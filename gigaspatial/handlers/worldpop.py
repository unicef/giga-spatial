"""
WorldPop data handler for population and demographic datasets.

This module provides handlers for interacting with the WorldPop repository. It supports:
- Population counts and densities (constrained and unconstrained).
- Age and sex structures (school age, under-18, and detailed cohorts).
- Degree of Urbanisation (DUG) classification grids.
- REST API integration for automated dataset discovery and metadata retrieval.
"""

import dataclasses
from pydantic.dataclasses import dataclass
from pydantic import (
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)
from pathlib import Path
import os
from typing import Optional, Union, Literal, List, Dict, Any, Tuple
import pandas as pd
import geopandas as gpd
import pycountry
import requests
import logging

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.processing.tif_processor import TifProcessor
from gigaspatial.handlers.base import (
    BaseHandlerConfig,
    BaseHandlerDownloader,
    BaseHandlerReader,
    BaseHandler,
    DownloadOptions,
    RasterReadOptions,
    ResolveOptions,
)
from gigaspatial.handlers.base.fetch import extract_zip_to_store, stream_to_store
from gigaspatial.handlers.base.options import split_kwargs
from gigaspatial.config import config as global_config


class WorldPopRestClient:
    """
    REST API client for WorldPop data access.

    This class provides direct access to the WorldPop REST API without any
    configuration dependencies, allowing flexible integration patterns.
    """

    def __init__(
        self,
        base_url: str = "https://www.worldpop.org/rest/data",
        stats_url: str = "https://api.worldpop.org/v1/services/stats",
        api_key: Optional[str] = None,
        timeout: int = 600,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the WorldPop REST API client.

        Args:
            base_url: Base endpoint for the WorldPop REST API.
            stats_url: Endpoint for the WorldPop statistics service.
            api_key: Optional API key for authentication and higher rate limits.
            timeout: Request timeout duration in seconds.
            logger: Component logger instance.
        """
        self.base_url = base_url.rstrip("/")
        self.stats_url = stats_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Setup session with default headers
        self.session = requests.Session()
        self.session.headers.update(
            {"Accept": "application/json", "User-Agent": "WorldPop-Python-Client/1.0"}
        )

        if self.api_key:
            self.session.headers["X-API-Key"] = self.api_key

    def get_available_projects(self) -> List[Dict[str, Any]]:
        """
        Retrieve a list of all available WorldPop projects.

        Returns:
            A list of project metadata dictionaries (alias, title, description).
        """
        try:
            response = self.session.get(self.base_url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch available project aliases: {e}")
            return []

    def get_project_sources(self, dataset_type: str) -> List[Dict[str, Any]]:
        """
        Retrieve available data sources for a specific project type.

        Args:
            dataset_type: Project alias (e.g., 'pop', 'births').

        Returns:
            A list of source metadata dictionaries.
        """
        try:
            url = f"{self.base_url}/{dataset_type}"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.RequestException as e:
            self.logger.error(
                f"Failed to fetch project sources for {dataset_type}: {e}"
            )
            return []

    def get_source_entities(
        self, dataset_type: str, category: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve entities (countries, regions) available for a source and project.

        Args:
            dataset_type: Project alias.
            category: Source alias (e.g., 'wpgp').

        Returns:
            A list of entity metadata dictionaries (including ISO3 codes).
        """
        try:
            url = f"{self.base_url}/{dataset_type}/{category}"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.RequestException as e:
            self.logger.error(
                f"Failed to fetch entities for {dataset_type}/{category}: {e}"
            )
            return []

    def get_datasets(self, dataset_type: str, category: str, params: dict):
        """
        Retrieve datasets matching specific query parameters.

        Args:
            dataset_type: Dataset type alias.
            category: Source category alias.
            params: Dictionary of query parameters (e.g., {'iso3': 'RWA'}).

        Returns:
            A list of dataset metadata dictionaries.
        """
        try:
            url = f"{self.base_url}/{dataset_type}/{category}"
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch datasets for {params}: {e}")
            return []

    def get_datasets_by_country(
        self, dataset_type: str, category: str, iso3: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all datasets available for a specific country.

        Args:
            dataset_type: Dataset type alias.
            category: Source category alias.
            iso3: ISO 3166-1 alpha-3 country code.

        Returns:
            A list of country-specific dataset metadata.
        """
        params = {"iso3": iso3}
        return self.get_datasets(dataset_type, category, params)

    def get_dataset_by_id(
        self, dataset_type: str, category: str, dataset_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get dataset information by ID.

        Args:
            dataset_type: Dataset type alias (e.g., 'pop', 'births')
            category: Category alias (e.g., 'wpgp', 'pic')
            dataset_id: Dataset ID

        Returns:
            Dataset dictionary or None if not found
        """
        params = {"id": dataset_id}
        return self.get_datasets(dataset_type, category, params)

    def find_dataset(
        self,
        dataset_type: str,
        category: str,
        iso3: str,
        year: Union[str, int],
        **filters,
    ) -> Optional[Dict[str, Any]]:
        """
        Locate a specific dataset by year and optional metadata filters.

        Args:
            dataset_type: Dataset type alias.
            category: Source category alias.
            iso3: ISO 3166-1 alpha-3 code.
            year: Population year to match.
            **filters: Additional metadata filters (e.g., resolution='1km').

        Returns:
            The matching dataset dictionary, or None if not found.
        """
        datasets = self.get_country_datasets(dataset_type, category, iso3)
        year_str = str(year)

        for dataset in datasets:
            if dataset.get("popyear") == year_str:
                # Check additional filters
                match = True
                for key, value in filters.items():
                    if key in dataset and dataset[key] != value:
                        match = False
                        break

                if match:
                    return dataset

        return None

    def list_years_for_country(
        self, dataset_type: str, category: str, iso3: str
    ) -> List[int]:
        """
        List all available years for a specific country and dataset.

        Args:
            dataset_type: Dataset type alias
            category: Category alias
            iso3: ISO3 country code

        Returns:
            Sorted list of available years
        """
        datasets = self.get_datasets_by_country(dataset_type, category, iso3)
        years = []

        for dataset in datasets:
            try:
                year = int(dataset.get("popyear", 0))
                if year > 0:
                    years.append(year)
            except (ValueError, TypeError):
                continue

        return sorted(years)

    def search_datasets(
        self,
        dataset_type: Optional[str] = None,
        category: Optional[str] = None,
        iso3: Optional[str] = None,
        year: Optional[Union[str, int]] = None,
        **filters,
    ) -> List[Dict[str, Any]]:
        """
        Search for datasets with flexible filtering.

        Args:
            dataset_type: Optional dataset type filter
            category: Optional category filter
            iso3: Optional country filter
            year: Optional year filter
            **filters: Additional filters

        Returns:
            List of matching datasets
        """
        results = []

        if dataset_type:
            if category:
                # If we have country-specific filters
                if iso3:
                    datasets = self.get_datasets_by_country(
                        dataset_type, category, iso3
                    )
                    for dataset in datasets:
                        match = True

                        # Check year filter
                        if year and dataset.get("popyear") != str(year):
                            match = False

                        # Check additional filters
                        for key, value in filters.items():
                            if key in dataset and dataset[key] != value:
                                match = False
                                break

                        if match:
                            results.append(dataset)
                else:
                    return self.get_source_entities(dataset_type, category)
            else:
                return self.get_project_sources(dataset_type)
        else:
            return self.get_available_projects()

        return results

    def get_dataset_info(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract useful information from a dataset dictionary.

        Args:
            dataset: Dataset dictionary from API

        Returns:
            Cleaned dataset information
        """
        return {
            "id": dataset.get("id"),
            "title": dataset.get("title"),
            "description": dataset.get("desc"),
            "doi": dataset.get("doi"),
            "citation": dataset.get("citation"),
            "data_format": dataset.get("data_format"),
            "year": dataset.get("popyear"),
            "country": dataset.get("country"),
            "iso3": dataset.get("iso3"),
            "continent": dataset.get("continent"),
            "download_urls": dataset.get("files", []),
            "image_url": dataset.get("url_img"),
            "summary_url": dataset.get("url_summary"),
            "license": dataset.get("license"),
            "organization": dataset.get("organisation"),
            "author": dataset.get("author_name"),
            "maintainer": dataset.get("maintainer_name"),
            "project": dataset.get("project"),
            "category": dataset.get("category"),
            "date_created": dataset.get("date"),
            "public": dataset.get("public") == "Y",
            "archived": dataset.get("archive") == "Y",
        }

    def close(self):
        """Close the session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants (shared between resolution function & config class)
# ─────────────────────────────────────────────────────────────────────────────

AVAILABLE_YEARS_GR1: List[int] = list(range(2000, 2021))  # 2000-2020
AVAILABLE_YEARS_GR2: List[int] = list(range(2015, 2031))  # 2015-2030
AVAILABLE_RESOLUTIONS: List[int] = [100, 1000]


# ─────────────────────────────────────────────────────────────────────────────
# DemographicPathFilter — value object for demographic file filtering
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DemographicPathFilter:
    """
    Immutable value object encapsulating all demographic path-filtering logic.

    Built from the demographic resolve options (`WPResolveOptions`) and
    applied when a country's files are resolved.

    Attributes:
        sex_filters:   Set of sex codes to keep (``"M"``, ``"F"``, ``"T"``, ``"F_M"``).
        level_filters: Set of education-level codes to keep (``"PRIMARY"``, ``"SECONDARY"``).
        ages_filter:   Exact age values to keep.
        min_age:       Inclusive lower bound on age.
        max_age:       Inclusive upper bound on age.
    """

    sex_filters: Optional[frozenset] = None
    level_filters: Optional[frozenset] = None
    ages_filter: Optional[frozenset] = None
    min_age: Optional[int] = None
    max_age: Optional[int] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_kwargs(cls, **kwargs) -> "DemographicPathFilter":
        """
        Build a ``DemographicPathFilter`` from reader/caller keyword arguments.

        Recognised keys: ``sex``, ``education_level`` / ``level``,
        ``ages``, ``min_age``, ``max_age``.

        Args:
            **kwargs: Arbitrary keyword arguments from the call site.

        Returns:
            A frozen ``DemographicPathFilter`` instance.
        """

        def _to_str_frozenset(v) -> Optional[frozenset]:
            if v is None:
                return None
            if isinstance(v, (list, tuple, set, frozenset)):
                return frozenset(str(x).upper() for x in v)
            return frozenset({str(v).upper()})

        def _to_int_frozenset(v) -> Optional[frozenset]:
            if v is None:
                return None
            if isinstance(v, (list, tuple, set, frozenset)):
                return frozenset(int(x) for x in v)
            return frozenset({int(v)})

        sex_filters = _to_str_frozenset(kwargs.get("sex"))
        level_filters = _to_str_frozenset(
            kwargs.get("education_level") or kwargs.get("level")
        )
        ages_filter = _to_int_frozenset(kwargs.get("ages"))  # ← int frozenset
        raw_min = kwargs.get("min_age")
        raw_max = kwargs.get("max_age")

        return cls(
            sex_filters=sex_filters,
            level_filters=level_filters,
            ages_filter=ages_filter,
            min_age=int(raw_min) if raw_min is not None else None,
            max_age=int(raw_max) if raw_max is not None else None,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_filters(self) -> bool:
        """Return ``True`` if any filter criterion has been set."""
        return any(
            v is not None
            for v in (
                self.sex_filters,
                self.level_filters,
                self.ages_filter,
                self.min_age,
                self.max_age,
            )
        )

    # ------------------------------------------------------------------
    # Core filtering
    # ------------------------------------------------------------------

    def filter_paths(
        self,
        paths: List[Path],
        project: str,
        school_age: bool,
        _logger: Optional[logging.Logger] = None,
    ) -> List[Path]:
        """
        Filter a list of demographic TIF paths according to the stored criteria.

        Args:
            paths:      List of candidate local ``Path`` objects.
            project:    WorldPop project string (e.g. ``"age_structures"``).
            school_age: Whether this is a school-age dataset.
            _logger:    Optional logger for per-file warnings.

        Returns:
            Filtered list of ``Path`` objects.
        """
        log = _logger or logging.getLogger(self.__class__.__name__)
        filtered: List[Path] = []

        for p in paths:
            stem = os.path.splitext(p.name)[0]
            parts = stem.split("_")

            sex_val: Optional[str] = None
            age_val: Optional[int] = None
            education_level_val: Optional[str] = None

            is_school_age_filename = any(
                lvl in stem.upper() for lvl in ["PRIMARY", "SECONDARY"]
            )
            is_under_18_filename = "UNDER_18" in stem.upper() or "UNDER" in stem.upper()

            # ── Parse filename ───────────────────────────────────────────
            if is_under_18_filename:
                if len(parts) > 1:
                    sex_val = parts[1].upper()

            elif is_school_age_filename:
                if len(parts) >= 4:
                    if (
                        len(parts) > 2
                        and parts[1].upper() == "F"
                        and parts[2].upper() == "M"
                    ):
                        sex_val = "F_M"
                        if len(parts) > 3:
                            education_level_val = parts[3].upper()
                    elif len(parts) > 1:
                        sex_val = parts[1].upper()
                        if len(parts) > 2:
                            education_level_val = parts[2].upper()

            else:
                if len(parts) >= 4:
                    sex_val = parts[1].upper()
                    try:
                        age_val = int(parts[2])
                    except (ValueError, IndexError):
                        age_val = None

            # ── Sex filter ──────────────────────────────────────────────
            if self.sex_filters:
                sex_ok = (
                    ("F_M" in self.sex_filters and sex_val == "F_M")
                    or ("F" in self.sex_filters and sex_val == "F")
                    or ("M" in self.sex_filters and sex_val == "M")
                    or ("T" in self.sex_filters and sex_val == "T")
                )
                if not sex_ok:
                    continue
            elif is_under_18_filename:
                if sex_val != "T":
                    continue
            elif project == "age_structures" and school_age:
                if sex_val != "F_M":
                    continue

            # ── Education level filter ──────────────────────────────────
            if self.level_filters and is_school_age_filename:
                if (
                    education_level_val is None
                    or education_level_val not in self.level_filters
                ):
                    continue

            # ── Age filters ─────────────────────────────────────────────
            if (
                (
                    self.ages_filter is not None
                    or self.min_age is not None
                    or self.max_age is not None
                )
                and not is_school_age_filename
                and not is_under_18_filename
            ):
                if age_val is not None:
                    if self.ages_filter is not None and age_val not in self.ages_filter:
                        continue
                    if self.min_age is not None and age_val < self.min_age:
                        continue
                    if self.max_age is not None and age_val > self.max_age:
                        continue
                else:
                    log.warning(
                        f"Could not parse age from filename {p.name} but age filters "
                        f"were applied. Skipping file."
                    )
                    continue

            filtered.append(p)

        return filtered


@dataclasses.dataclass(frozen=True)
class WPResolveOptions(ResolveOptions):
    """
    Resolve options of WorldPop datasets: the demographic cohorts to keep.

    They select which age-structure files a country resolves to, so files
    outside the cohorts are neither downloaded nor loaded.

    Attributes:
        sex: Sex code(s) to keep (``"M"``, ``"F"``, ``"T"``, ``"F_M"``).
        education_level: Education level(s) to keep (``"PRIMARY"``,
            ``"SECONDARY"``); ``level`` is an alias.
        ages: Exact age value(s) to keep.
        min_age: Inclusive lower bound on age.
        max_age: Inclusive upper bound on age.
    """

    sex: Any = None
    education_level: Any = None
    level: Any = None
    ages: Any = None
    min_age: Optional[int] = None
    max_age: Optional[int] = None

    def __post_init__(self):
        for name in ("sex", "education_level", "level", "ages"):
            value = getattr(self, name)
            if isinstance(value, (list, tuple, set)):
                object.__setattr__(self, name, frozenset(value))

    @property
    def demographic_filter(self) -> DemographicPathFilter:
        return DemographicPathFilter.from_kwargs(
            sex=self.sex,
            education_level=self.education_level,
            level=self.level,
            ages=self.ages,
            min_age=self.min_age,
            max_age=self.max_age,
        )


# ─────────────────────────────────────────────────────────────────────────────
# resolve_dataset_category - pure, testable, side-effect-free
# ─────────────────────────────────────────────────────────────────────────────


def resolve_dataset_category(
    release: str,
    project: str,
    year: int,
    resolution: int,
    un_adjusted: bool,
    constrained: bool,
    school_age: bool,
    under_18: bool,
    dug_level: str = "L1",
) -> Tuple[str, Dict[str, Any], List[str]]:
    """
    Determine the WorldPop dataset category string and normalised field values.

    This is a **pure function**: it performs no I/O, raises ``ValueError`` for
    invalid combinations, and returns warnings as strings rather than logging
    them — leaving side effects to the caller.

    Args:
        release:     ``"GR1"`` or ``"GR2"``.
        project:     ``"pop"``, ``"age_structures"``, or ``"degree_of_urbanization"``.
        year:        Reference year for the dataset.
        resolution:  Spatial resolution in metres (100 or 1000).
        un_adjusted: Whether to use UN-adjusted totals.
        constrained: Whether to use building-constrained datasets.
        school_age:  Whether to select school-age demographic cohorts.
        under_18:    Whether to select under-18 demographic cohorts.
        dug_level:   Urbanisation grid level (``"L1"`` or ``"L2"``).

    Returns:
        A 3-tuple ``(dataset_category, normalized_fields, warnings)`` where:

        - ``dataset_category`` is the resolved category string.
        - ``normalized_fields`` is a ``dict`` of field names to normalised values
          that **differ** from the inputs (e.g. ``{"year": 2020}`` when a year
          override was applied).
        - ``warnings`` is a list of human-readable warning strings.

    Raises:
        ValueError: When the combination of parameters is invalid.
    """
    normalized: Dict[str, Any] = {}
    warnings: List[str] = []

    # ── DUG branch ───────────────────────────────────────────────────────────
    if project == "degree_of_urbanization":
        if year not in AVAILABLE_YEARS_GR2:
            raise ValueError(
                f"Degree of Urbanisation datasets support years 2015-2030, "
                f"got year={year}."
            )
        if school_age:
            warnings.append(
                "`school_age` is not applicable for degree_of_urbanization. Ignoring."
            )
            normalized["school_age"] = False
        if under_18:
            warnings.append(
                "`under_18` is not applicable for degree_of_urbanization. Ignoring."
            )
            normalized["under_18"] = False
        if un_adjusted:
            warnings.append(
                "`un_adjusted` is not applicable for degree_of_urbanization. Ignoring."
            )
            normalized["un_adjusted"] = False

        return "dug_g2_v1", normalized, warnings

    # ── GR2 branch ───────────────────────────────────────────────────────────
    if release == "GR2":
        if school_age:
            raise ValueError(
                "School age population datasets are not available in the GR2 release. "
                "Use project='age_structures' with release='GR1'."
            )
        if year not in AVAILABLE_YEARS_GR2:
            raise ValueError(f"GR2 release supports years 2015-2030, got year={year}.")
        if not constrained:
            raise ValueError(
                "GR2 release is currently only available as constrained datasets. "
                "Set constrained=True, or switch to release='GR1' for unconstrained data."
            )
        if un_adjusted:
            warnings.append(
                "GR2 datasets do not support UN adjustment. `un_adjusted` set to False."
            )
            normalized["un_adjusted"] = False
            un_adjusted = False

        if project == "pop":
            if under_18:
                raise ValueError(
                    "`under_18=True` is only valid for project='age_structures'. "
                    "Set project='age_structures' or under_18=False."
                )
            category = (
                "G2_CN_POP_R25A_100m" if resolution == 100 else "G2_CN_POP_R25A_1km"
            )

        elif project == "age_structures":
            if under_18:
                if resolution != 100:
                    warnings.append(
                        "Under-18 datasets are only available at 100m resolution. "
                        "`resolution` set to 100."
                    )
                    normalized["resolution"] = 100
                category = "G2_Age_U18_R25A_100m"
            else:
                category = (
                    "G2_CN_Age_R25A_100m" if resolution == 100 else "G2_CN_Age_R25A_1km"
                )
        else:
            raise ValueError(f"Unsupported project for GR2: {project!r}")

        return category, normalized, warnings

    # ── GR1 branch ───────────────────────────────────────────────────────────
    if year not in AVAILABLE_YEARS_GR1:
        raise ValueError(
            f"Year {year} is not available in the GR1 release. "
            f"Available years: 2000-2020. "
            f"For years 2015-2030 use release='GR2'."
        )

    if project == "age_structures":
        if school_age:
            if resolution == 100:
                warnings.append(
                    "School age datasets are only available at 1km resolution. "
                    "`resolution` set to 1000."
                )
                normalized["resolution"] = 1000
            if year != 2020:
                warnings.append(
                    "School age datasets are only available for 2020. `year` set to 2020."
                )
                normalized["year"] = 2020
            if un_adjusted:
                warnings.append(
                    "School age datasets are only available without UN adjustment. "
                    "`un_adjusted` set to False."
                )
                normalized["un_adjusted"] = False
            if constrained:
                warnings.append(
                    "School age datasets are only available unconstrained. "
                    "`constrained` set to False."
                )
                normalized["constrained"] = False
            category = "sapya1km"

        else:  # non-school age_structures
            if resolution == 1000:
                warnings.append(
                    "Age structures datasets are only available at 100m resolution. "
                    "`resolution` set to 100."
                )
                normalized["resolution"] = 100
            if not constrained:
                if un_adjusted:
                    warnings.append(
                        "Age structures unconstrained datasets are only available without "
                        "UN adjustment. `un_adjusted` set to False."
                    )
                    normalized["un_adjusted"] = False
                    un_adjusted = False
                category = "aswpgp"
            else:
                if un_adjusted:
                    if year != 2020:
                        warnings.append(
                            "Age structures constrained datasets with UN adjustment are only "
                            "available for 2020. `year` set to 2020."
                        )
                        normalized["year"] = 2020
                    category = "ascicua_2020"
                else:
                    if year != 2020:
                        raise ValueError(
                            "Age structures constrained datasets without UN adjustment are "
                            "only available for 2020. Please set `year` to 2020."
                        )
                    category = "ascic_2020"

    elif project == "pop":
        if school_age:
            raise ValueError(
                f"Received unexpected value '{school_age}' for `school_age` with "
                f"project='pop'. For school age population, use project='age_structures'."
            )
        if under_18:
            raise ValueError(
                "`under_18=True` is only available with release='GR2' and project='age_structures'."
            )
        if constrained:
            if year != 2020:
                warnings.append(
                    "Population constrained datasets are only available for 2020. "
                    "`year` set to 2020."
                )
                normalized["year"] = 2020
            if resolution != 100:
                warnings.append(
                    "Population constrained datasets are only available at 100m resolution. "
                    "`resolution` set to 100."
                )
                normalized["resolution"] = 100
            category = "cic2020_UNadj_100m" if un_adjusted else "cic2020_100m"
        else:
            category = (
                f"wpgp{'unadj' if un_adjusted else ''}"
                if resolution == 100
                else ("wpicuadj1km" if un_adjusted else "wpic1km")
            )
    else:
        raise ValueError(f"Unsupported project for GR1: {project!r}")

    return category, normalized, warnings


# ─────────────────────────────────────────────────────────────────────────────
# WPPopulationConfig
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"), kw_only=True)
class WPPopulationConfig(BaseHandlerConfig):
    """
    Configuration for WorldPop dataset retrieval.

    This class defines the parameters for selecting specific WorldPop datasets,
    including release versions (GR1/GR2), projects (pop/age/urban), and
    constraints (unconstrained/constrained).

    After construction the Pydantic model validator calls
    :func:`resolve_dataset_category` and:

    * Populates ``dataset_category`` with the resolved string.
    * Applies normalised field overrides (e.g. ``year``, ``resolution``) and
      logs any resulting warnings.

    Attributes:
        release:            Dataset release generation (``"GR1"`` or ``"GR2"``).
        project:            Category of data (population, age structures, or urbanization).
        year:               Reference year for the dataset.
        resolution:         Spatial resolution in metres (100 or 1000).
        un_adjusted:        If ``True``, uses UN-adjusted population totals.
        constrained:        If ``True``, uses building-constrained datasets.
        school_age:         If ``True``, filters for school-age demographic cohorts.
        under_18:           If ``True``, filters for under-18 demographic cohorts.
        dug_level:          Urbanisation classification level (``"L1"`` or ``"L2"``).
        dataset_category:   Resolved category string (populated by validator).
    """

    client = WorldPopRestClient()
    units_match_country = True
    options_type = WPResolveOptions

    # ── Module-level constants re-exported as class attributes ───────────────
    AVAILABLE_YEARS_GR1: List[int] = Field(
        default_factory=lambda: list(range(2000, 2021))
    )
    AVAILABLE_YEARS_GR2: List[int] = Field(
        default_factory=lambda: list(range(2015, 2031))
    )
    AVAILABLE_RESOLUTIONS: List[int] = Field(default_factory=lambda: [100, 1000])

    # ─- User-facing configuration fields ─────────────────────────────────────
    base_path: Path = Field(default=global_config.get_path("worldpop", "bronze"))

    release: Literal["GR1", "GR2"] = Field(
        ...,
        description=(
            "WorldPop dataset release. "
            "'GR1': years 2000-2020, pop & age_structures, constrained & unconstrained. "
            "'GR2': years 2015-2030, pop only, constrained only, no UN adjustment."
        ),
    )
    project: Literal["pop", "age_structures", "degree_of_urbanization"] = Field(...)
    year: int = Field(...)
    resolution: int = Field(...)
    un_adjusted: bool = Field(...)
    constrained: bool = Field(...)
    school_age: bool = Field(...)
    under_18: bool = Field(
        ...,
        description=(
            "If True, fetch under-18 population datasets. "
            "Only available for project='age_structures' with release='GR2'."
        ),
    )
    dug_level: Literal["L1", "L2"] = Field(
        default="L1",
        description=(
            "Degree of Urbanisation grid level. Only applies when project='degree_of_urbanization'. "
            "'L1': 3-class classification (urban/suburban/rural). "
            "'L2': 7-class detailed classification."
        ),
    )

    # ── Derived / internal fields ─────────────────────────────────────────────
    dataset_category: str = Field(default="", init=False)

    # ── Simple field validators ───────────────────────────────────────────────

    @field_validator("year")
    def validate_year(cls, value: int) -> int:
        """Reject years that fall outside all known WorldPop release windows."""
        all_valid = set(range(2000, 2021)) | set(range(2015, 2031))
        if value not in all_valid:
            raise ValueError(
                f"No datasets found for the provided year: {value}. "
                f"Valid years: 2000-2020 (GR1) or 2015-2030 (GR2/DUG)."
            )
        return value

    @field_validator("resolution")
    def validate_resolution(cls, value: int) -> int:
        """Reject resolutions not supported by any WorldPop dataset."""
        if value in AVAILABLE_RESOLUTIONS:
            return value
        raise ValueError(
            f"No datasets found for the provided resolution: {value}. "
            f"Available resolutions: {AVAILABLE_RESOLUTIONS}."
        )

    # ── Model-level validator - delegates to resolve_dataset_category ─────────

    @model_validator(mode="after")
    def validate_configuration(self) -> "WPPopulationConfig":
        """
        Resolve and normalise the configuration by delegating to
        :func:`resolve_dataset_category`.

        * Calls the pure resolver with the current field values.
        * Applies any normalised field overrides returned by the resolver
          (e.g. correcting ``year``, ``resolution``, boolean flags).
        * Logs each warning emitted by the resolver.
        * Stores the resolved ``dataset_category``.

        Returns:
            ``self`` (mutated in-place per Pydantic ``mode="after"`` convention).
        """
        category, normalized, warnings = resolve_dataset_category(
            release=self.release,
            project=self.project,
            year=self.year,
            resolution=self.resolution,
            un_adjusted=self.un_adjusted,
            constrained=self.constrained,
            school_age=self.school_age,
            under_18=self.under_18,
            dug_level=self.dug_level,
        )

        # Apply normalised overrides
        for field_name, new_value in normalized.items():
            object.__setattr__(self, field_name, new_value)

        # Surface warnings through the instance logger
        for msg in warnings:
            self.logger.warning(msg)

        self.dataset_category = category
        return self

    # ── Path resolution ───────────────────────────────────────────────────────

    def get_relevant_data_units_by_key(
        self, search_key: str, options: WPResolveOptions
    ) -> List[str]:
        """
        Identify the WorldPop files of a country.

        For ``age_structures``, ``.tif`` files outside the demographic
        cohorts in ``options`` are left out. Zip resources are kept; their
        extracted files are filtered by :meth:`get_stored_paths`.

        Args:
            search_key: ISO 3166-1 alpha-3 country code.
            options: The resolve options.

        Returns:
            The URLs of the relevant files.
        """
        api_dataset_type = (
            "dug" if self.project == "degree_of_urbanization" else self.project
        )
        datasets = self.client.search_datasets(
            api_dataset_type, self.dataset_category, search_key, self.year
        )

        if not datasets:
            raise RuntimeError(
                f"No WorldPop datasets found for country: {search_key}, "
                f"project: {self.project}, category: {self.dataset_category}, year: {self.year}. "
                "Please check the configuration parameters."
            )

        ZIP_CATEGORIES = {"sapya1km", "G2_Age_U18_R25A_100m"}

        if self.project == "degree_of_urbanization":
            files = [
                file
                for file in datasets[0].get("files", [])
                if file.endswith(".tif") and f"GRID_{self.dug_level}_" in file
            ]
        else:
            files = [
                file
                for file in datasets[0].get("files", [])
                if (self.dataset_category in ZIP_CATEGORIES or file.endswith(".tif"))
            ]

        dem_filter = options.demographic_filter
        if self.project == "age_structures" and dem_filter.has_filters:
            kept = set(
                dem_filter.filter_paths(
                    [Path(f) for f in files if not self._is_archive(f)],
                    project=self.project,
                    school_age=self.school_age,
                    _logger=self.logger,
                )
            )
            files = [f for f in files if self._is_archive(f) or Path(f) in kept]

        return files

    def get_data_unit_path(self, unit: str, options=None) -> Path:
        """
        Resolve a WorldPop file URL to its local storage path.

        Args:
            unit:     The remote file URL.
            options:  The resolve options (unused).

        Returns:
            Absolute local path for the data file.
        """
        return self.base_path / unit.split("GIS/")[1]

    def get_stored_paths(
        self, unit: str, options: Optional[WPResolveOptions] = None
    ) -> List[Path]:
        """
        Return the stored files of a WorldPop resource.

        A ``.tif`` resource is stored as itself. A zip resource (school-age
        and under-18 cohorts) is stored as the ``.tif`` files extracted from
        it, filtered for ``age_structures`` by the demographic cohorts in
        ``options``; their names are known only after extraction, so this is
        empty before download.

        Args:
            unit:     Remote URL of the resource.
            options:  The resolve options.

        Returns:
            The resource's stored file paths.
        """
        if not self._is_archive(unit):
            return [self.get_data_unit_path(unit)]

        tifs = self._extracted_tifs(unit)
        if self.project != "age_structures":
            return tifs
        return (options or WPResolveOptions()).demographic_filter.filter_paths(
            tifs,
            project=self.project,
            school_age=self.school_age,
            _logger=self.logger,
        )

    def is_unit_available(self, unit: str, paths: List[Path]) -> bool:
        """A zip resource is available once any of its ``.tif`` files is extracted."""
        if self._is_archive(unit):
            return bool(self._extracted_tifs(unit))
        return super().is_unit_available(unit, paths)

    @staticmethod
    def _is_archive(unit: str) -> bool:
        return unit.lower().endswith(".zip")

    def _extracted_tifs(self, unit: str) -> List[Path]:
        output_dir = self.get_data_unit_path(unit).parent
        if not self.data_store.is_dir(str(output_dir)):
            return []
        return [
            Path(f)
            for f in self.data_store.list_files(str(output_dir))
            if f.lower().endswith(".tif")
        ]

    # ── Search key ────────────────────────────────────────────────────────────

    def get_search_key(self, source, options: WPResolveOptions) -> str:
        """
        Identify the country code from a geographic source for dataset search.

        Args:
            source:   Country name or ISO code.
            options:  The resolve options.

        Returns:
            The ISO 3166-1 alpha-3 country code.
        """
        if not isinstance(source, str):
            raise ValueError(
                f"Unsupported source type: {type(source)}. "
                "Please use country-based (str) filtering."
            )
        return pycountry.countries.lookup(source).alpha_3

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        base = (
            f"WPPopulationConfig("
            f"release={self.release}, "
            f"project={self.project}, "
            f"year={self.year}"
        )
        if self.project == "degree_of_urbanization":
            return base + f", dug_level={self.dug_level})"

        return (
            base + f", resolution={self.resolution}, "
            f"un_adjusted={self.un_adjusted}, "
            f"constrained={self.constrained}, "
            f"school_age={self.school_age}, "
            f"under_18={self.under_18})"
        )


class WPPopulationDownloader(BaseHandlerDownloader):
    """
    Downloader for WorldPop datasets.

    Handles the acquisition of demographic rasters from the WorldPop repository,
    including specialized logic for extracting zipped age structures.
    """

    options_type = DownloadOptions

    def __init__(
        self,
        config: Union[WPPopulationConfig, dict[str, Union[str, int]]],
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the WorldPop downloader.

        Args:
            config: WorldPop configuration object or parameter dictionary.
            data_store: Storage interface for local persistence.
            logger: Component logger instance.
        """
        config = (
            config
            if isinstance(config, WPPopulationConfig)
            else WPPopulationConfig(**config)
        )
        super().__init__(config=config, data_store=data_store, logger=logger)

    def download_data_unit(self, url: str, options: DownloadOptions):
        """
        Download a WorldPop data resource (TIF or ZIP).

        Args:
            url: Remote URL of the resource.
            options: The download options.

        Returns:
            The stored path, or for a zip the paths of its extracted ``.tif``
            files.

        Raises:
            requests.RequestException: If the download fails.
            zipfile.BadZipFile: If a zip download is not a zip archive.
            FileNotFoundError: If a zip holds no ``.tif`` file.
        """
        fetch = dict(
            session=self.config.client.session,
            show_progress=options.show_progress,
            timeout=self.config.client.timeout,
        )
        stored = self.config.get_data_unit_path(url)
        if url.lower().endswith(".zip"):
            return extract_zip_to_store(
                url, stored.parent, self.data_store, r"(?i).*\.tif$", **fetch
            )
        return stream_to_store(url, stored, self.data_store, **fetch)


class WPPopulationReader(BaseHandlerReader):
    """
    Reader for WorldPop datasets.

    Interfaces with local demographic rasters to facilitate data loading
    and spatial aggregation.
    """

    options_type = RasterReadOptions

    def __init__(
        self,
        config: Union[WPPopulationConfig, dict[str, Union[str, int]]],
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the WorldPop reader.

        Args:
            config: WorldPop configuration object or parameter dictionary.
            data_store: Storage interface for local persistence.
            logger: Component logger instance.
        """
        config = (
            config
            if isinstance(config, WPPopulationConfig)
            else WPPopulationConfig(**config)
        )
        super().__init__(config=config, data_store=data_store, logger=logger)

    def load_from_paths(
        self, source_data_path: List[Union[str, Path]], options: RasterReadOptions
    ) -> Union[List[TifProcessor], TifProcessor]:
        """
        Load WorldPop rasters from local storage.

        Args:
            source_data_path: List of absolute paths to demographic TIFs.
            options: ``merge_rasters`` aggregates all rasters into a single processor.

        Returns:
            A TifProcessor or list of TifProcessors for the requested data.
        """
        return self._load_raster_data(
            raster_paths=source_data_path, merge_rasters=options.merge_rasters
        )


class WPPopulationHandler(BaseHandler):
    """
    Handler for WorldPop Populations datasets.

    This class provides a unified interface for downloading and loading WP Population data.
    It manages the lifecycle of configuration, downloading, and reading components.
    """

    def __init__(
        self,
        release: Literal["GR1", "GR2"] = "GR2",
        project: Literal["pop", "age_structures"] = "pop",
        year: int = 2025,
        resolution: int = 1000,
        un_adjusted: bool = False,
        constrained: bool = True,
        school_age: bool = False,
        under_18: bool = False,
        dug_level: Literal["L1", "L2"] = "L1",
        config: Optional[WPPopulationConfig] = None,
        downloader: Optional[WPPopulationDownloader] = None,
        reader: Optional[WPPopulationReader] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        """
        Initialize the WorldPop demographic handler.

        Args:
            release: Generation of datasets to target.
            project: Specific data project (population, age, urbanization).
            year: Dataset reference year.
            resolution: Grid resolution (100m or 1km).
            un_adjusted: If True, uses UN-adjusted variants.
            constrained: If True, target building-constrained rasters.
            school_age: If True, targets school-age cohorts.
            under_18: If True, targets under-18 cohorts.
            dug_level: Urbanization classification granularity.
            config: Optional configuration override.
            downloader: Optional downloader instance.
            reader: Optional reader instance.
            data_store: Storage interface for persistence.
            logger: Component logger instance.
            **kwargs: Additional configuration overrides.
        """
        self._release = release
        self._project = project
        self._year = year
        self._resolution = resolution
        self._un_adjusted = un_adjusted
        self._constrained = constrained
        self._school_age = school_age
        self._under_18 = under_18
        self._dug_level = dug_level
        super().__init__(
            config=config,
            downloader=downloader,
            reader=reader,
            data_store=data_store,
            logger=logger,
            **kwargs,
        )

    def create_config(
        self, data_store: DataStore, logger: logging.Logger, **kwargs
    ) -> WPPopulationConfig:
        """
        Develop a WorldPop configuration instance.

        Args:
            data_store: Storage backend for local files.
            logger: Component logger.
            **kwargs: Configuration overrides.

        Returns:
            A configured WPPopulationConfig instance.
        """
        return WPPopulationConfig(
            release=self._release,
            project=self._project,
            year=self._year,
            resolution=self._resolution,
            un_adjusted=self._un_adjusted,
            constrained=self._constrained,
            school_age=self._school_age,
            under_18=self._under_18,
            dug_level=self._dug_level,
            data_store=data_store,
            logger=logger,
            **kwargs,
        )

    def create_downloader(
        self,
        config: WPPopulationConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> WPPopulationDownloader:
        """
        Develop a WorldPop downloader instance.

        Args:
            config: Handler configuration.
            data_store: Storage backend for local files.
            logger: Component logger.
            **kwargs: Downloader parameters.

        Returns:
            A configured WPPopulationDownloader instance.
        """
        return WPPopulationDownloader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )

    def create_reader(
        self,
        config: WPPopulationConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> WPPopulationReader:
        """
        Develop a WorldPop reader instance.

        Args:
            config: Handler configuration.
            data_store: Storage backend for local files.
            logger: Component logger.
            **kwargs: Reader parameters.

        Returns:
            A configured WPPopulationReader instance.
        """
        return WPPopulationReader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )

    def load_into_dataframe(
        self,
        source: str,
        ensure_available: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Acquire WorldPop data and load it into a tabular DataFrame.

        Args:
            source: Geographic area (country ISO3).
            ensure_available: If True, executes download if data is missing locally.
            **kwargs: Handler options go to ``load_data``, the rest to
                ``TifProcessor.to_dataframe``.

        Returns:
            A pandas DataFrame containing demographic data.
        """
        load_kwargs, frame_kwargs = split_kwargs(
            kwargs, self.config, self.downloader, self.reader
        )
        tif_processors = self.load_data(
            source=source, ensure_available=ensure_available, **load_kwargs
        )
        if isinstance(tif_processors, TifProcessor):
            return tif_processors.to_dataframe(**frame_kwargs)

        return pd.concat(
            [tp.to_dataframe(**frame_kwargs) for tp in tif_processors], ignore_index=True
        )

    def load_into_geodataframe(
        self,
        source: str,
        ensure_available: bool = True,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """
        Acquire WorldPop data and load it into a geospatial GeoDataFrame.

        Args:
            source: Geographic area (country ISO3).
            ensure_available: If True, executes download if data is missing locally.
            **kwargs: Handler options go to ``load_data``, the rest to
                ``TifProcessor.to_geodataframe``.

        Returns:
            A GeoDataFrame containing demographic data with geometries.
        """
        load_kwargs, frame_kwargs = split_kwargs(
            kwargs, self.config, self.downloader, self.reader
        )
        tif_processors = self.load_data(
            source=source, ensure_available=ensure_available, **load_kwargs
        )
        if isinstance(tif_processors, TifProcessor):
            return tif_processors.to_geodataframe(**frame_kwargs)

        return pd.concat(
            [tp.to_geodataframe(**frame_kwargs) for tp in tif_processors], ignore_index=True
        )

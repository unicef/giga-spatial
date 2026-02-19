from pydantic.dataclasses import dataclass
from pydantic import (
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)
from pathlib import Path
import functools
import multiprocessing
import os
from typing import Optional, Union, Literal, List, Dict, Any
import numpy as np
import pandas as pd
import geopandas as gpd
import pycountry
import requests
from tqdm import tqdm
import logging
import zipfile
import tempfile

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.processing.tif_processor import TifProcessor
from gigaspatial.handlers.base import (
    BaseHandlerConfig,
    BaseHandlerDownloader,
    BaseHandlerReader,
    BaseHandler,
)
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
            base_url: Base URL for the WorldPop REST API
            stats_url: URL for the WorldPop statistics API
            api_key: Optional API key for higher rate limits
            timeout: Request timeout in seconds
            logger: Optional logger instance
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
        Get list of all available projects (e.g., population, births, pregnancies, etc.).

        Returns:
            List of project dictionaries with alias, name, title, and description
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
        Get available sources for a specific project type.

        Args:
            dataset_type: Project type alias (e.g., 'pop', 'births', 'pregnancies')

        Returns:
            List of source dictionaries with alias and name
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
        Get list of entities (countries, global, continental) available for a specific project type and source.

        Args:
            dataset_type: Project type alias (e.g., 'pop', 'births')
            category: Source alias (e.g., 'wpgp', 'pic')

        Returns:
            List of entity dictionaries with id and iso3 codes (if applicable)
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
        Get all datasets available for the params.

        Args:
            dataset_type: Dataset type alias (e.g., 'pop', 'births')
            category: Category alias (e.g., 'wpgp', 'pic')
            params: Query parameters (e.g., {'iso3`:'RWA'})

        Returns:
            List of dataset dictionaries with metadata and file information
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
        Get all datasets available for a specific country.

        Args:
            dataset_type: Dataset type alias (e.g., 'pop', 'births')
            category: Category alias (e.g., 'wpgp', 'pic')
            iso3: ISO3 country code (e.g., 'USA', 'BRA')

        Returns:
            List of dataset dictionaries with metadata and file information
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
        Find a specific dataset by year and optional filters.

        Args:
            dataset_type: Dataset type alias
            category: Category alias
            iso3: ISO3 country code
            year: Year to search for
            **filters: Additional filters (e.g., gender='F', resolution='1km')

        Returns:
            Dataset dictionary or None if not found
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


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class WPPopulationConfig(BaseHandlerConfig):

    client = WorldPopRestClient()

    AVAILABLE_YEARS_GR1: List = Field(default=list(range(2000, 2021)))  # 2000–2020
    AVAILABLE_YEARS_GR2: List = Field(default=list(range(2015, 2031)))  # 2015–2030
    AVAILABLE_RESOLUTIONS: List = Field(default=[100, 1000])

    # user config
    base_path: Path = Field(default=global_config.get_path("worldpop", "bronze"))

    release: Literal["GR1", "GR2"] = Field(
        ...,
        description=(
            "WorldPop dataset release. "
            "'GR1': years 2000-2020, pop & age_structures, constrained & unconstrained. "
            "'GR2': years 2015-2030, pop only, constrained only, no UN adjustment."
        ),
    )
    project: Literal["pop", "age_structures"] = Field(...)
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

    def _filter_age_sex_paths(self, paths: List[Path], filters: Dict) -> List[Path]:
        """
        Helper to filter a list of WorldPop age_structures paths based on sex, age,
        and education level filters.

        Supported filename patterns:
        - School age:  DJI_M_SECONDARY_2020_1km.tif       (ISO3_SEX_LEVEL_YEAR_RES.tif)
                    DJI_F_M_SECONDARY_2020_1km.tif
        - Under-18:    rwa_T_Under_18_2020_CN_100m.tif     (ISO3_SEX_Under_18_YEAR_...tif)
                    rwa_F_Under_18_2020_CN_100m.tif
                    rwa_M_Under_18_2020_CN_100m.tif
        - Non-school:  RWA_F_25_2020.tif                  (ISO3_SEX_AGE_YEAR.tif)
        """
        sex_filters = filters.get("sex_filters")
        level_filters = filters.get("level_filters")
        ages_filter = filters.get("ages_filter")
        min_age = filters.get("min_age")
        max_age = filters.get("max_age")

        filtered_paths: List[Path] = []

        for p in paths:
            bn = p.name
            stem = os.path.splitext(bn)[0]
            parts = stem.split("_")

            sex_val, age_val, education_level_val = None, None, None

            # Detect file type by keywords in the stem
            is_school_age_filename = any(
                lvl in stem.upper() for lvl in ["PRIMARY", "SECONDARY"]
            )
            is_under_18_filename = "UNDER_18" in stem.upper() or "UNDER" in stem.upper()

            if is_under_18_filename:
                # Pattern: ISO3_SEX_Under_18_YEAR_... where SEX is T, F, or M
                if len(parts) > 1:
                    sex_val = parts[1].upper()  # T, F, or M

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

            else:  # Non-school age: RWA_F_25_2020.tif
                if len(parts) >= 4:
                    sex_val = parts[1].upper()
                    try:
                        age_val = int(parts[2])
                    except (ValueError, IndexError):
                        age_val = None

            # ── Sex filter ───────────────────────────────────────────────────────
            if sex_filters:
                sex_ok = (
                    ("F_M" in sex_filters and sex_val == "F_M")
                    or ("F" in sex_filters and sex_val == "F")
                    or ("M" in sex_filters and sex_val == "M")
                    or ("T" in sex_filters and sex_val == "T")
                )
                if not sex_ok:
                    continue
            elif is_under_18_filename:
                # Default for under-18 with no sex filter: return only the total (T) file
                if sex_val != "T":
                    continue
            elif self.project == "age_structures" and self.school_age:
                # Default for school_age with no sex filter: return only F_M combined file
                if sex_val != "F_M":
                    continue

            # ── Education level filter (school_age only) ─────────────────────────
            if level_filters and is_school_age_filename:
                if (
                    education_level_val is None
                    or education_level_val not in level_filters
                ):
                    continue

            # ── Age filters (non-school, non-under-18 only) ───────────────────────
            if (
                (ages_filter is not None or min_age is not None or max_age is not None)
                and not is_school_age_filename
                and not is_under_18_filename
            ):
                if age_val is not None:
                    if ages_filter is not None and age_val not in ages_filter:
                        continue
                    if min_age is not None and age_val < int(min_age):
                        continue
                    if max_age is not None and age_val > int(max_age):
                        continue
                else:
                    self.logger.warning(
                        f"Could not parse age from filename {p.name} but age filters "
                        f"were applied. Skipping file."
                    )
                    continue

            filtered_paths.append(p)

        return filtered_paths

    @field_validator("year")
    def validate_year(cls, value: int) -> int:
        all_valid = set(range(2000, 2021)) | set(range(2015, 2031))
        if value not in all_valid:
            raise ValueError(
                f"No datasets found for the provided year: {value}. "
                f"Valid years: 2000-2020 (GR1) or 2015-2030 (GR2)."
            )
        return value

    @field_validator("resolution")
    def validate_resolution(cls, value: int) -> int:
        if value in [100, 1000]:
            return value
        raise ValueError(
            f"No datasets found for the provided resolution: {value}. "
            f"Available resolutions: [100, 1000]."
        )

    @model_validator(mode="after")
    def validate_configuration(self):
        """
        Validate and normalise configuration based on dataset availability constraints.

        GR1 (pop):
        - Constrained: only 2020 at 100m, with or without UN adjustment.
        - Unconstrained: 100m or 1km, with or without UN adjustment, years 2000-2020.

        GR1 (age_structures):
        - School age: only 2020, 1km, unconstrained, no UN adjustment.
        - Non-school age: 100m only.
        - Unconstrained: no UN adjustment.
        - Constrained + UN adjusted: only 2020.
        - Constrained, no UN adjustment: only 2020.

        GR2 (pop):
        - Constrained datasets only.
        - Years 2015-2030 at 100m or 1km.
        - No UN adjustment.

        GR2 (age_structures):
        - Constrained datasets only, no UN adjustment, no school age.
        - Years 2015-2030.
        - Full age structures: 100m or 1km.
        - Under-18 population (under_18=True): 100m only.
        """

        # ── GR2 branch ───────────────────────────────────────────────────────
        if self.release == "GR2":
            if self.school_age:
                raise ValueError(
                    "School age population datasets are not available in the GR2 release. "
                    "Use project='age_structures' with release='GR1'."
                )
            if self.year not in self.AVAILABLE_YEARS_GR2:
                raise ValueError(
                    f"GR2 release supports years 2015-2030, got year={self.year}."
                )
            if not self.constrained:
                raise ValueError(
                    "GR2 release is currently only available as constrained datasets. "
                    "Set constrained=True, or switch to release='GR1' for unconstrained data."
                )
            if self.un_adjusted:
                self.logger.warning(
                    "GR2 datasets do not support UN adjustment. `un_adjusted` set to False."
                )
                self.un_adjusted = False

            if self.project == "pop":
                if self.under_18:
                    raise ValueError(
                        "`under_18=True` is only valid for project='age_structures'. "
                        "Set project='age_structures' or under_18=False."
                    )
                self.dataset_category = (
                    "G2_CN_POP_R25A_100m"
                    if self.resolution == 100
                    else "G2_CN_POP_R25A_1km"
                )

            elif self.project == "age_structures":
                if self.under_18:
                    if self.resolution != 100:
                        self.logger.warning(
                            "Under-18 datasets are only available at 100m resolution. "
                            "`resolution` set to 100."
                        )
                        self.resolution = 100
                    self.dataset_category = "G2_Age_U18_R25A_100m"
                else:
                    self.dataset_category = (
                        "G2_CN_Age_R25A_100m"
                        if self.resolution == 100
                        else "G2_CN_Age_R25A_1km"
                    )

            return self

        # ── GR1 branch ───────────────────────────────────────────────────────
        if self.year not in self.AVAILABLE_YEARS_GR1:
            raise ValueError(
                f"Year {self.year} is not available in the GR1 release. "
                f"Available years: 2000-2020. "
                f"For years 2015-2030 use release='GR2'."
            )

        if self.project == "age_structures":
            if self.school_age:
                if self.resolution == 100:
                    self.logger.warning(
                        "School age datasets are only available at 1km resolution. "
                        "`resolution` set to 1000."
                    )
                    self.resolution = 1000
                if self.year != 2020:
                    self.logger.warning(
                        "School age datasets are only available for 2020. "
                        "`year` set to 2020."
                    )
                    self.year = 2020
                if self.un_adjusted:
                    self.logger.warning(
                        "School age datasets are only available without UN adjustment. "
                        "`un_adjusted` set to False."
                    )
                    self.un_adjusted = False
                if self.constrained:
                    self.logger.warning(
                        "School age datasets are only available unconstrained. "
                        "`constrained` set to False."
                    )
                    self.constrained = False
                self.dataset_category = "sapya1km"

            else:  # non-school age_structures
                if self.resolution == 1000:
                    self.logger.warning(
                        "Age structures datasets are only available at 100m resolution. "
                        "`resolution` set to 100."
                    )
                    self.resolution = 100
                if not self.constrained:
                    if self.un_adjusted:
                        self.logger.warning(
                            "Age structures unconstrained datasets are only available without "
                            "UN adjustment. `un_adjusted` set to False."
                        )
                        self.un_adjusted = False
                    self.dataset_category = "aswpgp"
                else:
                    if self.un_adjusted:
                        if self.year != 2020:
                            self.logger.warning(
                                "Age structures constrained datasets with UN adjustment are only "
                                "available for 2020. `year` set to 2020."
                            )
                            self.year = 2020
                        self.dataset_category = "ascicua_2020"
                    else:
                        if self.year != 2020:
                            raise ValueError(
                                "Age structures constrained datasets without UN adjustment are "
                                "only available for 2020. Please set `year` to 2020."
                            )
                        self.dataset_category = "ascic_2020"

        elif self.project == "pop":
            if self.school_age:
                raise ValueError(
                    f"Received unexpected value '{self.school_age}' for `school_age` with "
                    f"project='pop'. For school age population, use project='age_structures'."
                )
            if self.under_18:
                raise ValueError(
                    "`under_18=True` is only available with release='GR2' and project='age_structures'."
                )
            if self.constrained:
                if self.year != 2020:
                    self.logger.warning(
                        "Population constrained datasets are only available for 2020. "
                        "`year` set to 2020."
                    )
                    self.year = 2020
                if self.resolution != 100:
                    self.logger.warning(
                        "Population constrained datasets are only available at 100m resolution. "
                        "`resolution` set to 100."
                    )
                    self.resolution = 100
                self.dataset_category = (
                    "cic2020_UNadj_100m" if self.un_adjusted else "cic2020_100m"
                )
            else:
                self.dataset_category = (
                    f"wpgp{'unadj' if self.un_adjusted else ''}"
                    if self.resolution == 100
                    else ("wpicuadj1km" if self.un_adjusted else "wpic1km")
                )

        return self

    def get_relevant_data_units_by_geometry(
        self, geometry: str, **kwargs
    ) -> List[Dict[str, Any]]:
        datasets = self.client.search_datasets(
            self.project, self.dataset_category, geometry, self.year
        )

        if not datasets:
            raise RuntimeError(
                f"No WorldPop datasets found for country: {geometry}, "
                f"project: {self.project}, category: {self.dataset_category}, year: {self.year}. "
                "Please check the configuration parameters."
            )
        ZIP_CATEGORIES = {"sapya1km", "G2_Age_U18_R25A_100m"}

        files = [
            file
            for file in datasets[0].get("files", [])
            if (self.dataset_category in ZIP_CATEGORIES or file.endswith(".tif"))
        ]

        return files

    def get_data_unit_path(self, unit: str, **kwargs) -> Path:
        """
        Given a WP file url, return the corresponding path.
        """
        return self.base_path / unit.split("GIS/")[1]

    def get_data_unit_paths(self, units: Union[List[str], str], **kwargs) -> list:
        """
        Given WP file url(s), return the corresponding local file paths.

        - For school_age age_structures (zip resources), if extracted .tif files are present
        in the target directory, return those; otherwise, return the zip path(s) to allow
        the downloader to fetch and extract them.
        - For non-school_age age_structures (individual .tif URLs), you can filter by sex and age
        using kwargs: sex, ages, min_age, max_age.
        """
        if not isinstance(units, list):
            units = [units]

        # Extract optional filters
        sex = kwargs.get("sex")
        education_level = kwargs.get("education_level") or kwargs.get("level")
        ages_filter = kwargs.get("ages")
        min_age = kwargs.get("min_age")
        max_age = kwargs.get("max_age")

        def _to_set(v):
            if v is None:
                return None
            if isinstance(v, (list, tuple, set)):
                return {str(x).upper() for x in v}
            return {str(v).upper()}

        sex_filters = _to_set(sex)
        level_filters = _to_set(education_level)

        # 1) School-age branch (zip → extracted tifs)
        if self.project == "age_structures" and (self.school_age or self.under_18):
            resolved_paths: List[Path] = []
            for url in units:
                output_dir = self.get_data_unit_path(url).parent

                if self.data_store.is_dir(str(output_dir)):
                    try:
                        all_extracted_tifs = [
                            Path(f)
                            for f in self.data_store.list_files(str(output_dir))
                            if f.lower().endswith(".tif")
                        ]
                        # Apply filters to extracted tifs
                        filtered_tifs = self._filter_age_sex_paths(
                            all_extracted_tifs,
                            {
                                "sex_filters": sex_filters,
                                "level_filters": level_filters,
                            },
                        )
                        resolved_paths.extend(filtered_tifs)
                    except Exception:
                        resolved_paths.append(self.get_data_unit_path(url))  # Fallback
                else:
                    resolved_paths.append(
                        self.get_data_unit_path(url)
                    )  # Fallback if not extracted yet

            return resolved_paths

        # 2) Non-school_age age_structures (individual tif URLs) with DEFERRED sex/age filters
        if self.project == "age_structures" and not self.school_age:
            # Store filters in a way that the reader can access them if needed
            self._temp_age_sex_filters = {
                "sex_filters": sex_filters,
                "ages_filter": ages_filter,
                "min_age": min_age,
                "max_age": max_age,
            }
            # Here, we don't apply the filters yet. We return all potential paths.
            # The actual filtering will happen in the reader or during TifProcessor loading.
            return [self.get_data_unit_path(unit) for unit in units]

        # Default behavior for all other datasets
        return [self.get_data_unit_path(unit) for unit in units]

    def extract_search_geometry(self, source, **kwargs):
        """
        Override the method since geometry extraction does not apply.
        Returns country iso3 for dataset search
        """
        if not isinstance(source, str):
            raise ValueError(
                f"Unsupported source type: {type(source)}"
                "Please use country-based (str) filtering."
            )

        return pycountry.countries.lookup(source).alpha_3

    def __repr__(self) -> str:

        return (
            f"WPPopulationConfig("
            f"release={self.release}, "
            f"project={self.project}, "
            f"year={self.year}, "
            f"resolution={self.resolution}, "
            f"un_adjusted={self.un_adjusted}, "
            f"constrained={self.constrained}, "
            f"school_age={self.school_age}, "
            f"under_18={self.under_18}"
            f")"
        )


class WPPopulationDownloader(BaseHandlerDownloader):

    def __init__(
        self,
        config: Union[WPPopulationConfig, dict[str, Union[str, int]]],
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the downloader.

        Args:
            config: Configuration for the WorldPop dataset, either as a WPPopulationConfig object or a dictionary of parameters
            data_store: Optional data storage interface. If not provided, uses LocalDataStore.
            logger: Optional custom logger. If not provided, uses default logger.
        """
        config = (
            config
            if isinstance(config, WPPopulationConfig)
            else WPPopulationConfig(**config)
        )
        super().__init__(config=config, data_store=data_store, logger=logger)

    def download_data_unit(self, url, **kwargs):
        """Download data file for a url. If a zip, extract contained .tif files."""
        # If the resource is a zip (e.g., school age datasets), download to temp and extract .tif files
        if url.lower().endswith(".zip"):
            temp_downloaded_path: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".zip"
                ) as temp_file:
                    temp_downloaded_path = Path(temp_file.name)
                    response = self.config.client.session.get(
                        url, stream=True, timeout=self.config.client.timeout
                    )
                    response.raise_for_status()

                    total_size = int(response.headers.get("content-length", 0))

                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {os.path.basename(temp_downloaded_path)}",
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                temp_file.write(chunk)
                                pbar.update(len(chunk))

                extracted_files: List[Path] = []
                output_dir = self.config.get_data_unit_path(url).parent
                with zipfile.ZipFile(str(temp_downloaded_path), "r") as zip_ref:
                    members = [
                        m for m in zip_ref.namelist() if m.lower().endswith(".tif")
                    ]
                    for member in members:
                        extracted_path = output_dir / Path(member).name
                        with zip_ref.open(member) as source:
                            file_content = source.read()
                            self.data_store.write_file(
                                str(extracted_path), file_content
                            )
                        extracted_files.append(extracted_path)
                        self.logger.info(f"Extracted {member} to {extracted_path}")

                return extracted_files

            except requests.RequestException as e:
                self.logger.error(f"Failed to download {url}: {e}")
                return None
            except zipfile.BadZipFile:
                self.logger.error("Downloaded file is not a valid zip archive.")
                return None
            except Exception as e:
                self.logger.error(f"Unexpected error processing zip for {url}: {e}")
                return None
            finally:
                if temp_downloaded_path and temp_downloaded_path.exists():
                    try:
                        temp_downloaded_path.unlink()
                    except OSError as e:
                        self.logger.warning(
                            f"Could not delete temporary file {temp_downloaded_path}: {e}"
                        )

        # Otherwise, download as a regular file (e.g., .tif)
        try:
            response = self.config.client.session.get(
                url, stream=True, timeout=self.config.client.timeout
            )
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            file_path = self.config.get_data_unit_path(url)

            with self.data_store.open(str(file_path), "wb") as file:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {os.path.basename(file_path)}",
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))

            self.logger.info(f"Successfully downloaded: {file_path}")
            return file_path

        except requests.RequestException as e:
            self.logger.error(f"Failed to download {url}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error downloading {url}: {e}")
            return None

    def download_data_units(
        self,
        urls: List[str],
        **kwargs,
    ) -> List[str]:
        """Download data files for multiple urls."""

        with multiprocessing.Pool(self.config.n_workers) as pool:
            download_func = functools.partial(self.download_data_unit)
            results = list(
                tqdm(
                    pool.imap(download_func, urls),
                    total=len(urls),
                    desc=f"Downloading data",
                )
            )

        # Flatten results and filter out None
        flattened: List[Path] = []
        for item in results:
            if item is None:
                continue
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)

        return flattened


class WPPopulationReader(BaseHandlerReader):

    def __init__(
        self,
        config: Union[WPPopulationConfig, dict[str, Union[str, int]]],
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the reader.

        Args:
            config: Configuration for the WorldPop dataset, either as a WPPopulationConfig object or a dictionary of parameters
            data_store: Optional data storage interface. If not provided, uses LocalDataStore.
            logger: Optional custom logger. If not provided, uses default logger.
        """
        config = (
            config
            if isinstance(config, WPPopulationConfig)
            else WPPopulationConfig(**config)
        )
        super().__init__(config=config, data_store=data_store, logger=logger)

    def load_from_paths(
        self,
        source_data_path: List[Union[str, Path]],
        merge_rasters: bool = False,
        **kwargs,
    ) -> Union[List[TifProcessor], TifProcessor]:
        """
        Load TifProcessors of WP datasets.
        Args:
            source_data_path: List of file paths to load
            merge_rasters: If True, all rasters will be merged into a single TifProcessor.
                           Defaults to False.
        Returns:
            Union[List[TifProcessor], TifProcessor]: List of TifProcessor objects for accessing the raster data or a single
                                                    TifProcessor if merge_rasters is True.
        """
        # Apply deferred age/sex filters if present and applicable
        if (
            hasattr(self.config, "_temp_age_sex_filters")
            and self.config.project == "age_structures"
            and not self.config.school_age
        ):
            # Ensure source_data_path is a list of Path objects for consistent filtering
            source_data_path = [
                Path(p) if isinstance(p, str) else p for p in source_data_path
            ]
            filtered_paths = self.config._filter_age_sex_paths(
                source_data_path, self.config._temp_age_sex_filters
            )
            # Clear the temporary filter after use
            del self.config._temp_age_sex_filters
            if not filtered_paths:
                self.logger.warning(
                    "No WorldPop age_structures paths matched the applied filters."
                )
                return []  # Return empty list if no paths after filtering
            source_data_path = filtered_paths

        return self._load_raster_data(
            raster_paths=source_data_path, merge_rasters=merge_rasters
        )

    def load(self, source, merge_rasters: bool = False, **kwargs):
        return super().load(source=source, merge_rasters=merge_rasters, **kwargs)


class WPPopulationHandler(BaseHandler):
    """
    Handler for WorldPop Populations datasets.

    This class provides a unified interface for downloading and loading WP Population data.
    It manages the lifecycle of configuration, downloading, and reading components.
    """

    def __init__(
        self,
        release: Literal["GR1", "GR2"] = "GR1",
        project: Literal["pop", "age_structures"] = "pop",
        year: int = 2020,
        resolution: int = 1000,
        un_adjusted: bool = True,
        constrained: bool = False,
        school_age: bool = False,
        under_18: bool = False,
        config: Optional[WPPopulationConfig] = None,
        downloader: Optional[WPPopulationDownloader] = None,
        reader: Optional[WPPopulationReader] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        self._release = release
        self._project = project
        self._year = year
        self._resolution = resolution
        self._un_adjusted = un_adjusted
        self._constrained = constrained
        self._school_age = school_age
        self._under_18 = under_18
        super().__init__(
            config=config,
            downloader=downloader,
            reader=reader,
            data_store=data_store,
            logger=logger,
        )

    def create_config(
        self, data_store: DataStore, logger: logging.Logger, **kwargs
    ) -> WPPopulationConfig:
        """
        Create and return a WPPopulationConfig instance.

        Args:
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional configuration parameters

        Returns:
            Configured WPPopulationConfig instance
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
        Create and return a WPPopulationDownloader instance.

        Args:
            config: The configuration object
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional downloader parameters

        Returns:
            Configured WPPopulationDownloader instance
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
        Create and return a WPPopulationReader instance.

        Args:
            config: The configuration object
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional reader parameters

        Returns:
            Configured WPPopulationReader instance
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
        Load GHSL data into a pandas DataFrame.

        Args:
            source: The data source specification
            ensure_available: If True, ensure data is downloaded before loading
            **kwargs: Additional parameters passed to load methods

        Returns:
            DataFrame containing the GHSL data
        """
        tif_processors = self.load_data(
            source=source, ensure_available=ensure_available, **kwargs
        )
        if isinstance(tif_processors, TifProcessor):
            return tif_processors.to_dataframe(**kwargs)

        return pd.concat(
            [tp.to_dataframe(**kwargs) for tp in tif_processors], ignore_index=True
        )

    def load_into_geodataframe(
        self,
        source: str,
        ensure_available: bool = True,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """
        Load GHSL data into a geopandas GeoDataFrame.

        Args:
            source: The data source specification
            ensure_available: If True, ensure data is downloaded before loading
            **kwargs: Additional parameters passed to load methods

        Returns:
            GeoDataFrame containing the GHSL data
        """
        tif_processors = self.load_data(
            source=source, ensure_available=ensure_available, **kwargs
        )
        if isinstance(tif_processors, TifProcessor):
            return tif_processors.to_geodataframe(**kwargs)

        return pd.concat(
            [tp.to_geodataframe(**kwargs) for tp in tif_processors], ignore_index=True
        )

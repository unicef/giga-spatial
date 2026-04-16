"""
Administrative boundary resolution and ingestion engine.

Provides the `AdminBoundaries` class, a specialized entity table for handling
spatial data from various administrative boundary providers (GADM, GeoRepo,
Natural Earth, etc.). Supports automated resolution based on ISO codes and
admin levels.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, ClassVar, Dict, Optional, Union
from urllib.error import HTTPError
import logging

import geopandas as gpd
import pycountry
from shapely.geometry import shape

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.handlers.hdx import HDXConfig

from gigaspatial.core.schemas.admin_boundary import AdminBoundary, AdminBoundaryTable


class AdminBoundaries(AdminBoundaryTable):
    """
    Handler for administrative boundary data from multiple external sources.

    Extends `AdminBoundaryTable` with factory methods that resolve and ingest
    boundaries from GADM, GeoRepo (UNICEF), geoBoundaries (HDX), Natural Earth,
    and internal data stores.

    Examples:
        >>> boundaries = AdminBoundaries.create("BRA", admin_level=1)
        >>> gdf = boundaries.to_geodataframe()
    """

    logger: ClassVar[logging.Logger] = global_config.get_logger("AdminBoundaries")

    # ------------------------------------------------------------------
    # Field mapping: source column names → AdminBoundary field names
    # ------------------------------------------------------------------

    _schema_config: ClassVar[Dict[str, Dict[str, str]]] = {
        "gadm": {
            "country_iso": "GID_0",
            "boundary_id": "GID_{level}",
            "name": "NAME_{level}",
            "parent_id": "GID_{parent_level}",
        },
        "internal": {
            "boundary_id": "admin{level}_id_giga",
            "name": "name",
            "name_en": "name_en",
            "country_iso": "iso_3166_1_alpha_3",
        },
        "geoBoundaries": {
            "boundary_id": "shapeID",
            "name": "shapeName",
            "country_iso": "shapeGroup",
        },
    }

    @property
    def boundaries(self) -> List[AdminBoundary]:
        """
        Backwards-compatible alias for ``entities``.

        Previously ``AdminBoundaries`` stored boundaries in a ``boundaries``
        list field. This property maintains compatibility with any code that
        accesses ``.boundaries`` directly.

        Returns:
            List of ``AdminBoundary`` entities identical to ``self.entities``.
        """
        return self.entities

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _map_fields(
        cls,
        gdf: gpd.GeoDataFrame,
        source: str,
        current_level: int,
    ) -> gpd.GeoDataFrame:
        """
        Rename raw source columns to AdminBoundary field names.

        Args:
            gdf: Raw GeoDataFrame from the data source.
            source: Key in ``_schema_config`` (e.g. ``"gadm"``).
            current_level: Admin level integer used for template substitution.

        Returns:
            GeoDataFrame with columns renamed to match AdminBoundary fields.
        """
        config = cls._schema_config.get(source, {})
        parent_level = current_level - 1

        field_mapping: Dict[str, str] = {}
        for target_field, src_col in config.items():
            if "{parent_level}" in src_col:
                field_mapping[src_col.format(parent_level=parent_level)] = target_field
            elif "{level}" in src_col:
                field_mapping[src_col.format(level=current_level)] = target_field
            else:
                field_mapping[src_col] = target_field

        return gdf.rename(columns=field_mapping)

    @classmethod
    def _gdf_to_table(
        cls,
        gdf: gpd.GeoDataFrame,
        admin_level: int,
    ) -> "AdminBoundaries":
        """
        Convert a GeoDataFrame (already column-mapped) into an AdminBoundaries instance.

        Sets ``admin_level`` on every row and delegates validation to
        ``AdminBoundaryTable.from_dataframe``.

        Args:
            gdf: GeoDataFrame with columns matching AdminBoundary field names.
            admin_level: Integer admin level to stamp on every entity.

        Returns:
            AdminBoundaries instance.
        """
        import pandas as pd

        df = pd.DataFrame(gdf)
        df["admin_level"] = admin_level
        # Ensure geometry column is named correctly for EntityProcessor
        if "geometry" not in df.columns and gdf.geometry.name:
            df["geometry"] = gdf.geometry
        return cls.from_dataframe(df, entity_class=AdminBoundary, clean=True)

    @classmethod
    def _create_empty(cls) -> "AdminBoundaries":
        """Return an empty AdminBoundaries instance."""
        return cls(entities=[])

    # ------------------------------------------------------------------
    # Source-specific factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_gadm(
        cls,
        country_code: str,
        admin_level: int = 0,
        **kwargs,
    ) -> "AdminBoundaries":
        """
        Load boundaries from the GADM 4.1 dataset.

        Args:
            country_code: ISO 3166-1 alpha-3 country code (e.g., "BRA").
            admin_level: Administrative level (0-4).
            **kwargs: Additional parameters.

        Returns:
            An AdminBoundaries instance populated from GADM.
        """
        url = (
            f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/"
            f"gadm41_{country_code}_{admin_level}.json"
        )
        cls.logger.info(
            f"Loading GADM data for {country_code} admin_level={admin_level} from {url}"
        )
        try:
            gdf = gpd.read_file(url)
            gdf = cls._map_fields(gdf, "gadm", admin_level)

            if admin_level == 0:
                gdf["country_iso"] = gdf["boundary_id"]
                gdf["name"] = gdf["COUNTRY"]
            elif admin_level == 1:
                gdf["country_iso"] = gdf["parent_id"]

            return cls._gdf_to_table(gdf, admin_level)

        except (ValueError, HTTPError, FileNotFoundError) as e:
            cls.logger.warning(
                f"GADM load failed for {country_code} level {admin_level}: {e}"
            )
            return cls._create_empty()

    @classmethod
    def from_georepo(
        cls,
        country_code: str,
        admin_level: int = 0,
        **kwargs,
    ) -> "AdminBoundaries":
        """
        Load boundaries from the UNICEF GeoRepo API.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.
            admin_level: Administrative level (0-4).
            **kwargs: Additional parameters.

        Returns:
            An AdminBoundaries instance populated from GeoRepo.
        """
        cls.logger.info(
            f"Loading GeoRepo data for {country_code} admin_level={admin_level}"
        )
        from gigaspatial.handlers.unicef.georepo import get_country_boundaries_by_iso3

        geojson = get_country_boundaries_by_iso3(country_code, admin_level=admin_level)
        features = geojson.get("features", [])
        parent_level = admin_level - 1

        records = []
        for feat in features:
            props = feat.get("properties", {})
            geom = feat.get("geometry")
            records.append(
                {
                    "boundary_id": props.get("ucode"),
                    "name": props.get("name"),
                    "name_en": props.get("name_en"),
                    "geometry": shape(geom) if geom else None,
                    "parent_id": (
                        props.get(f"adm{parent_level}_ucode")
                        if admin_level > 0
                        else None
                    ),
                    "country_iso": country_code,
                    "admin_level": admin_level,
                }
            )

        import pandas as pd

        df = gpd.GeoDataFrame(records, geometry="geometry", crs=4326)
        cls.logger.info(f"Fetched {len(df)} boundaries from GeoRepo.")
        return cls.from_dataframe(df, entity_class=AdminBoundary, clean=True)

    @classmethod
    def from_geoboundaries(
        cls,
        country_code: str,
        admin_level: int = 0,
    ) -> "AdminBoundaries":
        """
        Load boundaries from the geoBoundaries dataset via HDX.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.
            admin_level: Administrative level (0-4).

        Returns:
            An AdminBoundaries instance populated from geoBoundaries.

        Raises:
            ValueError: If no matching dataset or resource is found on HDX.
        """
        cls.logger.info(
            f"Searching HDX geoBoundaries for {country_code} level {admin_level}"
        )
        country_datasets = HDXConfig.search_datasets(
            query=(
                'dataseries_name:"geoBoundaries - Subnational Administrative Boundaries"'
                f' AND groups:"{country_code.lower()}"'
            ),
            rows=1,
        )
        if not country_datasets:
            raise ValueError(f"No geoBoundaries dataset found for {country_code}.")

        resources = [
            r
            for r in country_datasets[0].get_resources()
            if r.data["name"]
            == f"geoBoundaries-{country_code.upper()}-ADM{admin_level}.geojson"
        ]
        if not resources:
            raise ValueError(
                f"No geoBoundaries resource for {country_code} ADM{admin_level}."
            )

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                _, local_path = resources[0].download(folder=tmpdir)
                with open(local_path, "rb") as f:
                    gdf = gpd.read_file(f)

            gdf = cls._map_fields(gdf, "geoBoundaries", admin_level)
            return cls._gdf_to_table(gdf, admin_level)

        except (ValueError, HTTPError, FileNotFoundError) as e:
            cls.logger.warning(
                f"geoBoundaries load failed for {country_code} level {admin_level}: {e}"
            )
            return cls._create_empty()

    @classmethod
    def from_data_store(
        cls,
        path: Union[str, Path],
        data_store: DataStore = None,
        admin_level: int = 0,
        **kwargs,
    ) -> "AdminBoundaries":
        """
        Load boundaries from an internal data store.

        Args:
            path: Path to the dataset file within the store.
            data_store: DataStore instance providing file access.
            admin_level: Administrative level (0-4).
            **kwargs: Additional arguments for `read_dataset`.

        Returns:
            An AdminBoundaries instance.
        """
        from gigaspatial.core.io.readers import read_dataset

        cls.logger.info(f"Loading from data store at {path}, level={admin_level}")
        try:
            gdf = read_dataset(str(path), data_store=data_store, **kwargs)
            if gdf.empty:
                cls.logger.warning(f"Empty dataset at {path}.")
                return cls._create_empty()

            gdf = cls._map_fields(gdf, "internal", admin_level)

            if admin_level == 0:
                gdf["boundary_id"] = gdf.get("country_iso", gdf.get("boundary_id"))
            else:
                # Derive parent_id by stripping the last segment of the boundary_id
                gdf["parent_id"] = gdf["boundary_id"].apply(lambda x: x[:-3])

            return cls._gdf_to_table(gdf, admin_level)

        except (FileNotFoundError, KeyError) as e:
            cls.logger.warning(f"Data store load failed at {path}: {e}")
            return cls._create_empty()

    @classmethod
    def from_global_country_boundaries(cls, scale: str = "medium") -> "AdminBoundaries":
        """
        Load global country boundaries (admin level 0) from Natural Earth.

        Args:
            scale: Resolution ('large' -> 10m, 'medium' -> 50m, 'small' -> 110m).

        Returns:
            An AdminBoundaries instance with global country polygons.

        Raises:
            ValueError: If an invalid scale is provided.
        """
        scale_map = {"large": "10m", "medium": "50m", "small": "110m"}
        if scale not in scale_map:
            raise ValueError(
                f"Invalid scale '{scale}'. Choose from 'large', 'medium', 'small'."
            )
        res = scale_map[scale]
        url = (
            f"https://naciscdn.org/naturalearth/{res}/cultural/"
            f"ne_{res}_admin_0_countries.zip"
        )
        cls.logger.info(f"Loading Natural Earth boundaries from {url}")
        try:
            gdf = gpd.read_file(url)
            records = []
            for _, row in gdf.iterrows():
                iso = row.get("ISO_A3_EH") or row.get("ISO_A3") or row.get("ADM0_A3")
                name = row.get("NAME") or row.get("ADMIN") or row.get("SOVEREIGNT")
                if not iso or not name or row.get("geometry") is None:
                    continue
                records.append(
                    {
                        "boundary_id": iso,
                        "name": name,
                        "country_iso": iso,
                        "geometry": row["geometry"],
                        "admin_level": 0,
                    }
                )
            import pandas as pd

            cls.logger.info(
                f"Loaded {len(records)} country polygons from Natural Earth."
            )
            return cls.from_dataframe(
                gpd.GeoDataFrame(records, geometry="geometry", crs=4326),
                entity_class=AdminBoundary,
                clean=True,
            )
        except Exception as e:
            cls.logger.error(f"Natural Earth load failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Unified factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        country_code: Optional[str] = None,
        admin_level: int = 0,
        data_store: Optional[DataStore] = None,
        path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> "AdminBoundaries":
        """
        Unified factory method to resolve administrative boundaries.

        Automatically selects the best available source in the following order:
        1. Internal DataStore (if provided).
        2. UNICEF GeoRepo API.
        3. GADM 4.1.
        4. geoBoundaries (HDX).

        Args:
            country_code: ISO country code or name.
            admin_level: Administrative level (0=country, 1=state, etc.).
            data_store: Optional DataStore for internal lookups.
            path: Optional specific file path in the DataStore.
            **kwargs: Forwarded to specific source loaders.

        Returns:
            An AdminBoundaries instance.

        Raises:
            ValueError: If essential parameters are missing or invalid.
            RuntimeError: If all data sources fail to resolve.
        """
        cls.logger.info(
            f"AdminBoundaries.create country={country_code} level={admin_level} "
            f"data_store={data_store is not None} path={path is not None}"
        )

        use_data_store = data_store is not None and (
            global_config.ADMIN_BOUNDARIES_DATA_DIR is not None or path is not None
        )

        if not country_code and not data_store:
            raise ValueError("Either country_code or data_store must be provided.")

        if use_data_store and not path and not country_code:
            raise ValueError(
                "When data_store is provided, also supply path or country_code."
            )

        # ── 1. DataStore path ──────────────────────────────────────────
        if use_data_store:
            iso3 = cls._resolve_iso3(country_code)
            if path is None and iso3:
                path = global_config.get_admin_path(
                    country_code=iso3, admin_level=admin_level
                )
            return cls.from_data_store(
                path, data_store=data_store, admin_level=admin_level, **kwargs
            )

        # ── 2-4. Remote sources ────────────────────────────────────────
        iso3 = cls._resolve_iso3(country_code)  # raises ValueError on bad code

        if cls._try_georepo(iso3, admin_level):
            return cls.from_georepo(iso3, admin_level=admin_level)

        cls.logger.info("Attempting GADM fallback.")
        result = cls.from_gadm(iso3, admin_level, **kwargs)
        if result.entities:
            return result

        cls.logger.info("Attempting geoBoundaries fallback.")
        try:
            return cls.from_geoboundaries(iso3, admin_level)
        except Exception as e:
            cls.logger.error(f"All sources failed. Last error: {e}")
            raise RuntimeError(
                f"Failed to load boundaries for {country_code} "
                f"(GeoRepo, GADM, geoBoundaries all failed)."
            ) from e

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_iso3(country_code: Optional[str]) -> Optional[str]:
        """Resolve any country identifier to ISO 3166-1 alpha-3."""
        if country_code is None:
            return None
        try:
            return pycountry.countries.lookup(country_code).alpha_3
        except LookupError as e:
            raise ValueError(f"Invalid country code '{country_code}': {e}") from e

    @classmethod
    def _try_georepo(cls, iso3_code: str, admin_level: int) -> bool:
        """
        Test GeoRepo connectivity without raising.

        Returns:
            True if GeoRepo is reachable, False otherwise.
        """
        try:
            from gigaspatial.handlers.unicef.georepo import GeoRepoClient

            client = GeoRepoClient()
            if client.check_connection():
                cls.logger.info("GeoRepo connection OK.")
                return True
            cls.logger.info("GeoRepo connection returned False.")
            return False
        except ImportError:
            cls.logger.info("GeoRepo client not available.")
            return False
        except (ValueError, Exception) as e:
            cls.logger.warning(f"GeoRepo check failed: {e}")
            return False

    def __repr__(self) -> str:
        countries = self.get_countries()
        levels = self.get_admin_levels()
        return (
            f"AdminBoundaries("
            f"n={len(self.entities)}, "
            f"levels={list(levels)}, "
            f"countries={sorted(countries)}"
            f")"
        )

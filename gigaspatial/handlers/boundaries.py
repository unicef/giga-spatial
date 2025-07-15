from pydantic import BaseModel, Field
from typing import Optional, ClassVar, Union, Dict, List
import geopandas as gpd
from pathlib import Path
from urllib.error import HTTPError
from shapely.geometry import Polygon, MultiPolygon, shape
import tempfile
import pycountry

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.readers import read_dataset
from gigaspatial.handlers.hdx import HDXConfig
from gigaspatial.config import config as global_config


class AdminBoundary(BaseModel):
    """Base class for administrative boundary data with flexible fields."""

    id: str = Field(..., description="Unique identifier for the administrative unit")
    name: str = Field(..., description="Primary local name")
    geometry: Union[Polygon, MultiPolygon] = Field(
        ..., description="Geometry of the administrative boundary"
    )

    name_en: Optional[str] = Field(
        None, description="English name if different from local name"
    )
    parent_id: Optional[str] = Field(
        None, description="ID of parent administrative unit"
    )
    country_code: Optional[str] = Field(
        None, min_length=3, max_length=3, description="ISO 3166-1 alpha-3 country code"
    )

    class Config:
        arbitrary_types_allowed = True


class AdminBoundaries(BaseModel):
    """Base class for administrative boundary data with flexible fields."""

    boundaries: List[AdminBoundary] = Field(default_factory=list)
    level: int = Field(
        ...,
        ge=0,
        le=4,
        description="Administrative level (e.g., 0=country, 1=state, etc.)",
    )

    logger: ClassVar = global_config.get_logger("AdminBoundaries")

    _schema_config: ClassVar[Dict[str, Dict[str, str]]] = {
        "gadm": {
            "country_code": "GID_0",
            "id": "GID_{level}",
            "name": "NAME_{level}",
            "parent_id": "GID_{parent_level}",
        },
        "internal": {
            "id": "admin{level}_id_giga",
            "name": "name",
            "name_en": "name_en",
            "country_code": "iso_3166_1_alpha_3",
        },
        "geoBoundaries": {
            "id": "shapeID",
            "name": "shapeName",
            "country_code": "shapeGroup",
        },
    }

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert the AdminBoundaries to a GeoDataFrame."""
        if not self.boundaries:
            if hasattr(self, "_empty_schema"):
                columns = self._empty_schema
            else:
                columns = ["id", "name", "country_code", "geometry"]
                if self.level > 0:
                    columns.append("parent_id")

            return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=4326)

        return gpd.GeoDataFrame(
            [boundary.model_dump() for boundary in self.boundaries],
            geometry="geometry",
            crs=4326,
        )

    @classmethod
    def get_schema_config(cls) -> Dict[str, Dict[str, str]]:
        """Return field mappings for different data sources"""
        return cls._schema_config

    @classmethod
    def from_gadm(
        cls, country_code: str, admin_level: int = 0, **kwargs
    ) -> "AdminBoundaries":
        """Load and create instance from GADM data."""
        url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{country_code}_{admin_level}.json"
        cls.logger.info(
            f"Loading GADM data for country: {country_code}, admin level: {admin_level} from URL: {url}"
        )
        try:
            gdf = gpd.read_file(url)

            gdf = cls._map_fields(gdf, "gadm", admin_level)

            if admin_level == 0:
                gdf["country_code"] = gdf["id"]
                gdf["name"] = gdf["COUNTRY"]
            elif admin_level == 1:
                gdf["country_code"] = gdf["parent_id"]

            boundaries = [
                AdminBoundary(**row_dict) for row_dict in gdf.to_dict("records")
            ]
            cls.logger.info(f"Created {len(boundaries)} AdminBoundary objects.")
            return cls(
                boundaries=boundaries, level=admin_level, country_code=country_code
            )

        except (ValueError, HTTPError, FileNotFoundError) as e:
            cls.logger.warning(
                f"Error loading GADM data for {country_code} at admin level {admin_level}: {str(e)}"
            )
            cls.logger.info("Falling back to empty instance")
            return cls._create_empty_instance(country_code, admin_level, "gadm")

    @classmethod
    def from_data_store(
        cls,
        data_store: DataStore,
        path: Union[str, "Path"],
        admin_level: int = 0,
        **kwargs,
    ) -> "AdminBoundaries":
        """Load and create instance from internal data store."""
        cls.logger.info(
            f"Loading data from data store at path: {path}, admin level: {admin_level}"
        )
        try:
            gdf = read_dataset(data_store, str(path), **kwargs)

            if gdf.empty:
                cls.logger.warning(f"No data found at {path}.")
                return cls._create_empty_instance(None, admin_level, "internal")

            gdf = cls._map_fields(gdf, "internal", admin_level)

            if admin_level == 0:
                gdf["id"] = gdf["country_code"]
            else:
                gdf["parent_id"] = gdf["id"].apply(lambda x: x[:-3])

            boundaries = [
                AdminBoundary(**row_dict) for row_dict in gdf.to_dict("records")
            ]
            cls.logger.info(f"Created {len(boundaries)} AdminBoundary objects.")
            return cls(boundaries=boundaries, level=admin_level)

        except (FileNotFoundError, KeyError) as e:
            cls.logger.warning(
                f"No data found at {path} for admin level {admin_level}: {str(e)}"
            )
            cls.logger.info("Falling back to empty instance")
            return cls._create_empty_instance(None, admin_level, "internal")

    @classmethod
    def from_georepo(
        cls,
        country_code: str = None,
        admin_level: int = 0,
        **kwargs,
    ) -> "AdminBoundaries":
        """
        Load and create instance from GeoRepo (UNICEF) API.

        Args:
            country: Country name (if using name-based lookup)
            iso3: ISO3 code (if using code-based lookup)
            admin_level: Administrative level (0=country, 1=state, etc.)
            api_key: GeoRepo API key (optional)
            email: GeoRepo user email (optional)
            kwargs: Extra arguments (ignored)

        Returns:
            AdminBoundaries instance
        """
        cls.logger.info(
            f"Loading data from UNICEF GeoRepo for country: {country_code}, admin level: {admin_level}"
        )
        from gigaspatial.handlers.unicef_georepo import get_country_boundaries_by_iso3

        # Fetch boundaries from GeoRepo
        geojson = get_country_boundaries_by_iso3(country_code, admin_level=admin_level)

        features = geojson.get("features", [])
        boundaries = []
        parent_level = admin_level - 1

        for feat in features:
            props = feat.get("properties", {})
            geometry = feat.get("geometry")
            shapely_geom = shape(geometry) if geometry else None
            # For admin_level 0, no parent_id
            parent_id = None
            if admin_level > 0:
                parent_id = props.get(f"adm{parent_level}_ucode")

            boundary = AdminBoundary(
                id=props.get("ucode"),
                name=props.get("name"),
                name_en=props.get("name_en"),
                geometry=shapely_geom,
                parent_id=parent_id,
                country_code=country_code,
            )
            boundaries.append(boundary)

        cls.logger.info(
            f"Created {len(boundaries)} AdminBoundary objects from GeoRepo data."
        )

        # Try to infer country_code from first boundary if not set
        if boundaries and not boundaries[0].country_code:
            boundaries[0].country_code = boundaries[0].id[:3]

        return cls(boundaries=boundaries, level=admin_level)

    @classmethod
    def from_geoboundaries(cls, country_code, admin_level: int = 0):
        cls.logger.info(
            f"Searching for geoBoundaries data for country: {country_code}, admin level: {admin_level}"
        )

        country_datasets = HDXConfig.search_datasets(
            query=f'dataseries_name:"geoBoundaries - Subnational Administrative Boundaries" AND groups:"{country_code.lower()}"',
            rows=1,
        )
        if not country_datasets:
            cls.logger.error(f"No datasets found for country: {country_code}")
            raise ValueError(
                "No resources found for the specified country. Please check your search parameters and try again."
            )

        cls.logger.info(f"Found dataset: {country_datasets[0].get('title', 'Unknown')}")

        resources = [
            resource
            for resource in country_datasets[0].get_resources()
            if (
                resource.data["name"]
                == f"geoBoundaries-{country_code.upper()}-ADM{admin_level}.geojson"
            )
        ]

        if not resources:
            cls.logger.error(
                f"No resources found for {country_code} at admin level {admin_level}"
            )
            raise ValueError(
                "No resources found for the specified criteria. Please check your search parameters and try again."
            )

        cls.logger.info(f"Found resource: {resources[0].data.get('name', 'Unknown')}")

        try:
            cls.logger.info("Downloading and processing boundary data...")
            with tempfile.TemporaryDirectory() as tmpdir:
                url, local_path = resources[0].download(folder=tmpdir)
                cls.logger.debug(f"Downloaded file to temporary path: {local_path}")
                with open(local_path, "rb") as f:
                    gdf = gpd.read_file(f)

            gdf = cls._map_fields(gdf, "geoBoundaries", admin_level)
            boundaries = [
                AdminBoundary(**row_dict) for row_dict in gdf.to_dict("records")
            ]
            cls.logger.info(
                f"Successfully created {len(boundaries)} AdminBoundary objects"
            )
            return cls(boundaries=boundaries, level=admin_level)

        except (ValueError, HTTPError, FileNotFoundError) as e:
            cls.logger.warning(
                f"Error loading geoBoundaries data for {country_code} at admin level {admin_level}: {str(e)}"
            )
            cls.logger.info("Falling back to empty instance")
            return cls._create_empty_instance(
                country_code, admin_level, "geoBoundaries"
            )

    @classmethod
    def from_global_country_boundaries(cls, scale: str = "medium") -> "AdminBoundaries":
        """
        Load global country boundaries from Natural Earth Data.

        Args:
            scale (str): One of 'large', 'medium', 'small'.
                - 'large'  -> 10m
                - 'medium' -> 50m
                - 'small'  -> 110m
        Returns:
            AdminBoundaries: All country boundaries at admin_level=0
        """
        scale_map = {
            "large": "10m",
            "medium": "50m",
            "small": "110m",
        }
        if scale not in scale_map:
            raise ValueError(
                f"Invalid scale '{scale}'. Choose from 'large', 'medium', 'small'."
            )
        scale_folder = scale_map[scale]
        url = f"https://naciscdn.org/naturalearth/{scale_folder}/cultural/ne_{scale_folder}_admin_0_countries.zip"
        cls.logger.info(f"Loading Natural Earth global country boundaries from {url}")
        try:
            gdf = gpd.read_file(url)
            # Map fields to AdminBoundary schema
            boundaries = []
            for _, row in gdf.iterrows():
                iso_a3 = row.get("ISO_A3_EH") or row.get("ISO_A3") or row.get("ADM0_A3")
                name = row.get("NAME") or row.get("ADMIN") or row.get("SOVEREIGNT")
                geometry = row.get("geometry")
                if not iso_a3 or not name or geometry is None:
                    continue
                boundary = AdminBoundary(
                    id=iso_a3,
                    name=name,
                    geometry=geometry,
                    country_code=iso_a3,
                )
                boundaries.append(boundary)
            cls.logger.info(
                f"Loaded {len(boundaries)} country boundaries from Natural Earth."
            )
            return cls(boundaries=boundaries, level=0)
        except Exception as e:
            cls.logger.error(f"Failed to load Natural Earth global boundaries: {e}")
            raise

    @classmethod
    def create(
        cls,
        country_code: Optional[str] = None,
        admin_level: int = 0,
        data_store: Optional[DataStore] = None,
        path: Optional[Union[str, "Path"]] = None,
        **kwargs,
    ) -> "AdminBoundaries":
        """
        Factory method to create an AdminBoundaries instance using various data sources,
        depending on the provided parameters and global configuration.

        Loading Logic:
            1. If a `data_store` is provided and either a `path` is given or
               `global_config.ADMIN_BOUNDARIES_DATA_DIR` is set:
                - If `path` is not provided but `country_code` is, the path is constructed
                  using `global_config.get_admin_path()`.
                - Loads boundaries from the specified data store and path.

            2. If only `country_code` is provided (no data_store):
                - Attempts to load boundaries from GeoRepo (if available).
                - If GeoRepo is unavailable, attempts to load from GADM.
                - If GADM fails, falls back to geoBoundaries.
                - Raises an error if all sources fail.

            3. If neither `country_code` nor `data_store` is provided:
                - Raises a ValueError.

        Args:
            country_code (Optional[str]): ISO country code (2 or 3 letter) or country name.
            admin_level (int): Administrative level (0=country, 1=state/province, etc.).
            data_store (Optional[DataStore]): Optional data store instance for loading from existing data.
            path (Optional[Union[str, Path]]): Optional path to data file (used with data_store).
            **kwargs: Additional arguments passed to the underlying creation methods.

        Returns:
            AdminBoundaries: Configured instance.

        Raises:
            ValueError: If neither country_code nor (data_store, path) are provided,
                        or if country_code lookup fails.
            RuntimeError: If all data sources fail to load boundaries.

        Examples:
            # Load from a data store (path auto-generated if not provided)
            boundaries = AdminBoundaries.create(country_code="USA", admin_level=1, data_store=store)

            # Load from a specific file in a data store
            boundaries = AdminBoundaries.create(data_store=store, path="data.shp")

            # Load from online sources (GeoRepo, GADM, geoBoundaries)
            boundaries = AdminBoundaries.create(country_code="USA", admin_level=1)
        """
        cls.logger.info(
            f"Creating AdminBoundaries instance. Country: {country_code}, "
            f"admin level: {admin_level}, data_store provided: {data_store is not None}, "
            f"path provided: {path is not None}"
        )

        from_data_store = data_store is not None and (
            global_config.ADMIN_BOUNDARIES_DATA_DIR is not None or path is not None
        )

        # Validate input parameters
        if not country_code and not data_store:
            raise ValueError("Either country_code or data_store must be provided.")

        if from_data_store and not path and not country_code:
            raise ValueError(
                "If data_store is provided, either path or country_code must also be specified."
            )

        # Handle data store path first
        if from_data_store:
            iso3_code = None
            if country_code:
                try:
                    iso3_code = pycountry.countries.lookup(country_code).alpha_3
                except LookupError as e:
                    raise ValueError(f"Invalid country code '{country_code}': {e}")

            # Generate path if not provided
            if path is None and iso3_code:
                path = global_config.get_admin_path(
                    country_code=iso3_code,
                    admin_level=admin_level,
                )

            return cls.from_data_store(data_store, path, admin_level, **kwargs)

        # Handle country code path
        if country_code is not None:
            try:
                iso3_code = pycountry.countries.lookup(country_code).alpha_3
            except LookupError as e:
                raise ValueError(f"Invalid country code '{country_code}': {e}")

            # Try GeoRepo first
            if cls._try_georepo(iso3_code, admin_level):
                return cls.from_georepo(iso3_code, admin_level=admin_level)

            # Fallback to GADM
            try:
                cls.logger.info("Attempting to load from GADM.")
                return cls.from_gadm(iso3_code, admin_level, **kwargs)
            except Exception as e:
                cls.logger.warning(
                    f"GADM loading failed: {e}. Falling back to geoBoundaries."
                )

            # Final fallback to geoBoundaries
            try:
                return cls.from_geoboundaries(iso3_code, admin_level)
            except Exception as e:
                cls.logger.error(f"All data sources failed. geoBoundaries error: {e}")
                raise RuntimeError(
                    f"Failed to load administrative boundaries for {country_code} "
                    f"from all available sources (GeoRepo, GADM, geoBoundaries)."
                ) from e

        # This should never be reached due to validation above
        raise ValueError("Unexpected error: no valid data source could be determined.")

    @classmethod
    def _try_georepo(cls, iso3_code: str, admin_level: int) -> bool:
        """Helper method to test GeoRepo availability.

        Args:
            iso3_code: ISO3 country code
            admin_level: Administrative level

        Returns:
            bool: True if GeoRepo is available and working, False otherwise
        """
        try:
            from gigaspatial.handlers.unicef_georepo import GeoRepoClient

            client = GeoRepoClient()
            if client.check_connection():
                cls.logger.info("GeoRepo connection successful.")
                return True
            else:
                cls.logger.info("GeoRepo connection failed.")
                return False

        except ImportError:
            cls.logger.info("GeoRepo client not available (import failed).")
            return False
        except ValueError as e:
            cls.logger.warning(f"GeoRepo initialization failed: {e}")
            return False
        except Exception as e:
            cls.logger.warning(f"GeoRepo error: {e}")
            return False

    @classmethod
    def _create_empty_instance(
        cls, country_code: Optional[str], admin_level: int, source_type: str
    ) -> "AdminBoundaries":
        """Create an empty instance with the required schema structure."""
        # for to_geodataframe() to use later
        instance = cls(boundaries=[], level=admin_level, country_code=country_code)

        schema_fields = set(cls.get_schema_config()[source_type].keys())
        schema_fields.update(["geometry", "country_code", "id", "name", "name_en"])
        if admin_level > 0:
            schema_fields.add("parent_id")

        instance._empty_schema = list(schema_fields)
        return instance

    @classmethod
    def _map_fields(
        cls,
        gdf: gpd.GeoDataFrame,
        source: str,
        current_level: int,
    ) -> gpd.GeoDataFrame:
        """Map source fields to schema fields"""
        config = cls.get_schema_config().get(source, {})
        parent_level = current_level - 1

        field_mapping = {}
        for k, v in config.items():
            if "{parent_level}" in v:
                field_mapping[v.format(parent_level=parent_level)] = k
            elif "{level}" in v:
                field_mapping[v.format(level=current_level)] = k
            else:
                field_mapping[v] = k

        return gdf.rename(columns=field_mapping)

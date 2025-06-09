from pydantic import BaseModel, Field
from typing import Optional, ClassVar, Union, Dict, List
import geopandas as gpd
from pathlib import Path
from urllib.error import HTTPError
from shapely.geometry import Polygon, MultiPolygon, shape
import pycountry

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.readers import read_dataset
from gigaspatial.config import config


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
        # extra = "allow"
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

    logger: ClassVar = config.get_logger("AdminBoundaries")

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
    }

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
    def create(
        cls,
        country_code: Optional[str] = None,
        admin_level: int = 0,
        data_store: Optional[DataStore] = None,
        path: Optional[Union[str, "Path"]] = None,
        **kwargs,
    ) -> "AdminBoundaries":
        """Factory method to create AdminBoundaries instance from either GADM or data store."""
        cls.logger.info(
            f"Creating AdminBoundaries instance. Country: {country_code}, admin level: {admin_level}, data_store provided: {data_store is not None}, path provided: {path is not None}"
        )
        iso3_code = pycountry.countries.lookup(country_code).alpha_3
        if data_store is not None:
            if path is None:
                if country_code is None:
                    ValueError(
                        "If data_store is provided, path or country_code must also be specified."
                    )
                path = config.get_admin_path(
                    country_code=iso3_code,
                    admin_level=admin_level,
                )
            return cls.from_data_store(data_store, path, admin_level, **kwargs)
        elif country_code is not None:
            from gigaspatial.handlers.unicef_georepo import GeoRepoClient

            try:
                client = GeoRepoClient()
                if client.check_connection():
                    cls.logger.info("GeoRepo connection successful.")
                    return cls.from_georepo(
                        iso3_code,
                        admin_level=admin_level,
                    )
            except ValueError as e:
                cls.logger.warning(
                    f"GeoRepo initialization failed: {str(e)}. Falling back to GADM."
                )
            except Exception as e:
                cls.logger.warning(f"GeoRepo error: {str(e)}. Falling back to GADM.")

            return cls.from_gadm(iso3_code, admin_level, **kwargs)
        else:
            raise ValueError(
                "Either country_code or (data_store, path) must be provided."
            )

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

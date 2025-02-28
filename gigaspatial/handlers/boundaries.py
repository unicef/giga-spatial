from pydantic import BaseModel, Field
from typing import Optional, ClassVar, Union, Dict, List
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon

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
        cls, country_code: str, admin_level: Optional[int] = 0, **kwargs
    ) -> gpd.GeoDataFrame:
        """Load GADM data from URL"""
        url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{country_code}_{admin_level}.json"
        gdf = gpd.read_file(url)

        gdf = cls._map_fields(gdf, "gadm", admin_level)

        if admin_level == 0:
            gdf["country_code"] = gdf["id"]
            gdf["name"] = gdf["COUNTRY"]
        elif admin_level == 1:
            gdf["country_code"] = gdf["parent_id"]

        return cls(
            boundaries=[
                AdminBoundary(**row_dict) for row_dict in gdf.to_dict("records")
            ],
            level=admin_level,
        )

    @classmethod
    def from_data_store(
        cls,
        data_store: DataStore,
        path: Union[str, "Path"],
        admin_level: Optional[int] = 0,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """Load internal data from datastore"""
        gdf = read_dataset(data_store, str(path), **kwargs)
        gdf = cls._map_fields(gdf, "internal", admin_level)

        if admin_level == 0:
            gdf["id"] = gdf["country_code"]
        else:
            gdf["parent_id"] = gdf["id"].apply(lambda x: x[:-3])

        return cls(
            boundaries=[
                AdminBoundary(**row_dict) for row_dict in gdf.to_dict("records")
            ],
            level=admin_level,
        )

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
        if data_store is not None:
            if path is None:
                if country_code is None:
                    ValueError(
                        "If data_store is provided, path or country_code must also be specified."
                    )
                path = config.get_admin_path(
                    country_code=country_code, admin_level=admin_level
                )
            return cls.from_data_store(data_store, path, admin_level, **kwargs)
        elif country_code is not None:
            return cls.from_gadm(country_code, admin_level, **kwargs)
        else:
            raise ValueError(
                "Either country_code or (data_store, path) must be provided."
            )

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
        return gpd.GeoDataFrame(
            [boundary.model_dump() for boundary in self.boundaries],
            geometry="geometry",
            crs=4326,
        )

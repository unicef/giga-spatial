import pandas as pd
import geopandas as gpd
import mercantile
from shapely.geometry import box
from shapely.strtree import STRtree
from shapely import MultiPolygon, Polygon, Point
import json
from pathlib import Path
from pydantic import BaseModel, Field, PrivateAttr
from typing import List, Union, Iterable, Optional, Tuple

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore


class MercatorTiles(BaseModel):
    zoom_level: int = Field(..., ge=0, le=20)
    quadkeys: List[str] = Field(default_factory=list)
    data_store: DataStore = Field(default_factory=LocalDataStore, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_quadkeys(cls, quadkeys: List[str]) -> "MercatorTiles":
        """Create MercatorTiles from list of quadkeys."""
        return MercatorTiles(zoom_level=len(quadkeys[0]), quadkeys=quadkeys)

    @classmethod
    def from_points(
        cls, points: List[Union[Point, Tuple[float, float]]], zoom_level: int
    ) -> "MercatorTiles":
        """Create MercatorTiles from a list of points or lat-lon pairs."""
        quadkeys = {
            (
                mercantile.quadkey(mercantile.tile(p.x, p.y, zoom_level))
                if isinstance(p, Point)
                else mercantile.quadkey(mercantile.tile(p[1], p[0], zoom_level))
            )
            for p in points
        }
        return cls(zoom_level=zoom_level, quadkeys=list(quadkeys))

    @staticmethod
    def from_bounds(
        xmin: float, ymin: float, xmax: float, ymax: float, zoom_level: int
    ) -> "MercatorTiles":
        """Create MercatorTiles from boundary coordinates."""
        return MercatorTiles(
            zoom_level=zoom_level,
            quadkeys=[
                mercantile.quadkey(tile)
                for tile in mercantile.tiles(xmin, ymin, xmax, ymax, zoom_level)
            ],
        )

    @staticmethod
    def from_polygon(
        polygon: Polygon, zoom_level: int, predicate: str = "intersects"
    ) -> "MercatorTiles":
        """Create MercatorTiles from a polygon."""
        tiles = list(mercantile.tiles(*polygon.bounds, zoom_level))
        quadkeys_boxes = [
            (mercantile.quadkey(t), box(*mercantile.bounds(t))) for t in tiles
        ]
        quadkeys, boxes = zip(*quadkeys_boxes) if quadkeys_boxes else ([], [])

        if not boxes:
            return MercatorTiles(zoom_level=zoom_level, quadkeys=[])

        s = STRtree(boxes)
        result = s.query(polygon, predicate=predicate)
        return MercatorTiles(
            zoom_level=zoom_level, quadkeys=[quadkeys[i] for i in result]
        )

    @staticmethod
    def from_multipolygon(
        multipolygon: MultiPolygon, zoom_level: int, predicate: str = "intersects"
    ) -> "MercatorTiles":
        """Create MercatorTiles from a MultiPolygon."""
        quadkeys = {
            quadkey
            for geom in multipolygon.geoms
            for quadkey in MercatorTiles.from_polygon(
                geom, zoom_level, predicate
            ).quadkeys
        }
        return MercatorTiles(zoom_level=zoom_level, quadkeys=list(quadkeys))

    @classmethod
    def from_json(
        cls, data_store: DataStore, file: Union[str, Path], **kwargs
    ) -> "MercatorTiles":
        """Load MercatorTiles from a JSON file."""
        with data_store.open(str(file), "r") as f:
            data = json.load(f)
            if isinstance(data, list):  # If file contains only quadkeys
                data = {
                    "zoom_level": len(data[0]) if data else 0,
                    "quadkeys": data,
                    **kwargs,
                }
            else:
                data.update(kwargs)
            instance = cls(**data)
            instance.data_store = data_store
            return instance

    def filter_quadkeys(self, quadkeys: Iterable[str]) -> "MercatorTiles":
        """Filter quadkeys by a given set of quadkeys."""
        return MercatorTiles(
            zoom_level=self.zoom_level,
            quadkeys=list(set(self.quadkeys) & set(quadkeys)),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with quadkey and centroid coordinates."""
        tiles_data = [mercantile.quadkey_to_tile(q) for q in self.quadkeys]
        bounds_data = [mercantile.bounds(tile) for tile in tiles_data]

        centroids = [
            (
                (bounds.south + bounds.north) / 2,  # latitude
                (bounds.west + bounds.east) / 2,  # longitude
            )
            for bounds in bounds_data
        ]

        return pd.DataFrame(
            {
                "quadkey": self.quadkeys,
                "latitude": [c[0] for c in centroids],
                "longitude": [c[1] for c in centroids],
            }
        )

    def to_geoms(self) -> List[box]:
        return [
            box(*mercantile.bounds(mercantile.quadkey_to_tile(q)))
            for q in self.quadkeys
        ]

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert to GeoPandas GeoDataFrame."""
        return gpd.GeoDataFrame(
            {"quadkey": self.quadkeys, "geometry": self.to_geoms()}, crs="EPSG:4326"
        )

    def save(self, file: Union[str, Path], format: str = "json") -> None:
        """Save MercatorTiles to file in specified format."""
        with self.data_store.open(str(file), "wb" if format == "parquet" else "w") as f:
            if format == "parquet":
                self.to_geodataframe().to_parquet(f, index=False)
            elif format == "geojson":
                f.write(self.to_geodataframe().to_json(drop_id=True))
            elif format == "json":
                json.dump(self.quadkeys, f)
            else:
                raise ValueError(f"Unsupported format: {format}")

    def __len__(self) -> int:
        return len(self.quadkeys)


class CountryMercatorTiles(MercatorTiles):
    """MercatorTiles specialized for country-level operations.

    This class extends MercatorTiles to work specifically with country boundaries.
    It can only be instantiated through the create() classmethod.
    """

    _country: str = PrivateAttr()

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "CountryMercatorTiles cannot be instantiated directly. "
            "Use CountryMercatorTiles.create() instead."
        )

    @classmethod
    def create(
        cls,
        country: str,
        zoom_level: int,
        data_store: Optional[DataStore] = None,
        country_geom_path: Optional[Union[str, Path]] = None,
        predicate: str = "intersects",
    ) -> "CountryMercatorTiles":
        """Create CountryMercatorTiles for a specific country."""
        instance = super().__new__(cls)
        super(CountryMercatorTiles, instance).__init__(
            zoom_level=zoom_level,
            quadkeys=[],
            data_store=data_store or LocalDataStore(),
        )

        instance._country = country

        country_geom = instance._load_country_geometry(
            country, data_store, country_geom_path
        )

        tiles = (
            MercatorTiles.from_multipolygon
            if isinstance(country_geom, MultiPolygon)
            else MercatorTiles.from_polygon
        )(country_geom, zoom_level, predicate)

        instance.quadkeys = tiles.quadkeys
        return instance

    @property
    def country(self) -> str:
        """Get country identifier."""
        return self._country

    def _load_country_geometry(
        self,
        country,
        data_store: Optional[DataStore] = None,
        path: Optional[Union[str, Path]] = None,
    ) -> Union[Polygon, MultiPolygon]:
        """Load country boundary geometry from DataStore or GADM."""
        from gigaspatial.handlers.boundaries import AdminBoundaries

        gdf_admin0 = AdminBoundaries.create(
            country_code=country, admin_level=0, data_store=data_store, path=path
        ).to_geodataframe()

        return gdf_admin0.geometry.iloc[0]

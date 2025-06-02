import pandas as pd
import geopandas as gpd
import mercantile
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
from shapely import MultiPolygon, Polygon, Point
import json
from pathlib import Path
from pydantic import BaseModel, Field, PrivateAttr
from typing import List, Union, Iterable, Optional, Tuple
import pycountry

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore


class MercatorTiles(BaseModel):
    zoom_level: int = Field(..., ge=0, le=20)
    quadkeys: List[str] = Field(default_factory=list)
    data_store: DataStore = Field(default_factory=LocalDataStore, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_quadkeys(cls, quadkeys: List[str]):
        """Create MercatorTiles from list of quadkeys."""
        return cls(zoom_level=len(quadkeys[0]), quadkeys=set(quadkeys))

    @classmethod
    def from_bounds(
        cls, xmin: float, ymin: float, xmax: float, ymax: float, zoom_level: int
    ):
        """Create MercatorTiles from boundary coordinates."""
        return cls(
            zoom_level=zoom_level,
            quadkeys=[
                mercantile.quadkey(tile)
                for tile in mercantile.tiles(xmin, ymin, xmax, ymax, zoom_level)
            ],
        )

    @classmethod
    def from_spatial(
        cls,
        source: Union[
            BaseGeometry,
            gpd.GeoDataFrame,
            List[Union[Point, Tuple[float, float]]],  # points
        ],
        zoom_level: int,
        predicate: str = "intersects",
        **kwargs,
    ):
        if isinstance(source, gpd.GeoDataFrame):
            if source.crs != "EPSG:4326":
                source = source.to_crs("EPSG:4326")
            source = source.geometry.unary_union

        if isinstance(source, BaseGeometry):
            return cls.from_geometry(
                geometry=source, zoom_level=zoom_level, predicate=predicate, **kwargs
            )
        elif isinstance(source, Iterable) and all(
            len(pt) == 2 or isinstance(pt, Point) for pt in source
        ):
            return cls.from_points(geometry=source, zoom_level=zoom_level, **kwargs)
        else:
            raise

    @classmethod
    def from_geometry(
        cls,
        geometry: BaseGeometry,
        zoom_level: int,
        predicate: str = "intersects",
        **kwargs,
    ):
        """Create MercatorTiles from a polygon."""
        tiles = list(mercantile.tiles(*geometry.bounds, zoom_level))
        quadkeys_boxes = [
            (mercantile.quadkey(t), box(*mercantile.bounds(t))) for t in tiles
        ]
        quadkeys, boxes = zip(*quadkeys_boxes) if quadkeys_boxes else ([], [])

        if not boxes:
            return MercatorTiles(zoom_level=zoom_level, quadkeys=[])

        s = STRtree(boxes)
        result = s.query(geometry, predicate=predicate)
        return cls(
            zoom_level=zoom_level, quadkeys=[quadkeys[i] for i in result], **kwargs
        )

    @classmethod
    def from_points(
        cls, points: List[Union[Point, Tuple[float, float]]], zoom_level: int, **kwargs
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
        return cls(zoom_level=zoom_level, quadkeys=list(quadkeys), **kwargs)

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

    country: str = Field(..., exclude=True)

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
        predicate: str = "intersects",
        data_store: Optional[DataStore] = None,
        country_geom_path: Optional[Union[str, Path]] = None,
    ):
        """Create CountryMercatorTiles for a specific country."""
        from gigaspatial.handlers.boundaries import AdminBoundaries

        instance = super().__new__(cls)
        super(CountryMercatorTiles, instance).__init__(
            zoom_level=zoom_level,
            quadkeys=[],
            data_store=data_store or LocalDataStore(),
            country=pycountry.countries.lookup(country).alpha_3,
        )

        country_geom = (
            AdminBoundaries.create(
                country_code=country,
                data_store=data_store,
                path=country_geom_path,
            )
            .boundaries[0]
            .geometry
        )

        tiles = MercatorTiles.from_geometry(country_geom, zoom_level, predicate)

        instance.quadkeys = tiles.quadkeys
        return instance

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from datetime import datetime
import json
import requests
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Literal, Optional

from gigaspatial.grid.mercator_tiles import CountryMercatorTiles
from gigaspatial.core.io.readers import read_dataset
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.config import config


class OoklaSpeedtestTileConfig(BaseModel):
    service_type: Literal["fixed", "mobile"]
    year: int
    quarter: int
    data_store: DataStore = Field(default_factory=LocalDataStore, exclude=True)
    base_path: Path = Field(
        default=config.get_path("ookla_speedtest", "bronze"), exclude=True
    )

    class Config:
        arbitrary_types_allowed = True

    @property
    def quarter_start(self):
        if not 1 <= self.quarter <= 4:
            raise ValueError("Quarter must be within [1, 2, 3, 4]")

        month = [1, 4, 7, 10]
        return datetime(self.year, month[self.quarter - 1], 1)

    @property
    def tile_name(self):
        return f"{self.quarter_start:%Y-%m-%d}_performance_{self.service_type}_tiles.parquet"

    @property
    def tile_url(self):
        base_url = "https://ookla-open-data.s3.amazonaws.com/parquet/performance"
        qs_dt = self.quarter_start
        return f"{base_url}/type={self.service_type}/year={qs_dt:%Y}/quarter={self.quarter}/{qs_dt:%Y-%m-%d}_performance_{self.service_type}_tiles.parquet"

    def download_tile(self):
        path = str(self.base_path / self.tile_name)
        if not self.data_store.file_exists(path):
            response = requests.get(self.tile_url)
            response.raise_for_status()
            self.data_store.write_file(path, response.content)

    def read_tile(self):
        path = str(self.base_path / self.tile_name)

        if self.data_store.file_exists(path):
            df = read_dataset(self.data_store, path)
            return df
        else:
            self.download_tile()
            df = self.read_tile()
            return df


class OoklaSpeedtestConfig(BaseModel):
    tiles: List[OoklaSpeedtestTileConfig] = Field(default_factory=list)

    @classmethod
    def from_available_ookla_tiles(
        cls, data_store: DataStore = None, base_path: Path = None
    ):
        data_store = data_store or LocalDataStore()
        base_path = base_path or config.get_path("ookla_speedtest", "bronze")

        # first data year
        start_year = 2019
        # max data year
        max_year = datetime.today().year
        max_quarter = np.floor((datetime.today().month - 1) / 3)
        if max_quarter == 0:
            max_year -= 1
            max_quarter = 4

        ookla_tiles = []
        for year in range(start_year, max_year + 1):
            for quarter in range(1, 5):
                if year == max_year and quarter > max_quarter:
                    continue
                for type in ["fixed", "mobile"]:
                    ookla_tiles.append(
                        OoklaSpeedtestTileConfig(
                            service_type=type,
                            year=year,
                            quarter=quarter,
                            data_store=data_store,
                            base_path=base_path,
                        )
                    )
        return cls(tiles=ookla_tiles)


class OoklaSpeedtestTile(BaseModel):
    quadkey: str
    tile: str
    avg_d_kbps: float
    avg_u_kbps: float
    avg_lat_ms: float
    avg_lat_down_ms: Optional[float] = None
    avg_lat_up_ms: Optional[float] = None
    tests: int
    devices: int

    model_config = ConfigDict(extra="allow")


class CountryOoklaTiles(BaseModel):
    country: str
    service_type: str
    year: int
    quarter: int
    quadkeys: List[OoklaSpeedtestTile]

    @staticmethod
    def from_country(country, ookla_tile_config: OoklaSpeedtestTileConfig):
        # load country zoom level 16 quadkeys
        country_tiles = CountryMercatorTiles.create(country, 16)

        # read ookla tiles for the config
        ookla_tiles = ookla_tile_config.read_tile()

        # filter country tiles by ookla tile quadkeys
        country_ookla_tiles = country_tiles.filter_quadkeys(ookla_tiles.quadkey)
        if len(country_ookla_tiles):
            df_quadkeys = country_ookla_tiles.to_dataframe().merge(
                ookla_tiles, on="quadkey", how="left"
            )
            return CountryOoklaTiles(
                country=country,
                service_type=ookla_tile_config.service_type,
                year=ookla_tile_config.year,
                quarter=ookla_tile_config.quarter,
                quadkeys=[
                    OoklaSpeedtestTile(**tile_dict)
                    for tile_dict in df_quadkeys.to_dict("records")
                ],
            )
        else:
            return CountryOoklaTiles(
                country=country,
                service_type=ookla_tile_config.service_type,
                year=ookla_tile_config.year,
                quarter=ookla_tile_config.quarter,
                quadkeys=[],
            )

    def to_dataframe(self):
        if len(self):
            return pd.DataFrame([q.model_dump() for q in self.quadkeys])
        else:
            return pd.DataFrame(
                columns=[
                    "quadkey",
                    "tile",
                    "avg_d_kbps",
                    "avg_u_kbps",
                    "avg_lat_ms",
                    "avg_lat_down_ms",
                    "avg_lat_up_ms",
                    "tests",
                    "devices",
                ]
            )

    def to_geodataframe(self):
        if len(self):
            df = self.to_dataframe()
            df["geometry"] = df.tile.apply(wkt.loads)
            df.drop(columns="tile", inplace=True)
            return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        else:
            return gpd.GeoDataFrame(
                columns=[
                    "quadkey",
                    "avg_d_kbps",
                    "avg_u_kbps",
                    "avg_lat_ms",
                    "avg_lat_down_ms",
                    "avg_lat_up_ms",
                    "tests",
                    "devices",
                    "geometry",
                ]
            )

    def __len__(self):
        return len(self.quadkeys)

"""
Tropical cyclone tracks from the ECMWF essential portal, in memory.

``EcmwfTropicalCycloneFetcher`` returns positions for one forecast cycle and
does not write a data store. The stored form of the same tracks is
``EcmwfTropicalCycloneHandler``.

Per-storm BUFR files are listed at https://essential.ecmwf.int/. The heading
centre ECMF is the IFS deterministic track and ECEP is the IFS ENS track.
AIFS Single and AIFS ENS, and an IFS system with no essential file, use the
open-data BUFR track (``type=tf``).

IFS tracks run to 360 h at 00 and 12 UTC and to 144 h at 06 and 18 UTC.
AIFS tracks run to 360 h at every cycle.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from gigaspatial.fetchers.ecmwf.catalogue import (
    ATTRIBUTION,
    ESSENTIAL_ROOT,
    OPEN_DATA_ROOT,
    expand_systems,
    parse_date,
    parse_time,
)
from gigaspatial.handlers.base.config import source_geometry
from gigaspatial.handlers.ecmwf.tropical_cyclone import (
    TRACK_COLUMNS,
    download_track_payloads,
    points_from_payloads,
)


class EcmwfTropicalCycloneFetcher:
    """
    Fetch ECMWF tropical-cyclone tracks for one cycle.

    Args:
        essential_root: Listing root for https://essential.ecmwf.int/.
        open_data_root: Open-data root used when a system has no essential file.
    """

    def __init__(
        self,
        essential_root: str = ESSENTIAL_ROOT,
        open_data_root: str = OPEN_DATA_ROOT,
    ):
        self.essential_root = essential_root
        self.open_data_root = open_data_root

    def fetch(self, source=None, *, system: str = "all", date, time: int = 0) -> gpd.GeoDataFrame:
        """
        Return track positions for the requested systems.

        Args:
            source: Optional place. Positions outside it are dropped.
            system: ``all`` or one of ``ifs``, ``ifs-ens``, ``aifs-single``,
                ``aifs-ens``.
            date: Forecast run date.
            time: Cycle hour, 0, 6, 12, or 18.

        Returns:
            A GeoDataFrame in EPSG:4326. One row is one storm position.
            ``maximum_wind`` is the track's reported 10 m wind, in m/s, when
            the BUFR message was decoded.

        Raises:
            FileNotFoundError: If a requested system has no track file.
            RuntimeError: If a file was downloaded and no position could be read.
        """
        when = parse_date(date)
        hour = parse_time(time)
        frames = []
        for name in expand_systems(system):
            payloads = download_track_payloads(
                when, hour, name, self.essential_root, self.open_data_root
            )
            frame = points_from_payloads(payloads, name)
            located = frame.dropna(subset=["latitude", "longitude"])
            if located.empty:
                raise RuntimeError(
                    f"No positions could be read for {name} on {when.isoformat()} "
                    f"{hour:02d} UTC. Install ecCodes to decode a combined BUFR trajectory."
                )
            frames.append(located)
        table = pd.concat(frames, ignore_index=True)
        geometry = [Point(lon, lat) for lon, lat in zip(table.longitude, table.latitude)]
        tracks = gpd.GeoDataFrame(table, geometry=geometry, crs="EPSG:4326")
        tracks.attrs["attribution"] = ATTRIBUTION
        if source is None:
            return tracks
        place = source if isinstance(source, BaseGeometry) else source_geometry(source)
        return tracks[tracks.geometry.intersects(place)].reset_index(drop=True)


__all__ = ["EcmwfTropicalCycloneFetcher", "TRACK_COLUMNS"]

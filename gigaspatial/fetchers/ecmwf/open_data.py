"""
ECMWF real-time open data, returned in memory.

Publisher: ECMWF.
Page: https://www.ecmwf.int/en/forecasts/datasets/open-data
Licence: Creative Commons CC-BY-4.0. The files are a rolling archive of the
latest forecast cycles on data.ecmwf.int. 0.1° IFS files are published about
two hours after the matching 0.25° files.

``EcmwfOpenDataFetcher`` downloads GRIB2 for one of the four systems:

- IFS deterministic (``ifs``): 00/12 UTC through 360 h, 06/18 UTC through 90 h.
- IFS ENS (``ifs-ens``): 00/12 UTC through 360 h, 06/18 UTC through 144 h.
- AIFS Single (``aifs-single``) and AIFS ENS (``aifs-ens``): every cycle
  through 360 h, at 0.25° only.

Omitting ``param`` downloads every field in the step file. That is the full
open-data variable set for the system, including pressure levels and, at
0.1°, fields that are not in the core 0.25° list. Naming a parameter uses
the file's index and HTTP range requests, so a wind query does not pull the
pressure-level file's other messages when they can be skipped.

``fetch_wind`` reads 10u and 10v, adds wind speed, and reads maximum 10 m
gust (``10fg`` through 90 h, ``10fg3`` after that) for the IFS systems.
Passed a track table, it crops each storm to the box around its own track
and keeps gust on those same cells.
"""

from __future__ import annotations

import os
import tempfile
from typing import Iterable, List, Optional, Sequence, Union

import pandas as pd
import requests
import xarray as xr
from shapely.geometry.base import BaseGeometry

from gigaspatial.fetchers.ecmwf.catalogue import (
    ATTRIBUTION,
    GUST_SHORT_NAMES,
    OPEN_DATA_ROOT,
    SYSTEM_LABELS,
    canonical_system,
    check_resol,
    check_step,
    field_urls,
    forecast_steps,
    gust_param,
    index_url,
    merge_ranges,
    parse_date,
    parse_index,
    parse_time,
    select_messages,
    variable_names,
)
from gigaspatial.fetchers.ecmwf.wind import (
    DEFAULT_MARGIN_DEGREES,
    StormWindow,
    add_gust,
    add_wind_speed,
    crop_dataset,
    pair_storm_fields,
)
from gigaspatial.handlers.base.config import source_geometry


class EcmwfOpenDataFetcher:
    """
    Download ECMWF open-data forecast fields and return them in memory.

    The result is an xarray Dataset on the native lat/lon grid. A source
    crops that grid to the source bounds. Values are not resampled.

    Args:
        root: Open-data root. Defaults to ``https://data.ecmwf.int/forecasts``.
        timeout: Seconds to wait for each response.
        session: ``requests`` session. Tests pass a stub here.
    """

    def __init__(
        self,
        root: str = OPEN_DATA_ROOT,
        timeout: float = 120,
        session: Optional[requests.Session] = None,
    ):
        self.root = root.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    @staticmethod
    def variables(system: str) -> tuple:
        """Return the documented atmospheric short names for ``system``."""
        return variable_names(system)

    @staticmethod
    def steps(system: str, time: int) -> tuple:
        """Return the published forecast steps, in hours."""
        return forecast_steps(system, time)

    def fetch(
        self,
        source=None,
        *,
        system: str = "ifs",
        date,
        time: int = 0,
        step: Union[int, Sequence[int], None] = None,
        resol: str = "0p25",
        param: Optional[Iterable[str]] = None,
        levtype: Optional[str] = None,
        number: Optional[Iterable] = None,
    ) -> xr.Dataset:
        """
        Download forecast fields and open them as one dataset.

        Args:
            source: Country, geometry, points, or a frame. Omit it for the
                global grid. A source selects native cells inside its bounds.
            system: ``ifs``, ``ifs-ens``, ``aifs-single``, or ``aifs-ens``.
            date: Forecast run date.
            time: Cycle hour, 0, 6, 12, or 18.
            step: One step, several steps, or None for every published step.
            resol: ``0p25`` or ``0p1``. 0.1° is IFS only.
            param: Short names to keep. None downloads every field in the file.
            levtype: ``sfc`` or ``pl``. None keeps both. At 0.1°, ``sfc`` is
                the surface file, which also holds soil levels.
            number: Ensemble members to keep when ``param`` is set.

        Returns:
            The forecast on its native grid, cropped to ``source`` when given.

        Raises:
            ValueError: If the system, resolution, or step is not published.
            requests.RequestException: If a download fails.
            ImportError: If cfgrib is not installed.
        """
        raw = self.download_fields(
            system=system,
            date=date,
            time=time,
            step=step,
            resol=resol,
            param=param,
            levtype=levtype,
            number=number,
        )
        dataset = open_grib_bytes(raw)
        dataset.attrs["attribution"] = ATTRIBUTION
        dataset.attrs["system"] = canonical_system(system)
        dataset.attrs["resolution"] = check_resol(system, resol)
        if source is not None:
            dataset = crop_dataset(dataset, _geometry(source))
        return dataset

    def download_fields(
        self,
        *,
        system: str = "ifs",
        date,
        time: int = 0,
        step: Union[int, Sequence[int], None] = None,
        resol: str = "0p25",
        param: Optional[Iterable[str]] = None,
        levtype: Optional[str] = None,
        number: Optional[Iterable] = None,
    ) -> bytes:
        """
        Download the GRIB2 messages for one run and return them concatenated.

        ``param`` omitted downloads each step file whole. A parameter list
        reads the ``.index`` file and requests only those byte ranges.
        """
        when = parse_date(date)
        hour = parse_time(time)
        system = canonical_system(system)
        resol = check_resol(system, resol)
        steps = _steps(system, hour, step)
        names = None if param is None else [str(item) for item in param]
        chunks = []
        for one in steps:
            for url in field_urls(
                when, hour, system, resol, one, root=self.root, levtype=levtype
            ):
                if _whole_file(names, levtype, resol):
                    chunks.append(self._read(url))
                else:
                    chunks.append(
                        self._read_selection(url, names, one, levtype, number)
                    )
        return b"".join(chunks)

    def fetch_wind(
        self,
        source=None,
        tracks: Optional[pd.DataFrame] = None,
        *,
        margin_degrees: float = DEFAULT_MARGIN_DEGREES,
        system: str = "ifs",
        date,
        time: int = 0,
        step: Union[int, Sequence[int], None] = None,
        resol: str = "0p25",
        number: Optional[Iterable] = None,
    ) -> Union[xr.Dataset, List[StormWindow]]:
        """
        Download 10u, 10v, and gust, and add 10 m wind speed.

        Gust is requested for the IFS systems. AIFS open data has 10u and 10v
        and no maximum-gust field, so those windows carry wind speed only.

        Args:
            source: Optional place. With ``tracks``, storms whose area misses
                this place are dropped. Without ``tracks``, the grid is cropped
                to the source.
            tracks: Track table with a storm id, latitude, and longitude.
                Each storm is cropped to the box around its track.
            margin_degrees: Padding around the track, in degrees.
            system, date, time, step, resol, number: The forecast to read.
                See ``fetch``.

        Returns:
            A dataset when ``tracks`` is omitted, otherwise one
            ``StormWindow`` per storm (and ensemble member, when present).
        """
        when = parse_date(date)
        hour = parse_time(time)
        system = canonical_system(system)
        steps = _steps(system, hour, step)
        params = ["10u", "10v"]
        if system in {"ifs", "ifs-ens"}:
            params.extend(gust_param(one) for one in steps)
            # Ask for both index names so a mixed step list still matches.
            params.extend(GUST_SHORT_NAMES)
        params = list(dict.fromkeys(params))
        dataset = self.fetch(
            None,
            system=system,
            date=when,
            time=hour,
            step=steps,
            resol=resol,
            param=params,
            levtype="sfc",
            number=number,
        )
        dataset = add_gust(add_wind_speed(dataset))
        dataset.attrs["system_label"] = SYSTEM_LABELS[system]
        if tracks is None:
            if source is not None:
                return crop_dataset(dataset, _geometry(source))
            return dataset
        windows = pair_storm_fields(tracks, dataset, margin_degrees)
        if source is None:
            return windows
        place = _geometry(source)
        return [window for window in windows if window.area.intersects(place)]

    def _read(self, url: str) -> bytes:
        response = self.session.get(url, timeout=self.timeout)
        try:
            response.raise_for_status()
            return response.content
        finally:
            response.close()

    def _read_selection(self, url, params, step, levtype, number) -> bytes:
        index_body = self._read(index_url(url))
        rows = select_messages(
            parse_index(index_body.decode("utf-8")),
            param=params,
            step=step,
            levtype=levtype,
            number=number,
        )
        if params:
            found = {str(row.get("param")) for row in rows}
            required = [name for name in params if name not in GUST_SHORT_NAMES]
            missing = [name for name in required if name not in found]
            if missing:
                raise ValueError(f"No index entries at {url} for {missing}.")
        elif not rows:
            raise ValueError(f"No index entries at {url} match the request.")
        if not rows:
            return b""
        chunks = []
        for offset, length in merge_ranges(rows):
            end = offset + length - 1
            response = self.session.get(
                url,
                headers={"Range": f"bytes={offset}-{end}"},
                timeout=self.timeout,
            )
            try:
                response.raise_for_status()
                chunks.append(response.content)
            finally:
                response.close()
        return b"".join(chunks)


def _whole_file(params, levtype, resol: str) -> bool:
    if params:
        return False
    if levtype not in (None, "", "all") and resol == "0p25":
        return False
    return True


def _steps(system: str, hour: int, step) -> List[int]:
    if step is None:
        return list(forecast_steps(system, hour))
    if isinstance(step, (str, int)):
        values = [step]
    else:
        values = list(step)
    return [check_step(system, hour, int(item)) for item in values]


def _geometry(source) -> BaseGeometry:
    if isinstance(source, BaseGeometry):
        return source
    return source_geometry(source)


def open_grib_bytes(raw: bytes) -> xr.Dataset:
    """Open concatenated GRIB2 with cfgrib and merge the hypercubes."""
    if not raw:
        raise ValueError("ECMWF download was empty.")
    try:
        import cfgrib
    except ImportError as exc:
        raise ImportError(
            "Reading ECMWF GRIB2 needs cfgrib and the ECMWF ecCodes library. "
            "Install them with pip install 'giga-spatial[ecmwf]'."
        ) from exc
    handle = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
    path = handle.name
    opened = []
    try:
        handle.write(raw)
        handle.close()
        opened = list(cfgrib.open_datasets(path))
        if not opened:
            raise ValueError("GRIB2 file has no datasets.")
        loaded = [dataset.load() for dataset in opened]
    finally:
        for dataset in opened:
            dataset.close()
        os.unlink(path)
    if len(loaded) == 1:
        return loaded[0]
    return xr.merge(loaded, compat="override", join="outer")

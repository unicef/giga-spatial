"""
ECMWF open-data catalogue: models, steps, variables, and file URLs.

The atmospheric products are the public subset described at
https://www.ecmwf.int/en/forecasts/datasets/open-data
Licence: Creative Commons CC-BY-4.0.

IFS fields are published at 0.25° and, from 20 October 2026, at 0.1°.
AIFS Single and AIFS ENS stay at 0.25°. At 0.1° each step is two files,
surface (``-sfc``, which also holds soil levels) and pressure level (``-pl``).
The 0.25° file is one combined GRIB2. Tropical-cyclone tracks are BUFR4
and are published at 0.25° for every system.

Forecast horizons follow the open-data schedule:

- IFS deterministic, 00 and 12 UTC: 0–144 h every 3 h, then 150–360 h every 6 h.
  06 and 18 UTC: 0–90 h every 3 h.
- IFS ENS, 00 and 12 UTC: the same 0–360 h schedule.
  06 and 18 UTC: 0–144 h every 3 h.
- AIFS Single and AIFS ENS, every cycle: 0–360 h every 6 h.

Track files use one step in the name, the end of the trajectory: 360 h at
00 and 12 UTC, 144 h at 06 and 18 UTC for both IFS systems, and 360 h for
every AIFS cycle.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from typing import Iterable, Mapping, Optional, Sequence, Tuple

OPEN_DATA_PAGE = "https://www.ecmwf.int/en/forecasts/datasets/open-data"
OPEN_DATA_ROOT = "https://data.ecmwf.int/forecasts"
ESSENTIAL_ROOT = "https://essential.ecmwf.int"
LICENCE = "CC-BY-4.0"
ATTRIBUTION = (
    "ECMWF open data, licensed under Creative Commons CC-BY-4.0 "
    "(https://www.ecmwf.int/en/forecasts/datasets/open-data)."
)

# IFS cycle 50r1. From this date the 06 and 18 UTC deterministic runs are
# stored under stream=oper. Earlier runs used stream=scda.
IFS_50R1 = date(2026, 5, 12)
RUN_HOURS = (0, 6, 12, 18)
RESOLUTIONS = ("0p25", "0p1")
IFS_ONLY_RESOLUTIONS = frozenset({"0p1"})
SPLIT_RESOLUTIONS = frozenset({"0p1"})

SYSTEMS = ("ifs", "ifs-ens", "aifs-single", "aifs-ens")
SYSTEM_LABELS = {
    "ifs": "IFS deterministic",
    "ifs-ens": "IFS ENS",
    "aifs-single": "AIFS Single",
    "aifs-ens": "AIFS ENS",
}

# URL ``type`` values. IFS ENS is one ``ef`` file (control and perturbed).
# AIFS ENS keeps control (``cf``) and perturbed (``pf``) as separate files.
FILE_TYPES = {
    "ifs": ("fc",),
    "ifs-ens": ("ef",),
    "aifs-single": ("fc",),
    "aifs-ens": ("cf", "pf"),
}

# Index ``type`` values inside those files.
INDEX_TYPES = {
    "ifs": ("fc",),
    "ifs-ens": ("cf", "pf"),
    "aifs-single": ("fc",),
    "aifs-ens": ("cf", "pf"),
}

_ALIASES = {
    "ifs": "ifs",
    "ifs-deterministic": "ifs",
    "ifs_deterministic": "ifs",
    "hres": "ifs",
    "ifs-ens": "ifs-ens",
    "ifs_ens": "ifs-ens",
    "ens": "ifs-ens",
    "aifs-single": "aifs-single",
    "aifs_single": "aifs-single",
    "aifs-ens": "aifs-ens",
    "aifs_ens": "aifs-ens",
}

# Originating centre in an essential.ecmwf.int WMO heading.
# ECMF is the deterministic IFS product, ECEP the ensemble IFS product.
DISS_CENTRES = {
    "ECMF": "ifs",
    "ECEP": "ifs-ens",
}

# (short name, long name, param id, level type)
_IFS_ATMOSPHERE: Tuple[Tuple[str, str, str, str], ...] = (
    ("z", "Geopotential", "129", "pl"),
    ("t", "Temperature", "130", "pl"),
    ("u", "U wind component", "131", "pl"),
    ("v", "V wind component", "132", "pl"),
    ("q", "Specific humidity", "133", "pl"),
    ("w", "Vertical velocity", "135", "pl"),
    ("vo", "Relative vorticity", "138", "pl"),
    ("d", "Divergence", "155", "pl"),
    ("gh", "Geopotential height", "156", "pl"),
    ("r", "Relative humidity", "157", "pl"),
    ("vsw", "Volumetric soil water (layers 1–4)", "39-42", "sol"),
    ("sot", "Soil temperature (layers 1–4)", "139,170,183,236", "sol"),
    ("msl", "Mean sea-level pressure", "151", "sfc"),
    ("sp", "Surface pressure", "134", "sfc"),
    ("2t", "2 m temperature", "167", "sfc"),
    ("2d", "2 m dewpoint temperature", "168", "sfc"),
    ("skt", "Skin temperature", "235", "sfc"),
    ("10u", "10 m U wind component", "165", "sfc"),
    ("10v", "10 m V wind component", "166", "sfc"),
    ("100u", "100 m U wind component", "228246", "sfc"),
    ("100v", "100 m V wind component", "228247", "sfc"),
    ("10fg", "Maximum 10 m wind gust", "49", "sfc"),
    ("mx2t3", "Maximum 2 m temperature (3 h)", "228026", "sfc"),
    ("mn2t3", "Minimum 2 m temperature (3 h)", "228027", "sfc"),
    ("mx2t6", "Maximum 2 m temperature (6 h)", "121", "sfc"),
    ("mn2t6", "Minimum 2 m temperature (6 h)", "122", "sfc"),
    ("tcc", "Total cloud cover", "164", "sfc"),
    ("hcc", "High cloud cover", "188", "sfc"),
    ("mcc", "Medium cloud cover", "187", "sfc"),
    ("lcc", "Low cloud cover", "186", "sfc"),
    ("tcw", "Total column water", "136", "sfc"),
    ("tcwv", "Total column water vapour", "137", "sfc"),
    ("cp", "Convective precipitation", "143", "sfc"),
    ("tp", "Total precipitation", "228", "sfc"),
    ("tprate", "Total precipitation rate", "260048", "sfc"),
    ("tciw", "Total column cloud ice water", "79", "sfc"),
    ("tclw", "Total column cloud liquid water", "78", "sfc"),
    ("tcrw", "Total column rain water", "228089", "sfc"),
    ("sf", "Snowfall (water equivalent)", "144", "sfc"),
    ("sd", "Snow depth (water equivalent)", "141", "sfc"),
    ("rsn", "Snow density", "33", "sfc"),
    ("asn", "Snow albedo", "32", "sfc"),
    ("mucape", "Most-unstable CAPE", "228235", "sfc"),
    ("ptype", "Precipitation type", "260015", "sfc"),
    ("ro", "Runoff", "205", "sfc"),
    ("ssr", "Surface net shortwave radiation", "176", "sfc"),
    ("ssrd", "Surface downward shortwave radiation", "169", "sfc"),
    ("str", "Surface net longwave radiation", "177", "sfc"),
    ("strd", "Surface downward longwave radiation", "175", "sfc"),
    ("ttr", "Top net longwave radiation", "179", "sfc"),
    ("ewss", "Eastward turbulent surface stress", "180", "sfc"),
    ("nsss", "Northward turbulent surface stress", "181", "sfc"),
    ("zos", "Sea surface height", "262124", "sfc"),
    ("sithick", "Sea ice thickness", "262000", "sfc"),
    ("sve", "Eastward sea-water velocity", "262140", "sfc"),
    ("svn", "Northward sea-water velocity", "262139", "sfc"),
    ("lsm", "Land-sea mask", "172", "sfc"),
    ("sdor", "Standard deviation of sub-grid orography", "160", "sfc"),
    ("slor", "Slope of sub-grid orography", "163", "sfc"),
)

_AIFS_ATMOSPHERE: Tuple[Tuple[str, str, str, str], ...] = (
    ("z", "Geopotential", "129", "sfc"),
    ("t", "Temperature", "130", "pl"),
    ("u", "U component of the wind", "131", "pl"),
    ("v", "V component of the wind", "132", "pl"),
    ("q", "Specific humidity", "133", "pl"),
    ("sp", "Surface pressure", "134", "sfc"),
    ("w", "Vertical velocity", "135", "pl"),
    ("tcw", "Total column water", "136", "sfc"),
    ("msl", "Mean sea level pressure", "151", "sfc"),
    ("sdor", "Standard deviation of sub-gridscale orography", "160", "sfc"),
    ("slor", "Slope of sub-gridscale orography", "163", "sfc"),
    ("10u", "10 metre U wind component", "165", "sfc"),
    ("10v", "10 metre V wind component", "166", "sfc"),
    ("2t", "2 metre temperature", "167", "sfc"),
    ("2d", "2 metre dewpoint temperature", "168", "sfc"),
    ("ssrd", "Surface short-wave radiation downwards", "169", "sfc"),
    ("lsm", "Land-sea mask", "172", "sfc"),
    ("strd", "Surface long-wave radiation downwards", "175", "sfc"),
    ("skt", "Skin temperature", "235", "sfc"),
    ("lcc", "Low cloud cover", "3073", "sfc"),
    ("mcc", "Medium cloud cover", "3074", "sfc"),
    ("hcc", "High cloud cover", "3075", "sfc"),
    ("cp", "Convective precipitation", "228143", "sfc"),
    ("sf", "Snowfall water equivalent", "228144", "sfc"),
    ("tcc", "Total cloud cover", "228164", "sfc"),
    ("tp", "Total precipitation", "228228", "sfc"),
    ("100u", "100 metre U wind component", "228246", "sfc"),
    ("100v", "100 metre V wind component", "228247", "sfc"),
    ("rowe", "Runoff water equivalent (surface plus subsurface)", "231002", "sfc"),
    ("vsw", "Volumetric soil moisture (layers 1-2)", "260199", "sol"),
    ("fscov", "Fraction of snow cover", "260289", "sfc"),
    ("sot", "Soil temperature (layers 1-2)", "260360", "sol"),
)

# Documented atmospheric parameters. Omitting ``param`` on a download reads
# every message in the file, including 0.1° fields beyond this list.
VARIABLES = {
    "ifs": _IFS_ATMOSPHERE,
    "ifs-ens": _IFS_ATMOSPHERE,
    "aifs-single": _AIFS_ATMOSPHERE,
    "aifs-ens": _AIFS_ATMOSPHERE,
}

# 10fg is the maximum 10 m gust on steps through 90 h. Later IFS steps are
# indexed as 10fg3, the same quantity over a 3-hour window.
GUST_SHORT_NAMES = ("10fg", "10fg3")
WIND_COMPONENT_NAMES = (("10u", "u10"), ("10v", "v10"))
SURFACE_GEOPOTENTIAL = ("z", "Geopotential (orography, step 0)", "129", "sfc")


def canonical_system(name: str) -> str:
    """Return the catalogue key for a model name."""
    key = _ALIASES.get(str(name).strip().lower())
    if key is None:
        known = ", ".join(SYSTEMS)
        raise ValueError(f"Unknown ECMWF system {name!r}. Known systems: {known}.")
    return key


def expand_systems(name: str) -> Tuple[str, ...]:
    """Return one system, or all four when ``name`` is ``all``."""
    if str(name).strip().lower() == "all":
        return SYSTEMS
    return (canonical_system(name),)


def canonical_resol(value) -> str:
    """Return ``0p25`` or ``0p1``."""
    text = str(value).strip().lower().replace("°", "").replace(" ", "")
    if text in {"0p25", "0.25", "0.25deg", "025"}:
        return "0p25"
    if text in {"0p1", "0.1", "0.10", "0.10deg"}:
        return "0p1"
    raise ValueError(f"Resolution must be 0.25 or 0.1 degrees, got {value!r}.")


def check_resol(system: str, resol: str) -> str:
    """Return a resolution the system publishes."""
    system = canonical_system(system)
    resol = canonical_resol(resol)
    if resol in IFS_ONLY_RESOLUTIONS and system not in {"ifs", "ifs-ens"}:
        label = SYSTEM_LABELS[system]
        raise ValueError(
            f"{label} open data is published at 0.25 degrees. "
            f"0.1 degrees is published for the IFS model."
        )
    return resol


def parse_date(value) -> date:
    """Read a forecast date from a date, datetime, or ``YYYY-MM-DD`` / ``YYYYMMDD`` string."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1]
    for candidate, fmt in ((text[:10], "%Y-%m-%d"), (text[:8], "%Y%m%d")):
        try:
            return datetime.strptime(candidate, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot read an ECMWF run date from {value!r}.")


def parse_time(value) -> int:
    """Return a cycle hour: 0, 6, 12, or 18."""
    if isinstance(value, time):
        hour = value.hour
    else:
        text = str(value).strip().lower().rstrip("z")
        if not text:
            raise ValueError("ECMWF cycle hour is missing.")
        hour = int(text)
        if hour >= 100:
            hour //= 100
    if hour not in RUN_HOURS:
        raise ValueError(f"ECMWF cycle hour must be one of {RUN_HOURS}, got {value!r}.")
    return hour


def _hours(start: int, stop: int, step: int) -> Tuple[int, ...]:
    return tuple(range(start, stop + 1, step))


def forecast_steps(system: str, hour: int) -> Tuple[int, ...]:
    """Return every published forecast step, in hours, for one cycle."""
    system = canonical_system(system)
    hour = parse_time(hour)
    if system in {"aifs-single", "aifs-ens"}:
        return _hours(0, 360, 6)
    if hour in (0, 12):
        return _hours(0, 144, 3) + _hours(150, 360, 6)
    if system == "ifs":
        return _hours(0, 90, 3)
    return _hours(0, 144, 3)


def track_step_hours(system: str, hour: int) -> int:
    """Return the step written into a tropical-cyclone track file name."""
    system = canonical_system(system)
    hour = parse_time(hour)
    if system in {"aifs-single", "aifs-ens"}:
        return 360
    if hour in (0, 12):
        return 360
    return 144


def check_step(system: str, hour: int, step: int) -> int:
    """Return ``step`` when that system publishes it at ``hour``."""
    step = int(step)
    published = forecast_steps(system, hour)
    if step not in published:
        label = SYSTEM_LABELS[canonical_system(system)]
        raise ValueError(
            f"Step {step} h is outside the {label} {hour:02d} UTC range "
            f"({published[0]}–{published[-1]} h)."
        )
    return step


def gust_param(step: int) -> str:
    """
    Return the open-data short name of maximum 10 m gust at one step.

    Through 90 h the index name is ``10fg``. Later steps are ``10fg3``.
    """
    return "10fg" if int(step) <= 90 else "10fg3"


def stream_for(system: str, when: date, hour: int) -> str:
    """Return the open-data stream directory for one run."""
    system = canonical_system(system)
    hour = parse_time(hour)
    if system == "ifs" and hour in (6, 18) and when < IFS_50R1:
        return "scda"
    if system in {"ifs-ens", "aifs-ens"}:
        return "enfo"
    return "oper"


def variable_names(system: str) -> Tuple[str, ...]:
    """Return the documented atmospheric short names for one system."""
    system = canonical_system(system)
    names = tuple(row[0] for row in VARIABLES[system])
    if system == "ifs":
        # Surface geopotential (orography) shares the short name ``z`` with
        # the pressure-level field and is requested as levtype=sfc.
        return names
    return names


def level_suffixes(resol: str, levtype: Optional[str]) -> Tuple[str, ...]:
    """
    Return the file-name suffixes for a resolution.

    ``0p25`` is one file and has an empty suffix. ``0p1`` splits surface
    (including soil) and pressure levels.
    """
    resol = canonical_resol(resol)
    if resol not in SPLIT_RESOLUTIONS:
        return ("",)
    if levtype in (None, "", "all"):
        return ("-sfc", "-pl")
    key = str(levtype).lower()
    if key in {"sfc", "sol", "surface"}:
        return ("-sfc",)
    if key in {"pl", "pressure"}:
        return ("-pl",)
    raise ValueError("levtype must be 'sfc', 'pl', or omitted.")


def field_urls(
    when: date,
    hour: int,
    system: str,
    resol: str,
    step: int,
    *,
    root: str = OPEN_DATA_ROOT,
    levtype: Optional[str] = None,
) -> Tuple[str, ...]:
    """Return the GRIB2 URLs for one forecast step."""
    system = canonical_system(system)
    resol = check_resol(system, resol)
    hour = parse_time(hour)
    step = check_step(system, hour, step)
    stream = stream_for(system, when, hour)
    model = "ifs" if system in {"ifs", "ifs-ens"} else system
    stamp = f"{when:%Y%m%d}{hour:02d}0000"
    folder = f"{root}/{when:%Y%m%d}/{hour:02d}z/{model}/{resol}/{stream}"
    urls = []
    for file_type in FILE_TYPES[system]:
        for suffix in level_suffixes(resol, levtype):
            name = f"{stamp}-{step}h-{stream}-{file_type}{suffix}.grib2"
            urls.append(f"{folder}/{name}")
    return tuple(urls)


def track_url(
    when: date,
    hour: int,
    system: str,
    *,
    root: str = OPEN_DATA_ROOT,
) -> str:
    """Return the open-data BUFR URL of one system's tropical-cyclone tracks."""
    system = canonical_system(system)
    hour = parse_time(hour)
    step = track_step_hours(system, hour)
    stream = stream_for(system, when, hour)
    model = "ifs" if system in {"ifs", "ifs-ens"} else system
    stamp = f"{when:%Y%m%d}{hour:02d}0000"
    name = f"{stamp}-{step}h-{stream}-tf.bufr"
    return f"{root}/{when:%Y%m%d}/{hour:02d}z/{model}/0p25/{stream}/{name}"


def index_url(grib_url: str) -> str:
    """Return the JSON-lines index that belongs to a GRIB2 URL."""
    if grib_url.endswith(".grib2"):
        return grib_url[: -len("grib2")] + "index"
    return grib_url + ".index"


def parse_index(text: str) -> Tuple[Mapping, ...]:
    """Read an ECMWF ``.index`` file. Each line is one GRIB message."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return tuple(rows)


def _as_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(str(value).split("-")[0])
    except (TypeError, ValueError):
        return None


def select_messages(
    rows: Sequence[Mapping],
    *,
    param: Optional[Iterable[str]] = None,
    step: Optional[int] = None,
    levtype: Optional[str] = None,
    number: Optional[Iterable] = None,
) -> Tuple[Mapping, ...]:
    """
    Keep index rows for the requested fields.

    ``param`` omitted keeps every parameter. ``levtype='sfc'`` also keeps
    soil levels, which ECMWF stores in the surface file.
    """
    wanted = None if param is None else {str(item) for item in param}
    levels = None
    if levtype not in (None, "", "all"):
        key = str(levtype).lower()
        levels = {"sfc", "sol"} if key in {"sfc", "sol", "surface"} else {key}
    members = None if number is None else {str(item) for item in number}
    kept = []
    for row in rows:
        if step is not None and _as_int(row.get("step")) != int(step):
            continue
        if levels is not None and str(row.get("levtype", "")) not in levels:
            continue
        if wanted is not None and str(row.get("param")) not in wanted:
            continue
        if members is not None and str(row.get("number")) not in members:
            continue
        kept.append(row)
    return tuple(kept)


def merge_ranges(rows: Sequence[Mapping]) -> Tuple[Tuple[int, int], ...]:
    """Merge ``_offset`` / ``_length`` pairs into the fewest byte ranges."""
    spans = sorted(
        (int(row["_offset"]), int(row["_length"]))
        for row in rows
        if row.get("_offset") is not None and row.get("_length") is not None
    )
    if not spans:
        return ()
    merged = [[spans[0][0], spans[0][1]]]
    for offset, length in spans[1:]:
        end = merged[-1][0] + merged[-1][1]
        if offset <= end:
            merged[-1][1] = max(end, offset + length) - merged[-1][0]
        else:
            merged.append([offset, length])
    return tuple((offset, length) for offset, length in merged)

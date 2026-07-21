from enum import Enum
import uuid
import numpy as np

from typing import List, Dict, Union

from gigaspatial.config import config

EnumValue = Union[str, Enum]


def enum_value(value: EnumValue) -> str:
    """Return a canonical string value from an enum or string."""
    return value.value if isinstance(value, Enum) else value


class DataConfidence(str, Enum):
    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    ESTIMATED = "estimated"


class PowerSource(str, Enum):
    GRID = "grid"
    GENERATOR = "generator"
    SOLAR = "solar"


class InfrastructureStatus(str, Enum):
    """Enum for infrastructure operational status."""

    PROPOSED = "proposed"
    PLANNED = "planned"
    UNDER_CONSTRUCTION = "underconstruction"
    OPERATIONAL = "operational"
    DECOMMISSIONED = "decommissioned"
    INACTIVE = "inactive"


class RadioType(str, Enum):
    """Enum for different radio technology types."""

    TWO_G = "2G"
    THREE_G = "3G"
    FOUR_G = "4G"
    FIVE_G = "5G"


class WirelessAccessServiceType(str, Enum):
    """Access-service roles supported by a wireless site."""

    MOBILE = "mobile"
    FIXED_WIRELESS = "fixed_wireless"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class WirelessAccessTechnology(str, Enum):
    """Technologies used to provide wireless access services."""

    GSM = "2g"
    UMTS = "3g"
    LTE = "4g"
    NR = "5g"
    WIFI = "wifi"
    WIMAX = "wimax"
    PROPRIETARY = "proprietary"
    OTHER = "other"


class SpectrumType(str, Enum):
    """Enum for spectrum access and licensing categories."""

    LICENSED = "licensed"
    SHARED = "shared"
    UNLICENSED = "unlicensed"


RADIO_ALIASES: Dict[RadioType, List[str]] = {
    RadioType.FIVE_G: [
        "5g",
        "nr",
        "new radio",
        "5g nr",
        "5g sa",
        "5g nsa",
        "nr5g",
    ],
    RadioType.FOUR_G: [
        "4g",
        "lte",
        "lte-a",
        "lte advanced",
        "4g lte",
        "e-utra",
        "fdd-lte",
        "tdd-lte",
    ],
    RadioType.THREE_G: [
        "3g",
        "cdma",
        "umts",
        "wcdma",
        "hspa",
        "hspa+",
        "hsdpa",
        "hsupa",
        "evdo",
        "cdma2000",
    ],
    RadioType.TWO_G: [
        "2g",
        "gsm",
        "gprs",
        "edge",
        "cdmaone",
    ],
}

# Reverse lookup: alias string → RadioType value, built once at module level
RADIO_ALIAS_MAP = {
    alias: radio_type.value
    for radio_type, aliases in RADIO_ALIASES.items()
    for alias in aliases
}

WIRELESS_ACCESS_TECHNOLOGY_ALIASES: Dict[
    WirelessAccessTechnology,
    List[str],
] = {
    WirelessAccessTechnology.GSM: [
        "gsm",
        "gprs",
        "edge",
        "2g",
        "cdmaone",
    ],
    WirelessAccessTechnology.UMTS: [
        "umts",
        "wcdma",
        "hspa",
        "hspa+",
        "hsdpa",
        "hsupa",
        "3g",
        "cdma2000",
        "evdo",
    ],
    WirelessAccessTechnology.LTE: [
        "lte",
        "lte-a",
        "lte advanced",
        "4g",
        "4g lte",
        "e-utra",
        "fdd-lte",
        "tdd-lte",
    ],
    WirelessAccessTechnology.NR: [
        "nr",
        "new radio",
        "5g",
        "5g nr",
        "5g sa",
        "5g nsa",
        "nr5g",
    ],
    WirelessAccessTechnology.WIFI: [
        "wifi",
        "wi-fi",
        "wifi 5",
        "wi-fi 5",
        "wifi 6",
        "wi-fi 6",
        "wifi 6e",
        "wi-fi 6e",
        "wifi 7",
        "wi-fi 7",
        "802.11",
        "802.11ac",
        "802.11ax",
        "802.11be",
    ],
    WirelessAccessTechnology.WIMAX: [
        "wimax",
        "wi-max",
        "802.16",
        "802.16e",
        "802.16m",
    ],
    WirelessAccessTechnology.PROPRIETARY: [
        "proprietary",
        "vendor proprietary",
        "private wireless",
        "fixed wireless proprietary",
    ],
}

WIRELESS_ACCESS_TECHNOLOGY_ALIAS_MAP: Dict[str, str] = {
    alias: technology.value
    for technology, aliases in WIRELESS_ACCESS_TECHNOLOGY_ALIASES.items()
    for alias in aliases
}

ACCESS_SERVICE_TYPE_ALIAS_MAP: Dict[str, str] = {
    "mobile": WirelessAccessServiceType.MOBILE.value,
    "cellular": WirelessAccessServiceType.MOBILE.value,
    "mobile broadband": WirelessAccessServiceType.MOBILE.value,
    "fixed wireless": WirelessAccessServiceType.FIXED_WIRELESS.value,
    "fixed_wireless": WirelessAccessServiceType.FIXED_WIRELESS.value,
    "fwa": WirelessAccessServiceType.FIXED_WIRELESS.value,
    "wireless broadband": WirelessAccessServiceType.FIXED_WIRELESS.value,
    "mixed": WirelessAccessServiceType.MIXED.value,
    "unknown": WirelessAccessServiceType.UNKNOWN.value,
}

WIRELESS_ACCESS_TECHNOLOGY_TO_RADIO_TYPE: Dict[
    WirelessAccessTechnology,
    RadioType,
] = {
    WirelessAccessTechnology.GSM: RadioType.TWO_G,
    WirelessAccessTechnology.UMTS: RadioType.THREE_G,
    WirelessAccessTechnology.LTE: RadioType.FOUR_G,
    WirelessAccessTechnology.NR: RadioType.FIVE_G,
}


NULL_LIKE_VALUES = [
    # numeric sentinels
    np.nan,
    # float("inf"),
    # float("-inf"),
    # empty / whitespace strings
    "",
    " ",
    # explicit null strings (case variants covered by str normalization)
    "null",
    "NULL",
    "Null",
    "none",
    "None",
    "NONE",
    # not available
    "n/a",
    "N/A",
    "n/A",
    "NA",
    "na",
    # not applicable / not defined
    "nd",
    "ND",
    "n.d.",
    "N.D.",
    # unknown
    "unknown",
    "Unknown",
    "UNKNOWN",
    # missing
    "missing",
    "Missing",
    "MISSING",
    # placeholder / filler
    "-",
    "--",
    "---",
    "?",
    "??",
    # zeros used as sentinels (careful — only for non-numeric fields)
    # "0",
    # excel / spreadsheet artifacts
    "#N/A",
    "#NA",
    "#VALUE!",
    "#NULL!",
    "#REF!",
]

# Fixed namespace UUID for all GigaSpatial entity ID generation
# Ensures IDs are stable, reproducible, and scoped to GigaSpatial
ENTITY_UUID_NAMESPACE = uuid.UUID(config.ENTITY_ID_NAMESPACE)

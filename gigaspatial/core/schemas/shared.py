from enum import Enum
import uuid
import numpy as np
from gigaspatial.config import config


class DataConfidence(str, Enum):
    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    ESTIMATED = "estimated"


class PowerSource(str, Enum):
    GRID = "grid"
    GENERATOR = "generator"
    SOLAR = "solar"


class RadioType(str, Enum):
    """Enum for different radio technology types."""

    TWO_G = "2G"
    THREE_G = "3G"
    FOUR_G = "4G"
    FIVE_G = "5G"


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

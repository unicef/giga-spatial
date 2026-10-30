"""Node and edge kinds, stored as categoricals with the library's kinds first."""

from typing import Any, Tuple

import numpy as np
import pandas as pd

NODE_KINDS = ("road", "point", "split")
EDGE_KINDS = ("road", "split", "connection", "point_link", "component_link", "added", "bridge")


def as_kinds(values: Any, default: str, known: Tuple[str, ...], length: int) -> pd.Categorical:
    """Kinds as a categorical whose categories start with ``known``; missing values become ``default``."""
    if values is None:
        codes = np.full(length, known.index(default), dtype=np.int8)
        return pd.Categorical.from_codes(codes, categories=list(known))
    if np.ndim(values) == 0:
        values = np.full(length, values, dtype=object)
    if isinstance(getattr(values, "dtype", None), pd.CategoricalDtype) and not pd.isna(values).any():
        categorical = pd.Categorical(values)
    else:
        categorical = pd.Categorical(pd.Series(values, dtype=object).fillna(default))
    extra = [c for c in categorical.categories if c not in known]
    return categorical.set_categories(list(known) + extra)

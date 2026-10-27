"""Normalisation of node and edge attributes into types Parquet can store.

osmnx simplification merges OSM ways, so an attribute such as ``osmid``,
``highway``, or ``name`` holds a list on some edges and a single value on
others. Parquet cannot store a column that mixes the two.
"""

from itertools import chain
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa

_LIST_LIKE = (list, tuple, np.ndarray)
_SCALARS = (str, bool, int, float, np.generic)
_BOOLEAN_RULES = {"oneway": all, "reversed": any}


def normalize_attributes(
    table: pd.DataFrame, *, int_lists: Iterable[str] = (), skip: Iterable[str] = ("geometry",)
) -> pd.DataFrame:
    """A table whose attribute columns all have one storable type.

    Numeric, boolean, and categorical columns are kept as they are. For object
    and text columns:

    - columns in ``int_lists`` become a list of integers on every row;
    - ``oneway`` lists are true only if every value is true, and ``reversed``
      lists are true if any value is;
    - other columns holding lists become a list of integers when every value
      is an integer, and ``;``-joined text otherwise, without repeats;
    - text columns, and columns mixing text with numbers, become categorical
      text.

    Missing values stay missing. Unchanged columns are shared with ``table``.

    Args:
        table: Node or edge table.
        int_lists: Columns to store as lists of integers, such as ``osmid``.
        skip: Columns to leave alone.

    Returns:
        A new table with the same index and column order.
    """
    int_lists = set(int_lists)
    skip = set(skip)
    columns = {}
    for name in table.columns:
        column = table[name]
        textual = column.dtype == object or (
            pd.api.types.is_string_dtype(column.dtype) and not isinstance(column.dtype, pd.CategoricalDtype)
        )
        if name in skip or (not textual and name not in int_lists):
            columns[name] = column
        else:
            columns[name] = _normalize_column(name, column, force_int_list=name in int_lists)
    return pd.DataFrame(columns, index=table.index, copy=False)


def _is_missing(value: Any) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and value != value)


def _normalize_column(name: str, column: pd.Series, force_int_list: bool) -> Any:
    values = column.to_numpy(dtype=object)
    inferred = pd.api.types.infer_dtype(values, skipna=True)
    if not force_int_list:
        if inferred == "string":
            return pd.Categorical(values)
        if inferred in ("empty", "boolean", "integer", "floating", "decimal", "bytes"):
            return column
        if inferred == "mixed-integer-float":
            return pd.to_numeric(column)

    is_list = np.fromiter((isinstance(v, _LIST_LIKE) for v in values), dtype=bool, count=len(values))
    if force_int_list:
        integers = _integer_lists(values, is_list)
        if integers is None:
            raise ValueError(f"Column {name!r} must hold integers to be stored as lists of integers")
        return integers
    if not is_list.any():
        if all(isinstance(v, _SCALARS) or _is_missing(v) for v in values):
            return pd.Categorical([None if _is_missing(v) else str(v) for v in values])
        return column

    listed = np.flatnonzero(is_list)
    single = np.flatnonzero(~is_list)
    if name in _BOOLEAN_RULES:
        rule = _BOOLEAN_RULES[name]
        out = values.copy()
        out[listed] = [bool(rule(v)) for v in values[listed]]
        missing = pd.isna(out)
        if not missing.any():
            return out.astype(bool)
        return pd.array([None if gone else bool(v) for v, gone in zip(out, missing)], dtype="boolean")
    if not _has_text(values, listed, single):
        integers = _integer_lists(values, is_list)
        if integers is not None:
            return integers
    out = values.copy()
    out[listed] = [";".join(dict.fromkeys(str(x) for x in v)) for v in values[listed]]
    if pd.api.types.infer_dtype(values[single], skipna=True) not in ("string", "empty"):
        out[single] = [None if _is_missing(v) else str(v) for v in values[single]]
    return pd.Categorical(out)


def _has_text(values: np.ndarray, listed: np.ndarray, single: np.ndarray) -> bool:
    """Whether the first list or the first single value holds text, which rules out integer lists cheaply."""
    for v in values[listed]:
        if len(v):
            if isinstance(v[0], str):
                return True
            break
    for v in values[single]:
        if not _is_missing(v):
            return isinstance(v, str)
    return False


def _integer_lists(values: np.ndarray, is_list: np.ndarray) -> Any:
    """An Arrow list<int64> column, single values wrapped; None if any value is not an integer."""
    missing = np.fromiter(
        (not listed and _is_missing(v) for v, listed in zip(values, is_list)), dtype=bool, count=len(values)
    )
    lengths = np.fromiter(
        (len(v) if listed else (0 if gone else 1) for v, listed, gone in zip(values, is_list, missing)),
        dtype=np.int64,
        count=len(values),
    )
    flat = np.array(
        list(chain.from_iterable(v if listed else (v,) for v, listed, gone in zip(values, is_list, missing) if not gone))
    )
    if len(flat) == 0:
        flat = np.empty(0, dtype=np.int64)
    if flat.dtype.kind not in "iu":
        return None
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32 if lengths.sum() < 2**31 else np.int64)
    array = pa.ListArray.from_arrays(
        pa.array(offsets), pa.array(flat.astype(np.int64)), mask=pa.array(missing) if missing.any() else None
    )
    return pd.arrays.ArrowExtensionArray(array)

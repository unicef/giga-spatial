"""Array helpers shared by the network modules."""

import numpy as np
import pandas as pd


def next_keys(
    old_a: np.ndarray, old_b: np.ndarray, old_key: np.ndarray, new_a: np.ndarray, new_b: np.ndarray
) -> np.ndarray:
    """Keys for new edges between node positions (a, b), continuing after the existing keys of each pair."""
    existing = pd.DataFrame({"a": old_a, "b": old_b, "key": old_key}).groupby(["a", "b"])["key"].max().rename("base")
    pairs = pd.DataFrame({"a": new_a, "b": new_b})
    base = pairs.join(existing, on=["a", "b"])["base"].fillna(-1).to_numpy() + 1
    return (base + pairs.groupby(["a", "b"]).cumcount().to_numpy()).astype(np.int64)


def as_positions(positions: np.ndarray, n_nodes: int) -> np.ndarray:
    return positions.astype(np.int32) if n_nodes < 2**31 else positions


def repeated_rows(a: np.ndarray, b: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Rows whose (a, b, k) appeared in an earlier row; sorting needs far less memory than hashing."""
    repeated = np.zeros(len(a), dtype=bool)
    if len(a) < 2:
        return repeated
    span = int(max(a.max(), b.max())) + 1
    if span < 2**24:
        combined = (a.astype(np.int64) * span + b) * 2**15 + k
        order = np.argsort(combined, kind="stable")
        sorted_combined = combined[order]
        repeated[order[1:][sorted_combined[1:] == sorted_combined[:-1]]] = True
        return repeated
    order = np.lexsort((k, b, a))
    same = np.ones(len(a) - 1, dtype=bool)
    for column in (a, b, k):
        sorted_column = column[order]
        same &= sorted_column[1:] == sorted_column[:-1]
    repeated[order[1:][same]] = True
    return repeated

"""
Confirmed TifProcessor defects.

Each test asserts the target behaviour from
.scratch/tif-processor-hardening/spec.md and is marked ``xfail(strict=True)``
with its defect ID and the ticket that fixes it. When a fix lands, its tests
start passing; strict mode then fails the run until the marker is removed.

Batched-sampling tests run under a deadline: while a defect stands, a pool
worker's setup can raise, and the pool then respawns workers forever.
"""

import copy
import gc
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from _tif_fakes import BAND, BANDS_FOR_MODE, DEFAULT_NODATA, EAST_BOUNDS, TWO_BANDS
from gigaspatial.processing.tif_processor import TifProcessor


def breaks(defect: str, reason: str):
    return pytest.mark.xfail(strict=True, reason=f"[{defect}] {reason}")


M1_3 = breaks("M1-3 → 03", "to_dataframe_chunked calls a helper that does not exist")
M1_4 = breaks(
    "M1-4 → 04", "the memory guard raises UnboundLocalError instead of warning"
)
M1_5 = breaks("M1-5 → 05", "a sparse pixel graph without edges raises IndexError")
M1_6 = breaks("M1-6 → 06", "a mixed-CRS merge without target_crs only logs a warning")
M1_7 = breaks("M1-7 → 07", "a mean merge casts the mean back to the source dtype")
M1_8 = breaks("M1-8 → 07", "a band_names list is indexed by the 1-based band number")
LATER_POINT_NODATA = breaks("later", "point samples return the raw nodata value")
LATER_NAN_NODATA = breaks("later", "NaN nodata is never masked")
LATER_POOLED_BANDS = breaks("later", "polygon statistics pool all bands into one value")
LATER_CLIPPED_WORKERS = breaks(
    "later", "batched worker setup ignores a clipped file on a non-local data store"
)

SOUTH_WEST = box(0.001, 0.001, 0.019, 0.019)
POLYGONS = [
    box(0.0, 0.02, 0.02, 0.04),
    box(0.02, 0.0, 0.04, 0.02),
    box(0.0, 0.0, 0.08, 0.04),
]
DEADLINE = 10


def merged(rasters):
    west = rasters.put("west.tif", BAND)
    east = rasters.put("east.tif", BAND + 16, bounds=EAST_BOUNDS)
    return TifProcessor([west, east], data_store=rasters.store)


def reprojected(rasters):
    path = rasters.put("a.tif")
    return TifProcessor(path, data_store=rasters.store, target_crs="EPSG:3857")


def clipped(rasters):
    tp = TifProcessor(rasters.put("a.tif"), data_store=rasters.store)
    return tp.clip_to_geometry(SOUTH_WEST)


DERIVED = [merged, reprojected, clipped]


def quarters(tp):
    """The four quadrants of a processor's bounds, in its own CRS."""
    left, bottom, right, top = tp.bounds
    mid_x, mid_y = (left + right) / 2, (bottom + top) / 2
    return [
        box(left, mid_y, mid_x, top),
        box(mid_x, mid_y, right, top),
        box(left, bottom, mid_x, mid_y),
        box(mid_x, bottom, right, mid_y),
    ]


def working_file_exists(tp):
    return Path(tp._temp_dir).is_dir() and Path(tp.dataset_path).is_file()


def batched(tp, polygons=POLYGONS):
    return tp.sample_by_polygons_batched(
        polygons, n_workers=2, batch_size=1, show_progress=False
    )


def pickled(tp):
    return pickle.loads(pickle.dumps(tp))


# ─────────────────────────────────────────────────────────────────────────────
# M1-1: data loss and ownership
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("derive", DERIVED)
def test_batched_sampling_leaves_a_derived_processor_intact(
    local_rasters, start_method, deadline, derive
):
    tp = derive(local_rasters)
    polygons = quarters(tp)
    expected = tp.sample_by_polygons(polygons)

    with deadline(DEADLINE):
        first = batched(tp, polygons)
    assert working_file_exists(tp)
    with deadline(DEADLINE):
        second = batched(tp, polygons)

    np.testing.assert_allclose(first, expected)
    np.testing.assert_allclose(second, expected)


@pytest.mark.parametrize("make_copy", [copy.copy, copy.deepcopy, pickled])
def test_dropping_a_copy_leaves_the_original_intact(local_rasters, make_copy):
    tp = merged(local_rasters)
    duplicate = make_copy(tp)

    del duplicate
    gc.collect()

    assert working_file_exists(tp)
    assert len(tp.to_dataframe()) == 32


def test_cleanup_twice_then_collection_touches_no_other_processor(
    local_rasters, monkeypatch
):
    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    tp = merged(local_rasters)
    other = merged(local_rasters)
    duplicate_of_other = copy.deepcopy(other)

    tp.cleanup()
    tp.cleanup()
    duplicate_of_other.cleanup()
    del tp, duplicate_of_other
    gc.collect()

    assert not unraisable
    assert working_file_exists(other)
    assert len(other.to_dataframe()) == 32


# ─────────────────────────────────────────────────────────────────────────────
# M1-2: use after cleanup
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("derive", DERIVED)
def test_derived_processor_used_after_cleanup_says_it_was_cleaned_up(
    local_rasters, derive
):
    tp = derive(local_rasters)

    tp.cleanup()

    with pytest.raises(FileNotFoundError, match="cleaned up"):
        with tp.open_dataset():
            pass


@pytest.mark.parametrize("derive", DERIVED)
def test_wrapped_errors_after_cleanup_carry_the_cause(local_rasters, derive):
    tp = derive(local_rasters)

    tp.cleanup()

    with pytest.raises(ValueError, match="cleaned up"):
        tp.to_dataframe()


# ─────────────────────────────────────────────────────────────────────────────
# M1-3: chunked conversion of multi-band rasters
# ─────────────────────────────────────────────────────────────────────────────


@M1_3
@pytest.mark.parametrize("mode", ["rgb", "rgba", "multi"])
def test_chunked_multi_band_conversion_matches_to_dataframe(local_rasters, mode):
    data = BANDS_FOR_MODE[mode].copy()
    data[0, 1, 2] = DEFAULT_NODATA
    path = local_rasters.put("a.tif", data)
    tp = TifProcessor(path, data_store=local_rasters.store, mode=mode)

    chunked = tp.to_dataframe_chunked(chunk_size=1)

    pd.testing.assert_frame_equal(chunked, tp.to_dataframe(), check_exact=False)


# ─────────────────────────────────────────────────────────────────────────────
# M1-4: memory guard
# ─────────────────────────────────────────────────────────────────────────────


@M1_4
def test_batched_sampling_warns_on_low_memory_and_still_runs(
    local_rasters, low_memory, deadline
):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    with pytest.warns(ResourceWarning, match="n_workers"), deadline(DEADLINE):
        result = batched(tp, POLYGONS[:2])

    np.testing.assert_allclose(result, [3.5, 13.5])


# ─────────────────────────────────────────────────────────────────────────────
# M1-5: sparse pixel graphs without edges
# ─────────────────────────────────────────────────────────────────────────────


@M1_5
def test_sparse_graph_without_adjacent_valid_pixels_is_empty(local_rasters):
    checkerboard = np.where(
        np.indices((4, 4)).sum(axis=0) % 2 == 0, BAND, DEFAULT_NODATA
    )
    path = local_rasters.put("a.tif", checkerboard.astype(np.float32))
    tp = TifProcessor(path, data_store=local_rasters.store)

    matrix = tp.to_graph(connectivity=4, graph_type="sparse")

    assert matrix.shape == (8, 8) and matrix.nnz == 0


@M1_5
def test_sparse_graph_without_valid_pixels_is_zero_by_zero(local_rasters):
    path = local_rasters.put("a.tif", np.full((4, 4), DEFAULT_NODATA, np.float32))
    tp = TifProcessor(path, data_store=local_rasters.store)

    matrix = tp.to_graph(graph_type="sparse")

    assert matrix.shape == (0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# M1-6: mixed-CRS merges
# ─────────────────────────────────────────────────────────────────────────────


@M1_6
def test_mixed_crs_merge_without_target_crs_raises(local_rasters):
    west = local_rasters.put("west.tif", BAND)
    east = local_rasters.put("east_3857.tif", BAND, bounds=EAST_BOUNDS, crs="EPSG:3857")

    with pytest.raises(ValueError) as error:
        TifProcessor([west, east], data_store=local_rasters.store)

    message = str(error.value)
    for part in ("east_3857.tif", "EPSG:4326", "EPSG:3857", "target_crs"):
        assert part in message


# ─────────────────────────────────────────────────────────────────────────────
# M1-7: mean-merge dtype
# ─────────────────────────────────────────────────────────────────────────────


@M1_7
def test_mean_merge_of_integers_keeps_the_fraction(local_rasters):
    ones = local_rasters.put("ones.tif", np.full((4, 4), 1, np.uint8), nodata=None)
    twos = local_rasters.put("twos.tif", np.full((4, 4), 2, np.uint8), nodata=None)

    tp = TifProcessor([ones, twos], data_store=local_rasters.store, merge_method="mean")

    assert tp.dtype == "float32"
    assert tp.to_dataframe()["pixel_value"].unique().tolist() == [1.5]


# ─────────────────────────────────────────────────────────────────────────────
# M1-8: band names
# ─────────────────────────────────────────────────────────────────────────────


@M1_8
def test_single_band_name_list_names_the_value_column(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    assert list(tp.to_dataframe(band_names=["pop"]).columns) == ["lon", "lat", "pop"]


@M1_8
def test_band_name_list_too_short_for_the_band_says_so(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    with pytest.raises(ValueError, match="band_names"):
        tp.to_dataframe(band_names=[])


# ─────────────────────────────────────────────────────────────────────────────
# Later: tracked, not fixed in Milestone 1
# ─────────────────────────────────────────────────────────────────────────────


@LATER_POINT_NODATA
def test_point_sample_at_a_nodata_pixel_is_nan(local_rasters):
    data = BAND.copy()
    data[0, 0] = DEFAULT_NODATA
    tp = TifProcessor(local_rasters.put("a.tif", data), data_store=local_rasters.store)

    samples = tp.sample_by_coordinates([(0.005, 0.035), (0.015, 0.035)])

    assert np.isnan(samples[0]) and samples[1] == 2.0


@LATER_NAN_NODATA
def test_drop_nodata_drops_nan_nodata_pixels(local_rasters):
    data = BAND.copy()
    data[0, 0] = np.nan
    path = local_rasters.put("a.tif", data, nodata=float("nan"))
    tp = TifProcessor(path, data_store=local_rasters.store)

    df = tp.to_dataframe(drop_nodata=True)

    assert len(df) == 15 and not df["pixel_value"].isna().any()


@LATER_POOLED_BANDS
def test_multi_band_polygon_statistics_are_per_band(local_rasters):
    path = local_rasters.put("a.tif", TWO_BANDS)
    tp = TifProcessor(path, data_store=local_rasters.store, mode="multi")

    result = tp.sample_by_polygons(POLYGONS[:1])

    np.testing.assert_allclose(np.asarray(result[0], dtype=float), [3.5, 35.0])


@LATER_CLIPPED_WORKERS
def test_batched_sampling_of_a_clipped_processor_on_a_non_local_store(
    memory_rasters, deadline
):
    tp = TifProcessor(memory_rasters.put("a.tif"), data_store=memory_rasters.store)
    south = tp.clip_to_geometry(box(0.001, 0.001, 0.039, 0.019))
    polygons = [box(0.0, 0.0, 0.02, 0.02), box(0.02, 0.0, 0.04, 0.02)]
    expected = south.sample_by_polygons(polygons)
    np.testing.assert_allclose(expected, [11.5, 13.5])

    with deadline(DEADLINE):
        result = batched(south, polygons)

    np.testing.assert_allclose(result, expected)

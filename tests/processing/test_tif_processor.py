"""
Characterization of TifProcessor's public behaviour.

These tests pin what TifProcessor does today, so that the Milestone 1 fixes and
the Milestone 2 restructuring (.scratch/tif-processor-hardening/spec.md) cannot
change anything else unnoticed. Confirmed defects live in
test_tif_processor_known_breaks.py.

All rasters are tiny synthetic GeoTIFFs; nothing touches the network.
"""

import copy
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box

from _tif_fakes import (
    BAND,
    BANDS_FOR_MODE,
    DEFAULT_BOUNDS,
    DEFAULT_NODATA,
    EAST_BOUNDS,
    RGB,
    RGBA,
    TWO_BANDS,
)
from gigaspatial.processing.tif_processor import TifProcessor


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────


def test_opens_a_single_stored_file(local_rasters):
    path = local_rasters.put("a.tif")

    tp = TifProcessor(path, data_store=local_rasters.store)

    assert tp.to_dataframe()["pixel_value"].tolist() == BAND.ravel().tolist()


def test_missing_file_raises_file_not_found(local_rasters, tmp_path):
    with pytest.raises(FileNotFoundError):
        TifProcessor(str(tmp_path / "missing.tif"), data_store=local_rasters.store)


def test_one_item_path_list_collapses_to_a_single_file(local_rasters):
    path = local_rasters.put("a.tif")

    tp = TifProcessor([path], data_store=local_rasters.store)

    assert (tp.is_merged, tp.source_count, tp.dataset_path) == (False, 1, path)


@pytest.mark.parametrize("method", ["first", "last", "min", "max", "mean"])
def test_merges_two_adjacent_files(local_rasters, method):
    west = local_rasters.put("west.tif", BAND)
    east = local_rasters.put("east.tif", BAND + 16, bounds=EAST_BOUNDS)

    tp = TifProcessor([west, east], data_store=local_rasters.store, merge_method=method)

    assert (tp.is_merged, tp.source_count) == (True, 2)
    assert (tp.width, tp.height) == (8, 4)
    assert tp.bounds == pytest.approx((0.0, 0.0, 0.08, 0.04))
    expected = np.hstack([BAND, BAND + 16])
    np.testing.assert_array_equal(
        tp.to_dataframe()["pixel_value"].to_numpy().reshape(4, 8), expected
    )


@pytest.mark.parametrize(
    "method, expected",
    [
        ("first", BAND),
        ("last", np.full((4, 4), 5.0)),
        ("min", np.minimum(BAND, 5)),
        ("max", np.maximum(BAND, 5)),
        ("mean", (BAND + 5) / 2),
    ],
)
def test_merge_method_decides_overlapping_float_pixels(local_rasters, method, expected):
    first = local_rasters.put("first.tif", BAND)
    second = local_rasters.put("second.tif", np.full((4, 4), 5, dtype=np.float32))

    tp = TifProcessor(
        [first, second], data_store=local_rasters.store, merge_method=method
    )

    assert tp.dtype == "float32"
    np.testing.assert_allclose(
        tp.to_dataframe()["pixel_value"].to_numpy().reshape(4, 4), expected
    )


def test_merge_with_target_crs_reprojects_each_file(local_rasters):
    west = local_rasters.put("west.tif", BAND)
    east_metres = (4452.78, 0.0, 8905.56, 4452.78)
    east = local_rasters.put("east.tif", BAND, bounds=east_metres, crs="EPSG:3857")

    tp = TifProcessor(
        [west, east], data_store=local_rasters.store, target_crs="EPSG:3857"
    )

    assert tp.crs == "EPSG:3857"
    assert tp.bounds.left == pytest.approx(0.0, abs=1e-6)
    assert tp.bounds.right == pytest.approx(8905.56, abs=1.0)


def test_merge_accepts_epsg_and_equivalent_wkt(local_rasters):
    west = local_rasters.put("west.tif", BAND)
    east = local_rasters.put(
        "east.tif", BAND, bounds=EAST_BOUNDS, crs=CRS.from_epsg(4326).to_wkt()
    )

    tp = TifProcessor([west, east], data_store=local_rasters.store)

    assert (tp.crs, tp.width) == ("EPSG:4326", 8)


def test_target_crs_reprojects_at_construction(local_rasters):
    path = local_rasters.put("a.tif")

    tp = TifProcessor(path, data_store=local_rasters.store, target_crs="EPSG:3857")

    assert tp.crs == "EPSG:3857"
    assert tp.bounds.left == pytest.approx(0.0, abs=1e-6)
    assert tp.bounds.right == pytest.approx(4452.78, abs=0.01)
    assert tp.bounds.top == pytest.approx(4452.78, abs=0.1)


@pytest.mark.parametrize(
    "mode, bands",
    [
        ("single", BAND),
        ("rgb", RGB),
        ("rgba", RGBA),
        ("multi", TWO_BANDS),
        ("multi", RGB),
    ],
)
def test_mode_accepts_its_band_count(local_rasters, mode, bands):
    path = local_rasters.put("a.tif", bands)

    tp = TifProcessor(path, data_store=local_rasters.store, mode=mode)

    assert tp.count == (bands.shape[0] if bands.ndim == 3 else 1)


@pytest.mark.parametrize(
    "mode, bands, message",
    [
        ("single", TWO_BANDS, "SINGLE mode requires a 1-band"),
        ("rgb", TWO_BANDS, "RGB mode requires a 3-band"),
        ("rgba", RGB, "RGBA mode requires a 4-band"),
        ("multi", BAND, "Multi mode requires a TIF file with 2 or more bands"),
    ],
)
def test_mode_rejects_other_band_counts(local_rasters, mode, bands, message):
    path = local_rasters.put("a.tif", bands)

    with pytest.raises(ValueError, match=message):
        TifProcessor(path, data_store=local_rasters.store, mode=mode)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────────────────────────────────────


def test_metadata_of_a_single_file(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    assert tp.crs == "EPSG:4326"
    assert tp.bounds == BoundingBox(*DEFAULT_BOUNDS)
    assert tp.transform == from_bounds(*DEFAULT_BOUNDS, width=4, height=4)
    assert tp.resolution == pytest.approx((0.01, 0.01))
    assert (tp.count, tp.dtype, tp.nodata) == (1, "float32", DEFAULT_NODATA)
    assert (tp.width, tp.height) == (4, 4)
    assert (tp.is_merged, tp.source_count) == (False, 1)


INFO_KEYS = {
    "count", "width", "height", "crs", "bounds", "transform", "dtypes",
    "nodata", "mode", "is_merged", "source_count",
}  # fmt: skip


def test_raster_info_keys(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    info = tp.get_raster_info()

    assert set(info) == INFO_KEYS
    assert (info["count"], info["dtypes"], info["mode"]) == (1, "float32", "single")


def test_raster_info_with_statistics(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    info = tp.get_raster_info(include_statistics=True)

    assert set(info) == INFO_KEYS | {"statistics"}
    stats = info["statistics"]
    assert set(stats) == {"bands", "overall", "approximate"}
    assert stats["overall"]["min"] == 1.0 and stats["overall"]["max"] == 16.0
    assert stats["overall"]["mean"] == pytest.approx(8.5)
    assert stats["overall"]["count"] == 16 and stats["overall"]["sum"] == 136.0
    assert [b["band"] for b in stats["bands"]] == [1]


# ─────────────────────────────────────────────────────────────────────────────
# Conversion
# ─────────────────────────────────────────────────────────────────────────────


def test_single_band_dataframe_columns_and_pixel_centres(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    df = tp.to_dataframe()

    assert list(df.columns) == ["lon", "lat", "pixel_value"]
    assert len(df) == 16
    assert df.iloc[0].tolist() == pytest.approx([0.005, 0.035, 1.0])
    assert df.iloc[-1].tolist() == pytest.approx([0.035, 0.005, 16.0])


def test_single_band_drop_nodata(local_rasters):
    data = BAND.copy()
    data[0, 0] = DEFAULT_NODATA
    tp = TifProcessor(local_rasters.put("a.tif", data), data_store=local_rasters.store)

    assert tp.to_dataframe()["pixel_value"].min() == 2.0
    assert len(tp.to_dataframe(drop_nodata=False)) == 16
    assert tp.to_dataframe(drop_nodata=False)["pixel_value"].min() == DEFAULT_NODATA


def test_single_band_value_thresholds_are_exclusive(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    df = tp.to_dataframe(min_value=4, max_value=10)

    assert df["pixel_value"].tolist() == [5.0, 6.0, 7.0, 8.0, 9.0]


def test_single_band_string_band_name(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    assert list(tp.to_dataframe(band_names="pop").columns) == ["lon", "lat", "pop"]


def test_multi_band_names_come_from_descriptions_or_band_number(local_rasters):
    path = local_rasters.put("a.tif", TWO_BANDS, descriptions=["pop", None])
    tp = TifProcessor(path, data_store=local_rasters.store, mode="multi")

    df = tp.to_dataframe()

    assert list(df.columns) == ["lon", "lat", "pop", "band_2"]
    assert df.iloc[1][["pop", "band_2"]].tolist() == [2.0, 20.0]


def test_multi_band_explicit_band_names(local_rasters):
    path = local_rasters.put("a.tif", TWO_BANDS)
    tp = TifProcessor(path, data_store=local_rasters.store, mode="multi")

    assert list(tp.to_dataframe(band_names=["a", "b"]).columns) == [
        "lon",
        "lat",
        "a",
        "b",
    ]


def test_rgb_dataframe(local_rasters):
    tp = TifProcessor(
        local_rasters.put("a.tif", RGB), data_store=local_rasters.store, mode="rgb"
    )

    df = tp.to_dataframe()

    assert list(df.columns) == ["lon", "lat", "red", "green", "blue"]
    assert df.iloc[0][["red", "green", "blue"]].tolist() == [1.0, 101.0, 201.0]


def test_rgba_dataframe_normalizes_alpha_above_one(local_rasters):
    tp = TifProcessor(
        local_rasters.put("a.tif", RGBA), data_store=local_rasters.store, mode="rgba"
    )

    df = tp.to_dataframe()

    assert list(df.columns) == ["lon", "lat", "red", "green", "blue", "alpha"]
    assert df["alpha"].unique().tolist() == [1.0]


def test_rgba_dataframe_keeps_alpha_already_in_unit_range(local_rasters):
    data = RGBA.copy()
    data[3] = 0.5
    tp = TifProcessor(
        local_rasters.put("a.tif", data), data_store=local_rasters.store, mode="rgba"
    )

    assert tp.to_dataframe()["alpha"].unique().tolist() == [0.5]


def test_geodataframe_has_one_box_per_pixel_in_the_raster_crs(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    gdf = tp.to_geodataframe()

    assert len(gdf) == 16 and gdf.crs == "EPSG:4326"
    assert gdf.geometry.iloc[0].equals_exact(box(0.0, 0.03, 0.01, 0.04), 1e-9)
    assert gdf["pixel_value"].tolist() == BAND.ravel().tolist()


@pytest.mark.parametrize("chunk_size", [1, 3, None])
def test_single_band_chunked_matches_to_dataframe(local_rasters, chunk_size):
    data = BAND.copy()
    data[1, 2] = DEFAULT_NODATA
    tp = TifProcessor(local_rasters.put("a.tif", data), data_store=local_rasters.store)

    chunked = tp.to_dataframe_chunked(chunk_size=chunk_size)

    pd.testing.assert_frame_equal(chunked, tp.to_dataframe(), check_exact=False)


# ─────────────────────────────────────────────────────────────────────────────
# Clipping
# ─────────────────────────────────────────────────────────────────────────────

SOUTH_WEST = box(0.001, 0.001, 0.019, 0.019)


def test_clip_to_geometry_returns_a_smaller_processor(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    clipped = tp.clip_to_geometry(SOUTH_WEST)

    assert isinstance(clipped, TifProcessor)
    assert clipped.bounds == pytest.approx((0.0, 0.0, 0.02, 0.02))
    assert clipped.to_dataframe()["pixel_value"].tolist() == [9.0, 10.0, 13.0, 14.0]
    assert (clipped.crs, clipped.mode) == ("EPSG:4326", "single")


def test_clip_to_bounds_returns_a_smaller_processor(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    clipped = tp.clip_to_bounds(SOUTH_WEST.bounds)

    assert clipped.bounds == pytest.approx((0.0, 0.0, 0.02, 0.02))
    assert clipped.to_dataframe()["pixel_value"].tolist() == [9.0, 10.0, 13.0, 14.0]


def test_clip_to_bounds_reprojects_bounds_given_in_another_crs(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)
    metres = gpd.GeoSeries([SOUTH_WEST], crs="EPSG:4326").to_crs("EPSG:3857")

    clipped = tp.clip_to_bounds(tuple(metres.total_bounds), bounds_crs="EPSG:3857")

    assert clipped.to_dataframe()["pixel_value"].tolist() == [9.0, 10.0, 13.0, 14.0]


def test_clip_without_processor_returns_array_transform_and_metadata(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    array, transform, meta = tp.clip_to_geometry(
        SOUTH_WEST, return_clipped_processor=False
    )

    np.testing.assert_array_equal(array, [[[9.0, 10.0], [13.0, 14.0]]])
    assert transform == from_bounds(0.0, 0.0, 0.02, 0.02, width=2, height=2)
    assert (meta["width"], meta["height"], meta["nodata"]) == (2, 2, DEFAULT_NODATA)


def test_clip_to_bounds_without_processor_returns_a_tuple(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    array, _, _ = tp.clip_to_bounds(SOUTH_WEST.bounds, return_clipped_processor=False)

    assert array.shape == (1, 2, 2)


def test_clip_to_a_non_overlapping_geometry_raises(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    with pytest.raises(ValueError, match="does not overlap"):
        tp.clip_to_geometry(box(10, 10, 11, 11))


# ─────────────────────────────────────────────────────────────────────────────
# Reprojection
# ─────────────────────────────────────────────────────────────────────────────


def test_to_crs_returns_a_processor_in_the_target_crs(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    result = tp.to_crs("EPSG:3857")

    assert isinstance(result, TifProcessor)
    assert result.crs == "EPSG:3857"
    assert result.bounds.right == pytest.approx(4452.78, abs=0.01)
    assert tp.crs == "EPSG:4326"


def test_to_crs_to_the_same_crs_returns_an_independent_copy(local_rasters):
    path = local_rasters.put("a.tif")
    tp = TifProcessor(path, data_store=local_rasters.store)

    result = tp.to_crs("EPSG:4326")

    assert result is not tp and result.dataset_path != path
    pd.testing.assert_frame_equal(result.to_dataframe(), tp.to_dataframe())
    result.cleanup()
    assert Path(path).is_file()


def test_to_crs_defaults_to_the_processors_reprojection_resolution(local_rasters):
    tp = TifProcessor(
        local_rasters.put("a.tif"),
        data_store=local_rasters.store,
        reprojection_resolution=(2000.0, 2000.0),
    )

    inherited = tp.to_crs("EPSG:3857")
    overridden = tp.to_crs("EPSG:3857", resolution=(500.0, 500.0))

    assert inherited.resolution == pytest.approx((2000.0, 2000.0))
    assert overridden.resolution == pytest.approx((500.0, 500.0))


def test_reproject_to_is_removed():
    assert not hasattr(TifProcessor, "reproject_to")


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

POINTS = [(0.005, 0.035), (0.035, 0.005)]
NORTH_WEST = box(0.0, 0.02, 0.02, 0.04)  # pixels 1, 2, 5, 6
SOUTH_EAST = box(0.02, 0.0, 0.04, 0.02)  # pixels 11, 12, 15, 16
OUTSIDE = box(10, 10, 11, 11)
POLYGONS = [NORTH_WEST, SOUTH_EAST, OUTSIDE, box(0.0, 0.0, 0.04, 0.04)]


def test_sample_by_coordinates_single_band(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    np.testing.assert_array_equal(tp.sample_by_coordinates(POINTS), [1.0, 16.0])


def test_sample_by_coordinates_multi_band(local_rasters):
    tp = TifProcessor(
        local_rasters.put("a.tif", TWO_BANDS),
        data_store=local_rasters.store,
        mode="multi",
    )

    np.testing.assert_array_equal(
        tp.sample_by_coordinates(POINTS), [[1.0, 10.0], [16.0, 160.0]]
    )


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("rgb", {"red": [1.0, 16.0], "green": [101.0, 116.0], "blue": [201.0, 216.0]}),
        (
            "rgba",
            {
                "red": [1.0, 16.0],
                "green": [101.0, 116.0],
                "blue": [201.0, 216.0],
                "alpha": [255.0, 255.0],
            },
        ),
    ],
)
def test_sample_by_coordinates_colour_modes_return_a_dict_per_channel(
    local_rasters, mode, expected
):
    path = local_rasters.put("a.tif", BANDS_FOR_MODE[mode])
    tp = TifProcessor(path, data_store=local_rasters.store, mode=mode)

    samples = tp.sample_by_coordinates(POINTS)

    assert {k: [float(x) for x in v] for k, v in samples.items()} == expected


def test_sample_by_polygons_one_statistic(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    np.testing.assert_allclose(
        tp.sample_by_polygons(POLYGONS), [3.5, 13.5, np.nan, 8.5], equal_nan=True
    )


def test_sample_by_polygons_list_of_statistics(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    result = tp.sample_by_polygons([NORTH_WEST, OUTSIDE], stat=["mean", "max"])

    assert result[0] == {"mean": 3.5, "max": 6.0}
    assert set(result[1]) == {"mean", "max"} and np.isnan(result[1]["mean"])


def test_sample_by_polygons_count(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    assert tp.sample_by_polygons([NORTH_WEST, SOUTH_EAST], stat="count").tolist() == [
        4,
        4,
    ]


def test_sample_by_polygons_callable(local_rasters):
    def spread(values):
        return values.max() - values.min()

    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    assert tp.sample_by_polygons([NORTH_WEST], stat=spread).tolist() == [5.0]
    assert tp.sample_by_polygons([NORTH_WEST], stat=[spread]) == [{"spread": 5.0}]


def test_sample_by_polygons_batched_matches_sample_by_polygons(
    local_rasters, start_method
):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)
    expected = tp.sample_by_polygons(POLYGONS)
    np.testing.assert_allclose(expected, [3.5, 13.5, np.nan, 8.5], equal_nan=True)

    batched = tp.sample_by_polygons_batched(
        POLYGONS, n_workers=2, batch_size=1, show_progress=False
    )

    np.testing.assert_allclose(batched, expected, equal_nan=True)
    assert tp.to_dataframe()["pixel_value"].tolist() == BAND.ravel().tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Pixel graph
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("connectivity, edges", [(4, 24), (8, 42)])
def test_networkx_pixel_graph(local_rasters, connectivity, edges):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    graph = tp.to_graph(connectivity=connectivity)

    assert (graph.number_of_nodes(), graph.number_of_edges()) == (16, edges)
    assert graph.nodes[0] == {"value": 1.0}
    assert graph.edges[0, 1]["weight"] == 2.0


@pytest.mark.parametrize("connectivity, edges", [(4, 24), (8, 42)])
def test_sparse_pixel_graph(local_rasters, connectivity, edges):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)

    matrix = tp.to_graph(connectivity=connectivity, graph_type="sparse")

    assert matrix.shape == (16, 16) and matrix.nnz == 2 * edges
    assert (matrix != matrix.T).nnz == 0
    assert (matrix[0, 1], matrix[1, 0]) == (2.0, 2.0)


def test_pixel_graph_with_coordinates_skips_nodata(local_rasters):
    data = BAND.copy()
    data[0, 1] = DEFAULT_NODATA
    tp = TifProcessor(local_rasters.put("a.tif", data), data_store=local_rasters.store)

    graph = tp.to_graph(include_coordinates=True)

    assert graph.number_of_nodes() == 15
    assert graph.nodes[0] == pytest.approx({"value": 1.0, "x": 0.005, "y": 0.035})
    assert not graph.has_edge(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Saving
# ─────────────────────────────────────────────────────────────────────────────


def test_save_to_file_round_trips_through_the_data_store(rasters):
    tp = TifProcessor(rasters.put("a.tif"), data_store=rasters.store)

    out = tp.save_to_file(rasters.output("out.tif"))

    assert out == Path(rasters.output("out.tif")) and rasters.store.file_exists(
        str(out)
    )
    saved = TifProcessor(str(out), data_store=rasters.store)
    assert (saved.crs, saved.bounds, saved.nodata) == (tp.crs, tp.bounds, tp.nodata)
    assert saved.to_dataframe().equals(tp.to_dataframe())


def test_save_array_to_file_uses_the_processor_georeferencing(rasters):
    tp = TifProcessor(rasters.put("a.tif"), data_store=rasters.store)

    out = tp.save_array_to_file(BAND * 2, rasters.output("doubled.tif"))

    saved = TifProcessor(str(out), data_store=rasters.store)
    assert (saved.crs, saved.bounds, saved.nodata) == (tp.crs, tp.bounds, tp.nodata)
    assert saved.to_dataframe()["pixel_value"].tolist() == (BAND * 2).ravel().tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle
#
# The processor's temporary directory has no public handle; _temp_dir is the
# narrowest one.
# ─────────────────────────────────────────────────────────────────────────────


def test_cleanup_deletes_the_temporary_directory(local_rasters):
    tp = TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store)
    temp_dir = Path(tp._temp_dir)
    assert temp_dir.is_dir()

    tp.cleanup()

    assert not temp_dir.exists()


def test_context_manager_cleans_up_on_exit(local_rasters):
    with TifProcessor(local_rasters.put("a.tif"), data_store=local_rasters.store) as tp:
        temp_dir = Path(tp._temp_dir)
        assert temp_dir.is_dir()

    assert not temp_dir.exists()


def test_processor_over_a_stored_file_stays_readable_after_cleanup(rasters):
    tp = TifProcessor(rasters.put("a.tif"), data_store=rasters.store)

    tp.cleanup()
    assert tp.to_dataframe()["pixel_value"].tolist() == BAND.ravel().tolist()
    tp.cleanup()

    assert tp.to_dataframe()["pixel_value"].tolist() == BAND.ravel().tolist()


def test_deepcopy_stays_readable_after_the_original_is_cleaned_up(rasters):
    tp = TifProcessor(rasters.put("a.tif"), data_store=rasters.store)
    duplicate = copy.deepcopy(tp)

    tp.cleanup()

    assert duplicate.to_dataframe()["pixel_value"].tolist() == BAND.ravel().tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Local and non-local data stores give the same results
# ─────────────────────────────────────────────────────────────────────────────


def test_opens_and_converts_through_either_store(rasters):
    tp = TifProcessor(
        rasters.put("a.tif", TWO_BANDS), data_store=rasters.store, mode="multi"
    )

    assert (tp.crs, tp.count, tp.bounds) == (
        "EPSG:4326",
        2,
        BoundingBox(*DEFAULT_BOUNDS),
    )
    assert tp.to_dataframe()["band_2"].tolist() == (BAND * 10).ravel().tolist()


def test_merges_through_either_store(rasters):
    west = rasters.put("west.tif", BAND)
    east = rasters.put("east.tif", BAND + 16, bounds=EAST_BOUNDS)

    tp = TifProcessor([west, east], data_store=rasters.store, merge_method="max")

    np.testing.assert_array_equal(
        tp.to_dataframe()["pixel_value"].to_numpy().reshape(4, 8),
        np.hstack([BAND, BAND + 16]),
    )


def test_clips_through_either_store(rasters):
    tp = TifProcessor(rasters.put("a.tif"), data_store=rasters.store)

    clipped = tp.clip_to_geometry(SOUTH_WEST)

    assert clipped.to_dataframe()["pixel_value"].tolist() == [9.0, 10.0, 13.0, 14.0]


def test_samples_through_either_store(rasters):
    tp = TifProcessor(rasters.put("a.tif"), data_store=rasters.store)

    np.testing.assert_array_equal(tp.sample_by_coordinates(POINTS), [1.0, 16.0])
    np.testing.assert_allclose(
        tp.sample_by_polygons(POLYGONS), [3.5, 13.5, np.nan, 8.5], equal_nan=True
    )


def test_batched_samples_through_either_store(rasters, start_method):
    tp = TifProcessor(rasters.put("a.tif"), data_store=rasters.store)

    batched = tp.sample_by_polygons_batched(
        POLYGONS, n_workers=2, batch_size=1, show_progress=False
    )

    np.testing.assert_allclose(batched, [3.5, 13.5, np.nan, 8.5], equal_nan=True)


# ─────────────────────────────────────────────────────────────────────────────
# Merged and clipped processors behave like one over a stored file
# ─────────────────────────────────────────────────────────────────────────────


def merged_west_east(rasters):
    west = rasters.put("west.tif", BAND)
    east = rasters.put("east.tif", BAND + 16, bounds=EAST_BOUNDS)
    return TifProcessor([west, east], data_store=rasters.store)


def clipped_south(rasters):
    tp = TifProcessor(rasters.put("a.tif"), data_store=rasters.store)
    return tp.clip_to_geometry(box(0.001, 0.001, 0.039, 0.019))


@pytest.mark.parametrize(
    "derive, polygon_means, pixels",
    [(merged_west_east, [3.5, 13.5], 32), (clipped_south, [np.nan, 13.5], 8)],
)
def test_derived_processor_samples_converts_and_saves(
    rasters, derive, polygon_means, pixels
):
    tp = derive(rasters)

    np.testing.assert_allclose(
        tp.sample_by_polygons([NORTH_WEST, SOUTH_EAST]), polygon_means, equal_nan=True
    )
    assert tp.sample_by_coordinates([(0.035, 0.005)]).tolist() == [16.0]
    assert len(tp.to_geodataframe()) == pixels
    saved = TifProcessor(
        str(tp.save_to_file(rasters.output("out.tif"))), data_store=rasters.store
    )
    assert saved.to_dataframe().equals(tp.to_dataframe())

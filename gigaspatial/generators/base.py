import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Type, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.core.io.writers import write_dataset
from gigaspatial.handlers.ghsl import GHSLDataHandler
from gigaspatial.handlers.rwi import RWIHandler
from gigaspatial.handlers.worldpop import WPPopulationHandler
from gigaspatial.processing.geo import (
    aggregate_points_to_zones,
    aggregate_polygons_to_zones,
    buffer_geodataframe,
)
from gigaspatial.processing.tif_processor import TifProcessor

_GEO_SUFFIXES = frozenset({".geojson", ".gpkg", ".shp"})
_TIF_INIT_KEYS = frozenset(
    {
        "mode",
        "merge_method",
        "target_crs",
        "resampling_method",
        "reprojection_resolution",
    }
)


class BaseViewGeneratorConfig(BaseModel):
    """Base configuration for view generators.

    .. deprecated::
        Pass ``ensure_available=`` directly to the generator constructor and
        pass a destination file path directly to :meth:`BaseViewGenerator.save_view`.
    """

    base_path: Path = Field(
        default=global_config.ROOT_DATA_DIR / global_config.VIEWS_DATA_DIR
    )
    output_format: str = "parquet"
    ensure_available: bool = True


class BaseViewGenerator:
    """Internal base class for POI and zonal view generators.

    Unifies canonical ``id_column`` view management, defensive view copies,
    ``save_view`` persistence via :func:`write_dataset`, ``overwrite``
    collision checks, empty-feature handling, and shared spatial/dataset
    enrichment pipelines.
    """

    id_column: str = "id"
    _default_config_cls: Type[BaseViewGeneratorConfig] = BaseViewGeneratorConfig
    _is_point_generator: bool = False
    _coverage_label: str = "items"
    _default_built_s_radius: Optional[float] = None
    _default_aqueduct_buffer_meters: Optional[float] = None

    def __init__(
        self,
        *,
        base_gdf: gpd.GeoDataFrame,
        ensure_available: bool = True,
        config: Optional[BaseViewGeneratorConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if config is not None:
            warnings.warn(
                f"{type(config).__name__} is deprecated and will be removed in a future "
                "minor release; pass `ensure_available=` directly to the generator "
                "constructor and a target file path directly to `save_view(path)`.",
                DeprecationWarning,
                stacklevel=3,
            )
            self.config = config
            self.ensure_available = (
                config.ensure_available if ensure_available is True else ensure_available
            )
        else:
            self.ensure_available = ensure_available
            self.config = self._default_config_cls(ensure_available=ensure_available)

        self.data_store = data_store or LocalDataStore()
        self.logger = logger or global_config.get_logger(self.__class__.__name__)
        if not hasattr(self, "_country"):
            self._country: Optional[str] = None

        self._base_gdf: gpd.GeoDataFrame = base_gdf
        self._view: pd.DataFrame = self._base_gdf.drop(columns=["geometry"]).copy()
        self._transformed_geoms_cache: Dict[str, List] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # View access, export, and persistence
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def view(self) -> pd.DataFrame:
        """Defensive copy of the current geometry-free view DataFrame."""
        return self._view.copy()

    def to_dataframe(self) -> pd.DataFrame:
        """Return a defensive copy of the current view as a pandas DataFrame."""
        return self.view

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Return the current view merged with geometries in the generator's CRS."""
        merged = self._view.merge(
            self._base_gdf[[self.id_column, "geometry"]],
            on=self.id_column,
            how="left",
        )
        return gpd.GeoDataFrame(
            merged,
            geometry="geometry",
            crs=self._base_gdf.crs,
        )

    def save_view(
        self,
        path: Optional[Union[str, Path]] = None,
        include_geometry: Optional[bool] = None,
        output_format: Optional[str] = None,
        **kwargs,
    ) -> Path:
        """Save the current view to ``path`` using :func:`write_dataset`.

        Args:
            path: Target file path (with extension, e.g. ``"views/out.parquet"``
                or ``"views/out.geojson"``). Legacy bare names resolved via
                ``config.base_path`` and ``output_format`` are deprecated.
            include_geometry: Whether to attach geometry via :meth:`to_geodataframe`.
                If ``None`` (default), geometry is attached automatically for
                geospatial extensions (``.geojson``, ``.gpkg``, ``.shp``).
            output_format: Deprecated format override when passing a bare name.
            **kwargs: Additional keyword arguments forwarded to :func:`write_dataset`.

        Returns:
            Path to the written output file.
        """
        used_legacy_name = False
        if path is None and "name" in kwargs:
            path = kwargs.pop("name")
            used_legacy_name = True

        if path is None:
            raise ValueError("path must be provided to save_view.")

        # Support legacy positional call: save_view("name", "csv")
        if isinstance(include_geometry, str):
            if output_format is not None:
                raise TypeError(
                    "Cannot specify output_format both positionally and by keyword."
                )
            output_format = include_geometry
            include_geometry = None

        raw_path = Path(path)
        if used_legacy_name or output_format is not None or raw_path.suffix == "":
            warnings.warn(
                "Passing a bare view name or `output_format` to `save_view` is "
                "deprecated; pass a target file path with extension (e.g., "
                "`save_view('views/out.parquet')`) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            fmt = (output_format or self.config.output_format).lstrip(".")
            if raw_path.suffix:
                output_path = (
                    raw_path.with_suffix(f".{fmt}")
                    if raw_path.suffix.lower() != f".{fmt.lower()}"
                    else raw_path
                )
            else:
                output_path = Path(self.config.base_path) / f"{raw_path}.{fmt}"
        else:
            output_path = raw_path

        suffix = output_path.suffix.lower()
        attach_geometry = (
            suffix in _GEO_SUFFIXES if include_geometry is None else include_geometry
        )
        if not attach_geometry and suffix in _GEO_SUFFIXES:
            raise ValueError(
                f"Cannot save geospatial format '{suffix}' with include_geometry=False."
            )

        self.logger.info(f"Saving view to {output_path}")
        data_to_save = self.to_geodataframe() if attach_geometry else self.to_dataframe()
        write_dataset(
            data=data_to_save,
            path=str(output_path),
            data_store=self.data_store,
            **kwargs,
        )
        return output_path

    # ──────────────────────────────────────────────────────────────────────────
    # View mutation, collision validation, and shared utilities
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_output_columns(
        self, columns: Iterable[str], *, overwrite: bool
    ) -> None:
        if overwrite:
            return
        colliding = [
            col
            for col in columns
            if col != self.id_column and col in self._view.columns
        ]
        if colliding:
            raise ValueError(
                f"Output column(s) already exist in view: {colliding}. "
                "Pass overwrite=True or choose a different output_column."
            )

    def _update_view(
        self, new_data: pd.DataFrame, *, overwrite: bool = False
    ) -> pd.DataFrame:
        """Merge enrichment columns from ``new_data`` into ``self._view``."""
        if self.id_column not in new_data.columns:
            raise ValueError(
                f"new_data DataFrame must contain '{self.id_column}' column. "
                f"Available columns: {list(new_data.columns)}"
            )

        new_cols = [col for col in new_data.columns if col != self.id_column]
        self._validate_output_columns(new_cols, overwrite=overwrite)

        original_ids = set(self._view[self.id_column])
        new_ids = set(new_data[self.id_column])
        missing_ids = original_ids - new_ids
        extra_ids = new_ids - original_ids

        if missing_ids:
            self.logger.warning(
                f"{len(missing_ids)} {self._coverage_label} will have NaN values for new columns"
            )
        if extra_ids:
            self.logger.warning(
                f"{len(extra_ids)} extra id(s) in new_data not present in view will be ignored"
            )

        new_data_indexed = new_data.set_index(self.id_column).copy()
        base = self._view.set_index(self.id_column)
        if overwrite:
            to_drop = [col for col in new_data_indexed.columns if col in base.columns]
            if to_drop:
                base = base.drop(columns=to_drop)

        updated = base.join(new_data_indexed, how="left").reset_index()
        updated.index.name = None
        self._view = updated
        self.logger.debug(f"View updated with columns: {list(new_data_indexed.columns)}")
        return self.view

    def _fill_empty_columns(
        self, fill_values: Dict[str, Any], *, overwrite: bool = False
    ) -> pd.DataFrame:
        """Add or overwrite ``fill_values`` columns across all rows in ``self._view``."""
        self._validate_output_columns(fill_values.keys(), overwrite=overwrite)
        n = len(self._view)
        data: Dict[str, Any] = {self.id_column: self._view[self.id_column].to_numpy()}
        for col, val in fill_values.items():
            if isinstance(val, (bool, np.bool_)):
                data[col] = np.full(n, bool(val), dtype=bool)
            elif isinstance(val, (int, np.integer)):
                data[col] = np.full(n, int(val), dtype=int)
            elif isinstance(val, (float, np.floating)) and np.isnan(val):
                data[col] = np.full(n, np.nan, dtype=float)
            else:
                data[col] = [val] * n
        return self._update_view(pd.DataFrame(data), overwrite=overwrite)

    def add_variable_to_view(
        self,
        data_dict: Union[Dict, pd.Series],
        column_name: str,
        overwrite: bool = False,
    ) -> None:
        """Add a mapping or Series keyed by ``self.id_column`` to the view."""
        self._validate_output_columns([column_name], overwrite=overwrite)
        if isinstance(data_dict, pd.Series):
            new_df = (
                data_dict.rename(column_name)
                .rename_axis(self.id_column)
                .reset_index()
            )
        elif not data_dict:
            new_df = pd.DataFrame(
                {
                    self.id_column: pd.Series(dtype=self._view[self.id_column].dtype),
                    column_name: pd.Series(dtype=float),
                }
            )
        else:
            new_df = pd.DataFrame(
                {
                    self.id_column: list(data_dict.keys()),
                    column_name: list(data_dict.values()),
                }
            )
        self._update_view(new_df, overwrite=overwrite)
        self.logger.info(f"Added variable '{column_name}' to the view.")

    def chain_operations(self, operations: List[dict]) -> "BaseViewGenerator":
        """Execute a sequence of ``{'method': ..., 'kwargs': ...}`` operations."""
        for op in operations:
            method_name = op["method"]
            kwargs = op.get("kwargs", {})
            if hasattr(self, method_name):
                getattr(self, method_name)(**kwargs)
            else:
                raise AttributeError(f"Method {method_name} not found")
        return self

    def validate_data_coverage(self, data_bounds: gpd.GeoDataFrame) -> dict:
        """Calculate how many generator places lie within ``data_bounds``."""
        bounds_aligned = (
            data_bounds.to_crs(self._base_gdf.crs)
            if data_bounds.crs is not None and data_bounds.crs != self._base_gdf.crs
            else data_bounds
        )
        within_mask = self._base_gdf.within(bounds_aligned.union_all())
        total = len(self._base_gdf)
        covered = int(within_mask.sum())
        pct = (covered / total) * 100 if total > 0 else 0.0
        label = self._coverage_label
        return {
            f"total_{label}": total,
            f"covered_{label}": covered,
            "coverage_percentage": pct,
            f"uncovered_{label}": total - covered,
        }

    def _resolve_ensure_available(self, ensure_available: Optional[bool]) -> bool:
        return self.ensure_available if ensure_available is None else ensure_available

    def _resolve_country(
        self, country: Optional[Union[str, List[str]]]
    ) -> Union[str, List[str]]:
        resolved = getattr(self, "_country", None) if country is None else country
        if resolved is None:
            raise ValueError(
                "country must be provided when the generator was not initialized from a country."
            )
        return resolved

    def _resolve_target_geometries(
        self,
        map_radius_meters: Optional[float] = None,
        *,
        require_polygons: bool = False,
    ) -> gpd.GeoDataFrame:
        if not self._is_point_generator:
            return self._base_gdf[[self.id_column, "geometry"]]

        if map_radius_meters is not None:
            if map_radius_meters <= 0:
                raise ValueError("map_radius_meters must be positive.")
            return buffer_geodataframe(
                self._base_gdf[[self.id_column, "geometry"]],
                buffer_distance_meters=map_radius_meters,
                cap_style="round",
            )

        if require_polygons:
            raise ValueError(
                "map_radius_meters must be provided for this operation on PoiViewGenerator."
            )

        return self._base_gdf[[self.id_column, "geometry"]]

    def _get_transformed_geometries(self, target_crs) -> List:
        key = str(target_crs)
        if key not in self._transformed_geoms_cache:
            self._transformed_geoms_cache[key] = (
                self._base_gdf.to_crs(target_crs).geometry.tolist()
            )
        return self._transformed_geoms_cache[key]

    @staticmethod
    def _ensure_tif_list(obj) -> List[TifProcessor]:
        """Normalize handler.load_data output into a flat list of TifProcessor."""
        if obj is None:
            return []
        if isinstance(obj, TifProcessor):
            return [obj]
        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            items = list(obj)
            if not items:
                return []
            if isinstance(items[0], TifProcessor):
                return items
            flat: List[TifProcessor] = []
            for it in items:
                if isinstance(it, TifProcessor):
                    flat.append(it)
                elif isinstance(it, Iterable) and not isinstance(it, (str, bytes)):
                    flat.extend(list(it))
            return flat
        raise TypeError(
            f"Unexpected type from load_data: {type(obj).__name__}; "
            f"expected TifProcessor or iterable of TifProcessor."
        )

    @staticmethod
    def _cleanup_tifs(tifs: Iterable[Any]) -> None:
        for tp in tifs:
            cleanup = getattr(tp, "cleanup", None)
            if callable(cleanup):
                cleanup()

    @staticmethod
    def _resolve_deprecated_ghsl_params(
        year: int,
        resolution: int,
        dataset_year: Optional[int],
        dataset_resolution: Optional[int],
    ) -> tuple[int, int]:
        if dataset_year is not None:
            warnings.warn(
                "'dataset_year' is deprecated and will be removed in a future minor "
                "release; use 'year' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            year = dataset_year
        if dataset_resolution is not None:
            warnings.warn(
                "'dataset_resolution' is deprecated and will be removed in a future minor "
                "release; use 'resolution' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            resolution = dataset_resolution
        return year, resolution

    # ──────────────────────────────────────────────────────────────────────────
    # Low-level spatial primitives (compute helpers + view-updating methods)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_spatial_output_schema(
        value_columns: Optional[Union[str, List[str]]],
        aggregation: Union[str, Dict[str, str]],
        output_suffix: str,
        output_column: Optional[str],
        *,
        default_count_col: str,
    ) -> Dict[str, Any]:
        if value_columns is None:
            if aggregation != "count":
                raise ValueError(
                    "value_columns must be specified for aggregation methods other than 'count'"
                )
            out_col = (
                output_column
                if output_column is not None
                else f"{default_count_col}{output_suffix}"
            )
            return {out_col: 0}

        cols = [value_columns] if isinstance(value_columns, str) else list(value_columns)
        if output_column is not None and len(cols) > 1:
            raise ValueError(
                "output_column cannot be used with multiple value_columns; use output_suffix instead."
            )

        if isinstance(aggregation, str):
            methods = {col: aggregation for col in cols}
        elif isinstance(aggregation, dict):
            missing = [col for col in cols if col not in aggregation]
            extra = [col for col in aggregation if col not in cols]
            if missing:
                raise ValueError(f"Missing aggregation methods for columns: {missing}")
            if extra:
                raise ValueError(
                    f"Aggregation methods specified for columns not in value_columns: {extra}"
                )
            methods = dict(aggregation)
        else:
            raise TypeError("aggregation must be a str or dict")

        if output_column is not None:
            col = cols[0]
            return {output_column: 0 if methods[col] == "count" else np.nan}

        return {
            f"{col}{output_suffix}": (0 if methods[col] == "count" else np.nan)
            for col in cols
        }

    def _prepare_point_mapping(
        self,
        value_columns: Optional[Union[str, List[str]]],
        aggregation: Union[str, Dict[str, str]],
        output_suffix: str,
        output_column: Optional[str],
        map_radius_meters: Optional[float],
    ) -> tuple[Dict[str, Any], gpd.GeoDataFrame]:
        fill_values = self._resolve_spatial_output_schema(
            value_columns=value_columns,
            aggregation=aggregation,
            output_suffix=output_suffix,
            output_column=output_column,
            default_count_col="point_count",
        )
        target_gdf = self._resolve_target_geometries(
            map_radius_meters, require_polygons=True
        )
        return fill_values, target_gdf

    def _compute_point_aggregation(
        self,
        points: Union[pd.DataFrame, gpd.GeoDataFrame],
        target_gdf: gpd.GeoDataFrame,
        output_cols: List[str],
        value_columns: Optional[Union[str, List[str]]] = None,
        aggregation: Union[str, Dict[str, str]] = "count",
        predicate: str = "within",
        output_suffix: str = "",
        output_column: Optional[str] = None,
    ) -> pd.DataFrame:
        result = aggregate_points_to_zones(
            points=points,
            zones=target_gdf,
            value_columns=value_columns,
            aggregation=aggregation,
            point_zone_predicate=predicate,
            zone_id_column=self.id_column,
            output_suffix=output_suffix,
            drop_geometry=True,
        )
        if output_column is not None:
            if value_columns is None:
                src_col = f"point_count{output_suffix}"
            else:
                first_col = (
                    value_columns
                    if isinstance(value_columns, str)
                    else value_columns[0]
                )
                src_col = f"{first_col}{output_suffix}"
            if src_col != output_column and src_col in result.columns:
                result = result.rename(columns={src_col: output_column})
        return result[[self.id_column, *output_cols]]

    def _aggregate_points(
        self,
        points: Union[pd.DataFrame, gpd.GeoDataFrame],
        value_columns: Optional[Union[str, List[str]]] = None,
        aggregation: Union[str, Dict[str, str]] = "count",
        predicate: str = "within",
        output_suffix: str = "",
        output_column: Optional[str] = None,
        map_radius_meters: Optional[float] = None,
    ) -> pd.DataFrame:
        fill_values, target_gdf = self._prepare_point_mapping(
            value_columns=value_columns,
            aggregation=aggregation,
            output_suffix=output_suffix,
            output_column=output_column,
            map_radius_meters=map_radius_meters,
        )
        return self._compute_point_aggregation(
            points=points,
            target_gdf=target_gdf,
            output_cols=list(fill_values.keys()),
            value_columns=value_columns,
            aggregation=aggregation,
            predicate=predicate,
            output_suffix=output_suffix,
            output_column=output_column,
        )

    def map_points(
        self,
        points: Union[pd.DataFrame, gpd.GeoDataFrame],
        value_columns: Optional[Union[str, List[str]]] = None,
        aggregation: Union[str, Dict[str, str]] = "count",
        predicate: str = "within",
        output_suffix: str = "",
        output_column: Optional[str] = None,
        map_radius_meters: Optional[float] = None,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """Aggregate point data onto generator zones or POI buffers and update ``self.view``."""
        fill_values, target_gdf = self._prepare_point_mapping(
            value_columns=value_columns,
            aggregation=aggregation,
            output_suffix=output_suffix,
            output_column=output_column,
            map_radius_meters=map_radius_meters,
        )
        self._validate_output_columns(fill_values.keys(), overwrite=overwrite)

        is_empty = (
            points is None
            or (hasattr(points, "empty") and points.empty)
            or (hasattr(points, "__len__") and len(points) == 0)
        )
        if is_empty:
            self.logger.warning("Empty point data provided; filling empty output columns.")
            return self._fill_empty_columns(fill_values, overwrite=overwrite)

        mapped = self._compute_point_aggregation(
            points=points,
            target_gdf=target_gdf,
            output_cols=list(fill_values.keys()),
            value_columns=value_columns,
            aggregation=aggregation,
            predicate=predicate,
            output_suffix=output_suffix,
            output_column=output_column,
        )
        return self._update_view(mapped, overwrite=overwrite)

    def _prepare_polygon_mapping(
        self,
        value_columns: Optional[Union[str, List[str]]],
        aggregation: Union[str, Dict[str, str]],
        predicate: Literal["intersects", "within", "fractional"],
        output_suffix: str,
        output_column: Optional[str],
        map_radius_meters: Optional[float],
    ) -> tuple[Dict[str, Any], gpd.GeoDataFrame]:
        if (
            self._is_point_generator
            and map_radius_meters is None
            and predicate in ("within", "fractional")
        ):
            raise ValueError(
                f"map_radius_meters must be provided on PoiViewGenerator for predicate='{predicate}'."
            )

        fill_values = self._resolve_spatial_output_schema(
            value_columns=value_columns,
            aggregation=aggregation,
            output_suffix=output_suffix,
            output_column=output_column,
            default_count_col="polygon_count",
        )
        target_gdf = self._resolve_target_geometries(
            map_radius_meters, require_polygons=False
        )
        return fill_values, target_gdf

    def _compute_polygon_aggregation(
        self,
        polygons: Union[pd.DataFrame, gpd.GeoDataFrame],
        target_gdf: gpd.GeoDataFrame,
        output_cols: List[str],
        value_columns: Optional[Union[str, List[str]]] = None,
        aggregation: Union[str, Dict[str, str]] = "count",
        predicate: Literal["intersects", "within", "fractional"] = "intersects",
        output_suffix: str = "",
        output_column: Optional[str] = None,
    ) -> pd.DataFrame:
        result = aggregate_polygons_to_zones(
            polygons=polygons,
            zones=target_gdf,
            value_columns=value_columns,
            aggregation=aggregation,
            predicate=predicate,
            zone_id_column=self.id_column,
            output_suffix=output_suffix,
            drop_geometry=True,
            output_column=output_column,
        )
        return result[[self.id_column, *output_cols]]

    def _aggregate_polygons(
        self,
        polygons: Union[pd.DataFrame, gpd.GeoDataFrame],
        value_columns: Optional[Union[str, List[str]]] = None,
        aggregation: Union[str, Dict[str, str]] = "count",
        predicate: Literal["intersects", "within", "fractional"] = "intersects",
        output_suffix: str = "",
        output_column: Optional[str] = None,
        map_radius_meters: Optional[float] = None,
    ) -> pd.DataFrame:
        fill_values, target_gdf = self._prepare_polygon_mapping(
            value_columns=value_columns,
            aggregation=aggregation,
            predicate=predicate,
            output_suffix=output_suffix,
            output_column=output_column,
            map_radius_meters=map_radius_meters,
        )
        return self._compute_polygon_aggregation(
            polygons=polygons,
            target_gdf=target_gdf,
            output_cols=list(fill_values.keys()),
            value_columns=value_columns,
            aggregation=aggregation,
            predicate=predicate,
            output_suffix=output_suffix,
            output_column=output_column,
        )

    def map_polygons(
        self,
        polygons: Union[pd.DataFrame, gpd.GeoDataFrame],
        value_columns: Optional[Union[str, List[str]]] = None,
        aggregation: Union[str, Dict[str, str]] = "count",
        predicate: Literal["intersects", "within", "fractional"] = "intersects",
        output_suffix: str = "",
        output_column: Optional[str] = None,
        map_radius_meters: Optional[float] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Aggregate polygon data onto generator zones or POIs/buffers and update ``self.view``."""
        fill_values, target_gdf = self._prepare_polygon_mapping(
            value_columns=value_columns,
            aggregation=aggregation,
            predicate=predicate,
            output_suffix=output_suffix,
            output_column=output_column,
            map_radius_meters=map_radius_meters,
        )
        self._validate_output_columns(fill_values.keys(), overwrite=overwrite)

        is_empty = (
            polygons is None
            or (hasattr(polygons, "empty") and polygons.empty)
            or (hasattr(polygons, "__len__") and len(polygons) == 0)
        )
        if is_empty:
            self.logger.warning("Empty polygon data provided; filling empty output columns.")
            return self._fill_empty_columns(fill_values, overwrite=overwrite)

        mapped = self._compute_polygon_aggregation(
            polygons=polygons,
            target_gdf=target_gdf,
            output_cols=list(fill_values.keys()),
            value_columns=value_columns,
            aggregation=aggregation,
            predicate=predicate,
            output_suffix=output_suffix,
            output_column=output_column,
        )
        return self._update_view(mapped, overwrite=overwrite)

    def _sample_rasters(
        self,
        raster_data: Union[TifProcessor, List[TifProcessor]],
        stat: str = "mean",
        output_column: str = "raster_value",
        map_radius_meters: Optional[float] = None,
        **kwargs,
    ) -> pd.DataFrame:
        raster_processor: Optional[TifProcessor] = None
        owned_processor = False

        tif_kwargs = {k: v for k, v in kwargs.items() if k in _TIF_INIT_KEYS}
        sample_kwargs = {k: v for k, v in kwargs.items() if k not in _TIF_INIT_KEYS}

        if isinstance(raster_data, TifProcessor):
            raster_processor = raster_data
        elif isinstance(raster_data, list) and all(
            isinstance(x, TifProcessor) for x in raster_data
        ):
            if not raster_data:
                self.logger.warning("No valid raster data provided")
                fill = 0 if stat == "count" else np.nan
                dtype = int if stat == "count" else float
                return pd.DataFrame(
                    {
                        self.id_column: self._base_gdf[self.id_column].to_numpy(),
                        output_column: np.full(len(self._base_gdf), fill, dtype=dtype),
                    }
                )

            if len(raster_data) > 1:
                all_source_paths = [tp.dataset_path for tp in raster_data]
                self.logger.info(
                    f"Merging {len(all_source_paths)} rasters into a single TifProcessor for raster aggregation."
                )
                raster_processor = TifProcessor(
                    dataset_path=all_source_paths,
                    data_store=self.data_store,
                    **tif_kwargs,
                )
                owned_processor = True
            else:
                raster_processor = raster_data[0]
        else:
            raise ValueError(
                "raster_data must be a TifProcessor object or a list of TifProcessor objects."
            )

        try:
            raster_crs = raster_processor.crs
            if self._is_point_generator:
                if map_radius_meters is not None:
                    buffers_gdf = self._resolve_target_geometries(map_radius_meters)
                    sampled_values = raster_processor.sample_by_polygons(
                        polygon_list=buffers_gdf.to_crs(raster_crs).geometry,
                        stat=stat,
                    )
                else:
                    coord_list = (
                        self._base_gdf.to_crs(raster_crs)
                        .get_coordinates()
                        .to_numpy()
                    )
                    sampled_values = raster_processor.sample_by_coordinates(
                        coordinate_list=coord_list, **sample_kwargs
                    )
            else:
                if raster_crs != self._base_gdf.crs:
                    self.logger.info(f"Projecting zones to raster CRS: {raster_crs}")
                    zone_geoms = self._get_transformed_geometries(raster_crs)
                else:
                    zone_geoms = self._base_gdf.geometry.tolist()

                sampled_values = raster_processor.sample_by_polygons(
                    polygon_list=zone_geoms, stat=stat
                )

            return pd.DataFrame(
                {
                    self.id_column: self._base_gdf[self.id_column].to_numpy(),
                    output_column: sampled_values,
                }
            )
        finally:
            if owned_processor and raster_processor is not None:
                raster_processor.cleanup()

    def map_rasters(
        self,
        raster_data: Union[TifProcessor, List[TifProcessor]],
        stat: str = "mean",
        output_column: str = "raster_value",
        map_radius_meters: Optional[float] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Sample raster data onto generator zones or POIs/buffers and update ``self.view``."""
        self._validate_output_columns([output_column], overwrite=overwrite)
        sampled = self._sample_rasters(
            raster_data=raster_data,
            stat=stat,
            output_column=output_column,
            map_radius_meters=map_radius_meters,
            **kwargs,
        )
        return self._update_view(sampled, overwrite=overwrite)

    # ──────────────────────────────────────────────────────────────────────────
    # Shared dataset enrichment methods
    # ──────────────────────────────────────────────────────────────────────────

    def map_wp_pop(
        self,
        country: Optional[Union[str, List[str]]] = None,
        map_radius_meters: Optional[float] = None,
        resolution: int = 1000,
        predicate: Literal[
            "centroid_within", "intersects", "fractional", "within"
        ] = "intersects",
        output_column: str = "population",
        handler: Optional[WPPopulationHandler] = None,
        ensure_available: Optional[bool] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Map WorldPop population data onto POIs or zones and update ``self.view``."""
        self._validate_output_columns([output_column], overwrite=overwrite)
        country = self._resolve_country(country)
        eff_ensure = self._resolve_ensure_available(ensure_available)
        countries_list = [country] if isinstance(country, str) else list(country)

        handler = handler or WPPopulationHandler(
            resolution=resolution,
            data_store=self.data_store,
            **kwargs,
        )

        stat = (
            "median" if handler.config.project == "degree_of_urbanization" else "sum"
        )

        if handler.config.project == "age_structures" and len(countries_list) > 1:
            raise ValueError(
                "For 'age_structures' project, only a single country can be processed at a time."
            )

        self.logger.info(
            f"Mapping WorldPop Population data (year: {handler.config.year}, resolution: {handler.config.resolution}m)"
        )

        if predicate == "fractional" and handler.config.resolution == 100:
            self.logger.warning(
                "Fractional aggregations only supported for datasets with 1000m resolution. Using `intersects` as predicate"
            )
            predicate = "intersects"

        if predicate == "centroid_within":
            if handler.config.project == "age_structures":
                iso = countries_list[0]
                raw = handler.load_data(
                    iso,
                    ensure_available=eff_ensure,
                    **handler.select_options(kwargs),
                )
                all_tif_processors = self._ensure_tif_list(raw)
                if not all_tif_processors:
                    return self._fill_empty_columns(
                        {output_column: np.nan}, overwrite=overwrite
                    )

                self.logger.info(
                    f"Sampling individual age_structures rasters using '{stat}' statistic and summing results."
                )
                total_vals: Optional[np.ndarray] = None
                id_vals = self._base_gdf[self.id_column].to_numpy()
                try:
                    for tif_processor in all_tif_processors:
                        single_raster_df = self._sample_rasters(
                            raster_data=tif_processor,
                            stat=stat,
                            output_column=output_column,
                            map_radius_meters=map_radius_meters,
                        )
                        vals = single_raster_df[output_column].to_numpy(dtype=float)
                        total_vals = vals if total_vals is None else (total_vals + vals)
                finally:
                    self._cleanup_tifs(all_tif_processors)

                combined = pd.DataFrame(
                    {self.id_column: id_vals, output_column: total_vals}
                )
                return self._update_view(combined, overwrite=overwrite)

            tif_processors: List[TifProcessor] = []
            try:
                for iso in countries_list:
                    raw = handler.load_data(
                        iso,
                        ensure_available=eff_ensure,
                        **handler.select_options(kwargs),
                    )
                    tif_processors.extend(self._ensure_tif_list(raw))

                return self.map_rasters(
                    raster_data=tif_processors,
                    stat=stat,
                    output_column=output_column,
                    map_radius_meters=map_radius_meters,
                    overwrite=overwrite,
                )
            finally:
                self._cleanup_tifs(tif_processors)

        # Polygon-based path ("intersects", "fractional", "within")
        self._resolve_target_geometries(map_radius_meters, require_polygons=True)
        gdf_pop = pd.concat(
            [
                handler.load_into_geodataframe(
                    c, ensure_available=eff_ensure, **kwargs
                )
                for c in countries_list
            ],
            ignore_index=True,
        )
        return self.map_polygons(
            polygons=gdf_pop,
            value_columns="pixel_value",
            aggregation=stat,
            predicate=predicate,
            output_column=output_column,
            map_radius_meters=map_radius_meters,
            overwrite=overwrite,
        )

    def map_ghsl(
        self,
        handler: Optional[GHSLDataHandler] = None,
        stat: str = "mean",
        output_column: Optional[str] = None,
        map_radius_meters: Optional[float] = None,
        ensure_available: Optional[bool] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Map Global Human Settlement Layer (GHSL) raster data and update ``self.view``."""
        coord_system = kwargs.pop("coord_system", None)
        if handler is None:
            if not coord_system:
                coord_system = 4326 if self._base_gdf.crs == "EPSG:4326" else 54009
            handler = GHSLDataHandler(
                data_store=self.data_store, coord_system=coord_system, **kwargs
            )

        column_name = (
            output_column
            if output_column
            else f"{handler.config.product.lower()}_{stat}"
        )
        self._validate_output_columns([column_name], overwrite=overwrite)

        eff_ensure = self._resolve_ensure_available(ensure_available)
        self.logger.info(
            f"Mapping {handler.config.product} data (year: {handler.config.year}, resolution: {handler.config.resolution}m)"
        )
        target_gdf = self._resolve_target_geometries(map_radius_meters).copy()
        raw = handler.load_data(
            target_gdf,
            ensure_available=eff_ensure,
            merge_rasters=True,
            **handler.select_options(kwargs),
        )
        tifs = self._ensure_tif_list(raw)
        try:
            return self.map_rasters(
                raster_data=tifs,
                stat=stat,
                output_column=column_name,
                map_radius_meters=map_radius_meters,
                overwrite=overwrite,
            )
        finally:
            self._cleanup_tifs(tifs)

    def map_built_s(
        self,
        year: int = 2020,
        resolution: int = 100,
        stat: str = "sum",
        output_column: str = "built_surface_m2",
        map_radius_meters: Optional[float] = None,
        handler: Optional[GHSLDataHandler] = None,
        ensure_available: Optional[bool] = None,
        overwrite: bool = False,
        dataset_year: Optional[int] = None,
        dataset_resolution: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Map GHSL Built-up Surface (GHS_BUILT_S) data and update ``self.view``."""
        resolved_year, resolved_resolution = self._resolve_deprecated_ghsl_params(
            year, resolution, dataset_year, dataset_resolution
        )
        effective_radius = (
            self._default_built_s_radius
            if map_radius_meters is None
            else map_radius_meters
        )
        handler = handler or GHSLDataHandler(
            product="GHS_BUILT_S",
            year=resolved_year,
            resolution=resolved_resolution,
            data_store=self.data_store,
            **kwargs,
        )
        return self.map_ghsl(
            handler=handler,
            stat=stat,
            output_column=output_column,
            map_radius_meters=effective_radius,
            ensure_available=ensure_available,
            overwrite=overwrite,
            **kwargs,
        )

    def map_smod(
        self,
        year: int = 2020,
        resolution: int = 1000,
        stat: str = "median",
        output_column: str = "smod_class",
        map_radius_meters: Optional[float] = None,
        handler: Optional[GHSLDataHandler] = None,
        ensure_available: Optional[bool] = None,
        overwrite: bool = False,
        dataset_year: Optional[int] = None,
        dataset_resolution: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Map GHSL Settlement Model (GHS_SMOD) data and update ``self.view``."""
        resolved_year, resolved_resolution = self._resolve_deprecated_ghsl_params(
            year, resolution, dataset_year, dataset_resolution
        )
        coord_system = kwargs.pop("coord_system", 54009)
        handler = handler or GHSLDataHandler(
            product="GHS_SMOD",
            year=resolved_year,
            resolution=resolved_resolution,
            data_store=self.data_store,
            coord_system=coord_system,
            **kwargs,
        )
        return self.map_ghsl(
            handler=handler,
            stat=stat,
            output_column=output_column,
            map_radius_meters=map_radius_meters,
            ensure_available=ensure_available,
            overwrite=overwrite,
            **kwargs,
        )

    def map_ghsl_pop(
        self,
        resolution: int = 100,
        stat: str = "sum",
        output_column: str = "ghsl_pop",
        predicate: Literal["intersects", "fractional"] = "intersects",
        year: int = 2020,
        map_radius_meters: Optional[float] = None,
        handler: Optional[GHSLDataHandler] = None,
        ensure_available: Optional[bool] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Map GHSL Population (GHS_POP) data and update ``self.view``."""
        self._validate_output_columns([output_column], overwrite=overwrite)
        eff_ensure = self._resolve_ensure_available(ensure_available)
        handler = handler or GHSLDataHandler(
            product="GHS_POP",
            year=year,
            resolution=resolution,
            data_store=self.data_store,
            **kwargs,
        )

        if predicate == "fractional":
            if handler.config.resolution == 100:
                self.logger.warning(
                    "Fractional aggregations only supported for datasets with 1000m resolution. Using `intersects` as predicate"
                )
                predicate = "intersects"
            else:
                target_gdf = self._resolve_target_geometries(
                    map_radius_meters, require_polygons=True
                ).copy()
                gdf_pop = handler.load_into_geodataframe(
                    target_gdf,
                    ensure_available=eff_ensure,
                    **handler.select_options(kwargs),
                )
                return self.map_polygons(
                    polygons=gdf_pop,
                    value_columns="pixel_value",
                    aggregation="sum",
                    predicate="fractional",
                    output_column=output_column,
                    map_radius_meters=map_radius_meters,
                    overwrite=overwrite,
                )

        return self.map_ghsl(
            handler=handler,
            stat=stat,
            output_column=output_column,
            map_radius_meters=map_radius_meters,
            ensure_available=eff_ensure,
            overwrite=overwrite,
            **kwargs,
        )

    def map_rwi(
        self,
        country: Optional[Union[str, List[str]]] = None,
        predicate: Literal["intersects", "within", "centroid_within"] = "intersects",
        aggregation: Literal["mean", "median", "max", "min"] = "mean",
        output_column: Optional[str] = None,
        map_radius_meters: Optional[float] = None,
        handler: Optional[RWIHandler] = None,
        ensure_available: Optional[bool] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Map Relative Wealth Index (RWI) data and update ``self.view``."""
        target_col = (
            output_column if output_column is not None else f"rwi_{aggregation}"
        )
        self._validate_output_columns([target_col], overwrite=overwrite)

        country = self._resolve_country(country)
        eff_ensure = self._resolve_ensure_available(ensure_available)
        handler = handler or RWIHandler(data_store=self.data_store, **kwargs)

        if not handler.config.get_relevant_data_units(country):
            raise ValueError(f"No RWI data units found for country '{country}'.")

        if predicate == "centroid_within":
            data = handler.load_data(country, ensure_available=eff_ensure)
            return self.map_points(
                points=data,
                value_columns="rwi",
                aggregation=aggregation,
                output_column=target_col,
                map_radius_meters=map_radius_meters,
                overwrite=overwrite,
            )

        data = handler.load_as_geodataframe(country, ensure_available=eff_ensure)
        return self.map_polygons(
            polygons=data,
            value_columns="rwi",
            aggregation=aggregation,
            predicate=predicate,
            output_column=target_col,
            map_radius_meters=map_radius_meters,
            overwrite=overwrite,
        )

    def map_aqueduct_flood_hazard(
        self,
        handler=None,
        return_period: str = "rp00100",
        flood_type: str = "inunriver",
        climate_scenario: str = "historical",
        year: str = "1980",
        model: Optional[str] = "000000000WATCH",
        subsidence: Optional[str] = None,
        projection: Optional[str] = None,
        buffer_meters: Optional[float] = None,
        stat: str = "mean",
        output_column: Optional[str] = None,
        ensure_available: Optional[bool] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Map WRI Aqueduct 3.0 flood hazard depth and update ``self.view``."""
        from gigaspatial.handlers.wri import AqueductFloodHandler

        h = handler or AqueductFloodHandler(
            flood_type=flood_type,
            climate_scenario=climate_scenario,
            year=year,
            return_period=return_period,
            model=model,
            subsidence=subsidence,
            projection=projection,
            data_store=self.data_store,
        )

        col = output_column or (
            h.config._get_filename().replace(".tif", "") + "_meters"
        )
        self._validate_output_columns([col], overwrite=overwrite)

        effective_radius = (
            self._default_aqueduct_buffer_meters
            if buffer_meters is None
            else buffer_meters
        )
        eff_ensure = self._resolve_ensure_available(ensure_available)
        self.logger.info(
            f"Mapping WRI Aqueduct flood hazard ({h.config.DATASET_URL})"
        )

        target_gdf = self._resolve_target_geometries(effective_radius).copy()
        raw = h.load_data(
            target_gdf,
            ensure_available=eff_ensure,
            **kwargs,
        )
        tifs = self._ensure_tif_list(raw)
        try:
            return self.map_rasters(
                raster_data=tifs,
                stat=stat,
                output_column=col,
                map_radius_meters=effective_radius,
                overwrite=overwrite,
            )
        finally:
            self._cleanup_tifs(tifs)

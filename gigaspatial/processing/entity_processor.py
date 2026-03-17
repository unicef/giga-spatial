from __future__ import annotations

import unicodedata
import uuid
from functools import wraps
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Set,
    Union,
)

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from gigaspatial.config import config
from gigaspatial.core.schemas.shared import ENTITY_UUID_NAMESPACE, NULL_LIKE_VALUES
from gigaspatial.processing.geo import (
    annotate_with_admin_regions,
    convert_to_geodataframe,
    detect_coordinate_columns,
)

logger = config.get_logger("EntityProcessor")


class EntityProcessor:
    """
    Cleans and normalises raw entity DataFrames before Pydantic validation.

    Designed to operate at the Silver layer of a medallion architecture —
    applied after manual Bronze inspection and before Gold schema validation
    via ``EntityTable.from_file`` or ``EntityTable.from_dataframe``.

    Pipeline (executed in order by ``process``):

    .. code-block:: text

        1.  Lowercase + strip column names
        2.  Rename columns        (static aliases + coordinate auto-detection)
        3.  Strip string values   (NFKC-normalised, geometry-safe)
        4.  Coerce null sentinels (geometry-safe)
        5.  Repair coordinates    (trailing commas, merged lat/lon)
        6.  Coerce numeric cols   (pd.to_numeric, errors → NaN)
        7.  Normalise enum casing (LOWERCASE_COLUMNS)
        8.  Parse geometry        (WKT / WKB / Shapely pass-through)
        9.  Drop all-null rows
        10. Deduplicate           (geometry-safe, excludes Shapely cols)
        11. Boundary filter       (optional, requires ``country`` kwarg)
        12. Admin region annotation (optional, requires ``country`` kwarg)

    Class attributes (override in subclasses):
        LOWERCASE_COLUMNS: Columns to force-lowercase for enum matching.
        NUMERIC_COLUMNS:   Columns to coerce with ``pd.to_numeric``.
        COLUMN_ALIASES:    Static non-coordinate column renames applied
                           before coordinate auto-detection.
        verbose:           If ``False``, INFO-level pipeline step logs are
                           suppressed. WARNING and ERROR always surface.
                           Defaults to ``True``.

    Silver-layer utilities (call manually, not part of ``process``):
        filter_by_country_boundary  — clip rows to a country polygon
        validate_coordinates        — flag or drop invalid lat/lon rows
        deduplicate_by_proximity    — spatial dedup via KDTree
        assign_entity_id            — generate deterministic UUID3 identifiers

    Example::

        processor = CellTowerProcessor()
        processor.verbose = False          # suppress pipeline INFO logs

        df = pd.read_csv("bronze/ke/towers.csv")
        df = processor.process(df, country="KEN")
        tower_table = CellTowerTable.from_dataframe(df)
    """

    LOWERCASE_COLUMNS: ClassVar[List[str]] = []
    NUMERIC_COLUMNS: ClassVar[List[str]] = ["latitude", "longitude"]
    COLUMN_ALIASES: ClassVar[Dict[str, str]] = {}
    verbose: ClassVar[bool] = True

    def __init__(self, verbose: Optional[bool] = None):
        """
        Initialise the processor.

        Args:
            verbose: Override the class-level ``verbose`` setting.
        """
        if verbose is not None:
            self.verbose = verbose
        self.processing_logs: List[str] = []

    # ------------------------------------------------------------------
    # Internal logging helper
    # ------------------------------------------------------------------

    def _log(self, level: str, msg: str, *args) -> None:
        """
        Emit a log message, respecting the ``verbose`` flag.

        INFO-level messages are suppressed when ``verbose=False``.
        WARNING and ERROR messages are always emitted.

        Args:
            level: One of ``'debug'``, ``'info'``, ``'warning'``, ``'error'``.
            msg: Printf-style message string.
            *args: Arguments for printf-style message formatting.
        """
        if not self.verbose and level == "info":
            return

        formatted_msg = msg % args if args else msg
        if hasattr(self, "processing_logs"):
            self.processing_logs.append(f"[{level.upper()}] {formatted_msg}")

        getattr(logger, level)(msg, *args)

    def track_changes(func):
        """
        Decorator for EntityProcessor methods to automatically log shape changes.

        Calculates the difference in rows and columns before and after the
        decorated method runs, appending a summary to ``self.processing_logs``.
        """

        @wraps(func)
        def wrapper(self, df: pd.DataFrame, *args, **kwargs):
            if not hasattr(self, "processing_logs"):
                return func(self, df, *args, **kwargs)

            initial_rows, initial_cols = df.shape
            initial_col_names = set(df.columns)

            result_df = func(self, df, *args, **kwargs)

            final_rows, final_cols = result_df.shape
            final_col_names = set(result_df.columns)

            dropped_rows = initial_rows - final_rows
            added_cols = final_col_names - initial_col_names
            dropped_cols = initial_col_names - final_col_names

            changes = []
            if dropped_rows > 0:
                changes.append(f"dropped {dropped_rows} rows")
            elif dropped_rows < 0:
                changes.append(f"added {abs(dropped_rows)} rows")

            if added_cols:
                changes.append(f"added columns {sorted(list(added_cols))}")
            if dropped_cols:
                changes.append(f"dropped columns {sorted(list(dropped_cols))}")

            if changes:
                msg = f"STEP [{func.__name__.strip('_')}]: " + ", ".join(changes)
                self.processing_logs.append(msg)

            return result_df

        return wrapper

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Run the full cleaning pipeline on a raw DataFrame.

        Steps are executed in a fixed order designed to avoid order-dependent
        bugs (e.g. numeric coercion must follow string stripping; coordinate
        repair must precede numeric coercion).

        Args:
            df: Raw input data. May be a plain ``pd.DataFrame`` or a
                ``gpd.GeoDataFrame``; geometry columns are preserved throughout.
            **kwargs:
                country (str): Optional country identifier.  When provided:
                    - Rows outside the country boundary are removed via
                      ``filter_by_country_boundary``.
                    - Admin region columns are appended via
                      ``_annotate_with_admin_regions``.
                Additional kwargs are forwarded to both methods.

        Returns:
            Cleaned ``pd.DataFrame`` ready for Pydantic entity validation.
        """
        self.processing_logs = []  # Clear logs at start of process
        df = df.copy()

        # --- structural ---
        df.columns = [c.strip().lower() for c in df.columns]
        df = self._rename_columns(df)

        # --- string normalisation (geometry-safe, before numeric coercion) ---
        df = self._strip_strings(df)

        # --- null coercion (geometry-safe, on stripped values) ---
        df = self._coerce_nulls(df)

        # --- coordinate repair ---
        df = self._repair_coordinate_columns(df)

        # --- numeric coercion ---
        df = self._coerce_numeric_columns(df)

        # --- casing normalisation ---
        df = self._normalize_casing(df)

        # --- geometry parsing ---
        df = self._parse_geometry(df)

        # --- row cleanup ---
        df = self._drop_empty_rows(df)
        df = self._drop_duplicates(df)

        # --- enrichment ---
        country = kwargs.pop("country", None)
        if country:
            df = self._filter_by_country_boundary(df, boundary=country, **kwargs)
            df = self._annotate_with_admin_regions(df, country, **kwargs)

        return df

    # ------------------------------------------------------------------
    # Geometry column detection
    # ------------------------------------------------------------------

    @staticmethod
    def _get_geometry_cols(df: pd.DataFrame) -> set[str]:
        """
        Return column names that contain Shapely geometry objects.

        Detection strategy:

        1. If ``df`` is a ``GeoDataFrame``, include its declared geometry
           column (lowercased to match the column-normalisation step).
        2. For every remaining ``object``-dtype column, inspect the first
           non-null value; if it is a ``BaseGeometry`` instance, include it.

        This two-pass approach ensures both explicitly declared geometry
        columns and ad-hoc geometry-like columns are excluded from string
        and null-coercion operations.

        Args:
            df: DataFrame or GeoDataFrame to inspect.

        Returns:
            Set of lowercased column name strings identified as geometry.
        """
        from shapely.geometry.base import BaseGeometry

        geometry_cols: set[str] = set()

        if isinstance(df, gpd.GeoDataFrame):
            geometry_cols.add(df.geometry.name.strip().lower())

        for col in df.select_dtypes(include="object").columns:
            if col in geometry_cols:
                continue
            first_valid = df[col].dropna().iloc[:1]
            if not first_valid.empty and isinstance(first_valid.iloc[0], BaseGeometry):
                geometry_cols.add(col)

        return geometry_cols

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    @track_changes
    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns to canonical names in two passes.

        Pass 1 — static aliases:
            Applies ``COLUMN_ALIASES`` defined on the processor subclass.
            A source column is only renamed if the target name does not
            already exist, preventing accidental overwrites.

        Pass 2 — coordinate auto-detection:
            Calls ``detect_coordinate_columns`` to identify latitude /
            longitude columns by keyword matching when the canonical names
            ``'latitude'`` and ``'longitude'`` are not already present.
            Silently skipped for non-point entities (e.g. ``GigaGeoEntity``
            subclasses) that have no coordinate columns.

        Args:
            df: DataFrame after column lowercasing.

        Returns:
            DataFrame with columns renamed to canonical names.
        """
        self._log("info", "Renaming columns.")
        rename_map: Dict[str, str] = {}

        for src, tgt in self.COLUMN_ALIASES.items():
            if src in df.columns and tgt not in df.columns:
                rename_map[src] = tgt

        has_lat = "latitude" in df.columns
        has_lon = "longitude" in df.columns
        if not (has_lat and has_lon):
            try:
                lat_col, lon_col = detect_coordinate_columns(df)
                if not has_lat and lat_col != "latitude":
                    rename_map[lat_col] = "latitude"
                if not has_lon and lon_col != "longitude":
                    rename_map[lon_col] = "longitude"
            except ValueError as exc:
                logger.debug("Coordinate column detection skipped: %s", exc)

        if rename_map:
            logger.debug("Renaming columns: %s", rename_map)
            df = df.rename(columns=rename_map)

        return df

    @track_changes
    def _strip_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        NFKC-normalise and strip all string values in non-geometry columns.

        NFKC normalisation is applied before stripping to convert Unicode
        compatibility characters to their canonical ASCII equivalents:

        - Non-breaking spaces (``\\xa0``) → regular space
        - Thin spaces (``\\u2009``) → regular space
        - Fullwidth digits (``\\uff10``–``\\uff19``) → ASCII digits

        Geometry columns — both GeoDataFrame-declared and ad-hoc Shapely
        object columns — are excluded to prevent silent data corruption.

        Args:
            df: DataFrame after column renaming.

        Returns:
            DataFrame with string values normalised and stripped.
        """
        self._log("info", "Stripping whitespace from string columns.")
        geometry_cols = self._get_geometry_cols(df)
        str_cols = [
            col
            for col in df.select_dtypes(include="object").columns
            if col not in geometry_cols
        ]
        if not str_cols:
            return df

        def normalize_and_strip(v):
            if not isinstance(v, str):
                return v
            return unicodedata.normalize("NFKC", v).strip()

        df[str_cols] = df[str_cols].apply(lambda col: col.apply(normalize_and_strip))
        return df

    @track_changes
    def _coerce_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace null-like sentinel strings with ``None`` across non-geometry columns.

        Uses ``NULL_LIKE_VALUES`` — a module-level list covering common
        representations such as ``"none"``, ``"n/a"``, ``"nan"``, ``"-"``,
        ``"#N/A"``, and ``""`` (see module constants for the full list).

        Geometry columns are skipped because ``DataFrame.replace`` iterates
        over values for equality and Shapely objects are unhashable.

        Args:
            df: DataFrame after string stripping.

        Returns:
            DataFrame with null-like values replaced by ``None``.
        """
        self._log("info", "Coercing null-like values to None.")
        geometry_cols = self._get_geometry_cols(df)
        cols_to_coerce = [c for c in df.columns if c not in geometry_cols]
        df[cols_to_coerce] = df[cols_to_coerce].replace(NULL_LIKE_VALUES, None)
        return df

    @track_changes
    def _repair_coordinate_columns(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
    ) -> pd.DataFrame:
        """
        Repair common coordinate encoding errors before numeric coercion.

        Handles four cases found in real-world Bronze-layer data:

        - Whitespace padding:  ``'-16.5275 '``  → ``-16.5275``
        - Trailing comma:      ``'-13.457503,'`` → ``-13.457503``
        - Comma-merged pair:   ``'-13.457503, 25.326'`` →
          latitude ``-13.457503``, longitude ``25.326``
        - Space-merged pair:   ``'-12.325 25.326'`` →
          latitude ``-12.325``, longitude ``25.326``

        For merged pairs, the extracted longitude fills missing values in
        the longitude column only; existing non-null longitude values are
        never overwritten.

        Returns the DataFrame unchanged if ``lat_col`` is absent.

        Args:
            df: DataFrame after null coercion.
            lat_col: Name of the latitude column. Defaults to ``'latitude'``.
            lon_col: Name of the longitude column. Defaults to ``'longitude'``.

        Returns:
            DataFrame with coordinate columns repaired.
        """
        self._log("info", "Repairing coordinate columns.")
        if lat_col not in df.columns:
            return df

        def split_merged(v):
            if not isinstance(v, str):
                return v, None
            normalized = v.strip().replace(", ", " ").replace(",", " ")
            parts = normalized.split()
            if len(parts) == 2:
                try:
                    return float(parts[0]), float(parts[1])
                except ValueError:
                    pass
            try:
                return float(v.strip().strip(",").strip()), None
            except ValueError:
                return v, None

        split_results = df[lat_col].apply(split_merged)
        extracted_lats = split_results.apply(lambda x: x[0])
        extracted_lons = split_results.apply(lambda x: x[1])
        df[lat_col] = extracted_lats

        if lon_col in df.columns:
            lon_fill_mask = df[lon_col].isna() & extracted_lons.notna()
            if lon_fill_mask.any():
                df.loc[lon_fill_mask, lon_col] = extracted_lons[lon_fill_mask]
                self._log(
                    "info",
                    "%d longitude values recovered from merged lat/lon column.",
                    lon_fill_mask.sum(),
                )
        else:
            df[lon_col] = extracted_lons

        return df

    @track_changes
    def _coerce_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Coerce ``NUMERIC_COLUMNS`` to float via ``pd.to_numeric``.

        Values that cannot be parsed are silently set to ``NaN`` rather than
        raising, consistent with the soft-fail philosophy of the pipeline.
        Columns absent from ``df`` are skipped without warning.

        Args:
            df: DataFrame after coordinate repair.

        Returns:
            DataFrame with listed columns cast to float where possible.
        """
        self._log("info", "Coercing numeric columns: %s", self.NUMERIC_COLUMNS)
        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @track_changes
    def _normalize_casing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lowercase values in ``LOWERCASE_COLUMNS`` for consistent enum matching.

        Only applied to columns with ``object`` dtype; columns that have
        already been coerced to a numeric type are silently skipped.

        Args:
            df: DataFrame after numeric coercion.

        Returns:
            DataFrame with target columns lowercased.
        """
        self._log("info", "Normalising casing for columns: %s", self.LOWERCASE_COLUMNS)
        for col in self.LOWERCASE_COLUMNS:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].str.lower()
        return df

    @track_changes
    def _parse_geometry(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse the ``geometry`` column from WKT strings, WKB bytes, or
        existing Shapely objects (pass-through).

        Unparseable values are set to ``None`` and logged as warnings; rows
        are **not** dropped here — callers (``from_dataframe``) decide how
        to handle missing geometry via Pydantic validation failures.

        No-op if no ``geometry`` column is present in ``df``.

        Args:
            df: DataFrame after casing normalisation.

        Returns:
            DataFrame with ``geometry`` values as Shapely objects or ``None``.
        """
        self._log("info", "Parsing geometry column.")
        if "geometry" not in df.columns:
            logger.debug("No 'geometry' column found; skipping.")
            return df

        from shapely import wkt, wkb
        from shapely.geometry.base import BaseGeometry

        def _parse(value):
            if value is None:
                return None
            if isinstance(value, BaseGeometry):
                return value
            try:
                if isinstance(value, str):
                    return wkt.loads(value)
                if isinstance(value, (bytes, bytearray)):
                    return wkb.loads(value)
            except Exception as exc:
                logger.debug("Failed to parse geometry value %r: %s", value, exc)
            return None

        df["geometry"] = df["geometry"].apply(_parse)
        null_count = df["geometry"].isna().sum()
        if null_count:
            logger.warning("%d rows have unparseable or missing geometry.", null_count)
        return df

    @track_changes
    def _drop_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows where every value is ``None`` / ``NaN``.

        Applied after all coercion steps so that rows reduced to all-null
        by cleaning (e.g. a row containing only ``"N/A"`` values) are removed
        before deduplication and validation.

        Args:
            df: DataFrame after geometry parsing.

        Returns:
            DataFrame with all-null rows removed.
        """
        self._log("info", "Dropping all-null rows.")
        return df.dropna(how="all")

    @track_changes
    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate rows, excluding geometry columns from the comparison key.

        Shapely geometry objects are unhashable and raise ``TypeError`` when
        passed to ``drop_duplicates``.  The comparison subset is therefore
        all non-geometry columns; if none exist the DataFrame is returned
        unchanged.

        Args:
            df: DataFrame after empty-row removal.

        Returns:
            Deduplicated DataFrame.
        """
        self._log("info", "Dropping duplicate rows.")
        geometry_cols = self._get_geometry_cols(df)
        subset = [c for c in df.columns if c not in geometry_cols] or None
        return df.drop_duplicates(subset=subset)

    def _normalize_enum_column(
        self,
        df: pd.DataFrame,
        column: str,
        alias_map: Dict[str, str],
        valid_values: Set[str],
        required: bool = False,
    ) -> pd.DataFrame:
        """
        Normalise an enum-typed column via an alias map, setting unrecognised
        values to ``None``.

        Alias resolution is applied first, then any value not present in
        ``valid_values`` and not already ``None`` is nulled and logged as a
        warning. Values that are already ``None`` / ``NaN`` are left unchanged.

        Args:
            df: DataFrame containing the column to normalise.
            column: Name of the column to normalise.
            alias_map: Mapping of raw string value → canonical enum value.
                Applied via ``Series.replace`` after casing normalisation.
            valid_values: Complete set of accepted canonical string values.
                Typically ``{e.value for e in SomeEnum}``.
            required: If ``True``, log a ``WARNING`` when the column is absent;
                otherwise log at ``DEBUG`` level. Use ``True`` for columns
                that map to required Pydantic fields. Defaults to ``False``.

        Returns:
            DataFrame with the column normalised in-place.
        """
        if column not in df.columns:
            if required:
                logger.warning("Column '%s' not found, skipping normalisation.", column)
            else:
                logger.debug("Column '%s' not found, skipping normalisation.", column)
            return df

        df[column] = df[column].replace(alias_map)

        invalid_mask = ~df[column].isin(valid_values) & df[column].notna()
        if invalid_mask.any():
            logger.warning(
                "%d rows have unrecognised '%s' values: %s — setting to None.",
                invalid_mask.sum(),
                column,
                df.loc[invalid_mask, column].unique().tolist(),
            )
            df.loc[invalid_mask, column] = None

        return df

    def _annotate_with_admin_regions(
        self,
        df: pd.DataFrame,
        country,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Spatially annotate rows with administrative region labels.

        Converts ``df`` to a ``GeoDataFrame`` if it is not already one,
        performs the spatial join via ``annotate_with_admin_regions``, then
        drops the geometry column so the result is a plain ``DataFrame``
        consistent with the rest of the pipeline.

        Called automatically by ``process`` when a ``country`` kwarg is
        provided. Can also be called manually at the Silver layer for
        incremental enrichment.

        Args:
            df: DataFrame with ``latitude`` / ``longitude`` columns or an
                existing ``geometry`` column.
            country: Country identifier forwarded to
                ``annotate_with_admin_regions`` (ISO 3166-1 alpha-3 or
                country name depending on the annotator implementation).
            **kwargs: Additional keyword arguments forwarded to the annotator.

        Returns:
            Plain ``pd.DataFrame`` with admin-region columns appended and
            geometry column dropped.
        """
        self._log("info", "Annotating rows with administrative regions.")
        if not isinstance(df, gpd.GeoDataFrame):
            df = convert_to_geodataframe(df)
        df = annotate_with_admin_regions(df, country, **kwargs)
        return df.drop(columns="geometry")

    @track_changes
    def _filter_by_country_boundary(
        self,
        df: Union[pd.DataFrame, gpd.GeoDataFrame],
        boundary: Union[str, gpd.GeoDataFrame],
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        **kwargs,
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Filter rows to those spatially intersecting a country or administrative boundary.

        Rows whose geometry falls outside the dissolved boundary are removed.
        The boundary is dissolved into a single unified geometry before filtering
        to correctly handle multi-polygon and multi-feature country representations.

        For point-based DataFrames (``pd.DataFrame`` with lat/lon columns),
        the input is temporarily converted to a GeoDataFrame for the spatial
        filter then the original DataFrame type is returned.  For GeoDataFrames
        with polygon or multipolygon geometry (e.g. ``MobileCoverage``,
        ``AdminBoundary``), the existing geometry column is used directly and a
        ``GeoDataFrame`` is returned.

        ``intersects`` is used as the spatial predicate rather than ``within``
        so that polygon geometries straddling the boundary edge are retained
        rather than silently dropped.  For point geometries the two predicates
        are equivalent.

        Called automatically by ``process`` when a ``country`` kwarg is
        provided. Can also be called manually at the Silver layer for
        targeted boundary clipping.

        Args:
            df: Input DataFrame or GeoDataFrame containing entity data.
                - ``pd.DataFrame``: must contain ``lat_col`` and ``lon_col``.
                - ``gpd.GeoDataFrame``: existing geometry column is used directly;
                ``lat_col`` and ``lon_col`` are ignored.
            boundary: Boundary to filter against. Accepts:

                - ``str`` — ISO 3166-1 alpha-3 country code or country name,
                passed to ``AdminBoundaries.create`` to load the boundary
                from an open source or configured data store at admin level 0.
                - ``gpd.GeoDataFrame`` — pre-loaded boundary geometries,
                dissolved into a single union before filtering.

            lat_col: Name of the latitude column. Used only when ``df`` is a
                plain ``pd.DataFrame``. Defaults to ``'latitude'``.
            lon_col: Name of the longitude column. Used only when ``df`` is a
                plain ``pd.DataFrame``. Defaults to ``'longitude'``.
            **kwargs: Additional keyword arguments forwarded to
                ``AdminBoundaries.create`` when ``boundary`` is a string
                (e.g. ``data_store``, ``source``).

        Returns:
            - ``pd.DataFrame`` if the input was a plain DataFrame, index reset.
            - ``gpd.GeoDataFrame`` if the input was already a GeoDataFrame,
            index reset.

        Raises:
            ValueError: If ``boundary`` is not a ``str`` or ``gpd.GeoDataFrame``.
        """
        from gigaspatial.handlers.boundaries import AdminBoundaries

        # --- resolve boundary ---
        if isinstance(boundary, str):
            boundary_table = AdminBoundaries.create(boundary, admin_level=0, **kwargs)
            boundary_gdf = boundary_table.to_geodataframe()
        elif isinstance(boundary, gpd.GeoDataFrame):
            boundary_gdf = boundary
        else:
            raise ValueError(
                f"Unsupported boundary type: {type(boundary).__name__}. "
                "Expected a country string or GeoDataFrame."
            )

        boundary_union = boundary_gdf.dissolve().geometry.iloc[0]

        # --- resolve input geometry ---
        is_geodataframe = isinstance(df, gpd.GeoDataFrame)
        if is_geodataframe:
            # polygon/multipolygon entities — use existing geometry directly
            gdf = df
        else:
            # point entities — convert temporarily for spatial filter
            gdf = convert_to_geodataframe(df, lat_col=lat_col, lon_col=lon_col)

        # --- spatial filter (intersects handles both points and polygons) ---
        mask = gdf.geometry.intersects(boundary_union)

        removed = (~mask).sum()
        if removed:
            self._log("info", "%d rows removed outside the boundary.", removed)

        result = gdf.loc[mask].reset_index(drop=True)

        # return original type — drop temporary geometry for plain DataFrames
        if not is_geodataframe:
            return df.loc[mask].reset_index(drop=True)
        return result

    def validate_coordinates(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        drop_invalid: bool = True,
    ) -> pd.DataFrame:
        """
        Validate coordinate columns, flagging or dropping invalid rows.

        Three checks are performed independently so the log output
        distinguishes between different classes of data quality issue:

        - **Missing**: ``None`` or ``NaN`` in either coordinate column.
        - **Out-of-range**: latitude outside ``[-90, 90]`` or longitude
          outside ``[-180, 180]``.
        - **Null island**: both coordinates exactly ``0.0``, which is
          unambiguously invalid for real-world infrastructure entities.

        ``pd.to_numeric`` is applied before comparison so the method is
        safe to call on columns that have not yet been through
        ``_coerce_numeric_columns`` (e.g. during manual Silver inspection).

        Args:
            df: Input DataFrame containing coordinate columns.
            lat_col: Name of the latitude column. Defaults to ``'latitude'``.
            lon_col: Name of the longitude column. Defaults to ``'longitude'``.
            drop_invalid: If ``True`` (default), invalid rows are removed and
                a summary is logged. If ``False``, an ``is_valid_coordinate``
                boolean column is appended instead, allowing the caller to
                inspect and handle invalid rows manually before dropping.

        Returns:
            - ``drop_invalid=True``: DataFrame with invalid rows removed,
              index reset.
            - ``drop_invalid=False``: Original DataFrame with an appended
              ``is_valid_coordinate`` boolean column.

        Raises:
            ValueError: If ``lat_col`` or ``lon_col`` are not present in ``df``.

        Example::

            # inspect before dropping
            df = processor.validate_coordinates(df, drop_invalid=False)
            print(df[~df["is_valid_coordinate"]][["latitude", "longitude"]])

            # drop once satisfied
            df = df[df["is_valid_coordinate"]].drop(columns="is_valid_coordinate")
        """
        missing_cols = [c for c in (lat_col, lon_col) if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Coordinate column(s) not found in DataFrame: {missing_cols}. "
                f"Available columns: {df.columns.tolist()}"
            )

        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")

        null_mask = lat.isna() | lon.isna()
        range_mask = (lat < -90) | (lat > 90) | (lon < -180) | (lon > 180)
        null_island_mask = (lat == 0.0) & (lon == 0.0)
        invalid_mask = null_mask | range_mask | null_island_mask
        valid_mask = ~invalid_mask

        if null_mask.any():
            logger.warning("%d rows have missing coordinate values.", null_mask.sum())
        if range_mask.any():
            logger.warning(
                "%d rows have out-of-range coordinate values "
                "(lat outside [-90, 90] or lon outside [-180, 180]).",
                range_mask.sum(),
            )
        if null_island_mask.any():
            logger.warning(
                "%d rows have null island coordinates (0.0, 0.0).",
                null_island_mask.sum(),
            )

        total_invalid = invalid_mask.sum()
        if drop_invalid:
            if total_invalid:
                self._log(
                    "info",
                    "%d of %d rows removed due to invalid coordinates.",
                    total_invalid,
                    len(df),
                )
            return df.loc[valid_mask].reset_index(drop=True)
        else:
            df = df.copy()
            df["is_valid_coordinate"] = valid_mask
            self._log(
                "info",
                "%d of %d rows flagged as invalid coordinates.",
                total_invalid,
                len(df),
            )
            return df

    def deduplicate_by_proximity(
        self,
        df: pd.DataFrame,
        distance_threshold_m: float = 50,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        keep: str = "first",
    ) -> pd.DataFrame:
        """
        Remove near-duplicate rows within a spatial distance threshold.

        Uses a greedy forward scan over a KDTree built in UTM projection:
        each row is kept and all neighbours within ``distance_threshold_m``
        metres are marked as duplicates.  This is significantly faster than
        an all-pairs approach for large datasets and produces deterministic
        results given a fixed row order.

        Designed for cross-source deduplication at the Silver layer where
        the same physical site may appear in multiple source files with
        different identifiers.  For ID-based deduplication at the Gold layer
        use ``EntityTable.merge(deduplicate_by_id=True)`` instead.

        Args:
            df: Input DataFrame containing coordinate columns.
            distance_threshold_m: Maximum distance in metres below which two
                rows are considered duplicates. Defaults to ``50``.
            lat_col: Name of the latitude column. Defaults to ``'latitude'``.
            lon_col: Name of the longitude column. Defaults to ``'longitude'``.
            keep: Which occurrence to retain when duplicates are found.
                ``'first'`` retains the earliest row (default);
                ``'last'`` retains the latest row.

        Returns:
            DataFrame with near-duplicate rows removed, index reset.

        Raises:
            ValueError: If ``distance_threshold_m`` is negative.
            ValueError: If ``keep`` is not ``'first'`` or ``'last'``.
            ValueError: If ``lat_col`` or ``lon_col`` are absent from ``df``.
        """
        if distance_threshold_m < 0:
            raise ValueError("distance_threshold_m must be non-negative.")
        if keep not in ("first", "last"):
            raise ValueError(f"keep must be 'first' or 'last', got '{keep}'.")

        missing_cols = [c for c in (lat_col, lon_col) if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Coordinate column(s) not found in DataFrame: {missing_cols}. "
                f"Available columns: {df.columns.tolist()}"
            )

        if df.empty:
            return df.reset_index(drop=True)

        working_df = (
            df.iloc[::-1].reset_index(drop=True)
            if keep == "last"
            else df.reset_index(drop=True)
        )

        gdf = convert_to_geodataframe(working_df, lat_col=lat_col, lon_col=lon_col)

        from gigaspatial.processing.geo import estimate_utm_crs_with_fallback

        utm_crs = estimate_utm_crs_with_fallback(gdf)
        coords = gdf.to_crs(utm_crs).get_coordinates().to_numpy()

        tree = cKDTree(coords)
        kept_mask = np.ones(len(coords), dtype=bool)
        for i in range(len(coords)):
            if not kept_mask[i]:
                continue
            for j in tree.query_ball_point(coords[i], r=distance_threshold_m):
                if j != i:
                    kept_mask[j] = False

        removed = (~kept_mask).sum()
        if removed:
            self._log(
                "info",
                "%d of %d rows removed as near-duplicates "
                "(distance_threshold_m=%.1f).",
                removed,
                len(df),
                distance_threshold_m,
            )

        result = working_df.loc[kept_mask].reset_index(drop=True)
        if keep == "last":
            result = result.iloc[::-1].reset_index(drop=True)
        return result

    def assign_entity_id(
        self,
        df: pd.DataFrame,
        entity_type: str,
        source_columns: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """
        Assign deterministic UUID3 identifiers to entities in a DataFrame.

        Generates identifiers derived from ``source_columns`` values using
        ``uuid.uuid3`` against a fixed GigaSpatial-scoped namespace UUID,
        ensuring the same combination of source values always produces the
        same ID across runs and environments.

        Rows where any ``source_columns`` value is missing fall back to a
        random ``uuid.uuid4`` so no row is ever left without an identifier.
        If ``source_columns`` is empty or ``None``, all rows receive UUID4.

        Existing IDs are preserved unless ``overwrite=True``.

        Args:
            df: Input DataFrame containing entity data.
            entity_type: Entity type name used to derive the ID column name
                (e.g. ``'cell_tower'`` → ``'cell_tower_id'``).
            source_columns: Column names whose concatenated values (joined
                with ``'|'``) seed the UUID3 hash.  Columns absent from
                ``df`` are skipped with a warning and affected rows fall
                back to UUID4.  Defaults to ``None`` (all UUID4).
            overwrite: If ``True``, regenerate IDs for all rows, including
                those that already have a non-null ID.  Defaults to
                ``False``.

        Returns:
            DataFrame with a populated ``{entity_type}_id`` column.

        Raises:
            ValueError: If ``entity_type`` is empty or blank.

        Example::

            df = processor.assign_entity_id(
                df,
                entity_type="cell_tower",
                source_columns=["cell_tower_id_source", "country_iso"],
            )
        """
        if not entity_type or not entity_type.strip():
            raise ValueError("entity_type must not be empty.")

        df = df.copy()
        id_column = f"{entity_type}_id"
        source_columns = source_columns or []

        if id_column not in df.columns:
            df[id_column] = None

        missing_cols = [c for c in source_columns if c not in df.columns]
        if missing_cols:
            logger.warning(
                "Source columns %s not found in DataFrame. "
                "Affected rows will receive random UUID4 identifiers.",
                missing_cols,
            )
            source_columns = [c for c in source_columns if c in df.columns]

        mask = pd.Series(True, index=df.index) if overwrite else df[id_column].isna()

        if not mask.any():
            logger.debug("All rows already have IDs. Nothing to assign.")
            return df

        def generate_id(row: pd.Series) -> str:
            if source_columns:
                values = [row[col] for col in source_columns]
                if all(v is not None and str(v).strip() != "" for v in values):
                    concat = "|".join(str(v).strip() for v in values)
                    return str(uuid.uuid3(ENTITY_UUID_NAMESPACE, concat))
            return str(uuid.uuid4())

        df.loc[mask, id_column] = df.loc[mask].apply(generate_id, axis=1)

        if source_columns:
            has_all = df.loc[mask, source_columns].notna().all(axis=1)
            deterministic = has_all.sum()
            fallback = (~has_all).sum()
        else:
            deterministic, fallback = 0, mask.sum()

        self._log(
            "info",
            "Assigned IDs for %d rows: %d deterministic (UUID3), %d fallback (UUID4).",
            mask.sum(),
            deterministic,
            fallback,
        )
        return df

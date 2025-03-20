import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from typing import Literal, List, Tuple, Optional
import re

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.handlers.boundaries import AdminBoundaries
from gigaspatial.config import config

LOGGER = config.get_logger(__name__)


def detect_coordinate_columns(
    data, lat_keywords=None, lon_keywords=None, case_sensitive=False
) -> Tuple[str, str]:
    """
    Detect latitude and longitude columns in a DataFrame using keyword matching.

    Parameters:
    ----------
    data : pandas.DataFrame
        DataFrame to search for coordinate columns.
    lat_keywords : list of str, optional
        Keywords for identifying latitude columns. If None, uses default keywords.
    lon_keywords : list of str, optional
        Keywords for identifying longitude columns. If None, uses default keywords.
    case_sensitive : bool, optional
        Whether to perform case-sensitive matching. Default is False.

    Returns:
    -------
    tuple[str, str]
        Names of detected (latitude, longitude) columns.

    Raises:
    ------
    ValueError
        If no unique pair of latitude/longitude columns can be found.
    TypeError
        If input data is not a pandas DataFrame.
    """

    # Default keywords if none provided
    default_lat = [
        "latitude",
        "lat",
        "y",
        "lat_",
        "lat(s)",
        "_lat",
        "ylat",
        "latitude_y",
    ]
    default_lon = [
        "longitude",
        "lon",
        "long",
        "x",
        "lon_",
        "lon(e)",
        "long(e)",
        "_lon",
        "xlon",
        "longitude_x",
    ]

    lat_keywords = lat_keywords or default_lat
    lon_keywords = lon_keywords or default_lon

    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if not data.columns.is_unique:
        raise ValueError("DataFrame contains duplicate column names")

    def create_pattern(keywords):
        """Create regex pattern from keywords."""
        return "|".join(rf"\b{re.escape(keyword)}\b" for keyword in keywords)

    def find_matching_columns(columns, pattern, case_sensitive) -> List:
        """Find columns matching the pattern."""
        flags = 0 if case_sensitive else re.IGNORECASE
        return [col for col in columns if re.search(pattern, col, flags=flags)]

    try:
        # Create patterns
        lat_pattern = create_pattern(lat_keywords)
        lon_pattern = create_pattern(lon_keywords)

        # Find matching columns
        lat_cols = find_matching_columns(data.columns, lat_pattern, case_sensitive)
        lon_cols = find_matching_columns(data.columns, lon_pattern, case_sensitive)

        # Remove any longitude matches from latitude columns and vice versa
        lat_cols = [col for col in lat_cols if col not in lon_cols]
        lon_cols = [col for col in lon_cols if col not in lat_cols]

        # Detailed error messages based on what was found
        if not lat_cols and not lon_cols:
            columns_list = "\n".join(f"- {col}" for col in data.columns)
            raise ValueError(
                f"No latitude or longitude columns found. Available columns are:\n{columns_list}\n"
                f"Consider adding more keywords or checking column names."
            )

        if not lat_cols:
            found_lons = ", ".join(lon_cols)
            raise ValueError(
                f"Found longitude columns ({found_lons}) but no latitude columns. "
                "Check latitude keywords or column names."
            )

        if not lon_cols:
            found_lats = ", ".join(lat_cols)
            raise ValueError(
                f"Found latitude columns ({found_lats}) but no longitude columns. "
                "Check longitude keywords or column names."
            )

        if len(lat_cols) > 1 or len(lon_cols) > 1:
            raise ValueError(
                f"Multiple possible coordinate columns found:\n"
                f"Latitude candidates: {lat_cols}\n"
                f"Longitude candidates: {lon_cols}\n"
                "Please specify more precise keywords."
            )

        return lat_cols[0], lon_cols[0]

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise RuntimeError(f"Error detecting coordinate columns: {str(e)}")


def convert_to_geodataframe(
    data: pd.DataFrame, lat_col: str = None, lon_col: str = None, crs="EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Convert a pandas DataFrame to a GeoDataFrame, either from latitude/longitude columns
    or from a WKT geometry column.

    Parameters:
    ----------
    data : pandas.DataFrame
        Input DataFrame containing either lat/lon columns or a geometry column.
    lat_col : str, optional
        Name of the latitude column. Default is 'lat'.
    lon_col : str, optional
        Name of the longitude column. Default is 'lon'.
    crs : str or pyproj.CRS, optional
        Coordinate Reference System of the geometry data. Default is 'EPSG:4326'.

    Returns:
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the input data with a geometry column.

    Raises:
    ------
    TypeError
        If input is not a pandas DataFrame.
    ValueError
        If required columns are missing or contain invalid data.
    """

    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame")

    # Create a copy to avoid modifying the input
    df = data.copy()

    try:
        if "geometry" not in df.columns:
            # If column names not provided, try to detect them
            if lat_col is None or lon_col is None:
                try:
                    detected_lat, detected_lon = detect_coordinate_columns(df)
                    lat_col = lat_col or detected_lat
                    lon_col = lon_col or detected_lon
                except ValueError as e:
                    if "geometry" not in df.columns:
                        raise ValueError(
                            f"Could not automatically detect coordinate columns and no "
                            f"'geometry' column found. Error: {str(e)}"
                        )

            # Validate latitude/longitude columns exist
            if lat_col not in df.columns or lon_col not in df.columns:
                raise ValueError(
                    f"Could not find columns: {lat_col} and/or {lon_col} in the DataFrame"
                )

            # Check for missing values
            if df[lat_col].isna().any() or df[lon_col].isna().any():
                raise ValueError(
                    f"Missing values found in {lat_col} and/or {lon_col} columns"
                )

            # Validate latitude/longitude values
            if not (
                df[lat_col].between(-90, 90).all()
                and df[lon_col].between(-180, 180).all()
            ):
                raise ValueError(
                    f"Invalid values found: latitude must be between -90 and 90, "
                    f"longitude must be between -180 and 180"
                )

            # Create geometry from lat/lon
            geometry = gpd.points_from_xy(x=df[lon_col], y=df[lat_col], crs=crs)

        else:
            # Handle WKT geometry column
            try:
                # Check for missing values
                if df["geometry"].isna().any():
                    raise ValueError("Missing values found in geometry column")

                # Convert WKT strings to geometry objects
                geometry = df["geometry"].apply(wkt.loads)
                df = df.drop("geometry", axis=1)  # Remove WKT column

            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid geometry format in 'geometry' column: {str(e)}"
                )

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)

        return gdf

    except Exception as e:
        raise RuntimeError(f"Error converting to GeoDataFrame: {str(e)}")


def buffer_geodataframe(
    gdf: gpd.GeoDataFrame,
    buffer_distance_meters: float,
    cap_style: Literal["round", "square", "flat"] = "round",
    copy=True,
) -> gpd.GeoDataFrame:
    """
    Buffers a GeoDataFrame with a given buffer distance in meters.

    Parameters:
    - gdf : geopandas.GeoDataFrame
        The GeoDataFrame to be buffered.
    - buffer_distance_meters : float
        The buffer distance in meters.
    - cap_style : str, optional
        The style of caps. round, flat, square. Default is round.

    Returns:
    - geopandas.GeoDataFrame
        The buffered GeoDataFrame.
    """

    # Input validation
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("Input must be a GeoDataFrame")

    if not isinstance(buffer_distance_meters, (float, int)):
        raise TypeError("Buffer distance must be a number")

    if cap_style not in ["round", "square", "flat"]:
        raise ValueError("cap_style must be round, flat or square.")

    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame must have a defined CRS")

    # Create a copy if requested
    gdf_work = gdf.copy() if copy else gdf

    # Store input CRS
    input_crs = gdf_work.crs

    try:
        # Create a custom UTM CRS based on the calculated UTM zone
        utm_crs = gdf_work.estimate_utm_crs()

        # Transform to UTM, create buffer, and transform back
        gdf_work = gdf_work.to_crs(utm_crs)
        gdf_work["geometry"] = gdf_work["geometry"].buffer(
            buffer_distance_meters, cap_style=cap_style
        )
        gdf_work = gdf_work.to_crs(input_crs)

        return gdf_work

    except Exception as e:
        raise RuntimeError(f"Error during buffering operation: {str(e)}")


def add_spatial_jitter(
    df: pd.DataFrame,
    columns: List[str] = ["latitude", "longitude"],
    amount: float = 0.0001,
    seed=None,
    copy=True,
) -> pd.DataFrame:
    """
    Add random jitter to duplicated geographic coordinates to create slight separation
    between overlapping points.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame containing geographic coordinates.
    columns : list of str, optional
        Column names containing coordinates to jitter. Default is ['latitude', 'longitude'].
    amount : float or dict, optional
        Amount of jitter to add. If float, same amount used for all columns.
        If dict, specify amount per column, e.g., {'lat': 0.0001, 'lon': 0.0002}.
        Default is 0.0001 (approximately 11 meters at the equator).
    seed : int, optional
        Random seed for reproducibility. Default is None.
    copy : bool, optional
        Whether to create a copy of the input DataFrame. Default is True.

    Returns:
    -------
    pandas.DataFrame
        DataFrame with jittered coordinates for previously duplicated points.

    Raises:
    ------
    ValueError
        If columns don't exist or jitter amount is invalid.
    TypeError
        If input types are incorrect.
    """

    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if not all(col in df.columns for col in columns):
        raise ValueError(f"Not all columns {columns} found in DataFrame")

    # Handle jitter amounts
    if isinstance(amount, (int, float)):
        if amount <= 0:
            raise ValueError("Jitter amount must be positive")
        jitter_amounts = {col: amount for col in columns}
    elif isinstance(amount, dict):
        if not all(col in amount for col in columns):
            raise ValueError("Must specify jitter amount for each column")
        if not all(amt > 0 for amt in amount.values()):
            raise ValueError("All jitter amounts must be positive")
        jitter_amounts = amount
    else:
        raise TypeError("amount must be a number or dictionary")

    # Create copy if requested
    df_work = df.copy() if copy else df

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    try:
        # Find duplicated coordinates
        duplicate_mask = df_work.duplicated(subset=columns, keep=False)
        n_duplicates = duplicate_mask.sum()

        if n_duplicates > 0:
            # Add jitter to each column separately
            for col in columns:
                jitter = np.random.uniform(
                    low=-jitter_amounts[col],
                    high=jitter_amounts[col],
                    size=n_duplicates,
                )
                df_work.loc[duplicate_mask, col] += jitter

            # Validate results (ensure no remaining duplicates)
            if df_work.duplicated(subset=columns, keep=False).any():
                # If duplicates remain, recursively add more jitter
                df_work = add_spatial_jitter(
                    df_work,
                    columns=columns,
                    amount={col: amt * 2 for col, amt in jitter_amounts.items()},
                    seed=seed,
                    copy=False,
                )

        return df_work

    except Exception as e:
        raise RuntimeError(f"Error during jittering operation: {str(e)}")


def simplify_geometries(
    gdf: gpd.GeoDataFrame,
    tolerance: float = 0.01,
    preserve_topology: bool = True,
    geometry_column: str = "geometry",
) -> gpd.GeoDataFrame:
    """
    Simplify geometries in a GeoDataFrame to reduce file size and improve visualization performance.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing geometries to simplify.
    tolerance : float, optional
        Tolerance for simplification. Larger values simplify more but reduce detail (default is 0.01).
    preserve_topology : bool, optional
        Whether to preserve topology while simplifying. Preserving topology prevents invalid geometries (default is True).
    geometry_column : str, optional
        Name of the column containing geometries (default is "geometry").

    Returns
    -------
    geopandas.GeoDataFrame
        A new GeoDataFrame with simplified geometries.

    Raises
    ------
    ValueError
        If the specified geometry column does not exist or contains invalid geometries.
    TypeError
        If the geometry column does not contain valid geometries.

    Examples
    --------
    Simplify geometries in a GeoDataFrame:
    >>> simplified_gdf = simplify_geometries(gdf, tolerance=0.05)
    """

    # Check if the specified geometry column exists
    if geometry_column not in gdf.columns:
        raise ValueError(
            f"Geometry column '{geometry_column}' not found in the GeoDataFrame."
        )

    # Check if the specified column contains geometries
    if not gpd.GeoSeries(gdf[geometry_column]).is_valid.all():
        raise TypeError(
            f"Geometry column '{geometry_column}' contains invalid geometries."
        )

    # Simplify geometries (non-destructive)
    gdf_simplified = gdf.copy()
    gdf_simplified[geometry_column] = gdf_simplified[geometry_column].simplify(
        tolerance=tolerance, preserve_topology=preserve_topology
    )

    return gdf_simplified


def overlay_aggregate_polygon_data(
    base_gdf: gpd.GeoDataFrame,
    overlay_gdf: gpd.GeoDataFrame,
    overlay_columns: List[str],
    base_gdf_key: str,
    overlay_gdf_key: str = None,
    agg_func: str = "sum",
) -> gpd.GeoDataFrame:
    """
    Overlay polygon data and aggregate values over the base GeoDataFrame.

    Parameters:
    ----------
    base_gdf : geopandas.GeoDataFrame
        GeoDataFrame representing the base geometries.
    overlay_gdf : geopandas.GeoDataFrame
        GeoDataFrame with polygon geometries to overlay and aggregate.
    overlay_columns : list of str
        Columns in `overlay_gdf` to aggregate based on overlapping areas.
    base_gdf_key : str
        Column in `base_gdf` to use as the key for aggregation.
    overlay_gdf_key : str, optional
        Column in `overlay_gdf` to use as the index for merging. Defaults to the overlay GeoDataFrame's index.
    agg_func : str, callable, or dict, default="sum"
        Aggregation function or dictionary of column-specific aggregation functions.
        Examples: "sum", "mean", "max", or {"column1": "mean", "column2": "sum"}.

    Returns:
    -------
    geopandas.GeoDataFrame
        Base GeoDataFrame with aggregated values from the overlay.

    Raises:
    ------
    ValueError
        If the overlay GeoDataFrame has duplicate index values or missing columns.
    RuntimeError
        If any geometry operations fail.
    """

    # Validate inputs
    if not isinstance(base_gdf, gpd.GeoDataFrame) or not isinstance(
        overlay_gdf, gpd.GeoDataFrame
    ):
        raise TypeError("Both base_gdf and overlay_gdf must be GeoDataFrames.")
    if not set(overlay_columns).issubset(overlay_gdf.columns):
        missing_cols = set(overlay_columns) - set(overlay_gdf.columns)
        raise ValueError(f"Missing columns in overlay_gdf: {missing_cols}")
    if overlay_gdf.index.duplicated().any():
        raise ValueError("Overlay GeoDataFrame contains duplicate index values.")
    if base_gdf_key not in base_gdf.columns:
        raise ValueError(f"base_gdf_key '{base_gdf_key}' not found in base_gdf.")

    # Set index name for overlay_gdf
    index_name = (
        overlay_gdf_key if overlay_gdf_key else overlay_gdf.index.name or "index"
    )

    # Ensure geometries are valid
    overlay_gdf = overlay_gdf.copy()

    # Perform the spatial overlay
    try:
        gdf_overlayed = (
            overlay_gdf[overlay_columns + ["geometry"]]
            .reset_index()
            .overlay(base_gdf, how="intersection")
            .set_index(index_name)
        )
    except Exception as e:
        raise RuntimeError(f"Error during spatial overlay: {e}")

    # Compute UTM CRS for accurate area calculations
    overlay_utm_crs = overlay_gdf.estimate_utm_crs()

    # Calculate pixel areas in overlay_gdf
    overlay_gdf_utm = overlay_gdf.to_crs(overlay_utm_crs)
    pixel_area = overlay_gdf_utm.area
    pixel_area.name = "pixel_area"

    # Add pixel area to the overlayed data
    gdf_overlayed = gdf_overlayed.join(pixel_area, how="left")

    # Compute overlap fraction
    gdf_overlayed_utm = gdf_overlayed.to_crs(overlay_utm_crs)
    gdf_overlayed["overlap_fraction"] = (
        gdf_overlayed_utm.area / gdf_overlayed["pixel_area"]
    )

    # Adjust overlay column values by overlap fraction
    gdf_overlayed[overlay_columns] = gdf_overlayed[overlay_columns].mul(
        gdf_overlayed["overlap_fraction"], axis=0
    )

    # Validate and apply aggregation
    if isinstance(agg_func, str) or callable(agg_func):
        aggregation_dict = {col: agg_func for col in overlay_columns}
    elif isinstance(agg_func, dict):
        aggregation_dict = agg_func
        for col in overlay_columns:
            if col not in aggregation_dict:
                raise ValueError(f"Missing aggregation function for column: {col}")
    else:
        raise ValueError(
            "Invalid aggregation function. Must be a string, callable, or dictionary."
        )

    # Aggregate data
    aggregated = (
        gdf_overlayed[overlay_columns + [base_gdf_key]]
        .groupby(base_gdf_key)
        .agg(aggregation_dict)
        .reset_index()
    )

    # Aggregate overlay columns by the base GeoDataFrame key
    # aggregated = (
    #    gdf_overlayed.groupby(base_gdf_key)[overlay_columns].sum().reset_index()
    # )

    # Merge aggregated values back to the base GeoDataFrame
    result = base_gdf.merge(aggregated, on=base_gdf_key, how="left")

    return result


def count_points_within_polygons(base_gdf, count_gdf, base_gdf_key):
    """
    Counts the number of points from `count_gdf` that fall within each polygon in `base_gdf`.

    Parameters:
    ----------
    base_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing polygon geometries to count points within.
    count_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing point geometries to be counted.
    base_gdf_key : str
        Column name in `base_gdf` to use as the key for grouping and merging.

    Returns:
    -------
    geopandas.GeoDataFrame
        The `base_gdf` with an additional column containing the count of points within each polygon.

    Raises:
    ------
    ValueError
        If `base_gdf_key` is missing in `base_gdf`.
    """
    # Validate inputs
    if base_gdf_key not in base_gdf.columns:
        raise ValueError(f"Key column '{base_gdf_key}' not found in `base_gdf`.")

    # Ensure clean index for the join
    count_gdf = count_gdf.reset_index()

    # Spatial join: Find points intersecting polygons
    joined_gdf = gpd.sjoin(
        base_gdf, count_gdf[["geometry"]], how="left", predicate="intersects"
    )

    # Count points grouped by the base_gdf_key
    point_counts = joined_gdf.groupby(base_gdf_key)["index_right"].count()
    point_counts.name = "point_count"

    # Merge point counts back to the base GeoDataFrame
    result_gdf = base_gdf.merge(point_counts, on=base_gdf_key, how="left")

    # Fill NaN counts with 0 for polygons with no points
    result_gdf["point_count"] = result_gdf["point_count"].fillna(0).astype(int)

    return result_gdf


def map_points_within_polygons(base_points_gdf, polygon_gdf):
    """
    Maps whether each point in `base_points_gdf` is within any polygon in `polygon_gdf`.

    Parameters:
    ----------
    base_points_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing point geometries to check.
    polygon_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing polygon geometries.

    Returns:
    -------
    geopandas.GeoDataFrame
        The `base_points_gdf` with an additional column `is_within` (True/False).

    Raises:
    ------
    ValueError
        If the geometries in either GeoDataFrame are invalid or not of the expected type.
    """
    # Validate input GeoDataFrames
    if not all(base_points_gdf.geometry.geom_type == "Point"):
        raise ValueError("`base_points_gdf` must contain only Point geometries.")
    if not all(polygon_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])):
        raise ValueError(
            "`polygon_gdf` must contain only Polygon or MultiPolygon geometries."
        )

    if not base_points_gdf.crs == polygon_gdf.crs:
        raise ValueError("CRS of `base_points_gdf` and `polygon_gdf` must match.")

    # Perform spatial join to check if points fall within any polygon
    joined_gdf = gpd.sjoin(
        base_points_gdf, polygon_gdf[["geometry"]], how="left", predicate="within"
    )

    # Add `is_within` column to base_points_gdf
    base_points_gdf["is_within"] = base_points_gdf.index.isin(
        set(joined_gdf.index[~joined_gdf.index_right.isna()])
    )

    return base_points_gdf

def calculate_distance(lat1, lon1, lat2, lon2, R=6371e3):
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance

def annotate_with_admin_regions(
    gdf: gpd.GeoDataFrame, country_code: str, data_store: Optional[DataStore] = None, admin_id_column_suffix = "_giga"
) -> gpd.GeoDataFrame:
    """
    Annotate a GeoDataFrame with administrative region information.
    
    Performs a spatial join between the input points and administrative boundaries
    at levels 1 and 2, resolving conflicts when points intersect multiple admin regions.
    
    Args:
        gdf: GeoDataFrame containing points to annotate
        country_code: Country code for administrative boundaries
        data_store: Optional DataStore for loading admin boundary data
        
    Returns:
        GeoDataFrame with added administrative region columns
    """
    
    if not isinstance(gdf, gpd.GeoDataFrame):
       raise TypeError("gdf must be a GeoDataFrame")
       
    if gdf.empty:
        LOGGER.warning("Empty GeoDataFrame provided, returning as-is")
        return gdf

    # read country admin data
    admin1_data = AdminBoundaries.create(
        country_code=country_code, admin_level=1, data_store=data_store
    ).to_geodataframe()

    admin1_data.rename(columns={"id": f"admin1_id{admin_id_column_suffix}", "name": "admin1"}, inplace=True)
    admin1_data.drop(columns=["name_en", "parent_id", "country_code"], inplace=True)

    admin2_data = AdminBoundaries.create(
        country_code=country_code, admin_level=2, data_store=data_store
    ).to_geodataframe()

    admin2_data.rename(
        columns={
            "id": f"admin2_id{admin_id_column_suffix}",
            "parent_id": f"admin1_id{admin_id_column_suffix}",
            "name": "admin2",
        },
        inplace=True,
    )
    admin2_data.drop(columns=["name_en", "country_code"], inplace=True)

    # Join dataframes based on 'admin1_id_giga'
    admin_data = admin2_data.merge(
        admin1_data[[f"admin1_id{admin_id_column_suffix}", "admin1", "geometry"]],
        left_on=f"admin1_id{admin_id_column_suffix}",
        right_on=f"admin1_id{admin_id_column_suffix}",
        how="outer",
    )

    admin_data["geometry"] = admin_data.apply(
        lambda x: x.geometry_x if x.geometry_x else x.geometry_y, axis=1
    )

    admin_data = gpd.GeoDataFrame(
        admin_data.drop(columns=["geometry_x", "geometry_y"]),
        geometry="geometry",
        crs=4326,
    )

    if gdf.crs is None:
       LOGGER.warning("Input GeoDataFrame has no CRS, assuming EPSG:4326")
       gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs != "EPSG:4326":
        LOGGER.info(f"Reprojecting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs(epsg=4326)

    # spatial join gdf to admins
    gdf_w_admins = gdf.copy().sjoin(
        admin_data,
        how="left",
        predicate="intersects",
    )

    # Check for duplicates caused by points intersecting multiple polygons
    if len(gdf_w_admins) != len(gdf):
        LOGGER.warning(
            "Some points intersect multiple administrative boundaries. Resolving conflicts..."
        )

        # Group by original index and select the closest admin area for ties
        gdf_w_admins["distance"] = gdf_w_admins.apply(
            lambda row: row.geometry.distance(
                admin_data.loc[row.index_right, "geometry"].centroid
            ),
            axis=1,
        )

        # For points with multiple matches, keep the closest polygon
        gdf_w_admins = gdf_w_admins.loc[
            gdf_w_admins.groupby(gdf.index)["distance"].idxmin()
        ].drop(columns="distance")

    # Drop unnecessary columns and reset the index
    gdf_w_admins = gdf_w_admins.drop(columns="index_right").reset_index(drop=True)

    return gdf_w_admins
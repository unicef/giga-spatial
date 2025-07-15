import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import base
from typing import Literal, List, Tuple, Optional, Union, Dict
import re

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.config import config

LOGGER = config.get_logger("GigaSpatialProcessing")


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

            # Create geometry from lat/lon
            geometry = gpd.points_from_xy(x=df[lon_col], y=df[lat_col])

        else:
            # Check if geometry column already contains valid geometries
            if df["geometry"].apply(lambda x: isinstance(x, base.BaseGeometry)).all():
                geometry = df["geometry"]
            elif df["geometry"].apply(lambda x: isinstance(x, str)).all():
                # Convert WKT strings to geometry objects
                geometry = df["geometry"].apply(wkt.loads)
            else:
                raise ValueError(
                    "Invalid geometry format: contains mixed or unsupported types"
                )

        # drop the WKT column if conversion was done
        if (
            "geometry" in df.columns
            and not df["geometry"]
            .apply(lambda x: isinstance(x, base.BaseGeometry))
            .all()
        ):
            df = df.drop(columns=["geometry"])

        return gpd.GeoDataFrame(df, geometry=geometry, crs=crs)

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
        try:
            utm_crs = gdf_work.estimate_utm_crs()
        except Exception as e:
            LOGGER.warning(
                f"Warning: UTM CRS estimation failed, using Web Mercator. Error: {e}"
            )
            utm_crs = "EPSG:3857"  # Fallback to Web Mercator

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


def get_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate the centroids of a (Multi)Polygon GeoDataFrame.

    Parameters:
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing (Multi)Polygon geometries.

    Returns:
    -------
    geopandas.GeoDataFrame
        A new GeoDataFrame with Point geometries representing the centroids.

    Raises:
    ------
    ValueError
        If the input GeoDataFrame does not contain (Multi)Polygon geometries.
    """
    # Validate input geometries
    if not all(gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])):
        raise ValueError(
            "Input GeoDataFrame must contain only Polygon or MultiPolygon geometries."
        )

    # Calculate centroids
    centroids = gdf.copy()
    centroids["geometry"] = centroids.geometry.centroid

    return centroids


def add_area_in_meters(
    gdf: gpd.GeoDataFrame, area_column_name: str = "area_in_meters"
) -> gpd.GeoDataFrame:
    """
    Calculate the area of (Multi)Polygon geometries in square meters and add it as a new column.

    Parameters:
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing (Multi)Polygon geometries.
    area_column_name : str, optional
        Name of the new column to store the area values. Default is "area_m2".

    Returns:
    -------
    geopandas.GeoDataFrame
        The input GeoDataFrame with an additional column for the area in square meters.

    Raises:
    ------
    ValueError
        If the input GeoDataFrame does not contain (Multi)Polygon geometries.
    """
    # Validate input geometries
    if not all(gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])):
        raise ValueError(
            "Input GeoDataFrame must contain only Polygon or MultiPolygon geometries."
        )

    # Create a copy of the GeoDataFrame to avoid modifying the original
    gdf_with_area = gdf.copy()

    # Calculate the UTM CRS for accurate area calculation
    try:
        utm_crs = gdf_with_area.estimate_utm_crs()
    except Exception as e:
        LOGGER.warning(
            f"Warning: UTM CRS estimation failed, using Web Mercator. Error: {e}"
        )
        utm_crs = "EPSG:3857"  # Fallback to Web Mercator

    # Transform to UTM CRS and calculate the area in square meters
    gdf_with_area[area_column_name] = gdf_with_area.to_crs(utm_crs).geometry.area

    return gdf_with_area


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


def aggregate_points_to_zones(
    points: Union[pd.DataFrame, gpd.GeoDataFrame],
    zones: gpd.GeoDataFrame,
    value_columns: Optional[Union[str, List[str]]] = None,
    aggregation: Union[str, Dict[str, str]] = "count",
    point_zone_predicate: str = "within",
    zone_id_column: str = "zone_id",
    output_suffix: str = "",
    drop_geometry: bool = False,
) -> gpd.GeoDataFrame:
    """
    Aggregate point data to zones with flexible aggregation methods.

    Args:
        points (Union[pd.DataFrame, gpd.GeoDataFrame]): Point data to aggregate
        zones (gpd.GeoDataFrame): Zones to aggregate points to
        value_columns (Optional[Union[str, List[str]]]): Column(s) containing values to aggregate
            If None, only counts will be performed.
        aggregation (Union[str, Dict[str, str]]): Aggregation method(s) to use:
            - Single string: Use same method for all columns ("count", "mean", "sum", "min", "max")
            - Dict: Map column names to aggregation methods
        point_zone_predicate (str): Spatial predicate for point-to-zone relationship
            Options: "within", "intersects", "contains"
        zone_id_column (str): Column in zones containing zone identifiers
        output_suffix (str): Suffix to add to output column names
        drop_geometry (bool): Whether to drop the geometry column from output

    Returns:
        gpd.GeoDataFrame: Zones with aggregated point values

    Example:
        >>> poi_counts = aggregate_points_to_zones(pois, zones, aggregation="count")
        >>> poi_value_mean = aggregate_points_to_zones(
        ...     pois, zones, value_columns="score", aggregation="mean"
        ... )
        >>> poi_multiple = aggregate_points_to_zones(
        ...     pois, zones,
        ...     value_columns=["score", "visits"],
        ...     aggregation={"score": "mean", "visits": "sum"}
        ... )
    """
    # Input validation
    if not isinstance(zones, gpd.GeoDataFrame):
        raise TypeError("zones must be a GeoDataFrame")

    if zone_id_column not in zones.columns:
        raise ValueError(f"Zone ID column '{zone_id_column}' not found in zones")

    # Convert points to GeoDataFrame if necessary
    if not isinstance(points, gpd.GeoDataFrame):
        points_gdf = convert_to_geodataframe(points)
    else:
        points_gdf = points.copy()

    # Ensure CRS match
    if points_gdf.crs != zones.crs:
        points_gdf = points_gdf.to_crs(zones.crs)

    # Handle value columns
    if value_columns is not None:
        if isinstance(value_columns, str):
            value_columns = [value_columns]

        # Validate that all value columns exist
        missing_cols = [col for col in value_columns if col not in points_gdf.columns]
        if missing_cols:
            raise ValueError(f"Value columns not found in points data: {missing_cols}")

    # Handle aggregation method
    agg_funcs = {}

    if isinstance(aggregation, str):
        if aggregation == "count":
            # Special case for count (doesn't need value columns)
            agg_funcs["__count"] = "count"
        elif value_columns is not None:
            # Apply the same aggregation to all value columns
            agg_funcs = {col: aggregation for col in value_columns}
        else:
            raise ValueError(
                "Value columns must be specified for aggregation methods other than 'count'"
            )
    elif isinstance(aggregation, dict):
        # Validate dictionary keys
        if value_columns is None:
            raise ValueError(
                "Value columns must be specified when using a dictionary of aggregation methods"
            )

        missing_aggs = [col for col in value_columns if col not in aggregation]
        extra_aggs = [col for col in aggregation if col not in value_columns]

        if missing_aggs:
            raise ValueError(f"Missing aggregation methods for columns: {missing_aggs}")
        if extra_aggs:
            raise ValueError(
                f"Aggregation methods specified for non-existent columns: {extra_aggs}"
            )

        agg_funcs = aggregation
    else:
        raise TypeError("aggregation must be a string or dictionary")

    # Create a copy of the zones
    result = zones.copy()

    # Spatial join
    joined = gpd.sjoin(points_gdf, zones, how="inner", predicate=point_zone_predicate)

    # Perform aggregation
    if "geometry" in joined.columns and not all(
        value == "count" for value in agg_funcs.values()
    ):
        # Drop geometry for non-count aggregations to avoid errors
        joined = joined.drop(columns=["geometry"])

    if "__count" in agg_funcs:
        # Count points per zone
        counts = (
            joined.groupby(zone_id_column)
            .size()
            .reset_index(name=f"point_count{output_suffix}")
        )
        result = result.merge(counts, on=zone_id_column, how="left")
        result[f"point_count{output_suffix}"] = (
            result[f"point_count{output_suffix}"].fillna(0).astype(int)
        )
    else:
        # Aggregate values
        aggregated = joined.groupby(zone_id_column).agg(agg_funcs).reset_index()

        # Rename columns to include aggregation method
        if len(value_columns) > 0:
            # Handle MultiIndex columns from pandas aggregation
            if isinstance(aggregated.columns, pd.MultiIndex):
                aggregated.columns = [
                    (
                        f"{col[0]}_{col[1]}{output_suffix}"
                        if col[0] != zone_id_column
                        else zone_id_column
                    )
                    for col in aggregated.columns
                ]

            # Merge back to zones
            result = result.merge(aggregated, on=zone_id_column, how="left")

            # Fill NaN values with zeros
            for col in result.columns:
                if (
                    col != zone_id_column
                    and col != "geometry"
                    and pd.api.types.is_numeric_dtype(result[col])
                ):
                    result[col] = result[col].fillna(0)

    if drop_geometry:
        result = result.drop(columns=["geometry"])
        return result

    return result


def annotate_with_admin_regions(
    gdf: gpd.GeoDataFrame,
    country_code: str,
    data_store: Optional[DataStore] = None,
    admin_id_column_suffix="_giga",
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
    from gigaspatial.handlers.boundaries import AdminBoundaries

    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("gdf must be a GeoDataFrame")

    if gdf.empty:
        LOGGER.warning("Empty GeoDataFrame provided, returning as-is")
        return gdf

    # read country admin data
    admin1_data = AdminBoundaries.create(
        country_code=country_code, admin_level=1, data_store=data_store
    ).to_geodataframe()

    admin1_data.rename(
        columns={"id": f"admin1_id{admin_id_column_suffix}", "name": "admin1"},
        inplace=True,
    )
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

    admin_data["admin2"].fillna("Unknown", inplace=True)
    admin_data[f"admin2_id{admin_id_column_suffix}"] = admin_data[
        f"admin2_id{admin_id_column_suffix}"
    ].replace({np.nan: None})

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


def aggregate_polygons_to_zones(
    polygons: Union[pd.DataFrame, gpd.GeoDataFrame],
    zones: gpd.GeoDataFrame,
    value_columns: Union[str, List[str]],
    aggregation: Union[str, Dict[str, str]] = "sum",
    predicate: Literal["intersects", "within", "fractional"] = "intersects",
    zone_id_column: str = "zone_id",
    output_suffix: str = "",
    drop_geometry: bool = False,
) -> gpd.GeoDataFrame:
    """
    Aggregates polygon data to zones based on a specified spatial relationship.

    This function performs a spatial join between polygons and zones and then
    aggregates values from the polygons to their corresponding zones. The aggregation
    method depends on the `predicate` parameter, which determines the nature of the
    spatial relationship.

    Args:
        polygons (Union[pd.DataFrame, gpd.GeoDataFrame]):
            Polygon data to aggregate. Must be a GeoDataFrame or convertible to one.
        zones (gpd.GeoDataFrame):
            The target zones to which the polygon data will be aggregated.
        value_columns (Union[str, List[str]]):
            The column(s) in `polygons` containing the numeric values to aggregate.
        aggregation (Union[str, Dict[str, str]], optional):
            The aggregation method(s) to use. Can be a single string (e.g., "sum",
            "mean", "max") to apply the same method to all columns, or a dictionary
            mapping column names to aggregation methods (e.g., `{'population': 'sum'}`).
            Defaults to "sum".
        predicate (Literal["intersects", "within", "fractional"], optional):
            The spatial relationship to use for aggregation:
            - "intersects": Aggregates values for any polygon that intersects a zone.
            - "within": Aggregates values for polygons entirely contained within a zone.
            - "fractional": Performs area-weighted aggregation. The value of a polygon
              is distributed proportionally to the area of its overlap with each zone.
              This requires calculating a UTM CRS for accurate area measurements.
            Defaults to "intersects".
        zone_id_column (str, optional):
            The name of the column in `zones` that contains the unique zone identifiers.
            Defaults to "zone_id".
        output_suffix (str, optional):
            A suffix to add to the names of the new aggregated columns in the output
            GeoDataFrame. Defaults to "".
        drop_geometry (bool, optional):
            If True, the geometry column will be dropped from the output GeoDataFrame.
            Defaults to False.

    Returns:
        gpd.GeoDataFrame:
            The `zones` GeoDataFrame with new columns containing the aggregated values.
            Zones with no intersecting or contained polygons will have `0` values.

    Raises:
        TypeError: If `zones` is not a GeoDataFrame or `polygons` cannot be converted.
        ValueError: If `zone_id_column` or any `value_columns` are not found, or
                    if the geometry types in `polygons` are not polygons.
        RuntimeError: If an error occurs during the area-weighted aggregation process.

    Example:
        >>> import geopandas as gpd
        >>> # Assuming 'landuse_polygons' and 'grid_zones' are GeoDataFrames
        >>> # Aggregate total population within each grid zone using area-weighting
        >>> pop_by_zone = aggregate_polygons_to_zones(
        ...     landuse_polygons,
        ...     grid_zones,
        ...     value_columns="population",
        ...     predicate="fractional",
        ...     aggregation="sum",
        ...     output_suffix="_pop"
        ... )
        >>> # Aggregate the count of landuse parcels intersecting each zone
        >>> count_by_zone = aggregate_polygons_to_zones(
        ...     landuse_polygons,
        ...     grid_zones,
        ...     value_columns="parcel_id",
        ...     predicate="intersects",
        ...     aggregation="count"
        ... )
    """
    # Input validation
    if not isinstance(zones, gpd.GeoDataFrame):
        raise TypeError("zones must be a GeoDataFrame")

    if zones.empty:
        raise ValueError("zones GeoDataFrame is empty")

    if zone_id_column not in zones.columns:
        raise ValueError(f"Zone ID column '{zone_id_column}' not found in zones")

    if predicate not in ["intersects", "within", "fractional"]:
        raise ValueError(
            f"Unsupported predicate: {predicate}. Predicate can be one of `intersects`, `within`, `fractional`"
        )

    # Convert polygons to GeoDataFrame if necessary
    if not isinstance(polygons, gpd.GeoDataFrame):
        try:
            polygons_gdf = convert_to_geodataframe(polygons)
        except Exception as e:
            raise TypeError(
                f"polygons must be a GeoDataFrame or convertible to one: {e}"
            )
    else:
        polygons_gdf = polygons.copy()

    if polygons_gdf.empty:
        LOGGER.warning("Empty polygons GeoDataFrame provided")
        return zones

    # Validate geometry types
    non_polygon_geoms = [
        geom_type
        for geom_type in polygons_gdf.geometry.geom_type.unique()
        if geom_type not in ["Polygon", "MultiPolygon"]
    ]
    if non_polygon_geoms:
        raise ValueError(
            f"Input contains non-polygon geometries: {non_polygon_geoms}. "
            "Use aggregate_points_to_zones for point data."
        )

    # Process value columns
    if isinstance(value_columns, str):
        value_columns = [value_columns]

    # Validate that all value columns exist
    missing_cols = [col for col in value_columns if col not in polygons_gdf.columns]
    if missing_cols:
        raise ValueError(f"Value columns not found in polygons data: {missing_cols}")

    # Ensure CRS match
    if polygons_gdf.crs != zones.crs:
        polygons_gdf = polygons_gdf.to_crs(zones.crs)

    # Handle aggregation method
    agg_funcs = _process_aggregation_methods(aggregation, value_columns)

    # Prepare minimal zones for spatial operations (only zone_id_column and geometry)
    minimal_zones = zones[[zone_id_column, "geometry"]].copy()

    if predicate == "fractional":
        aggregated_data = _fractional_aggregation(
            polygons_gdf, minimal_zones, value_columns, agg_funcs, zone_id_column
        )
    else:
        aggregated_data = _simple_aggregation(
            polygons_gdf,
            minimal_zones,
            value_columns,
            agg_funcs,
            zone_id_column,
            predicate,
        )

    # Merge aggregated results back to complete zones data
    result = zones.merge(
        aggregated_data[[col for col in aggregated_data.columns if col != "geometry"]],
        on=zone_id_column,
        how="left",
    )

    # Fill NaN values with zeros for the newly aggregated columns only
    aggregated_cols = [col for col in result.columns if col not in zones.columns]
    for col in aggregated_cols:
        if pd.api.types.is_numeric_dtype(result[col]):
            result[col] = result[col].fillna(0)

    # Apply output suffix consistently to result columns only
    if output_suffix:
        rename_dict = {col: f"{col}{output_suffix}" for col in aggregated_cols}
        result = result.rename(columns=rename_dict)

    if drop_geometry:
        result = result.drop(columns=["geometry"])

    return result


def _process_aggregation_methods(aggregation, value_columns):
    """Process and validate aggregation methods"""
    if isinstance(aggregation, str):
        return {col: aggregation for col in value_columns}
    elif isinstance(aggregation, dict):
        # Validate dictionary keys
        missing_aggs = [col for col in value_columns if col not in aggregation]
        extra_aggs = [col for col in aggregation if col not in value_columns]

        if missing_aggs:
            raise ValueError(f"Missing aggregation methods for columns: {missing_aggs}")
        if extra_aggs:
            raise ValueError(
                f"Aggregation methods specified for non-existent columns: {extra_aggs}"
            )

        return aggregation
    else:
        raise TypeError("aggregation must be a string or dictionary")


def _fractional_aggregation(
    polygons_gdf, zones, value_columns, agg_funcs, zone_id_column
):
    """Perform area-weighted (fractional) aggregation"""
    try:
        # Compute UTM CRS for accurate area calculations
        try:
            overlay_utm_crs = polygons_gdf.estimate_utm_crs()
        except Exception as e:
            LOGGER.warning(f"UTM CRS estimation failed, using Web Mercator. Error: {e}")
            overlay_utm_crs = "EPSG:3857"  # Fallback to Web Mercator

        # Prepare polygons for overlay - only necessary columns
        polygons_utm = polygons_gdf.to_crs(overlay_utm_crs)
        polygons_utm["orig_area"] = polygons_utm.area

        # Keep only necessary columns
        overlay_cols = value_columns + ["geometry", "orig_area"]
        overlay_gdf = polygons_utm[overlay_cols].copy()

        # Prepare zones for overlay
        zones_utm = zones.to_crs(overlay_utm_crs)

        # Perform the spatial overlay
        gdf_overlayed = gpd.overlay(overlay_gdf, zones_utm, how="intersection")

        if gdf_overlayed.empty:
            LOGGER.warning("No intersections found during fractional aggregation")
            return zones

        # Calculate fractional areas
        gdf_overlayed["intersection_area"] = gdf_overlayed.area
        gdf_overlayed["area_fraction"] = (
            gdf_overlayed["intersection_area"] / gdf_overlayed["orig_area"]
        )

        # Apply area weighting to value columns
        for col in value_columns:
            gdf_overlayed[col] = gdf_overlayed[col] * gdf_overlayed["area_fraction"]

        # Aggregate by zone ID
        aggregated = gdf_overlayed.groupby(zone_id_column)[value_columns].agg(agg_funcs)

        # Handle column naming for multi-level index
        aggregated = _handle_multiindex_columns(aggregated)

        # Reset index and merge back to zones
        aggregated = aggregated.reset_index()

        # Return only the aggregated data (will be merged with full zones later)
        return aggregated

    except Exception as e:
        raise RuntimeError(f"Error during area-weighted aggregation: {e}")


def _simple_aggregation(
    polygons_gdf, zones, value_columns, agg_funcs, zone_id_column, predicate
):
    """Perform simple (non-weighted) aggregation"""
    # Perform spatial join
    joined = gpd.sjoin(polygons_gdf, zones, how="inner", predicate=predicate)

    if joined.empty:
        LOGGER.warning(f"No {predicate} relationships found during spatial join")
        return zones

    # Remove geometry column for aggregation (keep only necessary columns)
    agg_cols = value_columns + [zone_id_column]
    joined_subset = joined[agg_cols].copy()

    # Group by zone ID and aggregate
    aggregated = joined_subset.groupby(zone_id_column)[value_columns].agg(agg_funcs)

    # Handle column naming for multi-level index
    aggregated = _handle_multiindex_columns(aggregated)

    # Reset index and merge back to zones
    aggregated = aggregated.reset_index()

    # Return only the aggregated data (will be merged with full zones later)
    return aggregated


def _handle_multiindex_columns(aggregated):
    """Handle multi-level column index from groupby aggregation"""
    if isinstance(aggregated.columns, pd.MultiIndex):
        # Flatten multi-level columns: combine column name with aggregation method
        aggregated.columns = [f"{col[0]}_{col[1]}" for col in aggregated.columns]
    return aggregated

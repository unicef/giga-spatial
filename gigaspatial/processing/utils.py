import pandas as pd
import uuid
from typing import List


def assign_id(
    df: pd.DataFrame, required_columns: List[str], id_column: str = "id"
) -> pd.DataFrame:
    """
    Generate IDs for any entity type in a pandas DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing entity data
        required_columns (List[str]): List of column names required for ID generation
        id_column (str): Name for the id column that will be generated

    Returns:
        pd.DataFrame: DataFrame with generated id column
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Check if ID column exists, if not create it with None values
    if id_column not in df.columns:
        df[id_column] = None

    # Check required columns exist
    if not all(col in df.columns for col in required_columns):
        return df

    # Create identifier concat for UUID generation
    df["identifier_concat"] = (
        df[required_columns].astype(str).fillna("").agg("".join, axis=1)
    )

    # Generate UUIDs only where all required fields are present and no existing ID
    mask = df[id_column].isna()
    for col in required_columns:
        mask &= df[col].notna()

    # Apply UUID generation only where mask is True
    df.loc[mask, id_column] = df.loc[mask, "identifier_concat"].apply(
        lambda x: str(uuid.uuid3(uuid.NAMESPACE_DNS, x))
    )

    # Drop temporary column
    df = df.drop(columns=["identifier_concat"])

    return df

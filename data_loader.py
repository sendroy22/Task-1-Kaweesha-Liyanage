"""
Data Loader & Deduplication Module
Handles dataset ingestion, structural inspection, variable data structure analysis,
and duplicate identification/handling.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def load_dataset(file_path: Path | str = "Dataset for Data Analytics.xlsx") -> pd.DataFrame:
    """
    Load the Excel dataset into a pandas DataFrame.
    
    Parameters:
        file_path (Path | str): Path to the Excel file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    resolved_path = Path(file_path)
    if not resolved_path.is_absolute():
        resolved_path = Path(__file__).parent / resolved_path

    if not resolved_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {resolved_path}")

    print(f"[INFO] Loading dataset from: {resolved_path.name}")
    df = pd.read_excel(resolved_path)
    return df


def inspect_data_structures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inspect the data structure, data type, statistical classification,
    cardinality, and memory usage for each variable in the dataset.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: Comprehensive variable data structure table.
    """
    print("\n" + "=" * 90)
    print("VARIABLE DATA STRUCTURE & SCHEMA ANALYSIS")
    print("=" * 90)
    
    var_records = []
    
    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)
        non_null_series = series.dropna()
        
        # Underlying Python type of non-null element
        py_type = type(non_null_series.iloc[0]).__name__ if not non_null_series.empty else "None"
        
        # Statistical Classification
        nunique = series.nunique()
        if pd.api.types.is_datetime64_any_dtype(series):
            stat_type = "Temporal / Datetime"
        elif col in ["OrderID", "TrackingNumber"]:
            stat_type = "Unique Identifier (Key)"
        elif col in ["CustomerID"]:
            stat_type = "Entity Identifier (ID)"
        elif pd.api.types.is_integer_dtype(series):
            stat_type = f"Discrete Numeric (Range: {series.min()} - {series.max()})"
        elif pd.api.types.is_float_dtype(series):
            stat_type = f"Continuous Numeric (${series.min():.2f} - ${series.max():.2f})"
        elif nunique <= 10:
            stat_type = f"Categorical Nominal ({nunique} levels)"
        else:
            stat_type = f"Text / Categorical ({nunique} unique)"
            
        # Sample values string
        sample_vals = non_null_series.unique()[:3].tolist()
        sample_str = ", ".join([str(v)[:18] for v in sample_vals])
        
        # Memory consumption
        mem_kb = series.memory_usage(deep=True) / 1024
        
        var_records.append({
            "Variable": col,
            "Pandas Dtype": dtype_str,
            "Python Type": py_type,
            "Statistical Structure": stat_type,
            "Non-Null Count": f"{series.count():,}",
            "Null Count (%)": f"{series.isnull().sum()} ({series.isnull().mean()*100:.1f}%)",
            "Unique Values": f"{nunique:,}",
            "Sample Values": f"[{sample_str}]",
            "Memory (KB)": f"{mem_kb:.1f}"
        })
        
    struct_df = pd.DataFrame(var_records).set_index("Variable")
    print(struct_df.to_string())
    print("=" * 90)
    return struct_df


def inspect_dataset(df: pd.DataFrame) -> None:
    """
    Print initial summary information, dimensions, data types, missing value percentages,
    and detailed variable data structures.
    """
    print("=" * 90)
    print("STEP 1: INITIAL DATASET OVERVIEW & STRUCTURAL INSPECTION")
    print("=" * 90)
    print(f"Total Records (Rows): {df.shape[0]:,}")
    print(f"Total Variables (Columns): {df.shape[1]}")
    total_mem_kb = df.memory_usage(deep=True).sum() / 1024
    print(f"Total Memory Consumption: {total_mem_kb:.2f} KB ({total_mem_kb / 1024:.2f} MB)")
    print("=" * 90)
    
    # Detailed data structure inspection for every variable
    inspect_data_structures(df)


def identify_duplicates(df: pd.DataFrame, primary_key: str = "OrderID") -> pd.DataFrame:
    """
    Identify full-row duplicates and primary-key duplicates in the dataset.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        primary_key (str): Column name representing the primary unique identifier.
        
    Returns:
        pd.DataFrame: DataFrame containing detected duplicate rows.
    """
    print("\n" + "=" * 90)
    print("STEP 2: DUPLICATE VALUE IDENTIFICATION")
    print("=" * 90)
    
    full_duplicates_count = df.duplicated().sum()
    print(f"Full-Row Duplicates: {full_duplicates_count} ({full_duplicates_count / len(df):.2%})")
    
    key_duplicates_count = 0
    if primary_key in df.columns:
        key_duplicates_count = df.duplicated(subset=[primary_key]).sum()
        print(f"Primary Key ('{primary_key}') Duplicates: {key_duplicates_count} ({key_duplicates_count / len(df):.2%})")
    
    if full_duplicates_count > 0:
        print("\nSample Full Duplicate Rows:")
        print(df[df.duplicated(keep=False)].head(6))
    else:
        print("[SUCCESS] No full-row duplicates found.")
        
    print("=" * 90)
    return df[df.duplicated(keep=False)]


def handle_duplicates(df: pd.DataFrame, subset: list[str] | None = None, keep: str = "first") -> pd.DataFrame:
    """
    Handle duplicate records by removing redundant entries.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        subset (list[str] | None): Columns to consider for identifying duplicates.
        keep (str): Which duplicate to keep ('first', 'last', False).
        
    Returns:
        pd.DataFrame: Deduplicated DataFrame.
    """
    print("\n" + "=" * 90)
    print("STEP 3: DUPLICATE VALUE HANDLING")
    print("=" * 90)
    
    initial_rows = len(df)
    df_cleaned = df.drop_duplicates(subset=subset, keep=keep).copy()
    dropped_count = initial_rows - len(df_cleaned)
    
    if dropped_count > 0:
        print(f"[ACTION] Dropped {dropped_count} duplicate rows.")
    else:
        print("[ACTION] No rows dropped (0 duplicates present).")
        
    print(f"Rows remaining: {len(df_cleaned)}")
    print("=" * 90)
    return df_cleaned


if __name__ == "__main__":
    df = load_dataset()
    inspect_dataset(df)
    identify_duplicates(df)
    df_deduped = handle_duplicates(df)

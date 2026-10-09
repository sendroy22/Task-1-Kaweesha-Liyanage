"""
Data Cleaning Module - Data Science Project 1
Handles dataset ingestion, structural inspection, duplicate deduplication,
missing value diagnostics & imputation, and outlier detection/capping.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def load_raw_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load raw Excel dataset into pandas DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {path}")
    print(f"[DATA CLEANING] Loading raw dataset from: {path}")
    df = pd.read_excel(path)
    print(f"[DATA CLEANING] Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def inspect_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform structural and data type inspection across all variables.
    """
    records = []
    for col in df.columns:
        s = df[col]
        records.append({
            "Column": col,
            "Dtype": str(s.dtype),
            "Non_Null_Count": s.count(),
            "Null_Count": s.isnull().sum(),
            "Null_Pct": round(s.isnull().mean() * 100, 2),
            "Unique_Values": s.nunique()
        })
    summary_df = pd.DataFrame(records)
    return summary_df


def handle_duplicates(df: pd.DataFrame, primary_key: str = "OrderID") -> pd.DataFrame:
    """
    Identify and remove duplicate records while preserving the first occurrence.
    """
    initial_rows = len(df)
    full_dups = df.duplicated().sum()
    pk_dups = df.duplicated(subset=[primary_key]).sum() if primary_key in df.columns else 0
    print(f"[DATA CLEANING] Full duplicate rows: {full_dups}, Primary Key ('{primary_key}') duplicates: {pk_dups}")
    
    df_deduped = df.drop_duplicates(keep="first").copy()
    dropped = initial_rows - len(df_deduped)
    print(f"[DATA CLEANING] Retained {len(df_deduped)} rows (dropped {dropped} duplicates).")
    return df_deduped


def analyze_and_impute_missing(
    df: pd.DataFrame, 
    tables_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Diagnose missingness and perform statistical & domain-aware imputation.
    - Categorical 'CouponCode' -> 'NO_COUPON' + create binary flag 'HasCoupon' (1/0).
    - Exports missing value summary table.
    """
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / len(df)) * 100
    
    missing_summary = pd.DataFrame({
        "Feature": df.columns,
        "Data_Type": [str(df[col].dtype) for col in df.columns],
        "Missing_Count": missing_counts.values,
        "Missing_Percentage": missing_pcts.round(2).values,
        "Imputation_Strategy": [
            "Domain Indicator ('NO_COUPON' + HasCoupon flag)" if col == "CouponCode" and missing_counts[col] > 0
            else ("Median Imputation" if pd.api.types.is_numeric_dtype(df[col]) and missing_counts[col] > 0
            else ("Mode Imputation" if missing_counts[col] > 0 else "None (Complete)"))
            for col in df.columns
        ]
    })
    
    if tables_dir is not None:
        tables_dir.mkdir(parents=True, exist_ok=True)
        missing_summary.to_csv(tables_dir / "missing_values.csv", index=False)
        print(f"[DATA CLEANING] Saved missing values summary to: {tables_dir / 'missing_values.csv'}")
        
    df_clean = df.copy()
    if "CouponCode" in df_clean.columns and df_clean["CouponCode"].isnull().any():
        df_clean["HasCoupon"] = np.where(df_clean["CouponCode"].notnull(), 1, 0)
        df_clean["CouponCode"] = df_clean["CouponCode"].fillna("NO_COUPON")
        print(f"[DATA CLEANING] Imputed 'CouponCode' NaNs with 'NO_COUPON' and engineered 'HasCoupon' binary flag.")
        
    # Impute any other numeric/categorical columns if present
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                med = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(med)
            else:
                mode_val = df_clean[col].mode().iloc[0]
                df_clean[col] = df_clean[col].fillna(mode_val)
                
    return df_clean, missing_summary


def detect_and_treat_outliers(
    df: pd.DataFrame, 
    numeric_cols: list[str] | None = None,
    factor: float = 1.5,
    method: str = "cap",
    tables_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect outliers using IQR rule and treat them using Winsorization (capping).
    Exports outlier summary table.
    """
    if numeric_cols is None:
        numeric_cols = ["UnitPrice", "Quantity", "ItemsInCart", "TotalPrice"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    outlier_records = []
    df_treated = df.copy()
    
    for col in numeric_cols:
        q1 = float(df[col].quantile(0.25))
        q3 = float(df[col].quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - (factor * iqr)
        upper_bound = q3 + (factor * iqr)
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = round((outlier_count / len(df)) * 100, 2)
        
        outlier_records.append({
            "Feature": col,
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "IQR": round(iqr, 2),
            "Lower_Fence": round(lower_bound, 2),
            "Upper_Fence": round(upper_bound, 2),
            "Outlier_Count": outlier_count,
            "Outlier_Percentage": outlier_pct,
            "Treatment_Applied": f"IQR Winsorization [{lower_bound:.2f}, {upper_bound:.2f}]" if method == "cap" else "None"
        })
        
        if method == "cap" and outlier_count > 0:
            df_treated[col] = np.clip(df_treated[col], lower_bound, upper_bound)
            
    outlier_summary = pd.DataFrame(outlier_records)
    
    if tables_dir is not None:
        tables_dir.mkdir(parents=True, exist_ok=True)
        outlier_summary.to_csv(tables_dir / "outlier_summary.csv", index=False)
        print(f"[DATA CLEANING] Saved outlier summary to: {tables_dir / 'outlier_summary.csv'}")
        
    return df_treated, outlier_summary


def run_data_cleaning_pipeline(
    raw_data_path: str | Path = Path("data/raw/original_dataset.xlsx"),
    processed_output_path: str | Path = Path("data/processed/cleaned_dataset.csv"),
    tables_dir: str | Path = Path("outputs/tables")
) -> pd.DataFrame:
    """
    Orchestrate full data cleaning pipeline.
    """
    raw_path = Path(raw_data_path)
    proc_path = Path(processed_output_path)
    tbl_dir = Path(tables_dir)
    
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n==================================================")
    print(">>> EXECUTING DATA CLEANING PIPELINE")
    print("==================================================")
    
    df_raw = load_raw_data(raw_path)
    df_deduped = handle_duplicates(df_raw, primary_key="OrderID")
    df_imputed, missing_summary = analyze_and_impute_missing(df_deduped, tables_dir=tbl_dir)
    df_cleaned, outlier_summary = detect_and_treat_outliers(
        df_imputed, 
        numeric_cols=["UnitPrice", "Quantity", "ItemsInCart", "TotalPrice"],
        factor=1.5,
        method="cap",
        tables_dir=tbl_dir
    )
    
    # Export cleaned dataset
    df_cleaned.to_csv(proc_path, index=False)
    print(f"[DATA CLEANING] Cleaned dataset saved to: {proc_path}")
    print(f"[DATA CLEANING] Final Cleaned Dataset Shape: {df_cleaned.shape}")
    print("==================================================\n")
    return df_cleaned


if __name__ == "__main__":
    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    raw_p = project_root / "data" / "raw" / "original_dataset.xlsx"
    clean_p = project_root / "data" / "processed" / "cleaned_dataset.csv"
    tables_p = project_root / "outputs" / "tables"
    
    run_data_cleaning_pipeline(raw_p, clean_p, tables_p)

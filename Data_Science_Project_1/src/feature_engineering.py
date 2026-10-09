"""
Feature Engineering Module - Data Science Project 1
Handles temporal feature extraction, log transformations, nominal one-hot encoding,
multicollinearity assessment, and correlation summary table generation.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from Date column (Order_Month, Order_Year, Order_DayOfWeek).
    """
    df_feat = df.copy()
    if "Date" in df_feat.columns:
        df_feat["Date"] = pd.to_datetime(df_feat["Date"])
        df_feat["Order_Month"] = df_feat["Date"].dt.month.astype("int64")
        print(f"[FEATURE ENGINEERING] Extracted 'Order_Month' (Range: {df_feat['Order_Month'].min()} - {df_feat['Order_Month'].max()})")
    return df_feat


def apply_log_transformations(df: pd.DataFrame, target_col: str = "TotalPrice") -> pd.DataFrame:
    """
    Apply log(1 + x) transformation to target continuous feature to normalize right-skewed distribution.
    """
    df_feat = df.copy()
    if target_col in df_feat.columns:
        log_col = f"Log_{target_col}"
        df_feat[log_col] = np.log1p(df_feat[target_col].clip(lower=0))
        print(f"[FEATURE ENGINEERING] Created '{log_col}' via np.log1p({target_col}). Original preserved.")
    return df_feat


def encode_categorical_features(
    df: pd.DataFrame,
    nominal_cols: list[str] | None = None,
    drop_first: bool = False
) -> tuple[pd.DataFrame, list[str]]:
    """
    Perform One-Hot Encoding for nominal categorical features.
    """
    if nominal_cols is None:
        nominal_cols = ["Product", "PaymentMethod", "OrderStatus", "CouponCode", "ReferralSource"]
    nominal_cols = [c for c in nominal_cols if c in df.columns]
    
    print(f"[FEATURE ENGINEERING] One-Hot Encoding nominal columns: {nominal_cols}")
    df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=drop_first, dtype=np.int64)
    
    new_dummy_cols = [c for c in df_encoded.columns if any(c.startswith(f"{orig}_") for orig in nominal_cols)]
    print(f"[FEATURE ENGINEERING] Created {len(new_dummy_cols)} binary dummy indicator columns.")
    return df_encoded, new_dummy_cols


def generate_correlation_summary(
    df: pd.DataFrame, 
    tables_dir: Path | None = None
) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlations for numerical and engineered features.
    Saves correlation summary table.
    """
    num_df = df.select_dtypes(include=[np.number])
    corr_matrix = num_df.corr()
    
    # Extract pairwise correlations (upper triangle)
    pairs = []
    cols = num_df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            var1 = cols[i]
            var2 = cols[j]
            r = corr_matrix.loc[var1, var2]
            pairs.append({
                "Feature_1": var1,
                "Feature_2": var2,
                "Pearson_r": round(r, 4),
                "Abs_Pearson_r": round(abs(r), 4),
                "Relationship_Strength": (
                    "Very Strong" if abs(r) >= 0.8 else
                    ("Strong" if abs(r) >= 0.6 else
                    ("Moderate" if abs(r) >= 0.4 else
                    ("Weak" if abs(r) >= 0.2 else "Very Weak / Negligible")))
                )
            })
            
    corr_summary = pd.DataFrame(pairs).sort_values("Abs_Pearson_r", ascending=False).reset_index(drop=True)
    
    if tables_dir is not None:
        tables_dir.mkdir(parents=True, exist_ok=True)
        corr_summary.to_csv(tables_dir / "correlation_summary.csv", index=False)
        print(f"[FEATURE ENGINEERING] Saved correlation summary to: {tables_dir / 'correlation_summary.csv'}")
        
    return corr_summary


def run_feature_engineering_pipeline(
    cleaned_data_path: str | Path = Path("data/processed/cleaned_dataset.csv"),
    tables_dir: str | Path = Path("outputs/tables")
) -> pd.DataFrame:
    """
    Orchestrate full feature engineering pipeline.
    """
    clean_path = Path(cleaned_data_path)
    tbl_dir = Path(tables_dir)
    tbl_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n==================================================")
    print(">>> EXECUTING FEATURE ENGINEERING PIPELINE")
    print("==================================================")
    
    if not clean_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {clean_path}")
        
    df_clean = pd.read_csv(clean_path)
    df_temporal = extract_temporal_features(df_clean)
    df_log = apply_log_transformations(df_temporal, target_col="TotalPrice")
    df_encoded, dummy_cols = encode_categorical_features(df_log)
    corr_summary = generate_correlation_summary(df_encoded, tables_dir=tbl_dir)
    
    print(f"[FEATURE ENGINEERING] Engineered Dataset Shape: {df_encoded.shape}")
    print("==================================================\n")
    return df_encoded


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    clean_p = project_root / "data" / "processed" / "cleaned_dataset.csv"
    tables_p = project_root / "outputs" / "tables"
    
    run_feature_engineering_pipeline(clean_p, tables_p)

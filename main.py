"""
Data Science Project 1: Advanced Analytics Pipeline Runner
Orchestrates data loading, EDA, missing value handling, outlier detection,
statistical assumptions checking, and feature engineering.
"""

from data_loader import load_dataset, inspect_dataset, identify_duplicates, handle_duplicates
from eda import run_eda
from missing_values import identify_missing_values, handle_missing_values
from outliers import detect_outliers_iqr, detect_outliers_zscore, handle_outliers
from assumptions import check_statistical_assumptions
from feature_engineering import engineer_features
from structural_contracts import run_structural_contracts_phase


def main():
    print("=" * 80)
    print(">>> STARTING MODULAR DATA SCIENCE PIPELINE")
    print("=" * 80)
    
    # PHASE 1: Data Ingestion & Structural Inspection
    print("\n>>> [PHASE 1] DATA INGESTION & INSPECTION")
    df_raw = load_dataset()
    inspect_dataset(df_raw)
    
    # PHASE 2: Duplicate Identification & Deduplication
    print("\n>>> [PHASE 2] DUPLICATE CHECK & DEDUPLICATION")
    identify_duplicates(df_raw, primary_key="OrderID")
    df_deduped = handle_duplicates(df_raw)
    
    # PHASE 3: Missing Value Diagnostics & Statistical Imputation
    print("\n>>> [PHASE 3] MISSING VALUE HANDLING")
    identify_missing_values(df_deduped)
    df_clean = handle_missing_values(df_deduped, categorical_strategy="domain_indicator")
    
    # PHASE 4: Exploratory Data Analysis (EDA)
    print("\n>>> [PHASE 4] EXPLORATORY DATA ANALYSIS")
    run_eda(df_clean)
    
    # PHASE 5: Outlier Detection & Neutralization
    print("\n>>> [PHASE 5] OUTLIER DETECTION & TREATMENT")
    numeric_cols = ["UnitPrice", "Quantity", "ItemsInCart", "TotalPrice"]
    detect_outliers_iqr(df_clean, columns=numeric_cols, factor=1.5)
    detect_outliers_zscore(df_clean, columns=numeric_cols, threshold=3.0)
    df_treated = handle_outliers(df_clean, columns=numeric_cols, method="cap")
    
    # PHASE 6: Statistical Assumptions Verification
    print("\n>>> [PHASE 6] STATISTICAL ASSUMPTIONS TESTING")
    check_statistical_assumptions(df_treated)
    
    # PHASE 7: Feature Engineering & Transformation
    print("\n>>> [PHASE 7] PREDICTIVE FEATURE ENGINEERING")
    df_engineered = engineer_features(df_treated)
    
    # PHASE 8: Structural Contracts, Validation & Model-Ready Scaling
    print("\n>>> [PHASE 8] STRUCTURAL CONTRACTS & MODEL-READY VALIDATION")
    df_model_ready = run_structural_contracts_phase(
        df_engineered, 
        export_dataset=True, 
        output_path="final_model_ready_dataset.csv"
    )
    
    # Summary of Final Pipeline Output
    print("\n" + "=" * 80)
    print("[SUCCESS] DATA SCIENCE PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Final Model-Ready Dataset Shape: {df_model_ready.shape}")
    print("\nSample Validated Records (First 5 rows):")
    display_cols = [
        "OrderID", "Date", "Product", "TotalPrice", "CouponCode", "HasCoupon", "Order_Month"
    ]
    avail_cols = [c for c in display_cols if c in df_model_ready.columns]
    print(df_model_ready[avail_cols].head())
    print("=" * 80)
    
    return df_model_ready


if __name__ == "__main__":
    df_final = main()

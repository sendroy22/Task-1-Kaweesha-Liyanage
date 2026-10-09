"""
Validation & Structural Contracts Module - Data Science Project 1
Implements Pandera DataFrameSchema data contracts, domain boundary validation,
zero-null enforcement, duplicate prevention, and model-ready dataset export.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema


def define_pandera_schema(df: pd.DataFrame) -> DataFrameSchema:
    """
    Define dynamic Pandera DataFrameSchema enforcing strict structural and domain rules.
    """
    schema_columns = {}
    
    # Base numeric constraints
    if "Quantity" in df.columns:
        schema_columns["Quantity"] = Column(
            int,
            Check.in_range(1, 5),
            nullable=False,
            description="Quantity must be integer between 1 and 5 inclusive."
        )
    if "UnitPrice" in df.columns:
        schema_columns["UnitPrice"] = Column(
            float,
            Check.greater_than_or_equal_to(0.01),
            nullable=False,
            description="UnitPrice must be strictly positive."
        )
    if "TotalPrice" in df.columns:
        schema_columns["TotalPrice"] = Column(
            float,
            Check.greater_than_or_equal_to(0.0),
            nullable=False,
            description="TotalPrice must be non-negative."
        )
    if "ItemsInCart" in df.columns:
        schema_columns["ItemsInCart"] = Column(
            int,
            Check.in_range(1, 10),
            nullable=False,
            description="ItemsInCart must be integer between 1 and 10."
        )
    if "Order_Month" in df.columns:
        schema_columns["Order_Month"] = Column(
            int,
            Check.in_range(1, 12),
            nullable=False,
            description="Order_Month must be valid calendar month between 1 and 12."
        )
    if "HasCoupon" in df.columns:
        schema_columns["HasCoupon"] = Column(
            int,
            Check.isin([0, 1]),
            nullable=False,
            description="HasCoupon must be binary indicator {0, 1}."
        )
    if "Log_TotalPrice" in df.columns:
        schema_columns["Log_TotalPrice"] = Column(
            float,
            Check.greater_than_or_equal_to(0.0),
            nullable=False,
            description="Log_TotalPrice must be non-negative."
        )
        
    # Check all one-hot encoded columns (prefixed or dummy columns)
    for col in df.columns:
        if any(col.startswith(p) for p in ["Product_", "PaymentMethod_", "OrderStatus_", "CouponCode_", "ReferralSource_"]):
            schema_columns[col] = Column(
                int,
                Check.isin([0, 1]),
                nullable=False,
                description=f"One-hot indicator '{col}' must be binary {0, 1}."
            )
            
    # Identity and timestamp columns
    if "OrderID" in df.columns:
        schema_columns["OrderID"] = Column(
            str,
            nullable=False,
            unique=True,
            description="OrderID must be unique primary key."
        )
        
    schema = DataFrameSchema(
        columns=schema_columns,
        strict=False,
        coerce=True
    )
    return schema


def run_structural_validation(
    df: pd.DataFrame, 
    tables_dir: Path | None = None
) -> tuple[bool, pd.DataFrame]:
    """
    Run multi-point data contract and domain validation checks.
    Produces validation summary report table.
    """
    print("\n==================================================")
    print(">>> RUNNING STRUCTURAL CONTRACTS & DATA VALIDATION")
    print("==================================================")
    
    validation_records = []
    
    # 1. Total Missing Values Check
    total_nulls = int(df.isnull().sum().sum())
    validation_records.append({
        "Check_ID": "VAL-01",
        "Validation_Domain": "Completeness",
        "Rule_Description": "Zero missing values across all columns",
        "Target_Columns": "All Columns",
        "Observed_Metric": f"{total_nulls} nulls",
        "Pass_Status": "PASSED" if total_nulls == 0 else "FAILED"
    })
    
    # 2. Primary Key Uniqueness
    pk_dups = int(df.duplicated(subset=["OrderID"]).sum()) if "OrderID" in df.columns else 0
    validation_records.append({
        "Check_ID": "VAL-02",
        "Validation_Domain": "Uniqueness",
        "Rule_Description": "Primary key 'OrderID' must be 100% unique",
        "Target_Columns": "OrderID",
        "Observed_Metric": f"{pk_dups} duplicate keys",
        "Pass_Status": "PASSED" if pk_dups == 0 else "FAILED"
    })
    
    # 3. Numerical Boundary Checks
    num_rules = [
        ("Quantity", lambda s: (s >= 1) & (s <= 5), "Quantity in [1, 5]"),
        ("UnitPrice", lambda s: s > 0, "UnitPrice > 0"),
        ("TotalPrice", lambda s: s >= 0, "TotalPrice >= 0"),
        ("ItemsInCart", lambda s: (s >= 1) & (s <= 10), "ItemsInCart in [1, 10]"),
        ("Order_Month", lambda s: (s >= 1) & (s <= 12), "Order_Month in [1, 12]"),
        ("HasCoupon", lambda s: s.isin([0, 1]), "HasCoupon in {0, 1}"),
        ("Log_TotalPrice", lambda s: s >= 0, "Log_TotalPrice >= 0")
    ]
    
    check_idx = 3
    for col, rule_fn, desc in num_rules:
        if col in df.columns:
            valid_mask = rule_fn(df[col])
            violation_count = int((~valid_mask).sum())
            status = "PASSED" if violation_count == 0 else "FAILED"
            min_v = df[col].min()
            max_v = df[col].max()
            validation_records.append({
                "Check_ID": f"VAL-{check_idx:02d}",
                "Validation_Domain": "Domain Constraints",
                "Rule_Description": desc,
                "Target_Columns": col,
                "Observed_Metric": f"Min: {min_v}, Max: {max_v} (Violations: {violation_count})",
                "Pass_Status": status
            })
            check_idx += 1
            
    # 4. Binary dummy validity
    dummy_cols = [c for c in df.columns if any(c.startswith(p) for p in ["Product_", "PaymentMethod_", "OrderStatus_", "CouponCode_", "ReferralSource_"])]
    if dummy_cols:
        invalid_dummies = sum(int((~df[c].isin([0, 1])).sum()) for c in dummy_cols)
        validation_records.append({
            "Check_ID": f"VAL-{check_idx:02d}",
            "Validation_Domain": "Encoding Integrity",
            "Rule_Description": "All One-Hot dummies must be binary {0, 1}",
            "Target_Columns": f"{len(dummy_cols)} One-Hot Columns",
            "Observed_Metric": f"Violations: {invalid_dummies}",
            "Pass_Status": "PASSED" if invalid_dummies == 0 else "FAILED"
        })
        check_idx += 1
        
    # 5. Pandera Schema Engine Execution
    schema = define_pandera_schema(df)
    pandera_status = "PASSED"
    pandera_msg = "Schema verified with 0 schema errors."
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        pandera_status = "FAILED"
        pandera_msg = f"{len(err.failure_cases)} schema failure cases."
        print(f"[PANDERA SCHEMA ALERT] {pandera_msg}")
        
    validation_records.append({
        "Check_ID": f"VAL-{check_idx:02d}",
        "Validation_Domain": "Pandera Engine",
        "Rule_Description": "Automated Pandera DataFrameSchema lazy evaluation",
        "Target_Columns": "Validated Schema Columns",
        "Observed_Metric": pandera_msg,
        "Pass_Status": pandera_status
    })
    
    validation_summary = pd.DataFrame(validation_records)
    
    if tables_dir is not None:
        tables_dir.mkdir(parents=True, exist_ok=True)
        validation_summary.to_csv(tables_dir / "validation_summary.csv", index=False)
        print(f"[VALIDATION] Saved validation summary to: {tables_dir / 'validation_summary.csv'}")
        
    all_passed = all(r["Pass_Status"] == "PASSED" for r in validation_records)
    print(f"\n[VALIDATION] Overall Data Contract Status: {'[ALL CONTRACTS PASSED]' if all_passed else '[CONTRACT VIOLATIONS DETECTED]'}")
    print("==================================================\n")
    return all_passed, validation_summary


def run_validation_pipeline(
    engineered_df: pd.DataFrame,
    final_output_path: str | Path = Path("data/processed/final_model_ready_dataset.csv"),
    tables_dir: str | Path = Path("outputs/tables")
) -> pd.DataFrame:
    """
    Validate engineered dataset and export final model-ready dataset.
    """
    final_path = Path(final_output_path)
    tbl_dir = Path(tables_dir)
    
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)
    
    is_valid, val_summary = run_structural_validation(engineered_df, tables_dir=tbl_dir)
    
    # Export final model-ready dataset
    engineered_df.to_csv(final_path, index=False)
    print(f"[VALIDATION] Final Model-Ready Dataset saved to: {final_path}")
    print(f"[VALIDATION] Output Dataset Dimensions: {engineered_df.shape[0]} Rows x {engineered_df.shape[1]} Columns")
    return engineered_df


if __name__ == "__main__":
    from feature_engineering import run_feature_engineering_pipeline
    from data_cleaning import run_data_cleaning_pipeline
    
    project_root = Path(__file__).resolve().parent.parent
    raw_p = project_root / "data" / "raw" / "original_dataset.xlsx"
    clean_p = project_root / "data" / "processed" / "cleaned_dataset.csv"
    final_p = project_root / "data" / "processed" / "final_model_ready_dataset.csv"
    tables_p = project_root / "outputs" / "tables"
    
    df_clean = run_data_cleaning_pipeline(raw_p, clean_p, tables_p)
    df_eng = run_feature_engineering_pipeline(clean_p, tables_p)
    df_model_ready = run_validation_pipeline(df_eng, final_p, tables_p)

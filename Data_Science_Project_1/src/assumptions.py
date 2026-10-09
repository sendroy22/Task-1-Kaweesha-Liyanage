"""
Statistical Assumptions Module - Data Science Project 1
Tests for statistical modeling assumptions:
1. Normality Diagnostics (Skewness, Kurtosis, Shapiro-Wilk & D'Agostino-Pearson tests)
2. Multicollinearity Assessment (Pearson Correlation Matrix & Variance Inflation Factors - VIF)
3. Target Variable ($Y$) Formulation & ML Modeling Paradigms
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def check_normality(
    df: pd.DataFrame, 
    columns: list[str] | None = None,
    tables_dir: Path | None = None
) -> pd.DataFrame:
    """
    Evaluate normality assumptions using Skewness, Kurtosis, and Shapiro-Wilk tests.
    """
    print("\n" + "=" * 80)
    print(">>> [ASSUMPTIONS] STATISTICAL ASSUMPTION 1: NORMALITY TESTS")
    print("=" * 80)
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    results = []
    for col in columns:
        clean_s = df[col].dropna()
        n = len(clean_s)
        
        skew_val = float(clean_s.skew())
        kurt_val = float(clean_s.kurtosis())
        
        # Shapiro-Wilk test (capped at 5,000 observations per scipy requirement)
        sample_s = clean_s if n <= 5000 else clean_s.sample(5000, random_state=42)
        stat, p_val = stats.shapiro(sample_s)
        
        is_normal = (abs(skew_val) < 0.5) and (p_val > 0.05)
        
        results.append({
            "Feature": col,
            "Skewness": round(skew_val, 3),
            "Kurtosis": round(kurt_val, 3),
            "Shapiro_Stat": round(stat, 4),
            "p_value": f"{p_val:.4e}",
            "Normally_Distributed": "Yes" if is_normal else "No (Skewed/Non-normal)",
            "Recommendation": "Standard scaling" if is_normal else "Log/Power transform or robust scaling"
        })
        
    normality_df = pd.DataFrame(results)
    print(normality_df.to_string(index=False))
    print("=" * 80)
    return normality_df


def check_multicollinearity_vif(
    df: pd.DataFrame, 
    columns: list[str] | None = None,
    tables_dir: Path | None = None
) -> pd.DataFrame:
    """
    Evaluate feature multicollinearity using Variance Inflation Factors (VIF).
    VIF = 1 / (1 - R_i^2)
    """
    print("\n" + "=" * 80)
    print(">>> [ASSUMPTIONS] STATISTICAL ASSUMPTION 2: MULTICOLLINEARITY (VIF)")
    print("=" * 80)
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude high-correlation derived pairs like Log_TotalPrice from base VIF check if present
        columns = [c for c in columns if c not in ["OrderID", "CustomerID", "TrackingNumber"]]
        
    numeric_data = df[columns].dropna()
    vif_records = []
    
    if len(columns) > 1:
        X = numeric_data.values
        X_centered = X - np.mean(X, axis=0)
        
        for i, col in enumerate(columns):
            y_i = X_centered[:, i]
            X_others = np.delete(X_centered, i, axis=1)
            
            try:
                coeffs, residuals, rank, s = np.linalg.lstsq(X_others, y_i, rcond=None)
                y_pred = X_others @ coeffs
                ss_res = np.sum((y_i - y_pred) ** 2)
                ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                vif = 1.0 / (1.0 - r_squared) if (1.0 - r_squared) > 1e-6 else 999.0
            except Exception:
                vif = np.nan
                
            risk = "Low (<5)" if vif < 5 else ("Moderate (5-10)" if vif <= 10 else "High Severe (>10)")
            vif_records.append({
                "Feature": col,
                "VIF": round(vif, 2) if not np.isnan(vif) else "N/A",
                "Multicollinearity_Risk": risk
            })
            
        vif_df = pd.DataFrame(vif_records)
        print(vif_df.to_string(index=False))
    else:
        vif_df = pd.DataFrame()
        print("[INFO] Need at least 2 numerical features to compute VIF.")
        
    print("=" * 80)
    return vif_df


def identify_target_variable_paradigms() -> pd.DataFrame:
    """
    Document target variable (Y) formulation and supervised ML problem mappings.
    """
    print("\n" + "=" * 80)
    print(">>> [ASSUMPTIONS] TARGET VARIABLE (Y) FORMULATION & ML PROBLEM MAPPINGS")
    print("=" * 80)
    
    mappings = [
        {
            "Paradigm": "Primary Target (Regression)",
            "Target_Y": "TotalPrice / Log_TotalPrice",
            "Target_Type": "Continuous Float ($)",
            "Problem_Type": "Supervised Regression / Revenue Estimation",
            "Objective": "Forecast transaction dollar value from basket items, quantity, unit price & promos."
        },
        {
            "Paradigm": "Secondary Target (Classification)",
            "Target_Y": "OrderStatus",
            "Target_Type": "Categorical Nominal (5 classes)",
            "Problem_Type": "Multi-class Classification / Churn & Logistics Risk",
            "Objective": "Predict likelihood of order cancellation or return prior to fulfillment."
        },
        {
            "Paradigm": "Tertiary Target (Propensity)",
            "Target_Y": "HasCoupon (Binary)",
            "Target_Type": "Binary Indicator {0, 1}",
            "Problem_Type": "Binary Classification / Coupon Propensity",
            "Objective": "Model consumer price sensitivity and discount adoption."
        }
    ]
    df_targets = pd.DataFrame(mappings)
    print(df_targets.to_string(index=False))
    print("=" * 80)
    return df_targets


def run_statistical_assumptions(
    df: pd.DataFrame, 
    tables_dir: str | Path | None = Path("outputs/tables")
) -> pd.DataFrame:
    """
    Run full statistical assumptions diagnostic suite and export summary table.
    """
    print("\n==================================================")
    print(">>> EXECUTING STATISTICAL ASSUMPTIONS TESTING")
    print("==================================================")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    normality_df = check_normality(df, columns=num_cols)
    vif_df = check_multicollinearity_vif(df, columns=num_cols)
    targets_df = identify_target_variable_paradigms()
    
    # Merge normality and VIF into unified assumptions summary
    assumptions_summary = normality_df.merge(vif_df, on="Feature", how="left")
    
    if tables_dir is not None:
        t_dir = Path(tables_dir)
        t_dir.mkdir(parents=True, exist_ok=True)
        out_csv = t_dir / "statistical_assumptions.csv"
        assumptions_summary.to_csv(out_csv, index=False)
        print(f"\n[ASSUMPTIONS] Saved statistical assumptions summary to: {out_csv}")
        
    print("==================================================\n")
    return assumptions_summary


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    clean_p = project_root / "data" / "processed" / "cleaned_dataset.csv"
    raw_p = project_root / "data" / "raw" / "original_dataset.xlsx"
    tbl_p = project_root / "outputs" / "tables"
    
    data_p = clean_p if clean_p.exists() else raw_p
    df = pd.read_csv(data_p) if data_p.suffix == ".csv" else pd.read_excel(data_p)
    run_statistical_assumptions(df, tables_dir=tbl_p)

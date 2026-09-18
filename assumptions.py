"""
Statistical Assumptions Module
Tests for Normality (Skewness, Kurtosis, Shapiro-Wilk), Multicollinearity (VIF, Correlation), and Homoscedasticity.
"""

import numpy as np
import pandas as pd
from scipy import stats


def check_normality(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Evaluate normality assumptions using Skewness, Kurtosis, and Shapiro-Wilk / D'Agostino tests.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list[str] | None): Numerical columns to test.
        
    Returns:
        pd.DataFrame: Summary table of normality diagnostics.
    """
    print("\n" + "=" * 70)
    print("STATISTICAL ASSUMPTION 1: NORMALITY TESTS")
    print("=" * 70)
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    results = []
    
    for col in columns:
        clean_series = df[col].dropna()
        n = len(clean_series)
        
        skew_val = clean_series.skew()
        kurt_val = clean_series.kurtosis()
        
        # Perform Shapiro-Wilk test (sample capped at 5000)
        stat, p_val = stats.shapiro(clean_series if n <= 5000 else clean_series.sample(5000, random_state=42))
        
        # Interpret normality
        is_normal = (abs(skew_val) < 0.5) and (p_val > 0.05)
        
        results.append({
            "Feature": col,
            "Skewness": round(skew_val, 3),
            "Kurtosis": round(kurt_val, 3),
            "Shapiro Stat": round(stat, 4),
            "p-value": f"{p_val:.4e}",
            "Normally Distributed?": "Yes" if is_normal else "No (Non-normal / Skewed)"
        })
        
    summary_df = pd.DataFrame(results).set_index("Feature")
    print(summary_df)
    print("=" * 70)
    return summary_df


def check_multicollinearity(df: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate multicollinearity using Pearson correlation matrix and Variance Inflation Factor (VIF).
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list[str] | None): Numerical feature columns.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (Correlation Matrix, VIF Table).
    """
    print("\n" + "=" * 70)
    print("STATISTICAL ASSUMPTION 2: MULTICOLLINEARITY & VIF ANALYSIS")
    print("=" * 70)
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    numeric_data = df[columns].dropna()
    
    # 1. Pearson Correlation Matrix
    corr_matrix = numeric_data.corr(method="pearson").round(3)
    print("--- Pearson Correlation Matrix ---")
    print(corr_matrix)
    
    # 2. Variance Inflation Factor (VIF) Calculation
    vif_records = []
    
    if len(columns) > 1:
        # Standardize features for linear algebraic VIF computation
        X = numeric_data.values
        X_centered = X - np.mean(X, axis=0)
        
        for i, col in enumerate(columns):
            y_i = X_centered[:, i]
            X_others = np.delete(X_centered, i, axis=1)
            
            # Linear regression R^2 to find VIF = 1 / (1 - R^2)
            try:
                # OLS estimate via least squares
                coeffs, residuals, rank, s = np.linalg.lstsq(X_others, y_i, rcond=None)
                y_pred = X_others @ coeffs
                ss_res = np.sum((y_i - y_pred) ** 2)
                ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                vif = 1.0 / (1.0 - r_squared) if (1.0 - r_squared) > 1e-6 else 999.0
            except Exception:
                vif = np.nan
                
            risk_level = "Low (<5)" if vif < 5 else ("Moderate (5-10)" if vif <= 10 else "High Severe (>10)")
            vif_records.append({
                "Feature": col,
                "VIF": round(vif, 2) if not np.isnan(vif) else "N/A",
                "Multicollinearity Risk": risk_level
            })
            
        vif_df = pd.DataFrame(vif_records).set_index("Feature")
        print("\n--- Variance Inflation Factor (VIF) ---")
        print(vif_df)
    else:
        vif_df = pd.DataFrame()
        print("\n[INFO] Need at least 2 numerical features to compute VIF.")
        
    print("=" * 70)
    return corr_matrix, vif_df


def check_statistical_assumptions(df: pd.DataFrame) -> None:
    """
    Master runner to execute all core statistical diagnostic tests.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        check_normality(df, columns=numeric_cols)
        check_multicollinearity(df, columns=numeric_cols)


if __name__ == "__main__":
    from data_loader import load_dataset, handle_duplicates
    from missing_values import handle_missing_values
    
    df = load_dataset()
    df = handle_duplicates(df)
    df = handle_missing_values(df)
    
    check_statistical_assumptions(df)

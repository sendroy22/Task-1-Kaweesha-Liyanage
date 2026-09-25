"""
Statistical Assumptions Module
Tests for Normality (Skewness, Kurtosis, Shapiro-Wilk), Multicollinearity (Correlation Matrix, VIF),
and Target Variable ($Y$) Formulation.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def ensure_figures_dir(plot_path: str) -> None:
    """Ensure parent directory for saving figures exists."""
    plot_dir = os.path.dirname(plot_path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)


def check_normality(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Evaluate normality assumptions using Skewness, Kurtosis, and Shapiro-Wilk / D'Agostino tests.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list[str] | None): Numerical columns to test.
        
    Returns:
        pd.DataFrame: Summary table of normality diagnostics.
    """
    print("\n" + "=" * 75)
    print("STATISTICAL ASSUMPTION 1: NORMALITY TESTS")
    print("=" * 75)
    
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
    print("=" * 75)
    return summary_df


def check_multicollinearity(
    df: pd.DataFrame, 
    columns: list[str] | None = None,
    save_plot: bool = True,
    plot_path: str = os.path.join("figures", "correlation_matrix.png")
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate multicollinearity using Pearson correlation matrix and Variance Inflation Factor (VIF).
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list[str] | None): Numerical feature columns.
        save_plot (bool): Whether to save correlation heatmap figure.
        plot_path (str): Filepath to save heatmap.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (Correlation Matrix, VIF Table).
    """
    print("\n" + "=" * 75)
    print("STATISTICAL ASSUMPTION 2: CORRELATION MATRIX & MULTICOLLINEARITY (VIF)")
    print("=" * 75)
    
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
        
    # Graphical Correlation Heatmap
    if save_plot and not corr_matrix.empty:
        ensure_figures_dir(plot_path)
        fig, ax = plt.subplots(figsize=(8, 7))
        
        cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(cax, fraction=0.046, pad=0.04)
        
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="left", fontsize=10, fontweight="bold")
        ax.set_yticklabels(corr_matrix.index, fontsize=10, fontweight="bold")
        
        # Annotate correlation coefficient values inside each cell
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                text_color = "white" if abs(val) > 0.55 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=10, fontweight="bold")
                
        plt.title("Pearson Correlation Heatmap Matrix", fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\n[FIGURE SAVED] Correlation heatmap matrix saved to: '{plot_path}'")
        
    print("=" * 75)
    return corr_matrix, vif_df


def identify_target_variable(df: pd.DataFrame) -> dict:
    """
    Formulate and identify potential target ($Y$) variables in the dataset
    based on machine learning problem paradigms.
    """
    print("\n" + "=" * 75)
    print("TARGET VARIABLE ($Y$) FORMULATION & ML PROBLEM MAPPING")
    print("=" * 75)
    
    target_mappings = {
        "Primary Regression Target": {
            "Variable (Y)": "TotalPrice",
            "Data Type": "Continuous Float ($)",
            "Problem Type": "Continuous Regression / Revenue Prediction",
            "Business Objective": "Predict expected transaction revenue based on cart contents, quantity, unit pricing, promotions, and acquisition channel.",
            "Predictors (X)": ["Quantity", "UnitPrice", "ItemsInCart", "HasCoupon", "Product", "PaymentMethod", "ReferralSource", "Date_Temporal_Features"]
        },
        "Alternative Classification Target 1": {
            "Variable (Y)": "OrderStatus",
            "Data Type": "Categorical Nominal (5 classes)",
            "Problem Type": "Multi-class Classification / Fulfillment & Churn Risk",
            "Business Objective": "Predict order cancellation or return risk at checkout to optimize logistics and fraud detection.",
            "Predictors (X)": ["TotalPrice", "UnitPrice", "Quantity", "ItemsInCart", "PaymentMethod", "CouponCode", "ReferralSource"]
        },
        "Alternative Classification Target 2": {
            "Variable (Y)": "HasCoupon (or CouponCode)",
            "Data Type": "Binary Indicator (0 / 1)",
            "Problem Type": "Binary Classification / Propensity to Apply Discounts",
            "Business Objective": "Model customer price sensitivity and coupon utilization propensity for targeted promotions.",
            "Predictors (X)": ["TotalPrice", "Quantity", "Product", "ReferralSource", "ItemsInCart"]
        },
        "Alternative Regression Target 3": {
            "Variable (Y)": "Quantity",
            "Data Type": "Discrete Integer (1 - 5)",
            "Problem Type": "Discrete Regression / Demand Estimation",
            "Business Objective": "Forecast item order volume based on unit pricing and marketing referral channel.",
            "Predictors (X)": ["UnitPrice", "Product", "ReferralSource", "HasCoupon"]
        }
    }
    
    for category, details in target_mappings.items():
        print(f"\n>>> {category}: '{details['Variable (Y)']}'")
        print(f"    - Type:       {details['Problem Type']}")
        print(f"    - Objective:  {details['Business Objective']}")
        print(f"    - Predictors: {', '.join(details['Predictors (X)'])}")
        
    print("=" * 75)
    return target_mappings


def check_statistical_assumptions(df: pd.DataFrame) -> None:
    """
    Master runner to execute all core statistical diagnostic tests.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        check_normality(df, columns=numeric_cols)
        check_multicollinearity(df, columns=numeric_cols)
    identify_target_variable(df)


if __name__ == "__main__":
    from data_loader import load_dataset, handle_duplicates
    from missing_values import handle_missing_values
    
    df = load_dataset()
    df = handle_duplicates(df)
    df = handle_missing_values(df)
    
    check_statistical_assumptions(df)


"""
Exploratory Data Analysis (EDA) Module
Generates comprehensive descriptive statistics, category distribution reports, and bivariate business insights.
"""

import numpy as np
import pandas as pd


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute enhanced numerical summary statistics including central tendency,
    dispersion, skewness, kurtosis, and IQR.
    """
    print("\n" + "=" * 70)
    print("EDA: COMPREHENSIVE NUMERICAL DESCRIPTIVE STATISTICS")
    print("=" * 70)
    
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("[INFO] No numerical columns found.")
        return pd.DataFrame()
        
    stats_dict = {
        "Count": numeric_df.count(),
        "Mean": numeric_df.mean().round(2),
        "Std Dev": numeric_df.std().round(2),
        "Min": numeric_df.min().round(2),
        "25% (Q1)": numeric_df.quantile(0.25).round(2),
        "50% (Median)": numeric_df.median().round(2),
        "75% (Q3)": numeric_df.quantile(0.75).round(2),
        "Max": numeric_df.max().round(2),
        "IQR": (numeric_df.quantile(0.75) - numeric_df.quantile(0.25)).round(2),
        "Skewness": numeric_df.skew().round(2),
        "Kurtosis": numeric_df.kurtosis().round(2)
    }
    
    summary_table = pd.DataFrame(stats_dict)
    print(summary_table)
    print("=" * 70)
    return summary_table


def analyze_categorical_distributions(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    """
    Analyze frequency counts and percentages for nominal and ordinal categorical attributes.
    """
    print("\n" + "=" * 70)
    print("EDA: CATEGORICAL ATTRIBUTE DISTRIBUTIONS")
    print("=" * 70)
    
    if columns is None:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        # Filter out high-cardinality IDs for readable terminal summary
        columns = [c for c in columns if c not in ["OrderID", "CustomerID", "TrackingNumber"]]
        
    for col in columns:
        print(f"\n--- Distribution for '{col}' ---")
        counts = df[col].value_counts(dropna=False)
        pcts = (df[col].value_counts(dropna=False, normalize=True) * 100).round(2)
        dist_df = pd.DataFrame({"Count": counts, "Percentage (%)": pcts})
        print(dist_df)
        
    print("=" * 70)


def analyze_bivariate_relationships(df: pd.DataFrame) -> None:
    """
    Analyze bivariate relationships and domain business metrics.
    """
    print("\n" + "=" * 70)
    print("EDA: BIVARIATE & DOMAIN BUSINESS INSIGHTS")
    print("=" * 70)
    
    # 1. Total Revenue and Average Order Value by Product
    if "Product" in df.columns and "TotalPrice" in df.columns:
        print("\n--- Revenue & Order Metrics by Product Category ---")
        prod_metrics = df.groupby("Product").agg(
            Total_Orders=("OrderID", "count") if "OrderID" in df.columns else ("TotalPrice", "count"),
            Total_Revenue=("TotalPrice", "sum"),
            Avg_Order_Value=("TotalPrice", "mean"),
            Avg_Quantity=("Quantity", "mean") if "Quantity" in df.columns else ("TotalPrice", "mean")
        ).round(2).sort_values(by="Total_Revenue", ascending=False)
        print(prod_metrics)
        
    # 2. Performance by Referral Channel
    if "ReferralSource" in df.columns and "TotalPrice" in df.columns:
        print("\n--- Performance Metrics by Referral Source ---")
        referral_metrics = df.groupby("ReferralSource").agg(
            Orders=("TotalPrice", "count"),
            Total_Revenue=("TotalPrice", "sum"),
            Avg_Order_Value=("TotalPrice", "mean")
        ).round(2).sort_values(by="Total_Revenue", ascending=False)
        print(referral_metrics)
        
    # 3. Order Status Breakdown
    if "OrderStatus" in df.columns and "TotalPrice" in df.columns:
        print("\n--- Order Status vs. Revenue Impact ---")
        status_metrics = df.groupby("OrderStatus").agg(
            Order_Count=("TotalPrice", "count"),
            Total_Value=("TotalPrice", "sum")
        ).round(2).sort_values(by="Total_Value", ascending=False)
        print(status_metrics)
        
    print("=" * 70)


def run_eda(df: pd.DataFrame) -> None:
    """
    Master function to run full Exploratory Data Analysis suite.
    """
    compute_summary_statistics(df)
    analyze_categorical_distributions(df)
    analyze_bivariate_relationships(df)


if __name__ == "__main__":
    from data_loader import load_dataset, handle_duplicates
    from missing_values import handle_missing_values
    
    df = load_dataset()
    df = handle_duplicates(df)
    df = handle_missing_values(df)
    run_eda(df)

"""
Outliers Detection, Transformation & Handling Module
Provides IQR and Z-Score outlier detection, outlier row extraction,
log transformation (log(1 + x)), distribution comparisons, and robust handling (capping / trimming / transformation).
"""

import os
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for reliable headless plot saving
import matplotlib.pyplot as plt


def get_outlier_rows_iqr(
    df: pd.DataFrame, 
    columns: list[str] | None = None, 
    factor: float = 1.5,
    per_column: bool = False
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Extract the actual rows containing outliers detected using the Interquartile Range (IQR) method.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list[str] | None): Numerical columns to evaluate (defaults to all numeric).
        factor (float): IQR multiplier (default is 1.5 for standard outliers, 3.0 for extreme).
        per_column (bool): If True, returns a dictionary {column_name: DataFrame_of_outliers}.
                           If False, returns a consolidated DataFrame of all rows with at least one outlier,
                           including an 'Outlier_Reason' diagnostic column.
                           
    Returns:
        pd.DataFrame | dict[str, pd.DataFrame]: Outlier rows.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    outliers_dict = {}
    all_outlier_indices = set()
    bounds = {}
    
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (factor * iqr)
        upper_bound = q3 + (factor * iqr)
        bounds[col] = (lower_bound, upper_bound)
        
        col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].copy()
        if not col_outliers.empty:
            outliers_dict[col] = col_outliers
            all_outlier_indices.update(col_outliers.index)
            
    if per_column:
        return outliers_dict
        
    if not all_outlier_indices:
        return pd.DataFrame(columns=df.columns)
        
    # Return sorted subset of rows with diagnostic reason annotation
    outlier_df = df.loc[sorted(all_outlier_indices)].copy()
    
    outlier_reasons = []
    for idx in outlier_df.index:
        reasons = []
        for col in columns:
            lower_bound, upper_bound = bounds[col]
            val = outlier_df.loc[idx, col]
            if val < lower_bound:
                reasons.append(f"{col}={val:.2f} (< lower bound {lower_bound:.2f})")
            elif val > upper_bound:
                reasons.append(f"{col}={val:.2f} (> upper bound {upper_bound:.2f})")
        outlier_reasons.append("; ".join(reasons))
        
    outlier_df["Outlier_Reason"] = outlier_reasons
    return outlier_df


def get_outliers_iqr(
    df: pd.DataFrame, 
    columns: list[str] | None = None, 
    factor: float = 1.5,
    per_column: bool = False
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Alias for get_outlier_rows_iqr."""
    return get_outlier_rows_iqr(df, columns=columns, factor=factor, per_column=per_column)


def detect_outliers_iqr(
    df: pd.DataFrame, 
    columns: list[str] | None = None, 
    factor: float = 1.5,
    show_rows: bool = True
) -> pd.DataFrame:
    """
    Detect outliers in numerical features using the Interquartile Range (IQR) method
    and optionally display the detected outlier rows.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list[str] | None): Numerical columns to evaluate (defaults to all numeric).
        factor (float): IQR multiplier (default is 1.5 for standard outliers, 3.0 for extreme).
        show_rows (bool): If True, prints the detected outlier records.
        
    Returns:
        pd.DataFrame: Summary table of detected outliers per column.
    """
    print("\n" + "=" * 70)
    print(f"OUTLIER DETECTION (INTERQUARTILE RANGE - IQR METHOD, FACTOR={factor})")
    print("=" * 70)
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    results = []
    
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (factor * iqr)
        upper_bound = q3 + (factor * iqr)
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100
        
        results.append({
            "Feature": col,
            "Q1 (25%)": round(q1, 2),
            "Q3 (75%)": round(q3, 2),
            "IQR": round(iqr, 2),
            "Lower Bound": round(lower_bound, 2),
            "Upper Bound": round(upper_bound, 2),
            "Outlier Count": outlier_count,
            "Outlier (%)": f"{outlier_pct:.2f}%"
        })
        
    summary_df = pd.DataFrame(results).set_index("Feature")
    print(summary_df)
    print("=" * 70)
    
    if show_rows:
        outlier_rows_by_col = get_outlier_rows_iqr(df, columns=columns, factor=factor, per_column=True)
        if outlier_rows_by_col:
            print("\n>>> DETECTED OUTLIER ROWS (IQR METHOD):")
            for col, rows in outlier_rows_by_col.items():
                print(f"\n--- Outlier Records for Feature '{col}' ({len(rows)} rows) ---")
                display_cols = [c for c in ["OrderID", "Date", "Product", "Quantity", "UnitPrice", "TotalPrice", col] if c in rows.columns]
                display_cols = list(dict.fromkeys(display_cols))
                print(rows[display_cols].to_string())
            print("=" * 70)
        else:
            print("[INFO] No outlier rows detected across evaluated numerical features.")
            print("=" * 70)
            
    return summary_df


def get_outlier_rows_zscore(
    df: pd.DataFrame, 
    columns: list[str] | None = None, 
    threshold: float = 3.0,
    per_column: bool = False
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Extract the actual rows containing outliers detected using the Z-Score method (|Z| > threshold).
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list[str] | None): Numerical columns to evaluate.
        threshold (float): Z-score threshold (default is 3.0 standard deviations).
        per_column (bool): If True, returns a dict of {column: DataFrame of outlier rows}.
                           If False, returns a consolidated DataFrame of all outlier rows.
                           
    Returns:
        pd.DataFrame | dict[str, pd.DataFrame]: Outlier rows.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    outliers_dict = {}
    all_outlier_indices = set()
    
    for col in columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        if std_val == 0:
            continue
            
        z_scores = (df[col] - mean_val) / std_val
        col_outliers = df[z_scores.abs() > threshold].copy()
        if not col_outliers.empty:
            outliers_dict[col] = col_outliers
            all_outlier_indices.update(col_outliers.index)
            
    if per_column:
        return outliers_dict
        
    if not all_outlier_indices:
        return pd.DataFrame(columns=df.columns)
        
    outlier_df = df.loc[sorted(all_outlier_indices)].copy()
    return outlier_df


def detect_outliers_zscore(
    df: pd.DataFrame, 
    columns: list[str] | None = None, 
    threshold: float = 3.0,
    show_rows: bool = False
) -> pd.DataFrame:
    """
    Detect outliers in numerical features using the Z-Score method (|Z| > threshold).
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list[str] | None): Numerical columns to evaluate.
        threshold (float): Z-score threshold (default is 3.0 standard deviations).
        show_rows (bool): If True, prints the detected outlier records.
        
    Returns:
        pd.DataFrame: Summary table of detected outliers per column.
    """
    print("\n" + "=" * 70)
    print(f"OUTLIER DETECTION (Z-SCORE METHOD, THRESHOLD=|{threshold}|)")
    print("=" * 70)
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    results = []
    
    for col in columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        if std_val == 0:
            continue
            
        z_scores = (df[col] - mean_val) / std_val
        outliers = df[z_scores.abs() > threshold]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100
        
        results.append({
            "Feature": col,
            "Mean": round(mean_val, 2),
            "Std Dev": round(std_val, 2),
            "Outlier Count": outlier_count,
            "Outlier (%)": f"{outlier_pct:.2f}%"
        })
        
    summary_df = pd.DataFrame(results).set_index("Feature")
    print(summary_df)
    print("=" * 70)
    
    if show_rows:
        outlier_rows_by_col = get_outlier_rows_zscore(df, columns=columns, threshold=threshold, per_column=True)
        if outlier_rows_by_col:
            print("\n>>> DETECTED OUTLIER ROWS (Z-SCORE METHOD):")
            for col, rows in outlier_rows_by_col.items():
                print(f"\n--- Outlier Records for Feature '{col}' ({len(rows)} rows) ---")
                display_cols = [c for c in ["OrderID", "Date", "Product", "Quantity", "UnitPrice", "TotalPrice", col] if c in rows.columns]
                display_cols = list(dict.fromkeys(display_cols))
                print(rows[display_cols].to_string())
            print("=" * 70)
        else:
            print("[INFO] No outlier rows detected via Z-Score method.")
            print("=" * 70)
            
    return summary_df


def apply_log_transformation(
    df: pd.DataFrame, 
    source_column: str = "TotalPrice", 
    target_column: str = "Log_TotalPrice"
) -> pd.DataFrame:
    """
    Apply log(1 + x) transformation to a specified column while keeping the original column unchanged.
    No outliers are deleted or capped.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        source_column (str): Original column name (default 'TotalPrice').
        target_column (str): New transformed column name (default 'Log_TotalPrice').
        
    Returns:
        pd.DataFrame: DataFrame containing both the original and transformed columns.
    """
    print("\n" + "=" * 70)
    print(f"APPLYING LOG TRANSFORMATION: {target_column} = log(1 + {source_column})")
    print("=" * 70)
    
    df_transformed = df.copy()
    df_transformed[target_column] = np.log1p(df_transformed[source_column])
    
    print(f"[TRANSFORMATION] Created new column '{target_column}' using np.log1p({source_column}).")
    print(f"[INFO] Original '{source_column}' column is preserved unchanged.")
    print(f"[INFO] No outlier rows were deleted or capped.")
    print("=" * 70)
    return df_transformed


def compare_distributions(
    df: pd.DataFrame, 
    col1: str = "TotalPrice", 
    col2: str = "Log_TotalPrice",
    save_plot: bool = True,
    plot_path: str = os.path.join("figures", "distribution_comparison.png")
) -> pd.DataFrame:
    """
    Compare the statistical distributions of the original and log-transformed columns
    using summary statistics, ASCII distribution bins, and side-by-side histogram plots.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing both columns.
        col1 (str): Original column name (default 'TotalPrice').
        col2 (str): Transformed column name (default 'Log_TotalPrice').
        save_plot (bool): Whether to generate and save histogram plot images (default True).
        plot_path (str): Filepath to save the comparison plot image.
        
    Returns:
        pd.DataFrame: Summary statistics comparison table.
    """
    print("\n" + "=" * 70)
    print(f"DISTRIBUTION COMPARISON: '{col1}' vs. '{col2}'")
    print("=" * 70)
    
    metrics = {
        "Count": [len(df[col1]), len(df[col2])],
        "Mean": [df[col1].mean(), df[col2].mean()],
        "Std Dev": [df[col1].std(), df[col2].std()],
        "Median (50%)": [df[col1].median(), df[col2].median()],
        "IQR": [df[col1].quantile(0.75) - df[col1].quantile(0.25), df[col2].quantile(0.75) - df[col2].quantile(0.25)],
        "Min": [df[col1].min(), df[col2].min()],
        "25% (Q1)": [df[col1].quantile(0.25), df[col2].quantile(0.25)],
        "75% (Q3)": [df[col1].quantile(0.75), df[col2].quantile(0.75)],
        "Max": [df[col1].max(), df[col2].max()],
        "Skewness": [df[col1].skew(), df[col2].skew()],
        "Kurtosis": [df[col1].kurtosis(), df[col2].kurtosis()]
    }
    
    comp_df = pd.DataFrame(metrics, index=[col1, col2]).T
    comp_df["Change / Interpretation"] = [
        "Identical (No rows deleted)",
        f"{comp_df.loc['Mean', col1]:.2f} -> {comp_df.loc['Mean', col2]:.2f}",
        f"{comp_df.loc['Std Dev', col1]:.2f} -> {comp_df.loc['Std Dev', col2]:.2f} (Variance Stabilized)",
        f"{comp_df.loc['Median (50%)', col1]:.2f} -> {comp_df.loc['Median (50%)', col2]:.2f}",
        f"{comp_df.loc['IQR', col1]:.2f} -> {comp_df.loc['IQR', col2]:.2f} (Dispersion Scaled)",
        f"{comp_df.loc['Min', col1]:.2f} -> {comp_df.loc['Min', col2]:.2f}",
        f"{comp_df.loc['25% (Q1)', col1]:.2f} -> {comp_df.loc['25% (Q1)', col2]:.2f}",
        f"{comp_df.loc['75% (Q3)', col1]:.2f} -> {comp_df.loc['75% (Q3)', col2]:.2f}",
        f"{comp_df.loc['Max', col1]:.2f} -> {comp_df.loc['Max', col2]:.2f} (Compresses extreme right tail)",
        f"Skewness: {comp_df.loc['Skewness', col1]:.2f} (Right-skewed) -> {comp_df.loc['Skewness', col2]:.2f} (Normalized)",
        f"Kurtosis: {comp_df.loc['Kurtosis', col1]:.2f} -> {comp_df.loc['Kurtosis', col2]:.2f}"
    ]
    
    print("\n--- Summary Statistics Comparison ---")
    formatted_comp = comp_df.copy()
    for c in [col1, col2]:
        formatted_comp[c] = formatted_comp[c].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x))
    print(formatted_comp.to_string())
    print("=" * 70)
    
    # Text-based Frequency Bins (Histograms in Terminal)
    print(f"\n--- Frequency Distribution (Histogram Bins for '{col1}') ---")
    counts1, bin_edges1 = np.histogram(df[col1].dropna(), bins=10)
    for i in range(len(counts1)):
        bar = "#" * int(counts1[i] / (len(df) / 50))
        print(f"[{bin_edges1[i]:8.2f} - {bin_edges1[i+1]:8.2f}]: {counts1[i]:4d} ({counts1[i]/len(df)*100:5.1f}%) | {bar}")
        
    print(f"\n--- Frequency Distribution (Histogram Bins for '{col2}') ---")
    counts2, bin_edges2 = np.histogram(df[col2].dropna(), bins=10)
    for i in range(len(counts2)):
        bar = "#" * int(counts2[i] / (len(df) / 50))
        print(f"[{bin_edges2[i]:8.2f} - {bin_edges2[i+1]:8.2f}]: {counts2[i]:4d} ({counts2[i]/len(df)*100:5.1f}%) | {bar}")
    print("=" * 70)
    
    # Graphical Histograms with Matplotlib
    if save_plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Subplot 1: Original TotalPrice
        axes[0].hist(df[col1], bins=30, color="#2b5c8f", edgecolor="black", alpha=0.75, density=True)
        mean_val1 = df[col1].mean()
        med_val1 = df[col1].median()
        axes[0].axvline(mean_val1, color="red", linestyle="--", linewidth=1.8, label=f"Mean: {mean_val1:.2f}")
        axes[0].axvline(med_val1, color="green", linestyle="-.", linewidth=1.8, label=f"Median: {med_val1:.2f}")
        axes[0].set_title(f"Original Distribution: {col1}\n(Skewness = {df[col1].skew():.2f})", fontsize=12, fontweight="bold")
        axes[0].set_xlabel(f"{col1} ($)", fontsize=10)
        axes[0].set_ylabel("Density", fontsize=10)
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend()
        
        # Subplot 2: Log_TotalPrice
        axes[1].hist(df[col2], bins=30, color="#2e8b57", edgecolor="black", alpha=0.75, density=True)
        mean_val2 = df[col2].mean()
        med_val2 = df[col2].median()
        axes[1].axvline(mean_val2, color="red", linestyle="--", linewidth=1.8, label=f"Mean: {mean_val2:.2f}")
        axes[1].axvline(med_val2, color="green", linestyle="-.", linewidth=1.8, label=f"Median: {med_val2:.2f}")
        axes[1].set_title(f"Transformed Distribution: {col2}\n(Skewness = {df[col2].skew():.2f})", fontsize=12, fontweight="bold")
        axes[1].set_xlabel(f"{col2} (log scale)", fontsize=10)
        axes[1].set_ylabel("Density", fontsize=10)
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend()
        
        plt.suptitle("Comparison of Original vs. Log-Transformed Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        plot_dir = os.path.dirname(plot_path)
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
            
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[VISUALIZATION] Distribution comparison histogram saved to: '{plot_path}'")
        print("=" * 70)
        
    return comp_df


def handle_outliers(
    df: pd.DataFrame, 
    columns: list[str] | None = None, 
    method: Literal["cap", "trim", "log", "none"] = "none",
    factor: float = 1.5
) -> pd.DataFrame:
    """
    Treat detected outliers using Winsorization (capping), trimming, or log transformation.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list[str] | None): Columns to treat.
        method (str): 'none' (preserve raw values), 'log' (apply log1p), 'cap' (Winsorize to IQR bounds), or 'trim' (drop rows).
        factor (float): IQR factor used for boundary determination.
        
    Returns:
        pd.DataFrame: DataFrame with treated/transformed outliers.
    """
    print("\n" + "=" * 70)
    print(f"OUTLIER HANDLING (STRATEGY: {method.upper()})")
    print("=" * 70)
    
    if method == "none":
        print("[INFO] Outlier handling set to 'none'. Preserving raw distributions without deleting or capping.")
        print("=" * 70)
        return df
        
    if method == "log":
        df_treated = df.copy()
        target_cols = columns if columns is not None else ["TotalPrice"]
        for col in target_cols:
            if col in df_treated.columns and pd.api.types.is_numeric_dtype(df_treated[col]):
                new_col = f"Log_{col}"
                df_treated[new_col] = np.log1p(df_treated[col])
                print(f"[LOG TRANSFORMATION] Created '{new_col}' = log(1 + {col}) without capping or dropping.")
        print("=" * 70)
        return df_treated
        
    df_treated = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    for col in columns:
        q1 = df_treated[col].quantile(0.25)
        q3 = df_treated[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (factor * iqr)
        upper_bound = q3 + (factor * iqr)
        
        outlier_mask = (df_treated[col] < lower_bound) | (df_treated[col] > upper_bound)
        count = outlier_mask.sum()
        
        if count == 0:
            print(f"'{col}': 0 outliers found within [{lower_bound:.2f}, {upper_bound:.2f}].")
            continue
            
        if method == "cap":
            df_treated[col] = np.clip(df_treated[col], lower_bound, upper_bound)
            print(f"[CAPPING] '{col}': Capped {count} outliers to boundaries [{lower_bound:.2f}, {upper_bound:.2f}]")
        elif method == "trim":
            df_treated = df_treated[~outlier_mask]
            print(f"[TRIMMING] '{col}': Removed {count} outlier rows.")
            
    print(f"Final shape after outlier handling: {df_treated.shape}")
    print("=" * 70)
    return df_treated


if __name__ == "__main__":
    from data_loader import load_dataset, handle_duplicates
    from missing_values import handle_missing_values
    
    # 1. Load and prepare clean dataset
    df = load_dataset()
    df = handle_duplicates(df)
    df = handle_missing_values(df)
    
    numeric_cols = ["UnitPrice", "Quantity", "ItemsInCart", "TotalPrice"]
    
    # 2. Outlier detection (IQR & Z-Score) and inspection of outlier rows
    detect_outliers_iqr(df, columns=numeric_cols, factor=1.5, show_rows=True)
    detect_outliers_zscore(df, columns=numeric_cols, threshold=3.0)
    
    # 3. Create Log_TotalPrice using log(1 + TotalPrice) without capping/deleting any outliers
    df_transformed = apply_log_transformation(df, source_column="TotalPrice", target_column="Log_TotalPrice")
    
    # 4. Preview the first few rows with the new column
    print("\n>>> PREVIEW FIRST FEW ROWS (TotalPrice & Log_TotalPrice):")
    preview_cols = ["OrderID", "Product", "Quantity", "UnitPrice", "TotalPrice", "Log_TotalPrice"]
    print(df_transformed[preview_cols].head(10).to_string())
    print("=" * 70)
    
    # 5. Compare distributions using summary statistics and histograms
    compare_distributions(
        df_transformed, 
        col1="TotalPrice", 
        col2="Log_TotalPrice", 
        save_plot=True, 
        plot_path=os.path.join("figures", "distribution_comparison.png")
    )

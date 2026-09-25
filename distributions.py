"""
Variable Distribution Analysis Module
Evaluates and visualizes the statistical distributions of target dataset variables:
- Numerical: UnitPrice, TotalPrice, ItemsInCart
- Categorical: Product, PaymentMethod, OrderStatus, CouponCode, ReferralSource
- Temporal: Date (Monthly Orders distribution over time)
- Grouped Numerical: UnitPrice across each Product category
- Monthly Distribution: TotalPrice across each month

All generated distribution plots are exported to the 'figures/' directory.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plot generation
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# List of target variables as specified
TARGET_NUMERIC_VARS = ["UnitPrice", "TotalPrice", "ItemsInCart"]
TARGET_CATEGORICAL_VARS = ["Product", "PaymentMethod", "OrderStatus", "CouponCode", "ReferralSource"]
TARGET_DATE_VAR = "Date"


def ensure_figures_dir(plot_path: str) -> None:
    """Ensure parent directory for saving plots exists."""
    plot_dir = os.path.dirname(plot_path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)


def analyze_numeric_distributions(
    df: pd.DataFrame, 
    columns: list[str] | None = None,
    save_plot: bool = True,
    plot_path: str = os.path.join("figures", "numeric_distributions.png")
) -> pd.DataFrame:
    """
    Compute statistical distribution metrics and generate histogram/boxplot visualizations
    for target numerical variables (UnitPrice, TotalPrice, ItemsInCart).
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list[str] | None): Numerical columns to analyze.
        save_plot (bool): Whether to generate and save figure.
        plot_path (str): Filepath to save the numerical distribution plot.
        
    Returns:
        pd.DataFrame: Summary statistics table.
    """
    if columns is None:
        columns = [c for c in TARGET_NUMERIC_VARS if c in df.columns]
        
    print("\n" + "=" * 80)
    print("NUMERICAL VARIABLE DISTRIBUTIONS: " + ", ".join(columns))
    print("=" * 80)
    
    stats_records = []
    for col in columns:
        s = df[col].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        mean_val = s.mean()
        std_val = s.std()
        median_val = s.median()
        skew_val = s.skew()
        kurt_val = s.kurtosis()
        
        # Skewness classification
        if abs(skew_val) < 0.5:
            skew_desc = "Approx. Symmetric"
        elif skew_val >= 0.5:
            skew_desc = "Right-Skewed (Positive)"
        else:
            skew_desc = "Left-Skewed (Negative)"
            
        stats_records.append({
            "Variable": col,
            "Count": len(s),
            "Mean": round(mean_val, 2),
            "Std Dev": round(std_val, 2),
            "Min": round(s.min(), 2),
            "25% (Q1)": round(q1, 2),
            "50% (Median)": round(median_val, 2),
            "75% (Q3)": round(q3, 2),
            "Max": round(s.max(), 2),
            "IQR": round(iqr, 2),
            "Skewness": round(skew_val, 3),
            "Kurtosis": round(kurt_val, 3),
            "Distribution Shape": skew_desc
        })
        
    stats_df = pd.DataFrame(stats_records).set_index("Variable")
    print(stats_df.to_string())
    print("=" * 80)
    
    # Terminal ASCII Histograms
    for col in columns:
        print(f"\n--- Frequency Bins for '{col}' ---")
        counts, bin_edges = np.histogram(df[col].dropna(), bins=8)
        for i in range(len(counts)):
            bar = "#" * int(counts[i] / (len(df[col].dropna()) / 40))
            pct = (counts[i] / len(df[col].dropna())) * 100
            print(f"[{bin_edges[i]:8.2f} - {bin_edges[i+1]:8.2f}]: {counts[i]:4d} ({pct:5.1f}%) | {bar}")
            
    # Graphical Visualization
    if save_plot and columns:
        ensure_figures_dir(plot_path)
        num_cols = len(columns)
        fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 8))
        
        # Format axes as 2D array even if single column
        if num_cols == 1:
            axes = np.array([[axes[0]], [axes[1]]])
            
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        
        for i, col in enumerate(columns):
            color = palette[i % len(palette)]
            data = df[col].dropna()
            
            # Row 1: Histogram & Density
            ax_hist = axes[0, i]
            ax_hist.hist(data, bins=25, color=color, edgecolor="black", alpha=0.7, density=True)
            mean_v = data.mean()
            med_v = data.median()
            ax_hist.axvline(mean_v, color="#d62728", linestyle="--", linewidth=1.8, label=f"Mean: {mean_v:.1f}")
            ax_hist.axvline(med_v, color="#2ca02c", linestyle="-.", linewidth=1.8, label=f"Median: {med_v:.1f}")
            ax_hist.set_title(f"Histogram: {col}\n(Skew: {data.skew():.2f}, Kurt: {data.kurtosis():.2f})", fontsize=11, fontweight="bold")
            ax_hist.set_xlabel(col, fontsize=10)
            ax_hist.set_ylabel("Density", fontsize=10)
            ax_hist.grid(True, linestyle=":", alpha=0.6)
            ax_hist.legend(fontsize=9)
            
            # Row 2: Boxplot
            ax_box = axes[1, i]
            box_props = dict(patch_artist=True, boxprops=dict(facecolor=color, alpha=0.6), medianprops=dict(color="black", linewidth=2))
            ax_box.boxplot(data, vert=False, **box_props)
            ax_box.set_title(f"Box Plot: {col}", fontsize=11, fontweight="bold")
            ax_box.set_xlabel(col, fontsize=10)
            ax_box.grid(True, linestyle=":", alpha=0.6)
            
        plt.suptitle("Statistical Distributions of Numerical Variables", fontsize=14, fontweight="bold", y=1.00)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\n[FIGURE SAVED] Numerical distribution plots saved to: '{plot_path}'")
        print("=" * 80)
        
    return stats_df


def analyze_categorical_distributions(
    df: pd.DataFrame, 
    columns: list[str] | None = None,
    save_plot: bool = True,
    plot_path: str = os.path.join("figures", "categorical_distributions.png")
) -> dict[str, pd.DataFrame]:
    """
    Analyze frequency distributions, proportions, and dominant categories for target categorical variables
    (Product, PaymentMethod, OrderStatus, CouponCode, ReferralSource).
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list[str] | None): Categorical columns to analyze.
        save_plot (bool): Whether to generate and save figure.
        plot_path (str): Filepath to save the categorical distribution plot.
        
    Returns:
        dict[str, pd.DataFrame]: Dictionary mapping variable names to distribution tables.
    """
    if columns is None:
        columns = [c for c in TARGET_CATEGORICAL_VARS if c in df.columns]
        
    print("\n" + "=" * 80)
    print("CATEGORICAL VARIABLE DISTRIBUTIONS: " + ", ".join(columns))
    print("=" * 80)
    
    dist_tables = {}
    for col in columns:
        # Handle CouponCode missing values explicitly for clear distribution view
        series = df[col].copy()
        if series.isnull().any():
            series = series.fillna("Missing (No Coupon)")
            
        counts = series.value_counts(dropna=False)
        pcts = (series.value_counts(dropna=False, normalize=True) * 100).round(2)
        cum_pcts = pcts.cumsum().round(2)
        
        dist_df = pd.DataFrame({
            "Frequency": counts,
            "Percentage (%)": pcts,
            "Cumulative (%)": cum_pcts
        })
        dist_tables[col] = dist_df
        
        print(f"\n--- Distribution for '{col}' (Cardinality: {series.nunique()}) ---")
        print(dist_df.to_string())
        
    print("=" * 80)
    
    # Graphical Visualization
    if save_plot and columns:
        ensure_figures_dir(plot_path)
        num_cols = len(columns)
        n_rows = (num_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
        axes = axes.flatten()
        
        colors = ["#3470a3", "#e67e22", "#27ae60", "#9b59b6", "#e74c3c", "#1abc9c", "#34495e", "#f39c12"]
        
        for idx, col in enumerate(columns):
            ax = axes[idx]
            series = df[col].copy()
            if series.isnull().any():
                series = series.fillna("Missing (No Coupon)")
                
            val_counts = series.value_counts()
            bars = ax.bar(val_counts.index.astype(str), val_counts.values, color=colors[:len(val_counts)], edgecolor="black", alpha=0.8)
            
            # Value & percentage labels on top of bars
            total = len(series)
            for bar in bars:
                height = bar.get_height()
                pct = (height / total) * 100
                ax.annotate(f"{height}\n({pct:.1f}%)",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, fontweight="bold")
                            
            ax.set_title(f"Distribution: {col}", fontsize=11, fontweight="bold")
            ax.set_ylabel("Order Count", fontsize=10)
            ax.grid(axis="y", linestyle=":", alpha=0.6)
            ax.set_ylim(0, val_counts.max() * 1.22)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
            
        # Hide any unused subplots
        for j in range(num_cols, len(axes)):
            fig.delaxes(axes[j])
            
        plt.suptitle("Frequency Distributions of Categorical Variables", fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\n[FIGURE SAVED] Categorical distribution plots saved to: '{plot_path}'")
        print("=" * 80)
        
    return dist_tables


def analyze_temporal_distribution(
    df: pd.DataFrame, 
    date_column: str = TARGET_DATE_VAR,
    save_plot: bool = True,
    plot_path: str = os.path.join("figures", "monthly_orders_distribution.png")
) -> pd.DataFrame:
    """
    Analyze the temporal distribution of monthly transaction order volumes over time.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        date_column (str): Name of the Date column.
        save_plot (bool): Whether to generate and save figure.
        plot_path (str): Filepath to save the monthly orders distribution plot.
        
    Returns:
        pd.DataFrame: Monthly transaction frequency table.
    """
    if date_column not in df.columns:
        print(f"[WARNING] Date column '{date_column}' not found in DataFrame.")
        return pd.DataFrame()
        
    print("\n" + "=" * 80)
    print(f"TEMPORAL DISTRIBUTION: MONTHLY ORDERS ('{date_column}')")
    print("=" * 80)
    
    date_series = pd.to_datetime(df[date_column])
    start_date = date_series.min()
    end_date = date_series.max()
    span_days = (end_date - start_date).days
    
    print(f"Earliest Transaction Date: {start_date.strftime('%Y-%m-%d')}")
    print(f"Latest Transaction Date:   {end_date.strftime('%Y-%m-%d')}")
    print(f"Total Date Span:           {span_days} days (~{span_days / 30.4:.1f} months / {span_days / 365.25:.2f} years)")
    print(f"Total Transactions:        {len(date_series):,}")
    
    # Monthly Distribution
    monthly_counts = date_series.dt.to_period("M").value_counts().sort_index()
    monthly_pcts = (monthly_counts / len(date_series) * 100).round(2)
    cum_pcts = monthly_pcts.cumsum().round(2)
    
    monthly_df = pd.DataFrame({
        "Period": [str(p) for p in monthly_counts.index],
        "Order_Count": monthly_counts.values,
        "Percentage (%)": monthly_pcts.values,
        "Cumulative (%)": cum_pcts.values
    }).set_index("Period")
    
    print("\n--- Monthly Transaction Orders Distribution ---")
    print(monthly_df.to_string())
    print("=" * 80)
    
    # Graphical Visualization
    if save_plot:
        ensure_figures_dir(plot_path)
        fig, ax = plt.subplots(figsize=(15, 6))
        
        periods_str = [str(p) for p in monthly_counts.index]
        bars = ax.bar(periods_str, monthly_counts.values, color="#2b5c8f", edgecolor="black", alpha=0.75, width=0.7)
        ax.plot(periods_str, monthly_counts.values, color="#e74c3c", marker="o", linewidth=2.2, markersize=6, label="Monthly Trend")
        
        # Add count and percentage labels above bars
        total = len(date_series)
        mean_monthly = monthly_counts.mean()
        ax.axhline(mean_monthly, color="#27ae60", linestyle="--", linewidth=1.8, label=f"Average: {mean_monthly:.1f} orders/mo")
        
        for bar in bars:
            h = bar.get_height()
            p = (h / total) * 100
            ax.annotate(f"{h}\n({p:.1f}%)",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
                        
        ax.set_title("Monthly Order Volume Distribution (2023 - 2025)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Year-Month", fontsize=11)
        ax.set_ylabel("Number of Orders", fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.set_ylim(0, monthly_counts.max() * 1.25)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        ax.legend(fontsize=10, loc="upper left")
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\n[FIGURE SAVED] Monthly orders distribution plot saved to: '{plot_path}'")
        print("=" * 80)
        
    return monthly_df


def analyze_unitprice_by_product_distribution(
    df: pd.DataFrame, 
    product_col: str = "Product", 
    price_col: str = "UnitPrice",
    save_plot: bool = True,
    plot_path: str = os.path.join("figures", "unitprice_by_product_distribution.png")
) -> pd.DataFrame:
    """
    Analyze and compare the statistical distribution of UnitPrice across each Product category.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        product_col (str): Column name for Product categories.
        price_col (str): Column name for UnitPrice.
        save_plot (bool): Whether to generate and save figure.
        plot_path (str): Filepath to save the UnitPrice by Product distribution plot.
        
    Returns:
        pd.DataFrame: Summary statistics table of UnitPrice grouped by Product.
    """
    if product_col not in df.columns or price_col not in df.columns:
        print(f"[WARNING] Columns '{product_col}' or '{price_col}' not found in DataFrame.")
        return pd.DataFrame()
        
    print("\n" + "=" * 80)
    print(f"NUMERICAL VARIABLE DISTRIBUTIONS: '{price_col}' ACROSS EACH '{product_col}'")
    print("=" * 80)
    
    records = []
    products = df[product_col].dropna().unique()
    
    for prod in sorted(products):
        subset = df[df[product_col] == prod][price_col].dropna()
        q1 = subset.quantile(0.25)
        q3 = subset.quantile(0.75)
        iqr = q3 - q1
        mean_v = subset.mean()
        std_v = subset.std()
        med_v = subset.median()
        skew_v = subset.skew()
        kurt_v = subset.kurtosis()
        
        # Distribution shape assessment
        if abs(skew_v) < 0.5:
            shape_desc = "Approx. Symmetric"
        elif skew_v >= 0.5:
            shape_desc = "Right-Skewed"
        else:
            shape_desc = "Left-Skewed"
            
        records.append({
            "Product": prod,
            "Count": len(subset),
            "Mean ($)": round(mean_v, 2),
            "Std Dev ($)": round(std_v, 2),
            "Min ($)": round(subset.min(), 2),
            "25% Q1 ($)": round(q1, 2),
            "50% Median ($)": round(med_v, 2),
            "75% Q3 ($)": round(q3, 2),
            "Max ($)": round(subset.max(), 2),
            "IQR ($)": round(iqr, 2),
            "Skewness": round(skew_v, 3),
            "Kurtosis": round(kurt_v, 3),
            "Shape": shape_desc
        })
        
    stats_df = pd.DataFrame(records).set_index("Product")
    print(stats_df.to_string())
    print("=" * 80)
    
    # Terminal ASCII Frequency Bins per Product
    for prod in sorted(products):
        subset = df[df[product_col] == prod][price_col].dropna()
        counts, bin_edges = np.histogram(subset, bins=5)
        print(f"\n--- '{price_col}' Distribution Bins for Product: '{prod}' (n={len(subset)}) ---")
        for i in range(len(counts)):
            bar = "#" * int(counts[i] / (len(subset) / 30))
            pct = (counts[i] / len(subset)) * 100
            print(f"[{bin_edges[i]:8.2f} - {bin_edges[i+1]:8.2f}]: {counts[i]:3d} ({pct:5.1f}%) | {bar}")
            
    print("=" * 80)
    
    # Graphical Visualization: Individual Curve-type Distribution Graph for each Product
    if save_plot:
        ensure_figures_dir(plot_path)
        prod_list = sorted(products)
        num_prods = len(prod_list)
        
        # 4 rows x 2 columns grid for 7 products
        n_rows = 4
        n_cols = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))
        axes = axes.flatten()
        
        palette = ["#2b5c8f", "#d95f02", "#2e8b57", "#7570b3", "#e7298a", "#1b9e77", "#e6ab02", "#a6761d"]
        
        for idx, prod in enumerate(prod_list):
            ax = axes[idx]
            color = palette[idx % len(palette)]
            subset = df[df[product_col] == prod][price_col].dropna().values
            
            mean_v = subset.mean()
            med_v = float(np.median(subset))
            std_v = subset.std()
            skew_v = pd.Series(subset).skew()
            
            # 1. Background density histogram
            ax.hist(subset, bins=16, density=True, color=color, alpha=0.22, edgecolor="gray", linewidth=0.8, label="Binned Samples")
            
            # 2. Smooth Kernel Density Estimation (KDE) curve
            try:
                kde = gaussian_kde(subset)
                x_grid = np.linspace(max(0, subset.min() - 30), subset.max() + 30, 400)
                kde_curve = kde(x_grid)
                
                # Plot distribution curve
                ax.plot(x_grid, kde_curve, color=color, linewidth=2.6, label="Distribution Curve (KDE)")
                # Shaded area under curve
                ax.fill_between(x_grid, kde_curve, color=color, alpha=0.35)
            except Exception:
                pass
                
            # 3. Mean & Median Vertical Reference Lines
            ax.axvline(mean_v, color="#d62728", linestyle="--", linewidth=1.8, label=f"Mean: ${mean_v:.2f}")
            ax.axvline(med_v, color="#1e8449", linestyle="-.", linewidth=1.8, label=f"Median: ${med_v:.2f}")
            
            # Styling and annotations
            ax.set_title(f"Product: {prod} (n={len(subset)})\n[Mean: ${mean_v:.1f} | Med: ${med_v:.1f} | Skew: {skew_v:.2f}]", fontsize=11, fontweight="bold")
            ax.set_xlabel(f"{price_col} ($)", fontsize=10)
            ax.set_ylabel("Probability Density", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(fontsize=8.5, loc="upper right")
            ax.set_xlim(0, 750)
            
        # Hide any unused subplots (8th subplot)
        for j in range(num_prods, len(axes)):
            fig.delaxes(axes[j])
            
        plt.suptitle("Individual Unit Price Distribution Curves per Product Category", fontsize=15, fontweight="bold", y=0.995)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\n[FIGURE SAVED] UnitPrice by Product distribution curve plot saved to: '{plot_path}'")
        print("=" * 80)
        
    return stats_df


def analyze_totalprice_by_month_distribution(
    df: pd.DataFrame, 
    date_col: str = TARGET_DATE_VAR, 
    price_col: str = "TotalPrice",
    save_plot: bool = True,
    plot_path: str = os.path.join("figures", "totalprice_by_month_distribution.png")
) -> pd.DataFrame:
    """
    Analyze and visualize the statistical distribution of TotalPrice across each month.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        date_col (str): Column name for timestamps.
        price_col (str): Column name for invoice TotalPrice.
        save_plot (bool): Whether to generate and save figure.
        plot_path (str): Filepath to save the TotalPrice by month distribution plot.
        
    Returns:
        pd.DataFrame: Monthly TotalPrice summary statistics table.
    """
    if date_col not in df.columns or price_col not in df.columns:
        print(f"[WARNING] Columns '{date_col}' or '{price_col}' not found in DataFrame.")
        return pd.DataFrame()
        
    print("\n" + "=" * 80)
    print(f"NUMERICAL VARIABLE DISTRIBUTIONS: '{price_col}' ACROSS EACH MONTH ('{date_col}')")
    print("=" * 80)
    
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
    df_temp["YearMonth"] = df_temp[date_col].dt.to_period("M").astype(str)
    
    months = sorted(df_temp["YearMonth"].unique())
    records = []
    
    for m in months:
        subset = df_temp[df_temp["YearMonth"] == m][price_col].dropna()
        q1 = subset.quantile(0.25)
        q3 = subset.quantile(0.75)
        iqr = q3 - q1
        mean_v = subset.mean()
        std_v = subset.std()
        med_v = subset.median()
        skew_v = subset.skew()
        kurt_v = subset.kurtosis()
        total_rev = subset.sum()
        
        records.append({
            "Month": m,
            "Orders": len(subset),
            "Total Rev ($)": round(total_rev, 2),
            "Mean / AOV ($)": round(mean_v, 2),
            "Std Dev ($)": round(std_v, 2),
            "Min ($)": round(subset.min(), 2),
            "25% Q1 ($)": round(q1, 2),
            "50% Median ($)": round(med_v, 2),
            "75% Q3 ($)": round(q3, 2),
            "Max ($)": round(subset.max(), 2),
            "IQR ($)": round(iqr, 2),
            "Skewness": round(skew_v, 3) if not np.isnan(skew_v) else 0.0,
            "Kurtosis": round(kurt_v, 3) if not np.isnan(kurt_v) else 0.0
        })
        
    monthly_stats_df = pd.DataFrame(records).set_index("Month")
    print(monthly_stats_df.to_string())
    print("=" * 80)
    
    # Graphical Visualization: Monthly Total Prices Line Graph
    if save_plot:
        ensure_figures_dir(plot_path)
        fig, axes = plt.subplots(2, 1, figsize=(18, 12))
        
        data_by_month = [df_temp[df_temp["YearMonth"] == m][price_col].dropna().values for m in months]
        revs = np.array([d.sum() for d in data_by_month])
        means = np.array([d.mean() for d in data_by_month])
        x_indices = np.arange(len(months))
        
        # -------------------------------------------------------------
        # Panel 1: Monthly Total Invoiced Revenue Line Graph
        # -------------------------------------------------------------
        ax1 = axes[0]
        ax1.plot(x_indices, revs, color="#1f77b4", marker="o", markersize=6.5, linewidth=2.4, label="Monthly Total Revenue ($)")
        ax1.fill_between(x_indices, revs, color="#1f77b4", alpha=0.15)
        
        # Mean monthly revenue line
        mean_rev = np.mean(revs)
        ax1.axhline(mean_rev, color="#e74c3c", linestyle="--", linewidth=1.8, label=f"Average Monthly Revenue (${mean_rev:,.2f})")
        
        # Annotate peak and lowest revenue points
        max_rev_idx = np.argmax(revs)
        min_rev_idx = np.argmin(revs)
        ax1.scatter(x_indices[max_rev_idx], revs[max_rev_idx], color="#27ae60", s=80, zorder=5)
        ax1.annotate(f"Peak: ${revs[max_rev_idx]:,.2f} ({months[max_rev_idx]})",
                     xy=(x_indices[max_rev_idx], revs[max_rev_idx]),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold", color="#27ae60")
                     
        ax1.scatter(x_indices[min_rev_idx], revs[min_rev_idx], color="#c0392b", s=80, zorder=5)
        ax1.annotate(f"Min: ${revs[min_rev_idx]:,.2f} ({months[min_rev_idx]})",
                     xy=(x_indices[min_rev_idx], revs[min_rev_idx]),
                     xytext=(0, -14), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold", color="#c0392b")
                     
        for i, (x, y) in enumerate(zip(x_indices, revs)):
            if i % 2 == 0:
                ax1.annotate(f"${y/1000:.1f}k", (x, y), xytext=(0, 5), textcoords="offset points",
                             ha="center", fontsize=7.5, color="#333333")
                             
        ax1.set_title("Time Series Distribution of Monthly Total Revenue (Total Invoiced Sum)", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Year-Month Period", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Total Invoiced Amount ($)", fontsize=11, fontweight="bold")
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(months, rotation=45, ha="right", fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.55)
        ax1.set_ylim(0, max(revs) * 1.18)
        ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)
        
        # -------------------------------------------------------------
        # Panel 2: Monthly Average Total Price (AOV) Line Graph
        # -------------------------------------------------------------
        ax2 = axes[1]
        ax2.plot(x_indices, means, color="#27ae60", marker="s", markersize=6.5, linewidth=2.4, label="Monthly Average Order Value ($)")
        ax2.fill_between(x_indices, means, color="#27ae60", alpha=0.15)
        
        # Overall dataset mean line
        overall_mean_price = df_temp[price_col].mean()
        ax2.axhline(overall_mean_price, color="#2b5c8f", linestyle="--", linewidth=1.8, label=f"Overall Average Order Value (${overall_mean_price:.2f})")
        
        max_mean_idx = np.argmax(means)
        min_mean_idx = np.argmin(means)
        ax2.scatter(x_indices[max_mean_idx], means[max_mean_idx], color="#2980b9", s=80, zorder=5)
        ax2.annotate(f"Peak AOV: ${means[max_mean_idx]:.2f} ({months[max_mean_idx]})",
                     xy=(x_indices[max_mean_idx], means[max_mean_idx]),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold", color="#2980b9")
                     
        ax2.scatter(x_indices[min_mean_idx], means[min_mean_idx], color="#c0392b", s=80, zorder=5)
        ax2.annotate(f"Min AOV: ${means[min_mean_idx]:.2f} ({months[min_mean_idx]})",
                     xy=(x_indices[min_mean_idx], means[min_mean_idx]),
                     xytext=(0, -14), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold", color="#c0392b")
                     
        for i, (x, y) in enumerate(zip(x_indices, means)):
            if i % 2 == 0:
                ax2.annotate(f"${y:.0f}", (x, y), xytext=(0, 5), textcoords="offset points",
                             ha="center", fontsize=7.5, color="#333333")
                             
        ax2.set_title("Time Series Distribution of Monthly Average Total Price per Order (AOV)", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Year-Month Period", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Average Total Price ($)", fontsize=11, fontweight="bold")
        ax2.set_xticks(x_indices)
        ax2.set_xticklabels(months, rotation=45, ha="right", fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.55)
        ax2.set_ylim(0, max(means) * 1.18)
        ax2.legend(loc="upper right", fontsize=10, framealpha=0.9)
        
        plt.suptitle("Monthly Total Price Time Series Line Graph Analysis (2023 - 2025)", fontsize=15, fontweight="bold", y=0.995)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\n[FIGURE SAVED] Monthly TotalPrice line graph plot saved to: '{plot_path}'")
        print("=" * 80)
        
    return monthly_stats_df


def analyze_product_orders_time_series(
    df: pd.DataFrame,
    date_col: str = "Date",
    product_col: str = "Product",
    save_plot: bool = True,
    plot_path: str = os.path.join("figures", "product_orders_time_series.png")
) -> pd.DataFrame:
    """
    Analyze and visualize each product's monthly ordering count over time as separate time-series line graphs.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        date_col (str): Column name for order timestamps.
        product_col (str): Column name for product names.
        save_plot (bool): Whether to generate and save the line graphs.
        plot_path (str): Filepath to save the time series plot.
        
    Returns:
        pd.DataFrame: Monthly ordering count table per product.
    """
    if date_col not in df.columns or product_col not in df.columns:
        print(f"[WARNING] Columns '{date_col}' or '{product_col}' not found in DataFrame.")
        return pd.DataFrame()
        
    print("\n" + "=" * 80)
    print(f"TIME SERIES DISTRIBUTION: EACH '{product_col}' ORDERING COUNT ACROSS EACH MONTH")
    print("=" * 80)
    
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
    df_temp["YearMonth"] = df_temp[date_col].dt.to_period("M").astype(str)
    
    # 1. Pivot Table: Monthly Order Count per Product
    monthly_pivot = pd.crosstab(df_temp["YearMonth"], df_temp[product_col])
    months = sorted(monthly_pivot.index.tolist())
    products = sorted(monthly_pivot.columns.tolist())
    
    # 2. Add Monthly Total & Print Table
    monthly_pivot_display = monthly_pivot.copy()
    monthly_pivot_display["Total_Monthly_Orders"] = monthly_pivot_display.sum(axis=1)
    
    print("\n--- Monthly Ordering Count per Product (Cross-Tabulation) ---")
    print(monthly_pivot_display.to_string())
    print("-" * 80)
    
    # 3. Compute Product Summary Statistics across Months
    summary_records = []
    total_dataset_orders = len(df_temp)
    
    for prod in products:
        counts = monthly_pivot[prod]
        total_prod_orders = counts.sum()
        mean_monthly = counts.mean()
        std_monthly = counts.std()
        min_monthly = counts.min()
        max_monthly = counts.max()
        peak_months = counts[counts == max_monthly].index.tolist()
        lowest_months = counts[counts == min_monthly].index.tolist()
        share_pct = (total_prod_orders / total_dataset_orders) * 100
        
        summary_records.append({
            "Product": prod,
            "Total Orders": total_prod_orders,
            "Share (%)": round(share_pct, 2),
            "Monthly Mean": round(mean_monthly, 2),
            "Monthly Std": round(std_monthly, 2),
            "Min Count (Month)": f"{min_monthly} ({', '.join(lowest_months[:2])})",
            "Max Count (Month)": f"{max_monthly} ({', '.join(peak_months[:2])})"
        })
        
    summary_df = pd.DataFrame(summary_records).set_index("Product")
    print("\n--- Product Monthly Ordering Distribution Summary Metrics ---")
    print(summary_df.to_string())
    print("=" * 80)
    
    # 4. Graphical Visualization: Separate Time Series Line Graphs for Each Product
    if save_plot:
        ensure_figures_dir(plot_path)
        
        product_styling = {
            "Chair": {"color": "#16a085", "marker": "o"},
            "Desk": {"color": "#8e44ad", "marker": "s"},
            "Laptop": {"color": "#2980b9", "marker": "^"},
            "Monitor": {"color": "#c0392b", "marker": "D"},
            "Phone": {"color": "#d35400", "marker": "v"},
            "Printer": {"color": "#27ae60", "marker": "p"},
            "Tablet": {"color": "#f39c12", "marker": "*"}
        }
        default_colors = plt.cm.tab10.colors
        
        # 4x2 Grid of Separate Subplots for Each Product + Aggregate Total
        fig, axes = plt.subplots(4, 2, figsize=(19, 16))
        x_indices = np.arange(len(months))
        
        grid_axes = axes.flatten()
        
        for idx, prod in enumerate(products):
            ax = grid_axes[idx]
            style = product_styling.get(prod, {"color": default_colors[idx % 10], "marker": "o"})
            y_vals = monthly_pivot[prod].values
            mean_val = y_vals.mean()
            max_idx = np.argmax(y_vals)
            max_val = y_vals[max_idx]
            min_idx = np.argmin(y_vals)
            min_val = y_vals[min_idx]
            
            # Line + Area
            ax.plot(x_indices, y_vals, color=style["color"], marker=style["marker"], markersize=5.5, linewidth=2.0, label=f"{prod} Orders")
            ax.fill_between(x_indices, y_vals, color=style["color"], alpha=0.15)
            
            # Mean line
            ax.axhline(mean_val, color="#444444", linestyle="--", linewidth=1.3, label=f"Mean ({mean_val:.1f}/mo)")
            
            # Highlight peak and min
            ax.scatter(x_indices[max_idx], max_val, color="#c0392b", s=50, zorder=5)
            ax.annotate(f"Peak: {max_val} ({months[max_idx]})", (x_indices[max_idx], max_val),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold", color="#c0392b")
                        
            ax.scatter(x_indices[min_idx], min_val, color="#7f8c8d", s=40, zorder=5)
            ax.annotate(f"Min: {min_val}", (x_indices[min_idx], min_val),
                        xytext=(0, -12), textcoords="offset points",
                        ha="center", fontsize=7.5, color="#555555")
                        
            ax.set_title(f"{prod} Monthly Ordering Trend (Total: {y_vals.sum():,} | Mean: {mean_val:.1f}/mo)", fontsize=11, fontweight="bold", pad=6)
            ax.set_ylabel("Orders (Units)", fontsize=9.5, fontweight="bold")
            ax.set_ylim(0, max_val + 3)
            ax.grid(True, linestyle="--", alpha=0.5)
            
            # X-ticks with clear labels
            tick_indices = list(range(0, len(months), 3))
            if (len(months) - 1) not in tick_indices:
                tick_indices.append(len(months) - 1)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([months[i] for i in tick_indices], rotation=35, ha="right", fontsize=8.5)
            ax.legend(loc="upper right", fontsize=8.5, framealpha=0.85)
            
        # 8th subplot: Total Combined Store Demand
        ax_tot = grid_axes[7]
        total_monthly_orders = monthly_pivot.sum(axis=1).values
        mean_tot = total_monthly_orders.mean()
        max_tot_idx = np.argmax(total_monthly_orders)
        min_tot_idx = np.argmin(total_monthly_orders)
        
        ax_tot.plot(x_indices, total_monthly_orders, color="#2c3e50", marker="o", markersize=5.5, linewidth=2.2, label="Total Store Orders")
        ax_tot.fill_between(x_indices, total_monthly_orders, color="#2c3e50", alpha=0.15)
        ax_tot.axhline(mean_tot, color="#e74c3c", linestyle="--", linewidth=1.3, label=f"Overall Mean ({mean_tot:.1f}/mo)")
        
        ax_tot.scatter(x_indices[max_tot_idx], total_monthly_orders[max_tot_idx], color="#27ae60", s=50, zorder=5)
        ax_tot.annotate(f"Peak: {total_monthly_orders[max_tot_idx]} ({months[max_tot_idx]})",
                        (x_indices[max_tot_idx], total_monthly_orders[max_tot_idx]),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold", color="#27ae60")
                        
        ax_tot.set_title(f"All Products Combined Monthly Demand (Total: {total_dataset_orders:,} Orders)", fontsize=11, fontweight="bold", pad=6)
        ax_tot.set_ylabel("Orders (Units)", fontsize=9.5, fontweight="bold")
        ax_tot.set_ylim(0, total_monthly_orders.max() + 10)
        ax_tot.grid(True, linestyle="--", alpha=0.5)
        
        tick_indices = list(range(0, len(months), 3))
        if (len(months) - 1) not in tick_indices:
            tick_indices.append(len(months) - 1)
        ax_tot.set_xticks(tick_indices)
        ax_tot.set_xticklabels([months[i] for i in tick_indices], rotation=35, ha="right", fontsize=8.5)
        ax_tot.legend(loc="upper right", fontsize=8.5, framealpha=0.85)
        
        plt.suptitle("Individual Product Monthly Ordering Count Time Series (2023 - 2025)", fontsize=15, fontweight="bold", y=0.995)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\n[FIGURE SAVED] Separate product time-series line graphs saved to: '{plot_path}'")
        print("=" * 80)
        
    return monthly_pivot


def run_distribution_analysis(
    df: pd.DataFrame, 
    save_plots: bool = True,
    figures_dir: str = "figures"
) -> dict:
    """
    Master function to analyze all required variable distributions:
    - Numerical: UnitPrice, TotalPrice, ItemsInCart
    - Categorical: Product, PaymentMethod, OrderStatus, CouponCode, ReferralSource
    - Temporal: Date (Monthly Orders)
    - Grouped: UnitPrice across each Product
    - Monthly: TotalPrice across each Month
    - Time Series: Each Product ordering count across each Month (Line Graph)
    
    All visualization plots are saved inside `figures_dir`.
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    numeric_plot = os.path.join(figures_dir, "numeric_distributions.png")
    categorical_plot = os.path.join(figures_dir, "categorical_distributions.png")
    monthly_plot = os.path.join(figures_dir, "monthly_orders_distribution.png")
    unitprice_prod_plot = os.path.join(figures_dir, "unitprice_by_product_distribution.png")
    totalprice_month_plot = os.path.join(figures_dir, "totalprice_by_month_distribution.png")
    product_time_series_plot = os.path.join(figures_dir, "product_orders_time_series.png")
    
    num_stats = analyze_numeric_distributions(
        df, 
        columns=TARGET_NUMERIC_VARS, 
        save_plot=save_plots, 
        plot_path=numeric_plot
    )
    
    cat_stats = analyze_categorical_distributions(
        df, 
        columns=TARGET_CATEGORICAL_VARS, 
        save_plot=save_plots, 
        plot_path=categorical_plot
    )
    
    temp_stats = analyze_temporal_distribution(
        df, 
        date_column=TARGET_DATE_VAR, 
        save_plot=save_plots, 
        plot_path=monthly_plot
    )
    
    prod_price_stats = analyze_unitprice_by_product_distribution(
        df,
        product_col="Product",
        price_col="UnitPrice",
        save_plot=save_plots,
        plot_path=unitprice_prod_plot
    )
    
    monthly_price_stats = analyze_totalprice_by_month_distribution(
        df,
        date_col=TARGET_DATE_VAR,
        price_col="TotalPrice",
        save_plot=save_plots,
        plot_path=totalprice_month_plot
    )
    
    prod_ts_stats = analyze_product_orders_time_series(
        df,
        date_col=TARGET_DATE_VAR,
        product_col="Product",
        save_plot=save_plots,
        plot_path=product_time_series_plot
    )
    
    print("\n[SUCCESS] Variable distribution analysis complete.")
    if save_plots:
        print(f"[FIGURES CREATED IN '{figures_dir}/']:")
        print(f"  1. {numeric_plot}")
        print(f"  2. {categorical_plot}")
        print(f"  3. {monthly_plot}")
        print(f"  4. {unitprice_prod_plot}")
        print(f"  5. {totalprice_month_plot}")
        print(f"  6. {product_time_series_plot}")
        
    return {
        "numeric": num_stats,
        "categorical": cat_stats,
        "temporal": temp_stats,
        "unitprice_by_product": prod_price_stats,
        "totalprice_by_month": monthly_price_stats,
        "product_orders_time_series": prod_ts_stats
    }


if __name__ == "__main__":
    from data_loader import load_dataset, handle_duplicates
    
    # 1. Load dataset
    df = load_dataset()
    df = handle_duplicates(df)
    
    # 2. Run full distribution analysis on specified target variables
    run_distribution_analysis(df, save_plots=True, figures_dir="figures")

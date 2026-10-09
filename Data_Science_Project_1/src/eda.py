"""
Exploratory Data Analysis (EDA) Module - Data Science Project 1
Provides comprehensive descriptive statistics, categorical breakdowns,
bivariate/multivariate domain analyses, and automated diagnostic visualization generation.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def compute_numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute enhanced numerical summary statistics including central tendency,
    dispersion, IQR, skewness, and kurtosis.
    """
    print("\n" + "=" * 80)
    print(">>> [EDA] COMPREHENSIVE NUMERICAL DESCRIPTIVE STATISTICS")
    print("=" * 80)
    
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
    print(summary_table.to_string())
    print("=" * 80)
    return summary_table


def analyze_categorical_distributions(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """
    Analyze frequency counts, proportions, and unique level distributions for categorical variables.
    """
    print("\n" + "=" * 80)
    print(">>> [EDA] CATEGORICAL ATTRIBUTE DISTRIBUTIONS")
    print("=" * 80)
    
    if columns is None:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        # Filter high-cardinality IDs and address strings for clean terminal presentation
        columns = [c for c in columns if c not in ["OrderID", "CustomerID", "TrackingNumber", "ShippingAddress"] and df[c].nunique() <= 50]
        
    cat_summaries = {}
    for col in columns:
        print(f"\n--- Frequency Distribution for '{col}' ---")
        counts = df[col].value_counts(dropna=False)
        pcts = (df[col].value_counts(dropna=False, normalize=True) * 100).round(2)
        dist_df = pd.DataFrame({"Count": counts, "Percentage (%)": pcts})
        print(dist_df.to_string())
        cat_summaries[col] = dist_df
        
    print("=" * 80)
    return cat_summaries


def analyze_bivariate_relationships(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compute bivariate revenue, volume, and fulfillment metrics across business dimensions.
    """
    print("\n" + "=" * 80)
    print(">>> [EDA] BIVARIATE & COMMERCIAL DOMAIN INSIGHTS")
    print("=" * 80)
    
    bivariate_results = {}
    
    # 1. Product Revenue and Average Order Value
    if "Product" in df.columns and "TotalPrice" in df.columns:
        print("\n--- Revenue & Volume Metrics by Product Category ---")
        prod_metrics = df.groupby("Product").agg(
            Total_Orders=("OrderID" if "OrderID" in df.columns else "TotalPrice", "count"),
            Total_Revenue=("TotalPrice", "sum"),
            Avg_Order_Value=("TotalPrice", "mean"),
            Avg_UnitPrice=("UnitPrice", "mean") if "UnitPrice" in df.columns else ("TotalPrice", "mean"),
            Avg_Quantity=("Quantity", "mean") if "Quantity" in df.columns else ("TotalPrice", "mean")
        ).round(2).sort_values(by="Total_Revenue", ascending=False)
        print(prod_metrics.to_string())
        bivariate_results["Product_Metrics"] = prod_metrics
        
    # 2. Performance by Referral Source
    if "ReferralSource" in df.columns and "TotalPrice" in df.columns:
        print("\n--- Performance Metrics by Customer Referral Source ---")
        referral_metrics = df.groupby("ReferralSource").agg(
            Total_Orders=("TotalPrice", "count"),
            Total_Revenue=("TotalPrice", "sum"),
            Avg_Order_Value=("TotalPrice", "mean")
        ).round(2).sort_values(by="Total_Revenue", ascending=False)
        print(referral_metrics.to_string())
        bivariate_results["Referral_Metrics"] = referral_metrics
        
    # 3. Order Fulfillment Status vs. Financial Value
    if "OrderStatus" in df.columns and "TotalPrice" in df.columns:
        print("\n--- Fulfillment Status vs. Financial Volume Impact ---")
        status_metrics = df.groupby("OrderStatus").agg(
            Order_Count=("TotalPrice", "count"),
            Total_Value=("TotalPrice", "sum"),
            Avg_Value=("TotalPrice", "mean")
        ).round(2).sort_values(by="Total_Value", ascending=False)
        print(status_metrics.to_string())
        bivariate_results["Status_Metrics"] = status_metrics
        
    print("=" * 80)
    return bivariate_results


def generate_all_eda_visualizations(
    df: pd.DataFrame, 
    figures_dir: str | Path = Path("outputs/figures")
) -> None:
    """
    Generate and export all EDA visual artifacts:
    - Distribution plots (distributions/)
    - Boxplots (boxplots/)
    - Categorical bar charts (categorical_plots/)
    - Pearson correlation heatmap (correlation_heatmap.png)
    """
    fig_root = Path(figures_dir)
    dist_dir = fig_root / "distributions"
    box_dir = fig_root / "boxplots"
    cat_dir = fig_root / "categorical_plots"
    
    for d in [dist_dir, box_dir, cat_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    print(f"\n[EDA] Generating visualization figures to: {fig_root}")
    
    # Styling configuration
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    
    # 1. TotalPrice Distribution
    if "TotalPrice" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["TotalPrice"], kde=True, color="#2563eb", bins=30, ax=ax)
        ax.axvline(df["TotalPrice"].mean(), color="#dc2626", linestyle="--", linewidth=1.8, label=f"Mean: ${df['TotalPrice'].mean():.2f}")
        ax.axvline(df["TotalPrice"].median(), color="#16a34a", linestyle="-.", linewidth=1.8, label=f"Median: ${df['TotalPrice'].median():.2f}")
        ax.set_title("Distribution of TotalPrice (Invoice Amount)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("TotalPrice ($)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.legend()
        plt.tight_layout()
        fig.savefig(dist_dir / "totalprice_distribution.png", dpi=300)
        plt.close()
        
    # 2. Log_TotalPrice Distribution
    log_data = np.log1p(df["TotalPrice"]) if "TotalPrice" in df.columns else None
    if log_data is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(log_data, kde=True, color="#059669", bins=30, ax=ax)
        ax.axvline(log_data.mean(), color="#dc2626", linestyle="--", linewidth=1.8, label=f"Mean: {log_data.mean():.2f}")
        ax.axvline(log_data.median(), color="#2563eb", linestyle="-.", linewidth=1.8, label=f"Median: {log_data.median():.2f}")
        ax.set_title("Distribution of Log_TotalPrice (log(1 + TotalPrice))", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Log_TotalPrice", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.legend()
        plt.tight_layout()
        fig.savefig(dist_dir / "log_totalprice_distribution.png", dpi=300)
        plt.close()

    # 3. UnitPrice Distribution
    if "UnitPrice" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["UnitPrice"], kde=True, color="#7c3aed", bins=30, ax=ax)
        ax.axvline(df["UnitPrice"].mean(), color="#dc2626", linestyle="--", linewidth=1.8, label=f"Mean: ${df['UnitPrice'].mean():.2f}")
        ax.axvline(df["UnitPrice"].median(), color="#16a34a", linestyle="-.", linewidth=1.8, label=f"Median: ${df['UnitPrice'].median():.2f}")
        ax.set_title("Distribution of UnitPrice (Product Catalog Price)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("UnitPrice ($)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.legend()
        plt.tight_layout()
        fig.savefig(dist_dir / "unitprice_distribution.png", dpi=300)
        plt.close()

    # 4. Quantity Distribution
    if "Quantity" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.countplot(data=df, x="Quantity", color="#3b82f6", ax=ax)
        ax.set_title("Distribution of Order Quantity per Transaction", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Quantity (Units)", fontsize=11)
        ax.set_ylabel("Order Count", fontsize=11)
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
        plt.tight_layout()
        fig.savefig(dist_dir / "quantity_distribution.png", dpi=300)
        plt.close()

    # 5. ItemsInCart Distribution
    if "ItemsInCart" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.countplot(data=df, x="ItemsInCart", color="#8b5cf6", ax=ax)
        ax.set_title("Distribution of Items In Cart at Checkout", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Items in Cart", fontsize=11)
        ax.set_ylabel("Order Count", fontsize=11)
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
        plt.tight_layout()
        fig.savefig(dist_dir / "itemsincart_distribution.png", dpi=300)
        plt.close()

    # 6. Distribution Comparison (Original vs Log)
    if "TotalPrice" in df.columns and log_data is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(df["TotalPrice"], kde=True, color="#2563eb", bins=30, ax=axes[0])
        axes[0].set_title(f"Original TotalPrice\n(Skewness = {df['TotalPrice'].skew():.2f})", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("TotalPrice ($)", fontsize=10)
        axes[0].set_ylabel("Density", fontsize=10)

        sns.histplot(log_data, kde=True, color="#059669", bins=30, ax=axes[1])
        axes[1].set_title(f"Log-Transformed TotalPrice\n(Skewness = {log_data.skew():.2f})", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Log_TotalPrice", fontsize=10)
        axes[1].set_ylabel("Density", fontsize=10)
        plt.suptitle("Statistical Comparison: Original vs. Log-Transformed Distribution", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(dist_dir / "distribution_comparison.png", dpi=300)
        plt.close()

    # 7. Outlier Boxplots
    num_cols = [c for c in ["UnitPrice", "Quantity", "ItemsInCart", "TotalPrice"] if c in df.columns]
    if num_cols:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        palette = ["#2563eb", "#059669", "#d97706", "#dc2626"]
        for i, col in enumerate(num_cols):
            sns.boxplot(data=df, y=col, ax=axes[i], color=palette[i % len(palette)], flierprops={"marker": "o", "markersize": 5, "markerfacecolor": "red"})
            axes[i].set_title(f"Boxplot: {col}", fontsize=12, fontweight="bold")
            axes[i].set_ylabel(col, fontsize=10)
        plt.suptitle("Univariate Outlier Detection via Boxplots (IQR Rule)", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        fig.savefig(box_dir / "outlier_boxplots_numerical.png", dpi=300)
        plt.close()

    # 8. TotalPrice by Product Boxplot
    if "Product" in df.columns and "TotalPrice" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        order_p = df.groupby("Product")["TotalPrice"].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x="Product", y="TotalPrice", order=order_p, color="#3b82f6", ax=ax)
        ax.set_title("TotalPrice Distribution by Product Category", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Product Category", fontsize=11)
        ax.set_ylabel("TotalPrice ($)", fontsize=11)
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        fig.savefig(box_dir / "totalprice_by_product_boxplot.png", dpi=300)
        plt.close()

    # 9. UnitPrice by Product Boxplot
    if "Product" in df.columns and "UnitPrice" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        order_u = df.groupby("Product")["UnitPrice"].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x="Product", y="UnitPrice", order=order_u, color="#8b5cf6", ax=ax)
        ax.set_title("Catalog UnitPrice Dispersion by Product", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Product Category", fontsize=11)
        ax.set_ylabel("UnitPrice ($)", fontsize=11)
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        fig.savefig(box_dir / "unitprice_by_product_boxplot.png", dpi=300)
        plt.close()

    # 10. Categorical Frequency Charts
    cat_items = [
        ("Product", cat_dir / "product_distribution.png", "#2563eb", "Product Category Purchase Distribution"),
        ("PaymentMethod", cat_dir / "payment_method_distribution.png", "#059669", "Payment Method Adoption"),
        ("OrderStatus", cat_dir / "order_status_distribution.png", "#d97706", "Order Fulfillment Status"),
        ("CouponCode", cat_dir / "coupon_code_distribution.png", "#7c3aed", "Coupon Code Utilization (Including NO_COUPON)"),
        ("ReferralSource", cat_dir / "referral_source_distribution.png", "#dc2626", "Customer Acquisition Referral Channels")
    ]
    for col, out_p, col_hex, title in cat_items:
        if col in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            order = df[col].value_counts().index
            sns.countplot(data=df, x=col, order=order, color=col_hex, ax=ax)
            ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel(col, fontsize=11)
            ax.set_ylabel("Order Count", fontsize=11)
            plt.xticks(rotation=15, ha='right')
            for p in ax.patches:
                val = int(p.get_height())
                pct = (val / len(df)) * 100
                ax.annotate(f"{val} ({pct:.1f}%)", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
            plt.tight_layout()
            fig.savefig(out_p, dpi=300)
            plt.close()

    # 11. Bivariate Categorical Breakdown
    if "Product" in df.columns and "OrderStatus" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        crosstab_pct = pd.crosstab(df["Product"], df["OrderStatus"], normalize="index") * 100
        crosstab_pct.plot(kind="bar", stacked=True, colormap="tab10", ax=ax, edgecolor="black", alpha=0.85)
        ax.set_title("Order Fulfillment Status Proportions across Product Lines (%)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Product Category", fontsize=11)
        ax.set_ylabel("Proportion (%)", fontsize=11)
        ax.legend(title="OrderStatus", bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        fig.savefig(cat_dir / "bivariate_categorical_breakdown.png", dpi=300)
        plt.close()

    # 12. Correlation Heatmap
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        fig, ax = plt.subplots(figsize=(9, 7.5))
        corr_m = num_df.corr()
        mask = np.triu(np.ones_like(corr_m, dtype=bool))
        sns.heatmap(corr_m, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                    square=True, linewidths=1.0, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title("Pearson Correlation Heatmap of Numerical Features", fontsize=13, fontweight="bold", pad=14)
        plt.tight_layout()
        fig.savefig(fig_root / "correlation_heatmap.png", dpi=300)
        plt.close()

    print("[EDA] Visualizations generation completed successfully.")


def run_full_eda(
    data_path: str | Path = Path("data/raw/original_dataset.xlsx"),
    figures_dir: str | Path = Path("outputs/figures")
) -> None:
    """
    Master runner to execute the full Exploratory Data Analysis suite.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset for EDA not found at: {path}")
        
    print("\n==================================================")
    print(">>> EXECUTING EXPLORATORY DATA ANALYSIS PIPELINE")
    print("==================================================")
    
    if path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
        
    compute_numerical_summary(df)
    analyze_categorical_distributions(df)
    analyze_bivariate_relationships(df)
    generate_all_eda_visualizations(df, figures_dir=figures_dir)
    print("==================================================\n")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    raw_p = project_root / "data" / "raw" / "original_dataset.xlsx"
    clean_p = project_root / "data" / "processed" / "cleaned_dataset.csv"
    fig_p = project_root / "outputs" / "figures"
    
    target_data = clean_p if clean_p.exists() else raw_p
    run_full_eda(target_data, fig_p)

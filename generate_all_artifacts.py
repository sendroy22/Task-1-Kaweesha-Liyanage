"""
Master Generator Script for Data_Science_Project_1 Folder Structure
Sets up folders, copies raw data, runs data cleaning, feature engineering, validation,
generates all visualization figures, summary tables, Jupyter notebook, and PDF report.
"""

import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Path Definitions
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR / "Data_Science_Project_1"
DATA_RAW_DIR = PROJECT_DIR / "data" / "raw"
DATA_PROC_DIR = PROJECT_DIR / "data" / "processed"
SRC_DIR = PROJECT_DIR / "src"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
DIST_DIR = FIG_DIR / "distributions"
BOX_DIR = FIG_DIR / "boxplots"
CAT_DIR = FIG_DIR / "categorical_plots"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORT_DIR = PROJECT_DIR / "report"

# 1. Create all directories
for d in [DATA_RAW_DIR, DATA_PROC_DIR, SRC_DIR, NOTEBOOKS_DIR, DIST_DIR, BOX_DIR, CAT_DIR, TABLES_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(">>> [1/6] Created directory hierarchy.")

# 2. Copy Raw Dataset
src_raw = BASE_DIR / "Dataset for Data Analytics.xlsx"
dst_raw = DATA_RAW_DIR / "original_dataset.xlsx"
if src_raw.exists():
    shutil.copy2(src_raw, dst_raw)
    print(f">>> [2/6] Copied raw dataset to: {dst_raw}")
else:
    print(f"Warning: {src_raw} not found.")

# Set aesthetics for matplotlib/seaborn
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['figure.autolayout'] = False

# Import pipeline modules
import sys
sys.path.insert(0, str(SRC_DIR))
from data_cleaning import run_data_cleaning_pipeline
from feature_engineering import run_feature_engineering_pipeline
from validation import run_validation_pipeline

# 3. Execute Pipelines to generate cleaned data, engineered data, and tables
print(">>> [3/6] Running Data Cleaning, Feature Engineering & Validation pipelines...")
df_clean = run_data_cleaning_pipeline(
    raw_data_path=dst_raw,
    processed_output_path=DATA_PROC_DIR / "cleaned_dataset.csv",
    tables_dir=TABLES_DIR
)

df_eng = run_feature_engineering_pipeline(
    cleaned_data_path=DATA_PROC_DIR / "cleaned_dataset.csv",
    tables_dir=TABLES_DIR
)

df_final = run_validation_pipeline(
    engineered_df=df_eng,
    final_output_path=DATA_PROC_DIR / "final_model_ready_dataset.csv",
    tables_dir=TABLES_DIR
)

# 4. Generate High Quality Visualizations
print(">>> [4/6] Generating Figures & Visualizations...")

# --- A. DISTRIBUTIONS ---
# TotalPrice distribution
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df_clean["TotalPrice"], kde=True, color="#2563eb", bins=30, ax=ax)
ax.axvline(df_clean["TotalPrice"].mean(), color="#dc2626", linestyle="--", linewidth=1.8, label=f"Mean: ${df_clean['TotalPrice'].mean():.2f}")
ax.axvline(df_clean["TotalPrice"].median(), color="#16a34a", linestyle="-.", linewidth=1.8, label=f"Median: ${df_clean['TotalPrice'].median():.2f}")
ax.set_title("Distribution of TotalPrice (Invoice Amount)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("TotalPrice ($)", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.legend()
plt.tight_layout()
fig.savefig(DIST_DIR / "totalprice_distribution.png", dpi=300)
plt.close()

# Log_TotalPrice distribution
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df_final["Log_TotalPrice"], kde=True, color="#059669", bins=30, ax=ax)
ax.axvline(df_final["Log_TotalPrice"].mean(), color="#dc2626", linestyle="--", linewidth=1.8, label=f"Mean: {df_final['Log_TotalPrice'].mean():.2f}")
ax.axvline(df_final["Log_TotalPrice"].median(), color="#2563eb", linestyle="-.", linewidth=1.8, label=f"Median: {df_final['Log_TotalPrice'].median():.2f}")
ax.set_title("Distribution of Log_TotalPrice (log(1 + TotalPrice))", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Log_TotalPrice", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.legend()
plt.tight_layout()
fig.savefig(DIST_DIR / "log_totalprice_distribution.png", dpi=300)
plt.close()

# UnitPrice distribution
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df_clean["UnitPrice"], kde=True, color="#7c3aed", bins=30, ax=ax)
ax.axvline(df_clean["UnitPrice"].mean(), color="#dc2626", linestyle="--", linewidth=1.8, label=f"Mean: ${df_clean['UnitPrice'].mean():.2f}")
ax.axvline(df_clean["UnitPrice"].median(), color="#16a34a", linestyle="-.", linewidth=1.8, label=f"Median: ${df_clean['UnitPrice'].median():.2f}")
ax.set_title("Distribution of UnitPrice (Product Catalog Price)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("UnitPrice ($)", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.legend()
plt.tight_layout()
fig.savefig(DIST_DIR / "unitprice_distribution.png", dpi=300)
plt.close()

# Quantity distribution
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.countplot(data=df_clean, x="Quantity", palette="Blues_d", ax=ax)
ax.set_title("Distribution of Order Quantity per Transaction", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Quantity (Units)", fontsize=11)
ax.set_ylabel("Order Count", fontsize=11)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
fig.savefig(DIST_DIR / "quantity_distribution.png", dpi=300)
plt.close()

# ItemsInCart distribution
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.countplot(data=df_clean, x="ItemsInCart", palette="Purples_d", ax=ax)
ax.set_title("Distribution of Items In Cart at Checkout", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Items in Cart", fontsize=11)
ax.set_ylabel("Order Count", fontsize=11)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
fig.savefig(DIST_DIR / "itemsincart_distribution.png", dpi=300)
plt.close()

# Side-by-side distribution comparison (Original vs Log)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df_clean["TotalPrice"], kde=True, color="#2563eb", bins=30, ax=axes[0])
axes[0].set_title(f"Original TotalPrice\n(Skewness = {df_clean['TotalPrice'].skew():.2f})", fontsize=12, fontweight="bold")
axes[0].set_xlabel("TotalPrice ($)", fontsize=10)
axes[0].set_ylabel("Density", fontsize=10)

sns.histplot(df_final["Log_TotalPrice"], kde=True, color="#059669", bins=30, ax=axes[1])
axes[1].set_title(f"Log-Transformed TotalPrice\n(Skewness = {df_final['Log_TotalPrice'].skew():.2f})", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Log_TotalPrice", fontsize=10)
axes[1].set_ylabel("Density", fontsize=10)
plt.suptitle("Statistical Comparison: Original vs. Log-Transformed Distribution", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(DIST_DIR / "distribution_comparison.png", dpi=300)
plt.close()

# --- B. BOXPLOTS ---
# Outlier boxplots for numerical features
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
num_features = ["UnitPrice", "Quantity", "ItemsInCart", "TotalPrice"]
colors = ["#2563eb", "#059669", "#d97706", "#dc2626"]

for i, col in enumerate(num_features):
    sns.boxplot(data=df_clean, y=col, ax=axes[i], color=colors[i], flierprops={"marker": "o", "markersize": 5, "markerfacecolor": "red"})
    axes[i].set_title(f"Boxplot: {col}", fontsize=12, fontweight="bold")
    axes[i].set_ylabel(col, fontsize=10)
plt.suptitle("Univariate Outlier Detection via Boxplots (IQR Rule)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(BOX_DIR / "outlier_boxplots_numerical.png", dpi=300)
plt.close()

# TotalPrice by Product boxplot
fig, ax = plt.subplots(figsize=(10, 5.5))
order_prod = df_clean.groupby("Product")["TotalPrice"].median().sort_values(ascending=False).index
sns.boxplot(data=df_clean, x="Product", y="TotalPrice", order=order_prod, palette="Blues_r", ax=ax)
ax.set_title("TotalPrice Distribution by Product Category", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Product Category", fontsize=11)
ax.set_ylabel("TotalPrice ($)", fontsize=11)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
fig.savefig(BOX_DIR / "totalprice_by_product_boxplot.png", dpi=300)
plt.close()

# TotalPrice by Order_Month boxplot
fig, ax = plt.subplots(figsize=(11, 5.5))
sns.boxplot(data=df_final, x="Order_Month", y="TotalPrice", palette="viridis", ax=ax)
ax.set_title("Monthly TotalPrice Distribution & Seasonality", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Calendar Month (1=Jan .. 12=Dec)", fontsize=11)
ax.set_ylabel("TotalPrice ($)", fontsize=11)
plt.tight_layout()
fig.savefig(BOX_DIR / "totalprice_by_month_boxplot.png", dpi=300)
plt.close()

# UnitPrice by Product boxplot
fig, ax = plt.subplots(figsize=(10, 5.5))
order_unit = df_clean.groupby("Product")["UnitPrice"].median().sort_values(ascending=False).index
sns.boxplot(data=df_clean, x="Product", y="UnitPrice", order=order_unit, palette="Purples_r", ax=ax)
ax.set_title("Catalog UnitPrice Dispersion by Product", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Product Category", fontsize=11)
ax.set_ylabel("UnitPrice ($)", fontsize=11)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
fig.savefig(BOX_DIR / "unitprice_by_product_boxplot.png", dpi=300)
plt.close()

# --- C. CATEGORICAL PLOTS ---
cat_vars = [
    ("Product", CAT_DIR / "product_distribution.png", "Blues_d", "Product Category Purchase Distribution"),
    ("PaymentMethod", CAT_DIR / "payment_method_distribution.png", "Greens_d", "Payment Method Adoption"),
    ("OrderStatus", CAT_DIR / "order_status_distribution.png", "Oranges_d", "Order Fulfillment Status"),
    ("CouponCode", CAT_DIR / "coupon_code_distribution.png", "Purples_d", "Coupon Code Utilization (Including NO_COUPON)"),
    ("ReferralSource", CAT_DIR / "referral_source_distribution.png", "Reds_d", "Customer Acquisition Referral Channels")
]

for col, out_p, pal, title in cat_vars:
    if col in df_clean.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        order = df_clean[col].value_counts().index
        sns.countplot(data=df_clean, x=col, order=order, palette=pal, ax=ax)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel("Order Count", fontsize=11)
        plt.xticks(rotation=15, ha='right')
        for p in ax.patches:
            val = int(p.get_height())
            pct = (val / len(df_clean)) * 100
            ax.annotate(f"{val} ({pct:.1f}%)", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
        plt.tight_layout()
        fig.savefig(out_p, dpi=300)
        plt.close()

# Bivariate categorical breakdown (Product vs OrderStatus)
fig, ax = plt.subplots(figsize=(10, 5.5))
crosstab_pct = pd.crosstab(df_clean["Product"], df_clean["OrderStatus"], normalize="index") * 100
crosstab_pct.plot(kind="bar", stacked=True, colormap="tab10", ax=ax, edgecolor="black", alpha=0.85)
ax.set_title("Order Fulfillment Status Proportions across Product Lines (%)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Product Category", fontsize=11)
ax.set_ylabel("Proportion (%)", fontsize=11)
ax.legend(title="OrderStatus", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
fig.savefig(CAT_DIR / "bivariate_categorical_breakdown.png", dpi=300)
plt.close()

# --- D. CORRELATION HEATMAP ---
# Correlation heatmap of numerical + engineered features
corr_cols = ["Quantity", "UnitPrice", "TotalPrice", "ItemsInCart", "Order_Month", "HasCoupon", "Log_TotalPrice"]
avail_corr_cols = [c for c in corr_cols if c in df_final.columns]
corr_sub = df_final[avail_corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7.5))
mask = np.triu(np.ones_like(corr_sub, dtype=bool))
sns.heatmap(corr_sub, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            square=True, linewidths=1.0, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Pearson Correlation Heatmap of Numerical & Engineered Features", fontsize=13, fontweight="bold", pad=14)
plt.tight_layout()
fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=300)
plt.close()

print(">>> [4/6] Successfully generated all visualization figures.")

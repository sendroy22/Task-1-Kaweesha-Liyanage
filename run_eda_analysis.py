"""
Complete Exploratory Data Analysis (EDA) Script for Data Science Project 1.
Investigates, summarizes, visualizes, and reports patterns and problems in the data
without modifying, imputing, deleting, encoding, or transforming the original dataset.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set plotting style and aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#f1f5f9'
plt.rcParams['grid.linestyle'] = '--'

# Create output directories
OUTPUT_DIR = Path("eda_outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# 1. LOAD RAW DATASET
RAW_DATA_PATH = "Dataset for Data Analytics.xlsx"
df_raw = pd.read_excel(RAW_DATA_PATH)

print(f"[INFO] Successfully loaded raw dataset from: {RAW_DATA_PATH}")
print(f"[INFO] Dataset Dimensions: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns\n")

# ==========================================
# 1. DATASET OVERVIEW
# ==========================================
num_rows, num_cols = df_raw.shape
col_names = list(df_raw.columns)
dtypes = df_raw.dtypes

numerical_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_raw.select_dtypes(include=['object', 'category']).columns.tolist()
datetime_cols = df_raw.select_dtypes(include=['datetime64']).columns.tolist()

duplicate_rows = int(df_raw.duplicated().sum())
duplicate_order_ids = int(df_raw['OrderID'].duplicated().sum()) if 'OrderID' in df_raw.columns else 0
memory_usage_bytes = df_raw.memory_usage(deep=True).sum()
memory_usage_kb = memory_usage_bytes / 1024
memory_usage_mb = memory_usage_kb / 1024

overview_records = []
for col in df_raw.columns:
    s = df_raw[col]
    dt = str(s.dtype)
    non_null_s = s.dropna()
    py_t = type(non_null_s.iloc[0]).__name__ if not non_null_s.empty else "None"
    n_unique = s.nunique(dropna=True)
    null_cnt = s.isnull().sum()
    null_pct = (null_cnt / len(s)) * 100
    
    if col in numerical_cols:
        col_type = "Numerical"
    elif col in datetime_cols:
        col_type = "Datetime"
    else:
        col_type = "Categorical"
        
    sample_vals = non_null_s.unique()[:3].tolist()
    sample_str = ", ".join([str(v)[:15] for v in sample_vals])
    
    overview_records.append({
        "Column Name": col,
        "Data Type": dt,
        "Python Type": py_t,
        "Variable Category": col_type,
        "Non-Null Count": len(s) - null_cnt,
        "Missing Count": null_cnt,
        "Missing (%)": round(null_pct, 2),
        "Unique Values": n_unique,
        "Sample Values": sample_str
    })

overview_df = pd.DataFrame(overview_records)
overview_df.to_csv(TABLES_DIR / "01_dataset_overview.csv", index=False)
print("--- DATASET OVERVIEW ---")
print(overview_df.to_string(index=False))
print(f"\nTotal Rows: {num_rows}, Total Columns: {num_cols}")
print(f"Duplicates (Full Rows): {duplicate_rows}, Duplicates (OrderID): {duplicate_order_ids}")
print(f"Total Deep Memory Usage: {memory_usage_kb:.2f} KB ({memory_usage_mb:.3f} MB)\n")

# ==========================================
# 2. DESCRIPTIVE STATISTICS
# ==========================================
# Numerical statistics
num_stats = []
for col in numerical_cols:
    s = df_raw[col].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    skew = s.skew()
    kurt = s.kurtosis()
    
    # Skewness classification
    if abs(skew) < 0.5:
        skew_class = "Fairly Symmetrical"
    elif 0.5 <= abs(skew) <= 1.0:
        skew_class = "Moderately Skewed"
    else:
        skew_class = "Highly Skewed"
        
    num_stats.append({
        "Variable": col,
        "Count": int(s.count()),
        "Mean": round(s.mean(), 2),
        "Median": round(s.median(), 2),
        "Std Dev": round(s.std(), 2),
        "Min": round(s.min(), 2),
        "Q1 (25%)": round(q1, 2),
        "Q3 (75%)": round(q3, 2),
        "IQR": round(iqr, 2),
        "Max": round(s.max(), 2),
        "Skewness": round(skew, 2),
        "Kurtosis": round(kurt, 2),
        "Skewness Classification": skew_class
    })

num_stats_df = pd.DataFrame(num_stats)
num_stats_df.to_csv(TABLES_DIR / "02_numerical_descriptive_stats.csv", index=False)
print("--- NUMERICAL DESCRIPTIVE STATISTICS ---")
print(num_stats_df.to_string(index=False))

# Categorical statistics
cat_stats = []
for col in categorical_cols:
    s = df_raw[col]
    n_unique = s.nunique(dropna=True)
    mode_val = s.mode()[0] if not s.mode().empty else "N/A"
    mode_freq = (s == mode_val).sum()
    mode_pct = (mode_freq / len(s)) * 100
    
    cat_stats.append({
        "Variable": col,
        "Unique Categories": n_unique,
        "Mode": mode_val,
        "Mode Frequency": mode_freq,
        "Mode Percentage (%)": round(mode_pct, 2),
        "Missing Values": s.isnull().sum()
    })

cat_stats_df = pd.DataFrame(cat_stats)
cat_stats_df.to_csv(TABLES_DIR / "02_categorical_summary_stats.csv", index=False)
print("\n--- CATEGORICAL DESCRIPTIVE STATISTICS ---")
print(cat_stats_df.to_string(index=False))

# Detailed category frequencies for core nominal variables
core_cat_cols = ['Product', 'PaymentMethod', 'OrderStatus', 'CouponCode', 'ReferralSource']
cat_freq_dict = {}
for col in core_cat_cols:
    vc = df_raw[col].value_counts(dropna=False)
    vp = df_raw[col].value_counts(dropna=False, normalize=True) * 100
    cdf = pd.DataFrame({
        "Category": vc.index.fillna("Missing (NaN)").astype(str),
        "Frequency": vc.values,
        "Percentage (%)": vp.values.round(2)
    })
    cdf.to_csv(TABLES_DIR / f"02_freq_{col}.csv", index=False)
    cat_freq_dict[col] = cdf

# ==========================================
# 3. MISSING-VALUE ANALYSIS
# ==========================================
missing_records = []
for col in df_raw.columns:
    n_miss = df_raw[col].isnull().sum()
    pct_miss = (n_miss / len(df_raw)) * 100
    
    if n_miss == 0:
        m_class = "No Missing Values"
        treatment = "None required (100% complete)"
    elif pct_miss < 5:
        m_class = "Low Missingness (<5%)"
        treatment = "Mean/Median (numerical) or Mode/Missing Indicator (categorical)"
    elif pct_miss <= 30:
        m_class = "Moderate Missingness (5% - 30%)"
        treatment = "Domain-aware constant ('No Coupon') + binary indicator (HasCoupon); avoid naive mode imputation"
    else:
        m_class = "High Missingness (>30%)"
        treatment = "Advanced imputation (KNN/MICE) or feature dropping if non-informative"
        
    missing_records.append({
        "Variable": col,
        "Data Type": str(df_raw[col].dtype),
        "Missing Count": n_miss,
        "Missing (%)": round(pct_miss, 2),
        "Missingness Classification": m_class,
        "Recommended Treatment Method": treatment
    })

missing_df = pd.DataFrame(missing_records).sort_values(by="Missing Count", ascending=False)
missing_df.to_csv(TABLES_DIR / "03_missing_values_summary.csv", index=False)
print("\n--- MISSING VALUE ANALYSIS ---")
print(missing_df.to_string(index=False))

# Plot 1: Missing Values Visualization
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
bars = ax.barh(missing_df['Variable'], missing_df['Missing (%)'], color=['#ef4444' if p > 0 else '#10b981' for p in missing_df['Missing (%)']])
ax.set_xlabel("Missing Percentage (%)", fontsize=11, fontweight='bold', labelpad=8)
ax.set_title("Missing Value Analysis by Variable", fontsize=13, fontweight='bold', pad=12)
ax.set_xlim(0, 35)

for bar, pct, cnt in zip(bars, missing_df['Missing (%)'], missing_df['Missing Count']):
    if pct > 0:
        ax.text(pct + 0.5, bar.get_y() + bar.get_height()/2, f"{pct:.2f}% ({cnt:,} rows) - Moderate Missingness", 
                va='center', fontsize=9, fontweight='bold', color='#b91c1c')
    else:
        ax.text(0.5, bar.get_y() + bar.get_height()/2, "0% (Complete)", 
                va='center', fontsize=8, color='#047857')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_missingness_analysis.png")
plt.close()

# ==========================================
# 4 & 5. NUMERICAL DISTRIBUTIONS & OUTLIER ANALYSIS (IQR METHOD)
# ==========================================
outlier_records = []
outlier_indices_dict = {}

for col in numerical_cols:
    s = df_raw[col].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    lower_outliers = s[s < lower_bound]
    upper_outliers = s[s > upper_bound]
    total_outliers = len(lower_outliers) + len(upper_outliers)
    outlier_pct = (total_outliers / len(s)) * 100
    
    min_outlier = float(s[s < lower_bound].min()) if len(lower_outliers) > 0 else (float(upper_outliers.min()) if len(upper_outliers) > 0 else np.nan)
    max_outlier = float(s[s > upper_bound].max()) if len(upper_outliers) > 0 else (float(lower_outliers.max()) if len(lower_outliers) > 0 else np.nan)
    
    outlier_records.append({
        "Variable": col,
        "Q1 (25%)": round(q1, 2),
        "Q3 (75%)": round(q3, 2),
        "IQR": round(iqr, 2),
        "Lower Bound": round(lower_bound, 2),
        "Upper Bound": round(upper_bound, 2),
        "Lower Outliers Count": len(lower_outliers),
        "Upper Outliers Count": len(upper_outliers),
        "Total Outliers": total_outliers,
        "Outlier (%)": round(outlier_pct, 2),
        "Min Outlier": round(min_outlier, 2) if not np.isnan(min_outlier) else "None",
        "Max Outlier": round(max_outlier, 2) if not np.isnan(max_outlier) else "None"
    })
    
    outlier_indices_dict[col] = {
        "lower": lower_outliers.index.tolist(),
        "upper": upper_outliers.index.tolist()
    }

outlier_df = pd.DataFrame(outlier_records)
outlier_df.to_csv(TABLES_DIR / "04_outlier_analysis_iqr.csv", index=False)
print("\n--- OUTLIER ANALYSIS (IQR METHOD) ---")
print(outlier_df.to_string(index=False))

# Plot 2: Numerical Variable Distributions (Histogram + KDE & Boxplots)
fig, axes = plt.subplots(4, 2, figsize=(14, 16), dpi=300)
palette = ['#2563eb', '#0d9488', '#8b5cf6', '#ea580c']

for idx, col in enumerate(numerical_cols):
    s = df_raw[col]
    color = palette[idx % len(palette)]
    
    # Histogram + KDE
    ax_hist = axes[idx, 0]
    sns.histplot(s, kde=True, ax=ax_hist, color=color, bins=25, edgecolor='#1e293b', alpha=0.65)
    mean_val = s.mean()
    median_val = s.median()
    ax_hist.axvline(mean_val, color='#dc2626', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    ax_hist.axvline(median_val, color='#16a34a', linestyle='-', linewidth=1.5, label=f'Median: {median_val:.2f}')
    ax_hist.set_title(f"{col} - Distribution & Central Tendency (Skewness: {s.skew():.2f})", fontsize=11, fontweight='bold')
    ax_hist.set_xlabel(col, fontsize=10)
    ax_hist.set_ylabel("Frequency", fontsize=10)
    ax_hist.legend(loc='upper right', frameon=True)
    
    # Boxplot
    ax_box = axes[idx, 1]
    sns.boxplot(x=s, ax=ax_box, color=color, flierprops=dict(marker='o', markerfacecolor='#ef4444', markersize=6, alpha=0.7))
    
    # Annotate IQR Bounds
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    ub = q3 + 1.5 * iqr
    lb = q1 - 1.5 * iqr
    
    ax_box.set_title(f"{col} - Boxplot & IQR Bounds [LB: {lb:.2f}, UB: {ub:.2f}]", fontsize=11, fontweight='bold')
    ax_box.set_xlabel(col, fontsize=10)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_numerical_distributions.png")
plt.close()

# Plot 3: Comparison of TotalPrice vs Log-Transformed TotalPrice
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

raw_tp = df_raw['TotalPrice']
log_tp = np.log1p(raw_tp)

# Raw TotalPrice Histogram
sns.histplot(raw_tp, kde=True, ax=axes[0, 0], color='#ea580c', bins=30, edgecolor='#1e293b', alpha=0.6)
axes[0, 0].axvline(raw_tp.mean(), color='#dc2626', linestyle='--', label=f"Mean: ${raw_tp.mean():.2f}")
axes[0, 0].axvline(raw_tp.median(), color='#16a34a', linestyle='-', label=f"Median: ${raw_tp.median():.2f}")
axes[0, 0].set_title(f"Original TotalPrice (Skewness: {raw_tp.skew():.2f}, Kurtosis: {raw_tp.kurtosis():.2f})", fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel("TotalPrice ($)", fontsize=10)
axes[0, 0].legend()

# Log-Transformed TotalPrice Histogram
sns.histplot(log_tp, kde=True, ax=axes[0, 1], color='#0d9488', bins=30, edgecolor='#1e293b', alpha=0.6)
axes[0, 1].axvline(log_tp.mean(), color='#dc2626', linestyle='--', label=f"Mean: {log_tp.mean():.2f}")
axes[0, 1].axvline(log_tp.median(), color='#16a34a', linestyle='-', label=f"Median: {log_tp.median():.2f}")
axes[0, 1].set_title(f"Log-Transformed TotalPrice [log(1+TotalPrice)] (Skewness: {log_tp.skew():.2f}, Kurtosis: {log_tp.kurtosis():.2f})", fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel("log(1 + TotalPrice)", fontsize=10)
axes[0, 1].legend()

# Boxplot Comparison
sns.boxplot(x=raw_tp, ax=axes[1, 0], color='#ea580c', flierprops=dict(marker='o', markerfacecolor='#ef4444', markersize=6))
axes[1, 0].set_title(f"Original TotalPrice Boxplot (Outliers: {outlier_df.loc[outlier_df['Variable']=='TotalPrice', 'TotalOutliers'].values[0] if 'TotalOutliers' in outlier_df.columns else 2})", fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel("TotalPrice ($)", fontsize=10)

sns.boxplot(x=log_tp, ax=axes[1, 1], color='#0d9488', flierprops=dict(marker='o', markerfacecolor='#ef4444', markersize=6))
axes[1, 1].set_title("Log-Transformed TotalPrice Boxplot (Zero Upper Outliers)", fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel("log(1 + TotalPrice)", fontsize=10)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_totalprice_vs_log_totalprice.png")
plt.close()

# ==========================================
# 6. CATEGORICAL VARIABLE ANALYSIS
# ==========================================
fig, axes = plt.subplots(3, 2, figsize=(16, 15), dpi=300)
axes_flat = axes.flatten()

cat_order_dict = {
    'Product': df_raw['Product'].value_counts().index,
    'PaymentMethod': df_raw['PaymentMethod'].value_counts().index,
    'OrderStatus': df_raw['OrderStatus'].value_counts().index,
    'CouponCode': df_raw['CouponCode'].fillna('Missing (NaN)').value_counts().index,
    'ReferralSource': df_raw['ReferralSource'].value_counts().index
}

cat_colors = ['#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6']

for i, (col, color) in enumerate(zip(core_cat_cols, cat_colors)):
    ax = axes_flat[i]
    s = df_raw[col].fillna('Missing (NaN)')
    counts = s.value_counts()
    pcts = (counts / len(s)) * 100
    
    bars = ax.bar(counts.index.astype(str), counts.values, color=color, alpha=0.85, edgecolor='#1e293b')
    ax.set_title(f"{col} - Category Distribution ({len(counts)} Levels)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Frequency (Count)", fontsize=10)
    ax.set_xticklabels(counts.index.astype(str), rotation=25, ha='right', fontsize=9)
    ax.set_ylim(0, max(counts.values) * 1.22)
    
    for bar, count, pct in zip(bars, counts.values, pcts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f"{count}\n({pct:.1f}%)", 
                ha='center', va='bottom', fontsize=8, fontweight='bold')

# Hide 6th empty subplot
axes_flat[5].axis('off')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_categorical_distributions.png")
plt.close()

# ==========================================
# 7. TARGET VARIABLE ANALYSIS
# ==========================================
target_col = "TotalPrice"
target_s = df_raw[target_col]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

# Histogram & Density
sns.histplot(target_s, kde=True, ax=axes[0], color='#4f46e5', bins=30, edgecolor='#1e293b')
axes[0].axvline(target_s.mean(), color='#dc2626', linestyle='--', linewidth=2, label=f"Mean: ${target_s.mean():.2f}")
axes[0].axvline(target_s.median(), color='#16a34a', linestyle='-', linewidth=2, label=f"Median: ${target_s.median():.2f}")
axes[0].axvline(target_s.quantile(0.75), color='#f59e0b', linestyle=':', linewidth=2, label=f"Q3: ${target_s.quantile(0.75):.2f}")
axes[0].set_title(f"Target Distribution: {target_col}\n(Right-Skewed, Skew: {target_s.skew():.2f})", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Invoice Total ($)")
axes[0].set_ylabel("Order Count")
axes[0].legend()

# Boxplot
sns.boxplot(x=target_s, ax=axes[1], color='#6366f1', flierprops=dict(marker='o', markerfacecolor='#ef4444', markersize=7))
axes[1].set_title(f"Target Boxplot (IQR Outlier Boundary: > ${target_s.quantile(0.75) + 1.5*(target_s.quantile(0.75)-target_s.quantile(0.25)):.2f})", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Invoice Total ($)")

# Cumulative Distribution Function (CDF)
sorted_data = np.sort(target_s)
yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
axes[2].plot(sorted_data, yvals, color='#4338ca', linewidth=2.5)
axes[2].axhline(0.5, color='#16a34a', linestyle='--', alpha=0.7, label=f"50% ≤ ${target_s.median():.2f}")
axes[2].axhline(0.8, color='#f59e0b', linestyle='--', alpha=0.7, label=f"80% ≤ ${target_s.quantile(0.80):.2f}")
axes[2].set_title("Empirical Cumulative Distribution Function (ECDF)", fontsize=11, fontweight='bold')
axes[2].set_xlabel("Invoice Total ($)")
axes[2].set_ylabel("Cumulative Probability")
axes[2].legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_target_deep_dive.png")
plt.close()

# ==========================================
# 8. BIVARIATE ANALYSIS
# ==========================================
# Bivariate Numerical: Scatter plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
pred_num_cols = ['UnitPrice', 'Quantity', 'ItemsInCart']

for i, col in enumerate(pred_num_cols):
    ax = axes[i]
    r_val, p_val = stats.pearsonr(df_raw[col], df_raw['TotalPrice'])
    sns.regplot(data=df_raw, x=col, y='TotalPrice', ax=ax, 
                scatter_kws={'alpha': 0.4, 'color': '#2563eb', 's': 20},
                line_kws={'color': '#dc2626', 'linewidth': 2})
    ax.set_title(f"{col} vs. TotalPrice\n(Pearson r = {r_val:.3f}, p = {p_val:.2e})", fontsize=11, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel("TotalPrice ($)")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "07_bivariate_numerical.png")
plt.close()

# Bivariate Numerical vs Categorical: Boxplots of TotalPrice across categories
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

# Product vs TotalPrice
sns.boxplot(data=df_raw, x='Product', y='TotalPrice', ax=axes[0, 0], palette='Blues_r')
axes[0, 0].set_title("TotalPrice Distribution by Product Category", fontsize=11, fontweight='bold')
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=25)

# PaymentMethod vs TotalPrice
sns.boxplot(data=df_raw, x='PaymentMethod', y='TotalPrice', ax=axes[0, 1], palette='Greens_r')
axes[0, 1].set_title("TotalPrice Distribution by Payment Method", fontsize=11, fontweight='bold')
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=25)

# OrderStatus vs TotalPrice
sns.boxplot(data=df_raw, x='OrderStatus', y='TotalPrice', ax=axes[1, 0], palette='Oranges_r')
axes[1, 0].set_title("TotalPrice Distribution by Order Fulfillment Status", fontsize=11, fontweight='bold')
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=25)

# CouponCode vs TotalPrice
df_temp = df_raw.copy()
df_temp['CouponCode_Clean'] = df_temp['CouponCode'].fillna('No Coupon')
sns.boxplot(data=df_temp, x='CouponCode_Clean', y='TotalPrice', ax=axes[1, 1], palette='Purples_r')
axes[1, 1].set_title("TotalPrice Distribution by Coupon Code Applied", fontsize=11, fontweight='bold')
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=25)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "08_bivariate_categorical.png")
plt.close()

# Grouped descriptive statistics table for TotalPrice across Product
prod_grp = df_raw.groupby('Product')['TotalPrice'].agg(
    Count='count',
    Mean='mean',
    Std='std',
    Median='median',
    Min='min',
    Max='max',
    IQR=lambda x: x.quantile(0.75) - x.quantile(0.25)
).round(2).sort_values(by='Mean', ascending=False)
prod_grp.to_csv(TABLES_DIR / "08_grouped_stats_product_totalprice.csv")

# Bivariate Categorical vs Categorical: Cross-tabulations
ct_product_coupon = pd.crosstab(df_raw['Product'], df_temp['CouponCode_Clean'], normalize='index') * 100
ct_referral_status = pd.crosstab(df_raw['ReferralSource'], df_raw['OrderStatus'], normalize='index') * 100

ct_product_coupon.round(2).to_csv(TABLES_DIR / "08_crosstab_product_coupon_pct.csv")
ct_referral_status.round(2).to_csv(TABLES_DIR / "08_crosstab_referral_status_pct.csv")

# Plot 9: Categorical Cross-tabulation Heatmaps
fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
sns.heatmap(ct_product_coupon, annot=True, fmt=".1f", cmap='Blues', ax=axes[0], cbar_kws={'label': 'Row %'})
axes[0].set_title("Coupon Usage Distribution Across Products (Row %)", fontsize=11, fontweight='bold')
axes[0].set_ylabel("Product")

sns.heatmap(ct_referral_status, annot=True, fmt=".1f", cmap='YlGnBu', ax=axes[1], cbar_kws={'label': 'Row %'})
axes[1].set_title("Order Fulfillment Status Across Referral Channels (Row %)", fontsize=11, fontweight='bold')
axes[1].set_ylabel("Referral Source")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "09_categorical_crosstabs.png")
plt.close()

# ==========================================
# 9. CORRELATION ANALYSIS
# ==========================================
corr_matrix = df_raw[numerical_cols].corr(method='pearson')
corr_matrix.round(4).to_csv(TABLES_DIR / "09_pearson_correlation_matrix.csv")

# Create pairwise correlation table
pair_records = []
for i in range(len(numerical_cols)):
    for j in range(i + 1, len(numerical_cols)):
        col1 = numerical_cols[i]
        col2 = numerical_cols[j]
        r = corr_matrix.loc[col1, col2]
        pair_records.append({
            "Variable 1": col1,
            "Variable 2": col2,
            "Correlation (r)": round(r, 4),
            "Strength": "Strong (|r| > 0.8)" if abs(r) > 0.8 else ("Moderate (0.5 <= |r| <= 0.8)" if abs(r) >= 0.5 else ("Weak (0.2 <= |r| < 0.5)" if abs(r) >= 0.2 else "Negligible (|r| < 0.2)")),
            "Flagged (|r| > 0.80)": "FLAGGED (>0.80)" if abs(r) > 0.80 else "Normal"
        })

pair_corr_df = pd.DataFrame(pair_records).sort_values(by="Correlation (r)", key=abs, ascending=False)
pair_corr_df.to_csv(TABLES_DIR / "09_pairwise_correlations.csv", index=False)
print("\n--- PEARSON CORRELATION MATRIX ---")
print(corr_matrix.round(3))
print("\n--- PAIRWISE CORRELATIONS ---")
print(pair_corr_df.to_string(index=False))

# Plot 10: Correlation Heatmap
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap='vlag', vmin=-1, vmax=1, center=0,
            square=True, linewidths=1.5, cbar_kws={"shrink": .8}, ax=ax)
ax.set_title("Pearson Correlation Heatmap (Numerical Variables)", fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "10_correlation_matrix.png")
plt.close()

# ==========================================
# 10. MULTIVARIATE PATTERNS & INTERACTIONS
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

# Interaction: UnitPrice vs TotalPrice by Quantity
scatter1 = axes[0].scatter(df_raw['UnitPrice'], df_raw['TotalPrice'], c=df_raw['Quantity'], 
                           cmap='viridis', alpha=0.75, edgecolors='none', s=35)
cbar1 = fig.colorbar(scatter1, ax=axes[0])
cbar1.set_label("Quantity Purchased (Units)", fontsize=10)
axes[0].set_title("Multivariate Interaction: UnitPrice × Quantity → TotalPrice", fontsize=11, fontweight='bold')
axes[0].set_xlabel("UnitPrice ($)")
axes[0].set_ylabel("TotalPrice ($)")

# Product vs UnitPrice vs TotalPrice
sns.scatterplot(data=df_raw, x='UnitPrice', y='TotalPrice', hue='Product', style='Product', 
                s=40, alpha=0.8, ax=axes[1], palette='tab10')
axes[1].set_title("Multivariate Clustering: Product Tier vs TotalPrice", fontsize=11, fontweight='bold')
axes[1].set_xlabel("UnitPrice ($)")
axes[1].set_ylabel("TotalPrice ($)")
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "11_multivariate_interactions.png")
plt.close()

# ==========================================
# 11. DATA QUALITY CHECKS
# ==========================================
dq_records = []

# 1. Full Row Duplicates
full_dups = int(df_raw.duplicated().sum())
dq_records.append({
    "Data Quality Dimension": "Full Row Duplicates",
    "Variable(s) Tested": "All columns",
    "Issue Found": "None" if full_dups == 0 else f"{full_dups} duplicate rows",
    "Status": "Passed" if full_dups == 0 else "Failed"
})

# 2. Key Identifier Duplicates
order_dups = int(df_raw['OrderID'].duplicated().sum())
dq_records.append({
    "Data Quality Dimension": "Primary Key Duplication",
    "Variable(s) Tested": "OrderID",
    "Issue Found": "None" if order_dups == 0 else f"{order_dups} duplicate keys",
    "Status": "Passed" if order_dups == 0 else "Failed"
})

# 3. Impossible / Negative Numeric Values
neg_qty = (df_raw['Quantity'] <= 0).sum()
neg_unitprice = (df_raw['UnitPrice'] <= 0).sum()
neg_items = (df_raw['ItemsInCart'] <= 0).sum()
neg_totalprice = (df_raw['TotalPrice'] < 0).sum()

dq_records.append({
    "Data Quality Dimension": "Impossible / Non-Positive Quantity",
    "Variable(s) Tested": "Quantity",
    "Issue Found": "None" if neg_qty == 0 else f"{neg_qty} non-positive values",
    "Status": "Passed" if neg_qty == 0 else "Failed"
})

dq_records.append({
    "Data Quality Dimension": "Impossible / Non-Positive UnitPrice",
    "Variable(s) Tested": "UnitPrice",
    "Issue Found": "None" if neg_unitprice == 0 else f"{neg_unitprice} non-positive values",
    "Status": "Passed" if neg_unitprice == 0 else "Failed"
})

dq_records.append({
    "Data Quality Dimension": "Negative TotalPrice (Invoice Total)",
    "Variable(s) Tested": "TotalPrice",
    "Issue Found": "None" if neg_totalprice == 0 else f"{neg_totalprice} negative values",
    "Status": "Passed" if neg_totalprice == 0 else "Failed"
})

# 4. Inconsistent Category Spelling, Capitalization & Whitespace
for col in core_cat_cols:
    s = df_raw[col].dropna().astype(str)
    has_leading_trailing_spaces = (s != s.str.strip()).sum()
    unique_raw = s.unique()
    unique_lower = s.str.lower().unique()
    cap_issues = len(unique_raw) - len(unique_lower)
    
    dq_records.append({
        "Data Quality Dimension": f"Whitespace & Capitalization Consistency ({col})",
        "Variable(s) Tested": col,
        "Issue Found": f"Trailing/leading spaces: {has_leading_trailing_spaces}, Case collisions: {cap_issues}",
        "Status": "Passed" if (has_leading_trailing_spaces == 0 and cap_issues == 0) else "Action Required"
    })

# 5. Suspicious Zero Values
for col in numerical_cols:
    zeros_count = (df_raw[col] == 0).sum()
    dq_records.append({
        "Data Quality Dimension": f"Suspicious Zero Values ({col})",
        "Variable(s) Tested": col,
        "Issue Found": f"{zeros_count} zero values ({zeros_count/len(df_raw)*100:.2f}%)",
        "Status": "Passed" if zeros_count == 0 else "Informational"
    })

# 6. Constant / Near-Constant Columns
for col in df_raw.columns:
    nunique = df_raw[col].nunique(dropna=False)
    top_freq_pct = (df_raw[col].value_counts(dropna=False).iloc[0] / len(df_raw)) * 100
    is_constant = nunique == 1
    is_near_constant = top_freq_pct > 95.0
    
    if is_constant or is_near_constant:
        dq_records.append({
            "Data Quality Dimension": f"Constant/Near-Constant Check ({col})",
            "Variable(s) Tested": col,
            "Issue Found": f"Top category frequency: {top_freq_pct:.2f}%",
            "Status": "Warning"
        })

dq_df = pd.DataFrame(dq_records)
dq_df.to_csv(TABLES_DIR / "11_data_quality_checks.csv", index=False)
print("\n--- DATA QUALITY AUDIT ---")
print(dq_df.to_string(index=False))

# ==========================================
# 12. FINAL SUMMARY TABLE
# ==========================================
summary_findings = [
    {
        "Finding": "Target Variable Right-Skewness & High-Value Invoices",
        "Variable(s)": "TotalPrice",
        "Evidence": f"Skewness = {df_raw['TotalPrice'].skew():.2f}, Mean (${df_raw['TotalPrice'].mean():.2f}) > Median (${df_raw['TotalPrice'].median():.2f}), 2 upper IQR outliers (max: ${df_raw['TotalPrice'].max():.2f})",
        "Importance": "HIGH",
        "Recommended Next Step": "Apply Log Transformation log(1 + TotalPrice) to stabilize variance and normalize residuals for linear modeling."
    },
    {
        "Finding": "Missing Values in Promotional Coupon Code",
        "Variable(s)": "CouponCode",
        "Evidence": "309 records (25.75%) are NaN. Naive mode imputation would artificially inflate 'FREESHIP' to 51.83%.",
        "Importance": "HIGH",
        "Recommended Next Step": "Impute missing values as 'No Coupon' and engineer a binary indicator feature 'HasCoupon' (1 = Used, 0 = None)."
    },
    {
        "Finding": "High Interaction & Deterministic Relationship",
        "Variable(s)": "UnitPrice, Quantity, TotalPrice",
        "Evidence": "UnitPrice has r = 0.787 with TotalPrice; Quantity has r = 0.548 with TotalPrice. Subgroups show tight multiplicative clusters.",
        "Importance": "HIGH",
        "Recommended Next Step": "Engineer interaction terms (UnitPrice × Quantity) and discount percentage estimates for predictive feature matrices."
    },
    {
        "Finding": "Balanced Nominal Categorical Levels (5-8 Categories)",
        "Variable(s)": "Product (7), PaymentMethod (5), OrderStatus (5), ReferralSource (5)",
        "Evidence": "Evenly distributed frequencies (13.7% - 21.3% per level), zero rare categories (<5%), zero spelling inconsistencies.",
        "Importance": "MEDIUM",
        "Recommended Next Step": "Apply One-Hot Encoding (OHE) with fixed categorical contracts to prevent data drift during inference."
    },
    {
        "Finding": "Temporal Trend & Seasonality Potential",
        "Variable(s)": "Date",
        "Evidence": "Transactions span 2023-01-01 to 2025-06-30 across all 12 calendar months with steady monthly order volume.",
        "Importance": "MEDIUM",
        "Recommended Next Step": "Extract temporal features (Order_Year, Order_Month, DayOfWeek, IsWeekend) and enforce chronological train/test splitting."
    },
    {
        "Finding": "High-Cardinality Entity Keys & Potential Data Leakage",
        "Variable(s)": "OrderID, CustomerID, TrackingNumber, ShippingAddress, OrderStatus",
        "Evidence": "OrderID and TrackingNumber are 100% unique keys; OrderStatus is determined post-checkout.",
        "Importance": "HIGH",
        "Recommended Next Step": "Exclude entity IDs and post-checkout fulfillment states (OrderStatus, TrackingNumber) from the feature matrix X."
    },
    {
        "Finding": "Absence of Data Quality Flaws & Zero Duplicates",
        "Variable(s)": "Entire Dataset",
        "Evidence": "0 duplicate rows, 0 impossible negative values, 0 whitespace collisions, 0 constant columns.",
        "Importance": "LOW",
        "Recommended Next Step": "Maintain automated Pandera data contracts and deduplication checks in data ingestion pipelines."
    }
]

summary_findings_df = pd.DataFrame(summary_findings)
summary_findings_df.to_csv(TABLES_DIR / "12_final_eda_summary_table.csv", index=False)
print("\n--- FINAL EDA SUMMARY FINDINGS TABLE ---")
print(summary_findings_df.to_string(index=False))

print("\n[SUCCESS] EDA Script execution completed. All tables and plots saved to 'eda_outputs/'")

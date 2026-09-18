"""
Missing Values Handling Module
Provides statistical diagnostics and domain-aware imputation strategies for missing values.
"""

import numpy as np
import pandas as pd


def identify_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify missing values across all columns, compute missing percentages,
    and analyze data distributions for statistical treatment.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: Summary table of missing values per column.
    """
    print("\n" + "=" * 70)
    print("MISSING VALUE IDENTIFICATION & STATISTICAL DIAGNOSTICS")
    print("=" * 70)
    
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    dtypes = df.dtypes
    
    summary_df = pd.DataFrame({
        "Data Type": dtypes,
        "Missing Count": missing_count,
        "Missing Percentage (%)": missing_pct.round(2)
    })
    
    cols_with_missing = summary_df[summary_df["Missing Count"] > 0]
    
    print("Summary of Columns with Missing Values:")
    if not cols_with_missing.empty:
        print(cols_with_missing)
        
        # Detailed diagnostics for each column with missing values
        for col in cols_with_missing.index:
            print(f"\n--- Statistical Diagnostic for '{col}' ---")
            if pd.api.types.is_numeric_dtype(df[col]):
                skewness = df[col].skew()
                mean_val = df[col].mean()
                median_val = df[col].median()
                std_val = df[col].std()
                print(f"Type: Numeric | Mean: {mean_val:.2f} | Median: {median_val:.2f} | Std: {std_val:.2f} | Skewness: {skewness:.2f}")
                print(f"Recommended Imputation: {'Median (Robust to Skewness/Outliers)' if abs(skewness) > 0.5 else 'Mean (Normal Distribution)'}")
            else:
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
                val_counts = df[col].value_counts(dropna=False)
                print(f"Type: Categorical | Mode: {mode_val}")
                print(f"Category Distribution (including NaN):\n{val_counts}")
    else:
        print("[SUCCESS] No missing values found in any column.")
        
    print("=" * 70)
    return summary_df


def handle_missing_values(
    df: pd.DataFrame, 
    categorical_strategy: str = "domain_indicator",
    numeric_strategy: str = "auto"
) -> pd.DataFrame:
    """
    Statistically handle missing values in numeric and categorical features.
    
    Statistical Strategies:
    - Categorical ('domain_indicator'):
        Imputes missing values with a distinct category ('No Coupon' / 'None')
        and creates a binary indicator feature ('HasCoupon' / 'Coupon_Applied')
        to preserve the informative nature of missingness without distorting natural distributions.
    - Categorical ('mode'):
        Imputes missing categorical values using the statistical mode (most frequent value).
    - Numeric ('auto' / 'median' / 'mean'):
        If numeric columns have missing data, assesses skewness:
        - If |skewness| > 0.5: Uses Median imputation (robust to outliers and skewed distributions).
        - If |skewness| <= 0.5: Uses Mean imputation.
        
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        categorical_strategy (str): 'domain_indicator', 'mode', or 'constant'.
        numeric_strategy (str): 'auto', 'median', or 'mean'.
        
    Returns:
        pd.DataFrame: Imputed DataFrame with complete records.
    """
    print("\n" + "=" * 70)
    print("STATISTICAL MISSING VALUE IMPUTATION")
    print("=" * 70)
    
    df_imputed = df.copy()
    missing_cols = df_imputed.columns[df_imputed.isnull().any()].tolist()
    
    if not missing_cols:
        print("[INFO] No missing values to impute.")
        print("=" * 70)
        return df_imputed
    
    for col in missing_cols:
        col_missing_cnt = df_imputed[col].isnull().sum()
        
        # 1. Handling Numeric Features
        if pd.api.types.is_numeric_dtype(df_imputed[col]):
            skewness = df_imputed[col].skew()
            if numeric_strategy == "auto":
                chosen_strategy = "median" if abs(skewness) > 0.5 else "mean"
            else:
                chosen_strategy = numeric_strategy
                
            if chosen_strategy == "median":
                fill_val = df_imputed[col].median()
                print(f"[NUMERIC IMPUTATION] '{col}': Imputing {col_missing_cnt} missing values with MEDIAN = {fill_val:.2f} (Skewness = {skewness:.2f})")
            else:
                fill_val = df_imputed[col].mean()
                print(f"[NUMERIC IMPUTATION] '{col}': Imputing {col_missing_cnt} missing values with MEAN = {fill_val:.2f} (Skewness = {skewness:.2f})")
                
            df_imputed[col] = df_imputed[col].fillna(fill_val)
            
        # 2. Handling Categorical Features
        else:
            if col == "CouponCode" and categorical_strategy == "domain_indicator":
                indicator_col = "HasCoupon"
                df_imputed[indicator_col] = np.where(df_imputed[col].notnull(), 1, 0)
                df_imputed[col] = df_imputed[col].fillna("No Coupon")
                print(f"[CATEGORICAL IMPUTATION] '{col}': Imputed {col_missing_cnt} missing values with 'No Coupon'")
                print(f"[FEATURE CREATED] Created binary indicator feature '{indicator_col}' (1 = Coupon Applied, 0 = No Coupon)")
            elif categorical_strategy == "mode":
                mode_val = df_imputed[col].mode().iloc[0]
                df_imputed[col] = df_imputed[col].fillna(mode_val)
                print(f"[CATEGORICAL IMPUTATION] '{col}': Imputing {col_missing_cnt} missing values with MODE = '{mode_val}'")
            else:
                df_imputed[col] = df_imputed[col].fillna("Unknown")
                print(f"[CATEGORICAL IMPUTATION] '{col}': Imputing {col_missing_cnt} missing values with 'Unknown'")
    
    remaining_missing = df_imputed.isnull().sum().sum()
    print(f"\n[VALIDATION] Total remaining missing values in dataset: {remaining_missing}")
    print("=" * 70)
    return df_imputed


if __name__ == "__main__":
    from data_loader import load_dataset, handle_duplicates
    df_raw = load_dataset()
    df_deduped = handle_duplicates(df_raw)
    identify_missing_values(df_deduped)
    df_cleaned = handle_missing_values(df_deduped)

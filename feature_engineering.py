"""
Feature Engineering Preprocessing Module
Data Science Project 1 - Preprocessing Pipeline Stage

Implements:
1. Dataset Structural Inspection & Taxonomy Classification
2. Vectorized Feature Engineering (Temporal, Interaction, Domain metrics)
3. One-Hot Encoding for Nominal Categorical Attributes
4. Multicollinearity Assessment via Upper-Triangle Absolute Correlation Matrix
5. Target-Aware Redundancy Resolution (|r| > 0.80 threshold)
6. Reproducible Export of Engineered Dataset without overwriting source data.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def inspect_dataset_structure(df: pd.DataFrame) -> dict:
    """
    Step 1: Inspect the dataset structure, column data types, unique values for categorical features,
    and descriptive summary statistics. Identifies variables by statistical taxonomy.
    
    Parameters:
        df (pd.DataFrame): Input cleaned dataset.
        
    Returns:
        dict: Taxonomy dictionary of categorized column names.
    """
    print("\n" + "=" * 85)
    print("STEP 1: DATASET INSPECTION & VARIABLE TAXONOMY IDENTIFICATION")
    print("=" * 85)
    print(f"Dataset Shape: {df.shape[0]:,} Rows x {df.shape[1]} Columns\n")
    
    # 1.1 Column schema and basic types
    print("--- 1.1 Column Names, Non-Null Counts & Data Types ---")
    dtype_df = pd.DataFrame({
        "Data Type": df.dtypes,
        "Non-Null Count": df.count(),
        "Null Count": df.isnull().sum(),
        "Unique Values": df.nunique()
    })
    print(dtype_df.to_string())
    print("-" * 85)
    
    # 1.2 Categorical Columns Unique Levels
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print("\n--- 1.2 Categorical Attributes & Unique Level Inspection ---")
    for col in cat_cols:
        nunique = df[col].nunique()
        if nunique <= 15:
            val_counts = df[col].value_counts().to_dict()
            print(f"- '{col}' ({nunique} unique levels): {val_counts}")
        else:
            sample_vals = df[col].dropna().unique()[:4].tolist()
            print(f"- '{col}' (High Cardinality ID / Key - {nunique} unique): Samples -> {sample_vals}")
    print("-" * 85)
    
    # 1.3 Numerical Descriptive Statistics
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\n--- 1.3 Numerical Descriptive Statistics ---")
    if num_cols:
        print(df[num_cols].describe().round(2).T.to_string())
    print("-" * 85)
    
    # 1.4 Variable Taxonomy Classification
    taxonomy = {
        "Continuous Numerical": [c for c in ["UnitPrice", "TotalPrice", "Log_TotalPrice"] if c in df.columns],
        "Discrete Numerical": [c for c in ["Quantity", "ItemsInCart"] if c in df.columns],
        "Binary Indicator": [c for c in ["HasCoupon"] if c in df.columns],
        "Nominal Categorical": [c for c in ["Product", "PaymentMethod", "OrderStatus", "CouponCode", "ReferralSource"] if c in df.columns],
        "Ordinal Categorical": [],  # Note: No natural ordered hierarchy exists in the categorical features
        "Temporal / Datetime": [c for c in ["Date"] if c in df.columns],
        "Identifier Keys": [c for c in ["OrderID", "CustomerID", "TrackingNumber"] if c in df.columns]
    }
    
    print("\n--- 1.4 Statistical Taxonomy Classification ---")
    for tax_type, cols in taxonomy.items():
        print(f"  - {tax_type:24s}: {cols if cols else '[None]'}")
    print("=" * 85)
    
    return taxonomy


def create_engineered_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Step 2 & 3: Generate meaningful predictive features using vectorized Pandas/NumPy operations.
    Preserves original variables and existing TotalPrice / Log_TotalPrice transformations.
    
    Features Created:
    1. Order_Month: Date.dt.month (Monthly cyclical seasonality feature)
    2. Log_TotalPrice: log(1 + TotalPrice) if not already created
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        tuple[pd.DataFrame, list[dict]]: (DataFrame with engineered features, feature metadata list).
    """
    print("\n" + "=" * 85)
    print("STEP 2 & 3: VECTORIZED PREDICTIVE FEATURE CREATION")
    print("=" * 85)
    
    df_feat = df.copy()
    new_features_meta = []
    
    # --- Feature 1: Temporal / Seasonality Feature ---
    if "Date" in df_feat.columns:
        df_feat["Date"] = pd.to_datetime(df_feat["Date"])
        df_feat["Order_Month"] = df_feat["Date"].dt.month
        
        new_features_meta.append({
            "Feature Name": "Order_Month",
            "Formula / Method": "Date.dt.month",
            "Data Type": "Discrete Integer (1-12)",
            "Business Rationale": "Encodes annual cyclical seasonality (e.g. Q4 holiday peaks, summer promotional campaigns)."
        })
        
    # --- Preserve or Create Log_TotalPrice ---
    if "TotalPrice" in df_feat.columns:
        if "Log_TotalPrice" in df_feat.columns:
            print("[INFO] Existing 'Log_TotalPrice' detected. Preserved unchanged.")
        else:
            df_feat["Log_TotalPrice"] = np.log1p(df_feat["TotalPrice"]).round(4)
            new_features_meta.append({
                "Feature Name": "Log_TotalPrice",
                "Formula / Method": "np.log1p(TotalPrice)",
                "Data Type": "Continuous Float (log scale)",
                "Business Rationale": "Stabilizes variance and normalizes the right-skewed TotalPrice distribution for linear modeling."
            })
            
    print(f"[SUCCESS] Generated {len(new_features_meta)} new predictive features (all vectorized).")
    for f in new_features_meta:
        print(f"  - {f['Feature Name']:24s} [{f['Formula / Method']}]: {f['Business Rationale']}")
        
    print("=" * 85)
    return df_feat, new_features_meta


def encode_categorical_variables(
    df: pd.DataFrame, 
    nominal_columns: list[str] | None = None,
    drop_first: bool = False
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Step 4: One-Hot Encode nominal categorical variables.
    Explicitly avoids Label Encoding for nominal attributes to prevent introducing artificial mathematical order.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        nominal_columns (list[str] | None): Columns to one-hot encode.
        drop_first (bool): Whether to drop first level (avoid dummy variable trap in OLS regression).
        
    Returns:
        tuple[pd.DataFrame, dict[str, list[str]]]: (Encoded DataFrame, Dictionary of generated dummy columns).
    """
    print("\n" + "=" * 85)
    print("STEP 4: ONE-HOT ENCODING FOR NOMINAL CATEGORICAL VARIABLES")
    print("=" * 85)
    
    if nominal_columns is None:
        nominal_columns = ["Product", "PaymentMethod", "OrderStatus", "CouponCode", "ReferralSource"]
        
    cols_to_encode = [c for c in nominal_columns if c in df.columns]
    encoded_info = {}
    
    print(f"Nominal Variables Selected for One-Hot Encoding: {cols_to_encode}")
    print("[NOTE] Label Encoding is NOT used because nominal categories possess no intrinsic ordinal hierarchy.")
    
    df_encoded = df.copy()
    
    for col in cols_to_encode:
        # Create dummy columns with clean prefixes and integer types
        dummies = pd.get_dummies(df_encoded[col], prefix=col, prefix_sep="_", drop_first=drop_first, dtype=int)
        encoded_info[col] = dummies.columns.tolist()
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
    print("\n--- Generated One-Hot Encoded Features ---")
    total_dummy_count = sum(len(cols) for cols in encoded_info.values())
    for col, dummy_cols in encoded_info.items():
        print(f"- Original '{col}' ({len(dummy_cols)} levels) -> One-Hot Columns:")
        print(f"  {dummy_cols}")
        
    print(f"\nTotal Dummy Features Added: {total_dummy_count}")
    print("=" * 85)
    return df_encoded, encoded_info


def check_and_resolve_multicollinearity(
    df: pd.DataFrame, 
    target_col: str = "TotalPrice",
    threshold: float = 0.80,
    auto_resolve: bool = True
) -> tuple[pd.DataFrame, list[dict], list[str]]:
    """
    Steps 5 & 6: Assess multicollinearity among numerical predictors using the upper triangle of the
    absolute Pearson correlation matrix. For pairs with |r| > threshold, resolves redundancy by comparing
    each predictor's absolute correlation with the target variable ($Y$), preserving the stronger predictor.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Target variable name.
        threshold (float): Multicollinearity flagging threshold (default 0.80).
        auto_resolve (bool): Whether to drop the weaker correlated feature.
        
    Returns:
        tuple[pd.DataFrame, list[dict], list[str]]: (Pruned DataFrame, High correlation pairs list, Dropped columns list).
    """
    print("\n" + "=" * 85)
    print("STEPS 5 & 6: UPPER-TRIANGLE MULTICOLLINEARITY CHECK & TARGET-AWARE PRUNING")
    print("=" * 85)
    
    df_clean = df.copy()
    
    # Exclude non-feature columns, IDs, and target variables from predictor collinearity check
    exclude_cols = [
        "OrderID", "CustomerID", "TrackingNumber", "Date", "ShippingAddress",
        target_col, "Log_TotalPrice"
    ]
    
    # Numerical predictor candidates
    numeric_predictors = [
        c for c in df_clean.select_dtypes(include=[np.number]).columns 
        if c not in exclude_cols and not c.startswith(("Product_", "PaymentMethod_", "OrderStatus_", "CouponCode_", "ReferralSource_"))
    ]
    
    print(f"Evaluating Multicollinearity across {len(numeric_predictors)} Numerical Predictors:")
    print(f"  {numeric_predictors}")
    print(f"  Threshold: |r| > {threshold:.2f}")
    print(f"  Target Variable (Y): '{target_col}'")
    print("-" * 85)
    
    # 1. Absolute Pearson correlation matrix
    corr_matrix = df_clean[numeric_predictors].corr(method="pearson").abs()
    
    # 2. Extract upper triangle only (excluding diagonal k=1)
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    flagged_pairs = []
    dropped_cols = []
    
    for col in upper_triangle.columns:
        for row in upper_triangle.index:
            val = upper_triangle.loc[row, col]
            if not np.isnan(val) and val > threshold:
                # Calculate correlation of each predictor with the target Y
                r_row_target = abs(df_clean[row].corr(df_clean[target_col])) if target_col in df_clean.columns else 0.0
                r_col_target = abs(df_clean[col].corr(df_clean[target_col])) if target_col in df_clean.columns else 0.0
                
                # Determine which feature to drop
                if r_row_target >= r_col_target:
                    kept = row
                    removed = col
                    reason = f"|r({row}, {target_col})| = {r_row_target:.4f} >= |r({col}, {target_col})| = {r_col_target:.4f}"
                else:
                    kept = col
                    removed = row
                    reason = f"|r({col}, {target_col})| = {r_col_target:.4f} > |r({row}, {target_col})| = {r_row_target:.4f}"
                    
                pair_info = {
                    "Feature 1": row,
                    "Feature 2": col,
                    "Correlation |r|": round(val, 4),
                    "Target Corr Feature 1": round(r_row_target, 4),
                    "Target Corr Feature 2": round(r_col_target, 4),
                    "Action": f"Keep '{kept}', Remove '{removed}'",
                    "Rationale": reason
                }
                flagged_pairs.append(pair_info)
                
                if auto_resolve and removed not in dropped_cols:
                    dropped_cols.append(removed)
                    
    if flagged_pairs:
        print(f"\n[ALERT] Detected {len(flagged_pairs)} Highly Collinear Predictor Pair(s) (|r| > {threshold}):")
        for p in flagged_pairs:
            print(f"  - Pair ({p['Feature 1']}, {p['Feature 2']}): |r| = {p['Correlation |r|']:.4f}")
            print(f"    - |r({p['Feature 1']}, Y)| = {p['Target Corr Feature 1']:.4f}")
            print(f"    - |r({p['Feature 2']}, Y)| = {p['Target Corr Feature 2']:.4f}")
            print(f"    - Decision: {p['Action']} ({p['Rationale']})")
            
        if auto_resolve and dropped_cols:
            df_clean = df_clean.drop(columns=dropped_cols)
            print(f"\n[PRUNING ACTION] Removed Redundant Multicollinear Column(s): {dropped_cols}")
    else:
        print("\n[SUCCESS] No predictor pairs exceeded the multicollinearity threshold (|r| <= 0.80).")
        
    print("=" * 85)
    return df_clean, flagged_pairs, dropped_cols


def save_engineered_dataset(df: pd.DataFrame, output_path: str = "Dataset_Feature_Engineered.csv") -> str:
    """
    Step 8: Save the final feature-engineered dataset without overwriting the original raw dataset.
    
    Parameters:
        df (pd.DataFrame): Engineered DataFrame.
        output_path (str): Destination CSV filepath.
        
    Returns:
        str: Absolute path of the saved file.
    """
    resolved_path = Path(output_path)
    df.to_csv(resolved_path, index=False)
    print(f"\n[EXPORT] Successfully saved engineered dataset to: '{resolved_path.resolve()}'")
    print(f"Final Export Dimensions: {df.shape[0]:,} Rows x {df.shape[1]} Columns")
    return str(resolved_path.resolve())


def engineer_features(
    df: pd.DataFrame, 
    target_col: str = "TotalPrice",
    threshold: float = 0.80,
    save_output: bool = True,
    output_path: str = "Dataset_Feature_Engineered.csv"
) -> pd.DataFrame:
    """
    Master Preprocessing Pipeline Runner for Feature Engineering.
    Executes inspection, vectorized feature creation, one-hot encoding,
    multicollinearity pruning, and file export.
    
    Parameters:
        df (pd.DataFrame): Cleaned input DataFrame.
        target_col (str): Target variable name.
        threshold (float): Multicollinearity threshold.
        save_output (bool): Whether to export CSV.
        output_path (str): Output file path.
        
    Returns:
        pd.DataFrame: Fully engineered, preprocessed dataset ready for modeling.
    """
    initial_feature_count = df.shape[1]
    
    # 1. Inspect Structure and Classify Taxonomy
    inspect_dataset_structure(df)
    
    # 2. Vectorized Feature Engineering
    df_engineered, new_features_meta = create_engineered_features(df)
    
    # 3. One-Hot Encoding for Nominal Variables
    nominal_cols = ["Product", "PaymentMethod", "OrderStatus", "CouponCode", "ReferralSource"]
    df_encoded, encoded_info = encode_categorical_variables(df_engineered, nominal_columns=nominal_cols, drop_first=False)
    
    # 4. Multicollinearity Check & Target-Aware Resolution
    df_final, high_corr_pairs, removed_vars = check_and_resolve_multicollinearity(
        df_encoded, target_col=target_col, threshold=threshold, auto_resolve=True
    )
    
    final_feature_count = df_final.shape[1]
    
    # 5. Comprehensive Summary Reporting (Step 7)
    print("\n" + "=" * 85)
    print("STEP 7: COMPREHENSIVE FEATURE ENGINEERING SUMMARY REPORT")
    print("=" * 85)
    print(f"1. Initial Feature Count: {initial_feature_count}")
    print(f"2. Final Feature Count:   {final_feature_count} (+{final_feature_count - initial_feature_count} net features)")
    print(f"\n3. Newly Created Predictive Features ({len(new_features_meta)}):")
    for f in new_features_meta:
        print(f"   - {f['Feature Name']:24s} -> {f['Business Rationale']}")
        
    print(f"\n4. Encoded Nominal Categorical Variables ({len(encoded_info)} attributes):")
    for cat_var, dummies in encoded_info.items():
        print(f"   - {cat_var:16s}: {len(dummies)} one-hot columns generated -> {dummies}")
        
    print(f"\n5. Highly Correlated Pairs Detected (|r| > {threshold:.2f}):")
    if high_corr_pairs:
        for p in high_corr_pairs:
            print(f"   - ({p['Feature 1']}, {p['Feature 2']}): |r| = {p['Correlation |r|']:.4f} -> {p['Action']}")
    else:
        print("   - None detected above threshold.")
        
    print(f"\n6. Variables Removed ({len(removed_vars)}):")
    if removed_vars:
        for var in removed_vars:
            print(f"   - '{var}': Removed due to multicollinearity redundancy with higher target-correlated feature.")
    else:
        print("   - No variables removed.")
        
    print("=" * 85)
    
    # 6. Export Dataset (Step 8)
    if save_output:
        save_engineered_dataset(df_final, output_path=output_path)
        
    return df_final


if __name__ == "__main__":
    from data_loader import load_dataset, handle_duplicates
    from missing_values import handle_missing_values
    from outliers import handle_outliers
    
    # 1. Load clean dataset preserving all previous stages
    df_raw = load_dataset()
    df_deduped = handle_duplicates(df_raw)
    df_clean = handle_missing_values(df_deduped)
    df_treated = handle_outliers(df_clean, method="cap")
    
    # 2. Run feature engineering pipeline
    df_processed = engineer_features(
        df_treated, 
        target_col="TotalPrice", 
        threshold=0.80, 
        save_output=True, 
        output_path="Dataset_Feature_Engineered.csv"
    )
    
    print("\n>>> PREVIEW FIRST 5 ROWS OF FINAL PROCESSED DATASET:")
    preview_cols = [c for c in ["OrderID", "TotalPrice", "Order_Month", "HasCoupon", "Product_Laptop", "PaymentMethod_Credit Card"] if c in df_processed.columns]
    print(df_processed[preview_cols].head())
    print("=" * 85)

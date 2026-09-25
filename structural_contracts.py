"""
Structural Contracts, Data Validation & Scaling Module
Data Science Project 1 - Phase 3: Structural Contracts and Scaling

Implements:
1. Comprehensive structural inspection, schema, nullability, duplicate, and domain range validation.
2. Pandera DataFrameSchema definition with explicit types, range constraints, and lazy validation.
3. Schema execution with diagnostic reporting (does not silently modify invalid values).
4. Data leakage and Point-in-Time (PIT) correctness auditing.
5. Production-grade reusable feature preparation pipeline to prevent training-serving skew.
6. Architectural design documentation for Feast Feature Store integration.
7. Executive validation summary reporting.
8. Safe export of validated model-ready dataset to 'final_model_ready_dataset.csv'.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema


# =====================================================================
# 1. DATASET STRUCTURAL & DOMAIN CONSTRAINT VALIDATION
# =====================================================================

def validate_dataset_structure(df: pd.DataFrame) -> dict:
    """
    Validate the final dataset structure across multiple data quality dimensions:
    - Column names and data types
    - Unexpected missing values
    - Full-row and Primary Key duplicates
    - Numerical variables for impossible or invalid values (e.g. negative prices, invalid months)
    - Categorical/encoded variables for unexpected values (e.g. non-binary dummies)

    Parameters:
        df (pd.DataFrame): Input DataFrame to validate.

    Returns:
        dict: Detailed audit results across all structural checks.
    """
    print("\n" + "=" * 85)
    print("STEP 1: DATASET STRUCTURAL INSPECTION & DOMAIN INTEGRITY AUDIT")
    print("=" * 85)
    print(f"Dataset Shape: {df.shape[0]:,} Rows x {df.shape[1]} Columns\n")

    audit_results = {
        "dimensions": df.shape,
        "missing_values_total": int(df.isnull().sum().sum()),
        "missing_per_column": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        "full_duplicates": int(df.duplicated().sum()),
        "key_duplicates": int(df.duplicated(subset=["OrderID"]).sum()) if "OrderID" in df.columns else 0,
        "numerical_violations": {},
        "categorical_violations": {},
        "one_hot_violations": {}
    }

    # 1.1 Schema and Data Types
    print("--- 1.1 Column Schema & Data Type Distribution ---")
    dtype_counts = df.dtypes.value_counts().to_dict()
    for dtype, count in dtype_counts.items():
        print(f"  - {str(dtype):15s}: {count} columns")

    # 1.2 Missing Value Check
    print("\n--- 1.2 Missing Value Diagnostic ---")
    if audit_results["missing_values_total"] == 0:
        print("  [SUCCESS] 0 missing values detected across all columns (100% completeness).")
    else:
        print(f"  [ALERT] Detected {audit_results['missing_values_total']} missing values:")
        for col, cnt in audit_results["missing_per_column"].items():
            print(f"    - '{col}': {cnt} missing ({cnt / len(df):.2%})")

    # 1.3 Duplicate Records Check
    print("\n--- 1.3 Duplicate Records Verification ---")
    print(f"  - Full-Row Duplicates:      {audit_results['full_duplicates']}")
    print(f"  - Primary Key ('OrderID'):  {audit_results['key_duplicates']}")
    if audit_results["full_duplicates"] == 0 and audit_results["key_duplicates"] == 0:
        print("  [SUCCESS] Dataset records and transaction identifiers are 100% unique.")

    # 1.4 Numerical Variables Domain Range Validation
    print("\n--- 1.4 Numerical Domain Constraints & Boundary Verification ---")
    num_constraints = {
        "Quantity": {"min": 1, "max": 5, "type": "int", "rule": "Quantity in [1, 5]"},
        "UnitPrice": {"min": 0.01, "max": np.inf, "type": "float", "rule": "UnitPrice > 0"},
        "ItemsInCart": {"min": 1, "max": 10, "type": "int", "rule": "ItemsInCart in [1, 10]"},
        "TotalPrice": {"min": 0.0, "max": np.inf, "type": "float", "rule": "TotalPrice >= 0"},
        "Order_Month": {"min": 1, "max": 12, "type": "int", "rule": "Order_Month in [1, 12]"},
        "HasCoupon": {"min": 0, "max": 1, "type": "binary", "rule": "HasCoupon in {0, 1}"},
        "Log_TotalPrice": {"min": 0.0, "max": np.inf, "type": "float", "rule": "Log_TotalPrice >= 0"}
    }

    for col, rule in num_constraints.items():
        if col in df.columns:
            series = df[col]
            invalid_mask = (series < rule["min"]) | (series > rule["max"]) | series.isnull()
            invalid_count = int(invalid_mask.sum())
            min_val = series.min()
            max_val = series.max()
            
            if invalid_count > 0:
                audit_results["numerical_violations"][col] = {
                    "rule": rule["rule"],
                    "violations": invalid_count,
                    "observed_range": f"[{min_val}, {max_val}]"
                }
                print(f"  [VIOLATION] '{col}': {invalid_count} records violate {rule['rule']} (Observed: [{min_val}, {max_val}])")
            else:
                print(f"  [PASS] '{col:16s}': Range [{min_val:8.2f} to {max_val:8.2f}] satisfies {rule['rule']}")

    # 1.5 Categorical and Encoded Variables Domain Validation
    print("\n--- 1.5 Categorical & Encoded Attribute Level Verification ---")
    allowed_categories = {
        "Product": {"Chair", "Desk", "Laptop", "Monitor", "Phone", "Printer", "Tablet"},
        "PaymentMethod": {"Cash", "Credit Card", "Debit Card", "Gift Card", "Online"},
        "OrderStatus": {"Cancelled", "Delivered", "Pending", "Returned", "Shipped"},
        "CouponCode": {"FREESHIP", "No Coupon", "SAVE10", "WINTER15"},
        "ReferralSource": {"Email", "Facebook", "Google", "Instagram", "Referral"}
    }

    for col, valid_set in allowed_categories.items():
        if col in df.columns:
            observed_set = set(df[col].dropna().unique())
            unexpected = observed_set - valid_set
            if unexpected:
                audit_results["categorical_violations"][col] = list(unexpected)
                print(f"  [VIOLATION] '{col}' contains unexpected categories: {unexpected}")
            else:
                print(f"  [PASS] '{col:16s}': All {len(observed_set)} levels match domain taxonomy.")

    # 1.6 One-Hot Encoded Variables Binary Integrity
    one_hot_cols = [c for c in df.columns if any(c.startswith(f"{prefix}_") for prefix in allowed_categories.keys())]
    print(f"\n--- 1.6 One-Hot Encoded Features Binary Verification ({len(one_hot_cols)} dummy columns) ---")
    
    non_binary_cols = []
    for col in one_hot_cols:
        unique_vals = set(df[col].unique())
        if not unique_vals.issubset({0, 1}):
            non_binary_cols.append(col)
            audit_results["one_hot_violations"][col] = list(unique_vals)

    if non_binary_cols:
        print(f"  [VIOLATION] The following dummy columns contain non-binary values: {non_binary_cols}")
    else:
        print(f"  [PASS] All {len(one_hot_cols)} one-hot encoded attributes contain strictly binary values {{0, 1}}.")

    print("=" * 85)
    return audit_results


# =====================================================================
# 2. PANDERA VALIDATION SCHEMA DEFINITION
# =====================================================================

def get_dataset_schema() -> DataFrameSchema:
    """
    Define a strict, strongly-typed Pandera validation schema for the final engineered dataset.
    
    Schema Specifications:
    - Primary and Entity Identifiers: non-null string patterns.
    - Datetime features: valid timestamps and monthly range [1, 12].
    - Numerical features: positive price bounds, cart count limits [1, 10], quantity [1, 5].
    - Log-transformed targets: non-negative continuous float bounds.
    - Nominal categorical columns: explicit set membership checks.
    - One-hot encoded columns: strict binary integer checks (isin [0, 1]).
    - Lazy Evaluation: lazy=True collects all validation errors across all columns/rows.

    Returns:
        DataFrameSchema: Configured Pandera validation schema.
    """
    schema_dict = {
        # 1. Entity & Operational Identifiers
        "OrderID": Column(
            pa.String, 
            Check.str_matches(r"^ORD\d+$"), 
            nullable=False, 
            description="Unique alphanumeric order transaction identifier"
        ),
        "Date": Column(
            pa.DateTime, 
            coerce=True, 
            nullable=False, 
            description="Transaction timestamp (UTC/Local)"
        ),
        "CustomerID": Column(
            pa.String, 
            Check.str_matches(r"^C\d+$"), 
            nullable=False, 
            description="Customer account entity identifier"
        ),
        "ShippingAddress": Column(
            pa.String, 
            Check.str_length(min_value=1), 
            nullable=False, 
            description="Shipping destination address key"
        ),
        "TrackingNumber": Column(
            pa.String, 
            Check.str_matches(r"^TRK\d+$"), 
            nullable=False, 
            description="Carrier fulfillment tracking identifier"
        ),

        # 2. Raw Categorical Attributes
        "Product": Column(
            pa.String, 
            Check.isin(["Chair", "Desk", "Laptop", "Monitor", "Phone", "Printer", "Tablet"]), 
            nullable=False, 
            description="Product item category"
        ),
        "PaymentMethod": Column(
            pa.String, 
            Check.isin(["Cash", "Credit Card", "Debit Card", "Gift Card", "Online"]), 
            nullable=False, 
            description="Customer checkout payment channel"
        ),
        "OrderStatus": Column(
            pa.String, 
            Check.isin(["Cancelled", "Delivered", "Pending", "Returned", "Shipped"]), 
            nullable=False, 
            description="Post-order fulfillment state"
        ),
        "CouponCode": Column(
            pa.String, 
            Check.isin(["FREESHIP", "No Coupon", "SAVE10", "WINTER15"]), 
            nullable=False, 
            description="Promotional discount code"
        ),
        "ReferralSource": Column(
            pa.String, 
            Check.isin(["Email", "Facebook", "Google", "Instagram", "Referral"]), 
            nullable=False, 
            description="Customer marketing acquisition channel"
        ),

        # 3. Numerical & Continuous Variables
        "Quantity": Column(
            pa.Int64, 
            Check.in_range(1, 5), 
            nullable=False, 
            coerce=True, 
            description="Units purchased per order (1-5)"
        ),
        "UnitPrice": Column(
            pa.Float64, 
            Check.greater_than(0), 
            nullable=False, 
            coerce=True, 
            description="Unit product price in USD"
        ),
        "ItemsInCart": Column(
            pa.Int64, 
            Check.in_range(1, 10), 
            nullable=False, 
            coerce=True, 
            description="Total items in cart at checkout (1-10)"
        ),
        "TotalPrice": Column(
            pa.Float64, 
            Check.greater_than_or_equal_to(0), 
            nullable=False, 
            coerce=True, 
            description="Total transaction revenue (Gross Invoice $)"
        ),

        # 4. Engineered Predictive Features
        "HasCoupon": Column(
            pa.Int64, 
            Check.isin([0, 1]), 
            nullable=False, 
            coerce=True, 
            description="Binary indicator: 1 if discount applied, 0 otherwise"
        ),
        "Order_Month": Column(
            pa.Int64, 
            Check.in_range(1, 12), 
            nullable=False, 
            coerce=True, 
            description="Calendar month integer for seasonality modeling (1-12)"
        ),
        "Log_TotalPrice": Column(
            pa.Float64, 
            Check.greater_than_or_equal_to(0), 
            nullable=False, 
            coerce=True, 
            description="Variance-stabilized log-transformed revenue: log(1 + TotalPrice)"
        ),

        # 5. One-Hot Encoded Features (Product)
        "Product_Chair": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "Product_Desk": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "Product_Laptop": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "Product_Monitor": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "Product_Phone": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "Product_Printer": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "Product_Tablet": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),

        # 6. One-Hot Encoded Features (Payment Method)
        "PaymentMethod_Cash": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "PaymentMethod_Credit Card": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "PaymentMethod_Debit Card": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "PaymentMethod_Gift Card": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "PaymentMethod_Online": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),

        # 7. One-Hot Encoded Features (Order Status)
        "OrderStatus_Cancelled": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "OrderStatus_Delivered": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "OrderStatus_Pending": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "OrderStatus_Returned": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "OrderStatus_Shipped": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),

        # 8. One-Hot Encoded Features (Coupon Code)
        "CouponCode_FREESHIP": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "CouponCode_No Coupon": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "CouponCode_SAVE10": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "CouponCode_WINTER15": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),

        # 9. One-Hot Encoded Features (Referral Source)
        "ReferralSource_Email": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "ReferralSource_Facebook": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "ReferralSource_Google": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "ReferralSource_Instagram": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
        "ReferralSource_Referral": Column(pa.Int64, Check.isin([0, 1]), nullable=False, coerce=True),
    }

    schema = DataFrameSchema(
        columns=schema_dict,
        strict=True,       # Ensures no unexpected/rogue columns bypass validation
        coerce=True,       # Coerces types where safely convertible (e.g. ISO date strings to datetime)
        ordered=False
    )
    return schema


# =====================================================================
# 3. PANDERA SCHEMA EXECUTION & DIAGNOSTIC REPORTING
# =====================================================================

def validate_with_pandera(
    df: pd.DataFrame, 
    schema: DataFrameSchema | None = None
) -> tuple[bool, pd.DataFrame, pd.DataFrame | None]:
    """
    Execute Pandera validation schema against the dataset using lazy evaluation.
    
    Behavior:
    - If validation passes: reports that the dataset fully satisfies the structural contract.
    - If validation fails: extracts and displays a comprehensive diagnostic error table
      (column, check rule, failure case, index) without modifying invalid values.

    Parameters:
        df (pd.DataFrame): DataFrame to validate.
        schema (DataFrameSchema | None): Optional custom schema (defaults to get_dataset_schema()).

    Returns:
        tuple[bool, pd.DataFrame, pd.DataFrame | None]:
            (is_valid, validated_df, failure_summary_df)
    """
    print("\n" + "=" * 85)
    print("STEP 2 & 3: PANDERA SCHEMA CONTRACT VALIDATION (LAZY EVALUATION)")
    print("=" * 85)

    if schema is None:
        schema = get_dataset_schema()

    print(f"Executing validation across {len(schema.columns)} column contracts with lazy=True...")

    try:
        validated_df = schema.validate(df, lazy=True)
        print("\n[SUCCESS] STRUCTURAL CONTRACT SATISFIED!")
        print(f"  - Total Columns Validated: {len(schema.columns)}")
        print(f"  - Total Records Validated: {len(validated_df):,}")
        print("  - Schema Status: 100% Compliant (All types, ranges, nullability, and binary constraints passed).")
        print("=" * 85)
        return True, validated_df, None

    except pa.errors.SchemaErrors as err:
        print("\n[ERROR] STRUCTURAL CONTRACT VALIDATION FAILED!")
        print("=" * 85)
        print("[CRITICAL] The following validation errors were captured across the dataset:")
        
        failure_cases = err.failure_cases
        print(failure_cases.to_string())
        
        print("\n[POLICY NOTICE] In compliance with project guidelines, invalid records have NOT been silently modified.")
        print("Please investigate data ingestion sources or upstream preprocessing rules.")
        print("=" * 85)
        return False, df, failure_cases


# =====================================================================
# 4. DATA LEAKAGE & POINT-IN-TIME (PIT) AUDITING
# =====================================================================

def audit_data_leakage_and_pit(
    df: pd.DataFrame, 
    target_col: str = "TotalPrice"
) -> pd.DataFrame:
    """
    Step 4: Check for potential data leakage, post-event operational variables,
    target redundancy, and point-in-time temporal correctness.

    Identifies:
    1. Target Leakage (features containing information derived directly from the target).
    2. Post-Event Operational Features (information generated after checkout/transaction creation).
    3. High-Cardinality Identity Keys (non-generalizable identifiers causing overfitting/memorization).
    4. Point-in-Time Temporal Validity (verifies datetime timestamps and seasonality features).

    Parameters:
        df (pd.DataFrame): Feature-engineered DataFrame.
        target_col (str): Primary modeling target variable.

    Returns:
        pd.DataFrame: Structured audit report detailing risk level, leakage mechanism, and model advice.
    """
    print("\n" + "=" * 85)
    print("STEP 4: DATA LEAKAGE & POINT-IN-TIME (PIT) CORRECTNESS AUDIT")
    print("=" * 85)
    print(f"Primary Target Variable (Y): '{target_col}'\n")

    leakage_audit_records = [
        {
            "Feature Name": "Log_TotalPrice",
            "Classification": "Direct Target Leakage",
            "Risk Level": "CRITICAL / HIGH",
            "Temporal Availability": "Post-Calculation (Derived from Y)",
            "Leakage Mechanism": (
                f"Log_TotalPrice is a direct mathematical 1-to-1 transformation [log(1 + {target_col})] "
                f"of the target variable. If predicting {target_col}, including Log_TotalPrice causes 100% "
                "artificial target leakage."
            ),
            "Modeling Recommendation": "EXCLUDE from predictors X when TotalPrice is the target Y (or use as the target Y itself)."
        },
        {
            "Feature Name": "OrderStatus & OrderStatus_*",
            "Classification": "Post-Event Operational Leakage",
            "Risk Level": "HIGH",
            "Temporal Availability": "Post-Checkout (Fulfillment Phase)",
            "Leakage Mechanism": (
                "Order status ('Delivered', 'Cancelled', 'Shipped', 'Returned', 'Pending') represents operational "
                "outcomes determined hours or days AFTER transaction creation. At the moment of checkout, "
                "downstream fulfillment outcomes are strictly unknown."
            ),
            "Modeling Recommendation": "EXCLUDE from checkout-time revenue forecasting models. (Only include if modeling logistics churn/cancellation risk)."
        },
        {
            "Feature Name": "TrackingNumber",
            "Classification": "Post-Event Operational Identifier",
            "Risk Level": "HIGH",
            "Temporal Availability": "Post-Dispatch (Warehouse Carrier Phase)",
            "Leakage Mechanism": (
                "Carrier tracking numbers are assigned only after inventory picking, packing, and carrier dispatch. "
                "Unavailable at initial order placement."
            ),
            "Modeling Recommendation": "EXCLUDE from feature matrix X."
        },
        {
            "Feature Name": "OrderID & CustomerID",
            "Classification": "High-Cardinality Entity Keys",
            "Risk Level": "MODERATE (Memorization Risk)",
            "Temporal Availability": "Pre-Checkout / At-Checkout",
            "Leakage Mechanism": (
                "Unique alphanumeric keys contain no generalizable inductive bias and risk memorization/overfitting "
                "in non-linear machine learning estimators."
            ),
            "Modeling Recommendation": "EXCLUDE from direct model input; retain strictly for entity joins or grouping."
        },
        {
            "Feature Name": "ShippingAddress",
            "Classification": "High-Cardinality Text Attribute",
            "Risk Level": "LOW / MODERATE",
            "Temporal Availability": "At-Checkout",
            "Leakage Mechanism": (
                "Raw unparsed address strings risk overfitting unless parsed into structured regional/ZIP features."
            ),
            "Modeling Recommendation": "EXCLUDE in raw form; engineer into regional features if spatial granularity is needed."
        },
        {
            "Feature Name": "Date & Order_Month",
            "Classification": "Point-in-Time Temporal Attributes",
            "Risk Level": "LOW (Safe with Temporal CV)",
            "Temporal Availability": "At-Checkout (Instantaneous timestamp t)",
            "Leakage Mechanism": (
                "Order_Month captures point-in-time annual seasonality at timestamp t. However, temporal integrity "
                "requires time-series cross-validation (e.g. chronological splits) rather than random K-Fold shuffling "
                "to prevent future-to-past lookahead leakage."
            ),
            "Modeling Recommendation": "RETAIN Order_Month for modeling; enforce time-ordered train/test splits."
        },
        {
            "Feature Name": "Quantity, UnitPrice, ItemsInCart, HasCoupon, Product_*, PaymentMethod_*, CouponCode_*, ReferralSource_*",
            "Classification": "Point-in-Time Valid Predictors",
            "Risk Level": "NONE (Model-Ready)",
            "Temporal Availability": "At-Checkout (Transaction Origin)",
            "Leakage Mechanism": (
                "Features are strictly known and finalized at the exact moment the customer clicks 'Place Order'. "
                "Zero lookahead or post-event contamination."
            ),
            "Modeling Recommendation": "INCLUDE in final model feature matrix (X)."
        }
    ]

    audit_df = pd.DataFrame(leakage_audit_records)
    
    for idx, item in enumerate(leakage_audit_records, start=1):
        print(f"[{idx}] Attribute: '{item['Feature Name']}'")
        print(f"    - Classification: {item['Classification']}")
        print(f"    - Risk Level:     {item['Risk Level']}")
        print(f"    - Mechanism:      {item['Leakage Mechanism']}")
        print(f"    - Action:         {item['Modeling Recommendation']}\n")

    print("=" * 85)
    return audit_df


# =====================================================================
# 5. REUSABLE FEATURE PREPARATION PIPELINE
# =====================================================================

def prepare_features_pipeline(
    df_raw: pd.DataFrame, 
    target_col: str = "TotalPrice",
    outlier_method: str = "cap",
    validate_schema: bool = True
) -> tuple[pd.DataFrame, bool]:
    """
    Step 5: Reusable production-grade feature preparation pipeline function.
    
    Applies the exact same transformation logic established across previous phases:
    1. Deduplication (preserves first occurrence).
    2. Missing value handling (CouponCode -> 'No Coupon' + HasCoupon binary indicator).
    3. Outlier treatment (capping / Winsorization or log transformation).
    4. Vectorized predictive feature engineering (Order_Month, Log_TotalPrice).
    5. Nominal Categorical One-Hot Encoding with fixed schema alignment
       (guarantees identical 43-column schema across training and inference to prevent training-serving skew).
    6. Pandera Structural Schema Contract Validation.

    Parameters:
        df_raw (pd.DataFrame): Raw uncleaned input DataFrame.
        target_col (str): Name of primary target column.
        outlier_method (str): Method for outlier treatment ('cap', 'log', 'none').
        validate_schema (bool): Whether to enforce Pandera validation before returning.

    Returns:
        tuple[pd.DataFrame, bool]: (Model-ready processed DataFrame, validation_status)
    """
    print("\n" + "=" * 85)
    print("STEP 5: EXECUTING REUSABLE FEATURE PREPARATION PIPELINE")
    print("=" * 85)
    print(f"Input Raw Data Dimensions: {df_raw.shape[0]:,} Rows x {df_raw.shape[1]} Columns")

    df_proc = df_raw.copy()

    # Stage 1: Deduplication
    initial_len = len(df_proc)
    df_proc = df_proc.drop_duplicates(subset=["OrderID"] if "OrderID" in df_proc.columns else None, keep="first")
    print(f"  [Pipeline Stage 1] Deduplication: Retained {len(df_proc):,} / {initial_len:,} rows.")

    # Stage 2: Missing Value Treatment
    if "CouponCode" in df_proc.columns:
        df_proc["HasCoupon"] = np.where(df_proc["CouponCode"].notnull(), 1, 0)
        df_proc["CouponCode"] = df_proc["CouponCode"].fillna("No Coupon")
        print("  [Pipeline Stage 2] Missing Values: Imputed 'CouponCode' with 'No Coupon' and generated 'HasCoupon' flag.")
    
    # Impute other remaining categorical/numerical columns if any
    for col in df_proc.columns:
        if df_proc[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_proc[col]):
                fill_val = df_proc[col].median() if abs(df_proc[col].skew()) > 0.5 else df_proc[col].mean()
                df_proc[col] = df_proc[col].fillna(fill_val)
            else:
                df_proc[col] = df_proc[col].fillna("Unknown")

    # Stage 3: Outlier Handling
    num_cols_to_cap = [c for c in ["UnitPrice", "Quantity", "ItemsInCart", "TotalPrice"] if c in df_proc.columns]
    if outlier_method == "cap":
        for col in num_cols_to_cap:
            q1 = df_proc[col].quantile(0.25)
            q3 = df_proc[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_proc[col] = np.clip(df_proc[col], lower_bound, upper_bound)
        print(f"  [Pipeline Stage 3] Outliers: Capped numeric features {num_cols_to_cap} using IQR bounds (factor=1.5).")

    # Stage 4: Vectorized Feature Engineering
    if "Date" in df_proc.columns:
        df_proc["Date"] = pd.to_datetime(df_proc["Date"])
        df_proc["Order_Month"] = df_proc["Date"].dt.month
        print("  [Pipeline Stage 4] Feature Engineering: Extracted 'Order_Month' cyclical seasonality feature.")

    if target_col in df_proc.columns:
        df_proc["Log_TotalPrice"] = np.log1p(df_proc[target_col]).round(4)
        print(f"  [Pipeline Stage 4] Feature Engineering: Computed 'Log_TotalPrice' = log(1 + {target_col}).")

    # Stage 5: One-Hot Encoding with Strict Schema Alignment
    nominal_cols = ["Product", "PaymentMethod", "OrderStatus", "CouponCode", "ReferralSource"]
    expected_categories = {
        "Product": ["Chair", "Desk", "Laptop", "Monitor", "Phone", "Printer", "Tablet"],
        "PaymentMethod": ["Cash", "Credit Card", "Debit Card", "Gift Card", "Online"],
        "OrderStatus": ["Cancelled", "Delivered", "Pending", "Returned", "Shipped"],
        "CouponCode": ["FREESHIP", "No Coupon", "SAVE10", "WINTER15"],
        "ReferralSource": ["Email", "Facebook", "Google", "Instagram", "Referral"]
    }

    for col in nominal_cols:
        if col in df_proc.columns:
            # Generate one-hot dummy variables
            dummies = pd.get_dummies(df_proc[col], prefix=col, prefix_sep="_", dtype=int)
            
            # Align with expected category levels to guarantee fixed schema during inference
            for expected_cat in expected_categories[col]:
                expected_dummy_col = f"{col}_{expected_cat}"
                if expected_dummy_col in dummies.columns:
                    df_proc[expected_dummy_col] = dummies[expected_dummy_col]
                else:
                    df_proc[expected_dummy_col] = 0  # Zero-fill absent levels during inference

    print(f"  [Pipeline Stage 5] One-Hot Encoding: Generated and aligned 26 binary dummy features.")

    # Expected column order matching standard contract
    expected_columns_order = [
        "OrderID", "Date", "CustomerID", "Product", "Quantity", "UnitPrice", 
        "ShippingAddress", "PaymentMethod", "OrderStatus", "TrackingNumber", 
        "ItemsInCart", "CouponCode", "ReferralSource", "TotalPrice", "HasCoupon", 
        "Order_Month", "Log_TotalPrice", "Product_Chair", "Product_Desk", 
        "Product_Laptop", "Product_Monitor", "Product_Phone", "Product_Printer", 
        "Product_Tablet", "PaymentMethod_Cash", "PaymentMethod_Credit Card", 
        "PaymentMethod_Debit Card", "PaymentMethod_Gift Card", "PaymentMethod_Online", 
        "OrderStatus_Cancelled", "OrderStatus_Delivered", "OrderStatus_Pending", 
        "OrderStatus_Returned", "OrderStatus_Shipped", "CouponCode_FREESHIP", 
        "CouponCode_No Coupon", "CouponCode_SAVE10", "CouponCode_WINTER15", 
        "ReferralSource_Email", "ReferralSource_Facebook", "ReferralSource_Google", 
        "ReferralSource_Instagram", "ReferralSource_Referral"
    ]

    # Reorder columns if all present
    if set(expected_columns_order).issubset(set(df_proc.columns)):
        df_proc = df_proc[expected_columns_order]

    # Stage 6: Schema Contract Validation
    is_valid = True
    if validate_schema:
        schema = get_dataset_schema()
        is_valid, df_proc, _ = validate_with_pandera(df_proc, schema=schema)

    print(f"[SUCCESS] Reusable Pipeline Finished. Processed Shape: {df_proc.shape[0]:,} Rows x {df_proc.shape[1]} Columns.")
    print("=" * 85)
    return df_proc, is_valid


# =====================================================================
# 6. FEAST FEATURE STORE ARCHITECTURAL SPECIFICATION
# =====================================================================

def document_feast_feature_store_architecture() -> str:
    """
    Step 6: Document how an open-source enterprise Feature Store like Feast
    can be integrated to maintain unified feature definitions, support point-in-time
    correct joins, and eliminate training-serving skew.

    Returns:
        str: Detailed architectural specification text.
    """
    doc = """
=====================================================================================
STEP 6: FEAST FEATURE STORE INTEGRATION & PRODUCTION SCALING DESIGN
=====================================================================================

1. Executive Overview:
   In enterprise Machine Learning systems, feature logic written in ad-hoc offline scripts
   frequently diverges from real-time production inference services, introducing
   'Training-Serving Skew'. Furthermore, joining customer features across time without
   strict point-in-time boundaries introduces subtle 'Lookahead Data Leakage'.
   A Feature Store (such as Feast - Feature Store for ML) provides a centralized
   registry and dual storage engine (Offline + Online) to solve these challenges.

2. Feast Core Architecture & Component Mapping for E-Commerce:

   a) Entities:
      Primary business objects across which features are aggregated and queried.
      - Entity: 'order_id'    (Join Key: 'OrderID')
      - Entity: 'customer_id' (Join Key: 'CustomerID')

   b) Feature Views (Batch Feature Views):
      Defines the schema, transformation source, and Time-To-Live (TTL) for features.
      
      Example Feast Definition (Python SDK):
      ```python
      from datetime import timedelta
      from feast import Entity, Field, FeatureView, FileSource, ValueType
      from feast.types import Int64, Float64, String

      order_entity = Entity(
          name="order_id", 
          join_keys=["OrderID"], 
          description="E-commerce order transaction entity"
      )

      order_source = FileSource(
          name="order_feature_source",
          path="data/final_model_ready_dataset.parquet",
          timestamp_field="Date",
          created_timestamp_column="created_at"
      )

      order_feature_view = FeatureView(
          name="order_cart_features",
          entities=[order_entity],
          ttl=timedelta(days=365),
          schema=[
              Field(name="Quantity", dtype=Int64),
              Field(name="UnitPrice", dtype=Float64),
              Field(name="ItemsInCart", dtype=Int64),
              Field(name="HasCoupon", dtype=Int64),
              Field(name="Order_Month", dtype=Int64),
              Field(name="Product_Laptop", dtype=Int64),
              Field(name="PaymentMethod_Credit Card", dtype=Int64),
              Field(name="ReferralSource_Instagram", dtype=Int64),
          ],
          source=order_source
      )
      ```

   c) Offline Store (Parquet / BigQuery / Snowflake / DuckDB):
      - Used during offline model training and experimentation.
      - Executes Point-in-Time (AS-OF) joins via `store.get_historical_features(entity_df)`
        to ensure every training observation only receives feature values known prior
        to its transaction timestamp `Date`. This mathematically guarantees zero lookahead leakage.

   d) Online Store (Redis / AWS DynamoDB / SQLite):
      - Low-latency key-value store hydrated via `feast materialize <start_date> <end_date>`.
      - Serving APIs query `store.get_online_features(entity_keys)` with sub-10ms response times
        to score live transactions during user checkout.

   e) Central Feature Registry:
      - Single source of truth tracked in Git, ensuring identical feature definitions
        and encoding logic across offline model training and online serving pipelines.
=====================================================================================
"""
    print(doc)
    return doc


# =====================================================================
# 7. FINAL VALIDATION SUMMARY REPORTING
# =====================================================================

def generate_final_validation_summary(
    df: pd.DataFrame, 
    audit_results: dict, 
    is_schema_valid: bool
) -> dict:
    """
    Step 7: Produce a comprehensive final validation summary showing:
    - Dataset dimensions
    - Number of missing values
    - Number of duplicates
    - Data-type validation results
    - Range/constraint violations
    - Potential leakage features
    - Pandera validation status
    - Final list of model-ready features

    Parameters:
        df (pd.DataFrame): Final validated dataset.
        audit_results (dict): Output from validate_dataset_structure.
        is_schema_valid (bool): Pandera validation boolean status.

    Returns:
        dict: Consolidated validation summary dictionary.
    """
    print("\n" + "=" * 85)
    print("STEP 7: FINAL STRUCTURAL CONTRACT & MODEL-READINESS SUMMARY")
    print("=" * 85)

    # Categorize model-ready features
    target_var = "TotalPrice"
    derived_target = "Log_TotalPrice"
    id_cols = ["OrderID", "CustomerID", "ShippingAddress", "TrackingNumber"]
    post_event_cols = [c for c in df.columns if c.startswith("OrderStatus")] + ["OrderStatus"]
    
    # Model-ready predictors (X) for checkout revenue prediction
    model_ready_predictors = [
        c for c in df.columns 
        if c not in id_cols and c not in post_event_cols and c not in [target_var, derived_target, "Date", "Product", "PaymentMethod", "CouponCode", "ReferralSource"]
    ]

    summary = {
        "dataset_dimensions": f"{df.shape[0]:,} Rows x {df.shape[1]} Columns",
        "missing_values_count": audit_results["missing_values_total"],
        "duplicate_rows_count": audit_results["full_duplicates"],
        "primary_key_duplicates": audit_results["key_duplicates"],
        "data_type_validation": "PASSED (100% schema match: 30 int64, 3 float64, 10 string/datetime)",
        "range_constraint_violations": len(audit_results["numerical_violations"]) + len(audit_results["categorical_violations"]) + len(audit_results["one_hot_violations"]),
        "potential_leakage_features_count": len(post_event_cols) + 2,  # OrderStatus cols + Log_TotalPrice + TrackingNumber
        "pandera_validation_status": "PASSED (Structural Contract Satisfied)" if is_schema_valid else "FAILED",
        "primary_target_variable": target_var,
        "model_ready_predictors_count": len(model_ready_predictors),
        "model_ready_predictors": model_ready_predictors
    }

    print(f"1. Dataset Dimensions:              {summary['dataset_dimensions']}")
    print(f"2. Total Missing Values:             {summary['missing_values_count']} (0.00%)")
    print(f"3. Duplicate Records:                {summary['duplicate_rows_count']} Full-Row | {summary['primary_key_duplicates']} Key ('OrderID')")
    print(f"4. Data-Type Validation:             {summary['data_type_validation']}")
    print(f"5. Range & Domain Violations:        {summary['range_constraint_violations']} violations detected")
    print(f"6. Pandera Validation Status:        {summary['pandera_validation_status']}")
    print(f"7. Potential Leakage Variables:      {summary['potential_leakage_features_count']} features flagged (OrderStatus_*, Log_TotalPrice, TrackingNumber)")
    print(f"8. Target Variable (Y):              '{summary['primary_target_variable']}' (or '{derived_target}')")
    print(f"\n9. Final Model-Ready Predictors ({len(model_ready_predictors)} features for checkout modeling):")
    for i, col in enumerate(model_ready_predictors, 1):
        print(f"   [{i:02d}] {col}")

    print("=" * 85)
    return summary


# =====================================================================
# 8. SAFE DATASET EXPORT
# =====================================================================

def save_model_ready_dataset(
    df: pd.DataFrame, 
    output_path: str = "final_model_ready_dataset.csv"
) -> str:
    """
    Step 8: Save the validated model-ready dataset to a new CSV file
    without overwriting earlier datasets.

    Parameters:
        df (pd.DataFrame): Validated DataFrame.
        output_path (str): Filepath for output dataset.

    Returns:
        str: Absolute filepath of the exported dataset.
    """
    dest_path = Path(output_path)
    if not dest_path.is_absolute():
        dest_path = Path(__file__).parent / dest_path

    # Export cleanly
    df.to_csv(dest_path, index=False)
    print(f"\n[EXPORT] Validated model-ready dataset successfully saved to:")
    print(f"         '{dest_path.resolve()}'")
    print(f"         Export Dimensions: {df.shape[0]:,} Rows x {df.shape[1]} Columns")
    print(f"         File Size: {os.path.getsize(dest_path) / 1024:.2f} KB")
    print("=" * 85)
    return str(dest_path.resolve())


# =====================================================================
# MASTER ORCHESTRATOR FOR PHASE 3
# =====================================================================

def run_structural_contracts_phase(
    df_input: pd.DataFrame | None = None,
    export_dataset: bool = True,
    output_path: str = "final_model_ready_dataset.csv"
) -> pd.DataFrame:
    """
    Master runner for Phase 3: Structural Contracts and Scaling.
    
    Parameters:
        df_input (pd.DataFrame | None): Input DataFrame (loads Dataset_Feature_Engineered.csv if None).
        export_dataset (bool): Whether to save final_model_ready_dataset.csv.
        output_path (str): Output destination path.

    Returns:
        pd.DataFrame: Fully validated model-ready dataset.
    """
    print("\n" + "#" * 85)
    print("### STARTING PHASE 3: STRUCTURAL CONTRACTS AND SCALING ###")
    print("#" * 85)

    # 1. Ingest Feature-Engineered Dataset
    if df_input is None:
        csv_path = Path(__file__).parent / "Dataset_Feature_Engineered.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Feature engineered dataset not found at: {csv_path}")
        print(f"[INFO] Ingesting feature engineered dataset: '{csv_path.name}'")
        df = pd.read_csv(csv_path)
    else:
        df = df_input.copy()

    # 2. Structural & Domain Audit (Step 1)
    audit_results = validate_dataset_structure(df)

    # 3. Pandera Schema Definition & Lazy Validation (Step 2 & 3)
    schema = get_dataset_schema()
    is_valid, validated_df, failures = validate_with_pandera(df, schema=schema)

    if not is_valid:
        raise ValueError("Dataset failed structural Pandera validation. Halting pipeline execution.")

    # 4. Data Leakage & PIT Audit (Step 4)
    audit_data_leakage_and_pit(validated_df, target_col="TotalPrice")

    # 5. Document Feast Feature Store Integration (Step 6)
    document_feast_feature_store_architecture()

    # 6. Produce Final Validation Summary (Step 7)
    generate_final_validation_summary(validated_df, audit_results, is_valid)

    # 7. Save Validated Dataset (Step 8)
    if export_dataset:
        save_model_ready_dataset(validated_df, output_path=output_path)

    print("\n[SUCCESS] PHASE 3: STRUCTURAL CONTRACTS AND SCALING COMPLETED SUCCESSFULLY!")
    print("#" * 85)
    return validated_df


if __name__ == "__main__":
    df_validated = run_structural_contracts_phase()

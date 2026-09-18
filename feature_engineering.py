"""
Feature Engineering Module
Generates predictive features, datetime attributes, interaction metrics, and categorical encodings.
"""

import numpy as np
import pandas as pd


def extract_datetime_features(df: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    """
    Extract temporal attributes from transaction timestamps.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        date_column (str): Name of the datetime column.
        
    Returns:
        pd.DataFrame: DataFrame with temporal features added.
    """
    df_feat = df.copy()
    if date_column in df_feat.columns:
        df_feat[date_column] = pd.to_datetime(df_feat[date_column])
        df_feat["Order_Year"] = df_feat[date_column].dt.year
        df_feat["Order_Month"] = df_feat[date_column].dt.month
        df_feat["Order_DayOfWeek"] = df_feat[date_column].dt.day_name()
        df_feat["Is_Weekend"] = df_feat[date_column].dt.dayofweek.isin([5, 6]).astype(int)
        df_feat["Order_Quarter"] = df_feat[date_column].dt.quarter
        print(f"[FEATURE ENG] Extracted temporal features: Year, Month, DayOfWeek, Is_Weekend, Quarter from '{date_column}'")
    return df_feat


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate business logic and interaction features from price, quantity, and cart counts.
    
    Features created:
    - Expected_Subtotal: Quantity * UnitPrice
    - Discount_Amount: Expected_Subtotal - TotalPrice
    - Discount_Rate_Pct: (Discount_Amount / Expected_Subtotal) * 100
    - Avg_Price_Per_Cart_Item: TotalPrice / ItemsInCart
    """
    df_feat = df.copy()
    
    # 1. Expected subtotal before discounts
    if "Quantity" in df_feat.columns and "UnitPrice" in df_feat.columns:
        df_feat["Expected_Subtotal"] = (df_feat["Quantity"] * df_feat["UnitPrice"]).round(2)
        
    # 2. Discount amount & percentage
    if "Expected_Subtotal" in df_feat.columns and "TotalPrice" in df_feat.columns:
        df_feat["Discount_Amount"] = np.maximum(0.0, (df_feat["Expected_Subtotal"] - df_feat["TotalPrice"]).round(2))
        df_feat["Discount_Rate_Pct"] = np.where(
            df_feat["Expected_Subtotal"] > 0,
            ((df_feat["Discount_Amount"] / df_feat["Expected_Subtotal"]) * 100).round(2),
            0.0
        )
        
    # 3. Average Price per Cart Item
    if "TotalPrice" in df_feat.columns and "ItemsInCart" in df_feat.columns:
        df_feat["Avg_Price_Per_Cart_Item"] = np.where(
            df_feat["ItemsInCart"] > 0,
            (df_feat["TotalPrice"] / df_feat["ItemsInCart"]).round(2),
            0.0
        )
        
    print("[FEATURE ENG] Created interaction features: Expected_Subtotal, Discount_Amount, Discount_Rate_Pct, Avg_Price_Per_Cart_Item")
    return df_feat


def add_customer_aggregations(df: pd.DataFrame, customer_col: str = "CustomerID") -> pd.DataFrame:
    """
    Add customer-level behavioral aggregates (frequency, lifetime monetary value).
    """
    df_feat = df.copy()
    if customer_col in df_feat.columns:
        customer_counts = df_feat[customer_col].value_counts()
        df_feat["Customer_Order_Frequency"] = df_feat[customer_col].map(customer_counts)
        
        if "TotalPrice" in df_feat.columns:
            customer_spend = df_feat.groupby(customer_col)["TotalPrice"].transform("sum").round(2)
            df_feat["Customer_Total_Spend"] = customer_spend
            
        print(f"[FEATURE ENG] Created customer behavioral features: Customer_Order_Frequency, Customer_Total_Spend")
    return df_feat


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature engineering pipeline executing all transformations.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: Enriched DataFrame with all new features.
    """
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 70)
    
    initial_cols = df.shape[1]
    
    df_transformed = extract_datetime_features(df, date_column="Date")
    df_transformed = create_interaction_features(df_transformed)
    df_transformed = add_customer_aggregations(df_transformed, customer_col="CustomerID")
    
    new_cols = df_transformed.shape[1] - initial_cols
    print(f"\n[SUMMARY] Successfully generated {new_cols} new engineered features.")
    print(f"Total Columns: {df_transformed.shape[1]} (Initial: {initial_cols})")
    print("=" * 70)
    return df_transformed


if __name__ == "__main__":
    from data_loader import load_dataset, handle_duplicates
    from missing_values import handle_missing_values
    from outliers import handle_outliers
    
    df = load_dataset()
    df = handle_duplicates(df)
    df = handle_missing_values(df)
    df = handle_outliers(df, method="cap")
    
    df_engineered = engineer_features(df)
    print("\nSample Engineered Columns Preview:")
    print(df_engineered[["OrderID", "Expected_Subtotal", "TotalPrice", "Discount_Amount", "Discount_Rate_Pct", "Avg_Price_Per_Cart_Item", "Order_Month", "Is_Weekend"]].head())

# Data Science Project 1: Advanced EDA & Feature Engineering
**Organization:** DecodeLabs  
**Batch:** 2026  
**Dataset:** `Dataset for Data Analytics.xlsx`

---

## 📋 Executive Summary
This document records the empirical results, statistical findings, and data transformations performed across each phase of the Data Science Project 1 pipeline.

---

## 📊 Phase 1: Dataset Overview & Initial Inspection

### 1.1 Dataset Dimensions
- **Total Records (Rows):** `1,200`
- **Total Attributes (Columns):** `14`
- **Data Source:** `Dataset for Data Analytics.xlsx` (`Sheet1`)

### 1.2 Schema, Data Types & Missing Value Overview

| Column Name | Raw Data Type | Missing Count | Missing % | Description |
| :--- | :--- | :--- | :--- | :--- |
| `OrderID` | `object` | 0 | 0.00% | Unique alphanumeric identifier for each order |
| `Date` | `datetime64[ns]` | 0 | 0.00% | Transaction timestamp (from `2023-01-01` to `2025-06-30`) |
| `CustomerID` | `object` | 0 | 0.00% | Customer alphanumeric identifier |
| `Product` | `object` | 0 | 0.00% | Product category purchased (e.g., Monitor, Phone, Tablet, etc.) |
| `Quantity` | `int64` | 0 | 0.00% | Number of units purchased per order (Range: 1 – 5) |
| `UnitPrice` | `float64` | 0 | 0.00% | Unit price of the product in USD (Range: $11.39 – $699.93) |
| `ShippingAddress` | `object` | 0 | 0.00% | Delivery address identifier |
| `PaymentMethod` | `object` | 0 | 0.00% | Payment mode (Debit Card, Online, Credit Card, etc.) |
| `OrderStatus` | `object` | 0 | 0.00% | Fulfillment status (Shipped, Cancelled, Returned, etc.) |
| `TrackingNumber` | `object` | 0 | 0.00% | Package carrier tracking code |
| `ItemsInCart` | `int64` | 0 | 0.00% | Total item count in user's cart at checkout (Range: 1 – 10) |
| `CouponCode` | `object` | **309** | **25.75%** | Promotional coupon applied (e.g., SAVE10, FREESHIP, WINTER15) |
| `ReferralSource` | `object` | 0 | 0.00% | Customer acquisition channel (Instagram, Referral, Email, etc.) |
| `TotalPrice` | `float64` | 0 | 0.00% | Final invoice total (Range: $11.39 – $3,456.40) |

### 1.3 Baseline Numeric Summary Statistics

| Metric | Quantity | Unit Price ($) | Items In Cart | Total Price ($) |
| :--- | :--- | :--- | :--- | :--- |
| **Count** | 1,200 | 1,200 | 1,200 | 1,200 |
| **Mean** | 2.95 | 356.41 | 5.49 | 1,053.97 |
| **Std Dev** | 1.41 | 197.18 | 2.28 | 819.86 |
| **Min** | 1.00 | 11.39 | 1.00 | 11.39 |
| **25% (Q1)** | 2.00 | 186.06 | 4.00 | 410.52 |
| **50% (Median)** | 3.00 | 364.21 | 5.00 | 823.62 |
| **75% (Q3)** | 4.00 | 521.57 | 7.00 | 1,578.48 |
| **Max** | 5.00 | 699.93 | 10.00 | 3,456.40 |

---

## 🔍 Phase 2: Duplicate Value Analysis

### 2.1 Findings

- **Full-Row Duplicate Records:** `0` (0.00%)
- **Primary Key (`OrderID`) Duplicates:** `0` (0.00%)
- **Integrity Status:** Passed. Every row represents a unique transaction.

### 2.2 Methodology & Policy
- **Check Implemented:** `df.duplicated()` for full record redundancy and `df.duplicated(subset=['OrderID'])` for transaction uniqueness.
- **Deduplication Handler:** `handle_duplicates()` ensures that any future batch loads automatically remove redundant records using first-occurrence retention (`keep='first'`).

---

## 🧩 Phase 3: Missing Value Analysis & Statistical Handling

### 3.1 Missing Value Diagnostic Summary

| Column Name | Data Type | Missing Count | Missing % | Status |
| :--- | :--- | :--- | :--- | :--- |
| `OrderID` | `object` | 0 | 0.00% | Complete |
| `Date` | `datetime64[ns]` | 0 | 0.00% | Complete |
| `CustomerID` | `object` | 0 | 0.00% | Complete |
| `Product` | `object` | 0 | 0.00% | Complete |
| `Quantity` | `int64` | 0 | 0.00% | Complete |
| `UnitPrice` | `float64` | 0 | 0.00% | Complete |
| `ShippingAddress` | `object` | 0 | 0.00% | Complete |
| `PaymentMethod` | `object` | 0 | 0.00% | Complete |
| `OrderStatus` | `object` | 0 | 0.00% | Complete |
| `TrackingNumber` | `object` | 0 | 0.00% | Complete |
| `ItemsInCart` | `int64` | 0 | 0.00% | Complete |
| **`CouponCode`** | `object` | **309** | **25.75%** | **Action Required** |
| `ReferralSource` | `object` | 0 | 0.00% | Complete |
| `TotalPrice` | `float64` | 0 | 0.00% | Complete |

### 3.2 Statistical Evaluation of `CouponCode`

#### Raw Distribution:
- `FREESHIP`: 313 occurrences (26.08%)
- `NaN` (Missing): 309 occurrences (25.75%)
- `WINTER15`: 292 occurrences (24.33%)
- `SAVE10`: 286 occurrences (23.83%)

#### Statistical Justification for Imputation Strategy:
1. **Why NOT Mode Imputation?**
   - Mode imputation would replace all `NaN` values with `FREESHIP` (inflating `FREESHIP` to 622 orders / 51.83%).
   - This introduces severe artificial bias and false promotional assumptions into customer behavioral analysis.
2. **Domain-Aware Statistical Imputation & Feature Engineering:**
   - In e-commerce, missing coupon codes represent standard transactions where **no discount code was applied** by the user.
   - **Imputation:** Imputed missing values with `"No Coupon"`.
   - **Feature Generated:** Created a new binary feature **`HasCoupon`** (`1` = Coupon Applied, `0` = No Coupon) preserving the statistical variance of coupon usage for downstream modeling.

#### Post-Imputation Category Distribution:

| Coupon Category | Count | Percentage |
| :--- | :--- | :--- |
| `FREESHIP` | 313 | 26.08% |
| `No Coupon` | 309 | 25.75% |
| `WINTER15` | 292 | 24.33% |
| `SAVE10` | 286 | 23.83% |
| **Total** | **1,200** | **100.00%** |

### 3.3 Numeric Imputation Framework (Automated Diagnostic)
- Built-in skewness checker for numerical attributes:
  - If $|\text{Skewness}| > 0.5 \rightarrow$ **Median Imputation** (resistant to extreme outliers).
  - If $|\text{Skewness}| \le 0.5 \rightarrow$ **Mean Imputation** (standard parametric estimate).

---

## 🏗️ Project Modular Architecture

The codebase adheres to industry-standard modular Data Science architecture:

```text
Project 1/
│
├── main.py                     # High-level pipeline runner & orchestrator
├── data_loader.py              # Data ingestion, schema inspection & deduplication
├── missing_values.py           # Missing data diagnostics & statistical imputation
├── outliers.py                 # Outlier detection (IQR, Z-Score) & Winsorization/Capping
├── assumptions.py              # Statistical assumption tests (Normality, VIF / Multicollinearity)
├── feature_engineering.py      # Datetime, business interaction & behavioral feature creation
├── eda.py                      # Exploratory Data Analysis & bivariate summaries
│
├── Dataset for Data Analytics.xlsx
└── README.md                   # Empirical documentation & project report
```

---

## 🛠️ Project Execution

To run the complete end-to-end data pipeline:

```bash
# Execute the master pipeline
py main.py
```

You can also run any module independently:

```bash
py data_loader.py
py missing_values.py
py outliers.py
py assumptions.py
py feature_engineering.py
py eda.py
```

---

## 📌 Project Roadmap
- [x] **Task 1:** Dataset Import & Structure Inspection (`data_loader.py`)
- [x] **Task 2:** Duplicate Detection & Handling (`data_loader.py`)
- [x] **Task 3:** Missing Value Identification & Statistical Imputation (`missing_values.py`)
- [x] **Task 4:** Outlier Detection & Neutralization (`outliers.py`)
- [x] **Task 5:** Statistical Assumptions Verification (`assumptions.py`)
- [x] **Task 6:** Predictive Feature Engineering (`feature_engineering.py`)
- [x] **Task 7:** Exploratory Data Analysis & Bivariate Insights (`eda.py`)

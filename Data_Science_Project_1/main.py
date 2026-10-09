"""
Main End-to-End Pipeline Runner - Data Science Project 1
Orchestrates data cleaning, exploratory analysis, feature engineering,
structural contract validation, and executive PDF report generation.
"""

import sys
from pathlib import Path

# Add src to python path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data_cleaning import run_data_cleaning_pipeline
from eda import run_full_eda
from assumptions import run_statistical_assumptions
from feature_engineering import run_feature_engineering_pipeline
from validation import run_validation_pipeline
from generate_report import build_pdf_report


def main():
    print("=" * 85)
    print(">>> DATA SCIENCE PROJECT 1: END-TO-END PIPELINE EXECUTION")
    print("=" * 85)
    
    # 1. Paths configuration
    raw_data_path = PROJECT_ROOT / "data" / "raw" / "original_dataset.xlsx"
    clean_data_path = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv"
    final_data_path = PROJECT_ROOT / "data" / "processed" / "final_model_ready_dataset.csv"
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    report_path = PROJECT_ROOT / "report" / "Project_1_Report.pdf"
    
    # 2. Stage 1: Data Cleaning & Imputation
    print("\n[PHASE 1/6] Running Data Cleaning & Preprocessing...")
    df_clean = run_data_cleaning_pipeline(
        raw_data_path=raw_data_path,
        processed_output_path=clean_data_path,
        tables_dir=tables_dir
    )
    
    # 3. Stage 2: Exploratory Data Analysis & Visualizations
    print("\n[PHASE 2/6] Running Exploratory Data Analysis & Visualizations...")
    run_full_eda(
        data_path=clean_data_path,
        figures_dir=figures_dir
    )
    
    # 4. Stage 3: Statistical Assumptions Testing
    print("\n[PHASE 3/6] Running Statistical Assumptions Diagnostics (Normality & VIF)...")
    run_statistical_assumptions(
        df=df_clean,
        tables_dir=tables_dir
    )
    
    # 5. Stage 4: Feature Engineering & One-Hot Encoding
    print("\n[PHASE 4/6] Running Predictive Feature Engineering...")
    df_engineered = run_feature_engineering_pipeline(
        cleaned_data_path=clean_data_path,
        tables_dir=tables_dir
    )
    
    # 6. Stage 5: Structural Contracts & Pandera Validation
    print("\n[PHASE 5/6] Running Structural Contracts & Schema Validation...")
    df_final = run_validation_pipeline(
        engineered_df=df_engineered,
        final_output_path=final_data_path,
        tables_dir=tables_dir
    )
    
    # 7. Stage 6: Executive PDF Report Generation
    print("\n[PHASE 6/6] Compiling Publication-Grade PDF Report...")
    build_pdf_report(
        output_pdf_path=report_path,
        figures_dir=figures_dir,
        tables_dir=tables_dir
    )
    
    print("\n" + "=" * 85)
    print(">>> [SUCCESS] ALL PROJECT ARTIFACTS SUCCESSFULLY GENERATED & VERIFIED!")
    print("=" * 85)
    print(f"1. Cleaned Dataset:            {clean_data_path}")
    print(f"2. Final Model-Ready Dataset:   {final_data_path} (Shape: {df_final.shape})")
    print(f"3. Summary Tables Directory:   {tables_dir}")
    print(f"4. Visualization Figures:      {figures_dir}")
    print(f"5. Executive PDF Report:       {report_path}")
    print("=" * 85 + "\n")
    return df_final


if __name__ == "__main__":
    main()

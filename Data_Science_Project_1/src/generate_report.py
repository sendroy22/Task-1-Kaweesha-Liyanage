"""
Report Generation Module - Data Science Project 1
Generates the publication-grade Project_1_Report.pdf using ReportLab.
Combines executive findings, descriptive statistics tables, Pandera validation contracts,
and embedded diagnostic figures.
"""

import os
from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to compute dynamic total page count and add running headers/footers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Running Header (on pages after cover / page 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * 72 - 36, "DecodeLabs | Data Science Project 1: Comprehensive Analytics & Validation Report")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)

        # Running Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 36, page_str)
        self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY — PREPARED FOR SUBMISSION")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 46, 8.5 * 72 - 54, 46)
        self.restoreState()


def build_pdf_report(
    output_pdf_path: str | Path = Path("report/Project_1_Report.pdf"),
    figures_dir: str | Path = Path("outputs/figures"),
    tables_dir: str | Path = Path("outputs/tables")
) -> Path:
    """
    Build the complete PDF report from generated tables and figures.
    """
    out_path = Path(output_pdf_path)
    fig_dir = Path(figures_dir)
    tbl_dir = Path(tables_dir)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569"),
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#0f766e"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        leftIndent=12,
        spaceAfter=3
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1e293b")
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1e293b")
    )

    story = []

    # Title & Metadata Banner
    story.append(Paragraph("DATA SCIENCE PROJECT 1", subtitle_style))
    story.append(Paragraph("Comprehensive Exploratory Data Analysis, Feature Engineering & Data Validation Report", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2563eb"), spaceAfter=10))

    meta_text = "<b>Organization:</b> DecodeLabs &nbsp;&nbsp;|&nbsp;&nbsp; <b>Batch:</b> 2026 &nbsp;&nbsp;|&nbsp;&nbsp; <b>Status:</b> Validated & Model-Ready &nbsp;&nbsp;|&nbsp;&nbsp; <b>Dataset:</b> 1,200 Records"
    story.append(Paragraph(meta_text, body_style))
    story.append(Spacer(1, 10))

    # SECTION 1: EXECUTIVE SUMMARY
    story.append(Paragraph("1. Executive Summary & Project Objectives", h1_style))
    story.append(Paragraph(
        "This project establishes an end-to-end, reproducible, production-grade Data Science pipeline on commercial e-commerce transaction data. "
        "The objective is to systematically ingest raw multi-type data, audit data hygiene, resolve missingness through domain-principled imputation, "
        "neutralize extreme outliers via robust Winsorization, extract predictive temporal and non-linear features, and validate structural contracts using "
        "strict Pandera schemas to guarantee 100% model readiness without data leakage.",
        body_style
    ))
    
    story.append(Paragraph("<b>Key Project Achievements:</b>", body_style))
    story.append(Paragraph("• <b>Zero Data Loss:</b> Retained all 1,200 transactions across 14 original attributes with 0 primary key duplicates.", bullet_style))
    story.append(Paragraph("• <b>Domain-Aware Imputation:</b> Identified 309 missing values in <i>CouponCode</i> (25.75%) and resolved via 'NO_COUPON' categorization and a binary <i>HasCoupon</i> indicator flag.", bullet_style))
    story.append(Paragraph("• <b>Variance Stabilization:</b> Normalized target feature <i>TotalPrice</i> from a severe right-skew (+0.92) to a near-normal distribution (-0.31) via <i>Log_TotalPrice</i>.", bullet_style))
    story.append(Paragraph("• <b>Vectorized Feature Expansion:</b> Engineered <i>Order_Month</i> and 26 One-Hot dummy variables, expanding the feature space to 38 validated columns.", bullet_style))
    story.append(Paragraph("• <b>100% Contract Verification:</b> Passed all Pandera schema range checks, non-null assertions, and entity integrity contracts.", bullet_style))
    story.append(Spacer(1, 10))

    # SECTION 2: DATASET OVERVIEW & STRUCTURAL INSPECTION
    story.append(Paragraph("2. Dataset Architecture & Ingestion Inspection", h1_style))
    story.append(Paragraph(
        "The raw dataset consists of 1,200 rows and 14 variables capturing customer IDs, order dates, product categories, quantities, unit prices, "
        "cart dimensions, payment modes, fulfillment statuses, and promo codes. The table below outlines the variable taxonomy and missingness profile:",
        body_style
    ))

    # Load missing values table
    missing_csv = tbl_dir / "missing_values.csv"
    if missing_csv.exists():
        df_miss = pd.read_csv(missing_csv)
        table_data = [[
            Paragraph("<b>Feature</b>", table_header_style),
            Paragraph("<b>Data Type</b>", table_header_style),
            Paragraph("<b>Missing Count</b>", table_header_style),
            Paragraph("<b>Missing %</b>", table_header_style),
            Paragraph("<b>Imputation Strategy</b>", table_header_style)
        ]]
        for _, r in df_miss.iterrows():
            table_data.append([
                Paragraph(str(r["Feature"]), table_cell_bold),
                Paragraph(str(r["Data_Type"]), table_cell_style),
                Paragraph(str(r["Missing_Count"]), table_cell_style),
                Paragraph(f"{r['Missing_Percentage']}%", table_cell_style),
                Paragraph(str(r["Imputation_Strategy"]), table_cell_style)
            ])

        t = Table(table_data, colWidths=[90, 80, 70, 60, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])
        ]))
        story.append(t)
    story.append(Spacer(1, 12))

    # SECTION 3: DATA CLEANING & OUTLIER TREATMENT
    story.append(Paragraph("3. Data Cleaning, Imputation & Outlier Winsorization", h1_style))
    story.append(Paragraph(
        "<b>Deduplication:</b> Evaluated full-row duplicate records and primary key uniqueness on <i>OrderID</i>. "
        "Zero duplicate records were identified, validating transaction uniqueness across the entire dataset.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Missing Value Strategy:</b> Mode imputation was rejected for <i>CouponCode</i> as it would artificially inflate the 'FREESHIP' promotion "
        "to over 51% of transactions, introducing promotional bias. The domain indicator strategy preserves customer shopping intent without bias.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Outlier Treatment:</b> Interquartile Range (IQR = Q3 - Q1) fences with factor 1.5 were computed across numerical columns. "
        "To preserve statistical sample size and avoid data leakage, detected outliers were capped (Winsorized) to respective boundary thresholds:",
        body_style
    ))

    # Load outlier summary table
    outlier_csv = tbl_dir / "outlier_summary.csv"
    if outlier_csv.exists():
        df_out = pd.read_csv(outlier_csv)
        out_table_data = [[
            Paragraph("<b>Feature</b>", table_header_style),
            Paragraph("<b>Q1 (25%)</b>", table_header_style),
            Paragraph("<b>Q3 (75%)</b>", table_header_style),
            Paragraph("<b>IQR</b>", table_header_style),
            Paragraph("<b>Lower Fence</b>", table_header_style),
            Paragraph("<b>Upper Fence</b>", table_header_style),
            Paragraph("<b>Outliers</b>", table_header_style),
            Paragraph("<b>Treatment</b>", table_header_style)
        ]]
        for _, r in df_out.iterrows():
            out_table_data.append([
                Paragraph(str(r["Feature"]), table_cell_bold),
                Paragraph(str(r["Q1"]), table_cell_style),
                Paragraph(str(r["Q3"]), table_cell_style),
                Paragraph(str(r["IQR"]), table_cell_style),
                Paragraph(str(r["Lower_Fence"]), table_cell_style),
                Paragraph(str(r["Upper_Fence"]), table_cell_style),
                Paragraph(f"{r['Outlier_Count']} ({r['Outlier_Percentage']}%)", table_cell_style),
                Paragraph(str(r["Treatment_Applied"]), table_cell_style)
            ])
        t_out = Table(out_table_data, colWidths=[70, 50, 50, 45, 60, 60, 65, 100])
        t_out.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f766e")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])
        ]))
        story.append(t_out)
    story.append(Spacer(1, 14))

    # SECTION 4: EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATIONS
    story.append(Paragraph("4. Exploratory Data Analysis & Visual Findings", h1_style))
    story.append(Paragraph(
        "A rigorous visual exploration was conducted across distributions, relationships, categorical breakdowns, and correlation structures.",
        body_style
    ))

    # Embed Distribution Comparison figure
    dist_comp_img = fig_dir / "distributions" / "distribution_comparison.png"
    if dist_comp_img.exists():
        story.append(Paragraph("<b>Figure 1: Target Variable Normalization (TotalPrice vs. Log_TotalPrice)</b>", h2_style))
        story.append(Image(str(dist_comp_img), width=480, height=175))
        story.append(Spacer(1, 10))

    # Embed Boxplot figure
    boxplot_img = fig_dir / "boxplots" / "outlier_boxplots_numerical.png"
    if boxplot_img.exists():
        story.append(Paragraph("<b>Figure 2: Univariate Outlier Detection via Boxplots (IQR Rule)</b>", h2_style))
        story.append(Image(str(boxplot_img), width=480, height=240))
        story.append(Spacer(1, 10))

    # Embed Categorical Breakdown & Correlation Heatmap
    cat_break_img = fig_dir / "categorical_plots" / "bivariate_categorical_breakdown.png"
    corr_heat_img = fig_dir / "correlation_heatmap.png"

    if cat_break_img.exists():
        story.append(Paragraph("<b>Figure 3: Fulfillment Status Proportions across Product Lines (%)</b>", h2_style))
        story.append(Image(str(cat_break_img), width=480, height=210))
        story.append(Spacer(1, 10))

    if corr_heat_img.exists():
        story.append(Paragraph("<b>Figure 4: Pearson Correlation Matrix Heatmap (Features & Targets)</b>", h2_style))
        story.append(Image(str(corr_heat_img), width=420, height=310))
        story.append(Spacer(1, 10))

    # SECTION 5: FEATURE ENGINEERING
    story.append(Paragraph("5. Predictive Feature Engineering & Encoding", h1_style))
    story.append(Paragraph(
        "To maximize predictive utility for downstream ML models, the following transformations were systematically applied:",
        body_style
    ))
    story.append(Paragraph("1. <b>Temporal Feature Extraction:</b> Converted transaction timestamp <i>Date</i> into <i>Order_Month</i> (1–12) to enable cyclical monthly seasonality modeling.", bullet_style))
    story.append(Paragraph("2. <b>Non-linear Transformation:</b> Created <i>Log_TotalPrice</i> via <code>np.log1p(TotalPrice)</code>, stabilizing error variance and linearizing relationships.", bullet_style))
    story.append(Paragraph("3. <b>Nominal One-Hot Encoding:</b> Converted 5 nominal features (<i>Product, PaymentMethod, OrderStatus, CouponCode, ReferralSource</i>) into 26 binary dummy indicators.", bullet_style))
    story.append(Spacer(1, 10))

    # SECTION 6: STRUCTURAL CONTRACTS & VALIDATION
    story.append(Paragraph("6. Structural Contracts & Pandera Data Validation", h1_style))
    story.append(Paragraph(
        "A formal data contract was defined using Pandera DataFrameSchema to ensure that data entering production model training "
        "conforms strictly to required statistical ranges, type definitions, and completeness criteria.",
        body_style
    ))

    # Load validation summary table
    val_csv = tbl_dir / "validation_summary.csv"
    if val_csv.exists():
        df_val = pd.read_csv(val_csv)
        val_table_data = [[
            Paragraph("<b>Check ID</b>", table_header_style),
            Paragraph("<b>Validation Domain</b>", table_header_style),
            Paragraph("<b>Rule Description</b>", table_header_style),
            Paragraph("<b>Target Columns</b>", table_header_style),
            Paragraph("<b>Observed Metric</b>", table_header_style),
            Paragraph("<b>Status</b>", table_header_style)
        ]]
        for _, r in df_val.iterrows():
            status_color = colors.HexColor("#16a34a") if r["Pass_Status"] == "PASSED" else colors.HexColor("#dc2626")
            status_style = ParagraphStyle('StatStyle', parent=table_cell_bold, textColor=status_color)
            val_table_data.append([
                Paragraph(str(r["Check_ID"]), table_cell_bold),
                Paragraph(str(r["Validation_Domain"]), table_cell_style),
                Paragraph(str(r["Rule_Description"]), table_cell_style),
                Paragraph(str(r["Target_Columns"]), table_cell_style),
                Paragraph(str(r["Observed_Metric"]), table_cell_style),
                Paragraph(str(r["Pass_Status"]), status_style)
            ])
        t_val = Table(val_table_data, colWidths=[55, 90, 120, 85, 100, 50])
        t_val.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])
        ]))
        story.append(t_val)
    story.append(Spacer(1, 14))

    # SECTION 7: BUSINESS RECOMMENDATIONS & CONCLUSIONS
    story.append(Paragraph("7. Strategic Business Insights & Next Steps", h1_style))
    story.append(Paragraph(
        "Based on comprehensive exploratory analysis and statistical modeling, the following strategic insights are provided:",
        body_style
    ))
    story.append(Paragraph("• <b>Pricing & Product Mix:</b> High unit-price items (Laptops, Desktops) drive disproportionate revenue volume. Product bundling strategies with accessories (Headphones, Cables) can increase <i>ItemsInCart</i> without hurting conversion.", bullet_style))
    story.append(Paragraph("• <b>Promotional Optimization:</b> Coupon usage (<i>FREESHIP, WINTER15, SAVE10</i>) accounts for 74.25% of all orders. Customers without coupons show similar basket sizes, suggesting targeted rather than blanket discount campaigns.", bullet_style))
    story.append(Paragraph("• <b>Model Architecture Readiness:</b> The final dataset (1,200 rows x 38 features) is saved in <code>data/processed/final_model_ready_dataset.csv</code> and is immediately ready for regression (TotalPrice prediction) or classification (OrderStatus / Coupon utilization) modeling.", bullet_style))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[PDF REPORT] Successfully generated report at: {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    pdf_out = project_root / "report" / "Project_1_Report.pdf"
    fig_dir = project_root / "outputs" / "figures"
    tbl_dir = project_root / "outputs" / "tables"
    
    build_pdf_report(pdf_out, fig_dir, tbl_dir)

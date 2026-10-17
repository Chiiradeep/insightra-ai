from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)


def create_pdf_report(
    filename,
    dataset_name,
    data,
    ai_insights
):
    """
    Create a professional PDF report
    from the uploaded dataset.
    """

    # ==================================================
    # PDF DOCUMENT
    # ==================================================

    document = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=24,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=16,
        spaceBefore=15,
        spaceAfter=10
    )

    normal_style = ParagraphStyle(
        "NormalText",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15,
        spaceAfter=8
    )

    # ==================================================
    # CONTENT
    # ==================================================

    story = []

    # ==================================================
    # TITLE
    # ==================================================

    story.append(
        Paragraph(
            "DataNova AI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Automated Data Analysis Report",
            styles["Heading2"]
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            f"<b>Dataset:</b> {dataset_name}",
            normal_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Rows:</b> {data.shape[0]}",
            normal_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Columns:</b> {data.shape[1]}",
            normal_style
        )
    )

    story.append(Spacer(1, 15))

    # ==================================================
    # DATASET OVERVIEW
    # ==================================================

    story.append(
        Paragraph(
            "1. Dataset Overview",
            heading_style
        )
    )

    numeric_columns = data.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = data.select_dtypes(
        exclude="number"
    ).columns.tolist()

    overview_text = (
        f"The dataset contains "
        f"{data.shape[0]} rows and "
        f"{data.shape[1]} columns. "
        f"It contains "
        f"{len(numeric_columns)} numeric columns "
        f"and "
        f"{len(categorical_columns)} categorical columns."
    )

    story.append(
        Paragraph(
            overview_text,
            normal_style
        )
    )

    # ==================================================
    # DATA QUALITY
    # ==================================================

    story.append(
        Paragraph(
            "2. Data Quality",
            heading_style
        )
    )

    missing_values = int(
        data.isnull().sum().sum()
    )

    duplicate_rows = int(
        data.duplicated().sum()
    )

    quality_text = (
        f"Total missing values: "
        f"<b>{missing_values}</b><br/>"
        f"Duplicate rows: "
        f"<b>{duplicate_rows}</b>"
    )

    story.append(
        Paragraph(
            quality_text,
            normal_style
        )
    )

    # ==================================================
    # STATISTICAL SUMMARY
    # ==================================================

    story.append(
        Paragraph(
            "3. Statistical Summary",
            heading_style
        )
    )

    if numeric_columns:

        summary_data = [
            [
                "Column",
                "Average",
                "Minimum",
                "Maximum"
            ]
        ]

        for column in numeric_columns:

            summary_data.append(
                [
                    column,
                    f"{data[column].mean():.2f}",
                    f"{data[column].min():.2f}",
                    f"{data[column].max():.2f}"
                ]
            )

        table = Table(
            summary_data,
            repeatRows=1
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#1e293b")
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "ALIGN",
                        (1, 1),
                        (-1, -1),
                        "CENTER"
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [
                            colors.white,
                            colors.HexColor("#f1f5f9")
                        ]
                    )
                ]
            )
        )

        story.append(table)

    else:

        story.append(
            Paragraph(
                "No numeric columns are available.",
                normal_style
            )
        )

    # ==================================================
    # AI INSIGHTS
    # ==================================================

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "4. AI-Generated Insights",
            heading_style
        )
    )

    # Convert AI response into readable paragraphs

    if ai_insights:

        lines = ai_insights.split("\n")

        for line in lines:

            line = line.strip()

            if not line:
                story.append(
                    Spacer(1, 5)
                )
                continue

            # Remove markdown heading symbols
            clean_line = line.replace(
                "## ",
                ""
            )

            clean_line = clean_line.replace(
                "### ",
                ""
            )

            story.append(
                Paragraph(
                    clean_line,
                    normal_style
                )
            )

    else:

        story.append(
            Paragraph(
                "No AI insights were generated.",
                normal_style
            )
        )

    # ==================================================
    # COLUMN INFORMATION
    # ==================================================

    story.append(
        Paragraph(
            "5. Column Information",
            heading_style
        )
    )

    column_data = [
        [
            "Column",
            "Data Type",
            "Missing",
            "Unique"
        ]
    ]

    for column in data.columns:

        column_data.append(
            [
                str(column),
                str(data[column].dtype),
                str(data[column].isnull().sum()),
                str(data[column].nunique())
            ]
        )

    column_table = Table(
        column_data,
        repeatRows=1
    )

    column_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1e293b")
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor("#f1f5f9")
                    ]
                )
            ]
        )
    )

    story.append(
        column_table
    )

    # ==================================================
    # BUILD PDF
    # ==================================================

    document.build(story)

    return filename
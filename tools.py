import pandas as pd


# ============================================================
# CURRENT DATASET
# ============================================================

current_data = None


# ============================================================
# SET DATASET
# ============================================================

def set_dataset(data):
    """Store the uploaded dataset for analysis."""
    global current_data
    current_data = data


# ============================================================
# DATASET INFORMATION
# ============================================================

def get_dataset_info() -> str:
    """Return basic information about the current dataset."""

    if current_data is None:
        return "No dataset has been uploaded."

    return f"""
Dataset Information

Rows: {current_data.shape[0]}
Columns: {current_data.shape[1]}

Columns:
{list(current_data.columns)}

Data Types:
{current_data.dtypes.to_string()}
"""


# ============================================================
# NUMERIC COLUMNS
# ============================================================

def get_numeric_columns() -> str:
    """Return all numeric columns in the dataset."""

    if current_data is None:
        return "No dataset has been uploaded."

    columns = list(
        current_data.select_dtypes(
            include="number"
        ).columns
    )

    if not columns:
        return "No numeric columns found."

    return f"Numeric columns: {columns}"


# ============================================================
# CATEGORICAL COLUMNS
# ============================================================

def get_categorical_columns() -> str:
    """Return all categorical/text columns."""

    if current_data is None:
        return "No dataset has been uploaded."

    columns = list(
        current_data.select_dtypes(
            exclude="number"
        ).columns
    )

    if not columns:
        return "No categorical columns found."

    return f"Categorical columns: {columns}"


# ============================================================
# DATASET PREVIEW
# ============================================================

def get_dataset_preview() -> str:
    """Return the first five rows of the dataset."""

    if current_data is None:
        return "No dataset has been uploaded."

    return current_data.head().to_string(
        index=False
    )


# ============================================================
# COLUMN AVERAGE
# ============================================================

def get_column_average(column: str) -> float:
    """Calculate the average of a numeric column."""

    if current_data is None:
        return 0.0

    if column not in current_data.columns:
        return 0.0

    if not pd.api.types.is_numeric_dtype(
        current_data[column]
    ):
        return 0.0

    return float(
        current_data[column].mean()
    )


# ============================================================
# COLUMN SUM
# ============================================================

def get_column_sum(column: str) -> float:
    """Calculate the sum of a numeric column."""

    if current_data is None:
        return 0.0

    if column not in current_data.columns:
        return 0.0

    if not pd.api.types.is_numeric_dtype(
        current_data[column]
    ):
        return 0.0

    return float(
        current_data[column].sum()
    )


# ============================================================
# COLUMN MAXIMUM
# ============================================================

def get_column_max(column: str) -> float:
    """Find the maximum value in a numeric column."""

    if current_data is None:
        return 0.0

    if column not in current_data.columns:
        return 0.0

    if not pd.api.types.is_numeric_dtype(
        current_data[column]
    ):
        return 0.0

    return float(
        current_data[column].max()
    )


# ============================================================
# COLUMN MINIMUM
# ============================================================

def get_column_min(column: str) -> float:
    """Find the minimum value in a numeric column."""

    if current_data is None:
        return 0.0

    if column not in current_data.columns:
        return 0.0

    if not pd.api.types.is_numeric_dtype(
        current_data[column]
    ):
        return 0.0

    return float(
        current_data[column].min()
    )


# ============================================================
# TOP ROW
# ============================================================

def get_top_row(column: str) -> str:
    """Return the complete row with the highest value."""

    if current_data is None:
        return "No dataset has been uploaded."

    if column not in current_data.columns:
        return f"Column '{column}' was not found."

    if not pd.api.types.is_numeric_dtype(
        current_data[column]
    ):
        return f"Column '{column}' is not numeric."

    row = current_data.loc[
        current_data[column].idxmax()
    ]

    return row.to_string()


# ============================================================
# BOTTOM ROW
# ============================================================

def get_bottom_row(column: str) -> str:
    """Return the complete row with the lowest value."""

    if current_data is None:
        return "No dataset has been uploaded."

    if column not in current_data.columns:
        return f"Column '{column}' was not found."

    if not pd.api.types.is_numeric_dtype(
        current_data[column]
    ):
        return f"Column '{column}' is not numeric."

    row = current_data.loc[
        current_data[column].idxmin()
    ]

    return row.to_string()


# ============================================================
# UNIQUE VALUES
# ============================================================

def get_unique_values(column: str) -> str:
    """Return unique values from a column."""

    if current_data is None:
        return "No dataset has been uploaded."

    if column not in current_data.columns:
        return f"Column '{column}' was not found."

    values = (
        current_data[column]
        .dropna()
        .unique()
        .tolist()
    )

    return (
        f"Unique values in '{column}': "
        f"{values}"
    )


# ============================================================
# VALUE COUNTS
# ============================================================

def get_value_counts(column: str) -> str:
    """Count occurrences of each value in a column."""

    if current_data is None:
        return "No dataset has been uploaded."

    if column not in current_data.columns:
        return f"Column '{column}' was not found."

    counts = (
        current_data[column]
        .value_counts()
        .head(20)
    )

    return (
        f"Value counts for '{column}':\n"
        f"{counts.to_string()}"
    )


# ============================================================
# MISSING VALUES
# ============================================================

def get_missing_values() -> str:
    """Return missing-value counts for every column."""

    if current_data is None:
        return "No dataset has been uploaded."

    missing = (
        current_data
        .isnull()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    return (
        "Missing values by column:\n"
        f"{missing.to_string()}"
    )


# ============================================================
# GROUP AND AGGREGATE
# ============================================================

def group_and_aggregate(
    group_by: str,
    value_column: str,
    operation: str
) -> str:
    """
    Group a dataset by one column and perform an
    aggregation on another column.

    Supported operations:
    sum, average, mean, maximum, minimum, count
    """

    if current_data is None:
        return "No dataset has been uploaded."

    if group_by not in current_data.columns:
        return f"Column '{group_by}' was not found."

    if value_column not in current_data.columns:
        return (
            f"Column '{value_column}' "
            "was not found."
        )

    operation = operation.lower().strip()

    if operation in ["average", "mean"]:
        result = (
            current_data
            .groupby(group_by)[value_column]
            .mean()
            .sort_values(
                ascending=False
            )
        )

    elif operation == "sum":
        result = (
            current_data
            .groupby(group_by)[value_column]
            .sum()
            .sort_values(
                ascending=False
            )
        )

    elif operation in ["maximum", "max"]:
        result = (
            current_data
            .groupby(group_by)[value_column]
            .max()
            .sort_values(
                ascending=False
            )
        )

    elif operation in ["minimum", "min"]:
        result = (
            current_data
            .groupby(group_by)[value_column]
            .min()
            .sort_values(
                ascending=False
            )
        )

    elif operation == "count":

        result = (
            current_data
            .groupby(group_by)[value_column]
            .count()
            .sort_values(
                ascending=False
            )
        )

    else:

        return (
            "Unsupported operation. "
            "Use sum, average, maximum, "
            "minimum, or count."
        )

    return (
        f"Grouped analysis:\n"
        f"{result.to_string()}"
    )
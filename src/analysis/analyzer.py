import pandas as pd


# ============================================================
# NUMERIC SUMMARY
# ============================================================

def get_numeric_summary(df):
    """Return statistical summary for numeric columns."""

    return df.describe().T


# ============================================================
# CATEGORICAL SUMMARY
# ============================================================

def get_categorical_summary(df):
    """Return summary for categorical columns."""

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    if len(categorical_columns) == 0:
        return pd.DataFrame()

    summary = pd.DataFrame({
        "Column": categorical_columns,
        "Unique Values": [
            df[col].nunique()
            for col in categorical_columns
        ],
        "Missing Values": [
            df[col].isna().sum()
            for col in categorical_columns
        ]
    })

    return summary


# ============================================================
# CORRELATION
# ============================================================

def get_correlation(df):
    """Return correlation matrix for numeric columns."""

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    return numeric_df.corr()


# ============================================================
# BASIC DATASET INSIGHTS
# ============================================================

def get_basic_insights(df):
    """Generate basic automated insights about the dataset."""

    insights = []

    rows, columns = df.shape

    insights.append(
        f"The dataset contains {rows:,} rows and {columns} columns."
    )

    missing_values = int(
        df.isna().sum().sum()
    )

    if missing_values == 0:
        insights.append(
            "There are no missing values in the dataset."
        )
    else:
        insights.append(
            f"The dataset contains {missing_values:,} missing values."
        )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    if duplicate_rows == 0:
        insights.append(
            "There are no duplicate rows."
        )
    else:
        insights.append(
            f"The dataset contains {duplicate_rows:,} duplicate rows."
        )

    numeric_columns = len(
        df.select_dtypes(
            include="number"
        ).columns
    )

    categorical_columns = len(
        df.select_dtypes(
            include=["object", "category"]
        ).columns
    )

    insights.append(
        f"The dataset contains {numeric_columns} numeric columns "
        f"and {categorical_columns} categorical columns."
    )

    return insights


# ============================================================
# EXACT DATA ANALYSIS FOR AI ASSISTANT
# ============================================================

def get_column_info(df):
    """
    Return useful information about every column.

    This is used by the AI assistant to understand
    the structure of the uploaded dataset.
    """

    info = []

    for column in df.columns:

        series = df[column]

        info.append({
            "column": column,
            "data_type": str(series.dtype),
            "non_null": int(
                series.notna().sum()
            ),
            "missing": int(
                series.isna().sum()
            ),
            "unique_values": int(
                series.nunique(
                    dropna=True
                )
            )
        })

    return pd.DataFrame(info)


# ============================================================
# EXACT NUMERIC ANALYSIS
# ============================================================

def get_numeric_analysis(df):
    """
    Return exact statistics for all numeric columns.

    These calculations use the complete DataFrame,
    not a limited number of RAG-retrieved rows.
    """

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    results = []

    for column in numeric_columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if series.empty:
            continue

        results.append({
            "column": column,
            "count": int(
                series.count()
            ),
            "sum": float(
                series.sum()
            ),
            "mean": float(
                series.mean()
            ),
            "median": float(
                series.median()
            ),
            "minimum": float(
                series.min()
            ),
            "maximum": float(
                series.max()
            ),
            "standard_deviation": float(
                series.std()
            )
        })

    return pd.DataFrame(results)


# ============================================================
# COMPLETE DATASET PROFILE
# ============================================================

def get_dataset_profile(df):
    """
    Return a complete basic profile of the dataset.
    """

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    return {
        "rows": int(
            df.shape[0]
        ),

        "columns": int(
            df.shape[1]
        ),

        "column_names": df.columns.tolist(),

        "numeric_columns": numeric_columns,

        "categorical_columns": categorical_columns,

        "missing_values": int(
            df.isna().sum().sum()
        ),

        "duplicate_rows": int(
            df.duplicated().sum()
        )
    }
import pandas as pd


def generate_summary_report(df):
    """
    Generate a structured summary report for the dataset.
    """

    report = {}

    report["rows"] = int(df.shape[0])
    report["columns"] = int(df.shape[1])
    report["missing_values"] = int(df.isna().sum().sum())
    report["duplicate_rows"] = int(df.duplicated().sum())

    report["numeric_columns"] = df.select_dtypes(
        include="number"
    ).columns.tolist()

    report["categorical_columns"] = df.select_dtypes(
        include="object"
    ).columns.tolist()

    return report


def generate_column_report(df):
    """
    Generate detailed information for every column.
    """

    column_report = []

    for column in df.columns:
        column_report.append(
            {
                "Column": column,
                "Data Type": str(df[column].dtype),
                "Missing Values": int(df[column].isna().sum()),
                "Unique Values": int(df[column].nunique()),
            }
        )

    return pd.DataFrame(column_report)


def generate_statistical_report(df):
    """
    Generate descriptive statistics for numeric columns.
    """

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return pd.DataFrame()

    return numeric_df.describe().transpose()


def generate_category_report(df):
    """
    Generate frequency information for categorical columns.
    """

    categorical_columns = df.select_dtypes(
        include="object"
    ).columns

    results = []

    for column in categorical_columns:
        value_counts = df[column].value_counts().head(10)

        for value, count in value_counts.items():
            results.append(
                {
                    "Column": column,
                    "Value": value,
                    "Count": int(count),
                }
            )

    return pd.DataFrame(results)
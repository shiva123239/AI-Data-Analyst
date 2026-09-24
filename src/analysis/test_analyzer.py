import pandas as pd

from src.analysis.analyzer import (
    get_numeric_summary,
    get_categorical_summary,
    get_correlation,
    get_basic_insights,
    get_column_info,
    get_numeric_analysis,
    get_dataset_profile,
)


def main():

    print("\n" + "=" * 70)
    print("AI DATA ANALYST - ANALYZER TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    df = pd.read_csv("data/sales_data.csv")

    print("\nDataset loaded successfully.")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumns:")
    print(df.columns.tolist())

    # --------------------------------------------------------
    # NUMERIC SUMMARY
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("1. NUMERIC SUMMARY")
    print("-" * 70)

    numeric_summary = get_numeric_summary(df)

    print(numeric_summary)

    # --------------------------------------------------------
    # CATEGORICAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("2. CATEGORICAL SUMMARY")
    print("-" * 70)

    categorical_summary = get_categorical_summary(df)

    print(categorical_summary.to_string(index=False))

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("3. CORRELATION")
    print("-" * 70)

    correlation = get_correlation(df)

    print(correlation)

    # --------------------------------------------------------
    # BASIC INSIGHTS
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("4. BASIC DATASET INSIGHTS")
    print("-" * 70)

    insights = get_basic_insights(df)

    for insight in insights:
        print("-", insight)

    # --------------------------------------------------------
    # COLUMN INFORMATION
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("5. COLUMN INFORMATION")
    print("-" * 70)

    column_info = get_column_info(df)

    print(column_info.to_string(index=False))

    # --------------------------------------------------------
    # NUMERIC ANALYSIS
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("6. EXACT NUMERIC ANALYSIS")
    print("-" * 70)

    numeric_analysis = get_numeric_analysis(df)

    print(numeric_analysis.to_string(index=False))

    # --------------------------------------------------------
    # COMPLETE DATASET PROFILE
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("7. COMPLETE DATASET PROFILE")
    print("-" * 70)

    profile = get_dataset_profile(df)

    for key, value in profile.items():
        print(f"{key}: {value}")

    # --------------------------------------------------------
    # TEST COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANALYZER TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
import pandas as pd

from src.feature_engineering.feature_engineer import (
    engineer_features,
    get_feature_summary,
)


def create_test_dataframe():
    """Create a small sample dataset for testing."""

    return pd.DataFrame({
        "customer_id": [
            "C001",
            "C002",
            "C003",
            "C004",
        ],

        "date_of_birth": [
            "10-05-1990",
            "15-08-1985",
            "20-01-2000",
            "05-12-1975",
        ],

        "join_date": [
            "10-01-2022",
            "15-06-2023",
            "20-03-2024",
            "05-09-2021",
        ],

        "annual_income": [
            500000,
            750000,
            450000,
            5000000,
        ],

        "customer_segment": [
            "Premium",
            "Retail",
            "SME",
            "Premium",
        ],
    })


# =========================================================
# TEST 1
# =========================================================

def test_engineer_features_returns_dataframe():
    """Check that feature engineering returns a DataFrame."""

    df = create_test_dataframe()

    engineered_df, created_features = engineer_features(df)

    assert isinstance(engineered_df, pd.DataFrame)
    assert isinstance(created_features, list)


# =========================================================
# TEST 2
# =========================================================

def test_original_columns_are_preserved():
    """Check that all original columns are still present."""

    df = create_test_dataframe()

    engineered_df, created_features = engineer_features(df)

    for column in df.columns:
        assert column in engineered_df.columns


# =========================================================
# TEST 3
# =========================================================

def test_feature_engineering_creates_new_features():
    """Check that new features are created."""

    df = create_test_dataframe()

    engineered_df, created_features = engineer_features(df)

    assert engineered_df.shape[1] > df.shape[1]

    assert len(created_features) > 0


# =========================================================
# TEST 4
# =========================================================

def test_date_features_are_created():
    """Check that date-based features are created."""

    df = create_test_dataframe()

    engineered_df, created_features = engineer_features(df)

    possible_features = [
        "join_date_year",
        "join_date_month",
        "join_date_month_name",
        "join_date_quarter",
        "join_date_day_of_week",
        "join_date_day_name",
        "join_date_is_weekend",
        "date_of_birth_year",
        "date_of_birth_month",
        "date_of_birth_quarter",
        "age",
        "age_group",
    ]

    created = [
        feature
        for feature in possible_features
        if feature in engineered_df.columns
    ]

    assert len(created) > 0


# =========================================================
# TEST 5
# =========================================================

def test_income_features_are_created():
    """Check that income-related features are created."""

    df = create_test_dataframe()

    engineered_df, created_features = engineer_features(df)

    possible_features = [
        "annual_income_log",
        "annual_income_outlier",
    ]

    created = [
        feature
        for feature in possible_features
        if feature in engineered_df.columns
    ]

    assert len(created) > 0


# =========================================================
# TEST 6
# =========================================================

def test_feature_summary_returns_dictionary():
    """Check that the feature summary returns a dictionary."""

    df = create_test_dataframe()

    engineered_df, created_features = engineer_features(df)

    summary = get_feature_summary(
        df,
        engineered_df,
        created_features,
    )

    assert isinstance(summary, dict)


# =========================================================
# TEST 7
# =========================================================

def test_feature_summary_contains_required_information():
    """Check important summary information."""

    df = create_test_dataframe()

    engineered_df, created_features = engineer_features(df)

    summary = get_feature_summary(
        df,
        engineered_df,
        created_features,
    )

    assert "original_rows" in summary
    assert "original_columns" in summary
    assert "new_columns" in summary
    assert "total_columns" in summary
    assert "created_features" in summary


# =========================================================
# TEST 8
# =========================================================

def test_row_count_is_preserved():
    """Feature engineering should not remove rows."""

    df = create_test_dataframe()

    engineered_df, created_features = engineer_features(df)

    assert engineered_df.shape[0] == df.shape[0]


# =========================================================
# TEST 9
# =========================================================

def test_created_features_are_list():
    """Check that created_features is returned as a list."""

    df = create_test_dataframe()

    engineered_df, created_features = engineer_features(df)

    assert isinstance(created_features, list)


# =========================================================
# TEST 10
# =========================================================

def test_feature_summary_values_are_correct():
    """Check that summary values are correct."""

    df = create_test_dataframe()

    engineered_df, created_features = engineer_features(df)

    summary = get_feature_summary(
        df,
        engineered_df,
        created_features,
    )

    assert summary["original_rows"] == df.shape[0]

    assert summary["original_columns"] == df.shape[1]

    assert summary["total_columns"] == engineered_df.shape[1]

    assert summary["new_columns"] == (
        engineered_df.shape[1] - df.shape[1]
    )

    assert summary["created_features"] == created_features
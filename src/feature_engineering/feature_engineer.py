import pandas as pd
import numpy as np


# =========================================================
# FEATURE ENGINEERING ENGINE
# =========================================================

def engineer_features(df):
    """
    Automatically create useful features from the dataset.

    The function analyzes available columns and creates
    meaningful derived features without assuming a fixed
    dataset structure.

    Returns
    -------
    engineered_df : pandas.DataFrame
        Dataset containing original and engineered features.

    created_features : list
        Names of features created by this function.
    """

    # Work on a copy so the original DataFrame is not modified
    df = df.copy()

    created_features = []


    # =====================================================
    # 1. DATETIME FEATURE ENGINEERING
    # =====================================================

    # Store original columns first.
    # This prevents newly-created date columns from being
    # processed again.
    original_columns = df.columns.tolist()

    for column in original_columns:

        # Only try date conversion for object/string columns
        if (
            df[column].dtype == "object"
            or pd.api.types.is_string_dtype(df[column])
        ):

            # First try day-first parsing because datasets often
            # contain dates such as 10-05-1990.
            converted = pd.to_datetime(
                df[column],
                errors="coerce",
                dayfirst=True
            )

            valid_ratio = converted.notna().mean()

            # Only convert the column when most values are
            # successfully recognized as dates.
            if valid_ratio >= 0.70:

                df[column] = converted

                # -------------------------------------------------
                # Year
                # -------------------------------------------------

                year_column = f"{column}_year"

                if year_column not in df.columns:

                    df[year_column] = (
                        df[column].dt.year
                    )

                    created_features.append(
                        year_column
                    )


                # -------------------------------------------------
                # Month
                # -------------------------------------------------

                month_column = f"{column}_month"

                if month_column not in df.columns:

                    df[month_column] = (
                        df[column].dt.month
                    )

                    created_features.append(
                        month_column
                    )


                # -------------------------------------------------
                # Month Name
                # -------------------------------------------------

                month_name_column = (
                    f"{column}_month_name"
                )

                if month_name_column not in df.columns:

                    df[month_name_column] = (
                        df[column]
                        .dt
                        .month_name()
                    )

                    created_features.append(
                        month_name_column
                    )


                # -------------------------------------------------
                # Quarter
                # -------------------------------------------------

                quarter_column = (
                    f"{column}_quarter"
                )

                if quarter_column not in df.columns:

                    df[quarter_column] = (
                        df[column]
                        .dt
                        .quarter
                    )

                    created_features.append(
                        quarter_column
                    )


                # -------------------------------------------------
                # Day of Week
                # -------------------------------------------------

                weekday_column = (
                    f"{column}_day_of_week"
                )

                if weekday_column not in df.columns:

                    df[weekday_column] = (
                        df[column]
                        .dt
                        .dayofweek
                    )

                    created_features.append(
                        weekday_column
                    )


                # -------------------------------------------------
                # Day Name
                # -------------------------------------------------

                day_name_column = (
                    f"{column}_day_name"
                )

                if day_name_column not in df.columns:

                    df[day_name_column] = (
                        df[column]
                        .dt
                        .day_name()
                    )

                    created_features.append(
                        day_name_column
                    )


                # -------------------------------------------------
                # Weekend Indicator
                # -------------------------------------------------

                weekend_column = (
                    f"{column}_is_weekend"
                )

                if weekend_column not in df.columns:

                    df[weekend_column] = (
                        df[column]
                        .dt
                        .dayofweek
                        .isin([5, 6])
                        .astype(int)
                    )

                    created_features.append(
                        weekend_column
                    )


    # =====================================================
    # 2. AGE FEATURE ENGINEERING
    # =====================================================

    date_columns = [
        column
        for column in df.columns
        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        )
    ]

    birth_columns = [
        column
        for column in date_columns
        if any(
            keyword in column.lower()
            for keyword in [
                "birth",
                "dob",
                "date_of_birth"
            ]
        )
    ]

    for column in birth_columns:

        # -------------------------------------------------
        # Age
        # -------------------------------------------------

        age_column = "age"

        today = pd.Timestamp.today()

        df[age_column] = (
            (
                today
                - df[column]
            ).dt.days
            / 365.25
        ).round(0)

        df[age_column] = (
            df[age_column]
            .clip(
                lower=0,
                upper=120
            )
        )

        if age_column not in created_features:

            created_features.append(
                age_column
            )


        # -------------------------------------------------
        # Age Group
        # -------------------------------------------------

        age_group_column = "age_group"

        df[age_group_column] = pd.cut(
            df[age_column],
            bins=[
                0,
                18,
                25,
                35,
                45,
                55,
                65,
                120
            ],
            labels=[
                "Under 18",
                "18-24",
                "25-34",
                "35-44",
                "45-54",
                "55-64",
                "65+"
            ],
            include_lowest=True
        )

        if age_group_column not in created_features:

            created_features.append(
                age_group_column
            )


    # =====================================================
    # 3. TENURE FEATURE ENGINEERING
    # =====================================================

    join_columns = [
        column
        for column in date_columns
        if any(
            keyword in column.lower()
            for keyword in [
                "join",
                "joined",
                "registration",
                "signup",
                "sign_up",
                "start_date"
            ]
        )
    ]

    for column in join_columns:

        # -------------------------------------------------
        # Tenure in Days
        # -------------------------------------------------

        tenure_column = (
            f"{column}_tenure_days"
        )

        today = pd.Timestamp.today()

        df[tenure_column] = (
            today
            - df[column]
        ).dt.days

        df[tenure_column] = (
            df[tenure_column]
            .clip(lower=0)
        )

        created_features.append(
            tenure_column
        )


        # -------------------------------------------------
        # Tenure in Years
        # -------------------------------------------------

        tenure_years_column = (
            f"{column}_tenure_years"
        )

        df[tenure_years_column] = (
            df[tenure_column]
            / 365.25
        ).round(2)

        created_features.append(
            tenure_years_column
        )


    # =====================================================
    # 4. NUMERIC FEATURE ENGINEERING
    # =====================================================

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )


    # =====================================================
    # 4A. QUANTITY × PRICE = REVENUE
    # =====================================================

    quantity_columns = [
        column
        for column in numeric_columns
        if any(
            keyword in column.lower()
            for keyword in [
                "quantity",
                "qty",
                "units",
                "unit"
            ]
        )
    ]

    price_columns = [
        column
        for column in numeric_columns
        if any(
            keyword in column.lower()
            for keyword in [
                "price",
                "unit_price",
                "selling_price",
                "sale_price"
            ]
        )
    ]

    if quantity_columns and price_columns:

        quantity_column = (
            quantity_columns[0]
        )

        price_column = (
            price_columns[0]
        )

        # Check case-insensitively whether revenue already exists
        revenue_exists = any(
            column.lower() == "revenue"
            for column in df.columns
        )

        if not revenue_exists:

            df["revenue"] = (
                df[quantity_column]
                * df[price_column]
            )

            created_features.append(
                "revenue"
            )


    # =====================================================
    # 4B. REVENUE / QUANTITY = AVERAGE PRICE
    # =====================================================

    revenue_columns = [
        column
        for column in df.columns
        if any(
            keyword in column.lower()
            for keyword in [
                "revenue",
                "sales",
                "amount",
                "total"
            ]
        )
        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    if quantity_columns and revenue_columns:

        revenue_column = (
            revenue_columns[0]
        )

        quantity_column = (
            quantity_columns[0]
        )

        if (
            revenue_column != quantity_column
            and "average_price" not in df.columns
        ):

            df["average_price"] = (
                df[revenue_column]
                /
                df[quantity_column]
                .replace(
                    0,
                    np.nan
                )
            )

            df["average_price"] = (
                df["average_price"]
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
            )

            created_features.append(
                "average_price"
            )


    # =====================================================
    # 5. LOG TRANSFORMATIONS
    # =====================================================

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    for column in numeric_columns:

        # Avoid log transformations on IDs and
        # identifier-like fields.
        if any(
            keyword in column.lower()
            for keyword in [
                "id",
                "zip",
                "postal",
                "phone"
            ]
        ):

            continue


        # Get minimum safely
        try:

            minimum_value = (
                df[column]
                .min(
                    skipna=True
                )
            )

        except Exception:

            continue


        # Only create log features for positive data
        if (
            pd.notna(minimum_value)
            and minimum_value > 0
        ):

            log_column = (
                f"{column}_log"
            )

            if log_column not in df.columns:

                df[log_column] = (
                    np.log1p(
                        df[column]
                    )
                )

                created_features.append(
                    log_column
                )


    # =====================================================
    # 6. OUTLIER FLAGS
    # =====================================================

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    for column in numeric_columns:

        # Don't calculate outliers for identifiers,
        # dates represented by numeric parts, etc.
        if any(
            keyword in column.lower()
            for keyword in [
                "id",
                "zip",
                "postal",
                "phone",
                "year",
                "month",
                "day"
            ]
        ):

            continue


        try:

            q1 = (
                df[column]
                .quantile(0.25)
            )

            q3 = (
                df[column]
                .quantile(0.75)
            )

            iqr = q3 - q1

            if (
                pd.isna(iqr)
                or iqr == 0
            ):

                continue


            lower_bound = (
                q1 - 1.5 * iqr
            )

            upper_bound = (
                q3 + 1.5 * iqr
            )


            outlier_column = (
                f"{column}_outlier"
            )


            if outlier_column not in df.columns:

                df[outlier_column] = (
                    (
                        (df[column] < lower_bound)
                        |
                        (df[column] > upper_bound)
                    )
                    .astype(int)
                )

                created_features.append(
                    outlier_column
                )


        except Exception:

            continue


    # =====================================================
    # 7. MISSING VALUE FLAGS
    # =====================================================

    # Take a snapshot so that newly-created missing flags
    # aren't processed again.
    columns_for_missing_flags = (
        df.columns.tolist()
    )

    for column in columns_for_missing_flags:

        # Don't create flags for existing missing flags.
        if column.endswith(
            "_missing"
        ):

            continue


        missing_count = int(
            df[column]
            .isna()
            .sum()
        )


        if missing_count > 0:

            missing_column = (
                f"{column}_missing"
            )


            if missing_column not in df.columns:

                df[missing_column] = (
                    df[column]
                    .isna()
                    .astype(int)
                )

                created_features.append(
                    missing_column
                )


    # =====================================================
    # 8. REMOVE DUPLICATE FEATURE NAMES
    # =====================================================

    created_features = list(
        dict.fromkeys(
            created_features
        )
    )


    # =====================================================
    # RESULT
    # =====================================================

    return df, created_features


# =========================================================
# FEATURE SUMMARY
# =========================================================

def get_feature_summary(
    original_df,
    engineered_df,
    created_features
):
    """
    Return a summary of the feature engineering process.
    """

    return {
        "original_rows": int(
            original_df.shape[0]
        ),

        "original_columns": int(
            original_df.shape[1]
        ),

        "new_columns": int(
            engineered_df.shape[1]
            - original_df.shape[1]
        ),

        "total_columns": int(
            engineered_df.shape[1]
        ),

        "created_features": created_features
    }
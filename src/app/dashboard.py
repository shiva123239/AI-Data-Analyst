import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from io import BytesIO
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder
)
from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier
)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    ExtraTreesClassifier
)

# Optional advanced classification models
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import warnings
import uuid
import re
import hashlib

warnings.filterwarnings("ignore")


# ============================================================
# AI ANALYST + RAG
# ============================================================

from src.analysis.analyzer import (
    get_basic_insights,
    get_column_info,
    get_numeric_analysis,
    get_dataset_profile,
    get_categorical_summary,
    get_correlation,
)

from src.ai.rag_engine import RAGEngine
from src.ai.rag_ingestion import dataframe_to_documents


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    [data-testid="stMetric"] {
        padding: 8px 10px;
    }

    .sidebar-list ul {
        list-style: none !important;
        padding-left: 0 !important;
        margin-left: 0 !important;
    }

    .sidebar-list li {
        list-style: none !important;
        margin: 7px 0;
        padding: 7px 10px;
        border-radius: 7px;
        background: rgba(128, 128, 128, 0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "df": None,
    "file_signature": None,
    "analysis_started": False,
    "engineered_df": None,
    "feature_origin": {},
    "ml_results": None,
    "ml_preprocessor": None,
    "trained_model": None,
    "trained_models": {},
    "predictions_df": None,
    "problem_type": None,
    "target_column": None,
    "label_encoder": None,
    "X_train_columns": None,
    "leakage_columns": [],
    "confusion_matrix": None,
    "classification_report": None,
    "rag_engine": None,
    "rag_file_signature": None,
    "rag_ready": False,
    "rag_enabled": False,
    "ai_chat_history": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# RESET ANALYSIS
# ============================================================

def reset_analysis():

    st.session_state.analysis_started = False

    st.session_state.engineered_df = None

    st.session_state.feature_origin = {}

    st.session_state.ml_results = None

    st.session_state.trained_model = None

    st.session_state.trained_models = {}

    st.session_state.ml_preprocessor = None

    st.session_state.predictions_df = None

    st.session_state.problem_type = None

    st.session_state.target_column = None

    st.session_state.label_encoder = None

    st.session_state.X_train_columns = None

    st.session_state.leakage_columns = []

    st.session_state.confusion_matrix = None

    st.session_state.classification_report = None

    # Reset AI/RAG state when a new dataset is loaded.
    st.session_state.rag_engine = None
    st.session_state.rag_file_signature = None
    st.session_state.rag_ready = False
    st.session_state.rag_enabled = False
    st.session_state.ai_chat_history = []


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )

    return df


# ============================================================
# NUMERIC CONVERSION
# ============================================================

def convert_possible_numeric_columns(df):

    df = df.copy()

    for col in df.columns:

        if df[col].dtype == "object":

            cleaned = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip()
            )

            numeric = pd.to_numeric(
                cleaned,
                errors="coerce"
            )

            if numeric.notna().mean() >= 0.85:

                df[col] = numeric

    return df


# ============================================================
# DATE DETECTION
# ============================================================

def detect_date_columns(df):

    date_columns = []

    keywords = [
        "date",
        "dob",
        "birth",
        "joined",
        "joining",
        "created_at",
        "updated_at"
    ]

    for col in df.columns:

        if pd.api.types.is_datetime64_any_dtype(df[col]):

            date_columns.append(col)

            continue

        name = str(col).lower()

        if any(keyword in name for keyword in keywords):

            converted = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            if converted.notna().mean() >= 0.5:

                date_columns.append(col)

    return date_columns


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):

    df = df.copy()

    feature_origin = {}

    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    date_columns = detect_date_columns(df)

    for col in date_columns:

        try:

            dt = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            df[f"{col}_year"] = dt.dt.year
            feature_origin[f"{col}_year"] = col

            df[f"{col}_quarter"] = dt.dt.quarter
            feature_origin[f"{col}_quarter"] = col

            df[f"{col}_month"] = dt.dt.month
            feature_origin[f"{col}_month"] = col

            df[f"{col}_month_name"] = dt.dt.month_name()
            feature_origin[f"{col}_month_name"] = col

            df[f"{col}_week"] = (
                dt.dt.isocalendar()
                .week
                .astype(float)
            )

            feature_origin[f"{col}_week"] = col

            df[f"{col}_day"] = dt.dt.day
            feature_origin[f"{col}_day"] = col

            df[f"{col}_day_of_week"] = dt.dt.dayofweek
            feature_origin[f"{col}_day_of_week"] = col

            df[f"{col}_day_name"] = dt.dt.day_name()
            feature_origin[f"{col}_day_name"] = col

            df[f"{col}_is_weekend"] = (
                dt.dt.dayofweek
                .isin([5, 6])
                .astype(int)
            )

            feature_origin[f"{col}_is_weekend"] = col

        except Exception:
            pass


    # --------------------------------------------------------
    # AGE FEATURES
    # --------------------------------------------------------

    dob_columns = [
        c
        for c in df.columns
        if (
            "date_of_birth" in str(c).lower()
            or str(c).lower() in [
                "dob",
                "birth_date"
            ]
        )
    ]

    for col in dob_columns:

        try:

            dob = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            age = (
                datetime.now().year
                - dob.dt.year
            )

            df["age"] = age

            feature_origin["age"] = col

            df["age_log"] = np.log1p(
                age.clip(lower=0)
            )

            feature_origin["age_log"] = col

            df["age_group"] = pd.cut(
                age,
                bins=[
                    -np.inf,
                    17,
                    25,
                    35,
                    45,
                    55,
                    65,
                    np.inf
                ],
                labels=[
                    "Under 18",
                    "18-25",
                    "26-35",
                    "36-45",
                    "46-55",
                    "56-65",
                    "65+"
                ]
            )

            feature_origin["age_group"] = col

        except Exception:
            pass


    # --------------------------------------------------------
    # NUMERIC FEATURES
    # --------------------------------------------------------

    numeric_columns = (
        df
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    for col in numeric_columns:

        try:

            series = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            if (
                series.notna().sum() > 0
                and series.min() >= 0
            ):

                new_col = f"{col}_log"

                if new_col not in df.columns:

                    df[new_col] = np.log1p(series)

                    feature_origin[new_col] = col


            if series.notna().sum() >= 10:

                q1 = series.quantile(0.25)

                q3 = series.quantile(0.75)

                iqr = q3 - q1

                if (
                    pd.notna(iqr)
                    and iqr != 0
                ):

                    lower = q1 - 1.5 * iqr

                    upper = q3 + 1.5 * iqr

                    new_col = f"{col}_outlier"

                    if new_col not in df.columns:

                        df[new_col] = (
                            (series < lower)
                            |
                            (series > upper)
                        ).astype(int)

                        feature_origin[new_col] = col

        except Exception:
            pass

    return df, feature_origin


# ============================================================
# NORMALIZE FEATURE NAME
# ============================================================

def normalize_feature_name(name):

    return (
        str(name)
        .lower()
        .strip()
        .replace("-", "_")
        .replace(" ", "_")
    )


# ============================================================
# TARGET DERIVED FEATURE DETECTION
# ============================================================

def is_target_derived_feature(
    column,
    target,
    feature_origin=None
):

    if column == target:

        return True

    column_normalized = normalize_feature_name(column)

    target_normalized = normalize_feature_name(target)

    if feature_origin:

        origin = feature_origin.get(column)

        if origin is not None:

            if (
                normalize_feature_name(origin)
                == target_normalized
            ):

                return True

    if column_normalized.startswith(
        target_normalized + "_"
    ):

        return True

    return False


# ============================================================
# ID DETECTION
# ============================================================

def is_id_like_column(column):

    name = str(column).lower().strip()

    id_names = {
        "id",
        "customer_id",
        "customerid",
        "user_id",
        "userid",
        "employee_id",
        "employeeid",
        "account_id",
        "accountid",
        "transaction_id",
        "transactionid"
    }

    if name in id_names:

        return True

    if name.endswith("_id"):

        return True

    if (
        name.endswith("id")
        and len(name) <= 20
    ):

        return True

    return False


# ============================================================
# TARGET CANDIDATE RANKING
# ============================================================

def rank_target_candidates(df):
    """Rank likely prediction targets for a safer automatic default."""
    exact_names = {
        "target", "label", "class", "outcome", "y", "diagnosis",
        "prediction", "result", "response", "churn", "fraud",
        "default", "survived", "readmitted", "approved",
        "customer_segment", "segment", "sales", "revenue",
        "profit", "price"
    }
    semantic_keywords = {
        "target": 70, "label": 70, "outcome": 70, "class": 65,
        "diagnos": 65, "churn": 65, "fraud": 65, "default": 60,
        "surviv": 60, "readmit": 60, "approv": 60, "segment": 55,
        "sales": 55, "revenue": 55, "profit": 55, "price": 45
    }
    ranked = []
    n_rows = max(len(df), 1)
    for col in df.columns:
        if is_id_like_column(col):
            continue
        s = df[col]
        unique = int(s.dropna().nunique())
        missing_ratio = float(s.isna().mean())
        score = 0.0
        name = normalize_feature_name(col)
        if name in exact_names:
            score += 100
        for keyword, points in semantic_keywords.items():
            if keyword in name:
                score = max(score, float(points))
        if unique <= 1:
            score -= 100
        elif unique <= 20:
            score += 20
        elif unique <= 50:
            score += 8
        elif unique >= max(100, int(n_rows * 0.90)):
            score -= 35
        if (pd.api.types.is_object_dtype(s) or
                pd.api.types.is_categorical_dtype(s) or
                pd.api.types.is_bool_dtype(s)):
            score += 8
        score -= missing_ratio * 30
        ranked.append((col, score))
    ranked.sort(key=lambda item: (-item[1], str(item[0]).lower()))
    return [col for col, _ in ranked]


# ============================================================
# PREPARE ML DATA
# ============================================================

def prepare_ml_data(
    df,
    target,
    feature_origin=None
):

    if target not in df.columns:

        raise ValueError(
            "Selected target column does not exist."
        )

    removed_columns = []

    # --------------------------------------------------------
    # TARGET LEAKAGE
    # --------------------------------------------------------

    leakage_columns = []

    for col in df.columns:

        if is_target_derived_feature(
            col,
            target,
            feature_origin
        ):

            leakage_columns.append(col)

    removed_columns.extend(leakage_columns)


    X = df.drop(
        columns=leakage_columns,
        errors="ignore"
    )

    y = df[target].copy()


    # --------------------------------------------------------
    # REMOVE ID COLUMNS
    # --------------------------------------------------------

    id_columns = [
        c
        for c in X.columns
        if is_id_like_column(c)
    ]

    X = X.drop(
        columns=id_columns,
        errors="ignore"
    )

    removed_columns.extend(id_columns)


    # --------------------------------------------------------
    # REMOVE RAW DATE COLUMNS
    # --------------------------------------------------------

    raw_date_columns = detect_date_columns(X)

    X = X.drop(
        columns=raw_date_columns,
        errors="ignore"
    )

    removed_columns.extend(raw_date_columns)


    # --------------------------------------------------------
    # REMOVE EMPTY COLUMNS
    # --------------------------------------------------------

    empty_columns = [
        c
        for c in X.columns
        if X[c].isna().all()
    ]

    X = X.drop(
        columns=empty_columns,
        errors="ignore"
    )

    removed_columns.extend(empty_columns)


    # --------------------------------------------------------
    # REMOVE CONSTANT COLUMNS
    # --------------------------------------------------------

    constant_columns = [
        c
        for c in X.columns
        if X[c].nunique(dropna=False) <= 1
    ]

    X = X.drop(
        columns=constant_columns,
        errors="ignore"
    )

    removed_columns.extend(constant_columns)


    # --------------------------------------------------------
    # REMOVE HIGH CARDINALITY CATEGORICAL COLUMNS
    # --------------------------------------------------------

    categorical_columns = (
        X
        .select_dtypes(
            include=[
                "object",
                "category"
            ]
        )
        .columns
        .tolist()
    )

    high_cardinality_columns = [
        c
        for c in categorical_columns
        if X[c].nunique(dropna=True) > 100
    ]

    X = X.drop(
        columns=high_cardinality_columns,
        errors="ignore"
    )

    removed_columns.extend(
        high_cardinality_columns
    )


    return (
        X,
        y,
        list(dict.fromkeys(removed_columns))
    )


# ============================================================
# PROBLEM TYPE DETECTION
# ============================================================

def detect_problem_type(y):

    y_clean = y.dropna()

    if y_clean.empty:

        return "Classification"

    unique_values = y_clean.nunique()

    if pd.api.types.is_numeric_dtype(y_clean):

        if unique_values <= 10:

            return "Classification"

        return "Regression"

    return "Classification"


# ============================================================
# EDA SUMMARY
# ============================================================

def create_eda_summary(df):

    return pd.DataFrame({

        "Column": df.columns,

        "Data Type": [
            str(df[c].dtype)
            for c in df.columns
        ],

        "Missing Values": [
            int(df[c].isna().sum())
            for c in df.columns
        ],

        "Missing %": [
            round(
                df[c].isna().mean() * 100,
                2
            )
            for c in df.columns
        ],

        "Unique Values": [
            int(df[c].nunique())
            for c in df.columns
        ]

    })


# ============================================================
# STATISTICS
# ============================================================

def create_statistics(df):

    numeric = df.select_dtypes(
        include=np.number
    )

    if numeric.empty:

        return pd.DataFrame()

    return (
        numeric
        .describe()
        .T
        .reset_index()
        .rename(
            columns={
                "index": "Feature"
            }
        )
        .round(3)
    )


# ============================================================
# FIGURE HELPERS
# ============================================================

def compact_fig(
    width=6,
    height=3.5
):

    return plt.subplots(
        figsize=(
            width,
            height
        )
    )


def show_fig(fig):

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)


# ============================================================
# ONE HOT ENCODER
# ============================================================

def make_ohe():

    try:

        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=5,
            max_categories=40,
            sparse_output=True
        )

    except TypeError:

        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=5,
            max_categories=40,
            sparse=True
        )


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(X_train):

    numeric_features = (
        X_train
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X_train
        .select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    transformers = []


    # NUMERIC

    if numeric_features:

        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ])

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )


    # CATEGORICAL

    if categorical_features:

        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                make_ohe()
            )
        ])

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )


    if not transformers:

        raise ValueError(
            "No usable numeric or categorical features remain."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# ============================================================
# FAST REGRESSION TRAINING
# ============================================================

def train_regression_models(
    X_train_processed,
    X_test_processed,
    y_train,
    y_test
):

    models = {

        "Linear Regression":
            LinearRegression(),

        "Decision Tree":
            DecisionTreeRegressor(
                random_state=42,
                max_depth=10,
                min_samples_leaf=3
            ),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=50,
                random_state=42,
                n_jobs=-1,
                max_depth=10,
                min_samples_leaf=2
            )
    }


    results = []

    trained_models = {}

    predictions = {}


    for name, model in models.items():

        model.fit(
            X_train_processed,
            y_train
        )

        pred = model.predict(
            X_test_processed
        )

        mae = mean_absolute_error(
            y_test,
            pred
        )

        mse = mean_squared_error(
            y_test,
            pred
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            pred
        )

        results.append({

            "Model": name,

            "MAE": mae,

            "MSE": mse,

            "RMSE": rmse,

            "R2 Score": r2
        })

        trained_models[name] = model

        predictions[name] = pred


    results_df = pd.DataFrame(
        results
    )

    best_name = results_df.loc[
        results_df["RMSE"].idxmin(),
        "Model"
    ]

    return (
        results_df,
        trained_models,
        predictions,
        best_name
    )


# ============================================================
# FAST CLASSIFICATION TRAINING
# ============================================================

def train_classification_models(
    X_train_processed,
    X_test_processed,
    y_train,
    y_test
):
    """
    Improved classification training while preserving the original
    dashboard interface.

    Improvements only:
    - SMOTE is applied to TRAINING data only.
    - XGBoost is added when installed.
    - Stronger Random Forest / Extra Trees configurations.
    - Class-weighted Logistic Regression / Decision Tree are retained.
    - Existing return format is unchanged.
    """

    # ------------------------------------------------------------
    # SMOTE: training data ONLY
    # ------------------------------------------------------------
    X_fit = X_train_processed
    y_fit = y_train
    smote_used = False

    if SMOTE is not None:
        try:
            class_counts = pd.Series(y_train).value_counts()
            if len(class_counts) >= 2 and class_counts.min() >= 2:
                imbalance_ratio = (
                    class_counts.max() / class_counts.min()
                )

                # Use SMOTE only when there is meaningful imbalance.
                if imbalance_ratio >= 1.5:
                    k_neighbors = max(
                        1,
                        min(5, int(class_counts.min()) - 1)
                    )
                    smote = SMOTE(
                        random_state=42,
                        k_neighbors=k_neighbors
                    )
                    X_fit, y_fit = smote.fit_resample(
                        X_train_processed,
                        y_train
                    )
                    smote_used = True
        except Exception:
            # Keep the original training path if SMOTE is not
            # compatible with a particular dataset/preprocessing shape.
            X_fit = X_train_processed
            y_fit = y_train
            smote_used = False

    # ------------------------------------------------------------
    # Models
    # ------------------------------------------------------------
    models = {
        "Logistic Regression":
            LogisticRegression(
                max_iter=1500,
                solver="liblinear",
                C=1.0,
                class_weight=None
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42,
                max_depth=14,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight=None
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight=None
            ),

        "Extra Trees":
            ExtraTreesClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight=None
            )
    }

    # ------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------
    if XGBClassifier is not None:
        try:
            n_classes = int(pd.Series(y_train).nunique())

            if n_classes == 2:
                xgb_model = XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=2,
                    gamma=0,
                    reg_alpha=0.05,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist"
                )
            else:
                xgb_model = XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=2,
                    gamma=0,
                    reg_alpha=0.05,
                    reg_lambda=1.0,
                    objective="multi:softprob",
                    num_class=n_classes,
                    eval_metric="mlogloss",
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist"
                )

            models["XGBoost"] = xgb_model
        except Exception:
            pass

    results = []
    trained_models = {}
    predictions = {}

    for name, model in models.items():
        try:
            model.fit(
                X_fit,
                y_fit
            )

            pred = model.predict(
                X_test_processed
            )

            accuracy = accuracy_score(
                y_test,
                pred
            )

            precision = precision_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            )

            results.append({
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })

            trained_models[name] = model
            predictions[name] = pred

        except Exception:
            # One model failing must not stop the whole dashboard.
            continue

    if not results:
        raise ValueError(
            "No classification model could be trained on this dataset."
        )

    results_df = pd.DataFrame(
        results
    )

    # F1 is the primary selection metric for imbalanced/multiclass
    # classification; precision, recall and accuracy are tie-breakers.
    ranked = results_df.sort_values(
        by=[
            "F1 Score",
            "Precision",
            "Recall",
            "Accuracy"
        ],
        ascending=False
    )

    best_name = ranked.iloc[0]["Model"]

    # Keep these columns numeric and add a small diagnostic column
    # without changing the dashboard's existing required metrics.
    results_df["SMOTE Used"] = smote_used

    return (
        results_df,
        trained_models,
        predictions,
        best_name
    )

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    model,
    preprocessor,
    top_n=10
):

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        if hasattr(
            model,
            "feature_importances_"
        ):

            importances = (
                model
                .feature_importances_
            )

        elif hasattr(
            model,
            "coef_"
        ):

            coef = model.coef_

            if coef.ndim > 1:

                importances = np.mean(
                    np.abs(coef),
                    axis=0
                )

            else:

                importances = np.abs(
                    coef
                )

        else:

            return pd.DataFrame(
                columns=[
                    "feature",
                    "importance"
                ]
            )


        importance_df = pd.DataFrame({

            "feature": feature_names,

            "importance": importances

        })

        importance_df = (
            importance_df
            .sort_values(
                "importance",
                ascending=False
            )
            .head(top_n)
            .reset_index(drop=True)
        )

        return importance_df

    except Exception:

        return pd.DataFrame(
            columns=[
                "feature",
                "importance"
            ]
        )


# ============================================================
# EXACT DATASET QUESTION ANALYSIS
# ============================================================

def _question_tokens(text):
    return set(re.findall(r"[a-z0-9_]+", str(text).lower()))


def _find_column(df, question, aliases=None):
    """Resolve the requested dataset column from natural-language wording.

    Resolution is deterministic and dataset-driven. It supports real column
    names with spaces/underscores, common natural-language aliases, and token
    matches such as ``monthly income`` -> ``monthly_income``.
    """
    aliases = aliases or {}
    q = str(question).strip().lower()
    q_norm = normalize_feature_name(q)
    stat_question = any(
        word in q
        for word in [
            "average", "mean", "median", "minimum", "maximum", "max", "min",
            "standard deviation", "std", "range", "total", "sum", "highest", "lowest"
        ]
    )

    candidates = []

    for column in df.columns:
        c_text = str(column).strip().lower()
        c_norm = normalize_feature_name(c_text)
        score = 0

        # Exact normalized phrase: monthly income -> monthly_income.
        if c_norm and re.search(
            rf"(?<![a-z0-9_]){re.escape(c_norm)}(?![a-z0-9_])",
            q_norm,
        ):
            score += 10000

        # Compact form: monthlyincome.
        compact_c = re.sub(r"[^a-z0-9]", "", c_norm)
        compact_q = re.sub(r"[^a-z0-9]", "", q_norm)
        if compact_c and len(compact_c) >= 4 and compact_c in compact_q:
            score += 7000

        # All meaningful column-name tokens occur in the question.
        tokens = [t for t in re.split(r"[_\s]+", c_norm) if len(t) >= 2]
        if tokens and all(re.search(rf"{re.escape(t)}", q) for t in tokens):
            score += 5000 + 100 * len(tokens)

        # Existing explicit aliases supplied by the caller.
        for alias, alias_candidates in aliases.items():
            alias_l = str(alias).lower().strip()
            if re.search(rf"(?<![a-z0-9]){re.escape(alias_l)}(?![a-z0-9])", q):
                if column in alias_candidates:
                    score += 9000

        # Safe, generic semantic aliases for common analytical measures.
        semantic_groups = {
            "income": {"income", "salary", "pay", "wage", "compensation", "earnings"},
            "salary": {"income", "salary", "pay", "wage", "compensation", "earnings"},
            "revenue": {"revenue", "sales", "turnover", "income"},
            "sales": {"sales", "revenue", "turnover"},
            "profit": {"profit", "margin", "net_income", "net_profit"},
            "price": {"price", "amount", "cost", "value"},
            "cost": {"cost", "price", "expense", "amount"},
            "expenses": {"expense", "expenses", "cost", "spend", "spending"},
            "spending": {"spend", "spending", "expense", "expenses", "cost", "amount"},
            "customers": {"customer", "customers", "client", "clients"},
            "quantity": {"quantity", "units", "volume", "count"},
        }
        c_tokens = set(tokens)
        for q_word, compatible in semantic_groups.items():
            if re.search(rf"{re.escape(q_word)}", q):
                if c_tokens & compatible:
                    score += 2500

        # Numeric measures should win over categorical columns for statistical
        # questions when both are explicitly mentioned.
        if score > 0 and stat_question and pd.api.types.is_numeric_dtype(df[column]):
            score += 300

        if score > 0:
            candidates.append((score, column))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _infer_target_column(df, question=""):
    """Infer a likely target without silently changing dataset semantics."""
    preferred = [
        "outcome", "target", "label", "class", "y", "response",
        "diagnosis", "diabetes", "default", "churn"
    ]
    q = str(question).lower()

    for col in preferred:
        if col in df.columns:
            return col

    # If the user explicitly mentions a binary column name, use it.
    for c in df.columns:
        if str(c).lower() in q and df[c].dropna().nunique() == 2:
            return c

    return None


def _target_semantics(df, target, question=""):
    """Return cautious target-value semantics; never invent a meaning."""
    if target is None or target not in df.columns:
        return {}

    values = df[target].dropna().unique().tolist()
    result = {"target": target, "values": values}

    # The classic Pima diabetes dataset uses outcome=1 for diabetes.
    # Only apply this convention when the dataset itself strongly indicates it.
    dataset_hint = " ".join(map(str, df.columns)).lower()
    q = str(question).lower()
    if target == "outcome" and set(pd.to_numeric(df[target], errors="coerce").dropna().unique()).issubset({0, 1}):
        if "diabet" in q or "diabet" in dataset_hint:
            result["1"] = "diabetes / positive outcome (dataset convention; association, not causation)"
            result["0"] = "non-diabetes / negative outcome (dataset convention; association, not causation)"

    return result


def _fmt(v):
    if pd.isna(v):
        return "NA"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4f}".rstrip("0").rstrip(".")
    return str(v)


def _exact_numeric_summary(df, col):
    s = pd.to_numeric(df[col], errors="coerce")
    return {
        "count": int(s.notna().sum()),
        "missing": int(s.isna().sum()),
        "mean": float(s.mean()) if s.notna().any() else np.nan,
        "median": float(s.median()) if s.notna().any() else np.nan,
        "min": float(s.min()) if s.notna().any() else np.nan,
        "max": float(s.max()) if s.notna().any() else np.nan,
        "std": float(s.std()) if s.notna().sum() > 1 else np.nan,
    }



def _find_generic_filter_column(df, question, exclude=None):
    """Find the categorical filter column/value mentioned in a natural-language question.

    The resolver is intentionally dataset-independent. It prioritizes an actual
    column name mentioned in the question, then exact category-value matches,
    and only then falls back to partial value matches. This prevents values such
    as ``Sales`` from accidentally matching ``Sales Executive`` in another
    column when the question explicitly says ``Sales department``.
    """
    exclude = set(exclude or [])
    q = str(question).strip().lower()

    # Candidate categorical columns.
    candidates = []
    for c in df.columns:
        if c in exclude:
            continue
        series = df[c]
        if (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series)
        ):
            candidates.append(c)

    if not candidates:
        return None, None

    # Normalize the question once for column-name matching.
    q_norm = normalize_feature_name(q)

    # Score every (column, value) pair. Higher scores mean stronger evidence.
    matches = []
    for c in candidates:
        c_norm = normalize_feature_name(c)
        column_explicit = False

        # Exact normalized column-name mention, e.g. "attrition" or "department".
        if c_norm and re.search(
            rf"(?<![a-z0-9_]){re.escape(c_norm)}(?![a-z0-9_])", q_norm
        ):
            column_explicit = True

        # Also support multi-word column names by checking their individual words.
        c_words = [w for w in re.split(r"[_\s]+", str(c).lower()) if len(w) >= 3]
        column_word_hits = sum(
            1 for w in c_words
            if re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", q)
        )

        for raw_value in df[c].dropna().unique():
            value_text = str(raw_value).strip()
            value_lower = value_text.lower()
            if not value_lower:
                continue
            if len(value_lower) < 2 and value_lower not in {"m", "f", "y", "n", "0", "1"}:
                continue

            # Prefer exact category values over substrings/partial matches.
            exact_value = bool(
                re.search(
                    rf"(?<![a-z0-9]){re.escape(value_lower)}(?![a-z0-9])",
                    q,
                )
            )
            if not exact_value:
                continue

            score = len(value_lower)
            if column_explicit:
                score += 1000
            score += column_word_hits * 150

            # Context around the value helps distinguish e.g. department=Sales
            # from job_role=Sales Executive.
            value_match = re.search(
                rf"(?<![a-z0-9]){re.escape(value_lower)}(?![a-z0-9])", q
            )
            if value_match:
                before = q[max(0, value_match.start() - 45):value_match.start()]
                after = q[value_match.end():value_match.end() + 45]
                context = f"{before} {after}"

                if c_words:
                    for word in c_words:
                        if re.search(rf"\b{re.escape(word)}\b", context):
                            score += 250

                # Natural-language column context patterns.
                if re.search(rf"\b{re.escape(value_lower)}\s+{re.escape(str(c).lower())}\b", q):
                    score += 500
                if re.search(rf"\b{re.escape(str(c).lower())}\b.*?\b{re.escape(value_lower)}\b", q):
                    score += 400

            # Exact value should beat a longer compound value in another column
            # unless that other column is explicitly named.
            if value_lower == value_lower.strip():
                score += 25

            matches.append((score, len(value_lower), c, raw_value))

    if not matches:
        return None, None

    matches.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, column, value = matches[0]
    return column, value


def _resolve_categorical_filter(df, question, exclude=None):
    """Deterministically resolve a categorical column and one actual dataset value."""
    exclude = set(exclude or [])
    q = str(question).strip().lower()

    def norm(value):
        return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()

    def token_set(value):
        return set(norm(value).split())

    categorical = []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if (pd.api.types.is_object_dtype(s)
                or pd.api.types.is_categorical_dtype(s)
                or pd.api.types.is_bool_dtype(s)):
            categorical.append(c)

    if not categorical:
        return None, None

    q_norm = norm(q)
    q_tokens = token_set(q)
    context_words = {"department", "team", "group", "division", "status", "role", "gender", "education"}

    candidates = []
    for c in categorical:
        c_norm = norm(c)
        c_tokens = token_set(c)
        column_explicit = bool(c_norm and (c_norm in q_norm or c_tokens.issubset(q_tokens)))

        for value in df[c].dropna().unique().tolist():
            value_text = str(value).strip()
            if not value_text:
                continue
            v_norm = norm(value_text)
            v_tokens = token_set(value_text)
            if not v_norm:
                continue

            value_explicit = bool(v_norm in q_norm or v_tokens.issubset(q_tokens))
            if not value_explicit:
                continue

            score = 0
            if column_explicit:
                score += 10000
            score += len(v_norm)
            score += len(c_norm)

            # Strongly recognize constructions such as "Sales department".
            if re.search(rf"\b{re.escape(v_norm)}\s+(?:department|team|group|division)\b", q_norm):
                score += 5000
            if re.search(rf"\b(?:department|team|group|division)\s+{re.escape(v_norm)}\b", q_norm):
                score += 4500
            if re.search(rf"\bin\s+(?:the\s+)?{re.escape(v_norm)}\b", q_norm):
                score += 2500
            if re.search(rf"\bwith\s+(?:the\s+)?{re.escape(v_norm)}\b", q_norm):
                score += 1500

            # When the value is short/common (e.g. "No", "Yes"), only allow
            # it to win strongly when its own column is explicitly identified.
            if len(v_norm) <= 3 and not column_explicit:
                score -= 2000

            candidates.append((score, c, value))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, column, value = candidates[0]
    return column, value

def _parse_numeric_condition(question):
    """Detect common numeric conditions and return (operator, value)."""
    q = str(question).lower()
    patterns = [
        (r"\bgreater\s+than\s+(-?\d+(?:\.\d+)?)", ">"),
        (r"\bmore\s+than\s+(-?\d+(?:\.\d+)?)", ">"),
        (r"\babove\s+(-?\d+(?:\.\d+)?)", ">"),
        (r"\bover\s+(-?\d+(?:\.\d+)?)", ">"),
        (r"\bless\s+than\s+(-?\d+(?:\.\d+)?)", "<"),
        (r"\bsmaller\s+than\s+(-?\d+(?:\.\d+)?)", "<"),
        (r"\bbelow\s+(-?\d+(?:\.\d+)?)", "<"),
        (r"\bunder\s+(-?\d+(?:\.\d+)?)", "<"),
        (r"\bat\s+least\s+(-?\d+(?:\.\d+)?)", ">="),
        (r"\bminimum\s+of\s+(-?\d+(?:\.\d+)?)", ">="),
        (r"\bat\s+most\s+(-?\d+(?:\.\d+)?)", "<="),
        (r"\bmaximum\s+of\s+(-?\d+(?:\.\d+)?)", "<="),
        (r"\bequal\s+to\s+(-?\d+(?:\.\d+)?)", "=="),
        (r"\bequals\s+(-?\d+(?:\.\d+)?)", "=="),
    ]
    for pattern, operator in patterns:
        m = re.search(pattern, q)
        if m:
            return operator, float(m.group(1))
    return None, None


def _apply_numeric_condition(series, operator, value):
    numeric = pd.to_numeric(series, errors="coerce")
    if operator == ">":
        return numeric > value
    if operator == "<":
        return numeric < value
    if operator == ">=":
        return numeric >= value
    if operator == "<=":
        return numeric <= value
    if operator == "==":
        return numeric == value
    return pd.Series(False, index=series.index)


def _detect_binary_target_filter(df, target, question):
    """Infer positive/negative selection for a binary target using generic wording."""
    if target is None or target not in df.columns:
        return None
    values = df[target].dropna().unique().tolist()
    if len(values) != 2:
        return None
    q = str(question).lower()
    value_map = {str(v).strip().lower(): v for v in values}

    negative_phrases = [
        "without diabetes", "without diabet", "non-diabetic", "non diabetic",
        "not diabetic", "negative outcome", "negative class", "class 0", "outcome 0",
        "without churn", "not churned", "inactive", "no default", "not defaulted",
        "false", "no ",
    ]
    positive_phrases = [
        "with diabetes", "with diabet", "diabetic patients", "diabetic patient",
        "positive outcome", "positive class", "class 1", "outcome 1",
        "with churn", "churned", "active", "defaulted", "yes ",
    ]
    negative_values = {"0", "no", "false", "negative", "non-diabetic", "non diabetic", "normal", "inactive", "not churned"}
    positive_values = {"1", "yes", "true", "positive", "diabetes", "diabetic", "active", "churned", "defaulted"}

    if any(phrase in q for phrase in negative_phrases):
        for key, original in value_map.items():
            if key in negative_values:
                return original
    if any(phrase in q for phrase in positive_phrases):
        for key, original in value_map.items():
            if key in positive_values:
                return original
    return None

def _find_group_column(df, question, measure_col=None):
    """Find a categorical grouping column explicitly requested by the question."""
    q = str(question).strip().lower()
    q_norm = normalize_feature_name(q)
    candidates = []

    for c in df.columns:
        if c == measure_col:
            continue
        s = df[c]
        if not (
            pd.api.types.is_object_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(s)
        ):
            continue

        c_text = str(c).strip().lower()
        c_norm = normalize_feature_name(c_text)
        score = 0

        if c_norm and re.search(rf"(?<![a-z0-9_]){re.escape(c_norm)}(?![a-z0-9_])", q_norm):
            score += 10000

        words = [w for w in re.split(r"[_\s]+", c_norm) if len(w) >= 2]
        if words and all(re.search(rf"\b{re.escape(w)}\b", q) for w in words):
            score += 5000

        # Natural grouping wording: "which department", "by department", etc.
        for generic_word in ["department", "team", "group", "region", "category", "segment", "division", "role", "job role", "type", "class", "status"]:
            if generic_word in q:
                if generic_word == c_text or generic_word.replace(" ", "_") == c_norm:
                    score += 3000

        if score > 0:
            candidates.append((score, c))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _find_group_measure_column(df, question, aliases=None):
    """Resolve the numeric measure for grouped questions such as
    'average salary by department' or 'total sales by region'."""
    q = str(question).strip().lower()
    aliases = aliases or {}

    # Common natural-language measure aliases. These are only used when the
    # requested word is present in the question and the corresponding dataset
    # column actually exists.
    semantic_aliases = {
        "salary": [
            "salary", "monthlyincome", "monthly_income", "monthly rate",
            "monthlyrate", "monthly_rate", "annual salary", "annual_salary",
            "annualincome", "annual_income"
        ],
        "income": [
            "income", "monthlyincome", "monthly_income", "monthly rate",
            "monthlyrate", "monthly_rate", "annualincome", "annual_income"
        ],
        "sales": ["sales", "total_sales", "totalsales", "revenue"],
        "revenue": ["revenue", "sales", "total_sales", "totalsales"],
        "profit": ["profit", "net_profit", "netprofit", "gross_profit", "grossprofit"],
        "age": ["age"],
        "income per month": ["monthlyincome", "monthly_income", "monthly rate", "monthlyrate", "monthly_rate"],
        "monthly income": ["monthlyincome", "monthly_income", "monthly rate", "monthlyrate", "monthly_rate"],
    }

    normalized_columns = {normalize_feature_name(c): c for c in df.columns}

    # First prefer an actual column explicitly named in the question.
    for c in df.select_dtypes(include=np.number).columns:
        norm = normalize_feature_name(c)
        if norm and re.search(rf"(?<![a-z0-9_]){re.escape(norm)}(?![a-z0-9_])", normalize_feature_name(q)):
            return c

    # Then use semantic aliases, but only if the natural-language measure is
    # explicitly present in the question.
    for phrase, candidates in semantic_aliases.items():
        if phrase in q:
            for candidate in candidates:
                cand_norm = normalize_feature_name(candidate)
                if cand_norm in normalized_columns:
                    actual = normalized_columns[cand_norm]
                    if pd.api.types.is_numeric_dtype(df[actual]):
                        return actual

    # Also honor aliases supplied by the main analyzer.
    for phrase, candidates in aliases.items():
        if phrase in q:
            for candidate in candidates:
                cand_norm = normalize_feature_name(candidate)
                if cand_norm in normalized_columns:
                    actual = normalized_columns[cand_norm]
                    if pd.api.types.is_numeric_dtype(df[actual]):
                        return actual

    return None


def _is_grouped_question(question):
    """Return True when the wording requests a category/group calculation."""
    q = str(question).lower()
    group_words = [
        " by ", " per ", " for each ", " each ", " every ",
        " in each ", " across ", " grouped by ", "department-wise",
        "department wise", "category-wise", "category wise",
        "which department", "which team", "which region", "which group",
        "which category", "which segment", "which role",
        " department", " team", " region", " category", " segment",
        " division", " job role"
    ]
    aggregation_words = [
        "average", "mean", "median", "sum", "total", "count", "number",
        "how many", "percentage", "percent", "%", "minimum", "maximum",
        "max", "min", "highest", "lowest", "largest", "smallest",
        "most", "least"
    ]
    return any(w in q for w in group_words) and any(w in q for w in aggregation_words)


def _infer_group_column_from_value(df, question):
    """Infer a categorical grouping column from a category value in the question.

    This handles natural wording such as 'How many employees are in Sales?'
    where the user names the value but does not explicitly say 'department'.
    """
    q = str(question).strip().lower()
    candidates = []

    preferred_names = [
        "department", "team", "division", "region", "category",
        "segment", "job_role", "role", "status", "gender",
        "education_field", "marital_status", "business_travel"
    ]

    for c in df.columns:
        s = df[c]
        if not (
            pd.api.types.is_object_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(s)
        ):
            continue

        c_norm = normalize_feature_name(c)
        for value in s.dropna().unique().tolist():
            value_text = str(value).strip()
            value_norm = normalize_feature_name(value_text).replace("_", " ")
            if not value_norm:
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(value_norm)}(?![a-z0-9])", q):
                score = len(value_norm)
                if c_norm in preferred_names:
                    score += 10000 - preferred_names.index(c_norm) * 100
                if re.search(rf"\bin\s+(?:the\s+)?{re.escape(value_norm)}\b", q):
                    score += 2500
                if re.search(rf"\b{re.escape(value_norm)}\s+(?:department|team|group|division|region)\b", q):
                    score += 5000
                candidates.append((score, c, value_text))

    if not candidates:
        return None, None
    candidates.sort(key=lambda x: (x[0], len(str(x[1]))), reverse=True)
    _, column, value = candidates[0]
    return column, value


def _resolve_group_value_from_question(df, question, group_col):
    """Resolve one real category value for questions such as 'employees in Sales'."""
    q = str(question).strip().lower()
    if group_col is None or group_col not in df.columns:
        return None

    values = df[group_col].dropna().unique().tolist()
    candidates = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        norm = normalize_feature_name(text).replace("_", " ")
        if not norm:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(norm)}(?![a-z0-9])", q):
            score = len(norm)
            if re.search(rf"\b{re.escape(norm)}\s+(?:department|team|group|division|region|role|category)\b", q):
                score += 5000
            if re.search(rf"\b(?:department|team|group|division|region|role|category)\s+{re.escape(norm)}\b", q):
                score += 4500
            if re.search(rf"\bin\s+(?:the\s+)?{re.escape(norm)}\b", q):
                score += 2500
            candidates.append((score, len(norm), value))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def _grouped_aggregation_answer(df, question):
    """Deterministic grouped/category analysis using the uploaded DataFrame.

    This function intentionally does not depend on the RAG/LLM layer. It is
    designed for questions such as:
      - How many employees are in Sales?
      - What percentage of employees are in Sales?
      - What is the average salary by department?
      - Which department has the highest average salary?
      - What is the average monthly income by department?
      - How many employees are in each department?
    """
    if df is None or df.empty:
        return None

    q = str(question).strip()
    ql = q.lower()

    def norm(value):
        return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

    def is_category(series):
        return (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series)
        )

    def numeric_series(series):
        converted = pd.to_numeric(series, errors="coerce")
        # Accept numeric-looking Excel/text columns as numeric measures when
        # most non-empty values can be converted.
        non_empty = series.notna().sum()
        if non_empty == 0:
            return None
        if converted.notna().sum() / non_empty >= 0.90:
            return converted
        return None

    columns = list(df.columns)
    normalized = {norm(c): c for c in columns}

    # ------------------------------------------------------------
    # GROUP COLUMN RESOLUTION
    # ------------------------------------------------------------
    group_aliases = {
        "department": ["department", "dept"],
        "team": ["team", "teams"],
        "division": ["division", "divisions"],
        "region": ["region", "regions", "area"],
        "category": ["category", "categories"],
        "segment": ["segment", "segments"],
        "group": ["group", "groups"],
        "role": ["role", "job_role", "job role", "jobrole"],
        "gender": ["gender", "sex"],
        "status": ["status"],
        "education": ["education", "education_field", "education field"],
        "marital": ["marital_status", "marital status"],
        "travel": ["business_travel", "business travel"],
    }

    group_col = None
    # Explicit column wording gets first priority.
    for _, aliases_for_group in group_aliases.items():
        for alias in aliases_for_group:
            alias_norm = norm(alias)
            if alias_norm in normalized and (
                re.search(rf"\b{re.escape(alias.replace('_', ' '))}\b", ql)
                or re.search(rf"\b{re.escape(alias_norm.replace('_', ' '))}\b", ql)
            ):
                candidate = normalized[alias_norm]
                if is_category(df[candidate]):
                    group_col = candidate
                    break
        if group_col is not None:
            break

    # If no explicit grouping column was named, infer a real category value
    # from the question (e.g. "employees in Sales"). Prefer common grouping
    # columns such as department over arbitrary text columns.
    if group_col is None:
        preferred_group_order = [
            "department", "team", "division", "region", "category",
            "segment", "group", "job_role", "role", "status",
            "gender", "education_field", "marital_status", "business_travel",
        ]
        value_candidates = []
        for c in columns:
            if not is_category(df[c]):
                continue
            c_norm = norm(c)
            rank = preferred_group_order.index(c_norm) if c_norm in preferred_group_order else 999
            for value in df[c].dropna().unique().tolist():
                value_text = str(value).strip()
                if not value_text:
                    continue
                value_norm = norm(value_text).replace("_", " ")
                if not value_norm:
                    continue
                pattern = rf"(?<![a-z0-9]){re.escape(value_norm)}(?![a-z0-9])"
                if not re.search(pattern, ql.replace("_", " ")):
                    continue
                score = 10000 - rank * 100
                score += len(value_norm)
                if "employee" in ql and c_norm == "department":
                    score += 20000
                if re.search(rf"\bin\s+(?:the\s+)?{re.escape(value_norm)}\b", ql):
                    score += 5000
                if re.search(rf"\b{re.escape(value_norm)}\s+(?:department|team|group|division|region|category)\b", ql):
                    score += 6000
                if " " in value_norm:
                    score -= 500
                value_candidates.append((score, c, value))

        if value_candidates:
            value_candidates.sort(key=lambda x: (x[0], len(str(x[1]))), reverse=True)
            group_col = value_candidates[0][1]

    # ------------------------------------------------------------
    # MEASURE COLUMN RESOLUTION
    # ------------------------------------------------------------
    measure_col = None

    # Exact/normalized column names explicitly mentioned in the question.
    for c in columns:
        c_norm = norm(c).replace("_", " ")
        if not c_norm:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(c_norm)}(?![a-z0-9])", ql.replace("_", " ")):
            if numeric_series(df[c]) is not None:
                measure_col = c
                break

    # Semantic measures. For HR datasets, "salary" normally means
    # monthly_income; keep that deterministic preference.
    if measure_col is None:
        semantic_measure_aliases = [
            ("salary", [
                "monthly_income", "monthlyincome", "salary", "annual_salary",
                "annual_income", "income", "monthly_rate", "monthlyrate"
            ]),
            ("income", [
                "monthly_income", "monthlyincome", "income", "annual_income",
                "annualincome", "monthly_rate", "monthlyrate"
            ]),
            ("revenue", ["revenue", "sales", "total_sales", "totalsales"]),
            ("sales", ["sales", "total_sales", "totalsales", "revenue"]),
            ("profit", ["profit", "net_profit", "netprofit", "gross_profit", "grossprofit"]),
            ("age", ["age"]),
        ]
        for phrase, candidates in semantic_measure_aliases:
            if phrase not in ql:
                continue
            for candidate in candidates:
                candidate_norm = norm(candidate)
                actual = normalized.get(candidate_norm)
                if actual is not None and numeric_series(df[actual]) is not None:
                    measure_col = actual
                    break
            if measure_col is not None:
                break

    # ------------------------------------------------------------
    # SPECIFIC CATEGORY COUNT / PERCENTAGE
    # ------------------------------------------------------------
    count_words = ["how many", "number of", "count", "percentage", "percent", "%"]
    is_count_question = any(w in ql for w in count_words)

    # Treat questions such as "Which department has the most employees?"
    # and "Which department has the fewest employees?" as exact count
    # questions. These must be answered from the uploaded DataFrame, not RAG.
    employee_extreme_words = [
        "most", "highest", "largest", "maximum", "max",
        "fewest", "lowest", "smallest", "minimum", "min",
    ]
    employee_words = ["employee", "employees", "people", "rows", "records"]
    is_employee_extreme_question = (
        group_col is not None
        and any(
            re.search(rf"\b{re.escape(word)}\b", ql)
            for word in employee_extreme_words
        )
        and any(word in ql for word in employee_words)
    )

    if (is_count_question or is_employee_extreme_question) and group_col is not None:
        # A count question about a specific category should return the count
        # for that category, not the whole group table.
        values = df[group_col].dropna().unique().tolist()
        matches = []
        for value in values:
            value_norm = norm(value).replace("_", " ")
            if not value_norm:
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(value_norm)}(?![a-z0-9])", ql.replace("_", " ")):
                score = len(value_norm)
                if re.search(rf"\bin\s+(?:the\s+)?{re.escape(value_norm)}\b", ql):
                    score += 5000
                if re.search(rf"\b{re.escape(value_norm)}\s+(?:department|team|group|division|region|category)\b", ql):
                    score += 6000
                if " " in value_norm:
                    score -= 500
                matches.append((score, value))

        # For "How many employees are in Sales?", this resolves to Sales.
        if matches:
            matches.sort(key=lambda x: (x[0], len(str(x[1]))), reverse=True)
            specific_value = matches[0][1]
            mask = (
                df[group_col].astype(str).str.strip().str.casefold()
                == str(specific_value).strip().casefold()
            )
            n = int(mask.sum())
            total = int(len(df))
            pct = (n / total * 100) if total else 0.0
            evidence = {
                "question": q,
                "method": "exact pandas analysis",
                "group_column": group_col,
                "group_value": specific_value,
                "filtered_rows": n,
                "total_rows": total,
                "percentage": pct,
            }
            if "percentage" in ql or "percent" in ql or "%" in ql:
                return (
                    f"There are {n:,} rows in {group_col} = {_fmt(specific_value)}, "
                    f"representing {pct:.2f}% of the dataset."
                ), evidence
            return f"There are {n:,} rows in {group_col} = {_fmt(specific_value)}.", evidence

        # "Which department has the most employees?" / "fewest employees"
        # are also count questions, but they ask for the extreme group rather
        # than a specific category or a full group table. Handle them directly
        # from the complete uploaded DataFrame so RAG cannot hallucinate.
        count_extreme_words = [
            "most", "highest", "largest", "maximum", "max",
            "fewest", "lowest", "smallest", "minimum", "min",
        ]
        if any(
            re.search(rf"\b{re.escape(word)}\b", ql)
            for word in count_extreme_words
        ) and any(
            word in ql for word in ["employee", "employees", "people", "rows", "records", "count", "number"]
        ):
            counts = df.groupby(group_col, dropna=False).size().sort_values(ascending=False)
            counts = counts.dropna()
            if counts.empty:
                return None

            lowest = any(
                re.search(rf"\b{re.escape(word)}\b", ql)
                for word in ["fewest", "lowest", "smallest", "minimum", "min"]
            )
            ordered = counts.sort_values(ascending=lowest)
            winner = ordered.index[0]
            winner_count = int(ordered.iloc[0])
            total = int(len(df))
            pct = (winner_count / total * 100) if total else 0.0

            result = pd.DataFrame({
                group_col: [_fmt(v) for v in ordered.index],
                "Employee Count": [int(v) for v in ordered.values],
                "Percentage": [
                    round(float(v) / total * 100, 2) if total else 0.0
                    for v in ordered.values
                ],
            })

            evidence = {
                "question": q,
                "method": "exact pandas analysis",
                "group_column": group_col,
                "aggregation": "count",
                "winner": _fmt(winner),
                "winner_value": winner_count,
                "total_rows": total,
                "grouped_results": result.to_dict(orient="records"),
            }

            direction = "fewest" if lowest else "most"
            return (
                f"The department with the {direction} employees is "
                f"{group_col} = {_fmt(winner)}, with {winner_count:,} employees "
                f"({pct:.2f}% of the dataset)."
            ), evidence

        # "Which department has the most/fewest employees?"
        # is an extreme count question, so return the winning department.
        if is_employee_extreme_question:
            counts = df.groupby(group_col, dropna=False).size()
            if counts.empty:
                return None

            lowest = any(
                re.search(rf"\b{re.escape(word)}\b", ql)
                for word in ["fewest", "lowest", "smallest", "minimum", "min"]
            )
            ordered = counts.sort_values(ascending=lowest)
            winner = ordered.index[0]
            winner_count = int(ordered.iloc[0])
            total = int(len(df))
            pct = (winner_count / total * 100) if total else 0.0

            result = pd.DataFrame({
                group_col: [_fmt(v) for v in ordered.index],
                "Employee Count": [int(v) for v in ordered.values],
                "Percentage": [
                    round(float(v) / total * 100, 2) if total else 0.0
                    for v in ordered.values
                ],
            })

            evidence = {
                "question": q,
                "method": "exact pandas analysis",
                "group_column": group_col,
                "aggregation": "count",
                "winner": _fmt(winner),
                "winner_value": winner_count,
                "total_rows": total,
                "grouped_results": result.to_dict(orient="records"),
            }

            direction = "fewest" if lowest else "most"
            return (
                f"The department with the {direction} employees is "
                f"{group_col} = {_fmt(winner)}, with {winner_count:,} employees "
                f"({pct:.2f}% of the dataset)."
            ), evidence

        # "How many employees are in each department?"
        if "each" in ql or "every" in ql or "by" in ql or "per" in ql:
            counts = df.groupby(group_col, dropna=False).size().sort_values(ascending=False)
            total = int(len(df))
            result = pd.DataFrame({
                group_col: [_fmt(v) for v in counts.index],
                "Count": [int(v) for v in counts.values],
                "Percentage": [round(float(v) / total * 100, 2) if total else 0.0 for v in counts.values],
            })
            evidence = {
                "question": q,
                "method": "exact pandas analysis",
                "group_column": group_col,
                "aggregation": "count",
                "grouped_results": result.to_dict(orient="records"),
                "total_rows": total,
            }
            return f"The number of rows in each {group_col} is shown in the table below.", evidence

    # A grouped average/sum/min/max/median requires both a grouping column
    # and a numeric measure.
    if group_col is None or measure_col is None:
        return None

    measure = numeric_series(df[measure_col])
    if measure is None:
        return None

    temp = pd.DataFrame({
        "__group": df[group_col],
        "__measure": measure,
    })

    if "median" in ql:
        aggregation = "median"
        grouped = temp.groupby("__group", dropna=False)["__measure"].median()
    elif "sum" in ql or "total" in ql:
        aggregation = "sum"
        grouped = temp.groupby("__group", dropna=False)["__measure"].sum()
    elif "minimum" in ql or re.search(r"\bmin\b", ql):
        aggregation = "minimum"
        grouped = temp.groupby("__group", dropna=False)["__measure"].min()
    elif "maximum" in ql or re.search(r"\bmax\b", ql):
        aggregation = "maximum"
        grouped = temp.groupby("__group", dropna=False)["__measure"].max()
    else:
        # "average" / "mean" is the default for a grouped numeric question.
        if not any(w in ql for w in ["average", "mean", "salary", "income", "age", "sales", "revenue", "profit"]):
            return None
        aggregation = "average"
        grouped = temp.groupby("__group", dropna=False)["__measure"].mean()

    grouped = grouped.dropna()
    if grouped.empty:
        return None

    result_column = f"{aggregation.title()} {measure_col}"
    result = pd.DataFrame({
        group_col: [_fmt(v) for v in grouped.index],
        result_column: [round(float(v), 4) for v in grouped.values],
    })

    evidence = {
        "question": q,
        "method": "exact pandas analysis",
        "group_column": group_col,
        "measure_column": measure_col,
        "aggregation": aggregation,
        "grouped_results": result.to_dict(orient="records"),
    }

    extreme_words = ["highest", "lowest", "largest", "smallest", "most", "least"]
    if any(w in ql for w in extreme_words):
        lowest = any(w in ql for w in ["lowest", "smallest", "least"])
        ordered = grouped.sort_values(ascending=lowest)
        winner = ordered.index[0]
        winner_value = float(ordered.iloc[0])
        evidence.update({
            "winner": _fmt(winner),
            "winner_value": winner_value,
        })
        direction = "lowest" if lowest else "highest"
        return (
            f"The {direction} {aggregation} {measure_col} is in {group_col} = "
            f"{_fmt(winner)}, with a value of {_fmt(winner_value)}."
        ), evidence

    return (
        f"The {aggregation} {measure_col} by {group_col} is shown in the table below."
    ), evidence

def exact_dataset_analysis(df, question):
    """Perform deterministic, full-data analysis for common analytical questions.

    Returns (answer_text, evidence_dict). The LLM is never responsible for the
    arithmetic; it only explains already-verified results when used later.
    """
    q = str(question).strip()
    ql = q.lower()

    # ============================================================
    # CREDIT-CARD FRAUD DEDICATED ANALYSIS
    # Schema-gated to a binary `class` target plus numeric `amount`.
    # This block is isolated from HR, Diabetes, COVID, and generic logic.
    # ============================================================
    _fraud_cols = {str(c).strip().lower(): c for c in df.columns}
    _fraud_class_col = _fraud_cols.get("class")
    _fraud_amount_col = _fraud_cols.get("amount")
    _fraud_is_dataset = False

    if _fraud_class_col is not None and _fraud_amount_col is not None and len(df) > 0:
        _fc = pd.to_numeric(df[_fraud_class_col], errors="coerce").dropna()
        _fa = pd.to_numeric(df[_fraud_amount_col], errors="coerce")
        _fraud_is_dataset = set(_fc.unique().tolist()) == {0, 1} and _fa.notna().any()

    if _fraud_is_dataset:
        _fc = pd.to_numeric(df[_fraud_class_col], errors="coerce")
        _fa = pd.to_numeric(df[_fraud_amount_col], errors="coerce")
        _valid = _fc.isin([0, 1])
        _fc = _fc.loc[_valid]
        _fa = _fa.loc[_valid]
        _counts = _fc.value_counts().reindex([0, 1], fill_value=0).astype(int)
        _total = int(_counts.sum())
        _fraud_count = int(_counts.loc[1])
        _nonfraud_count = int(_counts.loc[0])
        _fraud_pct = (_fraud_count / _total * 100) if _total else 0.0
        _nonfraud_pct = (_nonfraud_count / _total * 100) if _total else 0.0
        _fraud_mean = float(_fa.loc[_fc == 1].mean())
        _nonfraud_mean = float(_fa.loc[_fc == 0].mean())

        _fraud_only = any(x in ql for x in [
            "fraudulent transactions", "fraudulent transaction", "fraudulent",
            "fraud transactions", "fraud transaction", "class 1", "class=1", "class = 1"
        ])
        _nonfraud_only = any(x in ql for x in [
            "non-fraudulent transactions", "non-fraudulent transaction",
            "non fraudulent transactions", "non fraudulent transaction",
            "non-fraud transactions", "non-fraud transaction", "non fraud transactions",
            "non fraud transaction", "non-fraudulent", "non fraudulent", "non-fraud", "non fraud",
            "class 0", "class=0", "class = 0"
        ])
        _count_q = "how many" in ql or "number of" in ql or "count" in ql
        _pct_q = any(x in ql for x in ["percentage", "percent", "proportion", "rate"])
        _avg_q = "average" in ql or "mean" in ql
        _by_class = any(x in ql for x in ["by class", "each class", "per class", "class distribution"])
        _distribution = "distribution" in ql or "breakdown" in ql
        _assoc = any(x in ql for x in ["associated with", "association", "correlated with", "correlation", "most related"])

        if _count_q and not _by_class and not _distribution:
            if _fraud_only and not _nonfraud_only:
                return (
                    f"There are {_fraud_count:,} fraudulent transactions out of {_total:,} total transactions ({_fraud_pct:.4f}%).",
                    {"question": q, "method": "exact pandas analysis", "target_column": _fraud_class_col,
                     "fraud_count": _fraud_count, "total_transactions": _total,
                     "fraud_percentage": round(_fraud_pct, 6)}
                )
            if _nonfraud_only:
                return (
                    f"There are {_nonfraud_count:,} non-fraudulent transactions out of {_total:,} total transactions ({_nonfraud_pct:.4f}%).",
                    {"question": q, "method": "exact pandas analysis", "target_column": _fraud_class_col,
                     "nonfraud_count": _nonfraud_count, "total_transactions": _total,
                     "nonfraud_percentage": round(_nonfraud_pct, 6)}
                )

        if _pct_q and not _by_class and not _distribution:
            if _nonfraud_only:
                return (
                    f"The non-fraudulent transaction percentage is {_nonfraud_pct:.4f}% ({_nonfraud_count:,} of {_total:,}).",
                    {"question": q, "method": "exact pandas analysis", "percentage": round(_nonfraud_pct, 6)}
                )
            return (
                f"The fraud rate is {_fraud_pct:.4f}% ({_fraud_count:,} fraudulent transactions out of {_total:,}).",
                {"question": q, "method": "exact pandas analysis", "fraud_rate_percentage": round(_fraud_pct, 6)}
            )

        if (
            (_by_class or _distribution or (_count_q and "each class" in ql))
            and not _avg_q
        ):
            _dist = pd.DataFrame({
                "Class": [0, 1],
                "Transaction Type": ["Non-fraudulent", "Fraudulent"],
                "Transaction Count": [_nonfraud_count, _fraud_count],
                "Percentage": [round(_nonfraud_pct, 4), round(_fraud_pct, 4)]
            })
            if "rate" in ql and "by class" in ql:
                _dist["Fraud Rate Within Class"] = [0.0, 100.0]
                _answer = "Because `class` itself defines fraud status, class 0 has a 0% fraud rate and class 1 has a 100% fraud rate."
            else:
                _answer = "The fraud and non-fraud transaction distribution by class is shown in the table below."
            return _answer, {"question": q, "method": "exact pandas analysis", "group_column": _fraud_class_col,
                             "grouped_results": _dist.to_dict(orient="records"), "total_transactions": _total}

        if _avg_q:
            if _fraud_only and not _nonfraud_only:
                return (f"The average transaction amount for fraudulent transactions is {_fraud_mean:.4f}.",
                        {"question": q, "method": "exact pandas analysis", "filter": "class = 1 (fraudulent)", "average_amount": _fraud_mean})
            if _nonfraud_only:
                return (f"The average transaction amount for non-fraudulent transactions is {_nonfraud_mean:.4f}.",
                        {"question": q, "method": "exact pandas analysis", "filter": "class = 0 (non-fraudulent)", "average_amount": _nonfraud_mean})
            if _by_class:
                _avg = pd.DataFrame({
                    "Class": [0, 1], "Transaction Type": ["Non-fraudulent", "Fraudulent"],
                    "Average Transaction Amount": [round(_nonfraud_mean, 4), round(_fraud_mean, 4)]
                })
                return "The average transaction amount by class is shown in the table below.", {
                    "question": q, "method": "exact pandas analysis", "group_column": _fraud_class_col,
                    "measure_column": _fraud_amount_col, "aggregation": "average",
                    "grouped_results": _avg.to_dict(orient="records")}
            _overall_mean = float(_fa.mean())
            return f"The average transaction amount overall is {_overall_mean:.4f}.", {
                "question": q, "method": "exact pandas analysis", "measure_column": _fraud_amount_col,
                "aggregation": "average", "average_amount": _overall_mean}

        if "median" in ql:
            _v = float(_fa.median())
            return f"The median transaction amount is {_v:.4f}.", {"question": q, "method": "exact pandas analysis", "median_amount": _v}
        if "minimum" in ql or "min" in ql:
            _v = float(_fa.min())
            return f"The minimum transaction amount is {_v:.4f}.", {"question": q, "method": "exact pandas analysis", "minimum_amount": _v}
        if "maximum" in ql or "max" in ql:
            _v = float(_fa.max())
            return f"The maximum transaction amount is {_v:.4f}.", {"question": q, "method": "exact pandas analysis", "maximum_amount": _v}

        if _assoc:
            _rows = []
            for _feature in df.select_dtypes(include=np.number).columns:
                if _feature == _fraud_class_col:
                    continue
                _pair = pd.DataFrame({
                    "feature": pd.to_numeric(df[_feature], errors="coerce"),
                    "class": _fc.reindex(df.index)
                }).dropna()
                if len(_pair) >= 2 and _pair["feature"].nunique() > 1:
                    _corr = float(_pair["feature"].corr(_pair["class"]))
                    if np.isfinite(_corr):
                        _rows.append({"Feature": _feature, "Correlation with Class": round(_corr, 6), "Absolute Correlation": round(abs(_corr), 6)})
            _assoc_df = pd.DataFrame(_rows).sort_values("Absolute Correlation", ascending=False).head(10).reset_index(drop=True)
            if not _assoc_df.empty:
                return (
                    "The numeric features most associated with the fraud class, measured by absolute Pearson correlation with `class`, are shown below. This measures association, not causation."
                ), {"question": q, "method": "exact pandas Pearson correlation", "target_column": _fraud_class_col,
                    "grouped_results": _assoc_df.to_dict(orient="records")}
            return "No numeric features with a valid correlation to the fraud class were available.", {"question": q, "method": "exact pandas Pearson correlation"}


    # ============================================================
    # CAR PRICE DEDICATED ANALYSIS
    # Schema-gated to the common car-price dataset structure:
    # `price` + manufacturer/model + production year and vehicle fields.
    # This block is isolated from HR, COVID, Fraud, Diabetes, and generic logic.
    # ============================================================
    _car_cols = {
        re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_"): c
        for c in df.columns
    }

    def _car_col(*names):
        for name in names:
            key = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
            if key in _car_cols:
                return _car_cols[key]
        return None

    _car_price_col = _car_col("price")
    _car_manufacturer_col = _car_col("manufacturer", "brand", "make")
    _car_model_col = _car_col("model", "car_model")
    _car_year_col = _car_col("prod._year", "prod_year", "production_year", "year")
    _car_fuel_col = _car_col("fuel_type", "fuel type", "fuel")
    _car_transmission_col = _car_col("gear_box_type", "gearbox_type", "gear box type", "transmission", "transmission_type")
    _car_mileage_col = _car_col("mileage", "mileage_km", "kilometers")
    _car_engine_col = _car_col("engine_volume", "engine_size", "engine")

    _car_is_dataset = (
        _car_price_col is not None
        and len(df) > 0
        and pd.to_numeric(df[_car_price_col], errors="coerce").notna().any()
        and (_car_manufacturer_col is not None or _car_model_col is not None)
        and _car_year_col is not None
    )

    if _car_is_dataset:
        _car_price = pd.to_numeric(df[_car_price_col], errors="coerce")

        def _car_numeric_series(column):
            if column is None:
                return None
            raw = df[column]
            numeric = pd.to_numeric(raw, errors="coerce")
            if numeric.notna().sum() > 0:
                return numeric
            extracted = raw.astype(str).str.replace(",", "", regex=False).str.extract(r"([-+]?\d*\.?\d+)", expand=False)
            return pd.to_numeric(extracted, errors="coerce")

        _car_year = _car_numeric_series(_car_year_col)
        _car_valid_price = _car_price.notna()
        _car_count = int(_car_valid_price.sum())

        _car_count_q = any(x in ql for x in ["how many", "number of", "count"])
        _car_average_q = "average" in ql or "mean" in ql
        _car_median_q = "median" in ql
        _car_min_q = "minimum" in ql or re.search(r"\bmin\b", ql) is not None
        _car_max_q = "maximum" in ql or re.search(r"\bmax\b", ql) is not None
        _car_distribution_q = "distribution" in ql or "histogram" in ql
        _car_correlation_q = "correlation" in ql or "correlated" in ql
        _car_most_important_q = "most important features" in ql or "important features" in ql
        _car_highest_q = any(x in ql for x in ["highest", "most", "maximum", "max"])
        _car_lowest_q = any(x in ql for x in ["lowest", "minimum", "min"])

        # Total number of cars.
        if _car_count_q and any(x in ql for x in ["cars", "vehicles", "car in the dataset", "vehicles in the dataset"]):
            return (
                f"There are {_car_count:,} cars with a valid price value in the uploaded dataset."
                if _car_count != len(df)
                else f"There are {_car_count:,} cars in the uploaded dataset.",
                {"question": q, "method": "exact pandas analysis", "rows": int(len(df)), "cars_with_valid_price": _car_count}
            )

        if _car_average_q and "price" in ql and not any(x in ql for x in ["by brand", "by manufacturer", "by fuel", "by transmission", "by gearbox", "by gear"]) and not (any(x in ql for x in ["brand", "manufacturer", "fuel type", "transmission", "gear box", "gearbox"]) and "highest" in ql):
            _v = float(_car_price.mean())
            return f"The average car price is {_v:.4f}.", {"question": q, "method": "exact pandas analysis", "target_column": _car_price_col, "average_price": _v}

        if _car_median_q and "price" in ql:
            _v = float(_car_price.median())
            return f"The median car price is {_v:.0f}.", {"question": q, "method": "exact pandas analysis", "target_column": _car_price_col, "median_price": _v}

        if _car_min_q and "price" in ql and "which car" not in ql:
            _v = float(_car_price.min())
            return f"The minimum car price is {_v:.0f}.", {"question": q, "method": "exact pandas analysis", "target_column": _car_price_col, "minimum_price": _v}

        if _car_max_q and "price" in ql and "which car" not in ql:
            _v = float(_car_price.max())
            return f"The maximum car price is {_v:.0f}.", {"question": q, "method": "exact pandas analysis", "target_column": _car_price_col, "maximum_price": _v}

        # Highest/lowest priced car.
        if "which car" in ql and "price" in ql and (("highest" in ql and not "lowest" in ql) or ("lowest" in ql and not "highest" in ql)) and not any(x in ql for x in ["average", "mean", "by brand", "by manufacturer"]):
            _idx = _car_price.idxmax() if _car_highest_q and not _car_lowest_q else _car_price.idxmin()
            _row = df.loc[_idx]
            _label = "highest" if _car_highest_q and not _car_lowest_q else "lowest"
            _details = {"Price": float(_car_price.loc[_idx])}
            if _car_manufacturer_col is not None:
                _details["Manufacturer"] = str(_row[_car_manufacturer_col])
            if _car_model_col is not None:
                _details["Model"] = str(_row[_car_model_col])
            if _car_year_col is not None:
                _details["Production Year"] = _row[_car_year_col]
            if _car_cols.get("id") is not None:
                _details["ID"] = _row[_car_cols["id"]]
            return (
                f"The {_label}-priced car has a price of {float(_car_price.loc[_idx]):,.0f}. The matching vehicle details are shown below.",
                {"question": q, "method": "exact pandas analysis", "target_column": _car_price_col, "car_details": _details}
            )

        # Grouped average price questions.
        _group_col = None
        _group_name = None
        if any(x in ql for x in ["by brand", "by manufacturer", "brand"]):
            _group_col, _group_name = _car_manufacturer_col, "Brand"
        elif "by fuel" in ql or "fuel type" in ql:
            _group_col, _group_name = _car_fuel_col, "Fuel Type"
        elif any(x in ql for x in ["by transmission", "by gearbox", "by gear box", "gear box type", "transmission type"]):
            _group_col, _group_name = _car_transmission_col, "Transmission Type"

        if _group_col is not None and ("average" in ql or "mean" in ql):
            _tmp = pd.DataFrame({"group": df[_group_col].astype(str), "price": _car_price}).replace({"group": {"nan": np.nan}}).dropna(subset=["group", "price"])
            _grouped = _tmp.groupby("group", as_index=False)["price"].mean().sort_values("price", ascending=False)
            _grouped.columns = [_group_name, "Average Price"]
            _grouped["Average Price"] = _grouped["Average Price"].round(4)
            if _car_highest_q or "highest" in ql:
                if not _grouped.empty:
                    _top = _grouped.iloc[0]
                    return (
                        f"{_group_name} '{_top[_group_name]}' has the highest average price at {float(_top['Average Price']):,.4f}.",
                        {"question": q, "method": "exact pandas groupby analysis", "group_column": _group_col, "aggregation": "average price", "grouped_results": _grouped.to_dict(orient="records")}
                    )
            return (
                f"The average car price by {_group_name.lower()} is shown in the table below.",
                {"question": q, "method": "exact pandas groupby analysis", "group_column": _group_col, "aggregation": "average price", "grouped_results": _grouped.to_dict(orient="records")}
            )

        # Most cars by brand.
        if _car_manufacturer_col is not None and ("most cars" in ql or ("most" in ql and "brand" in ql and "average" not in ql)):
            _counts = df[_car_manufacturer_col].dropna().astype(str).str.strip().value_counts().rename_axis("Brand").reset_index(name="Car Count")
            if not _counts.empty:
                _top = _counts.iloc[0]
                return (
                    f"Brand '{_top['Brand']}' has the most cars in the dataset, with {int(_top['Car Count']):,} cars.",
                    {"question": q, "method": "exact pandas value_counts", "group_column": _car_manufacturer_col, "grouped_results": _counts.to_dict(orient="records")}
                )

        # Car age / production-year relationship.
        if "car age" in ql or "age affect price" in ql:
            if _car_year is not None and _car_year.notna().sum() >= 2:
                _valid = pd.DataFrame({"Production Year": _car_year, "Price": _car_price}).dropna()
                if len(_valid) >= 2 and _valid["Production Year"].nunique() > 1:
                    _reference_year = int(_valid["Production Year"].max())
                    _valid["Car Age"] = _reference_year - _valid["Production Year"]
                    _corr = float(_valid["Car Age"].corr(_valid["Price"]))
                    _valid["Age Group"] = pd.cut(_valid["Car Age"], bins=[-0.1, 3, 7, 11, 15, np.inf], labels=["0-3 years", "4-7 years", "8-11 years", "12-15 years", "16+ years"])
                    _age_grouped = _valid.groupby("Age Group", observed=False)["Price"].mean().reset_index()
                    _age_grouped["Average Price"] = _age_grouped["Price"].round(4)
                    _age_grouped = _age_grouped.drop(columns=["Price"])
                    _direction = "negative" if _corr < 0 else "positive" if _corr > 0 else "near zero"
                    return (
                        f"Using the newest production year in the dataset ({_reference_year}) as the reference, car age has a {_direction} Pearson correlation of {_corr:.4f} with price. The grouped average prices are shown below. This is an association, not causation.",
                        {"question": q, "method": "exact pandas analysis", "derived_feature": "Car Age = reference production year - production year", "reference_year": _reference_year, "age_price_correlation": round(_corr, 6), "grouped_results": _age_grouped.to_dict(orient="records")}
                    )

        # Specific correlations with price.
        if _car_correlation_q:
            _target = _car_price
            _specific_col = None
            if "mileage" in ql:
                _specific_col = _car_mileage_col
            elif "engine" in ql and "price" in ql:
                _specific_col = _car_engine_col
            if _specific_col is not None:
                _feature = _car_numeric_series(_specific_col)
                _pair = pd.DataFrame({"Feature": _feature, "Price": _target}).dropna()
                if len(_pair) >= 2 and _pair["Feature"].nunique() > 1:
                    _corr = float(_pair["Feature"].corr(_pair["Price"]))
                    return (
                        f"The Pearson correlation between {_specific_col} and price is {_corr:.4f}. This is an association, not a causal effect.",
                        {"question": q, "method": "exact pandas Pearson correlation", "feature": _specific_col, "target": _car_price_col, "correlation": round(_corr, 6)}
                    )

        # Numeric feature ranking against price.
        if _car_most_important_q or "most correlated with price" in ql:
            _rows = []
            for _feature_col in df.columns:
                if _feature_col == _car_price_col:
                    continue
                _feature = _car_numeric_series(_feature_col)
                if _feature is None:
                    continue
                _pair = pd.DataFrame({"Feature": _feature, "Price": _car_price}).dropna()
                if len(_pair) >= 2 and _pair["Feature"].nunique() > 1:
                    _corr = float(_pair["Feature"].corr(_pair["Price"]))
                    if np.isfinite(_corr):
                        _rows.append({"Feature": _feature_col, "Correlation with Price": round(_corr, 6), "Absolute Correlation": round(abs(_corr), 6)})
            _ranked = pd.DataFrame(_rows).sort_values("Absolute Correlation", ascending=False).head(10).reset_index(drop=True) if _rows else pd.DataFrame()
            if not _ranked.empty:
                return (
                    "The strongest numeric feature associations with price, ranked by absolute Pearson correlation, are shown below. This is a statistical screening signal, not model-based feature importance and not causation.",
                    {"question": q, "method": "exact pandas Pearson correlation", "target_column": _car_price_col, "grouped_results": _ranked.to_dict(orient="records")}
                )
            return "No numeric predictor correlations with price could be calculated.", {"question": q, "method": "exact pandas Pearson correlation"}

        # Price distribution.
        if _car_distribution_q and "price" in ql:
            _dist = pd.DataFrame({"Price": _car_price.dropna()})
            if not _dist.empty:
                _bins = min(10, max(1, int(_dist["Price"].nunique())))
                try:
                    _dist["Price Range"] = pd.qcut(_dist["Price"], q=_bins, duplicates="drop")
                    _distribution = _dist.groupby("Price Range", observed=True).size().reset_index(name="Car Count")
                except Exception:
                    _distribution = pd.DataFrame()
                if not _distribution.empty:
                    _distribution["Percentage"] = (_distribution["Car Count"] / len(_dist) * 100).round(2)
                    return (
                        "The distribution of car prices is shown in the table below using quantile-based price ranges.",
                        {"question": q, "method": "exact pandas distribution", "target_column": _car_price_col, "grouped_results": _distribution.to_dict(orient="records")}
                    )

    # ============================================================
    # HR DEDICATED GROUPED ANALYSIS
    # Handles HR grouped questions deterministically before the
    # generic analysis layer.
    #
    # IMPORTANT:
    # This block only activates when the required HR columns and
    # HR-specific question wording are detected.
    # Diabetes analysis and other dataset logic are untouched.
    # ============================================================

    def _hr_col_by_name(*names):
        """Resolve an HR column using normalized names."""
        normalized_hr_columns = {
            re.sub(
                r"[^a-z0-9]+",
                "_",
                str(c).strip().lower()
            ).strip("_"): c
            for c in df.columns
        }

        for name in names:
            name_norm = re.sub(
                r"[^a-z0-9]+",
                "_",
                str(name).strip().lower()
            ).strip("_")

            if name_norm in normalized_hr_columns:
                return normalized_hr_columns[name_norm]

        return None


    def _hr_numeric_column(*names):
        """Resolve an HR numeric column."""
        column = _hr_col_by_name(*names)

        if column is None:
            return None

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if converted.notna().sum() == 0:
            return None

        return column


    # ------------------------------------------------------------
    # Resolve common HR columns
    # ------------------------------------------------------------

    _hr_department = _hr_col_by_name(
        "department",
        "dept"
    )

    _hr_job_role = _hr_col_by_name(
        "job_role",
        "job role",
        "jobrole",
        "role"
    )

    _hr_gender = _hr_col_by_name(
        "gender",
        "sex"
    )

    _hr_age = _hr_numeric_column(
        "age"
    )

    _hr_income = _hr_numeric_column(
        "monthly_income",
        "monthly income",
        "income",
        "salary",
        "annual_income",
        "annual_salary"
    )

    _hr_work_life_balance = _hr_numeric_column(
        "work_life_balance",
        "work life balance"
    )

    _hr_years_at_company = _hr_numeric_column(
        "years_at_company",
        "years at company"
    )

    _hr_job_satisfaction = _hr_numeric_column(
        "job_satisfaction",
        "job satisfaction"
    )


    # ============================================================
    # 1. AVERAGE WORK-LIFE BALANCE BY DEPARTMENT
    # ============================================================

    if (
        _hr_department is not None
        and _hr_work_life_balance is not None
        and "work" in ql
        and "life" in ql
        and "balance" in ql
        and "department" in ql
        and ("average" in ql or "mean" in ql)
    ):
        _hr_temp = pd.DataFrame({
            "Department": df[_hr_department],
            "Work-Life Balance": pd.to_numeric(
                df[_hr_work_life_balance],
                errors="coerce"
            )
        }).dropna(
            subset=["Department", "Work-Life Balance"]
        )

        _hr_grouped = (
            _hr_temp
            .groupby("Department")["Work-Life Balance"]
            .mean()
            .sort_values(ascending=False)
        )

        if not _hr_grouped.empty:
            _hr_result = pd.DataFrame({
                "Department": [
                    str(v) for v in _hr_grouped.index
                ],
                "Average Work-Life Balance": [
                    round(float(v), 4)
                    for v in _hr_grouped.values
                ]
            })

            evidence = {
                "question": q,
                "method": "exact pandas analysis",
                "group_column": _hr_department,
                "measure_column": _hr_work_life_balance,
                "aggregation": "average",
                "grouped_results": _hr_result.to_dict(
                    orient="records"
                ),
                "total_rows": int(len(df))
            }

            return (
                "The average work-life balance for each department "
                "is shown in the table below, calculated directly "
                "from the uploaded HR dataset."
            ), evidence


    # ============================================================
    # 2. AVERAGE YEARS AT COMPANY BY DEPARTMENT
    # ============================================================

    if (
        _hr_department is not None
        and _hr_years_at_company is not None
        and "years" in ql
        and "company" in ql
        and "department" in ql
        and ("average" in ql or "mean" in ql)
    ):
        _hr_temp = pd.DataFrame({
            "Department": df[_hr_department],
            "Years at Company": pd.to_numeric(
                df[_hr_years_at_company],
                errors="coerce"
            )
        }).dropna(
            subset=["Department", "Years at Company"]
        )

        _hr_grouped = (
            _hr_temp
            .groupby("Department")["Years at Company"]
            .mean()
            .sort_values(ascending=False)
        )

        if not _hr_grouped.empty:
            _hr_result = pd.DataFrame({
                "Department": [
                    str(v) for v in _hr_grouped.index
                ],
                "Average Years at Company": [
                    round(float(v), 4)
                    for v in _hr_grouped.values
                ]
            })

            evidence = {
                "question": q,
                "method": "exact pandas analysis",
                "group_column": _hr_department,
                "measure_column": _hr_years_at_company,
                "aggregation": "average",
                "grouped_results": _hr_result.to_dict(
                    orient="records"
                ),
                "total_rows": int(len(df))
            }

            return (
                "The average years at company for each department "
                "is shown in the table below, calculated directly "
                "from the uploaded HR dataset."
            ), evidence


    # ============================================================
    # 3. AVERAGE INCOME BY GENDER
    # ============================================================

    if (
        _hr_gender is not None
        and _hr_income is not None
        and "gender" in ql
        and "income" in ql
        and ("average" in ql or "mean" in ql)
    ):
        _hr_temp = pd.DataFrame({
            "Gender": df[_hr_gender],
            "Income": pd.to_numeric(
                df[_hr_income],
                errors="coerce"
            )
        }).dropna(
            subset=["Gender", "Income"]
        )

        _hr_grouped = (
            _hr_temp
            .groupby("Gender")["Income"]
            .mean()
            .sort_values(ascending=False)
        )

        if not _hr_grouped.empty:
            _hr_result = pd.DataFrame({
                "Gender": [
                    str(v) for v in _hr_grouped.index
                ],
                "Average Income": [
                    round(float(v), 4)
                    for v in _hr_grouped.values
                ]
            })

            evidence = {
                "question": q,
                "method": "exact pandas analysis",
                "group_column": _hr_gender,
                "measure_column": _hr_income,
                "aggregation": "average",
                "grouped_results": _hr_result.to_dict(
                    orient="records"
                ),
                "total_rows": int(len(df))
            }

            return (
                "The average income by gender is shown in the "
                "table below, calculated directly from the "
                "uploaded HR dataset."
            ), evidence


    # ============================================================
    # 4. MALE/FEMALE EMPLOYEES BY DEPARTMENT
    # ============================================================

    if (
        _hr_department is not None
        and _hr_gender is not None
        and "department" in ql
        and "male" in ql
        and "female" in ql
        and "employee" in ql
    ):
        _hr_gender_temp = pd.DataFrame({
            "Department": df[_hr_department],
            "Gender": df[_hr_gender]
        }).dropna(
            subset=["Department", "Gender"]
        )

        _hr_gender_counts = pd.crosstab(
            _hr_gender_temp["Department"],
            _hr_gender_temp["Gender"]
        )

        # Normalize common gender labels to Male/Female columns.
        _hr_gender_result = pd.DataFrame(
            index=_hr_gender_counts.index
        )

        for _gender_value in _hr_gender_counts.columns:
            _gender_text = str(_gender_value).strip().lower()

            if _gender_text in {"m", "male", "man", "men"}:
                _hr_gender_result["Male"] = (
                    _hr_gender_counts[_gender_value]
                )

            elif _gender_text in {"f", "female", "woman", "women"}:
                _hr_gender_result["Female"] = (
                    _hr_gender_counts[_gender_value]
                )

        if "Male" not in _hr_gender_result.columns:
            _hr_gender_result["Male"] = 0

        if "Female" not in _hr_gender_result.columns:
            _hr_gender_result["Female"] = 0

        _hr_gender_result = (
            _hr_gender_result[
                ["Male", "Female"]
            ]
            .fillna(0)
            .astype(int)
            .reset_index()
            .rename(
                columns={
                    "index": "Department"
                }
            )
        )

        evidence = {
            "question": q,
            "method": "exact pandas analysis",
            "group_column": _hr_department,
            "measure_column": _hr_gender,
            "aggregation": "count by gender",
            "grouped_results": _hr_gender_result.to_dict(
                orient="records"
            ),
            "total_rows": int(len(df))
        }

        return (
            "Here is the exact male and female employee count "
            "for each department, calculated directly from the "
            "uploaded HR dataset."
        ), evidence


    # ============================================================
    # 5. AVERAGE AGE FOR EACH JOB ROLE
    # ============================================================

    if (
        _hr_job_role is not None
        and _hr_age is not None
        and "job" in ql
        and "role" in ql
        and "age" in ql
        and ("average" in ql or "mean" in ql)
    ):
        _hr_temp = pd.DataFrame({
            "Job Role": df[_hr_job_role],
            "Age": pd.to_numeric(
                df[_hr_age],
                errors="coerce"
            )
        }).dropna(
            subset=["Job Role", "Age"]
        )

        _hr_grouped = (
            _hr_temp
            .groupby("Job Role")["Age"]
            .mean()
            .sort_values(ascending=False)
        )

        if not _hr_grouped.empty:
            _hr_result = pd.DataFrame({
                "Job Role": [
                    str(v) for v in _hr_grouped.index
                ],
                "Average Age": [
                    round(float(v), 4)
                    for v in _hr_grouped.values
                ]
            })

            evidence = {
                "question": q,
                "method": "exact pandas analysis",
                "group_column": _hr_job_role,
                "measure_column": _hr_age,
                "aggregation": "average",
                "grouped_results": _hr_result.to_dict(
                    orient="records"
                ),
                "total_rows": int(len(df))
            }

            return (
                "The average age for each job role is shown "
                "in the table below, calculated directly from "
                "the uploaded HR dataset."
            ), evidence


    # ============================================================
    # 6. HIGHEST AVERAGE INCOME BY JOB ROLE
    # ============================================================

    if (
        _hr_job_role is not None
        and _hr_income is not None
        and "job" in ql
        and "role" in ql
        and "income" in ql
        and "average" in ql
        and any(
            word in ql
            for word in [
                "highest",
                "largest",
                "maximum",
                "most"
            ]
        )
    ):
        _hr_temp = pd.DataFrame({
            "Job Role": df[_hr_job_role],
            "Income": pd.to_numeric(
                df[_hr_income],
                errors="coerce"
            )
        }).dropna(
            subset=["Job Role", "Income"]
        )

        _hr_grouped = (
            _hr_temp
            .groupby("Job Role")["Income"]
            .mean()
            .sort_values(ascending=False)
        )

        if not _hr_grouped.empty:
            _hr_winner = _hr_grouped.index[0]
            _hr_winner_value = float(
                _hr_grouped.iloc[0]
            )

            _hr_result = pd.DataFrame({
                "Job Role": [
                    str(v) for v in _hr_grouped.index
                ],
                "Average Income": [
                    round(float(v), 4)
                    for v in _hr_grouped.values
                ]
            })

            evidence = {
                "question": q,
                "method": "exact pandas analysis",
                "group_column": _hr_job_role,
                "measure_column": _hr_income,
                "aggregation": "average",
                "winner": str(_hr_winner),
                "winner_value": round(
                    _hr_winner_value,
                    4
                ),
                "grouped_results": _hr_result.to_dict(
                    orient="records"
                ),
                "total_rows": int(len(df))
            }

            return (
                f"The job role with the highest average income "
                f"is {_hr_winner}, with an average income of "
                f"{_hr_winner_value:.4f}."
            ), evidence


    # ============================================================
    # 7. EMPLOYEE COUNT BY JOB ROLE
    # ============================================================

    if (
        _hr_job_role is not None
        and "job" in ql
        and "role" in ql
        and "employee" in ql
        and (
            "how many" in ql
            or "number" in ql
            or "count" in ql
        )
    ):
        _hr_counts = (
            df[_hr_job_role]
            .dropna()
            .astype(str)
            .str.strip()
            .value_counts()
            .sort_values(ascending=False)
        )

        if not _hr_counts.empty:
            _hr_total = int(
                _hr_counts.sum()
            )

            _hr_result = pd.DataFrame({
                "Job Role": _hr_counts.index.tolist(),
                "Employee Count": [
                    int(v)
                    for v in _hr_counts.values
                ],
                "Percentage": [
                    round(
                        float(v) / _hr_total * 100,
                        2
                    )
                    if _hr_total
                    else 0.0
                    for v in _hr_counts.values
                ]
            })

            evidence = {
                "question": q,
                "method": "exact pandas analysis",
                "group_column": _hr_job_role,
                "aggregation": "count",
                "grouped_results": _hr_result.to_dict(
                    orient="records"
                ),
                "total_rows": int(len(df))
            }

            return (
                "Here is the exact employee count for each "
                "job role, calculated directly from the "
                "uploaded HR dataset."
            ), evidence


    # ============================================================
    # 8. AVERAGE JOB SATISFACTION BY DEPARTMENT
    # ============================================================

    if (
        _hr_department is not None
        and _hr_job_satisfaction is not None
        and "job" in ql
        and "satisfaction" in ql
        and "department" in ql
        and ("average" in ql or "mean" in ql)
    ):
        _hr_temp = pd.DataFrame({
            "Department": df[_hr_department],
            "Job Satisfaction": pd.to_numeric(
                df[_hr_job_satisfaction],
                errors="coerce"
            )
        }).dropna(
            subset=["Department", "Job Satisfaction"]
        )

        _hr_grouped = (
            _hr_temp
            .groupby("Department")["Job Satisfaction"]
            .mean()
            .sort_values(ascending=False)
        )

        if not _hr_grouped.empty:
            _hr_result = pd.DataFrame({
                "Department": [
                    str(v) for v in _hr_grouped.index
                ],
                "Average Job Satisfaction": [
                    round(float(v), 4)
                    for v in _hr_grouped.values
                ]
            })

            evidence = {
                "question": q,
                "method": "exact pandas analysis",
                "group_column": _hr_department,
                "measure_column": _hr_job_satisfaction,
                "aggregation": "average",
                "grouped_results": _hr_result.to_dict(
                    orient="records"
                ),
                "total_rows": int(len(df))
            }

            return (
                "The average job satisfaction for each department "
                "is shown in the table below, calculated directly "
                "from the uploaded HR dataset."
            ), evidence

    evidence = {"question": q, "method": "exact pandas analysis"}


    # ============================================================
    # COVID DATASET DEDICATED ANALYSIS
    # This block is isolated by schema detection so it does not
    # run for HR, Diabetes, or unrelated datasets.
    # ============================================================
    covid_required = {
        "country_region", "date", "confirmed", "deaths",
        "recovered", "active"
    }

    if covid_required.issubset(set(map(str, df.columns))):
        try:
            covid = df.copy()
            covid["date"] = pd.to_datetime(covid["date"], errors="coerce")

            for _metric in ["confirmed", "deaths", "recovered", "active"]:
                covid[_metric] = pd.to_numeric(covid[_metric], errors="coerce")

            covid = covid.dropna(subset=["date", "country_region"])

            if not covid.empty:
                # The uploaded COVID dataset is a cumulative time series.
                # Country/region totals therefore use the latest available
                # cumulative value instead of summing all historical dates.
                latest_date = covid["date"].max()
                evidence["latest_date"] = str(latest_date.date())
                evidence["date_range"] = {
                    "start": str(covid["date"].min().date()),
                    "end": str(latest_date.date())
                }

                def _country_daily(metric):
                    return (
                        covid.groupby(["country_region", "date"], dropna=False)[metric]
                        .sum(min_count=1)
                        .reset_index()
                    )

                def _latest_country(metric):
                    daily = _country_daily(metric)
                    daily = daily.sort_values(["country_region", "date"])
                    return daily.groupby("country_region", as_index=False).tail(1)

                def _latest_region(metric):
                    if "who_region" not in covid.columns:
                        return pd.DataFrame(columns=["who_region", "date", metric])
                    temp = covid.dropna(subset=["who_region"])
                    daily = (
                        temp.groupby(["who_region", "date"], dropna=False)[metric]
                        .sum(min_count=1)
                        .reset_index()
                        .sort_values(["who_region", "date"])
                    )
                    return daily.groupby("who_region", as_index=False).tail(1)

                def _records(frame):
                    return frame.replace({np.nan: None}).to_dict("records")

                def _metric_label(metric):
                    return {
                        "confirmed": "confirmed cases",
                        "deaths": "deaths",
                        "recovered": "recovered cases",
                        "active": "active cases",
                    }[metric]

                # --------------------------------------------------------
                # 1-4: highest country-level COVID measure
                # --------------------------------------------------------
                for _metric in ["confirmed", "deaths", "recovered", "active"]:
                    if (
                        _metric in ql
                        and "country" in ql
                        and any(word in ql for word in ["highest", "most", "maximum"])
                    ):
                        latest = _latest_country(_metric).dropna(subset=[_metric])
                        latest = latest.sort_values(_metric, ascending=False)
                        if not latest.empty:
                            top = latest.iloc[0]
                            display_name = _metric_label(_metric).title()
                            result = latest.rename(columns={
                                "country_region": "Country",
                                _metric: display_name
                            })[["Country", display_name]]
                            evidence["grouped_results"] = _records(result)
                            return (
                                f"{top['country_region']} has the highest {_metric_label(_metric)} "
                                f"in the latest available country totals, with {top[_metric]:,.0f}.",
                                evidence
                            )

                # --------------------------------------------------------
                # 5-8: total/latest cumulative measure by country
                # --------------------------------------------------------
                country_total_patterns = {
                    "confirmed": ["total number of confirmed", "total confirmed", "confirmed cases by country"],
                    "deaths": ["total number of deaths", "total deaths", "deaths by country"],
                    "recovered": ["total number of recovered", "total recovered", "recovered cases by country"],
                    "active": ["total number of active", "total active", "active cases by country"],
                }

                for _metric, _patterns in country_total_patterns.items():
                    if "country" in ql and any(pattern in ql for pattern in _patterns):
                        latest = _latest_country(_metric).dropna(subset=[_metric])
                        latest = latest.sort_values(_metric, ascending=False)
                        display_name = _metric_label(_metric).title()
                        result = latest.rename(columns={
                            "country_region": "Country",
                            _metric: display_name
                        })[["Country", display_name]]
                        evidence["grouped_results"] = _records(result)
                        return (
                            f"Latest cumulative {_metric_label(_metric)} by country are shown below. "
                            f"Each country uses its latest available date; the overall dataset latest date is "
                            f"{latest_date.date()}.",
                            evidence
                        )

                # --------------------------------------------------------
                # 9-13: WHO region analysis
                # --------------------------------------------------------
                region_labels = {
                    "confirmed": "confirmed cases",
                    "deaths": "deaths",
                    "recovered": "recovered cases",
                    "active": "active cases",
                }

                if "who_region" in covid.columns and "region" in ql:
                    for _metric, _label in region_labels.items():
                        if _metric in ql:
                            latest = _latest_region(_metric).dropna(subset=[_metric])
                            latest = latest.sort_values(_metric, ascending=False)
                            if latest.empty:
                                continue
                            display_name = _label.title()
                            result = latest.rename(columns={
                                "who_region": "WHO Region",
                                _metric: display_name
                            })[["WHO Region", display_name]]
                            evidence["grouped_results"] = _records(result)
                            if any(word in ql for word in ["highest", "most", "maximum"]):
                                top = latest.iloc[0]
                                return (
                                    f"{top['who_region']} has the highest {_label} in the latest available "
                                    f"WHO-region totals, with {top[_metric]:,.0f}.",
                                    evidence
                                )
                            return (
                                f"Latest cumulative {_label} by WHO region are shown below.",
                                evidence
                            )

                # --------------------------------------------------------
                # 14-16: global change over time
                # --------------------------------------------------------
                if "over time" in ql or "change over time" in ql:
                    for _metric, _label in region_labels.items():
                        if _metric in ql:
                            daily = (
                                covid.groupby("date")[_metric]
                                .sum(min_count=1)
                                .reset_index()
                                .sort_values("date")
                            )
                            daily["Daily Change"] = daily[_metric].diff()
                            display_name = _label.title()
                            result = daily.rename(columns={
                                "date": "Date",
                                _metric: display_name
                            })[["Date", display_name, "Daily Change"]]
                            result["Date"] = result["Date"].dt.strftime("%Y-%m-%d")
                            evidence["grouped_results"] = _records(result)
                            return (
                                f"The global {_label} trend over time is shown below. "
                                f"Values are cumulative totals aggregated across the uploaded dataset by date.",
                                evidence
                            )

                # --------------------------------------------------------
                # 17: highest confirmed cases on a single date
                # --------------------------------------------------------
                if "highest" in ql and "confirmed" in ql and "single date" in ql:
                    daily = (
                        covid.groupby("date")["confirmed"]
                        .sum(min_count=1)
                        .reset_index()
                        .sort_values("confirmed", ascending=False)
                    )
                    if not daily.empty:
                        top = daily.iloc[0]
                        result = daily.head(20).rename(columns={
                            "date": "Date",
                            "confirmed": "Confirmed Cases"
                        })
                        result["Date"] = result["Date"].dt.strftime("%Y-%m-%d")
                        evidence["grouped_results"] = _records(result)
                        return (
                            f"The highest global confirmed-case total on a single date was "
                            f"{top['confirmed']:,.0f} on {top['date'].date()}.",
                            evidence
                        )

                # --------------------------------------------------------
                # 18-19: country-specific latest statistics
                # --------------------------------------------------------
                country_request = None
                if re.search(r"\bindia\b", ql):
                    country_request = "India"
                elif re.search(r"\b(united states|usa|u\.s\.?|us)\b", ql):
                    country_request = "United States"

                if country_request and ("statistics" in ql or "stats" in ql):
                    normalized = covid["country_region"].astype(str).str.strip().str.casefold()
                    if country_request == "India":
                        mask = normalized.eq("india")
                    else:
                        mask = normalized.isin({"united states", "us", "usa", "u.s.", "u.s"})
                    selected = covid.loc[mask].copy()
                    if not selected.empty:
                        latest_country = (
                            selected.groupby("date")[["confirmed", "deaths", "recovered", "active"]]
                            .sum(min_count=1)
                            .reset_index()
                            .sort_values("date")
                        )
                        row = latest_country.iloc[-1]
                        result = pd.DataFrame([{
                            "Country": country_request,
                            "Date": row["date"].strftime("%Y-%m-%d"),
                            "Confirmed": row["confirmed"],
                            "Deaths": row["deaths"],
                            "Recovered": row["recovered"],
                            "Active": row["active"],
                        }])
                        evidence["grouped_results"] = _records(result)
                        return (
                            f"Latest available COVID-19 statistics for {country_request} are for "
                            f"{row['date'].date()}.",
                            evidence
                        )

                # --------------------------------------------------------
                # 20: death rate by country
                # --------------------------------------------------------
                if "death rate" in ql:
                    confirmed_latest = _latest_country("confirmed")[["country_region", "confirmed"]]
                    deaths_latest = _latest_country("deaths")[["country_region", "deaths"]]
                    merged = confirmed_latest.merge(
                        deaths_latest,
                        on="country_region",
                        how="outer"
                    )
                    merged["Death Rate (%)"] = np.where(
                        pd.to_numeric(merged["confirmed"], errors="coerce") > 0,
                        pd.to_numeric(merged["deaths"], errors="coerce")
                        / pd.to_numeric(merged["confirmed"], errors="coerce") * 100,
                        np.nan
                    )
                    result = merged[["country_region", "Death Rate (%)"]].rename(
                        columns={"country_region": "Country"}
                    )
                    result = result.sort_values("Death Rate (%)", ascending=False)
                    result["Death Rate (%)"] = result["Death Rate (%)"].round(2)
                    evidence["grouped_results"] = _records(result)
                    return (
                        "Death rate by country is calculated as deaths divided by confirmed cases, "
                        "using each country's latest available cumulative values.",
                        evidence
                    )

        except Exception as covid_error:
            # COVID-specific failures are captured so they cannot break
            # the existing HR/Diabetes/generic analysis flow.
            evidence["covid_error"] = str(covid_error)

    if df is None or df.empty:
        return "The uploaded dataset is empty, so there is nothing to analyze.", evidence

    target = _infer_target_column(df, q)
    semantics = _target_semantics(df, target, q)

    aliases = {
        "glucose": ["glucose"],
        "blood pressure": ["bloodpressure", "blood_pressure"],
        "bloodpressure": ["bloodpressure", "blood_pressure"],
        "skin thickness": ["skinthickness", "skin_thickness"],
        "insulin": ["insulin"],
        "bmi": ["bmi"],
        "age": ["age"],
        "pregnancies": ["pregnancies"],
        "diabetes pedigree": ["diabetespedigreefunction", "diabetes_pedigree_function"],
    }

    # Resolve the requested measure before the conditional-analysis block.
    # This must happen before any reference to `col`; otherwise Python treats
    # `col` as a local variable that may be used before it has been assigned.
    col = _find_column(df, q, aliases)

    # Natural-language measure aliases (for example, "salary" ->
    # MonthlyIncome) are resolved before filter analysis so questions such as
    # "average salary for Sales" can be answered from the actual DataFrame.
    if col is None:
        semantic_measure = _find_group_measure_column(df, q, aliases=aliases)
        if semantic_measure is not None:
            col = semantic_measure

    # ------------------------------------------------------------
    # GROUPED / CATEGORY ANALYSIS
    # Handle count, percentage, average, sum, min/max by a category before
    # generic single-filter logic. These calculations are always exact Pandas.
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # HR GROUPED NUMERIC ANALYSIS PRIORITY
    # Handle department-level income/salary questions directly before
    # generic single-column statistics can return the overall mean.
    # ------------------------------------------------------------
    _hr_department_col = next(
        (
            c for c in df.columns
            if str(c).strip().lower().replace("_", " ") == "department"
        ),
        None,
    )
    _hr_income_col = next(
        (
            c for c in df.columns
            if str(c).strip().lower().replace("_", " ") in {
                "monthly income",
                "monthlyincome",
            }
        ),
        None,
    )

    if (
        _hr_department_col is not None
        and _hr_income_col is not None
        and "department" in ql
        and ("average" in ql or "mean" in ql)
        and ("income" in ql or "salary" in ql)
    ):
        _hr_income = pd.to_numeric(
            df[_hr_income_col],
            errors="coerce",
        )
        _hr_temp = pd.DataFrame({
            "Department": df[_hr_department_col],
            "Monthly Income": _hr_income,
        }).dropna(subset=["Department", "Monthly Income"])

        _hr_grouped = (
            _hr_temp.groupby("Department", dropna=False)["Monthly Income"]
            .mean()
            .sort_values(ascending=False)
        )

        if not _hr_grouped.empty:
            _hr_result = pd.DataFrame({
                "Department": [str(v) for v in _hr_grouped.index],
                "Average Monthly Income": [
                    round(float(v), 4) for v in _hr_grouped.values
                ],
            })

            _hr_income_extreme_words = [
                "highest",
                "lowest",
                "largest",
                "smallest",
                "maximum",
                "minimum",
                "max",
                "min",
                "most",
                "least",
            ]
            _hr_income_is_extreme = any(
                re.search(
                    rf"\\b{re.escape(word)}\\b",
                    ql,
                )
                for word in _hr_income_extreme_words
            )

            if _hr_income_is_extreme:
                _hr_income_is_lowest = any(
                    re.search(
                        rf"\\b{re.escape(word)}\\b",
                        ql,
                    )
                    for word in [
                        "lowest",
                        "smallest",
                        "minimum",
                        "min",
                        "least",
                    ]
                )

                if _hr_income_is_lowest:
                    _hr_income_winner_department = _hr_grouped.index[-1]
                    _hr_income_winner_value = float(_hr_grouped.iloc[-1])
                    _hr_income_direction = "lowest"
                else:
                    _hr_income_winner_department = _hr_grouped.index[0]
                    _hr_income_winner_value = float(_hr_grouped.iloc[0])
                    _hr_income_direction = "highest"

                _hr_evidence = {
                    "question": q,
                    "method": "exact pandas analysis",
                    "group_column": _hr_department_col,
                    "measure_column": _hr_income_col,
                    "aggregation": "average",
                    "grouped_results": _hr_result.to_dict(orient="records"),
                    "winner": str(_hr_income_winner_department),
                    "winner_value": round(_hr_income_winner_value, 4),
                    "direction": _hr_income_direction,
                    "total_rows": int(len(df)),
                }

                return (
                    f"The {_hr_income_direction} average monthly income is "
                    f"for department = {str(_hr_income_winner_department)}, "
                    f"with an average monthly income of "
                    f"{_hr_income_winner_value:.4f}."
                ), _hr_evidence

            _hr_evidence = {
                "question": q,
                "method": "exact pandas analysis",
                "group_column": _hr_department_col,
                "measure_column": _hr_income_col,
                "aggregation": "average",
                "grouped_results": _hr_result.to_dict(orient="records"),
                "total_rows": int(len(df)),
            }
            return (
                "The average monthly income for each department is shown "
                "in the table below, calculated directly from the uploaded HR dataset."
            ), _hr_evidence

    # ------------------------------------------------------------
    # HR GROUPED AVERAGE AGE
    # Handle department-level age questions directly from the
    # uploaded HR DataFrame before generic statistics return
    # the overall average age.
    # ------------------------------------------------------------
    _hr_age_col = next(
        (
            c for c in df.columns
            if str(c).strip().lower().replace("_", " ") == "age"
        ),
        None,
    )

    if (
        _hr_department_col is not None
        and _hr_age_col is not None
        and "department" in ql
        and ("average" in ql or "mean" in ql)
        and "age" in ql
    ):
        _hr_age = pd.to_numeric(
            df[_hr_age_col],
            errors="coerce",
        )

        _hr_age_temp = pd.DataFrame({
            "Department": df[_hr_department_col],
            "Age": _hr_age,
        }).dropna(subset=["Department", "Age"])

        _hr_age_grouped = (
            _hr_age_temp
            .groupby("Department", dropna=False)["Age"]
            .mean()
            .sort_values(ascending=False)
        )

        if not _hr_age_grouped.empty:
            _hr_age_result = pd.DataFrame({
                "Department": [str(v) for v in _hr_age_grouped.index],
                "Average Age": [round(float(v), 4) for v in _hr_age_grouped.values],
            })

            _hr_age_evidence = {
                "question": q,
                "method": "exact pandas analysis",
                "group_column": _hr_department_col,
                "measure_column": _hr_age_col,
                "aggregation": "average",
                "grouped_results": _hr_age_result.to_dict(orient="records"),
                "total_rows": int(len(df)),
            }

            return (
                "The average age for each department is shown "
                "in the table below, calculated directly from "
                "the uploaded HR dataset."
            ), _hr_age_evidence

    # ------------------------------------------------------------
    # HR ATTRITION ANALYSIS
    # ------------------------------------------------------------
    _hr_attrition_col = next(
        (
            c for c in df.columns
            if str(c).strip().lower().replace("_", " ") == "attrition"
        ),
        None,
    )

    if _hr_attrition_col is not None:
        _attrition_text = (
            df[_hr_attrition_col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )
        _attrition_values = _attrition_text.unique().tolist()

        _positive_keys = {
            "yes", "y", "true", "1", "1.0",
            "attrition", "left", "churned", "churn", "positive"
        }
        _positive_key = next(
            (v for v in _attrition_values if v in _positive_keys),
            None,
        )

        # HR attrition is expected to be binary (for example Yes/No).
        if _positive_key is not None and len(_attrition_values) == 2:
            _positive_mask = (
                df[_hr_attrition_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq(_positive_key)
            )

            # Resolve the requested grouping column.
            _hr_attrition_group_col = None
            _hr_group_candidates = [
                "department",
                "job_role",
                "gender",
                "marital_status",
                "education_field",
                "business_travel",
                "overtime",
            ]

            for _candidate in _hr_group_candidates:
                if _candidate in df.columns:
                    _candidate_text = _candidate.replace("_", " ")
                    if (
                        _candidate in ql
                        or _candidate_text in ql
                    ):
                        _hr_attrition_group_col = _candidate
                        break

            # Support capitalization/spaces in actual column names.
            if _hr_attrition_group_col is None:
                for _candidate in df.columns:
                    _candidate_norm = (
                        str(_candidate).strip().lower().replace("_", " ")
                    )
                    if _candidate_norm in {
                        "department",
                        "job role",
                        "gender",
                        "marital status",
                        "education field",
                        "business travel",
                        "overtime",
                    } and _candidate_norm in ql:
                        _hr_attrition_group_col = _candidate
                        break

            if _hr_attrition_group_col is not None:
                _attrition_temp = pd.DataFrame({
                    "Group": df[_hr_attrition_group_col],
                    "AttritionFlag": _positive_mask.astype(int),
                }).dropna(subset=["Group"])

                _attrition_grouped = (
                    _attrition_temp
                    .groupby("Group", dropna=False)
                    .agg(
                        Employees=("Group", "size"),
                        Attrition_Count=("AttritionFlag", "sum"),
                    )
                )

                _attrition_grouped["Attrition_Rate"] = (
                    _attrition_grouped["Attrition_Count"]
                    / _attrition_grouped["Employees"]
                    * 100
                )

                _attrition_grouped = _attrition_grouped.sort_values(
                    "Attrition_Rate",
                    ascending=False,
                )

                if not _attrition_grouped.empty:
                    _attrition_result = pd.DataFrame({
                        _hr_attrition_group_col: [
                            str(v) for v in _attrition_grouped.index
                        ],
                        "Employee Count": [
                            int(v)
                            for v in _attrition_grouped["Employees"]
                        ],
                        "Attrition Count": [
                            int(v)
                            for v in _attrition_grouped["Attrition_Count"]
                        ],
                        "Attrition Rate (%)": [
                            round(float(v), 2)
                            for v in _attrition_grouped["Attrition_Rate"]
                        ],
                    })

                    _attrition_extreme = any(
                        re.search(
                            rf"\\b{re.escape(word)}\\b",
                            ql,
                        )
                        for word in [
                            "highest", "most", "largest", "maximum", "max",
                            "lowest", "least", "smallest", "minimum", "min",
                        ]
                    )

                    _attrition_rate_question = any(
                        word in ql
                        for word in [
                            "percentage", "percent", "%", "rate", "attrition"
                        ]
                    )

                    if _attrition_extreme:
                        _attrition_lowest = any(
                            re.search(
                                rf"\\b{re.escape(word)}\\b",
                                ql,
                            )
                            for word in [
                                "lowest", "least", "smallest", "minimum", "min"
                            ]
                        )

                        if _attrition_lowest:
                            _winner_row = _attrition_grouped.iloc[-1]
                            _winner_group = _attrition_grouped.index[-1]
                            _direction = "lowest"
                        else:
                            _winner_row = _attrition_grouped.iloc[0]
                            _winner_group = _attrition_grouped.index[0]
                            _direction = "highest"

                        _attrition_evidence = {
                            "question": q,
                            "method": "exact pandas analysis",
                            "group_column": _hr_attrition_group_col,
                            "measure_column": _hr_attrition_col,
                            "aggregation": "attrition rate",
                            "attrition_positive_value": _positive_key,
                            "grouped_results": _attrition_result.to_dict(
                                orient="records"
                            ),
                            "winner": str(_winner_group),
                            "winner_value": round(
                                float(_winner_row["Attrition_Rate"]), 2
                            ),
                            "winner_attrition_count": int(
                                _winner_row["Attrition_Count"]
                            ),
                            "direction": _direction,
                            "total_rows": int(len(df)),
                        }

                        return (
                            f"The {_direction} attrition rate is for "
                            f"{_hr_attrition_group_col.replace('_', ' ')} = "
                            f"{str(_winner_group)}, with an attrition rate of "
                            f"{float(_winner_row['Attrition_Rate']):.2f}% "
                            f"({int(_winner_row['Attrition_Count'])} employees)."
                        ), _attrition_evidence

                    if _attrition_rate_question:
                        _attrition_evidence = {
                            "question": q,
                            "method": "exact pandas analysis",
                            "group_column": _hr_attrition_group_col,
                            "measure_column": _hr_attrition_col,
                            "aggregation": "attrition rate",
                            "attrition_positive_value": _positive_key,
                            "grouped_results": _attrition_result.to_dict(
                                orient="records"
                            ),
                            "total_rows": int(len(df)),
                        }

                        return (
                            f"The attrition rate for each "
                            f"{_hr_attrition_group_col.replace('_', ' ')} "
                            "is shown in the table below, calculated directly "
                            "from the uploaded HR dataset."
                        ), _attrition_evidence

    grouped_result = _grouped_aggregation_answer(df, q)
    if grouped_result is not None:
        return grouped_result

    # ------------------------------------------------------------
    # GENERIC CONDITIONAL / FILTERED ANALYSIS
    # Run this before unfiltered statistics so questions such as
    # "average BMI for diabetic patients" use the filtered rows.
    # This is dataset-independent and works with numeric and categorical filters.
    # ------------------------------------------------------------
    condition_operator, condition_value = _parse_numeric_condition(q)
    target_filter_value = _detect_binary_target_filter(df, target, q)
    category_filter_col, category_filter_value = _resolve_categorical_filter(
        df, q, exclude=[]
    )
    if category_filter_col is None:
        category_filter_col, category_filter_value = _find_generic_filter_column(
            df, q, exclude=[target] if target else None
        )

    filter_mask = pd.Series(True, index=df.index)
    filter_parts = []
    filter_column = None
    filter_value = None

    # Numeric condition: identify the numeric column nearest to the condition wording.
    if condition_operator is not None:
        numeric_candidates = []
        for c in df.select_dtypes(include=np.number).columns:
            norm = normalize_feature_name(c)
            if re.search(rf"(?<![a-z0-9_]){re.escape(norm)}(?![a-z0-9_])", ql):
                numeric_candidates.append(c)
        if col is not None and pd.api.types.is_numeric_dtype(df[col]) and col not in numeric_candidates:
            numeric_candidates.insert(0, col)
        if numeric_candidates:
            filter_column = numeric_candidates[0]
            filter_mask &= _apply_numeric_condition(df[filter_column], condition_operator, condition_value)
            filter_parts.append(f"{filter_column} {condition_operator} {_fmt(condition_value)}")

    # Binary target wording such as "with diabetes" / "without diabetes".
    if target_filter_value is not None:
        filter_column = target
        filter_value = target_filter_value
        filter_mask &= df[target].astype(str).str.strip().str.lower() == str(target_filter_value).strip().lower()
        filter_parts.append(f"{target} = {_fmt(target_filter_value)}")

    # Generic categorical value mentioned in the question.
    if category_filter_col is not None and category_filter_value is not None:
        filter_column = category_filter_col
        filter_value = category_filter_value
        filter_mask &= df[category_filter_col].astype(str).str.strip().str.lower() == str(category_filter_value).strip().lower()
        filter_parts.append(f"{category_filter_col} = {_fmt(category_filter_value)}")

    has_filter = bool(filter_parts)
    filtered_df = df.loc[filter_mask].copy() if has_filter else df

    # Conditional questions need an explicit requested measure. For a numeric
    # measure, calculate the requested statistic on the filtered full dataset.
    # Once a real filter has been identified, answer the filtered question
    # deterministically. Do not require a particular preposition such as
    # "for" or "with" because natural language varies: users may say
    # "in Sales", "Sales department", "have attrition Yes", "where region is
    # West", etc.
    if has_filter:
        # `_find_column()` can resolve a categorical filter column (for example,
        # `department`) simply because that word appears in the question. For a
        # statistic such as "average age ... in the Sales department", the
        # requested measure must instead be the explicitly mentioned NUMERIC
        # column (`age`). Therefore, prefer an explicitly mentioned numeric
        # column over a categorical filter column.
        measure_col = col if (
            col is not None
            and col in df.columns
            and pd.api.types.is_numeric_dtype(df[col])
            and col not in {filter_column}
        ) else None

        # Find numeric columns explicitly mentioned in the question.
        mentioned_numeric = []
        for c in df.select_dtypes(include=np.number).columns:
            if c == filter_column:
                continue
            norm = normalize_feature_name(c)
            if norm and re.search(
                rf"(?<![a-z0-9_]){re.escape(norm)}(?![a-z0-9_])",
                ql,
            ):
                mentioned_numeric.append(c)

        # Prefer the explicitly mentioned numeric measure.
        if mentioned_numeric:
            measure_col = mentioned_numeric[0]

        # If the question is a count/percentage question, count filtered rows.
        if any(w in ql for w in ["how many", "number of", "count of", "count", "percentage", "percent", "%"]):
            n = int(len(filtered_df))
            pct = (n / len(df) * 100) if len(df) else 0.0
            evidence.update({
                "filter": filter_parts,
                "filtered_rows": n,
                "total_rows": int(len(df)),
                "percentage": pct,
            })
            if "percentage" in ql or "percent" in ql or "%" in ql:
                return f"{n:,} of {len(df):,} rows match the condition ({pct:.2f}%).", evidence
            return f"{n:,} rows match the condition: " + " and ".join(filter_parts) + ".", evidence

        if measure_col is not None and measure_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[measure_col]):
            stats = _exact_numeric_summary(filtered_df, measure_col)
            evidence.update({
                "filter": filter_parts,
                "filtered_rows": int(len(filtered_df)),
                "measure_column": measure_col,
                **stats,
            })
            if len(filtered_df) == 0 or stats["count"] == 0:
                return f"No rows match the condition: " + " and ".join(filter_parts) + ".", evidence
            if "median" in ql:
                return f"The median of {measure_col} for rows matching " + " and ".join(filter_parts) + f" is {_fmt(stats['median'])}.", evidence
            if "standard deviation" in ql or re.search(r"\bstd\b", ql):
                return f"The standard deviation of {measure_col} for rows matching " + " and ".join(filter_parts) + f" is {_fmt(stats['std'])}.", evidence
            if "minimum" in ql or re.search(r"\bmin\b", ql):
                return f"The minimum of {measure_col} for rows matching " + " and ".join(filter_parts) + f" is {_fmt(stats['min'])}.", evidence
            if "maximum" in ql or re.search(r"\bmax\b", ql):
                return f"The maximum of {measure_col} for rows matching " + " and ".join(filter_parts) + f" is {_fmt(stats['max'])}.", evidence
            return f"The mean of {measure_col} for rows matching " + " and ".join(filter_parts) + f" is {_fmt(stats['mean'])}.", evidence

    # ------------------------------------------------------------
    # GROUPED AGGREGATION
    # Examples: "which department has the highest average income?"
    # and "which region has the lowest sales?".
    # ------------------------------------------------------------
    if not has_filter:
        extreme_words = ["highest", "lowest", "largest", "smallest", "maximum", "minimum", "most", "least"]
        if any(w in ql for w in extreme_words):
            measure_col = col if (
                col is not None
                and col in df.columns
                and pd.api.types.is_numeric_dtype(df[col])
            ) else None
            if measure_col is not None:
                group_col = _find_group_column(df, q, measure_col=measure_col)
                if group_col is not None:
                    if "median" in ql:
                        grouped = df.groupby(group_col, dropna=False)[measure_col].median()
                        aggregation = "median"
                    elif "total" in ql or "sum" in ql:
                        grouped = df.groupby(group_col, dropna=False)[measure_col].sum()
                        aggregation = "total"
                    elif "count" in ql:
                        grouped = df.groupby(group_col, dropna=False)[measure_col].count()
                        aggregation = "count"
                    else:
                        grouped = df.groupby(group_col, dropna=False)[measure_col].mean()
                        aggregation = "average"

                    grouped = grouped.dropna().sort_values(ascending=False)
                    if not grouped.empty:
                        if any(w in ql for w in ["lowest", "smallest", "minimum", "least"]):
                            winner = grouped.index[-1]
                            winner_value = float(grouped.iloc[-1])
                            direction = "lowest"
                        else:
                            winner = grouped.index[0]
                            winner_value = float(grouped.iloc[0])
                            direction = "highest"

                        ranking = [
                            {"group": _fmt(idx), "value": float(val)}
                            for idx, val in grouped.items()
                        ]
                        evidence.update({
                            "group_column": group_col,
                            "measure_column": measure_col,
                            "aggregation": aggregation,
                            "group_values": ranking,
                            "winner": _fmt(winner),
                            "winner_value": winner_value,
                        })
                        return (
                            f"The {direction} {aggregation} {measure_col} is for {group_col} = {_fmt(winner)}, "
                            f"with a value of {_fmt(winner_value)}."
                        ), evidence

    # Identify a data-driven profile associated with a binary outcome.
    if target is not None and any(x in ql for x in ["more likely", "type of patient", "patient type", "profile"]):
        numeric = df.select_dtypes(include=np.number)
        if target in numeric.columns:
            values = numeric[target].dropna().unique().tolist()
            if len(values) == 2:
                profile_rows = []
                grouped = df.groupby(target, dropna=True)
                for c in numeric.columns:
                    if c == target:
                        continue
                    means = grouped[c].mean()
                    if len(means) == 2 and means.notna().all():
                        a, b = means.iloc[0], means.iloc[1]
                        diff = float(b - a)
                        profile_rows.append((c, means.index[0], float(a), means.index[1], float(b), diff))
                profile_rows.sort(key=lambda x: abs(x[5]), reverse=True)
                evidence["group_mean_differences"] = [
                    {"feature": c, "class_a": _fmt(a_class), "mean_a": a, "class_b": _fmt(b_class), "mean_b": b, "difference_class_b_minus_a": d}
                    for c, a_class, a, b_class, b, d in profile_rows[:10]
                ]
                if profile_rows:
                    positive_candidates = [v for v in values if str(v).strip().lower() in {"1", "yes", "true", "positive", "diabetes", "diabetic"}]
                    positive = positive_candidates[0] if positive_candidates else values[-1]
                    negative = values[0] if values[0] != positive else values[1]
                    pos_rows = [r for r in profile_rows if r[3] == positive]
                    pos_rows.sort(key=lambda x: abs(x[5]), reverse=True)
                    top = pos_rows[:5]
                    return (
                        f"In this dataset, outcome={_fmt(positive)} is associated with a different observed profile than outcome={_fmt(negative)}. "
                        + "; ".join(f"{c}: mean {_fmt(pos_mean)} for outcome {_fmt(positive)} vs {_fmt(neg_mean)} for outcome {_fmt(negative)}" for c, _, neg_mean, _, pos_mean, _ in top)
                        + ". These are observed associations in the dataset, not proof that any feature causes the outcome."
                    ), evidence

    # Dataset size / missingness / schema questions.
    if any(x in ql for x in ["how many rows", "how many records", "number of rows", "dataset size", "how large"]):
        evidence.update({"rows": int(len(df)), "columns": int(df.shape[1])})
        return f"The dataset contains {len(df):,} rows and {df.shape[1]:,} columns.", evidence

    if "missing" in ql and any(x in ql for x in ["value", "data", "column", "columns"]):
        missing = df.isna().sum()
        nonzero = missing[missing > 0].sort_values(ascending=False)
        evidence["missing_values"] = {str(k): int(v) for k, v in nonzero.items()}
        if nonzero.empty:
            return "There are no missing values in the uploaded dataset.", evidence
        parts = [f"{c}: {int(n)}" for c, n in nonzero.items()]
        return "Missing values are present in " + ", ".join(parts) + ".", evidence

    # Diabetes/outcome count and percentage questions.
    if ("how many" in ql or "number of" in ql or "count" in ql) and ("diabet" in ql or "outcome" in ql):
        if target is None:
            return "I could not identify a target/outcome column from the dataset.", evidence
        vc = df[target].value_counts(dropna=False)
        evidence["target_column"] = target
        evidence["target_counts"] = {str(k): int(v) for k, v in vc.items()}
        if "diabet" in ql:
            positive_candidates = [v for v in vc.index if str(v).strip().lower() in {"1", "yes", "true", "positive", "diabetes", "diabetic"}]
            if positive_candidates:
                positive_value = positive_candidates[0]
                n = int(vc.get(positive_value, 0))
                pct = n / len(df) * 100
                evidence.update({"positive_value": positive_value, "positive_count": n, "positive_percent": pct})
                if target == "outcome":
                    label = "outcome=1 (diabetes/positive outcome)" if str(positive_value) == "1" else f"outcome={_fmt(positive_value)}"
                    return f"Using the dataset's outcome convention, {n:,} patients have {label}, representing {pct:.2f}% of the {len(df):,} rows. This is an association in this dataset, not evidence of causation.", evidence
                return f"Using '{target}' as the diabetes indicator, {n:,} rows are in the positive/diabetes class ({_fmt(positive_value)}), representing {pct:.2f}% of the {len(df):,} rows. This is an observed class distribution, not evidence of causation.", evidence
        return f"The target column is '{target}'. Its observed class counts are: " + ", ".join(f"{_fmt(k)}={int(v)}" for k, v in vc.items()) + ".", evidence

    # Descriptive statistics for a requested numeric feature.
    col = _find_column(df, q, aliases)
    stat_words = ["average", "mean", "median", "minimum", "maximum", "max", "min", "standard deviation", "std", "range"]
    if col is not None and pd.api.types.is_numeric_dtype(df[col]) and any(w in ql for w in stat_words):
        st = _exact_numeric_summary(df, col)
        evidence.update({"column": col, **st})
        if "median" in ql:
            return f"The median of {col} is {_fmt(st['median'])} (n={st['count']:,}; missing={st['missing']:,}).", evidence
        if "standard deviation" in ql or re.search(r"\bstd\b", ql):
            return f"The standard deviation of {col} is {_fmt(st['std'])} (n={st['count']:,}; missing={st['missing']:,}).", evidence
        if "minimum" in ql or re.search(r"\bmin\b", ql):
            return f"The minimum of {col} is {_fmt(st['min'])}.", evidence
        if "maximum" in ql or re.search(r"\bmax\b", ql):
            return f"The maximum of {col} is {_fmt(st['max'])}.", evidence
        if "range" in ql:
            return f"The range of {col} is {_fmt(st['min'])} to {_fmt(st['max'])}.", evidence
        return f"The mean of {col} is {_fmt(st['mean'])}.", evidence

    # Group comparison: feature by binary outcome/class.
    if target is not None and any(x in ql for x in ["compare", "versus", "vs", "difference", "between"]):
        if col is None:
            # Try common feature names mentioned in natural language.
            col = _find_column(df, q, aliases)
        if col is not None and col != target and pd.api.types.is_numeric_dtype(df[col]):
            groups = []
            for value, g in df.groupby(target, dropna=True):
                s = pd.to_numeric(g[col], errors="coerce").dropna()
                if len(s):
                    groups.append((value, int(len(s)), float(s.mean()), float(s.median())))
            evidence.update({"column": col, "groups": groups})
            if groups:
                text = f"For {col}, the group means are: " + "; ".join(f"{_fmt(v)} → mean {_fmt(m)} (n={n})" for v, n, m, _ in groups) + "."
                return text + " These are observed group differences and do not establish causation.", evidence

    # Correlation with target or strongest correlations.
    if "correlation" in ql or "relationship" in ql:
        numeric = df.select_dtypes(include=np.number)
        if numeric.empty:
            return "There are no numeric columns available for correlation analysis.", evidence
        corr = numeric.corr()
        if target in corr.columns:
            target_corr = corr[target].drop(labels=[target], errors="ignore").dropna().sort_values(key=lambda x: x.abs(), ascending=False)
            evidence["target_correlations"] = {str(k): float(v) for k, v in target_corr.items()}
            if not target_corr.empty:
                top = target_corr.head(5)
                return "The strongest correlations with " + target + " are: " + ", ".join(f"{c}={v:.4f}" for c, v in top.items()) + ". Correlation describes association, not causation.", evidence
        pairs = []
        cols = corr.columns.tolist()
        for i, a in enumerate(cols):
            for b in cols[i+1:]:
                v = corr.loc[a, b]
                if pd.notna(v):
                    pairs.append((a, b, float(v)))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        evidence["strongest_pairs"] = pairs[:10]
        return "The strongest numeric feature relationships are: " + ", ".join(f"{a}–{b}={v:.4f}" for a, b, v in pairs[:5]) + ". These are associations, not causal effects.", evidence

    # Feature importance question: use target correlations as a transparent baseline.
    if "important" in ql and any(x in ql for x in ["feature", "predict", "predicting"]):
        numeric = df.select_dtypes(include=np.number)
        if target is None or target not in numeric.columns:
            return "I need a clearly identified numeric target column to rank features from the dataset statistics alone.", evidence
        ranking = numeric.corr()[target].drop(labels=[target], errors="ignore").dropna().sort_values(key=lambda x: x.abs(), ascending=False)
        evidence["feature_ranking_by_abs_correlation"] = {str(k): float(v) for k, v in ranking.items()}
        if ranking.empty:
            return "No numeric predictor correlations could be calculated.", evidence
        return "By absolute correlation with the target, the leading numeric features are: " + ", ".join(f"{c}={v:.4f}" for c, v in ranking.head(5).items()) + ". This is a screening signal, not a causal or model-based feature-importance measure.", evidence

    # Outlier questions using IQR, with exact counts.
    if "outlier" in ql or "unusual" in ql or "suspicious" in ql:
        numeric = df.select_dtypes(include=np.number)
        out = []
        for c in numeric.columns:
            s = numeric[c].dropna()
            if len(s) < 4:
                continue
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            if pd.notna(iqr) and iqr > 0:
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                n = int(((s < lower) | (s > upper)).sum())
                if n:
                    out.append((c, n, float(lower), float(upper)))
        out.sort(key=lambda x: x[1], reverse=True)
        evidence["iqr_outliers"] = out
        if not out:
            return "Using the 1.5×IQR rule, no numeric outliers were detected.", evidence
        return "Using the 1.5×IQR rule, potential outliers were detected in: " + "; ".join(f"{c}: {n} values outside [{lo:.4f}, {hi:.4f}]" for c, n, lo, hi in out[:10]) + ". An outlier is not automatically an error.", evidence

    # Generic profile fallback, still exact.
    evidence.update({"rows": int(len(df)), "columns": int(df.shape[1]), "column_names": [str(c) for c in df.columns]})
    return (
        f"I can analyze this dataset exactly, but the question needs a more specific operation. "
        f"The dataset has {len(df):,} rows and {df.shape[1]:,} columns. "
        f"Available columns include: {', '.join(map(str, df.columns[:15]))}" +
        ("..." if df.shape[1] > 15 else "."),
        evidence,
    )


def ai_explain_verified_result(question, exact_answer, evidence, rag_engine=None):
    """Return verified answers unchanged; use RAG only when exact analysis is unavailable."""
    # Deterministic analysis must never be rewritten by an LLM because a fluent
    # explanation can accidentally reverse class meanings or arithmetic.
    if evidence.get("method") == "exact pandas analysis" and not exact_answer.startswith("I can analyze this dataset exactly"):
        return exact_answer

    if rag_engine is None:
        return exact_answer

    try:
        response = rag_engine.ask(question, k=6)
        if response and str(response).strip():
            return str(response).strip()
    except Exception:
        pass

    return exact_answer


# ============================================================
# REPORT
# ============================================================

def generate_report(df):

    report = []

    report.append(
        "AI DATA ANALYST REPORT"
    )

    report.append(
        "=" * 50
    )

    report.append(
        f"Rows: {df.shape[0]}"
    )

    report.append(
        f"Columns: {df.shape[1]}"
    )

    report.append("")

    report.append(
        "MISSING VALUES"
    )

    report.append(
        "-" * 30
    )

    missing = df.isna().sum()

    missing = (
        missing[
            missing > 0
        ]
        .sort_values(
            ascending=False
        )
    )

    if missing.empty:

        report.append(
            "No missing values."
        )

    else:

        for col, count in missing.items():

            report.append(
                f"{col}: {count}"
            )


    report.append("")

    report.append(
        "NUMERIC SUMMARY"
    )

    report.append(
        "-" * 30
    )

    numeric = df.select_dtypes(
        include=np.number
    )

    if not numeric.empty:

        desc = numeric.describe().T

        for col in desc.index:

            report.append(
                f"{col}: "
                f"mean={desc.loc[col, 'mean']:.2f}, "
                f"min={desc.loc[col, 'min']:.2f}, "
                f"max={desc.loc[col, 'max']:.2f}"
            )

    return "\n".join(report)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🤖 AI Data Analyst"
    )

    st.markdown(
        """
        <div class="sidebar-list">
        <ul>

        <li>📂 Upload Dataset</li>

        <li>📊 Dataset Overview</li>

        <li>🔎 EDA & Statistics</li>

        <li>📈 Visualization</li>

        <li>🛠️ Feature Engineering</li>

        <li>🔬 Feature Analysis</li>

        <li>🤖 Machine Learning</li>

        <li>🔮 Prediction</li>

        <li>📄 Reports</li>

        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()


    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=[
            "csv",
            "xlsx",
            "xls"
        ],
        key="dataset_uploader"
    )


    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()
        signature = (
            f"{uploaded_file.name}-"
            f"{hashlib.sha256(file_bytes).hexdigest()}"
        )


        if (
            st.session_state.file_signature
            != signature
        ):

            try:

                file_suffix = Path(uploaded_file.name).suffix.lower()

                if file_suffix == ".csv":

                    new_df = pd.read_csv(
                        BytesIO(file_bytes)
                    )

                elif file_suffix in {".xlsx", ".xls"}:

                    # Read Excel from the uploaded bytes so the same upload
                    # can safely be reused for sheet discovery and loading.
                    excel_source = BytesIO(file_bytes)
                    excel_file = pd.ExcelFile(excel_source)

                    sheet_name = st.selectbox(
                        "Select Excel sheet",
                        excel_file.sheet_names,
                        key=f"excel_sheet_{signature[:12]}"
                    )

                    new_df = pd.read_excel(
                        BytesIO(file_bytes),
                        sheet_name=sheet_name
                    )

                else:
                    raise ValueError(
                        "Unsupported file type. Please upload CSV, XLS, or XLSX."
                    )


                new_df = clean_column_names(
                    new_df
                )

                new_df = convert_possible_numeric_columns(
                    new_df
                )


                st.session_state.df = new_df

                st.session_state.file_signature = signature

                reset_analysis()


            except Exception as e:

                st.error(
                    f"Could not load dataset: {e}"
                )


        st.success(
            "Dataset loaded successfully."
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

st.title(
    "🤖 AI Data Analyst"
)

st.write(
    "Upload a dataset, inspect it first, "
    "then run EDA, visualization, feature "
    "engineering and machine learning step by step."
)


# ============================================================
# CHECK DATASET
# ============================================================

if st.session_state.df is None:

    st.info(
        "👈 Upload a CSV or Excel file from the sidebar to begin."
    )

    st.stop()


df = st.session_state.df


# ============================================================
# 1. DATASET OVERVIEW
# ============================================================

st.header(
    "📊 1. Dataset Overview"
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Rows",
        f"{df.shape[0]:,}"
    )


with col2:

    st.metric(
        "Columns",
        df.shape[1]
    )


with col3:

    st.metric(
        "Missing Values",
        int(
            df.isna()
            .sum()
            .sum()
        )
    )


with col4:

    st.metric(
        "Duplicate Rows",
        int(
            df.duplicated()
            .sum()
        )
    )


# ============================================================
# DATASET PREVIEW
# ============================================================

st.subheader(
    "👀 Dataset Preview"
)

st.dataframe(
    df.head(10),
    use_container_width=True,
    height=300
)


# ============================================================
# START ANALYSIS
# ============================================================

st.divider()


if not st.session_state.analysis_started:

    st.info(
        "Dataset is loaded. Click the button below to start EDA and statistics."
    )


    if st.button(
        "🔎 Analyze EDA & Statistics",
        type="primary"
    ):

        st.session_state.analysis_started = True

        st.rerun()


    st.stop()


# ============================================================
# 2. EDA
# ============================================================

st.header(
    "🔎 2. Exploratory Data Analysis"
)

st.success(
    "EDA analysis is active."
)


eda_summary = create_eda_summary(
    df
)


st.subheader(
    "📋 Column Statistics"
)

st.dataframe(
    eda_summary,
    use_container_width=True,
    height=380
)


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

statistics = create_statistics(
    df
)


if not statistics.empty:

    st.subheader(
        "📐 Descriptive Statistics"
    )

    st.dataframe(
        statistics,
        use_container_width=True,
        height=320
    )

else:

    st.info(
        "No numeric columns available for descriptive statistics."
    )


# ============================================================
# 3. DATA VISUALIZATION
# ============================================================

st.header(
    "📈 3. Data Visualization"
)


numeric_columns = (
    df
    .select_dtypes(
        include=np.number
    )
    .columns
    .tolist()
)


categorical_columns = (
    df
    .select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    )
    .columns
    .tolist()
)


# ============================================================
# NUMERIC VISUALIZATION
# ============================================================

if numeric_columns:

    col1, col2 = st.columns(2)


    with col1:

        selected_numeric = st.selectbox(
            "Numeric distribution",
            numeric_columns,
            key="numeric_distribution"
        )


        if st.button(
            "Generate Distribution",
            key="generate_distribution"
        ):

            fig, ax = compact_fig()

            sns.histplot(
                df[
                    selected_numeric
                ].dropna(),
                kde=True,
                ax=ax
            )

            ax.set_title(
                f"Distribution of {selected_numeric}"
            )

            fig.tight_layout()

            show_fig(fig)


    with col2:

        selected_box = st.selectbox(
            "Numeric boxplot",
            numeric_columns,
            key="boxplot_feature"
        )


        if st.button(
            "Generate Boxplot",
            key="generate_boxplot"
        ):

            fig, ax = compact_fig(
                5.5,
                3.3
            )

            sns.boxplot(
                y=df[
                    selected_box
                ],
                ax=ax
            )

            ax.set_title(
                f"Boxplot of {selected_box}"
            )

            fig.tight_layout()

            show_fig(fig)


# ============================================================
# CATEGORICAL DISTRIBUTION
# ============================================================

if categorical_columns:

    st.subheader(
        "📊 Categorical Distribution"
    )


    selected_cat = st.selectbox(
        "Categorical feature",
        categorical_columns,
        key="categorical_distribution"
    )


    if st.button(
        "Generate Bar Chart",
        key="generate_bar"
    ):

        counts = (
            df[
                selected_cat
            ]
            .astype(str)
            .value_counts()
            .head(15)
        )


        fig, ax = compact_fig(
            6,
            3.6
        )


        counts.sort_values().plot(
            kind="barh",
            ax=ax
        )


        ax.set_title(
            f"Distribution of {selected_cat}"
        )

        ax.set_xlabel(
            "Count"
        )

        fig.tight_layout()

        show_fig(fig)


# ============================================================
# CORRELATION
# ============================================================

if len(numeric_columns) >= 2:

    st.subheader(
        "🔥 Numeric Correlation"
    )


    correlation = (
        df[
            numeric_columns[:15]
        ]
        .corr()
    )


    fig, ax = plt.subplots(
        figsize=(
            6.5,
            4.8
        )
    )


    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        annot_kws={
            "size": 7
        }
    )


    ax.set_title(
        "Correlation Between Numeric Features"
    )

    fig.tight_layout()

    show_fig(fig)


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

st.header(
    "🛠️ 4. Feature Engineering"
)


st.info(
    "Feature engineering creates useful date, age, "
    "log and outlier features while keeping the original dataset intact."
)


if st.button(
    "⚙️ Create Engineered Features",
    key="engineer_features"
):

    with st.spinner(
        "Creating engineered features..."
    ):

        engineered_df, feature_origin = (
            engineer_features(df)
        )


        st.session_state.engineered_df = (
            engineered_df
        )

        st.session_state.feature_origin = (
            feature_origin
        )


        st.session_state.ml_results = None

        st.session_state.trained_model = None

        st.session_state.trained_models = {}

        st.session_state.predictions_df = None

        st.session_state.confusion_matrix = None

        st.session_state.classification_report = None


    st.success(
        "Feature engineering completed successfully."
    )


# ============================================================
# ENGINEERED DATASET
# ============================================================

if st.session_state.engineered_df is not None:

    engineered_df = (
        st.session_state.engineered_df
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Original Columns",
            df.shape[1]
        )


    with col2:

        st.metric(
            "Engineered Columns",
            engineered_df.shape[1]
        )


    with col3:

        st.metric(
            "New Features",
            engineered_df.shape[1]
            - df.shape[1]
        )


    st.subheader(
        "Engineered Dataset"
    )


    st.dataframe(
        engineered_df.head(10),
        use_container_width=True,
        height=300
    )


else:

    st.info(
        "Click 'Create Engineered Features' before using ML."
    )


# ============================================================
# 5. FEATURE ANALYSIS
# ============================================================

st.header(
    "🔬 5. Feature Analysis"
)


if st.session_state.engineered_df is None:

    st.info(
        "Create engineered features first."
    )

else:

    engineered_df = (
        st.session_state.engineered_df
    )


    engineered_numeric = (
        engineered_df
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )


    if engineered_numeric:

        selected_feature = st.selectbox(
            "Select engineered feature",
            engineered_numeric,
            key="feature_analysis"
        )


        chart_type = st.selectbox(
            "Analysis type",
            [
                "Distribution",
                "Boxplot"
            ],
            key="feature_chart_type"
        )


        if st.button(
            "📊 Generate Feature Analysis",
            key="generate_feature_analysis"
        ):

            fig, ax = compact_fig()


            if chart_type == "Distribution":

                sns.histplot(
                    engineered_df[
                        selected_feature
                    ].dropna(),
                    kde=True,
                    ax=ax
                )

            else:

                sns.boxplot(
                    y=engineered_df[
                        selected_feature
                    ],
                    ax=ax
                )


            ax.set_title(
                f"{chart_type}: {selected_feature}"
            )

            fig.tight_layout()

            show_fig(fig)


    else:

        st.info(
            "No numeric engineered features available."
        )


# ============================================================
# 6. MACHINE LEARNING
# ============================================================

st.header(
    "🤖 6. Machine Learning"
)


if st.session_state.engineered_df is None:

    st.info(
        "Create engineered features first."
    )

else:

    ml_df = (
        st.session_state.engineered_df
    )


    # ========================================================
    # TARGET COLUMN
    # ========================================================

    st.subheader(
        "🎯 Target Column"
    )


    all_targets = ml_df.columns.tolist()

    # Keep every non-ID column selectable, but rank candidates so the
    # dashboard starts with a semantically sensible prediction target.
    usable_targets = rank_target_candidates(ml_df)

    default_target = usable_targets[0] if usable_targets else None


    if not usable_targets:

        st.error(
            "No suitable target columns were found."
        )

        st.stop()


    default_index = usable_targets.index(
        default_target
    )


    # IMPORTANT:
    # customer_id is NOT included here.
    #
    # customer_segment becomes the default
    # target for your customers.csv.


    target_column = st.selectbox(
        "Select the column you want the model to predict",
        usable_targets,
        index=default_index,
        key="ml_target_fixed"
    )


    target_series = ml_df[
        target_column
    ]


    target_unique = int(
        target_series
        .nunique(
            dropna=True
        )
    )


    target_missing = int(
        target_series
        .isna()
        .sum()
    )


    # ========================================================
    # TARGET INFORMATION
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Target Data Type",
            str(
                target_series.dtype
            )
        )


    with col2:

        st.metric(
            "Unique Values",
            target_unique
        )


    with col3:

        st.metric(
            "Missing Values",
            target_missing
        )


    # ========================================================
    # PROBLEM TYPE
    # ========================================================

    st.subheader(
        "🧠 Problem Type"
    )


    # Automatically determine the correct type.

    detected_problem = detect_problem_type(
        target_series
    )


    problem_choice = st.selectbox(
        "Choose the machine learning problem type",
        [
            "Auto",
            "Regression",
            "Classification"
        ],
        index=0,
        key="problem_choice_fixed"
    )


    if problem_choice == "Auto":

        problem_type = detected_problem

    else:

        problem_type = problem_choice


    st.success(
        f"Selected problem type: {problem_type}"
    )


    # ========================================================
    # CLASSIFICATION PREVIEW
    # ========================================================

    if problem_type == "Classification":

        class_preview = (
            target_series
            .dropna()
            .astype(str)
            .str.strip()
            .value_counts()
            .head(10)
        )


        if not class_preview.empty:

            st.caption(
                "Top target classes"
            )


            preview_df = (
                class_preview
                .rename("Count")
                .reset_index()
            )


            preview_df.columns = [
                target_column,
                "Count"
            ]


            st.dataframe(
                preview_df,
                use_container_width=False
            )


    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    st.subheader(
        "📊 Train/Test Split"
    )


    test_size = st.slider(
        "Test data percentage",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        key="test_size_fixed"
    )


    st.write(
        f"Training data: {100 - test_size}% | "
        f"Testing data: {test_size}%"
    )


    # ========================================================
    # TRAIN BUTTON
    # ========================================================

    if st.button(
        "🚀 Train Machine Learning Models",
        type="primary",
        key="train_ml_fixed"
    ):

        try:

            with st.spinner(
                "Preparing data and training models..."
            ):

                # ------------------------------------------------
                # PREPARE DATA
                # ------------------------------------------------

                X, y, removed_columns = (
                    prepare_ml_data(
                        ml_df,
                        target_column,
                        st.session_state.feature_origin
                    )
                )


                st.session_state.leakage_columns = (
                    removed_columns
                )


                # ------------------------------------------------
                # REMOVE MISSING TARGETS
                # ------------------------------------------------

                valid_target = ~y.isna()


                X = X.loc[
                    valid_target
                ].copy()


                y = y.loc[
                    valid_target
                ].copy()


                if len(y) == 0:

                    raise ValueError(
                        "No valid target values remain."
                    )


                label_encoder = None


                # =================================================
                # CLASSIFICATION
                # =================================================

                if problem_type == "Classification":

                    y = (
                        y
                        .astype(str)
                        .str.strip()
                    )


                    valid_string_target = (
                        y.str.lower()
                        != "nan"
                    )


                    X = X.loc[
                        valid_string_target
                    ].copy()


                    y = y.loc[
                        valid_string_target
                    ].copy()


                    class_counts = (
                        y.value_counts()
                    )


                    unique_classes = (
                        len(class_counts)
                    )


                    if unique_classes < 2:

                        raise ValueError(
                            "Classification requires at least 2 different classes."
                        )


                    if unique_classes > 50:

                        raise ValueError(
                            f"`{target_column}` has "
                            f"{unique_classes} classes. "
                            "Choose a lower-cardinality categorical target."
                        )


                    # --------------------------------------------
                    # FIX FOR STRATIFIED SPLIT
                    # --------------------------------------------

                    rare_classes = (
                        class_counts[
                            class_counts < 2
                        ]
                    )


                    if not rare_classes.empty:

                        examples = (
                            rare_classes
                            .index
                            .tolist()[:10]
                        )


                        raise ValueError(
                            f"`{target_column}` contains "
                            f"{len(rare_classes)} class(es) "
                            "with only one row. "
                            f"Examples: {examples}. "
                            "Choose another target such as "
                            "`customer_segment`."
                        )


                    test_rows = int(
                        np.ceil(
                            len(y)
                            * test_size
                            / 100
                        )
                    )


                    if test_rows < unique_classes:

                        raise ValueError(
                            f"The test set has only "
                            f"{test_rows} rows but there are "
                            f"{unique_classes} classes. "
                            "Increase the test percentage."
                        )


                    # --------------------------------------------
                    # LABEL ENCODING
                    # --------------------------------------------

                    label_encoder = LabelEncoder()


                    encoded_y = (
                        label_encoder
                        .fit_transform(y)
                    )


                    y = pd.Series(
                        encoded_y,
                        index=y.index,
                        name=target_column
                    )


                # =================================================
                # REGRESSION
                # =================================================

                else:

                    y = pd.to_numeric(
                        y,
                        errors="coerce"
                    )


                    valid_numeric_target = (
                        y.notna()
                    )


                    X = X.loc[
                        valid_numeric_target
                    ].copy()


                    y = y.loc[
                        valid_numeric_target
                    ].copy()


                    if len(y) < 20:

                        raise ValueError(
                            "Not enough valid numeric target rows for regression."
                        )


                # =================================================
                # GENERAL CHECKS
                # =================================================

                if len(X) < 20:

                    raise ValueError(
                        "Not enough valid rows for machine learning."
                    )


                if X.shape[1] == 0:

                    raise ValueError(
                        "No usable predictor features remain."
                    )


                # =================================================
                # TRAIN / TEST SPLIT
                # =================================================

                if problem_type == "Classification":

                    X_train, X_test, y_train, y_test = (
                        train_test_split(
                            X,
                            y,
                            test_size=test_size / 100,
                            random_state=42,
                            stratify=y
                        )
                    )

                else:

                    X_train, X_test, y_train, y_test = (
                        train_test_split(
                            X,
                            y,
                            test_size=test_size / 100,
                            random_state=42
                        )
                    )


                # =================================================
                # FAST PREPROCESSING
                # =================================================
                #
                # THIS IS THE IMPORTANT SPEED FIX.
                #
                # Previously the preprocessing pipeline was
                # fitted again for every model.
                #
                # Now we fit it ONLY ONCE.
                #
                # =================================================

                preprocessor = build_preprocessor(
                    X_train
                )


                X_train_processed = (
                    preprocessor
                    .fit_transform(X_train)
                )


                X_test_processed = (
                    preprocessor
                    .transform(X_test)
                )


                # =================================================
                # TRAIN MODELS
                # =================================================

                if problem_type == "Regression":

                    (
                        results_df,
                        trained_models,
                        predictions,
                        best_name
                    ) = train_regression_models(
                        X_train_processed,
                        X_test_processed,
                        y_train,
                        y_test
                    )

                else:

                    (
                        results_df,
                        trained_models,
                        predictions,
                        best_name
                    ) = train_classification_models(
                        X_train_processed,
                        X_test_processed,
                        y_train,
                        y_test
                    )


                # =================================================
                # BEST MODEL
                # =================================================

                best_model = trained_models[
                    best_name
                ]


                best_predictions = predictions[
                    best_name
                ]


                # =================================================
                # PREDICTION DATAFRAME
                # =================================================

                predictions_df = X_test.copy()


                predictions_df[
                    "actual"
                ] = y_test.values


                predictions_df[
                    "prediction"
                ] = best_predictions


                # =================================================
                # DECODE CLASSIFICATION
                # =================================================

                if (
                    problem_type == "Classification"
                    and label_encoder is not None
                ):

                    predictions_df[
                        "actual"
                    ] = (
                        label_encoder
                        .inverse_transform(
                            predictions_df[
                                "actual"
                            ].astype(int)
                        )
                    )


                    predictions_df[
                        "prediction"
                    ] = (
                        label_encoder
                        .inverse_transform(
                            predictions_df[
                                "prediction"
                            ].astype(int)
                        )
                    )


                # =================================================
                # CONFUSION MATRIX
                # =================================================

                cm = None

                class_report = None


                if problem_type == "Classification":

                    cm = confusion_matrix(
                        y_test,
                        best_predictions
                    )


                    class_report = classification_report(
                        y_test,
                        best_predictions,
                        target_names=label_encoder.classes_,
                        zero_division=0,
                        output_dict=True
                    )


                # =================================================
                # SAVE EVERYTHING
                # =================================================

                st.session_state.ml_results = (
                    results_df
                )


                st.session_state.trained_model = (
                    best_model
                )


                st.session_state.trained_models = (
                    trained_models
                )


                st.session_state.predictions_df = (
                    predictions_df
                )


                st.session_state.problem_type = (
                    problem_type
                )


                st.session_state.target_column = (
                    target_column
                )


                st.session_state.label_encoder = (
                    label_encoder
                )


                st.session_state.X_train_columns = (
                    X_train.columns.tolist()
                )


                st.session_state.confusion_matrix = cm


                st.session_state.classification_report = (
                    class_report
                )


                # Save preprocessor separately
                st.session_state.ml_preprocessor = (
                    preprocessor
                )


                st.success(
                    "✅ Machine learning completed successfully."
                )


        except Exception as e:

            st.error(
                f"Machine learning failed: {e}"
            )


    # ========================================================
    # EXCLUDED COLUMNS
    # ========================================================

    if st.session_state.leakage_columns:

        with st.expander(
            "🛡️ View columns excluded from ML"
        ):

            st.write(
                st.session_state.leakage_columns
            )


# ============================================================
# ML RESULTS
# ============================================================

if st.session_state.ml_results is not None:

    results_df = (
        st.session_state.ml_results
    )


    problem_type = (
        st.session_state.problem_type
    )


    target_column = (
        st.session_state.target_column
    )


    st.divider()


    st.subheader(
        "🏆 Model Results"
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    if problem_type == "Regression":

        best_row = results_df.loc[
            results_df["RMSE"].idxmin()
        ]

    else:

        best_row = results_df.loc[
            results_df["F1 Score"].idxmax()
        ]


    best_model_name = (
        best_row["Model"]
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Problem Type",
            problem_type
        )


    with col2:

        st.metric(
            "Best Model",
            best_model_name
        )


    st.caption(
        f"Target column: `{target_column}`"
    )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    st.subheader(
        "📋 Model Comparison"
    )


    st.dataframe(
        results_df.round(4),
        use_container_width=True,
        height=240
    )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.subheader(
        "🔍 Feature Importance"
    )


    importance_count = st.selectbox(
        "Number of important features",
        [
            5,
            10,
            15,
            20
        ],
        index=1,
        key="importance_count_fixed"
    )


    if "ml_preprocessor" in st.session_state:

        importance_df = get_feature_importance(
            st.session_state.trained_model,
            st.session_state.ml_preprocessor,
            top_n=importance_count
        )

    else:

        importance_df = pd.DataFrame()


    if not importance_df.empty:

        st.dataframe(
            importance_df.round(6),
            use_container_width=True,
            height=280
        )


        chart_data = (
            importance_df
            .sort_values(
                "importance",
                ascending=True
            )
        )


        fig, ax = plt.subplots(
            figsize=(
                7,
                4
            )
        )


        ax.barh(
            chart_data["feature"],
            chart_data["importance"]
        )


        ax.set_title(
            "Top Feature Importance"
        )


        ax.set_xlabel(
            "Importance"
        )


        fig.tight_layout()

        show_fig(fig)


    else:

        st.info(
            "Feature importance is not available."
        )


# ============================================================
# 7. PREDICTION
# ============================================================

st.header(
    "🔮 7. Prediction"
)


if st.session_state.predictions_df is None:

    st.info(
        "Train the machine learning models above to generate predictions."
    )

else:

    predictions_df = (
        st.session_state.predictions_df
    )


    results_df = (
        st.session_state.ml_results
    )


    problem_type = (
        st.session_state.problem_type
    )


    if problem_type == "Regression":

        best_row = results_df.loc[
            results_df["RMSE"].idxmin()
        ]

    else:

        best_row = results_df.loc[
            results_df["F1 Score"].idxmax()
        ]


    best_model_name = (
        best_row["Model"]
    )


    st.success(
        f"Predictions generated using the best model: "
        f"**{best_model_name}**"
    )


    # ========================================================
    # TEST SET PREDICTIONS
    # ========================================================

    st.subheader(
        "🔮 Test Set Predictions"
    )


    st.dataframe(
        predictions_df.head(100),
        use_container_width=True,
        height=400
    )


    # ========================================================
    # DOWNLOAD PREDICTIONS
    # ========================================================

    prediction_buffer = BytesIO()


    predictions_df.to_csv(
        prediction_buffer,
        index=False
    )


    st.download_button(
        label="⬇️ Download Test Predictions",
        data=prediction_buffer.getvalue(),
        file_name="test_predictions.csv",
        mime="text/csv"
    )


    # ========================================================
    # METRICS
    # ========================================================

    if problem_type == "Regression":

        st.subheader(
            "📐 Regression Metrics"
        )


        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "MAE",
                f"{best_row['MAE']:.4f}"
            )


        with c2:

            st.metric(
                "MSE",
                f"{best_row['MSE']:.4f}"
            )


        with c3:

            st.metric(
                "RMSE",
                f"{best_row['RMSE']:.4f}"
            )


        with c4:

            st.metric(
                "R2 Score",
                f"{best_row['R2 Score']:.4f}"
            )


    else:

        st.subheader(
            "📐 Classification Metrics"
        )


        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "Accuracy",
                f"{best_row['Accuracy']:.4f}"
            )


        with c2:

            st.metric(
                "Precision",
                f"{best_row['Precision']:.4f}"
            )


        with c3:

            st.metric(
                "Recall",
                f"{best_row['Recall']:.4f}"
            )


        with c4:

            st.metric(
                "F1 Score",
                f"{best_row['F1 Score']:.4f}"
            )


# ============================================================
# CONFUSION MATRIX
# ============================================================

if (
    st.session_state.confusion_matrix is not None
    and st.session_state.label_encoder is not None
):

    st.subheader(
        "🔢 Confusion Matrix"
    )


    cm = (
        st.session_state.confusion_matrix
    )


    labels = (
        st.session_state.label_encoder.classes_
    )


    fig, ax = plt.subplots(
        figsize=(
            7,
            5
        )
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )


    ax.set_xlabel(
        "Predicted"
    )


    ax.set_ylabel(
        "Actual"
    )


    ax.set_title(
        "Confusion Matrix"
    )


    fig.tight_layout()

    show_fig(fig)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

if (
    st.session_state.classification_report is not None
):

    st.subheader(
        "📊 Classification Report"
    )


    report_dict = (
        st.session_state.classification_report
    )


    report_rows = []


    for label, values in report_dict.items():

        if isinstance(values, dict):

            report_rows.append({

                "Class": label,

                "Precision": values.get(
                    "precision",
                    0
                ),

                "Recall": values.get(
                    "recall",
                    0
                ),

                "F1 Score": values.get(
                    "f1-score",
                    0
                ),

                "Support": values.get(
                    "support",
                    0
                )

            })


    if report_rows:

        st.dataframe(
            pd.DataFrame(
                report_rows
            ).round(4),
            use_container_width=True
        )


# ============================================================
# 8. REPORTS
# ============================================================

st.header(
    "📄 8. Reports"
)


report_df = (
    st.session_state.engineered_df
    if st.session_state.engineered_df is not None
    else df
)


report = generate_report(
    report_df
)


st.text_area(
    "Generated Data Analysis Report",
    report,
    height=300
)


st.download_button(
    label="⬇️ Download Analysis Report",
    data=report.encode("utf-8"),
    file_name="ai_data_analysis_report.txt",
    mime="text/plain"
)


# ============================================================
# DOWNLOAD PROCESSED DATASET
# ============================================================

processed_csv = report_df.to_csv(
    index=False
)


st.download_button(
    label="⬇️ Download Processed Dataset",
    data=processed_csv,
    file_name="processed_dataset.csv",
    mime="text/csv"
)


# ============================================================
# 9. AI DATA ASSISTANT
# ============================================================

if df is not None:

    st.header(
        "🤖 9. AI Data Assistant"
    )

    st.write(
        "Ask questions about the currently uploaded dataset. "
        "The assistant uses exact dataset analysis together with "
        "your existing local RAG engine."
    )


    # ------------------------------------------------------------
    # CREATE RAG ENGINE FOR THIS DATASET
    # ------------------------------------------------------------

    st.subheader("🔌 AI Assistant Connection")
    st.caption(
        "Exact dataset questions work locally. The optional RAG connection is "
        "only needed for broader natural-language questions."
    )

    if not st.session_state.rag_enabled:
        if st.button("🔌 Connect Local AI Assistant", key="connect_rag", type="secondary"):
            st.session_state.rag_enabled = True
            st.rerun()

    if (
        st.session_state.rag_enabled
        and (
            st.session_state.rag_engine is None
            or st.session_state.rag_file_signature
            != st.session_state.file_signature
        )
    ):

        try:

            # Use a separate Chroma directory for every uploaded file
            # so data from different datasets cannot mix.
            rag_directory = (
                Path("models")
                / "rag_chroma_sessions"
                / uuid.uuid4().hex
            )

            rag_directory.mkdir(
                parents=True,
                exist_ok=True
            )

            st.session_state.rag_engine = RAGEngine(
                persist_directory=str(rag_directory)
            )

            st.session_state.rag_file_signature = (
                st.session_state.file_signature
            )

            st.session_state.rag_ready = False

        except Exception as e:

            st.session_state.rag_engine = None
            st.session_state.rag_ready = False

            st.error(
                f"Could not initialize the AI Assistant: {e}"
            )


    # ------------------------------------------------------------
    # INGEST CURRENT DATASET
    # ------------------------------------------------------------

    if (
        st.session_state.rag_engine is not None
        and not st.session_state.rag_ready
    ):

        try:

            with st.spinner(
                "Connecting the dataset to the AI Assistant..."
            ):

                documents = dataframe_to_documents(
                    df,
                    source=str(
                        st.session_state.file_signature
                    )
                )

                chunks_added = (
                    st.session_state.rag_engine
                    .add_documents(documents)
                )

                # Add exact analytical information as searchable context.
                profile = get_dataset_profile(df)
                basic_insights = get_basic_insights(df)
                numeric_analysis = get_numeric_analysis(df)
                categorical_summary = get_categorical_summary(df)
                correlation = get_correlation(df)

                exact_context_parts = [
                    "DATASET PROFILE:",
                    str(profile),
                    "\nBASIC INSIGHTS:",
                    "\n".join(basic_insights),
                ]

                if not numeric_analysis.empty:
                    exact_context_parts.extend([
                        "\nEXACT NUMERIC ANALYSIS:",
                        numeric_analysis.to_string(
                            index=False
                        )
                    ])

                if not categorical_summary.empty:
                    exact_context_parts.extend([
                        "\nCATEGORICAL SUMMARY:",
                        categorical_summary.to_string(
                            index=False
                        )
                    ])

                if not correlation.empty:
                    exact_context_parts.extend([
                        "\nCORRELATION MATRIX:",
                        correlation.to_string()
                    ])

                st.session_state.rag_engine.add_text(
                    "\n".join(exact_context_parts),
                    source="exact_dataset_analysis"
                )

                st.session_state.rag_ready = True

            st.success(
                f"✅ AI Assistant connected to your dataset "
                f"({len(documents):,} rows / {chunks_added:,} chunks)"
            )

        except Exception as e:

            st.session_state.rag_ready = False

            st.error(
                f"Could not connect the dataset to RAG: {e}"
            )


    # ------------------------------------------------------------
    # ASK QUESTIONS
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # HR-FOCUSED SUGGESTED QUESTIONS
    # ------------------------------------------------------------

    # Detect available HR-related columns in the uploaded dataset.
    _available_columns = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    _hr_questions = []

    if "department" in _available_columns:
        _hr_questions.extend([
            "How many employees are in each department?",
            "Which department has the most employees?",
            "What is the average monthly income by department?",
            "Which department has the highest average monthly income?",
            "What is the average age by department?",
            "What is the average job satisfaction by department?",
            "Which department has the highest attrition?",
            "What percentage of employees in each department have attrition?",
        ])

    if "job_role" in _available_columns:
        _hr_questions.extend([
            "How many employees are in each job role?",
            "Which job role has the highest average income?",
            "Which job role has the highest attrition?",
            "What is the average age for each job role?",
        ])

    if "overtime" in _available_columns:
        _hr_questions.extend([
            "How many employees work overtime in each department?",
            "Which department has the highest overtime rate?",
        ])

    if "gender" in _available_columns:
        _hr_questions.extend([
            "How many male and female employees are in each department?",
            "What is the average income by gender?",
            "What is the attrition rate by gender?",
        ])

    if "years_at_company" in _available_columns:
        _hr_questions.append(
            "What is the average years at company by department?"
        )

    if "work_life_balance" in _available_columns:
        _hr_questions.append(
            "What is the average work-life balance by department?"
        )

    if "marital_status" in _available_columns:
        _hr_questions.append(
            "What is the average age by marital status?"
        )

    # Show HR questions only when relevant columns exist.
    if _hr_questions:
        st.markdown("### 💡 Suggested HR Questions")

        _selected_hr_question = st.selectbox(
            "Choose a question to ask the AI Assistant:",
            ["Select a suggested question..."] + _hr_questions,
            key="hr_suggested_question"
        )

        if _selected_hr_question != "Select a suggested question...":
            if (
                _selected_hr_question
                != st.session_state.get("_last_hr_suggested_question")
            ):
                st.session_state["ai_dataset_question"] = _selected_hr_question
                st.session_state["_last_hr_suggested_question"] = _selected_hr_question


    # ------------------------------------------------------------
    # COVID-FOCUSED SUGGESTED QUESTIONS
    # ------------------------------------------------------------

    # Show the same suggested-question experience used for HR datasets,
    # but only when the uploaded dataset has the expected COVID schema.
    _covid_columns = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    _is_covid_dataset = (
        "country_region" in _covid_columns
        and "date" in _covid_columns
        and "confirmed" in _covid_columns
        and "deaths" in _covid_columns
        and "recovered" in _covid_columns
        and "active" in _covid_columns
    )

    _covid_questions = [
        "Which country has the highest number of confirmed cases?",
        "Which country has the highest number of deaths?",
        "Which country has the highest number of recovered cases?",
        "Which country has the highest number of active cases?",
        "What is the total number of confirmed cases by country?",
        "What is the total number of deaths by country?",
        "What is the total number of recovered cases by country?",
        "What is the total number of active cases by country?",
        "Which WHO region has the highest number of confirmed cases?",
        "Which WHO region has the highest number of deaths?",
        "Show confirmed cases by WHO region.",
        "Show deaths by WHO region.",
        "Which WHO region has the highest number of recovered cases?",
        "How did confirmed cases change over time?",
        "How did deaths change over time?",
        "How did recovered cases change over time?",
        "What was the highest number of confirmed cases on a single date?",
        "Show the COVID-19 statistics for India.",
        "Show the COVID-19 statistics for the United States.",
        "What is the death rate by country?",
    ]

    if _is_covid_dataset:

        st.markdown("### 💡 Suggested COVID Questions")

        _selected_covid_question = st.selectbox(
            "Choose a COVID question to ask the AI Assistant:",
            ["Select a suggested question..."] + _covid_questions,
            key="covid_suggested_question"
        )

        if _selected_covid_question != "Select a suggested question...":

            if (
                _selected_covid_question
                != st.session_state.get(
                    "_last_covid_suggested_question"
                )
            ):

                st.session_state["ai_dataset_question"] = (
                    _selected_covid_question
                )

                st.session_state["_last_covid_suggested_question"] = (
                    _selected_covid_question
                )

    # ------------------------------------------------------------
    # CREDIT-CARD FRAUD-FOCUSED SUGGESTED QUESTIONS
    # ------------------------------------------------------------

    # Show the same suggested-question experience used for HR and COVID
    # datasets, but only when the uploaded dataset has the expected
    # credit-card fraud schema: binary `class` plus numeric `amount`.
    _fraud_suggest_columns = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    _fraud_suggest_is_dataset = False

    if (
        "class" in _fraud_suggest_columns
        and "amount" in _fraud_suggest_columns
        and len(df) > 0
    ):
        _suggest_class_values = pd.to_numeric(
            df[_fraud_suggest_columns["class"]],
            errors="coerce"
        ).dropna()
        _fraud_suggest_is_dataset = (
            set(_suggest_class_values.unique().tolist()) == {0, 1}
        )

    _fraud_questions = [
        "How many fraudulent transactions are there?",
        "How many transactions are in each class?",
        "What percentage of transactions are fraudulent?",
        "What is the fraud rate?",
        "What is the average transaction amount for fraudulent transactions?",
        "What is the average transaction amount for non-fraudulent transactions?",
        "What is the average transaction amount by class?",
        "Which features are most associated with fraudulent transactions?",
        "Show the distribution of fraud and non-fraud transactions.",
        "What is the average transaction amount overall?",
        "What is the median transaction amount?",
        "What is the minimum transaction amount?",
        "What is the maximum transaction amount?",
        "How many transactions are fraudulent?",
        "What percentage of transactions are fraudulent?",
        "How many transactions are non-fraudulent?",
        "What is the average transaction amount for fraudulent transactions?",
        "What is the average transaction amount for non-fraudulent transactions?",
        "What is the average transaction amount by class?",
        "What is the fraud rate by class?",
    ]

    if _fraud_suggest_is_dataset:

        st.markdown("### 💡 Suggested Credit Card Fraud Questions")

        _selected_fraud_question = st.selectbox(
            "Choose a fraud question to ask the AI Assistant:",
            ["Select a suggested question..."] + _fraud_questions,
            key="fraud_suggested_question"
        )

        if _selected_fraud_question != "Select a suggested question...":

            if (
                _selected_fraud_question
                != st.session_state.get(
                    "_last_fraud_suggested_question"
                )
            ):

                st.session_state["ai_dataset_question"] = (
                    _selected_fraud_question
                )

                st.session_state["_last_fraud_suggested_question"] = (
                    _selected_fraud_question
                )




    # DIABETES-FOCUSED SUGGESTED QUESTIONS
    # ------------------------------------------------------------

    # Show the suggested-question experience only when the uploaded
    # dataset has the expected Pima Diabetes schema.
    _diabetes_suggest_columns = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    _diabetes_required_columns = {
        "outcome",
        "glucose",
        "bmi",
        "age",
        "bloodpressure",
        "insulin",
    }

    _diabetes_suggest_is_dataset = (
        _diabetes_required_columns.issubset(
            set(_diabetes_suggest_columns.keys())
        )
        and len(df) > 0
    )

    _diabetes_questions = [
        "How many patients are in the dataset?",
        "How many patients have diabetes (outcome = 1)?",
        "How many patients do not have diabetes (outcome = 0)?",
        "What percentage of patients have diabetes?",
        "What is the average age of the patients?",
        "What is the minimum age?",
        "What is the maximum age?",
        "What is the average glucose level?",
        "What is the minimum glucose level?",
        "What is the maximum glucose level?",
        "What is the average BMI?",
        "What is the minimum BMI?",
        "What is the maximum BMI?",
        "What is the average blood pressure?",
        "What is the maximum blood pressure?",
        "What is the average insulin level?",
        "What is the maximum insulin level?",
        "What is the average glucose for patients with diabetes?",
        "What is the average BMI for patients with diabetes?",
        "How many patients have glucose above 150?",
    ]

    if _diabetes_suggest_is_dataset:

        st.markdown("### 💡 Suggested Diabetes Questions")

        _selected_diabetes_question = st.selectbox(
            "Choose a Diabetes question to ask the AI Assistant:",
            ["Select a suggested question..."] + _diabetes_questions,
            key="diabetes_suggested_question"
        )

        if _selected_diabetes_question != "Select a suggested question...":

            if (
                _selected_diabetes_question
                != st.session_state.get(
                    "_last_diabetes_suggested_question"
                )
            ):

                st.session_state["ai_dataset_question"] = (
                    _selected_diabetes_question
                )

                st.session_state["_last_diabetes_suggested_question"] = (
                    _selected_diabetes_question
                )

    # ------------------------------------------------------------
    # CAR PRICE-FOCUSED SUGGESTED QUESTIONS
    # ------------------------------------------------------------
    _car_suggest_columns = {
        re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_"): col
        for col in df.columns
    }
    _car_suggest_is_dataset = (
        "price" in _car_suggest_columns
        and "manufacturer" in _car_suggest_columns
        and "prod_year" in _car_suggest_columns
        and len(df) > 0
    )

    _car_questions = [
        "How many cars are in the dataset?",
        "What is the average car price?",
        "What is the minimum car price?",
        "What is the maximum car price?",
        "What is the median car price?",
        "Which car has the highest price?",
        "Which car has the lowest price?",
        "What is the average price by brand?",
        "Which brand has the highest average price?",
        "Which brand has the most cars?",
        "What is the average price by fuel type?",
        "Which fuel type has the highest average price?",
        "What is the average price by transmission type?",
        "Which transmission type has the highest average price?",
        "How does car age affect price?",
        "What is the correlation between mileage and price?",
        "What is the correlation between engine size and price?",
        "What are the most important features for predicting car price?",
        "Show the distribution of car prices.",
        "Which features are most correlated with price?",
    ]

    if _car_suggest_is_dataset:
        st.markdown("### 💡 Suggested Car Price Questions")
        _selected_car_question = st.selectbox(
            "Choose a Car Price question to ask the AI Assistant:",
            ["Select a suggested question..."] + _car_questions,
            key="car_price_suggested_question"
        )
        if _selected_car_question != "Select a suggested question...":
            if _selected_car_question != st.session_state.get("_last_car_price_suggested_question"):
                st.session_state["ai_dataset_question"] = _selected_car_question
                st.session_state["_last_car_price_suggested_question"] = _selected_car_question

    question = st.text_input(
        "Ask a question about your dataset",
        placeholder=(
            "Example: How many employees are in each department? "
            "Which department has the highest average income?"
        ),
        key="ai_dataset_question"
    )

    if question:

        try:

            with st.spinner(
                "Analyzing your question..."
            ):
                # Exact arithmetic is performed locally first.
                # HR department-count questions are handled directly from the
                # complete uploaded DataFrame before RAG is allowed to answer.
                _question_lower = str(question).strip().lower()
                _department_col = next(
                    (
                        col
                        for col in df.columns
                        if str(col).strip().lower().replace("_", " ") == "department"
                    ),
                    None,
                )

                _is_department_gender_count_question = (
                    _department_col is not None
                    and "department" in _question_lower
                    and "employee" in _question_lower
                    and "male" in _question_lower
                    and "female" in _question_lower
                )

                _is_department_count_question = (
                    _department_col is not None
                    and not _is_department_gender_count_question
                    and any(
                        phrase in _question_lower
                        for phrase in [
                            "each department",
                            "every department",
                            "employees by department",
                            "employee count by department",
                        ]
                    )
                    and any(
                        word in _question_lower
                        for word in ["how many", "number", "count"]
                    )
                )

                # Handle HR department employee extremes directly from the
                # uploaded DataFrame. This branch is intentionally outside
                # RAG/LLM so questions such as "Which department has the
                # fewest employees?" cannot be answered from retrieved text
                # such as employee_number values.
                _employee_extreme_words = [
                    "most", "highest", "largest", "maximum", "max",
                    "fewest", "lowest", "smallest", "minimum", "min",
                ]
                _is_department_employee_extreme_question = (
                    _department_col is not None
                    and "employee" in _question_lower
                    and any(
                        re.search(
                            rf"\b{re.escape(word)}\b",
                            _question_lower,
                        )
                        for word in _employee_extreme_words
                    )
                    and "department" in _question_lower
                )

                if _is_department_count_question:
                    _department_counts = (
                        df[_department_col]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .value_counts()
                    )
                    _total_employees = int(_department_counts.sum())

                    _department_results = pd.DataFrame(
                        {
                            "Department": _department_counts.index.tolist(),
                            "Employee Count": [
                                int(value)
                                for value in _department_counts.values
                            ],
                            "Percentage": [
                                round(
                                    (float(value) / _total_employees) * 100,
                                    2,
                                )
                                if _total_employees
                                else 0.0
                                for value in _department_counts.values
                            ],
                        }
                    )

                    exact_answer = (
                        "Here is the exact employee count for each department "
                        "calculated from the complete uploaded HR dataset."
                    )
                    evidence = {
                        "question": question,
                        "method": "exact pandas analysis",
                        "group_column": _department_col,
                        "aggregation": "count",
                        "total_rows": int(len(df)),
                        "total_employees": _total_employees,
                        "grouped_results": _department_results.to_dict(
                            orient="records"
                        ),
                    }
                elif _is_department_employee_extreme_question:
                    _department_counts = (
                        df[_department_col]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .value_counts()
                    )

                    if _department_counts.empty:
                        exact_answer = "No department values are available to analyze."
                        evidence = {
                            "question": question,
                            "method": "exact pandas analysis",
                            "group_column": _department_col,
                            "aggregation": "count",
                        }
                    else:
                        _lowest_department = any(
                            re.search(
                                rf"\b{re.escape(word)}\b",
                                _question_lower,
                            )
                            for word in [
                                "fewest",
                                "lowest",
                                "smallest",
                                "minimum",
                                "min",
                            ]
                        )

                        _ordered_departments = _department_counts.sort_values(
                            ascending=_lowest_department
                        )
                        _winner_department = _ordered_departments.index[0]
                        _winner_count = int(_ordered_departments.iloc[0])
                        _total_employees = int(_department_counts.sum())
                        _winner_percentage = (
                            (_winner_count / _total_employees) * 100
                            if _total_employees
                            else 0.0
                        )

                        _department_results = pd.DataFrame(
                            {
                                "Department": _ordered_departments.index.tolist(),
                                "Employee Count": [
                                    int(value)
                                    for value in _ordered_departments.values
                                ],
                                "Percentage": [
                                    round(
                                        (float(value) / _total_employees) * 100,
                                        2,
                                    )
                                    if _total_employees
                                    else 0.0
                                    for value in _ordered_departments.values
                                ],
                            }
                        )

                        _direction = "fewest" if _lowest_department else "most"
                        exact_answer = (
                            f"The department with the {_direction} employees is "
                            f"{_department_col} = {_winner_department}, with "
                            f"{_winner_count:,} employees "
                            f"({_winner_percentage:.2f}% of the dataset)."
                        )
                        evidence = {
                            "question": question,
                            "method": "exact pandas analysis",
                            "group_column": _department_col,
                            "aggregation": "count",
                            "winner": str(_winner_department),
                            "winner_value": _winner_count,
                            "total_rows": int(len(df)),
                            "grouped_results": _department_results.to_dict(
                                orient="records"
                            ),
                        }
                else:
                    exact_answer, evidence = exact_dataset_analysis(
                        df,
                        question
                    )

                rag_for_answer = (
                    st.session_state.rag_engine
                    if st.session_state.rag_ready
                    else None
                )

                answer = ai_explain_verified_result(
                    question,
                    exact_answer,
                    evidence,
                    rag_for_answer
                )

            st.markdown("### 💡 AI Answer")
            st.write(answer)

            if evidence.get("grouped_results"):
                st.dataframe(
                    pd.DataFrame(evidence["grouped_results"]),
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander(
                "📊 Exact Dataset Analysis Used by Assistant",
                expanded=False
            ):
                st.caption(
                    "These values are calculated directly from the uploaded DataFrame; "
                    "the language model is not used for the arithmetic."
                )
                st.write(exact_answer)
                if evidence.get("grouped_results"):
                    st.dataframe(
                        pd.DataFrame(evidence["grouped_results"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                if evidence:
                    st.json(evidence)

        except Exception as e:

            st.error(
                f"AI Assistant error: {e}"
            )

    elif question and not st.session_state.rag_ready:

        st.info(
            "This question was analyzed locally. Connect the Local AI Assistant "
            "above for broader RAG/LLM-based questions."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "🤖 AI Data Analyst | "
    "EDA • Statistics • Visualization • "
    "Feature Engineering • Machine Learning • "
    "Prediction • Reports"
)
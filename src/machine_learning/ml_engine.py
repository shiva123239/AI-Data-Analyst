"""
Machine Learning Engine
AI Data Analyst Advanced

Complete ML engine for the existing Streamlit application.

Supports:
- Automatic classification/regression detection
- Binary and multiclass classification
- Regression
- Leakage and identifier protection
- Safe preprocessing with sklearn Pipelines
- Optional treatment of domain-invalid zero values
- Multiple model training and comparison
- Cross-validation
- Classification metrics including ROC-AUC
- Confusion matrix
- Prediction probabilities
- Feature importance / coefficients
- Backward-compatible public function names
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# ============================================================
# CONSTANTS
# ============================================================

RANDOM_STATE = 42

# In the common diabetes dataset, zero is not a meaningful measured
# value for these physiological variables.  Do NOT include pregnancies:
# zero pregnancies is valid.
ZERO_AS_MISSING_COLUMNS = {
    "glucose",
    "bloodpressure",
    "blood_pressure",
    "skinthickness",
    "skin_thickness",
    "insulin",
    "bmi",
}

CLASSIFICATION_TARGET_NAMES = {
    "target",
    "label",
    "class",
    "outcome",
    "diagnosis",
    "prediction",
    "result",
}


# ============================================================
# HELPERS
# ============================================================

def _normalized_name(value: Any) -> str:
    """Normalize a column name for safe comparisons."""
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _empty_importance() -> pd.DataFrame:
    return pd.DataFrame(columns=["feature", "importance"])


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _classification_cv_folds(y: pd.Series, maximum: int = 5) -> int:
    """Choose a safe number of stratified CV folds."""
    counts = y.value_counts(dropna=False)
    if counts.empty:
        return 0
    minimum_class_count = int(counts.min())
    return min(maximum, minimum_class_count)


def _regression_cv_folds(y: pd.Series, maximum: int = 5) -> int:
    """Choose a safe number of regression CV folds."""
    return min(maximum, len(y))


def _get_prediction_probabilities(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray | None, list[Any] | None]:
    """Return probability predictions when the fitted model supports them."""
    model = pipeline.named_steps.get("model")

    if not hasattr(model, "predict_proba"):
        return None, None

    try:
        probabilities = pipeline.predict_proba(X_test)
        classes = getattr(model, "classes_", None)
        classes_list = classes.tolist() if classes is not None else None
        return np.asarray(probabilities), classes_list
    except Exception:
        return None, None


def _classification_auc(
    y_true: pd.Series,
    probabilities: np.ndarray | None,
    classes: list[Any] | None,
) -> float | None:
    """Calculate ROC-AUC safely for binary or multiclass predictions."""
    if probabilities is None or classes is None:
        return None

    try:
        if len(classes) == 2:
            # sklearn accepts the probability of the positive class.
            return float(roc_auc_score(y_true, probabilities[:, 1]))

        if len(classes) > 2:
            return float(
                roc_auc_score(
                    y_true,
                    probabilities,
                    multi_class="ovr",
                    average="weighted",
                )
            )
    except (ValueError, TypeError, IndexError):
        return None

    return None


def _classification_selection_score(metrics: dict[str, Any]) -> float:
    """Select models using F1, with ROC-AUC as a useful tie-breaker."""
    f1 = metrics.get("f1_score")
    auc = metrics.get("roc_auc")

    if f1 is None:
        return -np.inf

    return float(f1) + (0.001 * float(auc) if auc is not None else 0.0)


# ============================================================
# PROBLEM TYPE DETECTION
# ============================================================

def detect_problem_type(
    df: pd.DataFrame,
    target_column: str,
) -> str:
    """
    Automatically determine whether the selected target is a
    classification or regression target.

    Numeric targets with a small number of unique values are treated
    as classification. Common target names such as ``outcome`` are
    also treated as classification when they contain a small number
    of classes.
    """

    if df is None or df.empty:
        raise ValueError("The dataset is empty.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    target = df[target_column].dropna()

    if target.empty:
        raise ValueError(
            "The selected target column contains no usable values."
        )

    if (
        pd.api.types.is_object_dtype(target)
        or pd.api.types.is_categorical_dtype(target)
        or pd.api.types.is_bool_dtype(target)
    ):
        return "classification"

    if pd.api.types.is_numeric_dtype(target):
        unique_count = int(target.nunique())
        normalized_target = _normalized_name(target_column)

        # Binary / small-cardinality numeric targets are classification.
        if unique_count <= 10:
            return "classification"

        # Common target names can be used as a stronger hint, but only
        # when the number of classes is still reasonably small.
        if (
            normalized_target in CLASSIFICATION_TARGET_NAMES
            and unique_count <= 20
        ):
            return "classification"

        return "regression"

    return "classification"


# ============================================================
# IDENTIFIER REMOVAL
# ============================================================

def remove_identifier_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Remove obvious identifier columns from ML features."""

    X = X.copy()
    columns_to_remove: list[Any] = []

    exact_identifier_names = {
        "id",
        "customer_id",
        "employee_id",
        "user_id",
        "record_id",
        "account_id",
        "transaction_id",
        "product_id",
        "order_id",
        "member_id",
        "student_id",
        "application_id",
        "loan_id",
        "policy_id",
    }

    for column in X.columns:
        column_lower = _normalized_name(column)

        if (
            column_lower in exact_identifier_names
            or column_lower.endswith("_id")
        ):
            columns_to_remove.append(column)
            continue

        if pd.api.types.is_object_dtype(X[column]) or pd.api.types.is_categorical_dtype(X[column]):
            unique_count = X[column].nunique(dropna=True)
            total_count = len(X)

            if (
                total_count > 0
                and unique_count >= 0.95 * total_count
                and unique_count > 20
            ):
                columns_to_remove.append(column)

    if columns_to_remove:
        X = X.drop(
            columns=list(dict.fromkeys(columns_to_remove)),
            errors="ignore",
        )

    return X


# ============================================================
# TARGET DERIVED FEATURE REMOVAL
# ============================================================

def remove_target_derived_features(
    X: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """Remove the selected target and obvious target-derived features."""

    X = X.copy()
    target = _normalized_name(target_column)

    derived_suffixes = {
        "log",
        "log1p",
        "ln",
        "sqrt",
        "square",
        "squared",
        "cube",
        "cubed",
        "abs",
        "absolute",
        "rank",
        "zscore",
        "z_score",
        "standardized",
        "standardised",
        "scaled",
        "normalized",
        "normalised",
        "outlier",
        "is_outlier",
        "missing",
        "is_missing",
        "bin",
        "binned",
        "bucket",
        "bucketed",
        "category",
        "categorical",
    }

    columns_to_remove: list[Any] = []

    for column in X.columns:
        column_lower = _normalized_name(column)

        if column_lower == target:
            columns_to_remove.append(column)
            continue

        if column_lower.startswith(target + "_"):
            remainder = column_lower[len(target) + 1 :]
            first_part = remainder.split("_")[0]

            if (
                remainder in derived_suffixes
                or first_part in derived_suffixes
                or any(
                    remainder.startswith(suffix + "_")
                    for suffix in derived_suffixes
                )
            ):
                columns_to_remove.append(column)
                continue

        for prefix in derived_suffixes:
            if column_lower == f"{prefix}_{target}":
                columns_to_remove.append(column)
                break

    if columns_to_remove:
        X = X.drop(
            columns=list(dict.fromkeys(columns_to_remove)),
            errors="ignore",
        )

    return X


# ============================================================
# DIABETES / DOMAIN ZERO HANDLING
# ============================================================

def replace_domain_invalid_zeros(X: pd.DataFrame) -> pd.DataFrame:
    """
    Replace known domain-invalid zero measurements with NaN.

    This is intentionally limited to well-known medical measurement
    column names. Zero pregnancies, for example, remains a valid value.
    The replacement itself does not learn anything from the test set;
    the imputer in the sklearn pipeline learns replacement values from
    training data only.
    """

    X = X.copy()

    for column in X.columns:
        if _normalized_name(column) in ZERO_AS_MISSING_COLUMNS:
            if pd.api.types.is_numeric_dtype(X[column]):
                X[column] = X[column].replace(0, np.nan)

    return X


# ============================================================
# COMBINED FEATURE CLEANING
# ============================================================

def prepare_ml_features(
    df: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """Prepare features by removing target/leakage/identifier columns."""

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    X = df.drop(columns=[target_column]).copy()
    X = remove_target_derived_features(X, target_column)
    X = remove_identifier_columns(X)
    X = replace_domain_invalid_zeros(X)

    return X


# ============================================================
# PREPROCESSING
# ============================================================

def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a safe numeric/categorical preprocessing transformer."""

    numeric_features = (
        X.select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    categorical_features = (
        X.select_dtypes(
            include=["object", "category", "bool"]
        )
        .columns
        .tolist()
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_features:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError(
            "No usable numeric or categorical features are available."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_classification_models() -> dict[str, Any]:
    """Return robust classification models."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def get_regression_models() -> dict[str, Any]:
    """Return robust regression models."""

    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            random_state=RANDOM_STATE,
            max_depth=10,
            min_samples_leaf=3,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


# ============================================================
# CLASSIFICATION
# ============================================================

def train_classification_models(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Train, cross-validate and compare classification models."""

    data = df.copy()

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    data = data.dropna(subset=[target_column])

    if data.empty:
        raise ValueError(
            "No usable rows remain after removing missing target values."
        )

    y = data[target_column].copy()

    if y.nunique() < 2:
        raise ValueError(
            "Classification requires at least two target classes."
        )

    X = prepare_ml_features(data, target_column)

    if X.shape[1] == 0:
        raise ValueError(
            "No usable feature columns remain after removing identifiers "
            "and target-derived features."
        )

    class_counts = y.value_counts()
    stratify_value = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify_value,
    )

    preprocessor = create_preprocessor(X_train)
    models = get_classification_models()

    results: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, Pipeline] = {}
    cv_folds = _classification_cv_folds(y_train)

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        try:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            probabilities, classes = _get_prediction_probabilities(
                pipeline, X_test
            )

            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
            recall = recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
            auc = _classification_auc(
                y_test,
                probabilities,
                classes,
            )

            # Binary-specific metrics are useful for diabetes/outcome.
            sensitivity: float | None = None
            specificity: float | None = None

            if len(classes or []) == 2:
                try:
                    cm_binary = confusion_matrix(
                        y_test,
                        predictions,
                        labels=classes,
                    )
                    if cm_binary.shape == (2, 2):
                        tn, fp, fn, tp = cm_binary.ravel()
                        sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
                        specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
                except Exception:
                    pass

            metrics: dict[str, Any] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": auc,
                "sensitivity": sensitivity,
                "specificity": specificity,
            }

            if cv_folds >= 2:
                try:
                    cv = StratifiedKFold(
                        n_splits=cv_folds,
                        shuffle=True,
                        random_state=RANDOM_STATE,
                    )
                    cv_scores = cross_val_score(
                        pipeline,
                        X_train,
                        y_train,
                        cv=cv,
                        scoring="f1_weighted",
                        n_jobs=1,
                    )
                    metrics["cv_f1_mean"] = float(np.mean(cv_scores))
                    metrics["cv_f1_std"] = float(np.std(cv_scores))
                    metrics["cv_scores"] = [float(v) for v in cv_scores]
                except Exception as exc:
                    metrics["cv_error"] = str(exc)
            else:
                metrics["cv_f1_mean"] = None
                metrics["cv_f1_std"] = None
                metrics["cv_scores"] = []

            metrics["selection_score"] = _classification_selection_score(metrics)

            results[model_name] = metrics
            trained_models[model_name] = pipeline

        except Exception as exc:
            results[model_name] = {"error": str(exc)}

    valid_results = {
        name: metrics
        for name, metrics in results.items()
        if "f1_score" in metrics
    }

    if not valid_results:
        raise RuntimeError(
            "None of the classification models could be trained."
        )

    best_model_name = max(
        valid_results,
        key=lambda name: valid_results[name]["selection_score"],
    )

    best_model = trained_models[best_model_name]
    best_predictions = best_model.predict(X_test)
    best_probabilities, best_classes = _get_prediction_probabilities(
        best_model, X_test
    )
    cm = confusion_matrix(
        y_test,
        best_predictions,
        labels=best_classes if best_classes is not None else None,
    )

    return {
        "problem_type": "classification",
        "target_column": target_column,
        "results": results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "confusion_matrix": cm.tolist(),
        "classes": (
            best_classes
            if best_classes is not None
            else y.unique().tolist()
        ),
        "X_test": X_test,
        "y_test": y_test,
        "predictions": best_predictions,
        "probabilities": best_probabilities,
        "prediction_probabilities": best_probabilities,
        "cv_folds": cv_folds,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_columns": X.columns.tolist(),
        "class_distribution": {
            str(k): int(v) for k, v in y.value_counts().items()
        },
    }


# ============================================================
# REGRESSION
# ============================================================

def train_regression_models(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Train, cross-validate and compare regression models."""

    data = df.copy()

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    data = data.dropna(subset=[target_column])

    if data.empty:
        raise ValueError(
            "No usable rows remain after removing missing target values."
        )

    y = data[target_column].copy()

    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError(
            "Regression requires a numeric target column."
        )

    if y.nunique() < 2:
        raise ValueError(
            "Regression requires at least two different target values."
        )

    X = prepare_ml_features(data, target_column)

    if X.shape[1] == 0:
        raise ValueError(
            "No usable feature columns remain after removing identifiers "
            "and target-derived features."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    preprocessor = create_preprocessor(X_train)
    models = get_regression_models()

    results: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, Pipeline] = {}
    cv_folds = _regression_cv_folds(y_train)

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        try:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)

            metrics: dict[str, Any] = {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "r2_score": float(r2),
            }

            if cv_folds >= 2:
                try:
                    cv = KFold(
                        n_splits=cv_folds,
                        shuffle=True,
                        random_state=RANDOM_STATE,
                    )
                    cv_scores = cross_val_score(
                        pipeline,
                        X_train,
                        y_train,
                        cv=cv,
                        scoring="r2",
                        n_jobs=1,
                    )
                    metrics["cv_r2_mean"] = float(np.mean(cv_scores))
                    metrics["cv_r2_std"] = float(np.std(cv_scores))
                    metrics["cv_scores"] = [float(v) for v in cv_scores]
                except Exception as exc:
                    metrics["cv_error"] = str(exc)
            else:
                metrics["cv_r2_mean"] = None
                metrics["cv_r2_std"] = None
                metrics["cv_scores"] = []

            results[model_name] = metrics
            trained_models[model_name] = pipeline

        except Exception as exc:
            results[model_name] = {"error": str(exc)}

    valid_results = {
        name: metrics
        for name, metrics in results.items()
        if "r2_score" in metrics
    }

    if not valid_results:
        raise RuntimeError(
            "None of the regression models could be trained."
        )

    best_model_name = max(
        valid_results,
        key=lambda name: valid_results[name]["r2_score"],
    )

    best_model = trained_models[best_model_name]
    predictions = best_model.predict(X_test)

    return {
        "problem_type": "regression",
        "target_column": target_column,
        "results": results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "cv_folds": cv_folds,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_columns": X.columns.tolist(),
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    trained_pipeline: Pipeline,
) -> pd.DataFrame:
    """
    Extract feature importance from tree models or absolute coefficients
    from linear/logistic models.
    """

    if trained_pipeline is None:
        return _empty_importance()

    try:
        model = trained_pipeline.named_steps["model"]
        preprocessor = trained_pipeline.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return _empty_importance()

    importances: np.ndarray | None = None

    if hasattr(model, "feature_importances_"):
        try:
            importances = np.asarray(model.feature_importances_, dtype=float)
        except Exception:
            importances = None

    elif hasattr(model, "coef_"):
        try:
            coefficients = np.asarray(model.coef_, dtype=float)
            if coefficients.ndim == 1:
                importances = np.abs(coefficients)
            else:
                importances = np.mean(np.abs(coefficients), axis=0)
        except Exception:
            importances = None

    if importances is None or len(feature_names) != len(importances):
        return _empty_importance()

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    return (
        importance_df
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================
# PREDICTION HELPER
# ============================================================

def predict_with_model(
    trained_pipeline: Pipeline,
    input_data: pd.DataFrame,
) -> dict[str, Any]:
    """Generate prediction and probabilities using a trained pipeline."""

    if trained_pipeline is None:
        raise ValueError("A trained model is required.")

    if input_data is None or input_data.empty:
        raise ValueError("Prediction input is empty.")

    prepared = replace_domain_invalid_zeros(input_data.copy())
    predictions = trained_pipeline.predict(prepared)

    probabilities, classes = _get_prediction_probabilities(
        trained_pipeline,
        prepared,
    )

    result: dict[str, Any] = {
        "predictions": predictions,
        "probabilities": probabilities,
        "classes": classes,
    }

    if probabilities is not None and classes is not None:
        result["probability"] = probabilities.max(axis=1)

    return result


# ============================================================
# MAIN ML RUNNER
# ============================================================

def run_machine_learning(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str = "auto",
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Main entry point for the Machine Learning module."""

    if df is None or df.empty:
        raise ValueError("The dataset is empty.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if problem_type is None or str(problem_type).strip().lower() == "auto":
        problem_type = detect_problem_type(df, target_column)

    problem_type = str(problem_type).strip().lower()

    if problem_type in {"classification", "classify", "classification_problem"}:
        return train_classification_models(
            df=df,
            target_column=target_column,
            test_size=test_size,
        )

    if problem_type in {"regression", "regress", "regression_problem"}:
        return train_regression_models(
            df=df,
            target_column=target_column,
            test_size=test_size,
        )

    raise ValueError(
        "Unsupported problem type. Use 'auto', 'classification', "
        "or 'regression'."
    )

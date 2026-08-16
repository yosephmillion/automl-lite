import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder
)
from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import (
    SVC,
    SVR
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# JSON SAFE
# ============================================================

def safe_value(value):

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isfinite(value):
            return float(value)
        return None

    return value


# ============================================================
# ENCODER
# ============================================================

def create_preprocessor(X, scale_numeric=False):

    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    transformers = []

    # -----------------------------
    # Numeric
    # -----------------------------

    if numeric_columns:

        numeric_steps = [
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]

        if scale_numeric:

            numeric_steps.append(
                (
                    "scaler",
                    StandardScaler()
                )
            )

        from sklearn.pipeline import Pipeline as SkPipeline

        numeric_pipeline = SkPipeline(
            numeric_steps
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )
        )

    # -----------------------------
    # Categorical
    # -----------------------------

    if categorical_columns:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# ============================================================
# MODEL FACTORY
# ============================================================

def get_model(model_name, task):

    if task == "Classification":

        models = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=2000
                ),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42,
                    max_depth=10
                ),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=150,
                    random_state=42,
                    n_jobs=-1
                ),

            "KNN":
                KNeighborsClassifier(
                    n_neighbors=5
                ),

            "Naive Bayes":
                GaussianNB(),

            "SVM":
                SVC(
                    kernel="rbf"
                )
        }

    else:

        models = {

            "Linear Regression":
                LinearRegression(),

            "Decision Tree Regressor":
                DecisionTreeRegressor(
                    random_state=42,
                    max_depth=10
                ),

            "Random Forest Regressor":
                RandomForestRegressor(
                    n_estimators=150,
                    random_state=42,
                    n_jobs=-1
                ),

            "SVR":
                SVR(),

            "Ridge":
                Ridge(),

            "Lasso":
                Lasso(
                    max_iter=5000
                )
        }

    if model_name not in models:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return models[model_name]


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    df,
    target,
    task,
    model_name
):

    if df is None or df.empty:

        raise ValueError(
            "Dataset is empty."
        )

    if target not in df.columns:

        raise ValueError(
            f"Target '{target}' does not exist."
        )

    # Make a copy
    data = df.copy()

    # Remove rows where target is missing
    data = data.dropna(
        subset=[target]
    )

    if len(data) < 10:

        raise ValueError(
            "Not enough valid rows to train the model."
        )

    X = data.drop(
        columns=[target]
    )

    y = data[target]

    # ========================================================
    # Remove obviously useless ID-like columns
    # ========================================================

    columns_to_remove = []

    for column in X.columns:

        name = str(column).lower()

        unique_ratio = (
            X[column].nunique(dropna=True)
            / max(len(X), 1)
        )

        if (
            name == "id"
            or name.endswith("_id")
            or name.startswith("id_")
        ):

            columns_to_remove.append(column)

        elif unique_ratio > 0.98:

            columns_to_remove.append(column)

    if columns_to_remove:

        X = X.drop(
            columns=columns_to_remove
        )

    if X.shape[1] == 0:

        raise ValueError(
            "No usable feature columns remain after removing ID-like columns."
        )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if task == "Classification":

        y = y.astype(str)

        unique_classes = y.nunique()

        if unique_classes < 2:

            raise ValueError(
                "Classification target must contain at least two different classes."
            )

        if unique_classes > 50:

            raise ValueError(
                "This target has too many unique classes for a classification task. "
                "Choose another target or switch to Regression."
            )

        # Remove extremely rare classes
        class_counts = y.value_counts()

        valid_classes = class_counts[
            class_counts >= 2
        ].index

        mask = y.isin(valid_classes)

        X = X.loc[mask]
        y = y.loc[mask]

        if y.nunique() < 2:

            raise ValueError(
                "After removing invalid target values, fewer than two classes remain."
            )

        # Stratified split
        try:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )

        except ValueError:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42
            )

        # Scale for models that benefit from it
        scale_models = {
            "Logistic Regression",
            "KNN",
            "SVM"
        }

        preprocessor = create_preprocessor(
            X_train,
            scale_numeric=(
                model_name in scale_models
            )
        )

        model = get_model(
            model_name,
            task
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        cm = confusion_matrix(
            y_test,
            predictions
        )

        labels = sorted(
            [str(x) for x in y.unique()]
        )

        metrics = {

            "Accuracy":
                safe_value(accuracy),

            "Precision":
                safe_value(precision),

            "Recall":
                safe_value(recall),

            "F1 Score":
                safe_value(f1),

            "Confusion Matrix":
                cm.tolist(),

            "Labels":
                labels
        }

    # ========================================================
    # REGRESSION
    # ========================================================

    else:

        # Convert target to numeric
        y = pd.to_numeric(
            y,
            errors="coerce"
        )

        valid_mask = y.notna()

        X = X.loc[
            valid_mask
        ]

        y = y.loc[
            valid_mask
        ]

        if len(y) < 10:

            raise ValueError(
                "Not enough numeric target values for regression."
            )

        if y.nunique() < 2:

            raise ValueError(
                "Regression target must contain more than one unique value."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        scale_models = {
            "SVR",
            "Ridge",
            "Lasso"
        }

        preprocessor = create_preprocessor(
            X_train,
            scale_numeric=(
                model_name in scale_models
            )
        )

        model = get_model(
            model_name,
            task
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mse
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        metrics = {

            "MAE":
                safe_value(mae),

            "MSE":
                safe_value(mse),

            "RMSE":
                safe_value(rmse),

            "R² Score":
                safe_value(r2)
        }

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    feature_importance = None

    try:

        fitted_model = pipeline.named_steps[
            "model"
        ]

        fitted_preprocessor = pipeline.named_steps[
            "preprocessor"
        ]

        feature_names = (
            fitted_preprocessor
            .get_feature_names_out()
        )

        if hasattr(
            fitted_model,
            "feature_importances_"
        ):

            importances = (
                fitted_model.feature_importances_
            )

            feature_importance = {

                str(name): safe_value(value)

                for name, value in zip(
                    feature_names,
                    importances
                )
            }

        elif hasattr(
            fitted_model,
            "coef_"
        ):

            coefficients = fitted_model.coef_

            if coefficients.ndim > 1:

                coefficients = np.mean(
                    np.abs(coefficients),
                    axis=0
                )

            else:

                coefficients = np.abs(
                    coefficients
                )

            feature_importance = {

                str(name): safe_value(value)

                for name, value in zip(
                    feature_names,
                    coefficients
                )
            }

    except Exception as e:

        print(
            "Feature importance unavailable:",
            e
        )

    print("\n")
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print("Task   :", task)
    print("Model  :", model_name)
    print("Target :", target)
    print("Metrics:", metrics)
    print("=" * 60)

    return {

        "model": pipeline,

        "metrics": metrics,

        "y_test": y_test,

        "predictions": predictions,

        "feature_importance":
            feature_importance
    }
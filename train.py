import pandas as pd
import os
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("sqlite:///mlflow.db")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)



def train_models(
    data_path="data/processed/iris.csv",
    target_column="target"
):
    """
    Train multiple ML models using an automatic preprocessing pipeline.
    """

    os.makedirs("logs", exist_ok=True)

    # ============================================================
    # 1. LOAD DATASET
    # ============================================================

    df = pd.read_csv(data_path)

    print("\n========== DATASET ==========")
    print(f"Dataset: {data_path}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # ============================================================
    # 2. VALIDATE TARGET
    # ============================================================

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            f"not found in dataset."
        )

    # ============================================================
    # 3. SPLIT FEATURES AND TARGET
    # ============================================================

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    print(f"\nTarget column: {target_column}")
    print(f"Features: {X.columns.tolist()}")

    # ============================================================
    # 4. IDENTIFY COLUMN TYPES
    # ============================================================

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\nNumerical columns:")
    print(numerical_columns)

    print("\nCategorical columns:")
    print(categorical_columns)

    # ============================================================
    # 5. NUMERICAL PREPROCESSING
    # ============================================================

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # ============================================================
    # 6. CATEGORICAL PREPROCESSING
    # ============================================================

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # ============================================================
    # 7. COMBINE PREPROCESSING
    # ============================================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    # ============================================================
    # 8. TRAIN / TEST SPLIT
    # ============================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ============================================================
    # 9. DEFINE MODELS
    # ============================================================

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=200
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            random_state=42
        )
    }

    # ============================================================
    # 10. MLFLOW EXPERIMENT
    # ============================================================

    mlflow.set_experiment(
        "AutoMLOps_Experiment"
    )

    results = {}

    # ============================================================
    # 11. TRAIN EACH MODEL
    # ============================================================

    for name, model in models.items():

        print(f"\n{'=' * 50}")
        print(f"Training: {name}")
        print(f"{'=' * 50}")

        model_pipeline = Pipeline(
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

        with mlflow.start_run(
            run_name=name,
            nested=False
        ) as run:

            # ----------------------------------------------------
            # TRAIN
            # ----------------------------------------------------

            model_pipeline.fit(
                X_train,
                y_train
            )

            # ----------------------------------------------------
            # PREDICT
            # ----------------------------------------------------

            preds = model_pipeline.predict(
                X_test
            )

            # ----------------------------------------------------
            # METRICS
            # ----------------------------------------------------

            acc = accuracy_score(
                y_test,
                preds
            )

            prec = precision_score(
                y_test,
                preds,
                average="weighted",
                zero_division=0
            )

            rec = recall_score(
                y_test,
                preds,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                preds,
                average="weighted",
                zero_division=0
            )

            # ----------------------------------------------------
            # MLFLOW PARAMETERS
            # ----------------------------------------------------

            mlflow.log_param(
                "model_type",
                name
            )

            mlflow.log_param(
                "target_column",
                target_column
            )

            mlflow.log_param(
                "numerical_features",
                str(numerical_columns)
            )

            mlflow.log_param(
                "categorical_features",
                str(categorical_columns)
            )

            # ----------------------------------------------------
            # MLFLOW METRICS
            # ----------------------------------------------------

            mlflow.log_metric(
                "accuracy",
                acc
            )

            mlflow.log_metric(
                "precision",
                prec
            )

            mlflow.log_metric(
                "recall",
                rec
            )

            mlflow.log_metric(
                "f1_score",
                f1
            )

            # ----------------------------------------------------
            # SAVE MODEL TO MLFLOW
            # ----------------------------------------------------

            mlflow.sklearn.log_model(
                model_pipeline,
                "model"
            )

            # ----------------------------------------------------
            # STORE RESULTS
            # ----------------------------------------------------

            results[name] = {
                "model": model_pipeline,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "run_id": run.info.run_id
            }

            # ----------------------------------------------------
            # PRINT RESULTS
            # ----------------------------------------------------

            print(f"\n{name} Results:")

            print(
                f"Accuracy : {acc:.4f}"
            )

            print(
                f"Precision: {prec:.4f}"
            )

            print(
                f"Recall   : {rec:.4f}"
            )

            print(
                f"F1 Score : {f1:.4f}"
            )

    return results
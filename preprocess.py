import os
import pandas as pd
from sklearn.datasets import load_iris


# ============================================================
# EXISTING PIPELINE FUNCTION
# ============================================================

def load_and_save_data():
    """
    Existing AutoMLOps dataset loader.

    Currently loads the Iris dataset and saves it as a CSV.
    This function is intentionally preserved so that the
    existing main_pipeline.py continues to work.
    """

    # Ensure processed folder exists
    os.makedirs("data/processed", exist_ok=True)

    iris = load_iris()

    X = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    y = pd.Series(
        iris.target,
        name="target"
    )

    df = pd.concat([X, y], axis=1)

    df.to_csv(
        "data/processed/iris.csv",
        index=False
    )

    print("Dataset saved to data/processed/iris.csv")

    return df


# ============================================================
# GENERIC DATASET FUNCTIONS
# ============================================================

def load_dataset(file_path):
    """
    Load any CSV dataset into a pandas DataFrame.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    print("\n========== DATASET LOADED ==========")
    print(f"File: {file_path}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


def analyze_dataset(df, target_column=None):
    """
    Analyze the structure of a tabular dataset.

    Detects:
    - number of rows
    - number of columns
    - numerical columns
    - categorical columns
    - missing values
    - target column
    """

    print("\n========== DATASET ANALYSIS ==========")

    # Basic information
    print(f"\nRows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Column names
    print("\nColumns:")
    for column in df.columns:
        print(f" - {column}")

    # Data types
    print("\nData Types:")
    print(df.dtypes)

    # Missing values
    print("\nMissing Values:")
    missing_values = df.isnull().sum()

    for column, count in missing_values.items():
        print(f" - {column}: {count}")

    # Numerical columns
    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Categorical columns
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # If target is supplied, don't treat it as a feature
    if target_column in numerical_columns:
        numerical_columns.remove(target_column)

    if target_column in categorical_columns:
        categorical_columns.remove(target_column)

    print("\nNumerical Feature Columns:")
    print(numerical_columns)

    print("\nCategorical Feature Columns:")
    print(categorical_columns)

    # Target information
    if target_column is not None:

        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found."
            )

        print(f"\nTarget Column: {target_column}")

        print("\nTarget Data Type:")
        print(df[target_column].dtype)

        print("\nTarget Distribution:")
        print(df[target_column].value_counts())

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "missing_values": missing_values.to_dict(),
        "target_column": target_column
    }


def split_features_target(df, target_column):
    """
    Separate a dataset into:

    X = input features
    y = target variable
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    print("\n========== FEATURE / TARGET SPLIT ==========")

    print(f"Target Column: {target_column}")

    print(
        f"Feature Columns: {X.columns.tolist()}"
    )

    print(f"Feature Shape: {X.shape}")
    print(f"Target Shape: {y.shape}")

    return X, y


# ============================================================
# TEST GENERIC DATASET ANALYSIS
# ============================================================

if __name__ == "__main__":

    # Step 1:
    # Generate the existing Iris dataset
    load_and_save_data()

    # Step 2:
    # Load the generated CSV using the new generic loader
    df = load_dataset(
        "data/processed/iris.csv"
    )

    # Step 3:
    # Analyze the dataset
    analyze_dataset(
        df,
        target_column="target"
    )

    # Step 4:
    # Separate features and target
    X, y = split_features_target(
        df,
        target_column="target"
    )

    print("\n========== FINAL RESULT ==========")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
import os
import pandas as pd
from sklearn.datasets import load_iris

def load_and_save_data():
    # Ensure processed folder exists
    os.makedirs("data/processed", exist_ok=True)

    iris = load_iris()

    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")

    df = pd.concat([X, y], axis=1)

    df.to_csv("data/processed/iris.csv", index=False)

    print("Dataset saved to data/processed/iris.csv")

if __name__ == "__main__":
    load_and_save_data()
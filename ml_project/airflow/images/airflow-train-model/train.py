import os

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle


def save_object(obj, path: str, filename: str):
    filepath = os.path.join(path, filename)

    with open(filepath, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_model(model: str):
    if model == "logreg":
        return LogisticRegression(solver="liblinear", C=1.0, penalty="l1", max_iter=10000)
    elif model == "rf":
        return RandomForestClassifier(n_estimators=100, max_depth=3)
    else:
        return RandomForestClassifier()


@click.command("train")
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--model-name")
def train_model(input_dir: str, model_dir: str, model_name: str):
    if not input_dir or not model_dir or not model_name:
        return
    X_train = pd.read_csv(os.path.join(input_dir, "X_train_transformed.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"))
    model = get_model(model_name)
    model.fit(X_train, y_train)
    save_object(model, model_dir, f"model_{model_name}.pkl")


if __name__ == "__main__":
    train_model()

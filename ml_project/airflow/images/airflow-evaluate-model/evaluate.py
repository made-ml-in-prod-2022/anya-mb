import os
import pickle
import json

import click
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def load_object(filepath: str):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj


@click.command("evaluate")
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--model-name")
def evaluate_model(input_dir: str, model_dir: str, model_name: str):
    model = load_object(os.path.join(model_dir, f"model_{model_name}.pkl"))
    X_test = pd.read_csv(os.path.join(input_dir, "X_test_transformed.csv"))
    y_test = pd.read_csv(os.path.join(input_dir, "y_test.csv"))
    y_predict = model.predict(X_test)
    y_predict_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    roc_auc = roc_auc_score(y_test, y_predict_proba[:, 1])
    with open(os.path.join(model_dir, f"metric_{model_name}.json"), 'w') as f:
        json.dump({
            "accuracy": accuracy,
            "roc_auc": roc_auc
        }, f)


if __name__ == "__main__":
    evaluate_model()

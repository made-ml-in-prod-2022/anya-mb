import os
import pickle
from pathlib import Path
import numpy as np

import click
import pandas as pd
from constants import TARGET_COLUMN

from feature_transform import ExperimentalTransformer


def load_object(filepath: str):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj


@click.command("make_prediction")
@click.option("--input-datafile")
@click.option("--path-to-model")
@click.option("--output-dir")
def make_prediction(input_datafile: str, path_to_model: str, output_dir: str):
    model = load_object(path_to_model)
    data = pd.read_csv(input_datafile)

    X = data.drop(columns=[TARGET_COLUMN], axis=1)

    model_dir = Path(path_to_model).parent
    transformer = ExperimentalTransformer(model_dir)
    transformer.load_from_file()

    X_transformed = transformer.transform(X)

    y_predict = model.predict(X_transformed)

    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, "predictions.csv"), y_predict, delimiter=',')


if __name__ == "__main__":
    make_prediction()

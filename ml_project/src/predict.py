from os.path import join
from dataclasses import dataclass

from utils import read_data, load_object

import numpy as np
import pandas as pd


@dataclass
class PredictingParams:
    model_path: str
    data_path: str
    log_path: str
    predictions_path: str
    target_column: str


def save_predictions(predictions: np.ndarray, path_to_save: str):
    predictions_filepath = join(path_to_save, 'predictions.csv')
    pd.Series(predictions, name="prediction").to_csv(predictions_filepath)


def predict_pipeline(cfg: PredictingParams):
    print('Prediction pipeline')

    data = read_data(cfg.data_path)
    if cfg.target_column in data.columns:
        data = data.drop(columns=cfg.target_column)

    print('Reading data')

    model_filepath = join(cfg.model_path, 'model.pkl')
    model = load_object(model_filepath)

    predictions = model.predict(data)

    save_predictions(predictions, cfg.predictions_path)
    print('Predictions saved')

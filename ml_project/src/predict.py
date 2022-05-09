from os.path import join
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils import read_data, load_object


logger = logging.getLogger(__name__)


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
    logger.info(f'Start predict pipeline with parameters: {cfg}')

    data = read_data(cfg.data_path)
    if cfg.target_column in data.columns:
        data = data.drop(columns=cfg.target_column)

    logger.info(f'data.shape is {data.shape}')

    model_filepath = join(cfg.model_path, 'model.pkl')
    model = load_object(model_filepath)
    logger.info(f'Model is {model}')

    predictions = model.predict(data)
    logger.info(f'predictions.shape is {predictions.shape}')

    save_predictions(predictions, cfg.predictions_path)
    logger.info('Predictions saved')

import pytest
import pandas as pd
from feature_transform import ExperimentalTransformer


CAT_COLUMN_NAME = "legs_count"
NUM_COLUMN_NAME = "age"
CAT_FEATS_THRESHOLD = 3
MODEL_PATH = "/tmp"


@pytest.fixture
def transformer():
    return ExperimentalTransformer(
        MODEL_PATH,
        cat_feats_threshold=CAT_FEATS_THRESHOLD)


@pytest.fixture
def fit_data():
    data = [
        [10, 2],
        [20, 2],
        [30, 1],
    ]
    return pd.DataFrame(data, columns=[NUM_COLUMN_NAME, CAT_COLUMN_NAME])


@pytest.fixture
def transform_input_data():
    data = [
        [5, 2],
        [15, 1],
        [20, 1],
    ]
    return pd.DataFrame(data, columns=[NUM_COLUMN_NAME, CAT_COLUMN_NAME])

import logging
import pandas as pd
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from os.path import join

from sklearn.exceptions import NotFittedError

from utils import save_object, load_object

DEFAULT_THRESHOLD_FOR_CATEGORICAL = 6
logger = logging.getLogger(__name__)


@dataclass
class TransformerState:
    cat_features: list
    num_features: list
    ohe: OneHotEncoder
    scaler: StandardScaler


class ExperimentalTransformer(BaseEstimator):
    _TRANSFORMER_FILE_NAME = "transformer.pkl"

    def __init__(self, model_path, cat_feats_threshold=DEFAULT_THRESHOLD_FOR_CATEGORICAL):
        self.fitted = False
        self.n_unique_values_threshold_for_categorical = cat_feats_threshold
        self.state: TransformerState = None
        self.model_path = model_path

    def fit(self, X):
        cat_feats, num_feats = self._get_categorical_and_numerical_features_labels(X)

        ohe = OneHotEncoder()
        ohe.fit(X[cat_feats])

        scaler = StandardScaler()
        scaler.fit(X[num_feats])

        self.fitted = True
        self.state = TransformerState(cat_feats, num_feats, ohe, scaler)

        return self

    def save_to_file(self):
        if self.state is None:
            raise Exception("state is empty. Fit transformer first")
        save_object(self.state, self.model_path, self._TRANSFORMER_FILE_NAME)

    def load_from_file(self):
        self.fitted = True
        self.state = load_object(join(self.model_path, self._TRANSFORMER_FILE_NAME))

    def transform(self, X_):
        if (not self.fitted) or (not self.state):
            raise NotFittedError

        X = X_.copy()

        if not self.state:
            self.load_from_file()

        for col in self.state.cat_features + self.state.num_features:
            if col not in X:
                raise AttributeError

        hot = pd.DataFrame(self.state.ohe.transform(X[self.state.cat_features]).toarray())
        scal = pd.DataFrame(self.state.scaler.transform(X[self.state.num_features]))
        full_transformed = pd.concat([hot, scal], axis=1)

        full_transformed.columns = [f'col_{i}' for i in range(full_transformed.shape[1])]

        return full_transformed

    def fit_transform(self, X_):
        self.fit(X_)
        full_transformed = self.transform(X_)
        return full_transformed

    def _get_categorical_and_numerical_features_labels(self, X):
        cols_df = X.nunique().to_frame().reset_index()
        cols_df = cols_df.rename(columns={'index': 'column_name', 0: 'n_unique'})

        all_feats = cols_df['column_name'].tolist()

        cat_feats = cols_df[cols_df['n_unique'] < self.n_unique_values_threshold_for_categorical] \
            ['column_name'].tolist()
        num_feats = [x for x in all_feats if x not in cat_feats]

        return cat_feats, num_feats

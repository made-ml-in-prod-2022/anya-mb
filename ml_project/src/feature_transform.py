import logging
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from os.path import join
import pickle

from sklearn.exceptions import NotFittedError

from utils import save_object, load_object

DEFAULT_THRESHOLD_FOR_CATEGORICAL = 6
TRANSFORMER_STATE_FILENAME = 'transformer_state.pkl'
logger = logging.getLogger(__name__)


class TransformerState:
    def __init__(self, cat_feats, num_feats, scaler, ohe):
        self.cat_feats = cat_feats
        self.num_feats = num_feats
        self.scaler = scaler
        self.ohe = ohe


class ExperimentalTransformer(BaseEstimator):
    def __init__(self, model_path, cat_feats_threshold=DEFAULT_THRESHOLD_FOR_CATEGORICAL):
        self.fitted = False
        self.n_unique_values_threshold_for_categorical = cat_feats_threshold
        self.transformer_state = None
        self.model_path = model_path

    def fit(self, X):
        feats = self._get_categorical_and_numerical_features_labels(X)
        cat_feats, num_feats = feats

        ohe = OneHotEncoder()
        ohe.fit(X[cat_feats])

        scaler = StandardScaler()
        scaler.fit(X[num_feats])

        self.transformer_state = TransformerState(cat_feats, num_feats, scaler, ohe)

        self.fitted = True

        return self

    def save_to_file(self):
        if self.transformer_state is None:
            raise Exception("At least one param to save is None")
        save_object(self.transformer_state, self.model_path, TRANSFORMER_STATE_FILENAME)

    def transform(self, X_):
        if self.fitted is False:
            raise NotFittedError

        X = X_.copy()

        for col in self.transformer_state.cat_feats + self.transformer_state.num_feats:
            if col not in X:
                raise AttributeError

        if not self.transformer_state:
            self.transformer_state = load_object(join(self.model_path, TRANSFORMER_STATE_FILENAME))

        hot = pd.DataFrame(self.transformer_state.ohe.transform(X[self.transformer_state.cat_feats]).toarray())
        scal = pd.DataFrame(self.transformer_state.scaler.transform(X[self.transformer_state.num_feats]))
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

        return (cat_feats, num_feats)

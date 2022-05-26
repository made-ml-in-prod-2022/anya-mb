import logging
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from os.path import join
import pickle

from sklearn.exceptions import NotFittedError

from utils import save_object, load_object


N_UNIQUE_VALUES_THRESHOLD_FOR_CATEGORICAL = 6
logger = logging.getLogger(__name__)


class ExperimentalTransformer(BaseEstimator):
    def __init__(self, model_path):
        self.fitted = False
        self.N_UNIQUE_VALUES_THRESHOLD_FOR_CATEGORICAL = 6
        self.cat_feats = None
        self.num_feats = None
        self.scaler = None
        self.ohe = None
        self.model_path = model_path

    def fit(self, X):
        feats = self.get_categorical_and_numerical_features_labels(X)
        self.cat_feats, self.num_feats = feats

        self.ohe = OneHotEncoder()
        self.ohe.fit(X[self.cat_feats])
        save_object(self.ohe, self.model_path, 'ohe.pkl')

        self.scaler = StandardScaler()
        self.scaler.fit(X[self.num_feats])
        save_object(self.scaler, self.model_path, 'scaler.pkl')

        self.fitted = True

        return self

    def transform(self, X_):
        if (not self.fitted) or (not self.scaler) or (not self.ohe):
            raise NotFittedError

        X = X_.copy()

        for col in self.cat_feats + self.num_feats:
            if col not in X:
                raise AttributeError

        if not self.ohe:
            self.ohe = load_object(join(self.model_path, 'ohe.pkl'))

        if not self.scaler:
            self.scaler = load_object(join(self.model_path, 'scaler.pkl'))

        hot = pd.DataFrame(self.ohe.transform(X[self.cat_feats]).toarray())
        scal = pd.DataFrame(self.scaler.transform(X[self.num_feats]))
        full_transformed = pd.concat([hot, scal], axis=1)

        full_transformed.columns = [f'col_{i}' for i in range(full_transformed.shape[1])]

        return full_transformed

    def fit_transform(self, X_):
        self.fit(X_)
        full_transformed = self.transform(X_)
        return full_transformed

    def get_categorical_and_numerical_features_labels(self, X):
        cols_df = X.nunique().to_frame().reset_index()
        cols_df = cols_df.rename(columns={'index': 'column_name', 0: 'n_unique'})

        all_feats = cols_df['column_name'].tolist()

        cat_feats = cols_df[cols_df['n_unique'] < N_UNIQUE_VALUES_THRESHOLD_FOR_CATEGORICAL] \
            ['column_name'].tolist()
        num_feats = [x for x in all_feats if x not in cat_feats]

        return (cat_feats, num_feats)
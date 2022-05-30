from os.path import join

from feature_transform import ExperimentalTransformer
from train import MODEL_NAME
from utils import load_object


class ModelWrapper:
    def __init__(self, model_path):
        self.model_path = model_path

        model_filepath = join(model_path, MODEL_NAME)
        self.model = load_object(model_filepath)

        self.transformer = ExperimentalTransformer(model_path)
        self.transformer.load_from_file()

    def predict(self, df):
        data_transformed = self.transformer.transform(df)
        prediction = self.model.predict(data_transformed)
        return prediction

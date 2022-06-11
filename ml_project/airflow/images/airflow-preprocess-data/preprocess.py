import os
import pandas as pd
from feature_transform import ExperimentalTransformer
import click


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--output-model-dir")
def preprocess_data(input_dir: str, output_dir: str, output_model_dir: str):
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))

    transformer = ExperimentalTransformer(output_model_dir)

    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)

    X_train_transformed.to_csv(os.path.join(output_dir, "X_train_transformed.csv"), index=False)
    X_test_transformed.to_csv(os.path.join(output_dir, "X_test_transformed.csv"), index=False)
    transformer.save_to_file()


if __name__ == "__main__":
    preprocess_data()

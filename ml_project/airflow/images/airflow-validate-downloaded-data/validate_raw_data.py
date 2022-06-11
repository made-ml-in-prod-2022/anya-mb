import click
import pandas as pd


@click.command("validate")
@click.option("--path")
def validate_data(path: str):
    df = pd.read_csv(path + "/heart_cleveland_upload.csv")
    assert df.shape[0] > 100
    assert set(df.columns) == {"age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak",
                               "slope", "ca", "thal", "condition"}
    # extra validations here

if __name__ == "__main__":
    validate_data()

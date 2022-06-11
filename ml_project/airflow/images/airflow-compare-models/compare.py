import json
import os

import click


@click.command("compare")
@click.option("--model-dir")
def compare_models(model_dir: str):
    metric_files = [filename for filename in os.listdir(model_dir)
                    if filename.startswith("metric_")]
    metrics = [
        {
            "filename": filename,
            "metrics": json.load(open(os.path.join(model_dir, filename), 'r'))
        } for filename in metric_files
    ]

    # that key allows us to convert several metrics into one number, which is used to define the best model
    key = lambda x: x["metrics"]["accuracy"] + x["metrics"]["roc_auc"]
    sorted_metrics = sorted(metrics, key=key, reverse=True)

    best_model = sorted_metrics[0]

    with open(os.path.join(model_dir, "best_model.json"), "w") as f:
        json.dump({
            "best_model_filename": best_model["filename"]
        }, f)


if __name__ == "__main__":
    compare_models()

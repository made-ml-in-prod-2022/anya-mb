  --------------------------
  Machine Learning Project
  --------------------------

This repository contains an implementation of Random Forest Classifier
and Logistic Regression on heart disease dataset, complete with training, prediction,
configuration, and testing features.

# Installation

``` bash
pip install -r requirements.txt
```

# Usage

## Random Forest Classifier

-   **Train**:

    ``` bash
    python3 src/main.py mode=train model=rf
    ```

-   **Predict**:

    ``` bash
    python3 src/main.py mode=predict model=rf
    ```

## Logistic Regression

-   **Train**:

    ``` bash
    python3 src/main.py mode=train model=logreg
    ```

-   **Predict**:

    ``` bash
    python3 src/main.py mode=predict model=logreg
    ```

# Configuration

If you need to run the model on different data, modify the \'data_path\'
in [config/mode/predict.yaml](https://github.com/made-ml-in-prod-2022/anya-mb/blob/main/ml_project/config/mode/predict.yaml). The model can be changed
using the \'model_path\' parameter in
[config/general/general.yaml](https://github.com/made-ml-in-prod-2022/anya-mb/blob/main/ml_project/config/general/general.yaml).

# Tests

Run the tests using the following command:

``` bash
pytest -v src/tests.py
```

# Features

-   Modular project structure.
-   Comprehensive logging.
-   Configurable training using either JSON or YAML. Example
    configurations are provided.
-   Data classes are employed instead of plain dictionaries for
    configurations.
-   Custom transformer example: [Guide on Custom Transformers in
    Scikit-learn](https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156)
-   Dependencies are pinned for consistency.
-   Continuous Integration (CI) is set up using GitHub actions.

# Additional Features

-   Configuration with [Hydra](https://hydra.cc/docs/intro/).

# Acknowledgements

This project is a culmination of many learnings and experiments.
Feedback and contributions are welcome!

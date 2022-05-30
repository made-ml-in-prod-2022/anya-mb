import numpy as np


def test_fit_and_transform(transformer, fit_data, transform_input_data):
    transformer.fit(fit_data)
    assert transformer.fitted
    assert transformer.state.cat_features == ["legs_count"]
    assert transformer.state.num_features == ["age"]

    transformed_data = transformer.transform(transform_input_data)
    expected_transform_output_data = np.array([
        [0, 1, -1.837117],
        [1, 0, -0.612372],
        [1, 0, 0],
    ])
    assert np.allclose(transformed_data, expected_transform_output_data)


def test_fit_transform(transformer, transform_input_data):
    transformed_data = transformer.fit_transform(transform_input_data)
    expected_transform_output_data = np.array([
        [0, 1, -1.33630621],
        [1, 0, 0.26726124],
        [1, 0, 1.06904497],
    ])
    assert np.allclose(transformed_data, expected_transform_output_data)

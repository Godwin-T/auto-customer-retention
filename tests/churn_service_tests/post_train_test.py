import numpy as np
from churn_service_tests.utils import load_processed_data, predict


def test_data_invariace():

    gender_mapping = {"male": "female", "female": "male"}
    input_data, _ = load_processed_data()
    valid_output = predict(input_data)

    input_data["gender"] = input_data["gender"].map(
        gender_mapping
    )  # Introduce variance to gender
    variance_output = predict(input_data)  # Call prediction function
    no_match_output = np.sum((valid_output == variance_output))

    assert no_match_output == valid_output.shape[0]


def test_model_prediction():

    """Tests model prediction accuracy."""
    input_data, expected_output = load_processed_data()  # Prepare test input
    output = predict(input_data)  # Call prediction function

    assert output.shape == expected_output.shape  # test_output_size
    assert output.dtype == "int64"  # test_output_dtype

    output, expected_output = np.array(output), np.array(expected_output)
    accuracy = np.sum(output == expected_output) / expected_output.shape[0]
    assert accuracy >= 0.7

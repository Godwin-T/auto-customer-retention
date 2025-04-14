import pytest
import json
import os
from unittest.mock import patch, MagicMock
import pandas as pd
from deploy import create_app, load_config, get_default_config


# Set testing environment variable
os.environ["TESTING"] = "True"

# Mock the configuration
@pytest.fixture
def mock_config():
    return {
        "base": {"random_state": 42},
        "database": {
            "customer": {
                "database_path": "sqlite:///data/customer.db",
                "prediction_logs": "predictions",
            },
            "tracking": {"tracking_url": "sqlite:///mlruns.db"},
        },
        "data": {"process": {"path": "./data/processed"}},
        "hyperparameters": {"trees": 100},
    }


# Setup the Flask client for testing with mocked configuration
@pytest.fixture
def client(mock_config):
    with patch("deploy.initialize_mlflow"), patch("deploy.get_db_engine"):
        app = create_app(mock_config)
        with app.test_client() as client:
            yield client


# Test the predict endpoint
@pytest.fixture
def mock_data():
    return pd.DataFrame(
        {
            "customerid": [1, 2, 3],
            "churn": ["yes", "no", "yes"],
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4.0, 5.0, 6.0],
        }
    )


@patch("deploy.pull_data_from_db")
@patch("deploy.load_model")
@patch("deploy.input_data_processing")
@patch("deploy.output_data_processing")
@patch("deploy.push_data_to_db")
def test_predict(
    mock_push_data_to_db,
    mock_output_data_processing,
    mock_input_data_processing,
    mock_load_model,
    mock_pull_data_from_db,
    client,
    mock_data,
):

    # Mock the data returned by the pull_data_from_db function
    mock_pull_data_from_db.return_value = mock_data

    # Mock the model prediction result
    mock_model = MagicMock()
    mock_model.predict.return_value = [1, 0, 1]  # Mock prediction values
    mock_load_model.return_value = mock_model

    # Mock the input and output processing
    mock_input_data_processing.return_value = (mock_data["churn"], [{}], mock_data)
    mock_output_data_processing.return_value = mock_data

    # Call the /predict endpoint
    response = client.get("/predict")

    # Assert response status and success message
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert (
        response_json["Response"]
        == "The predictions have successfully been saved to database"
    )

    # Verify all the mocks were called
    mock_pull_data_from_db.assert_called_once()
    mock_load_model.assert_called_once()
    mock_input_data_processing.assert_called_once()
    mock_output_data_processing.assert_called_once()
    mock_push_data_to_db.assert_called_once()


# Test the deploy-auto endpoint
@patch("deploy.automated_deployment_workflow")
def test_deploy_auto(mock_deploy_auto, client):
    # Mock the automated_deployment_workflow function
    mock_deploy_auto.return_value = {"deployed": True, "Response": "Deployed"}

    # Call the /deploy-auto endpoint
    response = client.get("/deploy-auto")

    # Assert response status and success message
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json["Response"] == "Deployed"
    assert response_json["deployed"] is True

    # Verify the mock was called
    mock_deploy_auto.assert_called_once()


# Test the configuration loading
def test_load_config():
    # Test with environment variable not set
    with patch("deploy.os.getenv", return_value=None), patch(
        "deploy.open", side_effect=FileNotFoundError
    ):
        config = load_config()
        assert config == get_default_config()

    # Test with environment variable set but file not found
    with patch("deploy.os.getenv", return_value="nonexistent.yaml"), patch(
        "deploy.open", side_effect=FileNotFoundError
    ):
        config = load_config()
        assert config == get_default_config()


# Test the automated deployment workflow
@patch("deploy.initialize_mlflow")
@patch("deploy.extract_top_model")
@patch("deploy.register_model")
@patch("deploy.get_db_engine")
@patch("deploy.load_validation_data")
@patch("deploy.load_model")
@patch("mlflow.sklearn.load_model")
@patch("deploy.compare_models")
@patch("deploy.model_transition")
def test_automated_deployment_workflow(
    mock_transition,
    mock_compare,
    mock_load_mlflow,
    mock_load_model,
    mock_load_validation,
    mock_db_engine,
    mock_register,
    mock_extract,
    mock_init_mlflow,
    mock_config,
):
    from deploy import automated_deployment_workflow

    # Mock the client
    mock_client = MagicMock()
    mock_init_mlflow.return_value = mock_client

    # Mock extract top model
    mock_extract.return_value = ["run123"]

    # Mock model registration
    mock_register.return_value = "1"

    # Mock validation data
    mock_load_validation.return_value = ([], [])

    # Mock models
    production_model = MagicMock()
    challenger_model = MagicMock()
    mock_load_model.return_value = production_model
    mock_load_mlflow.return_value = challenger_model

    # Mock comparison results
    mock_compare.return_value = (True, {"production": {}, "challenger": {}})

    # Run the workflow
    result = automated_deployment_workflow(mock_config)

    # Check result
    assert result["deployed"] is True
    assert "model_version" in result
    assert result["Response"] == "Deployed"

    # Verify mocks were called
    mock_init_mlflow.assert_called_once()
    mock_extract.assert_called_once()
    mock_register.assert_called_once()
    mock_load_validation.assert_called_once()
    mock_load_model.assert_called_once()
    mock_load_mlflow.assert_called_once()
    mock_compare.assert_called_once()
    mock_transition.assert_called_once()

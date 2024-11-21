import os
import boto3
import pickle
import pandas as pd
import pkg_resources
from moto import mock_aws
from src.backend.churn_guard import (
    load_data_with_path,
    input_data_processing,
    output_data_processing,
    load_model_from_s3,
    upload_prediction_to_s3,
)

import warnings

warnings.filterwarnings("ignore")


def test_load_data():

    """Tests loading a CSV dataset."""

    data_path = "./tests/sample_data/raw_data.csv"
    input_data = pd.read_csv(data_path)
    input_data_dict = input_data.to_dict()

    data = load_data_with_path.fn(input_data_dict)  # Function to load data
    assert isinstance(data, pd.DataFrame)  # Check if data is a DataFrame
    assert data.shape[1] == 21  # Verify expected dimensions

    data_columns = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]

    input_data_dtypes = [
        "object",
        "object",
        "int64",
        "object",
        "object",
        "int64",
        "object",
        "object",
        "object",
        "object",
        "object",
        "object",
        "object",
        "object",
        "object",
        "object",
        "object",
        "object",
        "float64",
        "float64",
        "object",
    ]

    test_data_dtype = data.dtypes.apply(lambda x: x.name).tolist()

    assert data.columns.tolist() == data_columns  # Check column names
    assert test_data_dtype == input_data_dtypes  # Check columns dtypes


def test_input_data_processing():

    """Tests processing a Dataframe"""

    data_path = "./tests/sample_data/raw_data.csv"
    input_data = pd.read_csv(data_path)
    data = input_data_processing.fn(input_data.copy())

    assert input_data.shape[1] == data.shape[1]

    null_values = data.isnull().values.any()
    assert null_values == False, "Data consist od null values"

    # Check if any string in the list contains a space
    my_list = [str(col) for col in input_data.columns.tolist()]
    has_space = all(" " in s for s in my_list)
    assert has_space == False

    duplicates = all(data.duplicated())
    assert duplicates == False, "Data contain duplicates"


def test_output_data_processing():

    data_path = "./tests/sample_data/raw_data.csv"
    input_data = pd.read_csv(data_path)
    prediction = [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]
    output_data = output_data_processing.fn(input_data, prediction)

    assert isinstance(output_data, pd.DataFrame)


@mock_aws
def test_load_model_from_s3_moto():
    # Set up the mock S3 service
    s3 = boto3.client("s3")
    s3.create_bucket(Bucket="test-bucket")

    # Upload a fake model to the mock S3 bucket
    fake_model = ""
    model = {"model": fake_model}
    s3.put_object(Bucket="test-bucket", Key="model.pkl", Body=pickle.dumps(model))

    # Call the function that loads the model from S3
    loaded_model = load_model_from_s3.fn(
        s3, bucket_name="test-bucket", file_name="model.pkl"
    )

    # Assert the model was loaded correctly
    assert loaded_model == model, "Model not loading properly"


@mock_aws
def test_upload_file_to_s3():
    # Set up mock S3
    s3 = boto3.client("s3", region_name="us-east-1")
    bucket_name = "test-bucket"

    # Create a mock bucket
    s3.create_bucket(Bucket=bucket_name)

    # Simulate a file
    test_file_content = "this is a test file"
    test_file_path = "/tmp/test_file.pkl"

    # Write the test file to the local filesystem (simulating an actual file)
    with open(test_file_path, "wb") as f:
        pickle.dump(test_file_content, f)

    # Call the function to upload the file to S3
    upload_prediction_to_s3.fn(
        s3, test_file_path, bucket_name, "uploaded_test_file.pkl"
    )

    # Verify that the file exists in the mock S3 bucket
    response = s3.get_object(Bucket=bucket_name, Key="uploaded_test_file.pkl")
    stored_content = response["Body"].read()

    with open(test_file_path, "rb") as f:
        stored_content = pickle.load(f)

    assert stored_content == test_file_content, "File content not correct"

    # Clean up: remove the local test file
    os.remove(test_file_path)


def test_required_dependencies():

    with open("./src/backend/churn_guard/guard_deploy/requirements.txt") as f:
        required = f.read().splitlines()

    installed = [pkg.key for pkg in pkg_resources.working_set]

    missing = [pkg for pkg in required if pkg not in installed]

    assert len(missing) == 0, f"Missing dependencies: {', '.join(missing)}"

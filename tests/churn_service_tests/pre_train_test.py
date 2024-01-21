import pytest
import pandas as pd

from churn_service_tests.utils import load_raw_data, split_data


def test_load_data():

    """Tests loading a CSV dataset."""
    data = load_raw_data()  # Function to load data
    assert isinstance(data, pd.DataFrame)  # Check if data is a DataFrame
    assert data.shape == (7043, 21)  # Verify expected dimensions
    assert data.columns.tolist() == [
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
    ]  # Check column names


def test_dtypes():

    data = load_raw_data()
    data_types = data.dtypes
    data_types = [str(data_types[col]) for col in data.columns.to_list()]
    assert data_types == [
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
        "object",
        "object",
    ]


# def test_data_size():

#     data = load_raw_data()
#     no_records = data.shape[0]
#     assert no_records >= 5000


def test_data_leakage():

    (X_train, X_test) = split_data()
    concat_df = pd.concat([X_train, X_test])
    concat_df.drop_duplicates(inplace=True)
    assert concat_df.shape[0] == X_train.shape[0] + X_test.shape[0]

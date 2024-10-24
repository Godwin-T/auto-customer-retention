import pandas as pd
from app.churn_guard.utils.process_data import load_dataset, process_dataset


def test_load_dataset():

    input_data_path = "./sample_data/raw_data.csv"
    data = load_dataset(input_data_path)

    assert isinstance.fn(data, pd.DataFrame)
    assert data.shape == (11, 21)


def test_process_dataset():

    input_data_path = "./sample_data/raw_data.csv"
    data = load_dataset.fn(input_data_path)
    output_data = process_dataset.fn(data, target_column_name="churn")

    assert isinstance(output_data, pd.DataFrame)
    assert output_data["churn"].dtype == "int"
    assert output_data["totalcharges"].dtype == "float32"
    assert data.shape == (11, 21)

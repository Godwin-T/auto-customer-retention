import os
import time
import pytest
import shutil
import sqlite3

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from app.churn_guard.utils.evaluate import evaluate
from app.churn_guard.utils.modelhelper import save_model_to_dir
from app.churn_guard.train_pipeline.train import process_data
from app.churn_guard.utils.datahelper import (
    load_data_from_sqlite_db,
    load_data_from_mysql_db,
    create_mysql_database_table,
)


# Pytest fixture to create a temporary SQLite database
@pytest.fixture
def temp_sqlite_db():

    # Create a temporary SQLite database in the pytest-provided tmpdir
    tmpdir = "../tmp_db_dir"
    if not (os.path.exists(tmpdir)):
        os.mkdir(tmpdir)

    input_data_path = "./sample_data/raw_data.csv"
    tablename = "users"

    db_path = os.path.join(tmpdir, "test.db")
    conn = sqlite3.connect(db_path)

    df = pd.read_csv(input_data_path)
    df["log_time"] = time.time()

    df.to_sql(tablename, conn, if_exists="fail", index=False)
    conn.close()

    # Yield the temporary directory and db path for testing
    yield tmpdir, "test.db", df


@pytest.fixture(scope="session")
def db_engine():
    """Fixture to create a SQLAlchemy engine for an in-memory SQLite database."""
    engine = create_engine("sqlite:///:memory:", echo=True)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def temp_mysql_db(db_engine):

    engine = db_engine
    input_df_path = "./sample_data/processed_data.csv"
    input_df = pd.read_csv(input_df_path)

    create_mysql_database_table.fn(engine, input_df_path, tablename="test")
    output_df = load_data_from_mysql_db.fn(engine, tablename="test")

    Session = sessionmaker(bind=db_engine)
    session = Session()

    session.commit()

    yield session, output_df, input_df
    session.close()


# Test function for load_data_from_sqlite_db
def test_load_data_from_sqlite_db(temp_sqlite_db):
    tmpdir, dbname, expected_df = temp_sqlite_db
    db_directory = str(tmpdir)  # Convert tmpdir to string

    # Call the function to load data
    df = load_data_from_sqlite_db.fn(db_directory, dbname, "users")

    # Assert that the DataFrame returned by the function matches the expected DataFrame
    pd.testing.assert_frame_equal(df, expected_df)
    shutil.rmtree(tmpdir)


# Test function for load_data_from_sqlite_db
def test_load_data_from_myql_db(temp_mysql_db):

    _, output_df, expected_df = temp_mysql_db
    output_df.drop(["log_time"], inplace=True, axis=1)

    # Assert that the DataFrame returned by the function matches the expected DataFrame

    pd.testing.assert_frame_equal(output_df, expected_df)


def test_process_data():

    input_df_path = "./sample_data/processed_data.csv"
    input_df = pd.read_csv(input_df_path)

    features, target = process_data.fn(input_df, target_column="churn")

    assert features.shape == (7, 19)
    assert target.shape == (7,)


def test_model_training():

    input_df_path = "./sample_data/processed_data.csv"
    input_df = pd.read_csv(input_df_path)

    features, target = process_data(input_df, target_column="churn")
    features = features.to_dict(orient="records")

    lr_pipeline = make_pipeline(
        DictVectorizer(sparse=False), LogisticRegression(C=1)
    )  # make training pipeline

    lr_pipeline.fit(features, target)

    linear_model = lr_pipeline.named_steps["logisticregression"]
    coefficients = linear_model.coef_

    # Test if the model has been trained (check for model parameters)
    assert coefficients.shape[1] == 37, "Model output shape is incorrect"


def test_evaluate_model():

    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0]

    evaluation_results = evaluate(y_true, y_pred)

    assert evaluation_results["acc"] == 0.8, "Accuracy is out of range"
    assert evaluation_results["f1_score"] == 0.8, "f1_score is out of range"
    assert evaluation_results["precision"] == 1.0, "Precision is out of range"
    assert round(evaluation_results["recall"], 2) == 0.67, "Recall is out of range"


def test_model_saving():

    model_dir = "../models"
    model_name = "model.pkl"

    model_path = f"{model_dir}/{model_name}"

    assert save_model_to_dir.fn(model_name, model_path) == "Model saved successfully!"
    shutil.rmtree(model_dir)

import pytest
import pandas as pd
import sqlite3
from ingestion import ingest, process, create_app
import utils


# ---------------- Fixtures ----------------


@pytest.fixture
def dummy_dataframe():
    """Dummy dataframe representing raw customer data."""
    return pd.DataFrame(
        {
            "CustomerID": [1, 2],
            "TotalCharges": ["100.0", "200.0"],
            "Churn": ["Yes", "No"],
        }
    )


@pytest.fixture
def expected_processed_dataframe():
    """Expected processed dataframe after cleaning."""
    return pd.DataFrame({"totalcharges": [100.0, 200.0], "churn": [1, 0]})


@pytest.fixture
def dummy_config():
    """Mock config for the ingestion module."""
    return {
        "database": {"db_path": ":memory:"},
        "data": {
            "raw_data": {"path": "dummy.csv", "name": "raw_table"},
            "processed_data": {
                "name": "processed_table",
                "dropcols": [],
                "targetcolumn": "churn",
            },
        },
    }


@pytest.fixture
def db_engine():
    """SQLite in-memory DB engine."""
    return sqlite3.connect(":memory:")


@pytest.fixture(autouse=True)
def patch_load_config(monkeypatch, dummy_config):
    """Patch load_config to return dummy config for all tests."""
    monkeypatch.setattr("ingestion.load_config", lambda: dummy_config)


@pytest.fixture
def seed_raw_table(db_engine, dummy_dataframe):
    """Create and seed raw_table in the in-memory DB."""
    dummy_dataframe.to_sql("raw_table", db_engine, index=False, if_exists="replace")
    return db_engine


# ---------------- Tests ----------------


def test_push_and_pull_raw_data(monkeypatch, dummy_dataframe, db_engine):
    """Push dummy CSV data to DB and read it back."""
    monkeypatch.setattr(pd, "read_csv", lambda path: dummy_dataframe)

    utils.push_data_to_db(db_engine, "raw_table", dfpath="dummy.csv")
    df = utils.pull_data_from_db(db_engine, "raw_table")

    assert not df.empty
    expected_columns = ["CustomerID", "TotalCharges", "Churn", "date"]
    assert list(df.columns) == expected_columns
    assert df.shape == (2, 4)


def test_process_dataframe(dummy_dataframe, expected_processed_dataframe):
    """Check if process_dataframe transforms as expected."""
    processed = utils.process_dataframe(
        dummy_dataframe, target_col="Churn", drop_cols=["customerid"]  # use lowercase
    )

    pd.testing.assert_frame_equal(
        processed.reset_index(drop=True), expected_processed_dataframe
    )


def test_ingest_function(monkeypatch, dummy_dataframe, db_engine):
    """Run the ingest step and ensure no errors."""
    monkeypatch.setattr(pd, "read_csv", lambda path: dummy_dataframe)

    ingest(db_engine)

    df = utils.pull_data_from_db(db_engine, "raw_table")
    assert not df.empty
    assert "CustomerID" in df.columns


def test_process_function(seed_raw_table, expected_processed_dataframe):
    """Run the process function on real DB data."""
    process(seed_raw_table)

    df = utils.pull_data_from_db(seed_raw_table, "processed_table")
    assert not df.empty
    assert "churn" in df.columns
    assert "totalcharges" in df.columns

    # Compare values
    pd.testing.assert_frame_equal(
        df[["totalcharges", "churn"]].reset_index(drop=True),
        expected_processed_dataframe,
    )


def test_flask_process_endpoint(
    monkeypatch, seed_raw_table, expected_processed_dataframe
):
    """Test the /process Flask endpoint end-to-end."""
    app = create_app(engine=seed_raw_table)
    test_client = app.test_client()

    response = test_client.get("/process")

    assert response.status_code == 200

    # Validate that the data was actually processed in DB
    df = utils.pull_data_from_db(seed_raw_table, "processed_table")
    pd.testing.assert_frame_equal(
        df[["totalcharges", "churn"]].reset_index(drop=True),
        expected_processed_dataframe,
    )

import os
import json
import yaml
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, mock_open
import sqlite3

# Import the functions to test from your utils module
# These imports should work now without loading the config at import time
from .utils import (
    connect_sqlite,
    pull_data_from_db,
    validate_config,
    evaluate_model,
    load_config,
    initialize_config,
    get_customer_data_path,
)

# Sample config for testing
@pytest.fixture
def sample_config():
    return {
        "base": {"random_state": 42, "developer": "test", "artifact_path": "models"},
        "database": {
            "customer": {"database_path": "test.db"},
            "tracking": {
                "tracking_url": "sqlite:///mlflow.db",
                "experiment_name": "test",
            },
        },
        "data": {"test_size": 0.2},
        "hyperparameters": {
            "linear_model": {"min_c": 1, "max_c": 10, "interval": 1},
            "tree_models": {
                "criterion": ["gini", "entropy"],
                "min_depth": 1,
                "max_depth": 10,
                "min_sample_split": 2,
                "max_sample_split": 10,
                "min_sample_leaf": 1,
                "max_sample_leaf": 5,
            },
        },
    }


@pytest.fixture
def setup_config_env(sample_config, tmp_path):
    """Set up environment with temporary config file"""
    # Create a temporary config file
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config, f)

    # Patch environment variable to point to this config
    with patch.dict(os.environ, {"config_path": str(config_file)}):
        yield str(config_file)


@pytest.fixture
def mock_db():
    """Mock SQLite connection"""
    with patch("sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        yield mock_conn


# Tests for utils.py functions


def test_load_config(setup_config_env):
    """Test loading configuration from file"""
    config_path = setup_config_env
    config = load_config(config_path)

    assert config is not None
    assert "base" in config
    assert "random_state" in config["base"]
    assert config["base"]["random_state"] == 42


def test_initialize_config(setup_config_env):
    """Test initializing configuration from environment variable"""
    config = initialize_config()

    assert config is not None
    assert "base" in config
    assert config["base"]["random_state"] == 42


def test_get_customer_data_path(setup_config_env):
    """Test getting customer data path from config"""
    # Initialize config first
    initialize_config()

    # Now get the customer data path
    path = get_customer_data_path()
    assert path == "test.db"


def test_connect_sqlite(mock_db):
    """Test that connect_sqlite connects to the database"""
    conn = connect_sqlite("test.db")
    assert conn is not None
    sqlite3.connect.assert_called_once_with("test.db", check_same_thread=False)


def test_connect_sqlite_exception():
    """Test connect_sqlite handles exceptions properly"""
    with patch("sqlite3.connect", side_effect=Exception("Connection error")):
        with pytest.raises(Exception):
            connect_sqlite("test.db")


def test_validate_config_valid(sample_config):
    """Test validate_config with valid config"""
    assert validate_config(sample_config) is True


def test_validate_config_missing_section():
    """Test validate_config with missing section"""
    config = {"base": {}, "data": {}, "hyperparameters": {}}  # Missing database section
    with pytest.raises(ValueError, match="Missing required section: database"):
        validate_config(config)


def test_validate_config_missing_random_state():
    """Test validate_config with missing random_state"""
    config = {
        "base": {},  # Missing random_state
        "database": {},
        "data": {},
        "hyperparameters": {},
    }
    with pytest.raises(ValueError, match="Missing random_state in base configuration"):
        validate_config(config)


def test_pull_data_from_db():
    """Test pull_data_from_db function"""
    # Mock the database engine and pandas read_sql
    mock_engine = MagicMock()
    expected_data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

    with patch("pandas.read_sql", return_value=expected_data) as mock_read_sql:
        result = pull_data_from_db(mock_engine, "test_table")

        # Verify read_sql was called with correct parameters
        mock_read_sql.assert_called_once()
        args, _ = mock_read_sql.call_args
        assert args[0] == "SELECT * FROM test_table ORDER BY date DESC LIMIT 10000"
        assert args[1] == mock_engine

        # Verify the result is as expected
        pd.testing.assert_frame_equal(result, expected_data)


def test_pull_data_from_db_exception():
    """Test pull_data_from_db handles exceptions"""
    mock_engine = MagicMock()

    with patch("pandas.read_sql", side_effect=Exception("Database error")):
        result = pull_data_from_db(mock_engine, "test_table")
        assert result is None


def test_evaluate_model():
    """Test evaluate_model function"""
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 1, 1, 0]

    result = evaluate_model(y_true, y_pred)

    assert "accuracy_score" in result
    assert "precision_score" in result
    assert "recall_score" in result
    assert "f1_score" in result

    assert isinstance(result["accuracy_score"], float)
    assert isinstance(result["precision_score"], float)
    assert isinstance(result["recall_score"], float)
    assert isinstance(result["f1_score"], float)

    # Check actual values
    assert result["accuracy_score"] == 0.6  # 3 correct out of 5
    assert (
        abs(result["precision_score"] - 0.6667) < 0.001
    )  # 2 true positives out of 3 predicted positives
    assert (
        abs(result["recall_score"] - 0.6667) < 0.001
    )  # 2 true positives out of 3 actual positives
    assert abs(result["f1_score"] - 0.6667) < 0.001  # F1 score should be about 0.6667

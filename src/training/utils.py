import os
import yaml
import logging
import sqlite3
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Global config variable
config = None


def load_config(config_path):
    """Load configuration from YAML file."""
    global config
    try:
        with open(config_path) as conf_file:
            config = yaml.safe_load(conf_file)
        return config
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        raise


def initialize_config():
    """Initialize configuration from environment variable."""
    config_path = os.getenv("config_path")
    if config_path:
        return load_config(config_path)
    return None


def get_customer_data_path():
    """Get customer data path from config."""
    global config
    if not config:
        config = initialize_config()
    if config and "database" in config and "customer" in config["database"]:
        return config["database"]["customer"]["database_path"]
    return None


def connect_sqlite(dbpath: str) -> sqlite3.Connection:
    """Create and return a SQLite connection."""
    try:
        return sqlite3.connect(dbpath, check_same_thread=False)
    except Exception as e:
        print(dbpath)
        print(f"Error connecting to SQLite: {str(e)}")
        raise


def validate_config(config_dict):
    """Validate the configuration file for required fields and proper values."""
    required_sections = ["base", "database", "data", "hyperparameters"]
    for section in required_sections:
        if section not in config_dict:
            raise ValueError(f"Missing required section: {section}")
    # Validate specific required fields
    if "random_state" not in config_dict["base"]:
        raise ValueError("Missing random_state in base configuration")
    return True


def log_step(step_name):
    """Decorator for logging steps of a function."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            logging.info(f"Starting {step_name}")
            try:
                result = func(*args, **kwargs)
                logging.info(f"Completed {step_name}")
                return result
            except Exception as e:
                logging.error(f"Error in {step_name}: {str(e)}")
                raise

        return wrapper

    return decorator


def evaluate_model(y_true, y_pred):
    """Evaluate model performance using various metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred)
    out = {
        "accuracy_score": accuracy,
        "precision_score": precision,
        "recall_score": recall,
        "f1_score": f1score,
    }
    return out


def pull_data_from_db(db_engine, tablename: str):
    """Retrieve all data from a database table."""
    try:
        query = f"SELECT * FROM {tablename} ORDER BY date DESC LIMIT 10000"
        return pd.read_sql(query, db_engine)
    except Exception as e:
        print(f"Error pulling data from database: {str(e)}")
        return None

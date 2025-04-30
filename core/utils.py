"""
Database connection and shared database utilities
"""

import mlflow
import sqlite3
import logging
import pandas as pd
from typing import Optional
from config import load_config
from datetime import datetime

logger = logging.getLogger(__name__)


def connect_sqlite(dbpath: str) -> sqlite3.Connection:
    """Create and return a SQLite connection."""
    try:
        return sqlite3.connect(dbpath, check_same_thread=False)
    except Exception as e:
        print(dbpath)
        print(f"Error connecting to SQLite: {str(e)}")
        raise


def get_db_engine(config=None):
    """Get database engine from configuration"""
    if config is None:
        config = load_config()
    return connect_sqlite(config["database"]["db_path"])


def initialize_mlflow(config=None):
    """Initialize MLflow with the given configuration"""
    if config is None:
        config = load_config()

    tracking_uri = config["database"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["base"]["experiment_name"])
    return mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)


def push_data_to_db(
    db_engine, tablename: str, dfpath: str = None, data: pd.DataFrame = None
) -> None:
    """Save data to the configured database."""
    try:
        now = datetime.now()
        formatted_date = now.strftime("%Y-%m-%d")

        # Load data from file or use provided DataFrame
        if dfpath:
            data = pd.read_csv(dfpath)

        if data is None:
            raise ValueError("Either dfpath or data must be provided")

        data["date"] = formatted_date
        data.to_sql(tablename, db_engine, if_exists="append", index=False)
    except Exception as e:
        print(f"Error pushing data to database: {str(e)}")


def pull_data_from_db(
    db_engine, tablename: str, job_id: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Retrieve data from a database table, optionally filtered by job_id."""
    try:
        if job_id:
            query = f"SELECT * FROM {tablename} WHERE job_id = ?"
            return pd.read_sql(query, db_engine, params=(job_id,))
        else:
            query = f"SELECT * FROM {tablename}"
            return pd.read_sql(query, db_engine)
    except Exception as e:
        print(f"Error pulling data from database: {str(e)}")
        return None

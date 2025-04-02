import os
import yaml
import sqlite3
import datetime
import pandas as pd
from typing import Optional
from datetime import datetime

# from prefect import task
from dotenv import load_dotenv

load_dotenv()

config_path = os.getenv("config_path")
with open(config_path) as config:
    config = yaml.safe_load(config)
customer_data_path = config["database"]["customer"]["database_path"]


def connect_sqlite(dbpath: str) -> sqlite3.Connection:
    """Create and return a SQLite connection."""
    try:
        return sqlite3.connect(dbpath, check_same_thread=False)
    except Exception as e:
        print(dbpath)
        print(f"Error connecting to SQLite: {str(e)}")
        raise


# @task(name="Push data to database")
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


# @task(name="Pull data from database")
def pull_data_from_db(db_engine, tablename: str) -> Optional[pd.DataFrame]:
    """Retrieve all data from a database table."""
    try:
        query = f"SELECT * FROM {tablename}"
        return pd.read_sql(query, db_engine)
    except Exception as e:
        print(f"Error pulling data from database: {str(e)}")
        return None


def load_dataframe(filepath: str) -> pd.DataFrame:
    """Load data from CSV file into a DataFrame."""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading dataframe from {filepath}: {str(e)}")
        raise


# @task(name="Process raw data")
def process_dataframe(
    dataframe: pd.DataFrame, target_col: str, drop_cols: list = None
) -> pd.DataFrame:
    """Clean and preprocess the dataframe for analysis."""
    try:
        # Create a copy to avoid modifying the original
        df = dataframe.copy()

        # Clean column names
        df.columns = df.columns.str.replace(" ", "_").str.lower()

        # Clean categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            df[col] = df[col].str.replace(" ", "_").str.lower()

        # Drop specified columns
        if drop_cols:
            df = df.drop(drop_cols, axis=1)

        # Handle special case for totalcharges
        if "totalcharges" in df.columns:
            df = df[df["totalcharges"] != "_"]
            df["totalcharges"] = df["totalcharges"].astype("float32")

        # Convert target column to binary
        if target_col in df.columns:
            df[target_col] = (df[target_col] == "yes").astype(int)

        return df
    except Exception as e:
        print(f"Error processing dataframe: {str(e)}")
        raise

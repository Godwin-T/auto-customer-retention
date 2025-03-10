import os
import yaml
import mlflow
import sqlite3
import pandas as pd


from flask import Flask, request, jsonify
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine


dbname = os.getenv("customer_db")
dbpath = os.getenv("db_path")


def load_config(config_path):

    global config
    with open(config_path) as config:
        config = yaml.safe_load(config)


config_path = os.getenv("config_path")
load_config(config_path)
mlflow.set_registry_uri(config["tracking"]["tracking_url"])


def connect_sqlite(dbpath: str) -> sqlite3.Connection:
    """Create and return a SQLite connection."""
    try:
        return sqlite3.connect(dbpath, check_same_thread=False)
    except Exception as e:
        print(dbpath)
        print(f"Error connecting to SQLite: {str(e)}")
        raise


def get_engine():
    """Get appropriate database engine based on availability."""

    customer_data_path = f"{dbpath}/{dbname}"
    return connect_sqlite(customer_data_path), "sqlite"


# Initialize database connection once
db_engine, db_type = get_engine()

# @task(name="Pull data from database")
def pull_data_from_db(tablename: str):
    """Retrieve all data from a database table."""
    try:
        query = f"SELECT * FROM {tablename}"

        if db_type == "sqlite":
            return pd.read_sql(query, db_engine)
        else:
            return pd.read_sql(query, con=db_engine)
    except Exception as e:
        print(f"Error pulling data from database: {str(e)}")
        return None


# @task(name="Process input data")
def input_data_processing(data, column_to_drop="churn"):
    data.drop_duplicates(inplace=True)
    data.columns = data.columns.str.lower()
    data.columns = data.columns.str.replace(" ", "_").str.lower()

    # Drop the specified column if it exists
    if column_to_drop.lower() in data.columns:
        data.drop(columns=[column_to_drop.lower()], inplace=True)

    categorical_col = data.dtypes[data.dtypes == "object"].index.tolist()
    for col in categorical_col:
        data[col] = data[col].str.replace(" ", "_").str.lower()

    return data


# @task(name="Process output data")
def output_data_processing(data, prediction):

    data["churn"] = prediction
    data["churn"] = (data["churn"] >= 0.6).astype("int")
    output_data_frame = data

    return output_data_frame


def load_model(modelname, alias="Production"):
    # Load the model using the alias
    model_uri = f"models:/{modelname}@{alias}"
    model = mlflow.sklearn.load_model(model_uri)
    return model

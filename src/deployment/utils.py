import mlflow
import sqlite3
import pandas as pd
from datetime import datetime


def connect_sqlite(dbpath: str) -> sqlite3.Connection:
    """Create and return a SQLite connection."""
    try:
        return sqlite3.connect(dbpath, check_same_thread=False)
    except Exception as e:
        print(dbpath)
        print(f"Error connecting to SQLite: {str(e)}")
        raise


# @task(name="Pull data from database")
def pull_data_from_db(db_engine, tablename: str):
    """Retrieve all data from a database table."""
    try:
        query = f"SELECT * FROM {tablename}"
        return pd.read_sql(query, db_engine)
    except Exception as e:
        print(f"Error pulling data from database: {str(e)}")
        return None


# @task(name="Process input data")
def input_data_processing(data, column_to_drop="churn"):

    data.drop_duplicates(inplace=True)
    data.drop(["date"], axis=1, inplace=True)

    data.columns = data.columns.str.lower()
    data.columns = data.columns.str.replace(" ", "_").str.lower()

    # # Drop the specified column if it exists
    # if column_to_drop.lower() in data.columns:
    #     data.drop(columns=[column_to_drop.lower()], inplace=True)

    churn = data.pop("churn")

    categorical_col = data.dtypes[data.dtypes == "object"].index.tolist()
    for col in categorical_col:
        data[col] = data[col].str.replace(" ", "_").str.lower()
    data_dict = data.to_dict(orient="records")

    return churn, data_dict, data


def output_data_processing(data, prediction, churn, model_name="Production"):

    # Ensure the prediction column exists
    data = data.copy()
    data["prediction"] = prediction

    data["actual"] = (churn == "yes").astype("int")
    data["predicted"] = (data["prediction"] >= 0.6).astype(
        "int"
    )  # Convert churn threshold
    data["model_name"] = model_name

    # Construct output DataFrame
    output_data_frame = pd.DataFrame(
        {
            "customerid": data["customerid"],
            "model_name": data["model_name"],
            "predicted": data["predicted"],
            "actual": data["actual"],
        }
    )
    return output_data_frame


def load_model(modelname, alias="Production"):
    # Load the model using the alias
    model_uri = f"models:/{modelname}@{alias}"
    model = mlflow.sklearn.load_model(model_uri)
    return model


def push_data_to_db(db_engine, tablename: str, data: pd.DataFrame):
    """Save data to the configured database."""
    try:
        now = datetime.now()
        formatted_date = now.strftime("%Y-%m-%d")

        if data is None:
            raise ValueError("Data must be provided")

        data["prediction_date"] = formatted_date
        data.to_sql(tablename, db_engine, if_exists="append", index=False)
    except Exception as e:
        print(f"Error pushing data to database: {str(e)}")

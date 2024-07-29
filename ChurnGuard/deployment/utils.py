import os
import sqlite3
import requests
from datetime import datetime
import pandas as pd

CHILD_DIR = os.getcwd()
EVIDENTLY_SERVICE_ADDRESS = os.getenv("EVIDENTLY_SERVICE", "http://127.0.0.1:5000")
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")
MLFLOW_TRACKING_URI = "http://mlflow:5000"

MODEL_NAME = "Custormer-churn-models"
MODEL_STAGE = "Production"

BUCKETNAME = "newchurnbucket"
OBJECTNAME = "churn_model.pkl"

MODEL_PATH = "/home/models/churnmodel.pkl"
DB_NAME = "CustomerProflie.db"
DB_DIRECTORY = "/home/databases/"
DF_NAME = "Prediction"


def save_df(db_directory, dbname, dfname, data):

    db_path = os.path.join(db_directory, dbname)
    conn = sqlite3.connect(db_path)

    now = datetime.now()
    formatted_date = now.strftime("%d/%B/%Y")

    data["date"] = formatted_date
    data.to_sql(dfname, conn, if_exists="append", index=False)

    conn.close()


def send_to_evidently_service(record, prediction):
    rec = record.copy()
    rec["prediction"] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/churn", json=[rec])


# MLFLOW_TRACKING_URI = f"sqlite:///{CHILD_DIR}/mlflow.db"


# MODEL_DB_PATH = (
#     "/home/mlflow.db"  # f"{os.path.dirname(os.getcwd())}/training_pipeline/mlflow.db"
# )
# EVIDENTLY_SERVICE_ADDRESS = os.getenv("EVIDENTLY_SERVICE", "http://127.0.0.1:5000")
# MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")

# PROCESSED_DATASET = "../data/processed_data/churn.csv"

# MLFLOW_TRACKING_URI = f"sqlite:///{MODEL_DB_PATH}"

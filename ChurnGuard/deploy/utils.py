import os
import requests

CHILD_DIR = os.getcwd()
EVIDENTLY_SERVICE_ADDRESS = os.getenv("EVIDENTLY_SERVICE", "http://127.0.0.1:5000")
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")

MODEL_NAME = "Custormer-churn-models"
MODEL_STAGE = "Production"

BUCKETNAME = "newchurnbucket"
OBJECTNAME = "churn_model.pkl"

MLFLOW_TRACKING_URI = f"sqlite:///{CHILD_DIR}/mlflow.db"


def save_to_db(record, prediction, collection):
    rec = record.copy()
    rec["prediction"] = prediction
    collection.insert_one(rec)


def send_to_evidently_service(record, prediction):
    rec = record.copy()
    rec["prediction"] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/churn", json=[rec])

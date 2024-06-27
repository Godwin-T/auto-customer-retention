import os
import requests


MODEL_DB_PATH = (
    "/home/mlflow.db"  # f"{os.path.dirname(os.getcwd())}/training_pipeline/mlflow.db"
)
EVIDENTLY_SERVICE_ADDRESS = os.getenv("EVIDENTLY_SERVICE", "http://127.0.0.1:5000")
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")

PROCESSED_DATASET = "../data/processed_data/churn.csv"

MLFLOW_TRACKING_URI = f"sqlite:///{MODEL_DB_PATH}"

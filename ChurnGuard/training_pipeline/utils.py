import numpy as np
import pandas as pd
from datetime import datetime

from prefect import task
from sklearn.metrics import roc_curve
from datahelper import save_df, insert_collection

DB_DIRECTORY = "/home/databases/"
RAW_DATASET = "/home/data/raw_data/Churn.csv"
TARGET_COLUMN = "churn"
DROP_COLUMNS = ["customerid"]
PROCESSED_DATASET = "./data/churn-data/processed_data/churn.csv"
MODEL_PATH = "../models/churn_model.pkl"
METRICS_PATH = ""
PREDICTIONS_PATH = ""
ROC_CURVE_PATH = "../model_output/roc_curve.csv"
PREDICTION_URL = "http://127.0.0.1:9696/predict"
DB_NAME = "CustomerProflie.db"
PREDICTION_TABLE_NAME = "Predictions"
PROCESSED_DATASET_NAME = "TrainingData"
RAW_DATASET_NAME = "RawData"
ROC_TABLE_NAME = ""
MONGO_DBNAME = "MetricsDb"
COLLECTION_NAME = "Metrics"

# Define Metrics


@task(name="Metrics Saving")
def save_metrics(metrics):

    # if not os.path.exists(os.path.dirname(METRICS_PATH)):
    #     os.mkdir(os.path.dirname(METRICS_PATH))

    # with open(METRICS_PATH, "w") as json_file:
    #     json.dump(metrics, json_file)

    now = datetime.now()
    formatted_date = now.strftime("%d/%B/%Y")
    metrics["Date"] = formatted_date
    insert_collection(MONGO_DBNAME, COLLECTION_NAME, metrics)


@task(name="Save Prediction")
def save_predictions(y_test, y_pred):
    # Store predictions data for confusion matrix
    cdf = pd.DataFrame(
        np.column_stack([y_test, y_pred]), columns=["true_label", "predicted_label"]
    ).astype(int)

    save_df(DB_NAME, PREDICTION_TABLE_NAME, data=cdf)


#  cdf.to_csv(PREDICTIONS_PATH, index=None)


def save_roc_curve(y_test, y_pred_proba):
    # Calcualte ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    # Store roc curve data
    cdf = pd.DataFrame(np.column_stack([fpr, tpr]), columns=["fpr", "tpr"]).astype(
        float
    )
    save_df(DB_NAME, ROC_TABLE_NAME, data=cdf)
    # cdf.to_csv(ROC_CURVE_PATH, index=None)

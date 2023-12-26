import os
import shutil
from pathlib import Path

RAW_DATASET = "../raw_data/Telco-Customer-Churn.csv"
TARGET_COLUMN = "churn"
PROCESSED_DATASET = "../processed_data/churn.csv"
MODEL_PATH = "../models/churn_model.pkl"
METRICS_PATH = "../model_output/metrics.json"
PREDICTIONS_PATH = "../model_output/predictions.csv"
ROC_CURVE_PATH = "../model_output/roc_curve.csv"


def delete_and_recreate_dir(path):
    try:
        shutil.rmtree(path)
    except:
        pass
    finally:
        Path(path).mkdir(parents=True, exist_ok=True)

# # monitoring_service.py
import os
import yaml
import sqlite3
import threading
import pandas as pd
from typing import Dict, Any, Optional

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    TargetDriftPreset,
    DataQualityPreset,
)

# from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
from prometheus_client import Gauge
db_lock = threading.Lock()

# Define Prometheus metrics
data_drift_gauge = Gauge("data_drift", "Data Drift Status", ["model"])
feature_drift_gauge = Gauge(
    "feature_drift", "Feature Drift Status", ["feature", "model"]
)
feature_drift_value_guage = Gauge(
    "feature_drift_value", "Feature Drift Value", ["feature", "model"]
)
model_performance_gauge = Gauge(
    "model_performance", "Model Performance", ["metric", "model"]
)

# Load configuration
config_path = os.getenv("config_path", "config.yaml")
with open(config_path) as config_file:
    config = yaml.safe_load(config_file)

# Database connection
def connect_sqlite(dbpath: str) -> sqlite3.Connection:
    """Create and return a SQLite connection."""
    try:
        return sqlite3.connect(dbpath, check_same_thread=False, timeout=30)
    except Exception as e:
        print(f"Error connecting to SQLite: {str(e)}")
        raise


# Initialize DB connection
customerdb = config["database"]["db_path"]
db_engine = connect_sqlite(customerdb)


def fetch_data_from_db(tablename: str) -> Optional[pd.DataFrame]:
    """Retrieve all data from a database table."""
    try:
        with db_lock:
            query = f"SELECT * FROM {tablename}"
            return pd.read_sql(query, db_engine)
    except Exception as e:
        print(f"Error pulling data from database: {str(e)}")
        return None


def get_reference_data(model_name: str) -> pd.DataFrame:
    """Get reference data for a specific model from DB."""
    prediction_logs = fetch_data_from_db("prediction_logs")
    feature_logs = fetch_data_from_db("processdata")

    # Filter for specific model if needed
    if "model" in prediction_logs.columns:
        prediction_logs = prediction_logs[prediction_logs["model"] == model_name]

    # Join feature and prediction data
    feature_logs["actual"] = prediction_logs["actual"]
    feature_logs["predicted"] = prediction_logs["predicted"]

    # Use first 1000 records as reference
    reference_data = feature_logs.iloc[:1000]

    return reference_data


def get_prediction_logs(model_name: str) -> pd.DataFrame:
    """Get prediction logs for a specific model from DB."""
    prediction_logs = fetch_data_from_db("prediction_logs")

    if "model" in prediction_logs.columns:
        prediction_logs = prediction_logs[prediction_logs["model"] == model_name]

    return prediction_logs


def get_current_data(model_name: str) -> pd.DataFrame:
    """Get current data for a specific model from DB."""
    prediction_logs = fetch_data_from_db("prediction_logs")
    feature_logs = fetch_data_from_db("processdata")

    # Filter for specific model if needed
    if "model" in prediction_logs.columns:
        prediction_logs = prediction_logs[prediction_logs["model"] == model_name]

    # Join feature and prediction data
    feature_logs["actual"] = prediction_logs["actual"]
    feature_logs["predicted"] = prediction_logs["predicted"]

    # Use last 1000 records as current data
    current_data = feature_logs.iloc[:1000]  # -1000:

    return current_data


def analyze_drift(
    reference_data: pd.DataFrame, current_data: pd.DataFrame, model_name: str
):
    """Analyze drift between reference and current data and report to Prometheus."""
    categorical_cols = reference_data.select_dtypes(
        include=["object"]
    ).columns.to_list()
    numerical_cols = reference_data.select_dtypes(exclude=["object"]).columns.to_list()

    # Remove target and prediction columns from features
    # if "actual" in numerical_cols:
    #     numerical_cols.remove("actual")
    # if "predicted" in numerical_cols:
    #     numerical_cols.remove("predicted")

    # Define column mapping
    column_mapping = ColumnMapping(
        target="actual",
        prediction="predicted",
        numerical_features=numerical_cols,
        categorical_features=categorical_cols,
    )

    # Create a drift report
    drift_report = Report(
        metrics=[DataDriftPreset(), TargetDriftPreset(), DataQualityPreset()]
    )

    # Run the report
    drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    # Extract results
    results = {}
    drift_detected = False
    drift_results = drift_report.as_dict()

    # Extract dataset drift
    dataset_drift = next(
        (
            metric
            for metric in drift_results["metrics"]
            if metric["metric"] == "DatasetDriftMetric"
        ),
        None,
    )

    if dataset_drift:
        # Set drift_detected based on the dataset_drift result
        drift_detected = dataset_drift["result"]["dataset_drift"]
        data_drift_gauge.labels(model=model_name).set(1 if drift_detected else 0)

    # Track individual feature drift
    column_drifts = []
    for metric in drift_results["metrics"]:
        if metric["metric"] == "DataDriftTable":
            for key, val in metric["result"]["drift_by_columns"].items():
                feature_name = key
                feature_drift = val["drift_detected"]
                feature_drift_gauge.labels(feature=feature_name, model=model_name).set(
                    1 if feature_drift else 0
                )
                feature_drift_value_guage.labels(
                    feature=feature_name, model=model_name
                ).set(val["drift_score"])

                # Add to our drift tracking
                if feature_drift:
                    column_drifts.append(feature_name)

    # Update the drift_detected value if any column has drift
    if drift_detected:
        drift_detected = True

    results["drift_detected"] = drift_detected
    results["drifted_columns"] = column_drifts

    return results


def analyze_performance(model_name: str) -> Dict[str, Any]:
    """Analyze model performance metrics."""
    prediction_logs = fetch_data_from_db("prediction_logs")

    # Filter for specific model if needed
    if "model" in prediction_logs.columns:
        prediction_logs = prediction_logs[prediction_logs["model"] == model_name]

    # Calculate performance metrics
    actual = prediction_logs["actual"]
    predicted = prediction_logs["predicted"]

    # Simple accuracy calculation
    accuracy = (actual == predicted).mean()

    # Export to Prometheus
    model_performance_gauge.labels(metric="accuracy", model=model_name).set(accuracy)

    return {"accuracy": accuracy, "sample_count": len(actual)}


# API Endpoints
# @app.route("/data-metrics/drift/<model_name>", methods=["GET"])
def check_data_drift(model_name):
    """API endpoint to check data drift for a specific model."""
    try:
        reference_data = get_reference_data(model_name)
        current_data = get_current_data(model_name)

        drift_results = analyze_drift(reference_data, current_data, model_name)
        return drift_results
    except Exception as e:
        return {"error": str(e)}


# @app.route("/model-metrics/performance/<model_name>", methods=["GET"])
def check_model_performance(model_name):
    """API endpoint to check model performance metrics."""
    try:
        performance_results = analyze_performance(model_name)
        return performance_results
    except Exception as e:
        return {"error": str(e)}

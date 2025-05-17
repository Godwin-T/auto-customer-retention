# src/prefect/flows/monitoring_flow.py
from prefect import flow, task
import yaml
import os
import datetime
import logging
from monitoring_task import (
    check_data_drift_task,
    check_model_performance_task,
    # send_alerts_task
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config_path = os.getenv("config_path")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


@task(description="starting monitoring")
def log_monitoring_start():
    """Log the start of a monitoring run"""
    logger.info(f"Starting monitoring run at {datetime.datetime.now()}")
    return datetime.datetime.now()


@task(description="complete monitoring")
def log_monitoring_end(start_time):
    """Log the end of a monitoring run"""
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Completed monitoring run in {duration} seconds")
    return {
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration,
    }


@flow(name="ML Monitoring Flow")
def monitoring_flow():
    """Main monitoring flow for ML models"""
    start_time = log_monitoring_start()

    # Run monitoring for each configured model
    results = {}
    for model in config["models"]:
        model_name = model["name"]
        logger.info(f"Monitoring model: {model_name}")

        # Check for data drift
        drift_results = check_data_drift_task(model_name)

        # Check model performance
        performance_results = check_model_performance_task(model_name)

        # Combine results
        model_results = {
            "drift": drift_results,
            "performance": performance_results,
        }

        # Check if any alerts should be triggered
        # alerts_needed = False
        if drift_results.get("drift_detected", False):
            # alerts_needed = True
            logger.warning(f"Data drift detected for model {model_name}")

        if performance_results.get("accuracy", 1.0) < model.get("thresholds", {}).get(
            "accuracy", 0.75
        ):
            # alerts_needed = True
            logger.warning(f"Model {model_name} accuracy below threshold")

        # if alerts_needed:
        #     send_alerts_task(model_name, model_results)

        results[model_name] = model_results

    # Log end of monitoring
    run_info = log_monitoring_end(start_time)

    return {"results": results, "run_info": run_info}

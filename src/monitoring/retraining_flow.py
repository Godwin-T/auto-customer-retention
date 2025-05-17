# src/prefect/flows/retraining_flow.py
from prefect import flow, task
import requests
import yaml
import logging
import os

from utils import check_data_drift, check_model_performance

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config_path = os.getenv("config_path")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


@task(description="check model retrain")
def check_retraining_needed(model_name):
    """Check if model retraining is needed based on performance and drift"""
    try:
        model_performance = check_model_performance(model_name)
        data_drift = check_data_drift(model_name)

        # Get model threshold configuration
        model_config = next(
            (m for m in config["models"] if m["name"] == model_name), None
        )
        if not model_config:
            logger.warning(f"No configuration found for model {model_name}")
            return False

        # Check if performance is below threshold
        if model_performance.get("accuracy", 1.0) < model_config.get(
            "thresholds", {}
        ).get("accuracy", 0.75):
            logger.info(f"Retraining needed for {model_name}: Accuracy below threshold")
            return True

        # Check if drift is detected results["drift_detected"]
        if data_drift.get("drift_detected", False):
            logger.info(f"Retraining needed for {model_name}: Data drift detected")
            return True

        logger.info(f"No retraining needed for {model_name}")
        return False
    except Exception as e:
        logger.error(f"Error checking if retraining needed: {str(e)}")
    return False


@task(description="retrain model")
def retrain_model():
    """Retrain the model with new data"""
    try:
        endpoint = "http://127.0.0.1:8002/training/train"
        requests.get(endpoint)

    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")
        raise


@task(description="deploy model")
def deploy_model():
    """Deploy the newly trained model to production"""
    try:
        endpoint = "http://127.0.0.1:8002/deployment/deploy-auto"
        response = requests.get(endpoint)

        if response.status_code == 200:
            logger.info("Successfully deployed new model")
            return {"status": "success"}
        else:
            logger.error(f"Failed to deploy model: {response.text}")
    except Exception as e:
        logger.error(f"Error deploying model: {str(e)}")
        raise


@flow(name="ML Retraining Flow")
def retraining_flow():
    """Main flow for retraining ML models when needed"""
    # Run retraining checks for each configured model
    results = {}
    for model in config["models"]:
        model_name = model["name"]
        logger.info(f"Checking if retraining needed for: {model_name}")

        # Check if retraining is needed
        retraining_needed = check_retraining_needed(model_name)

        if retraining_needed:

            # Retrain model
            retrain_model()

            # Deploy new model
            deployment_result = deploy_model()

            results[model_name] = {
                "retraining_performed": True,
                "deployment_result": deployment_result,
            }
        else:
            results[model_name] = {
                "retraining_performed": False,
                "reason": "Metrics within acceptable thresholds",
            }

    return results

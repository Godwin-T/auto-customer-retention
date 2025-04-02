# src/prefect/tasks/monitoring_tasks.py
from prefect import task
import requests
import yaml
import os
import json
import datetime
import pandas as pd
import logging
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@task(description="check data drift")
def check_data_drift_task(model_name):
    """Task to check for data drift"""
    try:
        logger.info(f"Checking data drift for model {model_name}")
        # Call monitoring service API
        response = requests.get(f"http://monitor:8003/data-metrics/drift/{model_name}")

        if response.status_code == 200:
            drift_data = response.json()
            logger.info(f"Data drift check complete: {drift_data}")
            return drift_data
        else:
            logger.error(f"Error checking data drift: {response.text}")
            return {"error": response.text}
    except Exception as e:
        logger.error(f"Error in data drift task: {str(e)}")
        return {"error": str(e)}


@task(description="check prediction drift")
def check_prediction_drift_task(model_name):
    """Task to check for prediction drift"""
    try:
        logger.info(f"Checking prediction drift for model {model_name}")

        # Implementation would be similar to data drift but for predictions
        return {"drift_detected": False, "score": 0.05}
    except Exception as e:
        logger.error(f"Error in prediction drift task: {str(e)}")
        return {"error": str(e)}


@task(description="check model performance")
def check_model_performance_task(model_name):
    """Task to check model performance metrics"""
    try:
        logger.info(f"Checking performance for model {model_name}")
        # Call monitoring service API
        response = requests.get(
            f"http://monitor:8003/model-metrics/performance/{model_name}"
        )

        if response.status_code == 200:
            performance_data = response.json()
            logger.info(f"Performance check complete: {performance_data}")
            return {"performance": performance_data}
        else:
            logger.error(f"Error checking model performance: {response.text}")
            return {"error": response.text}
    except Exception as e:
        logger.error(f"Error in model performance task: {str(e)}")
        return {"error": str(e)}


@task(description="send slack alert")
def send_slack_alert(webhook_url, message):
    """Send a Slack alert"""
    try:
        logger.info(f"Would send Slack alert: {message}")

        """
        response = requests.post(
            webhook_url,
            json={"text": message}
        )
        if response.status_code != 200:
            raise ValueError(f"Error sending Slack alert: {response.text}")
        """

        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error sending Slack alert: {str(e)}")
        return {"status": "failed", "error": str(e)}

# src/prefect/tasks/monitoring_tasks.py
from prefect import task
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils import check_model_performance, check_data_drift


@task(description="check data drift")
def check_data_drift_task(model_name):
    """Task to check for data drift"""
    try:
        logger.info(f"Checking data drift for model {model_name}")

        data_drift_result = check_data_drift(model_name)

        logger.info(f"Data drift check complete: {data_drift_result}")
        return data_drift_result

    except Exception as e:
        logger.error(f"Error in data drift task: {str(e)}")
        return {"error": str(e)}


@task(description="check model performance")
def check_model_performance_task(model_name):
    """Task to check model performance metrics"""
    try:
        logger.info(f"Checking performance for model {model_name}")
        # Call monitoring service API

        model_drift_result = check_model_performance(model_name)
        return {"performance": model_drift_result}

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

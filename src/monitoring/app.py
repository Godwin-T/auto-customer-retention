import os
import yaml
import threading
import logging
import time
import sqlite3
from monitoring_flow import monitoring_flow
from retraining_flow import retraining_flow

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection settings
TABLE_CHECK_INTERVAL = 600  # 10 minutes in seconds


def load_config():
    """Load configuration from file specified in environment variables"""
    config_path = os.getenv("config_path")
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise


def check_table_exists():
    """
    Check if the required table exists in the database.

    Returns:
        bool: True if the table exists, False otherwise
    """
    config = load_config()
    table_name = "prediction_logs"
    dbpath = config["database"]["db_path"]

    conn = sqlite3.connect(dbpath)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type='table' AND name=?;
    """,
        (table_name,),
    )

    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def start_monitoring():
    while True:
        try:
            logger.info("Starting Monitoring Flow")
            monitoring_flow.serve(
                name="scheduled-comprehensive-monitoring",
                cron="*/30 * * * *",
                tags=["monitoring", "ml-ops", "drift"],
            )
        except Exception as e:
            logger.error(f"Monitoring Flow crashed: {e}. Restarting...")
            time.sleep(10)  # Wait before restarting


def start_retraining():
    while True:
        try:
            logger.info("Starting Retraining Flow")
            retraining_flow.serve(
                name="scheduled-model-training",
                cron="1 1 * * *",
                tags=["deployment", "ml-ops", "drift"],
            )
        except Exception as e:
            logger.error(f"Retraining Flow crashed: {e}. Restarting...")
            time.sleep(10)  # Wait before restarting


def main():
    while True:
        # Check if the required table exists
        if check_table_exists():
            # Table exists, start the monitoring and retraining flows
            t1 = threading.Thread(target=start_monitoring, daemon=True)
            t2 = threading.Thread(target=start_retraining, daemon=True)

            t1.start()
            t2.start()

            t1.join()
            t2.join()
        else:
            # Table doesn't exist, wait and try again
            time.sleep(TABLE_CHECK_INTERVAL)


if __name__ == "__main__":
    main()

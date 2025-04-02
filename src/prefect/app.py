import threading
import logging
import time
from monitoring_flow import monitoring_flow
from retraining_flow import retraining_flow

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_monitoring():
    while True:
        try:
            logger.info("Starting Monitoring Flow")
            monitoring_flow.serve(
                name="scheduled-comprehensive-monitoring",
                cron="*/3 * * * *",
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
                cron="*/5 * * * *",
                tags=["deployment", "ml-ops", "drift"],
            )
        except Exception as e:
            logger.error(f"Retraining Flow crashed: {e}. Restarting...")
            time.sleep(10)  # Wait before restarting


def main():
    t1 = threading.Thread(target=start_monitoring, daemon=True)
    t2 = threading.Thread(target=start_retraining, daemon=True)

    t1.start()
    t2.start()

    t1.join()
    t2.join()


if __name__ == "__main__":
    main()

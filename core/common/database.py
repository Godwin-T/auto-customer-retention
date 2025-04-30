"""
Database connection and shared database utilities
"""

import logging
import mlflow
from config import load_config
from utils import connect_sqlite

logger = logging.getLogger(__name__)


def get_db_engine(config=None):
    """Get database engine from configuration"""
    if config is None:
        config = load_config()

    return connect_sqlite(config["database"]["db_path"])


def initialize_mlflow(config=None):
    """Initialize MLflow with the given configuration"""
    if config is None:
        config = load_config()

    tracking_uri = config["database"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["training_config"]["base"]["experiment_name"])
    return mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

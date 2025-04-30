"""
Configuration loading and management
"""

import os
import yaml
import logging

logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from file specified in environment variables"""
    config_path = os.getenv("config_path")
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

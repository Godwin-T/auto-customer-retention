#!/usr/bin/env python3
"""
Modularized Flask application that combines:
- Data Ingestion
- Model Training
- Model Deployment
"""

import logging
from flask import Flask
from dotenv import load_dotenv
import warnings

# Import blueprints
from routes.routes import ingestion_bp, training_bp, deployment_bp
from config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore")


def create_app():
    """Initialize and configure the Flask application"""
    # Load environment variables
    load_dotenv()

    # Create Flask app
    app = Flask(__name__)

    # Load configuration
    app.config["deploy_config"] = load_config()

    # Register blueprints
    app.register_blueprint(ingestion_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(deployment_bp)

    return app


# Create the Flask app using the factory pattern
app = create_app()

# Only run the app if this script is executed directly
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8002)

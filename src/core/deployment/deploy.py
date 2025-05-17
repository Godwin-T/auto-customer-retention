"""
Routes for model deployment and prediction
"""

import pandas as pd
from flask import request, jsonify
from io import StringIO

from common.database import get_db_engine
from config import load_config
from utils import pull_data_from_db, push_data_to_db
from .utils import (
    input_data_processing,
    output_data_processing,
    load_model,
    automated_deployment_workflow,
)


def predict():
    """Make predictions using the deployed model"""

    config = load_config()
    model_name = "customerchurn"
    db_engine = get_db_engine()

    model = load_model(model_name, config)
    data = pull_data_from_db(db_engine, "processdata")

    if len(data):
        churn, data_dict, dataframe = input_data_processing(data)
        prediction = model.predict(data_dict)

        output_frame = output_data_processing(dataframe, prediction, churn)
        prediction_table_name = config["logs"]["prediction_logs"]
        push_data_to_db(
            db_engine,
            tablename=prediction_table_name,
            data=output_frame,
        )
    return jsonify(
        {
            "status": "success",
            "response": "The predictions have successfully been saved to database",
        }
    )


def batch_predict():
    """Make batch predictions from uploaded data"""

    config = load_config()
    model_name = config["model_registry"]["name"]
    db_engine = get_db_engine()

    model = load_model(model_name, config)
    # Get JSON data
    request_data = request.get_json()

    if not request_data or "csv_data" not in request_data:
        return jsonify({"status": "error", "message": "No data provided"}), 400

    # job_id = request_data.get("job_id")
    csv_data = request_data.get("csv_data")

    # Convert CSV string back to DataFrame
    csv_buffer = StringIO(csv_data)
    df = pd.read_csv(csv_buffer)

    # Now process the DataFrame
    churn, data_dict, dataframe = input_data_processing(df)

    prediction = model.predict(data_dict)

    output_frame = output_data_processing(dataframe, prediction, churn)
    prediction_table_name = config["logs"]["prediction_logs"]
    push_data_to_db(
        db_engine,
        tablename=prediction_table_name,
        data=output_frame,
    )

    return jsonify(
        {
            "status": "success",
            "message": "Batch prediction completed",
            # Include other relevant results
        }
    )


def deploy_auto():
    """Automated deployment endpoint that handles the entire workflow."""

    config = load_config()
    result = automated_deployment_workflow(config)
    return jsonify(result)

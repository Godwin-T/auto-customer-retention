import logging
from flask import Blueprint, jsonify
from training.train import streamlit_train_model
from deployment.deploy import predict, batch_predict, deploy_auto
from ingestion.ingest import (
    process_existing_data,
    upload_data,
    process_uploaded_data,
    get_data,
)


logger = logging.getLogger(__name__)
ingestion_bp = Blueprint("ingestion", __name__, url_prefix="/ingestion")
training_bp = Blueprint("training", __name__, url_prefix="/training")
deployment_bp = Blueprint("deployment", __name__, url_prefix="/deployment")


@ingestion_bp.route("/process", methods=["GET"])
def existing_data():

    try:
        result = process_existing_data()
        return result
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@ingestion_bp.route("/upload", methods=["POST"])
def receive_upload():
    try:
        result = upload_data()
        return result

    except Exception as e:
        logger.error(f"Error uploading data: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@ingestion_bp.route("/process_uploaded", methods=["POST"])
def process_streamlit_data():
    try:
        result = process_uploaded_data()
        return result
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@ingestion_bp.route("/get_data", methods=["GET"])
def get_specific_data():
    try:
        result = get_data()
        return result
    except Exception as e:
        logger.error(f"Error retrieving data: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@training_bp.route("/train", methods=["POST"])
def model_training():
    try:
        result = streamlit_train_model()
        return result
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@deployment_bp.route("/predict", methods=["GET"])
def prediction():
    try:
        result = predict()
        return result
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@deployment_bp.route("/batch_predict", methods=["POST"])
def batch_prediction():
    # try:
    result = batch_predict()
    return result

    # except Exception as e:
    #     logger.error(f"Error in batch prediction: {str(e)}")
    #     return jsonify({"status": "error", "message": str(e)}), 500


@deployment_bp.route("/deploy-auto", methods=["GET"])
def automated_deployment():
    # try:
    result = deploy_auto()
    return result
    # except Exception as e:
    #     logger.error(f"Error in automated deployment: {str(e)}")
    #     return jsonify({"status": "error", "message": str(e)}), 500

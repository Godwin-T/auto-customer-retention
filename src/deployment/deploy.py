import os
import yaml
import shutil
import logging
import pandas as pd
import mlflow
from flask import Flask, request, jsonify
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from utils import (
    pull_data_from_db,
    input_data_processing,
    output_data_processing,
    load_model,
    push_data_to_db,
    connect_sqlite,
)
from io import StringIO

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_app(config=None):
    """Factory function to create and configure the Flask app"""
    app = Flask("Deploy")

    # If config is not provided, load it
    if config is None:
        config = load_config()

    # Store config in app for access in routes
    app.config["deploy_config"] = config

    # Register routes
    register_routes(app)

    return app


def load_config():
    """Load configuration from file specified in environment variables"""
    load_dotenv()
    config_path = os.getenv("config_path", "./configs/deploy.yaml")

    try:
        with open(config_path) as config_file:
            config = yaml.safe_load(config_file)
            validate_config(config)
            return config
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}.")


def validate_config(config):
    """Validate the configuration file for required fields and proper values."""
    required_sections = ["logs", "database"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    return True


def initialize_mlflow(config):
    """Initialize MLflow with the given configuration"""
    tracking_db = config["database"]["tracking_uri"]
    mlflow.set_registry_uri(tracking_db)
    return MlflowClient(tracking_uri=tracking_db)


def get_db_engine(config):
    """Get database engine from configuration"""
    customer_db = config["database"]["db_path"]
    return connect_sqlite(customer_db)


def extract_top_model(client):
    """Extract the top performing model from MLflow"""
    # Get all experiments
    run_data = []
    experiments = client.search_experiments()

    for experiment in experiments:
        experiment_id = experiment.experiment_id

        # Get runs for the current experiment
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
        )

        for run in runs:
            try:
                metrics = run.data.metrics
                inference_time = (run.info.end_time - run.info.start_time) / 1000
                run_data.append(
                    {
                        "run_id": run.info.run_id,
                        "experiment_id": experiment_id,
                        "f1_score": metrics.get("f1_score", 0),
                        "accuracy_score": metrics.get("accuracy_score", 0),
                        "precision_score": metrics.get("precision_score", 0),
                        "recall_score": metrics.get("recall_score", 0),
                        "inference_time": inference_time,
                        "params": run.data.params,
                        "tags": run.data.tags.get("model_name", "unknown"),
                        "artifact_uri": run.info.artifact_uri,
                    }
                )
            except Exception as e:
                logger.error(f"Error processing run {run.info.run_id}: {e}")
                continue

    # Convert to DataFrame
    df = pd.DataFrame(run_data)
    if df.empty:
        logger.warning("No runs found.")
        return []

    # Sort by f1_score (descending) and inference_time (ascending)
    df_sorted = df.sort_values(
        by=["f1_score", "inference_time"], ascending=[False, True]
    )

    # Get top 1 run ID
    top_run_id = df_sorted.head(1)["run_id"].tolist()

    # Get all run IDs
    all_run_ids = df["run_id"].tolist()

    # Runs to delete (not in top 1)
    runs_to_delete = set(all_run_ids) - set(top_run_id)

    # Delete unwanted runs and their entire folders
    for run_id in runs_to_delete:
        try:
            # Get run folder path (remove 'file://' prefix)
            artifact_uri = df[df["run_id"] == run_id]["artifact_uri"].values[0]
            run_folder_path = os.path.dirname(artifact_uri.replace("file://", ""))

            # Delete run metadata from MLflow
            client.delete_run(run_id)

            # Delete the entire run folder
            if os.path.exists(run_folder_path):
                shutil.rmtree(run_folder_path)
                logger.info(
                    f"Deleted entire run folder for run {run_id} at {run_folder_path}"
                )

        except Exception as e:
            logger.error(f"Error deleting run {run_id}: {e}")

    logger.info(
        f"Deleted {len(runs_to_delete)} runs and their folders, keeping only the top model."
    )
    return top_run_id


def register_model(client, modelname, run_id):
    """Register a model with MLflow"""
    model_uri = f"runs:/{run_id}/model"

    # Check if the model is already registered
    try:
        client.get_registered_model(modelname)
        logger.info(
            f"Model '{modelname}' already registered. Creating a new model version."
        )
    except Exception:  # If model does not exist, register it
        client.create_registered_model(name=modelname)
        logger.info(f"Registered new model: {modelname}")

    # Create a new version of the model
    model_version = client.create_model_version(
        name=modelname, source=model_uri, run_id=run_id
    )
    logger.info(f"Model version {model_version.version} registered successfully!")

    return model_version.version


def model_transition(
    client, modelname, modelid, currentstage=None, newstage=None, modelversion=None
):
    """Transition a model to a new stage"""
    # Check if any model is already in Production
    try:
        production_models = client.get_model_version_by_alias(modelname, "Production")
    except:
        production_models = None

    # Get latest version
    latest_version = client.get_latest_versions(modelname)[0].version

    # If no model is in Production, move the latest model to Production
    if not production_models:
        # Add a tag to indicate this version is now in Production
        client.set_tag(modelid, "version", latest_version)
        client.set_registered_model_alias(modelname, "Production", latest_version)
        logger.info(
            f"Model version {latest_version} transitioned directly to Production."
        )
        return

    # If a model is already in Production and we have a new stage
    if newstage:
        modelversion = latest_version
        client.set_registered_model_alias(modelname, newstage, modelversion)
        logger.info(
            f"Model version {modelversion} transitioned from {currentstage} to {newstage}."
        )


def compare_models(production_model, challenger_model, validation_data):
    """Compare production model with challenger and decide whether to switch."""
    X_val, y_val = validation_data

    # Get predictions from both models
    prod_preds = production_model.predict(X_val)
    chal_preds = challenger_model.predict(X_val)

    # Calculate metrics
    from sklearn.metrics import f1_score, roc_auc_score

    metrics = {
        "production": {
            "f1": f1_score(y_val, (prod_preds > 0.5).astype(int)),
            "roc_auc": roc_auc_score(y_val, prod_preds),
        },
        "challenger": {
            "f1": f1_score(y_val, (chal_preds > 0.5).astype(int)),
            "roc_auc": roc_auc_score(y_val, chal_preds),
        },
    }

    # Decision logic - replace with your own criteria
    if (
        metrics["challenger"]["f1"] > metrics["production"]["f1"]
        and metrics["challenger"]["roc_auc"] > metrics["production"]["roc_auc"]
    ):
        return True, metrics
    else:
        return False, metrics


def load_validation_data(db_engine, tablename):
    """Load validation data from database"""
    df = pull_data_from_db(db_engine, tablename)

    if "date" in df.columns:
        df.drop(["date"], axis=1, inplace=True)

    y_val, x_val = df.pop("churn"), df
    x_val = x_val.to_dict(orient="records")
    # y_val = y_val.map({"yes": 1, "no": 0}).astype(int)

    return (x_val, y_val)


def automated_deployment_workflow(config):
    """Automated end-to-end workflow for deploying a new model."""
    try:
        # Initialize client
        client = initialize_mlflow(config)

        # 1. Extract top model
        top_run_ids = extract_top_model(client)
        if not top_run_ids:
            return {"deployed": False, "reason": "No models found"}

        top_run_id = top_run_ids[0]

        # 2. Register model
        model_version = register_model(client, "customerchurn", top_run_id)

        # 3. Load validation data
        db_engine = get_db_engine(config)
        val_data = load_validation_data(db_engine, "processdata")

        # 4. Compare with production model if exists
        try:
            production_model = load_model("customerchurn")
            challenger_model = mlflow.sklearn.load_model(f"runs:/{top_run_id}/model")

            should_deploy, metrics = compare_models(
                production_model, challenger_model, val_data
            )

            if should_deploy:
                # 5. Deploy new model if better
                model_transition(
                    client, "customerchurn", top_run_id, newstage="Production"
                )
                logger.info(f"New model {model_version} deployed to production")
                return {
                    "deployed": True,
                    "model_version": model_version,
                    "metrics": metrics,
                    "Response": "Deployed",
                }
            else:
                logger.info(
                    f"New model {model_version} not deployed (not better than production)"
                )
                return {
                    "deployed": False,
                    "model_version": model_version,
                    "metrics": metrics,
                    "Response": "Not deployed (not better than current model)",
                }
        except Exception as e:
            # No production model exists, deploy this one
            model_transition(client, "customerchurn", top_run_id, newstage="Production")
            logger.info(f"First model {model_version} deployed to production")
            return {
                "deployed": True,
                "model_version": model_version,
                "reason": "First model",
                "Response": "Deployed",
            }
    except Exception as e:
        logger.error(f"Deployment workflow error: {str(e)}")
        return {"error": str(e), "Response": f"Error: {str(e)}"}


def register_routes(app):
    """Register Flask routes"""

    @app.route("/predict", methods=["GET"])
    def predict():
        config = app.config["deploy_config"]
        model_name = "customerchurn"
        db_engine = get_db_engine(config)

        model = load_model(model_name)
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
            {"Response": "The predictions have successfully been saved to database"}
        )

    @app.route("/batch_predict", methods=["POST"])
    def batch_predict():
        config = app.config["deploy_config"]
        model_name = "customerchurn"
        db_engine = get_db_engine(config)

        model = load_model(model_name)
        try:
            # Get JSON data
            request_data = request.get_json()

            if not request_data or "csv_data" not in request_data:
                return jsonify({"status": "error", "message": "No data provided"}), 400

            job_id = request_data.get("job_id")
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

        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/deploy-auto", methods=["GET"])
    def deploy_auto():
        """Automated deployment endpoint that handles the entire workflow."""
        config = app.config["deploy_config"]
        result = automated_deployment_workflow(config)
        return jsonify(result)


# Create the Flask app using the factory pattern
app = create_app()

# Only run the app if this script is executed directly
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8002)

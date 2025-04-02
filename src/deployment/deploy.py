import os
import yaml
import shutil
import logging
import pandas as pd
import mlflow
import mlflow.pyfunc
from prefect import task, flow
from dotenv import load_dotenv
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify
from utils import (
    pull_data_from_db,
    input_data_processing,
    output_data_processing,
    load_model,
    push_data_to_db,
    connect_sqlite,
)

load_dotenv()


config_path = os.getenv("config_path")
with open(config_path) as config:
    config = yaml.safe_load(config)

customer_db = config["database"]["customer"]["database_path"]
tracking_db = config["database"]["tracking"]["tracking_url"]

mlflow.set_registry_uri(tracking_db)
db_engine = connect_sqlite(customer_db)


def validate_config(config):
    """Validate the configuration file for required fields and proper values."""
    required_sections = ["base", "database", "data", "hyperparameters"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")

    # Validate specific required fields
    if "random_state" not in config["base"]:
        raise ValueError("Missing random_state in base configuration")

    # Validate data paths exist
    data_path = config["data"]["process"]["path"]
    if not os.path.exists(data_path):
        raise ValueError(f"Data path does not exist: {data_path}")

    return True


def extract_top_model():
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
                        "artifact_uri": run.info.artifact_uri,  # Store artifact path
                    }
                )
            except Exception as e:
                print(f"Error processing run {run.info.run_id}: {e}")
                continue

    # Convert to DataFrame
    df = pd.DataFrame(run_data)
    if not df.empty:
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
                run_folder_path = os.path.dirname(
                    artifact_uri.replace("file://", "")
                )  # Remove /artifacts

                # Delete run metadata from MLflow
                client.delete_run(run_id)

                # Delete the entire run folder
                if os.path.exists(run_folder_path):
                    shutil.rmtree(run_folder_path)
                    print(
                        f"Deleted entire run folder for run {run_id} at {run_folder_path}"
                    )

            except Exception as e:
                print(f"Error deleting run {run_id}: {e}")
        print(
            f"Deleted {len(runs_to_delete)} runs and their folders, keeping only the top model."
        )
    else:
        print("No runs found.")
    return top_run_id


def register_model(modelname, run_id):
    model_uri = f"runs:/{run_id}/model"

    # Check if the model is already registered
    try:
        client.get_registered_model(
            modelname
        )  # If this succeeds, the model already exists
        print(f"Model '{modelname}' already registered. Creating a new model version.")
    except Exception:  # If model does not exist, register it
        client.create_registered_model(name=modelname)
        print(f"Registered new model: {modelname}")

    # Create a new version of the model
    model_version = client.create_model_version(
        name=modelname, source=model_uri, run_id=run_id
    )
    print(f"Model version {model_version.version} registered successfully!")

    return model_version.version


def model_transition(
    modelname, modelid, currentstage=None, newstage=None, modelversion=None
):
    """
    Transitions a model to a new stage. If no model is in production, it tags the latest version as 'Production' directly.

    :param modelname: Name of the registered model.
    :param currentstage: The current stage of the model (if applicable).
    :param newstage: The target stage to transition to.
    :param modelversion: The version of the model to transition (if applicable).
    """

    # Check if any model is already in Production
    try:
        production_models = client.get_model_version_by_alias(modelname, "Production")
    except:
        production_models = None
    # If no model is in Production, move the latest model to Production

    ###### Correct
    latest_version = client.get_latest_versions(modelname)[0].version
    if not production_models:
        latest_version = client.get_latest_versions(modelname)[0].version

        # Add a tag to indicate this version is now in Production
        client.set_tag(modelid, "version", latest_version)
        client.set_registered_model_alias(modelname, "Production", latest_version)

        print(f"Model version {latest_version} transitioned directly to Production.")
        return

    # If a model is already in Production, transition the given version if specified
    # if currentstage and newstage and modelversion:
    if newstage:
        modelversion = latest_version
        client.set_registered_model_alias(modelname, newstage, modelversion)

        print(
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


def load_validation_data(tablename):

    df = pull_data_from_db(db_engine, tablename)
    df.drop(["date"], axis=1, inplace=True)
    y_val, x_val = df.pop("churn"), df
    x_val = x_val.to_dict(orient="records")
    y_val = y_val.map({"yes": 1, "no": 0}).astype(int)

    return (x_val, y_val)


# 7. Add automated model deployment workflow
def automated_deployment_workflow():
    """Automated end-to-end workflow for deploying a new model."""
    try:
        # 1. Extract top model
        top_run_id = extract_top_model()[0]

        # 2. Register model
        model_version = register_model("customerchurn", top_run_id)

        # 3. Load validation data
        val_data = load_validation_data("processdata")

        # 4. Compare with production model if exists
        try:
            production_model = load_model("customerchurn")
            challenger_model = mlflow.sklearn.load_model(f"runs:/{top_run_id}/model")

            should_deploy, metrics = compare_models(
                production_model, challenger_model, val_data
            )

            if should_deploy:
                # 5. Deploy new model if better
                model_transition("customerchurn", top_run_id, newstage="Production")
                logging.info(f"New model {model_version} deployed to production")
                return {
                    "deployed": True,
                    "model_version": model_version,
                    "metrics": metrics,
                }
            else:
                logging.info(
                    f"New model {model_version} not deployed (not better than production)"
                )
                return {
                    "deployed": False,
                    "model_version": model_version,
                    "metrics": metrics,
                }
        except Exception as e:
            # No production model exists, deploy this one
            model_transition("customerchurn", top_run_id, newstage="Production")
            logging.info(f"First model {model_version} deployed to production")
            return {
                "deployed": True,
                "model_version": model_version,
                "reason": "First model",
            }
    except Exception as e:
        logging.error(f"Deployment workflow error: {str(e)}")
        return {"error": str(e)}


load_dotenv()
config_path = os.getenv("config_path")
with open(config_path) as config:
    config = yaml.safe_load(config)
validate_config(config)

mlflow.set_registry_uri(config["database"]["tracking"]["tracking_url"])
client = MlflowClient(tracking_uri=config["database"]["tracking"]["tracking_url"])
experiments = client.search_experiments()


# @log_step("Initializing resources")
# def initialize_resources():
#     """Initializes model and S3 bucket connection once when the app starts."""
#     global model
#     model_name = "customerchurn"
#     model = load_model(model_name)

app = Flask("Deploy")

# @app.before_request
# def check_resources():
#     global resources_initialized
#     resources_initialized = False
#     if not resources_initialized:
#         initialize_resources()
#         resources_initialized = True

model_name = "customerchurn"


@app.route("/predict", methods=["GET"])
def predict():

    model = load_model(model_name)
    data = pull_data_from_db(db_engine, "processdata")

    if len(data):

        churn, data_dict, dataframe = input_data_processing(data)
        prediction = model.predict(data_dict)

        output_frame = output_data_processing(dataframe, prediction, churn)
        prediction_table_name = config["database"]["customer"]["prediction_logs"]
        push_data_to_db(
            db_engine,
            tablename=prediction_table_name,
            data=output_frame,
        )
    return jsonify(
        {"Response": f"The predictions have successfully been saved to database"}
    )


# 8. Add deployment endpoint with automated workflow
@app.route("/deploy-auto", methods=["GET"])
def deploy_auto():
    """Automated deployment endpoint that handles the entire workflow."""
    result = automated_deployment_workflow()
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8002)

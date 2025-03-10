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
)

load_dotenv()

app = Flask("Deploy")
config_path = os.getenv("config_path")
logging.basicConfig(
    filename="train.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_config(config_path):

    global config
    with open(config_path) as config:
        config = yaml.safe_load(config)


load_config(config_path)
mlflow.set_registry_uri(config["tracking"]["tracking_url"])
client = MlflowClient(tracking_uri=config["tracking"]["tracking_url"])


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
            metrics = run.data.metrics
            inference_time = (run.info.end_time - run.info.start_time) / 1000
            try:
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


model_name = "customerchurn"
app = Flask("Churn")
# @task(name="Initailize resources")
def initialize_resources():
    """Initializes model and S3 bucket connection once when the app starts."""
    global model
    model = load_model(model_name)


@app.before_request
def check_resources():
    print("======================Resource initilized===========================")
    global resources_initialized
    resources_initialized = False
    if not resources_initialized:
        initialize_resources()
        resources_initialized = True


@app.route("/deploy", methods=["GET"])
def deploy():

    top_run_id = extract_top_model()[0]
    register_model("customerchurn", top_run_id)
    model_transition("customerchurn", top_run_id, newstage="Production")
    return jsonify({"Response": "Model Updated"})


@app.route("/predict", methods=["GET"])
# @flow(name="Prediction Flow")
def predict():

    print("============================================")
    data = pull_data_from_db("processdata")
    if len(data):
        dataframe = input_data_processing(data)
        model_data = dataframe.drop(["customerid"], axis=1)

        record_dicts = model_data.to_dict(orient="records")
        prediction = model.predict(record_dicts)

        output_frame = output_data_processing(dataframe, prediction)
        output_frame.to_csv("prediction.csv", index=False)
        return jsonify(
            {"Response": "The predictions have successfully been saved to database"}
        )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8002)


# @task(name="Change model")
# def compare_models(model_name):

#     production_alias = "Production"

#     latest_model = client.get_registered_model(model_name).latest_versions
#     latest_model_id = latest_model[0].run_id
#     latest_model_version = latest_model[0].version

#     if latest_model_version == 1:
#         client.set_registered_model_alias(
#             model_name, production_alias, latest_model_version
#         )
#         deploy_info = {"run_id": latest_model_id, "change": True}

#     else:
#         production_model_id = client.get_model_version_by_alias(
#             model_name, production_alias
#         ).run_id
#         production_metrics = get_metrics(client, production_model_id)
#         latest_metrics = get_metrics(client, latest_model_id)
#         deploy = compare_metrics(production_metrics, latest_metrics)

#         if deploy:
#             client.set_registered_model_alias(
#                 model_name, production_alias, latest_model_version
#             )
#             deploy_info = {"run_id": production_model_id, "change": True}
#         else:
#             deploy_info = {"run_id": production_model_id, "change": False}

#     return deploy_info

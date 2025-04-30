import os
import logging
import shutil
import pandas as pd
import mlflow
from mlflow.entities import ViewType

from common.database import initialize_mlflow
from mlflow.exceptions import MlflowException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_top_models(client, top_n=3):
    """Extract and retain top N performing models based on F1 score and inference time."""
    run_data = []
    experiments = client.search_experiments()

    for experiment in experiments:
        experiment_id = experiment.experiment_id
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

    df = pd.DataFrame(run_data)
    if df.empty:
        logger.warning("No runs found.")
        return []

    # Sort and retain top N
    df_sorted = df.sort_values(
        by=["f1_score", "inference_time"], ascending=[False, True]
    )
    top_df = df_sorted.head(top_n)
    top_run_ids = top_df["run_id"].tolist()

    # Delete other runs
    runs_to_delete = set(df["run_id"]) - set(top_run_ids)
    for run_id in runs_to_delete:
        try:
            artifact_uri = df[df["run_id"] == run_id]["artifact_uri"].values[0]
            run_folder_path = os.path.dirname(artifact_uri.replace("file://", ""))

            client.delete_run(run_id)
            if os.path.exists(run_folder_path):
                shutil.rmtree(run_folder_path)
                print(f"Deleted run {run_id}")

        except Exception as e:
            logger.error(f"Error deleting run {run_id}: {e}")

    print(f"Retained top {top_n} models. Deleted {len(runs_to_delete)} others.")
    return top_df


def register_models(client, top_df, model_name):
    """Register top models in the MLflow Model Registry."""
    registered_versions = []
    for _, row in top_df.iterrows():
        run_id = row["run_id"]
        # try:
        # result = mlflow.register_model(
        #     model_uri=f"runs:/{run_id}/model", name=model_name
        # )
        try:
            client.create_registered_model(model_name)
        except MlflowException as e:
            if "already exists" in str(e):
                pass  # Model already exists — safe to continue
            else:
                raise  # Something else went wrong — re-raise the exception

        model_uri = f"runs:/{run_id}/model"
        model_version = client.create_model_version(
            name=model_name, source=model_uri, run_id=run_id
        )
        registered_versions.append(model_version.version)  # result
        print(f"Registered model version {model_version.version} for run {run_id}")
        # except Exception as e:
        #     logger.error(f"Failed to register model from run {run_id}: {e}")
    return registered_versions


def get_latest_model_versions(client, model_name, stage=None):
    """Get latest model versions, optionally filtered by stage."""
    try:
        if stage:
            versions = client.get_latest_versions(model_name, stages=[stage])
        else:
            versions = client.get_latest_versions(model_name)
        return versions
    except Exception as e:
        logger.warning(f"Could not retrieve model versions (stage={stage}): {e}")
        return []


def get_model_metrics(client, run_id):
    """Extract key metrics from a model run."""
    if run_id is None:
        return None

    try:
        run = client.get_run(run_id)
        return {
            "f1_score": run.data.metrics.get("f1_score", 0),
            "accuracy_score": run.data.metrics.get("accuracy_score", 0),
            "precision_score": run.data.metrics.get("precision_score", 0),
            "recall_score": run.data.metrics.get("recall_score", 0),
            "inference_time": (run.info.end_time - run.info.start_time) / 1000,
        }
    except Exception as e:
        logger.error(f"Failed to get metrics for run {run_id}: {e}")
        return None


def is_better_model(new_metrics, comparison_metrics):
    """Return True if new model is better than comparison model based on metrics."""
    if comparison_metrics is None:
        return True  # No comparison model, allow promotion

    # Primary comparison based on F1 score
    if new_metrics["f1_score"] > comparison_metrics["f1_score"]:
        return True

    # If F1 scores are equal, compare inference time
    if (
        new_metrics["f1_score"] == comparison_metrics["f1_score"]
        and new_metrics["inference_time"] < comparison_metrics["inference_time"]
    ):
        return True

    return False


def promote_to_staging(client, model_name, registered_versions):
    """Promote the best model to Staging."""
    if not registered_versions:
        print("No models available to promote to Staging.")
        return None, None

    # Get the latest registered version (should be the best one)
    latest_version = max(registered_versions)

    # Check if there's already a staging model
    staging_versions = get_latest_model_versions(client, model_name, "Staging")

    # Get metrics for the new model
    model_version = client.get_model_version(model_name, latest_version)
    new_model_metrics = get_model_metrics(client, model_version.run_id)

    # If there's a staging model, only promote if new model is better
    if staging_versions:
        staging_metrics = get_model_metrics(client, staging_versions[0].run_id)
        if not is_better_model(new_model_metrics, staging_metrics):
            print("New model is not better than current Staging model. No promotion.")
            return None, None

    # Transition the model to Staging
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Staging",
            archive_existing_versions=False,  # Don't archive existing staging models yet
        )
        client.set_registered_model_alias(
            name=model_name, alias="staging", version=latest_version
        )
        print(f"Model version {latest_version} promoted to Staging.")
        return latest_version, new_model_metrics
    except Exception as e:
        print(f"Failed to promote model to Staging: {e}")
        return None, None


def promote_staging_to_production(client, model_name):
    """Evaluate if Staging model should be promoted to Production."""
    # Get current Staging model
    staging_versions = get_latest_model_versions(client, model_name, "Staging")
    if not staging_versions:
        print("No Staging model available for promotion to Production.")
        return False, None, None

    staging_version = staging_versions[0]
    staging_metrics = get_model_metrics(client, staging_version.run_id)

    # Get current Production model
    production_versions = get_latest_model_versions(client, model_name, "Production")
    if production_versions:
        production_version = production_versions[0]
        production_metrics = get_model_metrics(client, production_version.run_id)
    else:
        production_metrics = None

    # Compare and promote if better
    if is_better_model(staging_metrics, production_metrics):
        try:
            # Transition Staging model to Production, archive existing Production models
            client.transition_model_version_stage(
                name=model_name,
                version=staging_version.version,
                stage="Production",
                archive_existing_versions=True,  # Archive existing production models
            )
            client.set_registered_model_alias(
                name=model_name, alias="Production", version=staging_version.version
            )

            # Log details about model transition
            print(
                f"Model version {staging_version.version} promoted from Staging to Production."
            )

            # Archive previous production models
            if production_versions:
                print(
                    f"Previous Production model version {production_versions[0].version} has been archived."
                )

            return True, staging_version.version, staging_metrics
        except Exception as e:
            logger.error(f"Failed to promote Staging model to Production: {e}")
            return False, None, None
    else:
        print(
            "Staging model is not better than current Production model. No promotion."
        )
        return False, None, staging_metrics


def automated_deployment_workflow(config):
    """Main workflow function to manage the entire model deployment process.

    Returns a dictionary with deployment details and status.
    """
    try:
        client = initialize_mlflow(config)
        model_name = config["model_registry"]["name"]

        print("Starting automated deployment workflow...")

        # Step 1: Extract top models
        print("Step 1: Extracting top models...")
        top_df = extract_top_models(client, top_n=3)
        if top_df.empty:
            print("No models to evaluate. Workflow terminated.")
            return {
                "deployed": False,
                "reason": "No models to evaluate",
                "response": "No models available",
            }

        # Step 2: Register models
        print("Step 2: Registering top models...")
        registered_versions = register_models(client, top_df, model_name)

        # Step 3: Promote to staging
        print("Step 3: Promoting best model to Staging...")
        staging_version, staging_metrics = promote_to_staging(
            client, model_name, registered_versions
        )

        # Step 4: Evaluate for production
        print("Step 4: Evaluating Staging model for Production promotion...")
        if staging_version:
            promoted, production_version, metrics = promote_staging_to_production(
                client, model_name
            )
            if promoted:
                print(
                    "Workflow completed: New model promoted through Staging to Production."
                )
                return {
                    "deployed": True,
                    "model_version": production_version,
                    "metrics": metrics,
                    "response": "Deployed to Production",
                    "status": "New model promoted through Staging to Production",
                }
            else:
                print("Workflow completed: Model promoted to Staging only.")
                return {
                    "deployed": False,
                    "model_version": staging_version,
                    "metrics": metrics,
                    "response": "Not deployed to Production (not better than current model)",
                    "status": "Model promoted to Staging only",
                }
        else:
            print(
                "Workflow completed: No changes to Staging or Production environments."
            )
            return {
                "deployed": False,
                "reason": "Model not better than current Staging model",
                "response": "No changes to environments",
                "status": "No changes to Staging or Production environments",
            }
    except Exception as e:
        logger.error(f"Deployment workflow error: {str(e)}")
        return {
            "deployed": False,
            "error": str(e),
            "response": f"Error: {str(e)}",
            "status": "Error in deployment workflow",
        }


def input_data_processing(data, column_to_drop="churn"):
    data.drop_duplicates(inplace=True)
    if "date" in data.columns:
        data.drop(["date"], axis=1, inplace=True)

    data.columns = data.columns.str.lower()
    data.columns = data.columns.str.replace(" ", "_").str.lower()
    churn = data.pop("churn")

    categorical_col = data.dtypes[data.dtypes == "object"].index.tolist()
    for col in categorical_col:
        data[col] = data[col].str.replace(" ", "_").str.lower()
    data_dict = data.to_dict(orient="records")

    return churn, data_dict, data


def output_data_processing(data, prediction, churn, model_name="Production"):
    # Ensure the prediction column exists
    data = data.copy()
    data["prediction"] = prediction

    data["actual"] = (churn == "yes").astype("int")
    data["predicted"] = (data["prediction"] >= 0.6).astype(
        "int"
    )  # Convert churn threshold
    data["model_name"] = model_name

    # Construct output DataFrame
    output_data_frame = pd.DataFrame(
        {
            "customerid": data["customerid"],
            "model_name": data["model_name"],
            "predicted": data["predicted"],
            "actual": data["actual"],
        }
    )
    return output_data_frame


def load_model(modelname, config, alias="Production"):

    client = initialize_mlflow(config)
    model_versions = client.get_model_version_by_alias(modelname, alias)
    model_uri = f"models:/{modelname}/{model_versions.version}"

    # Load the model
    model = mlflow.sklearn.load_model(model_uri)
    return model

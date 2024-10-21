import os
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from app.churn_guard.utils.validate import compare_metrics, get_metrics

load_dotenv()


model_name = "Sklearn-linear-models"
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
client = MlflowClient(tracking_uri=tracking_uri)


def change_model_stage():

    production_alias = "Production"
    staged_alias = "Staged"

    latest_model = client.get_registered_model(model_name).latest_versions
    latest_model_id = latest_model[0].run_id
    latest_model_version = latest_model[0].version

    production_model = client.get_model_version_by_alias(model_name, production_alias)
    production_model_id = production_model.run_id

    production_metrics = get_metrics(client, production_model_id)
    latest_metrics = get_metrics(client, latest_model_id)
    deploy = compare_metrics(production_metrics, latest_metrics)

    if deploy:
        client.set_registered_model_alias(
            model_name, production_alias, latest_model_version
        )
        deploy_info = {"run_id": production_model_id, "change": True}
    else:
        deploy_info = {"run_id": production_model_id, "change": False}

    return deploy_info


def deploy_production():

    deploy_info = change_model_stage()
    run_id = deploy_info["run_id"]
    change = deploy_info["change"]
    with open("result.txt", "w") as f:
        f.write(str(change))

import os
import json

from prefect import task
from dotenv import load_dotenv
from app.churn_guard.utils.evaluate import evaluate_model

load_dotenv()


model_path = os.getenv("MODEL_PATH")
experiment_name = os.getenv("EXPERIMENT_NAME")

current_model = ""
new_model = ""


@task(name="Evaluate models")
def models_evaluation(current_model, new_model, X, y):

    current_model_evaluations = evaluate_model(current_model, X, y)
    new_model_evaluations = evaluate_model(new_model, X, y)

    with open("./curr_metric.json", "w") as json_file:
        json.dump(current_model_evaluations, json_file, indent=2)

    with open("./new_metric.json", "w") as json_file:
        json.dump(new_model_evaluations, json_file, indent=2)


@task(name="Compare model metrics")
def compare_metrics(previous_metrics, new_metrics):

    previous_f1 = previous_metrics["f1_score"]
    previous_acc = previous_metrics["accuracy"]

    new_f1 = new_metrics["f1_score"]
    new_acc = new_metrics["accuracy"]

    if (new_f1 > previous_f1) and (new_acc > previous_acc):
        deploy = True
    else:
        deploy = False

    return deploy


@task(name="Get model metrics")
def get_metrics(client, run_id):

    f1_score_history = client.get_metric_history(run_id, "f1_score")
    f1_score = [m.value for m in f1_score_history]

    acc_score_history = client.get_metric_history(run_id, "acc")
    acc_score = [m.value for m in acc_score_history]
    outpuct_dict = {"f1_score": f1_score, "accuracy": acc_score}

    return outpuct_dict

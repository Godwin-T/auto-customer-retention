import os
import json
import pickle
import mlflow

from prefect import task
from sklearn.pipeline import make_pipeline
from utils import MODEL_PATH

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Define Model Evaluation Metrics
def eval_metrics(y_true, prediction):

    f1 = f1_score(y_true, prediction)
    metrics = {
        "acc": accuracy_score(y_true, prediction),
        "f1_score": f1,
        "precision": precision_score(y_true, prediction),
        "recall": recall_score(y_true, prediction),
    }
    return metrics


# Define Model Training Function
@task(name="Train model")
def train_model(
    train_x,
    train_y,
    c_value=71,
    experiment_name="Customer_Churn_Predictions",
    tracking_uri="http://mlflow:5000",  # "sqlite:////home/databases/mlflow.db",
):

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    train_x = train_x.to_dict(orient="records")

    with mlflow.start_run():

        mlflow.set_tag("Developer", "Godwin")
        mlflow.set_tag("model", "Logistic Regression")
        mlflow.log_param("C", c_value)

        lr_pipeline = make_pipeline(
            DictVectorizer(sparse=False), LogisticRegression(C=c_value)
        )  # make training pipeline

        lr_pipeline.fit(train_x, train_y)
        prediction = lr_pipeline.predict(train_x)
        evaluation_result = eval_metrics(train_y, prediction)

        mlflow.log_metrics(evaluation_result)
        # artifact_path="/home/databases/artifacts"
        # os.makedirs(artifact_path, exist_ok=True)
        mlflow.sklearn.log_model(lr_pipeline, artifact_path="model")
        artifact_uri = mlflow.get_artifact_uri()
        print(f"Artifact uri: {artifact_uri}")

    return lr_pipeline, evaluation_result


# Define Model Evaluation Function
@task(name="Evaluate Model")
def evaluate_model(model, X_test, y_test, float_precision=4):

    X_test = X_test.to_dict(orient="records")
    prediction = model.predict(X_test)
    evaluation_result = eval_metrics(y_test, prediction)

    evaluation_result = json.loads(
        json.dumps(evaluation_result),
        parse_float=lambda x: round(float(x), float_precision),
    )
    return evaluation_result, prediction


# Define Model Saving Function


@task(name="Save Model")
def save_model(model):

    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.mkdir(os.path.dirname(MODEL_PATH))

    with open(MODEL_PATH, "wb") as f_out:
        pickle.dump(model, f_out)
    print("Model saved successfully!")

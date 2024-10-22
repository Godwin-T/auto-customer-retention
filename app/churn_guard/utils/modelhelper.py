import os
import pickle
import mlflow

import mlflow.entities
from prefect import task
from dotenv import load_dotenv
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from app.churn_guard.utils.evaluate import evaluate

load_dotenv()

# Define Model Training Function
@task(name="Train model")
def train_model(
    train_x,
    train_y,
    c_value=71,
    tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
    experiment_name="Customer_Churn_Prediction",
):
    # mlflow.delete_experiment(experiment_name)
    try:
        mlflow.create_experiment(
            experiment_name, artifact_location="s3://mlflowartifactsdb/"
        )
    except:
        pass
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
        evaluation_result = evaluate(train_y, prediction)

        mlflow.log_metrics(evaluation_result)
        mlflow.sklearn.log_model(
            lr_pipeline,
            artifact_path="mlflow",
            registered_model_name="Sklearn-models",
        )
        artifact_uri = mlflow.get_artifact_uri()
        print(f"Artifact uri: {artifact_uri}")

    return lr_pipeline, evaluation_result


# Define Model Saving Function
@task(name="Save Model to directory")
def save_model_to_dir(model, model_path):

    if not os.path.exists(os.path.dirname(model_path)):
        os.mkdir(os.path.dirname(model_path))

    with open(model_path, "wb") as f_out:
        pickle.dump(model, f_out)
    print("Model saved successfully!")
    return "Model saved successfully!"


# Define Model Saving Function
@task(name="Save Model to s3")
def save_model_to_s3(model, model_path):

    if not os.path.exists(os.path.dirname(model_path)):
        os.mkdir(os.path.dirname(model_path))

    with open(model_path, "wb") as f_out:
        pickle.dump(model, f_out)
    print("Model saved successfully!")
    return "Model saved successfully!"

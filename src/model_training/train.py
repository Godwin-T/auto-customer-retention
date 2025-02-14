# Importing Libraries
import os
import json
import pickle
import mlflow
import pandas as pd
import mlflow.entities

from prefect import task, flow
from dotenv import load_dotenv

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

from ..utils.evaluate import evaluate_model
from ..utils.modelhelper import train_model, save_model_to_dir
from ..utils.datahelper import load_data_from_relational_db


load_dotenv()

# target_column = os.getenv("TARGET_COLUMN")
db_dir = os.getenv("DB_DIRECTORY")

db_name = os.getenv("DB_NAME")
processed_dataset_name = os.getenv("PROCESSED_DATASET_NAME")
processed_dataset = os.getenv("PROCESSED_DATASET")

model_path = os.getenv("MODEL_PATH")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
experiment_name = os.getenv("EXPERIMENT_NAME")

model_name = "Custormer-churn-models"


@task(name="Process data")
def process_data(data, target_column):

    # Load data
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1993)
    output = (X_train, X_test, y_train, y_test)
    return output


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
            experiment_name, artifact_location="s3://mlflowartifactsdb/mlflow"
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
        # artifact_uri = mlflow.get_artifact_uri()
        # print(f"Artifact uri: {artifact_uri}")

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


@flow(name="Training and Model Evaluation")
def training_pipeline():

    # Load the processed dataset and split into train and test sets
    # X, y = load_data_from_db(PROCESSED_DATASET)

    print(
        "=================================Starting Model Training=================================================\n\n"
    )

    data = load_data_from_relational_db(
        dbprovider="mysql", tablename=processed_dataset_name
    )
    (X_train, X_test, y_train, y_test) = process_data(data, target_column="churn")

    model, train_eval_result = train_model(
        X_train, y_train
    )  # Train the model and get the evaluation results on the training set
    test_eval_result, _ = evaluate_model(
        model, X_test, y_test
    )  # Evaluate the model on the test set and get the evaluation results and predictions

    model_evaluation_result = {
        "Train evaluation result": train_eval_result,
        "Test evaluation result": test_eval_result,
    }

    # Print the training set evaluation results
    # save_model_to_dir(model, model_path)

    print("====================Train Set Metrics==================")
    print(json.dumps(train_eval_result, indent=2))
    print("======================================================")
    print()

    # Print the test set evaluation results
    print("====================Test Set Metrics==================")
    print(json.dumps(test_eval_result, indent=2))
    print("======================================================")

    return model_evaluation_result

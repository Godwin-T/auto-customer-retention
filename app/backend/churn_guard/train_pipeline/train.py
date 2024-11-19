# Importing Libraries
import os
import json
import pandas as pd
from prefect import task, flow
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split

from app.churn_guard import evaluate_model
from app.churn_guard import train_model, save_model_to_dir
from app.churn_guard import load_data_from_relational_db

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


@task(name="Process input data")
def process_data(data, target_column):

    # Load data
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1993)
    output = (X_train, X_test, y_train, y_test)
    return output


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

# Importing Libraries
print("Importing Libraries")
import json
import time
import mlflow
import pandas as pd
from prefect import task, flow
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from modelhelper import evaluate_model, train_model, save_model
from utils import (
    save_metrics,
    save_predictions,
    PROCESSED_DATASET,
    TARGET_COLUMN,
    DB_DIRECTORY,
    DB_NAME,
    PROCESSED_DATASET_NAME,
)
from datahelper import load_data


@task(name="Load data")
def load_data_from_db(db_dir, db_name, table_name):
    # Read the CSV file and split into features (X) and target variable (y)
    # data = pd.read_csv(file_path)
    data = load_data(db_dir, db_name, table_name)
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    return X, y


@flow(name="Training and Model Evaluation")
def main():
    # Load the processed dataset and split into train and test sets
    # X, y = load_data_from_db(PROCESSED_DATASET)
    X, y = load_data_from_db(DB_DIRECTORY, DB_NAME, PROCESSED_DATASET_NAME)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1993)

    # Train the model and get the evaluation results on the training set
    model, train_eval_result = train_model(X_train, y_train)

    # Evaluate the model on the test set and get the evaluation results and predictions
    test_eval_result, y_pred = evaluate_model(model, X_test, y_test)

    # Combine the evaluation results for both the training and test sets
    model_evaluation_result = {
        "Train evaluation result": train_eval_result,
        "Test evaluation result": test_eval_result,
    }

    # Print the training set evaluation results
    print("====================Train Set Metrics==================")
    print(json.dumps(train_eval_result, indent=2))
    print("======================================================")
    print()

    # Print the test set evaluation results
    print("====================Test Set Metrics==================")
    print(json.dumps(test_eval_result, indent=2))
    print("======================================================")

    # Save the overall model evaluation results, test set predictions, and the trained model
    # save_metrics(model_evaluation_result)
    # save_predictions(y_test, y_pred)
    save_model(model)

    MLFLOW_TRACKING_URI = "sqlite:////home/databases/mlflow.db"
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    runs = client.search_runs(
        experiment_ids="1",
        filter_string="metrics.f1_score >0.59",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.f1_score ASC"],
    )[0]

    run_id = runs.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="Custormer-churn-models")

    print("=================================")

    # # running loop from 0 to 4
    # for i in range(0,5):
    #     # adding 2 seconds time delay
    #     time.sleep(20000)


if __name__ == "__main__":
    main()

# Importing Libraries
import os
import json
import mlflow
import pandas as pd
from prefect import task, flow
from dotenv import load_dotenv

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

from app.churn_guard.utils.datahelper import load_data_from_sqlite_db
from app.churn_guard.utils.evaluate import evaluate_model
from app.churn_guard.utils.modelhelper import train_model, save_model

load_dotenv()

target_column = os.getenv("TARGET_COLUMN")
db_dir = os.getenv("DB_DIRECTORY")

db_name = os.getenv("DB_NAME")
processed_dataset_name = os.getenv("PROCESSED_DATASET_NAME")
processed_dataset = os.getenv("PROCESSED_DATASET")

model_path = os.getenv("MODEL_PATH")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
experiment_name = os.getenv("EXPERIMENT_NAME")

model_name = "Custormer-churn-models"


# @task(name="Load data")
def process_data(data, target_column):

    # Load data
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y


# @flow(name="Training and Model Evaluation")
def training_pipeline():

    # Load the processed dataset and split into train and test sets
    # X, y = load_data_from_db(PROCESSED_DATASET)

    print(
        "=================================Starting Model Training================================================="
    )

    data = load_data_from_sqlite_db(db_dir, db_name, processed_dataset_name)
    X, y = process_data(data, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1993)

    model, train_eval_result = train_model(
        X_train, y_train
    )  # Train the model and get the evaluation results on the training set
    test_eval_result, y_pred = evaluate_model(
        model, X_test, y_test
    )  # Evaluate the model on the test set and get the evaluation results and predictions

    model_evaluation_result = {
        "Train evaluation result": train_eval_result,
        "Test evaluation result": test_eval_result,
    }

    client = MlflowClient(tracking_uri=tracking_uri)
    runs = client.search_runs(
        experiment_ids="1",
        filter_string="metrics.f1_score >0.59",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.f1_score ASC"],
    )[0]

    run_id = runs.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=model_name)

    # Print the training set evaluation results
    save_model(model, model_path)

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

    return model_evaluation_result


# if __name__ == "__main__":
#     main()
# # running loop from 0 to 4
# for i in range(0,5):
#     # adding 2 seconds time delay
#     time.sleep(20000)

# Importing Libraries

import pandas as pd
import numpy as np

import mlflow

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import recall_score, f1_score
from sklearn.pipeline import make_pipeline
from prefect import task, flow

import warnings

warnings.filterwarnings("ignore")


@task(retries=3, retry_delay_seconds=2)
def read_dataframe(path):

    data = pd.read_csv(path)
    data.columns = data.columns.str.replace(" ", "_").str.lower()

    categorical_col = data.dtypes[data.dtypes == "object"].index.tolist()
    for col in categorical_col:
        data[col] = data[col].str.replace(" ", "_").str.lower()
    return data


# Data Preparation
@task
def prepare_data(data):

    data = data[data["totalcharges"] != "_"]
    data["totalcharges"] = data["totalcharges"].astype("float32")

    data["churn"] = (data["churn"] == "yes").astype(int)
    categorical_col = data.dtypes[data.dtypes == "object"].index.tolist()
    numerical_col = ["tenure", "totalcharges", "monthlycharges"]

    categorical_col.remove("customerid")
    feature_cols = categorical_col + numerical_col

    train_data, test_data = train_test_split(data, test_size=0.25, random_state=0)

    train_x = train_data.drop(["churn"], axis=1)
    test_x = test_data.drop(["churn"], axis=1)

    train_x = train_x[feature_cols].to_dict(orient="records")
    test_x = test_x[feature_cols].to_dict(orient="records")

    train_y = train_data.pop("churn")
    test_y = test_data.pop("churn")
    output = (train_x, train_y, test_x, test_y)
    return output


@task
def training_model(train_x, train_y):

    pipeline = make_pipeline(DictVectorizer(sparse=False), LogisticRegression(C=61))

    print("Training model")

    pipeline.fit(train_x, train_y)
    return pipeline


@task
def evaluating_model(model, test_x, test_y):

    pred = model.predict(test_x)
    accuracy_ = accuracy_score(test_y, pred)
    precision_ = precision_score(test_y, pred)
    recall_ = recall_score(test_y, pred)
    f1score_ = f1_score(test_y, pred)

    metrics = {
        "test_accuracy_score": accuracy_,
        "test_precision_score": precision_,
        "test_recall_score": recall_,
        "test_f1_score": f1score_,
    }

    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, artifact_path="models_mlflow")

    return metrics


@flow
def main_flow(path: str = "./data/Telco-Customer-Churn.csv"):

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Telcom Churn")

    data = read_dataframe(path)
    data = prepare_data(data)
    (train_x, train_y, test_x, test_y) = data
    model = training_model(train_x, train_y)
    metrics = evaluating_model(model, test_x, test_y)
    print("Complete")
    return metrics


# if __name__ == '__main__':
#     main_flow()

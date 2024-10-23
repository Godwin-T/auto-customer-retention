import os
import json
import mlflow
import sqlite3
import pandas as pd
import mlflow.entities
from typing import List
from datetime import datetime
from dotenv import load_dotenv
from prefect import task, flow
from sqlalchemy import create_engine
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

load_dotenv()

raw_dataset_path = os.getenv("RAW_DATASET_PATH")
raw_dataset_name = os.getenv("RAW_DATASET_NAME")
processed_dataset_name = os.getenv("PROCESSED_DATASET_NAME")

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
experiment_name = os.getenv("EXPERIMENT_NAME")

drop_columns = os.getenv("DROP_COLUMNS")
target_column_name = os.getenv("TARGET_COLUMN")

try:

    dbname = os.getenv("DBNAME")
    username = os.getenv("MYSQL_USERNAME")
    password = os.getenv("MYSQL_PASSWORD")
    hostname = os.getenv("HOSTNAME")

    engine = create_engine(
        f"mysql+mysqlconnector://{username}:{password}@{hostname}/{dbname}"
    )

except:
    pass


@task(name="Process model training input data")
def process_model_train_input(data, target_column):

    # Load data
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1993)
    output = (X_train, X_test, y_train, y_test)
    return output


# Load data
@task(name="Load raw dataset from path")
def load_raw_dataset_from_path(filepath: str):

    dataframe = pd.read_csv(filepath)
    return dataframe


# Prepare Data
@task(name="Process raw dataset")
def process_raw_data(dataframe: pd.DataFrame, target_column_name, drop_cols=None):

    dataframe.columns = dataframe.columns.str.replace(" ", "_").str.lower()

    categorical_col = dataframe.dtypes[dataframe.dtypes == "object"].index.tolist()
    for col in categorical_col:
        dataframe[col] = dataframe[col].str.replace(" ", "_").str.lower()

    # dataframe = dataframe.drop([drop_cols], axis=1)
    dataframe = dataframe[dataframe["totalcharges"] != "_"]
    dataframe["totalcharges"] = dataframe["totalcharges"].astype("float32")
    dataframe[target_column_name] = (dataframe[target_column_name] == "yes").astype(int)

    return dataframe


@task(name="Load data from sqlite")
def load_data_from_sqlite_db(db_directory, dbname, tablename):

    db_path = os.path.join(db_directory, dbname)
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {tablename}"
    # Load data into a DataFrame
    df = pd.read_sql(query, conn)
    return df


@task(name="Load data from mysql")
def load_data_from_mysql_db(sql_engine, tablename):

    query = f"SELECT * FROM {tablename}"
    df = pd.read_sql(query, con=sql_engine)
    return df


def load_data_from_relational_db(
    dbprovider="sqlite", db_directory=None, dbname=None, tablename="ProcessedData"
):

    if dbprovider == "sqlite":

        df = load_data_from_sqlite_db(db_directory, dbname, tablename)
    else:
        print("==============================")
        print(engine)
        print("==============================")
        df = load_data_from_mysql_db(engine, tablename)

    return df


# Define Model Evaluation Metrics
def evaluate(y_true, prediction):

    f1 = f1_score(y_true, prediction)
    metrics = {
        "acc": accuracy_score(y_true, prediction),
        "f1_score": f1,
        "precision": precision_score(y_true, prediction),
        "recall": recall_score(y_true, prediction),
    }
    return metrics


# Define Model Evaluation Function
@task(name="Evaluate Model")
def evaluate_model(model, X_test, y_test, float_precision=4):

    X_test = X_test.to_dict(orient="records")
    prediction = model.predict(X_test)
    evaluation_result = evaluate(y_test, prediction)

    evaluation_result = json.loads(
        json.dumps(evaluation_result),
        parse_float=lambda x: round(float(x), float_precision),
    )
    return evaluation_result, prediction


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

    print("Model Training Completed")


@task(name="Push data to sqlite")
def save_dataframe_to_sqlite(db_directory, dbname, tablename, dfpath=None, data=None):

    now = datetime.now()
    formatted_date = now.strftime("%d/%B/%Y")

    if dfpath:

        data = pd.read_csv(dfpath)
        data["date"] = formatted_date

    else:
        data["date"] = formatted_date

    db_path = os.path.join(db_directory, dbname)
    conn = sqlite3.connect(db_path)

    data.to_sql(tablename, conn, if_exists="append", index=False)
    conn.close()


@task(name="Push data to sqlite")
def save_dataframe_to_mysql(sql_engine, tablename, dfpath=None, data=None):

    now = datetime.now()
    formatted_date = now.strftime("%d/%B/%Y")

    if dfpath:

        data = pd.read_csv(dfpath)
        data["date"] = formatted_date

    else:
        data["date"] = formatted_date

    # Save the DataFrame to MySQL
    data.to_sql(name=tablename, con=sql_engine, if_exists="append", index=False)


def save_dataframe_to_relational_db(
    tablename,
    dbprovider="sqlite",
    db_directory=None,
    dbname=None,
    df_path=None,
    data=None,
):

    if dbprovider == "sqlite":

        save_dataframe_to_sqlite(db_directory, dbname, tablename, df_path, data)

    else:

        save_dataframe_to_mysql(engine, tablename, df_path, data)


@flow(name="Data Processing")
def data_processing_pipeline():

    input_data = load_raw_dataset_from_path(filepath=raw_dataset_path)
    save_dataframe_to_relational_db(
        tablename=raw_dataset_name, dbprovider="mysql", data=input_data
    )

    churn_data = process_raw_data(
        input_data, target_column_name, drop_cols=drop_columns
    )
    save_dataframe_to_relational_db(
        tablename=processed_dataset_name, dbprovider="mysql", data=churn_data
    )


@flow(name="Training and Model Evaluation")
def training_pipeline():

    data = load_data_from_relational_db(
        dbprovider="mysql", tablename=processed_dataset_name
    )
    (X_train, X_test, y_train, y_test) = process_model_train_input(
        data, target_column="churn"
    )

    train_model(X_train, y_train)  # Train model


@flow
def main():

    data_processing_pipeline()
    training_pipeline()


# if __name__ == "__main__":
#     main.deploy(
#         name="my-deployment",
#         work_pool_name="customer-retention",
#         image="freshinit/myprefectworkflow:v1",
#         push = False,
#         cron="0 */72 * * *"
#     )


# if __name__ == "__main__":

#     main.serve(
#         name="data=processing-deployment",
#         cron="*/3 * * * *"
#     )

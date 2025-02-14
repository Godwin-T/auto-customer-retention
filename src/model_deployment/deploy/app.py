import os
import boto3
import pickle
import mlflow
import pandas as pd


from dotenv import load_dotenv
from sqlalchemy import create_engine

from prefect import task, flow
from flask import Flask, request, jsonify
from mlflow.tracking import MlflowClient

load_dotenv()

dbname = os.getenv("DBNAME")
username = os.getenv("MYSQL_USERNAME")
password = os.getenv("MYSQL_PASSWORD")
hostname = os.getenv("HOSTNAME")

engine = create_engine(
    f"mysql+mysqlconnector://{username}:{password}@{hostname}/{dbname}"
)


tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
client = MlflowClient(tracking_uri=tracking_uri)


# @task(name="Connect to s3 bucket")
def connect_bucket(
    access_key_id=os.getenv("AWS_SERVER_PUBLIC_KEY"),
    access_secret_key=os.getenv("AWS_SERVER_SECRET_KEY"),
):

    s3_bucket = boto3.client(
        "s3", aws_access_key_id=access_key_id, aws_secret_access_key=access_secret_key
    )

    return s3_bucket


# @task(name="Load input data from path")
def load_data_with_path(data):

    data = pd.DataFrame(data)
    return data


# @task(name="Load input data from database")
def load_data_from_db(start_date=None, end_date=None):

    tablename = "RawData"

    if start_date:
        query = f"SELECT * FROM {tablename} WHERE date BETWEEN '{start_date}' AND '{end_date}' "
    else:
        query = f"SELECT * FROM {tablename}"

    df = pd.read_sql(query, con=engine)
    return df


# @task(name="Process input data")
def input_data_processing(data):

    data.drop_duplicates(inplace=True)
    data.columns = data.columns.str.lower()
    data.columns = data.columns.str.replace(" ", "_").str.lower()

    categorical_col = data.dtypes[data.dtypes == "object"].index.tolist()
    for col in categorical_col:
        data[col] = data[col].str.replace(" ", "_").str.lower()

    return data


# @task(name="Process output data")
def output_data_processing(data, prediction):

    data["churn"] = prediction
    data["churn"] = (data["churn"] >= 0.6).astype("int")
    output_data_frame = data
    # output_data_frame.drop(["churn"], axis=1, inplace=True)

    return output_data_frame


# @task(name="Load model from s3 bucket")
def load_model_from_s3(s3_bucket, bucket_name, file_name):

    obj = s3_bucket.get_object(Bucket=bucket_name, Key=file_name)
    model = obj["Body"].read()
    model = pickle.loads(model)
    return model


# @task(name="Load model from mlflow artifact store")
def load__mlflow_model(model_name, model_alias="Production"):

    model_info = client.get_model_version_by_alias(model_name, model_alias)
    model_id = model_info.run_id
    model = mlflow.pyfunc.load_model(f"runs:/{model_id}/mlflow")
    return model


# @task(name="Upload Prediction")
def upload_prediction_to_s3(s3_bucket, local_file_path, bucket_name, s3_object_name):

    s3_bucket.upload_file(local_file_path, bucket_name, s3_object_name)
    return "Object saved successfully"


bucket_name = os.getenv("BUCKETNAME")
file_name = os.getenv("OBJECTNAME")
s3_bucket = None
model = None
model_name = "Sklearn-models"
resources_initialized = False

app = Flask("Churn")

# @task(name="Initailize resources")
def initialize_resources():
    """Initializes model and S3 bucket connection once when the app starts."""
    global s3_bucket, model
    if not s3_bucket:
        s3_bucket = connect_bucket()  # Connect to S3 bucket once
    if not model:
        # model = load_model_from_s3(
        #     s3_bucket, bucket_name, file_name
        # )  # Load model from S3 once
        model = load__mlflow_model(model_name)


@app.before_request
def check_resources():
    print("======================Resource initilized===========================")
    global resources_initialized
    if not resources_initialized:
        initialize_resources()
        resources_initialized = True


@app.route("/predict", methods=["POST"])
# @flow(name="Prediction Flow")
def predict():

    print("============================================")
    data_path = request.get_json()

    dataframe = load_data_with_path(data_path)
    dataframe = input_data_processing(dataframe)
    model_data = dataframe.drop(["customerid"], axis=1)

    print("===================================================")

    record_dicts = model_data.to_dict(orient="records")
    prediction = model.predict(record_dicts)

    output_frame = output_data_processing(dataframe, prediction)
    output_frame.to_csv("prediction.csv", index=False)
    # upload_prediction_to_s3(s3_bucket, "prediction.csv", bucket_name, "prediction.csv")
    os.remove("prediction.csv")
    return jsonify(
        {"Response": "The predictions have successfully been saved to database"}
    )


if __name__ == "__main__":
    app.run(debug=True, port=9696)

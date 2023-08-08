import os
import requests

import pandas as pd
from flask import Flask, request, jsonify
from pymongo import MongoClient

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


child_directory = os.getcwd()
parent_directory = os.path.dirname(child_directory)

EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:5000')
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")

MLFLOW_TRACKING_URI = f"sqlite:///{parent_directory}/model/mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
runs = client.search_runs(experiment_ids='1',
                          filter_string="metrics.test_f1_score >0.595",
                          run_view_type=ViewType.ACTIVE_ONLY,
                          max_results=5,
                          order_by=["metrics.test_f1_score ASC"]
                        )

model_name = "Custormer-churn-models"
model_stage="Production"
loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")

def load_data(data):
    
    data =  {key.replace(' ', '_').lower(): val for key, val in data.items()}
    data['totalcharges'] = float(data['totalcharges'])
    return data

def prepare_data(data):

    customer_id = data['customerid']
    columns = ['gender', 'partner', 'dependents', 'phoneservice',
                       'multiplelines', 'internetservice', 'onlinesecurity',
                       'onlinebackup', 'deviceprotection', 'techsupport',
                       'streamingtv', 'streamingmovies','contract',
                       'paperlessbilling', 'paymentmethod','tenure', 
                       'totalcharges', 'monthlycharges']
    keys = data.keys()
    for key in list(keys):
        if key not in columns:
            data.pop(key)
    
    return customer_id,data


app = Flask('Churn')
mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction_service")
collection = db.get_collection("data")

@app.route("/predict", methods = ['POST'])
def predict():

    try:
        customer = request.get_json()
        data = load_data(customer)
        customer_id, record = prepare_data(data)
        prediction = loaded_model.predict(record)
        output = {'customerid': customer_id, 'churn':bool(prediction)}
        save_to_db(record, bool(prediction))
        send_to_evidently_service(record, bool(prediction))
        return jsonify(output)
    except:
        output = {'customerid': None, 'churn':None}
        return jsonify(output)
        


def save_to_db(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    collection.insert_one(rec)


def send_to_evidently_service(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/churn", json=[rec])

if __name__ == '__main__':
    app.run(debug=True, port=9696)
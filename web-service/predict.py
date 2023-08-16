#Importing Libraries
print('Importing Libraries')
import os
import mlflow
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


child_directory = os.getcwd()
parent_directory = os.path.dirname(child_directory)

MLFLOW_TRACKING_URI = f"sqlite:///{parent_directory}/mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
# runs = client.search_runs(
#     experiment_ids='1',
#     filter_string="metrics.test_f1_score >0.595",
#     run_view_type=ViewType.ACTIVE_ONLY,
#     max_results=5,
#     order_by=["metrics.test_f1_score ASC"]
# )

name = "Custormer-churn-models"
stage="Staging"

loaded_model = mlflow.pyfunc.load_model(f"models:/{name}/{stage}")

# logged_model = f'runs:{parent_directory}/9f00a33d4d4d43c1969d03d106fbf4d7/model'

# # Load model
# loaded_model = mlflow.pyfunc.load_model(logged_model)
print('Connected')

def load_data(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.replace(' ', '_').str.lower()

    categorical_col = data.dtypes[data.dtypes == 'object'].index.tolist()
    for col in categorical_col:
        data[col] = data[col].str.replace(' ', '_').str.lower()

    data = data[data['totalcharges'] != '_']
    data['totalcharges'] = data['totalcharges'].astype('float32')
    return data

def prepare_data(data):

    categorical_col = data.dtypes[data.dtypes == 'object'].index.tolist()
    numerical_col = ['tenure', 'totalcharges', 'monthlycharges']

    categorical_col.remove('customerid')
    feature_cols = categorical_col + numerical_col

    df_data = data[feature_cols].to_dict(orient = 'records')
    return data, df_data


app = Flask('Churn')

@app.route("/predict", methods = ['POST'])
def predict():

    customer = request.get_json()
    data = load_data(customer)
    data, df = prepare_data(data)
    prediction = loaded_model.predict(df)
    data['churn'] = prediction
    output = data[data['churn'] == 1]
    output.to_csv('predictions/churning customers.csv', index = False)
    output_text = {'Prediction Status': 'The model successufully predicted the customers that are likely to churn and save the file'}
    return jsonify(output_text)

if __name__ == '__main__':
    app.run(debug=True, port=5080)
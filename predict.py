#Importing Libraries
print('Importing Libraries')
import mlflow
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify



MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logged_model = 'runs:/b5eb75916ba045bc99b4902ea507a319/model'

# Load model
loaded_model = mlflow.pyfunc.load_model(logged_model)
print('Connected')

utils_path = 'Churn.bin'
with open(utils_path, 'rb') as f:
    model, encoder = pickle.load(f)


def data_loader(path):
    
    data = pd.read_csv(path)
    data.columns = data.columns.str.replace(' ', '_').str.lower()

    categorical_col = data.dtypes[data.dtypes == 'object'].index.tolist()
    for col in categorical_col:
        data[col] = data[col].str.replace(' ', '_').str.lower()
    return data

def data_encoder(data, feature_cols):

    data = encoder.transform(data[feature_cols].to_dict(orient = 'records'))
    return data


def file_data_prep(data):

    data = data[data['totalcharges'] != '_']
    data['totalcharges'] = data['totalcharges'].astype('float32')

    categorical_col = data.dtypes[data.dtypes == 'object'].index.tolist()
    numerical_col = ['tenure', 'totalcharges', 'monthlycharges']

    categorical_col.remove('customerid')
    feature_cols = categorical_col + numerical_col

    return data, feature_cols

app = Flask('Churn')

@app.route("/predict", methods = ['POST'])
def predict():

    customer = request.get_json()
    data = data_loader(customer)
    data, feature_cols = file_data_prep(data)
    encoded_data = data_encoder(data, feature_cols)
    prediction = loaded_model.predict(encoded_data)
    data['churn'] = prediction
    output = data[data['churn'] == 1]
    output.to_csv('data/churn.csv', index = False)
    output_text = {'Prediction Status': 'The model successufully predicted the customers that are likely to churn and save the file'}
    output = {'Prediction Shape':prediction.shape}
    return jsonify(output_text)

if __name__ == '__main__':
    app.run(debug=True, port=5080)
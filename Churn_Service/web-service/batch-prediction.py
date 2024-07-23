# Importing Libraries

import os
import mlflow
import pandas as pd
from flask import Flask, request, jsonify


child_directory = os.getcwd()
parent_directory = os.path.dirname(child_directory)

MLFLOW_TRACKING_URI = f"sqlite:///{parent_directory}/model/mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


model_name = "Custormer-churn-models"
model_stage = "Production"
loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")

# Load Data


def load_data(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.replace(" ", "_").str.lower()

    categorical_col = data.dtypes[data.dtypes == "object"].index.tolist()
    for col in categorical_col:
        data[col] = data[col].str.replace(" ", "_").str.lower()

    data = data[data["totalcharges"] != "_"]
    data["totalcharges"] = data["totalcharges"].astype("float32")
    return data


# Data Preparation


def prepare_data(data):

    categorical_col = data.dtypes[data.dtypes == "object"].index.tolist()
    numerical_col = ["tenure", "totalcharges", "monthlycharges"]

    categorical_col.remove("customerid")
    feature_cols = categorical_col + numerical_col

    df_data = data[feature_cols].to_dict(orient="records")
    return data, df_data


# Define Service for batch prediction

app = Flask("Churn")


@app.route("/predict", methods=["POST"])
def predict():

    customer = request.get_json()
    data = load_data(customer)
    data, df = prepare_data(data)
    prediction = loaded_model.predict(df)
    data["churn"] = prediction
    output = data[data["churn"] == 1]
    output.to_csv("predictions/churn.csv", index=False)
    output_text = {
        "Prediction Status": "The model successufully predicted the customers that are likely to churn and save the file"
    }
    return jsonify(output_text)


if __name__ == "__main__":
    app.run(debug=True, port=5080)

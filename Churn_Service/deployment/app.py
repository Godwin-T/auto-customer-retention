# import mlflow
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from utils import MODEL_PATH

# from pymongo import MongoClient


def load_model(model_path):

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def prepare_data(data):

    data = pd.DataFrame(data)
    data.columns = data.columns.str.lower()

    data.columns = data.columns.str.replace(" ", "_").str.lower()

    categorical_col = data.dtypes[data.dtypes == "object"].index.tolist()
    for col in categorical_col:
        data[col] = data[col].str.replace(" ", "_").str.lower()
    customer_id = data.pop("customerid")

    return customer_id, data


app = Flask("Churn")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()
    model = load_model(MODEL_PATH)
    customer_id, record = prepare_data(data)
    record = record.to_dict(orient="records")

    prediction = model.predict(record)
    prediction = [str(pred) for pred in prediction]
    customer_id = [str(id) for id in customer_id]
    output = {"customerid": customer_id, "churn": prediction}
    return jsonify(output)


# if __name__ == "__main__":
#     app.run(debug=True, port = 5080)

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")
# mongo_client = MongoClient(MONGODB_ADDRESS)
# db = mongo_client.get_database("prediction_service")
# collection = db.get_collection("data")
# save_to_db(record, bool(prediction))
# send_to_evidently_service(record, bool(prediction))

"""
Routes and utilities for model training
"""
from flask import jsonify, request
from sklearn.model_selection import train_test_split

from common.database import get_db_engine, initialize_mlflow
from config import load_config
from utils import pull_data_from_db
from training.models import linear_model, Tree


def process_data(tablename):
    """Process data for model training"""
    config = load_config()
    dbengine = get_db_engine()

    data = pull_data_from_db(dbengine, tablename)
    dframe = data.copy()
    if "date" in dframe.columns:
        dframe.drop(["date"], axis=1, inplace=True)
    customerid = dframe.pop("customerid") if "customerid" in dframe.columns else None

    y = dframe["churn"]
    X = dframe.drop(["churn"], axis=1)
    X = X.to_dict(orient="records")

    output_dframe = train_test_split(
        X,
        y,
        test_size=config["training_config"]["parameters"]["test_size"],
        random_state=11,
    )
    return customerid, output_dframe


# def train_model():
#     """Train model endpoint"""
#     initialize_mlflow()

#     params = request.get_json() or {}
#     job_id = params.get("job_id") if params else None

#     if job_id:
#         models = params.get("models", ["linear"])
#         problem_type = params.get("problem_type", "classification")

#         _, data = process_streamlit_data(job_id=job_id)

#         if "linear" in models:
#             linear_model(data)

#         if "random_forest" in models:
#             tree = Tree(data)
#             tree.train("randomforest")

#         # if "xgboost" in models:
#         #     xgb = XGBoost(data)
#         #     xgb.train("xgboost")

#     else:
#         _, data = process_data("processdata")
#         linear_model(data)

#         # Uncomment these if you want to train tree models and XGBoost
#         # tree = Tree(data)
#         # tree.train("randomforest")
#         # tree = Tree(data)
#         # tree.train("decisiontree")
#         # xgboost = XGBoost(data)
#         # xgboost.train("xgboost")

#     return jsonify({"status": "success", "response": "Model Training Complete"})


def train_model():
    """Train model endpoint"""
    initialize_mlflow()

    if request.method == "POST":
        params = request.get_json() or {}
    elif request.method == "GET":
        params = request.args.to_dict()
    else:
        params = {}

    job_id = params.get("job_id")

    if job_id:
        models = params.get("models", ["linear"])
        if isinstance(models, str):
            models = models.split(",")  # in case it's a comma-separated string in GET

        # problem_type = params.get("problem_type", "classification")

        _, data = process_streamlit_data(job_id=job_id)

        if "linear" in models:
            linear_model(data)

        if "random_forest" in models:
            tree = Tree(data)
            tree.train("randomforest")

        # if "xgboost" in models:
        #     xgb = XGBoost(data)
        #     xgb.train("xgboost")

    else:
        _, data = process_data("processdata")
        linear_model(data)

        # Uncomment these if you want to train tree models and XGBoost
        # tree = Tree(data)
        # tree.train("randomforest")
        # tree = Tree(data)
        # tree.train("decisiontree")
        # xgboost = XGBoost(data)
        # xgboost.train("xgboost")

    return jsonify({"status": "success", "response": "Model Training Complete"})


def process_streamlit_data(job_id=None):
    """Process data for model training"""
    config = load_config()
    dbengine = get_db_engine()
    tablename = config["data"]["streamlit"]["processed_data"]

    data = pull_data_from_db(dbengine, tablename, job_id=job_id)
    dframe = data.copy()
    dframe.drop(["job_id"], axis=1, inplace=True)

    if "date" in dframe.columns:
        dframe.drop(["date"], axis=1, inplace=True)

    customerid = dframe.pop("customerid") if "customerid" in dframe.columns else None

    y = dframe["churn"]
    X = dframe.drop(["churn"], axis=1)
    X = X.to_dict(orient="records")

    output_dframe = train_test_split(
        X,
        y,
        test_size=config["training_config"]["parameters"]["test_size"],
        random_state=11,
    )
    return customerid, output_dframe

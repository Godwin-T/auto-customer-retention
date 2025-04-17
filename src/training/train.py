# Import Libraries

import os
import yaml
import mlflow

from hyperopt.pyll import scope
from hyperopt import hp, STATUS_OK, fmin, Trials, tpe

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from dotenv import load_dotenv
from sklearn.pipeline import make_pipeline

from flask import Flask, request, jsonify
import warnings

warnings.filterwarnings("ignore")
from utils import evaluate_model, pull_data_from_db, connect_sqlite


load_dotenv()


def load_config():

    config_path = os.getenv("config_path")
    with open(config_path) as config:
        config = yaml.safe_load(config)
    return config


def get_engine():
    config = load_config()
    customer_data_path = config["database"]["db_path"]
    dbengine = connect_sqlite(customer_data_path)
    return dbengine


def initialize_mlflow():
    """Initialize MLflow with the given configuration"""
    config = load_config()
    tracking_uri = config["database"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["base"]["experiment_name"])


# Load and Process Data
def process_data(tablename):

    config = load_config()
    dbengine = get_engine()

    data = pull_data_from_db(dbengine, tablename)
    dframe = data.copy()
    dframe.drop(["date"], axis=1, inplace=True)
    customerid = dframe.pop("customerid")

    y = dframe["churn"]

    X = dframe.drop(["churn"], axis=1)
    X = X.to_dict(orient="records")

    output_dframe = train_test_split(
        X, y, test_size=config["parameters"]["test_size"], random_state=11
    )
    return customerid, output_dframe


def linear_model(data):

    config = load_config()
    developer = config["base"]["developer"]
    artifact_path = f"{config['base']['artifact_path']}"

    (train_x, test_x, train_y, test_y) = data
    lr_params = config["hyperparameters"]["linear_model"]
    c_values = range(lr_params["min_c"], lr_params["max_c"], lr_params["interval"])

    for val in c_values:

        with mlflow.start_run():
            mlflow.set_tag("developer", developer)
            mlflow.set_tag("model_name", "linearRegression")
            mlflow.log_param("c", val)

            lr_pipeline = make_pipeline(
                DictVectorizer(sparse=False), LogisticRegression(C=val)
            )
            lr_pipeline.fit(train_x, train_y)

            test_pred = lr_pipeline.predict(test_x)
            test_output_eval = evaluate_model(test_y, test_pred)
            mlflow.log_metrics(test_output_eval)
            mlflow.sklearn.log_model(lr_pipeline, artifact_path=artifact_path)


class Tree:
    def __init__(self, data):
        self.fullconfig = load_config()
        self.config = self.fullconfig["hyperparameters"]["tree_models"]
        self.developer = self.fullconfig["base"]["developer"]
        self.artifact_path = f"{self.fullconfig['base']['artifact_path']}"
        self.train_x, self.test_x, self.train_y, self.test_y = data

    def objective(self, params):

        model_name = params["model_name"]
        del params["model_name"]
        with mlflow.start_run():
            mlflow.set_tag("developer", self.developer)
            mlflow.set_tag("model_name", model_name)
            mlflow.log_params(params)

            if model_name == "decisiontree":
                pipeline = make_pipeline(
                    DictVectorizer(sparse=False), DecisionTreeClassifier(**params)
                )

            elif model_name == "randomforest":
                pipeline = make_pipeline(
                    DictVectorizer(sparse=False), RandomForestClassifier(**params)
                )

            else:
                print(f"{model_name} does not exist in models")

            pipeline.fit(self.train_x, self.train_y)
            prediction = pipeline.predict(self.test_x)
            prediction_eval = evaluate_model(self.test_y, prediction)

            mlflow.log_metrics(prediction_eval)
            mlflow.sklearn.log_model(pipeline, artifact_path=self.artifact_path)

        return {"loss": -prediction_eval["f1_score"], "status": STATUS_OK}

    # @log_step("Train Trea Model")
    def train(self, model_name):

        criterion = self.config["criterion"]
        min_depth, max_depth = self.config["min_depth"], self.config["max_depth"]
        min_samples_split, max_samples_split = (
            self.config["min_sample_split"],
            self.config["max_sample_split"],
        )
        min_samples_leaf, max_sample_leaf = (
            self.config["min_sample_leaf"],
            self.config["max_sample_leaf"],
        )

        space = {
            "max_depth": hp.randint("max_depth", min_depth, max_depth),
            "min_samples_split": hp.randint(
                "min_samples_split", min_samples_split, max_samples_split
            ),
            "min_samples_leaf": hp.randint(
                "min_samples_leaf", min_samples_leaf, max_sample_leaf
            ),
            "criterion": hp.choice("criterion", criterion),
            "model_name": model_name,
        }

        best_result = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=Trials(),
        )

        return best_result


class XGBoost:
    def __init__(self, data, num_boost_round=1000, early_stopping_rounds=50):

        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.booster = None
        self.vectorizer = DictVectorizer(sparse=False)

        self.fullconfig = load_config()
        self.config = self.fullconfig["hyperparameters"]["xgboost"]
        self.developer = self.fullconfig["base"]["developer"]
        self.artifact_path = f"{self.fullconfig['base']['artifact_path']}"
        self.train_x, self.test_x, self.train_y, self.test_y = data

    def fit(self, x, y):

        X_sparse = self.vectorizer.fit_transform(x)

        # Create xgb.DMatrix
        dtrain = xgb.DMatrix(X_sparse, label=y)
        self.booster = xgb.train(
            self.config,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            evals=[(dtrain, "train")],
            verbose_eval=50,
        )
        mlflow.xgboost.log_model(self.booster, artifact_path=self.artifact_path)

    def objective(self, config):

        model_name = config["model_name"]
        del config["model_name"]
        with mlflow.start_run():

            mlflow.set_tag("developer", self.developer)
            mlflow.set_tag("model_name", model_name)
            mlflow.log_params(config)

            self.fit(self.train_x, self.train_y)
            prediction = self.predict(self.test_x)
            prediction = (prediction >= 0.5).astype("int")

            prediction_eval = evaluate_model(self.test_y, prediction)
            mlflow.log_metrics(prediction_eval)
        return {"loss": -prediction_eval["f1_score"], "status": STATUS_OK}

    # @log_step("Train Xgboost")
    def inference(self, model_name):

        objective = self.config["objective"]
        metric = self.config["eval_metric"]
        min_learning_rate = self.config["min_learning_rate"]
        max_learning_rate = self.config["max_learning_rate"]
        min_depth, max_depth = self.config["min_depth"], self.config["max_depth"]
        min_child_weight, max_child_weight = (
            self.config["min_child_weight"],
            self.config["max_child_weight"],
        )

        search_space = {
            "max_depth": scope.int(hp.quniform("max_depth", min_depth, max_depth, 3)),
            "learning_rate": hp.loguniform(
                "learning_rate", min_learning_rate, max_learning_rate
            ),
            "min_child_weight": hp.loguniform(
                "min_child_weight", min_child_weight, max_child_weight
            ),
            "objective": objective,
            "eval_metric": metric,
            "seed": 11,
            "model_name": model_name,
        }

        best_result = fmin(
            fn=self.objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=50,
            trials=Trials(),
        )
        return best_result

    def predict(self, X):
        X_sparse = self.vectorizer.transform(X)

        # Create xgb.DMatrix
        dmatrix = xgb.DMatrix(X_sparse)

        # Use the trained model for predictions
        predictions = self.booster.predict(dmatrix)
        return predictions


app = Flask("Model_Training")


@app.route("/train", methods=["GET"])
def main():

    initialize_mlflow()

    _, data = process_data("processdata")
    linear_model(data)

    # tree = Tree(data)
    # tree.train("randomforest")
    # logging.info("Train Random Forest")
    # tree = Tree(data)
    # tree.train("decisiontree")
    # logging.info("Train Decision Tree")
    # xgboost = XGBoost( data)
    # xgboost_result = xgboost.train("xgboost")

    # logging.info("Train Xgboost")

    return jsonify({"response": "Model Training Complete"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8001)

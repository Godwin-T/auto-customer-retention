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

config_path = os.getenv("config_path")
with open(config_path) as config:
    config = yaml.safe_load(config)

customer_data_path = config["database"]["customer"]["database_path"]
dbengine = connect_sqlite(customer_data_path)

seed = config["base"]["random_state"]
developer = config["base"]["developer"]
artifact_path = f"{config['base']['artifact_path']}"

mlflow.set_tracking_uri(config["database"]["tracking"]["tracking_url"])
mlflow.set_experiment(config["database"]["tracking"]["experiment_name"])


# Load and Process Data
def process_data(tablename):

    data = pull_data_from_db(dbengine, tablename)
    dframe = data.copy()
    dframe.drop(["date"], axis=1, inplace=True)
    customerid = dframe.pop("customerid")

    y = dframe["churn"]
    y = y.map({"yes": 1, "no": 0}).astype(int)

    X = dframe.drop(["churn"], axis=1)
    X = X.to_dict(orient="records")

    output_dframe = train_test_split(
        X, y, test_size=config["data"]["test_size"], random_state=seed
    )
    return customerid, output_dframe


def linear_model(data):

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
    def __init__(self, configurations, data):
        self.config = configurations
        self.train_x, self.test_x, self.train_y, self.test_y = data

    def objective(self, params):

        model_name = params["model_name"]
        del params["model_name"]
        with mlflow.start_run():
            mlflow.set_tag("developer", developer)
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
            mlflow.sklearn.log_model(pipeline, artifact_path=artifact_path)

        return {"loss": -prediction_eval["f1_score"], "status": STATUS_OK}

    # @log_step("Train Trea Model")
    def inference(self, model_name):

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
    def __init__(self, params, data, num_boost_round=1000, early_stopping_rounds=50):
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.booster = None
        self.vectorizer = DictVectorizer(sparse=False)
        self.train_x, self.test_x, self.train_y, self.test_y = data

    def fit(self, x, y):

        X_sparse = self.vectorizer.fit_transform(x)

        # Create xgb.DMatrix
        dtrain = xgb.DMatrix(X_sparse, label=y)
        self.booster = xgb.train(
            self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            evals=[(dtrain, "train")],
            verbose_eval=50,
        )
        mlflow.xgboost.log_model(self.booster, artifact_path=artifact_path)

    def objective(self, params):

        model_name = params["model_name"]
        del params["model_name"]
        with mlflow.start_run():

            mlflow.set_tag("developer", developer)
            mlflow.set_tag("model_name", model_name)
            mlflow.log_params(params)

            self.fit(self.train_x, self.train_y)
            prediction = self.predict(self.test_x)
            prediction = (prediction >= 0.5).astype("int")

            prediction_eval = evaluate_model(self.test_y, prediction)
            mlflow.log_metrics(prediction_eval)
        return {"loss": -prediction_eval["f1_score"], "status": STATUS_OK}

    # @log_step("Train Xgboost")
    def inference(self, model_name):

        objective = self.params["objective"]
        metric = self.params["eval_metric"]
        min_learning_rate = self.params["min_learning_rate"]
        max_learning_rate = self.params["max_learning_rate"]
        min_depth, max_depth = self.params["min_depth"], self.params["max_depth"]
        min_child_weight, max_child_weight = (
            self.params["min_child_weight"],
            self.params["max_child_weight"],
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
            "seed": seed,
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

    _, data = process_data("processdata")
    linear_model(data)

    # params = config["hyperparameters"]["tree_models"]
    # tree = Tree(params, data)
    # tree.inference("randomforest")
    # logging.info("Train Random Forest")
    # tree = Tree(params, data)
    # tree.inference("decisiontree")
    # logging.info("Train Decision Tree")
    # params = config["hyperparameters"]["xgboost"]
    # # xgboost = XGBoost(params, data)
    # xgboost_result = xgboost.inference("xgboost")

    # logging.info("Train Xgboost")

    return jsonify({"response": "Model Training Complete"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8001)

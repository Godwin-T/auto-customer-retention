"""
Model definitions and training functions
"""

import logging
import mlflow
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from hyperopt import hp, STATUS_OK, fmin, Trials, tpe
from hyperopt.pyll import scope

from config import load_config
from .evaluate import evaluate

logger = logging.getLogger(__name__)


def linear_model(data):
    """Train logistic regression model with different C values"""
    config = load_config()
    developer = config["training_config"]["base"]["developer"]
    artifact_path = f"{config['training_config']['base']['artifact_path']}"

    (train_x, test_x, train_y, test_y) = data

    lr_params = config["training_config"]["hyperparameters"]["linear_model"]
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
            test_output_eval = evaluate(test_y, test_pred)
            mlflow.log_metrics(test_output_eval)
            mlflow.sklearn.log_model(lr_pipeline, artifact_path=artifact_path)


class Tree:
    """Tree-based models (Decision Tree, Random Forest)"""

    def __init__(self, data):
        self.fullconfig = load_config()
        self.config = self.fullconfig["training_config"]["hyperparameters"][
            "tree_models"
        ]
        self.developer = self.fullconfig["training_config"]["base"]["developer"]
        self.artifact_path = (
            f"{self.fullconfig['training_config']['base']['artifact_path']}"
        )
        self.train_x, self.test_x, self.train_y, self.test_y = data

    def objective(self, params):
        """Hyperopt objective function"""
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

    def train(self, model_name):
        """Train tree model with hyperparameter optimization"""
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
        self.config = self.fullconfig["training_config"]["hyperparameters"]["xgboost"]
        self.developer = self.fullconfig["training_config"]["base"]["developer"]
        self.artifact_path = (
            f"{self.fullconfig['training_config']['base']['artifact_path']}"
        )
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

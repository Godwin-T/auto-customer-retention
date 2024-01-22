import mlflow
from prefect import task

from hyperopt.pyll import scope
from hyperopt import hp, STATUS_OK, fmin, Trials, tpe

from sklearn.feature_extraction import DictVectorizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
from sklearn.pipeline import make_pipeline

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import warnings

warnings.filterwarnings("ignore")


# Set Mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Customer_Churn_Predictions")

# Linear Regression Model
@task
def train_LR(data, c_values=range(1, 100, 10)):

    (train_x, test_x, train_y, test_y) = data

    best_c_value = 0
    for val in c_values:

        with mlflow.start_run():
            mlflow.set_tag("Developer", "Godwin")
            mlflow.set_tag("model", "Logistic Regression")
            mlflow.set_tag("C", val)

            lr_pipeline = make_pipeline(
                DictVectorizer(sparse=False), LogisticRegression(C=val)
            )
            lr_pipeline.fit(train_x, train_y)

            test_pred = lr_pipeline.predict(test_x)
            test_output_eval = evaluate_model(test_y, test_pred)

            mlflow.log_metrics(test_output_eval)
            mlflow.sklearn.log_model(lr_pipeline, artifact_path="models_mlflow")

        if test_output_eval["f1_score"] > best_c_value:
            best_c_value = val

    best_result = {"C": best_c_value}
    metrics = {
        "Accuracy": test_output_eval["accuracy_score"],
        "F1_Score": test_output_eval["f1_score"],
    }
    print(metrics)
    return best_result


# Decision Tree Model
@task
def train_DT(data):

    (train_x, test_x, train_y, test_y) = data

    def objective(params):

        with mlflow.start_run():
            mlflow.set_tag("Developer", "Godwin")
            mlflow.set_tag("model", "DecisionTree")
            mlflow.log_params(params)

            pipeline = make_pipeline(
                DictVectorizer(sparse=False), DecisionTreeClassifier(**params)
            )
            pipeline.fit(train_x, train_y)

            prediction = pipeline.predict(test_x)
            prediction_eval = evaluate_model(test_y, prediction)

            mlflow.log_metrics(prediction_eval)
            mlflow.sklearn.log_model(pipeline, artifact_path="models_mlflow")

        return {"loss": -prediction_eval["f1_score"], "status": STATUS_OK}

    space = {
        "max_depth": hp.randint("max_depth", 1, 15),
        "min_samples_split": hp.randint("min_samples_split", 2, 15),
        "min_samples_leaf": hp.randint("min_samples_leaf", 1, 15),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
    }

    best_result = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=Trials()
    )
    return best_result


# Random Forest Model
@task
def train_RF(data):

    (train_x, test_x, train_y, test_y) = data

    def objective(params):

        with mlflow.start_run():
            mlflow.set_tag("Developer", "Godwin")
            mlflow.set_tag("model", "RandonForest")
            mlflow.log_params(params)

            pipeline = make_pipeline(
                DictVectorizer(sparse=False), RandomForestClassifier(**params)
            )
            pipeline.fit(train_x, train_y)

            prediction = pipeline.predict(test_x)
            prediction_eval = evaluate_model(test_y, prediction)

            mlflow.log_metrics(prediction_eval)
            mlflow.sklearn.log_model(pipeline, artifact_path="models_mlflow")

        return {"loss": -prediction_eval["f1_score"], "status": STATUS_OK}

    space = {
        "n_estimators": hp.choice(
            "n_estimators",
            [
                2,
                5,
                10,
                20,
                30,
                50,
                100,
            ],
        ),
        "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 5)),
        #'min_samples_split': hp.randint("min_samples_split", 2, 15),
        # 'min_samples_leaf': hp.randint("min_samples_leaf", 1, 15),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
    }

    best_result = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=Trials()
    )
    return best_result


# Define Xgboost
class XGBoostTrainer:
    def __init__(self, params, num_boost_round=1000, early_stopping_rounds=50):
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.booster = None
        self.dict_vectorizer = DictVectorizer(sparse=False)

    def fit(self, X, y):
        # Assuming X, y are your training data and labels
        # Convert the input features to a sparse matrix using DictVectorizer

        X_sparse = self.dict_vectorizer.fit_transform(X)

        # Create xgb.DMatrix
        dtrain = xgb.DMatrix(X_sparse, label=y)

        # Train the XGBoost model
        self.booster = xgb.train(
            self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            evals=[(dtrain, "train")],
            verbose_eval=50,
        )
        mlflow.xgboost.log_model(self.booster, artifact_path="models_mlflow")

        return self

    def predict(self, X):
        # Convert the input features to a sparse matrix using DictVectorizer
        X_sparse = self.dict_vectorizer.transform(X)

        # Create xgb.DMatrix
        dmatrix = xgb.DMatrix(X_sparse)

        # Use the trained model for predictions
        predictions = self.booster.predict(dmatrix)
        return predictions


# Xgboost Model
@task
def train_XGB(data):

    (train_x, test_x, train_y, test_y) = data

    def objective(params):

        with mlflow.start_run():

            mlflow.set_tag("Developer", "Godwin")
            mlflow.set_tag("model", "Xgboost")

            mlflow.log_params(params)

            booster = make_pipeline(XGBoostTrainer(params=params))

            booster.fit(train_x, train_y)
            prediction = booster.predict(test_x)
            prediction = (prediction >= 0.5).astype("int")

            prediction_eval = evaluate_model(test_y, prediction)
            mlflow.log_metrics(prediction_eval)

        return {"loss": -prediction_eval["f1_score"], "status": STATUS_OK}

    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 3)),
        "learning_rate": hp.loguniform("learning_rate", -3, 0),
        "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "seed": 42,
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials(),
    )
    return best_result


def evaluate_model(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred)

    out = {
        "accuracy_score": accuracy,
        "precision_score": precision,
        "recall_score": recall,
        "f1_score": f1score,
    }
    return out

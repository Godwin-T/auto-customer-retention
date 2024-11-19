import os
import json

from prefect import task
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Define Model Evaluation Metrics
def evaluate(y_true, prediction):

    f1 = f1_score(y_true, prediction)
    metrics = {
        "acc": accuracy_score(y_true, prediction),
        "f1_score": f1,
        "precision": precision_score(y_true, prediction),
        "recall": recall_score(y_true, prediction),
    }
    return metrics


# Define Model Evaluation Function
@task(name="Evaluate Model")
def evaluate_model(model, X_test, y_test, float_precision=4):

    X_test = X_test.to_dict(orient="records")
    prediction = model.predict(X_test)
    evaluation_result = evaluate(y_test, prediction)

    evaluation_result = json.loads(
        json.dumps(evaluation_result),
        parse_float=lambda x: round(float(x), float_precision),
    )
    return evaluation_result, prediction

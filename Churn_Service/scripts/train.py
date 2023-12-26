# Importing Libraries
print("Importing Libraries")
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from model import evaluate_model, train_model, save_model
from metrics import save_metrics, save_predictions
from utils_and_constants import PROCESSED_DATASET, TARGET_COLUMN


def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    return X, y


def main():

    X, y = load_data(PROCESSED_DATASET)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1993)

    model, train_eval_result = train_model(X_train, y_train)
    test_eval_result, y_pred = evaluate_model(model, X_test, y_test)
    model_evaluation_result = {
        "Train evaluation result": train_eval_result,
        "Test evaluation result": test_eval_result,
    }

    print("====================Train Set Metrics==================")
    print(json.dumps(train_eval_result, indent=2))
    print("======================================================")
    print()
    print("====================Test Set Metrics==================")
    print(json.dumps(test_eval_result, indent=2))
    print("======================================================")

    save_metrics(model_evaluation_result)
    save_predictions(y_test, y_pred)
    save_model(model)


if __name__ == "__main__":
    main()

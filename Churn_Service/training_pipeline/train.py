# Importing Libraries
print("Importing Libraries")
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from model import evaluate_model, train_model, save_model
from metrics import save_metrics, save_predictions
from constants import PROCESSED_DATASET, TARGET_COLUMN


def load_data(file_path):
    # Read the CSV file and split into features (X) and target variable (y)
    data = pd.read_csv(file_path)
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    return X, y


def main():
    # Load the processed dataset and split into train and test sets
    X, y = load_data(PROCESSED_DATASET)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1993)

    # Train the model and get the evaluation results on the training set
    model, train_eval_result = train_model(X_train, y_train)

    # Evaluate the model on the test set and get the evaluation results and predictions
    test_eval_result, y_pred = evaluate_model(model, X_test, y_test)

    # Combine the evaluation results for both the training and test sets
    model_evaluation_result = {
        "Train evaluation result": train_eval_result,
        "Test evaluation result": test_eval_result,
    }

    # Print the training set evaluation results
    print("====================Train Set Metrics==================")
    print(json.dumps(train_eval_result, indent=2))
    print("======================================================")
    print()

    # Print the test set evaluation results
    print("====================Test Set Metrics==================")
    print(json.dumps(test_eval_result, indent=2))
    print("======================================================")

    # Save the overall model evaluation results, test set predictions, and the trained model
    save_metrics(model_evaluation_result)
    save_predictions(y_test, y_pred)
    save_model(model)


if __name__ == "__main__":
    main()

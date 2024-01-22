import pandas as pd
from prefect import task, flow
from sklearn.model_selection import train_test_split
from model import train_LR, train_DT, train_RF, train_XGB
from utils import PROCESSED_DATASET


# Load Data
@task
def load_data(path):

    data = pd.read_csv(path)
    y = data["churn"]
    X = data.drop(["churn"], axis=1)
    return (X, y)


@task
def split_data(data):

    X, y = data
    X = X.to_dict(orient="record")
    (train_x, test_x, train_y, test_y) = train_test_split(
        X, y, test_size=0.3, random_state=1993
    )
    return (train_x, test_x, train_y, test_y)


@flow(name="Training and Model Evaluation")
def main():
    # Load the processed dataset and split into train and test sets
    data = load_data(PROCESSED_DATASET)
    data = split_data(data)

    # Train the model and get the evaluation results on the training set
    print("Training Linear Regression..")
    lr_best_params = train_LR(data)
    print("Successfully Trained Linear Regression")
    print()

    print("Training Decision Tree..")
    dt_best_params = train_DT(data)
    print("Successfully Trained Decision Tree")
    print()

    print("Training Random Forest..")
    rf_best_params = train_RF(data)
    print("Successfully Trained Random Forest")
    print()

    print("Training Xgboost..")
    xgb_best_params = train_XGB(data)
    print("Successfully Trained Xgboost")
    print()

    print("Best Linear Regression Parameters")
    print(lr_best_params)
    print()
    print("=========================================================================")

    print("Best Decision Tree Parameters")
    print(dt_best_params)
    print()
    print("=========================================================================")

    print("Best Random Forest Parameters")
    print(rf_best_params)
    print()
    print("=========================================================================")

    print("Best Xgboost Parameters")
    print(xgb_best_params)
    print()
    print("=========================================================================")


if __name__ == "__main__":
    main()

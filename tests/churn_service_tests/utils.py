import pickle
import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data():

    data_path = "../Churn_Service/data/raw_data/Telco-Customer-Churn.csv"
    df = pd.read_csv(data_path)
    return df


def load_processed_data():

    data_path = "../Churn_Service/data/processed_data/churn.csv"
    df = pd.read_csv(data_path)
    y, X = df.pop("churn"), df
    return X, y


def split_data():

    df = load_raw_data()
    (
        X_train,
        X_test,
    ) = train_test_split(df, test_size=0.24, random_state=25)
    return (X_train, X_test)


def load_model():

    model_path = "../Churn_Service/models/churn_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def predict(data):

    model = load_model()
    data = data.to_dict(orient="records")
    prediction = model.predict(data)
    return prediction

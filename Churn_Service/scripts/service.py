import pandas as pd
import requests


def predict_churn(url, data_path):

    data_frame = pd.read_csv(data_path)
    data_dict = data_frame.to_dict()

    response = requests.post(url, json=data_dict).json()
    ids = response["customerid"]
    prediction = [int(i) for i in response["churn"]]
    dicts = {"customerid": ids, "churn": prediction}
    output_frame = pd.DataFrame(dicts)

    return output_frame

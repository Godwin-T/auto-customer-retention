# Import Libraries

import os
import json
from datetime import datetime

from time import sleep
import pandas as pd
import requests


child_dir = os.getcwd()
parent_dir = os.path.dirname(child_dir)
file_name = "Telco-Customer-Churn.csv"

file_path = f"{parent_dir}/data/{file_name}"

data = pd.read_csv(file_path)
data = data.to_dict(orient="records")


class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


# Send Data
with open("target.csv", "w") as f_target:

    url = "http://127.0.0.1:9696/predict"
    for row in data:
        print("===============")
        response = requests.post(url, json=row).json()
        print(
            f"Customer ID: {response['customerid']}      Prediction: {response['churn']}"
        )
        sleep(1)

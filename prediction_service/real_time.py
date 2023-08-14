import os 
import json
import uuid
from datetime import datetime

from time import sleep
import pandas as pd
import requests

file_name = 'Telco-Customer-Churn.csv'
directory = os.getcwd()
data_path = f'{directory}/{file_name}'

table = pd.read_csv(data_path)
data = table.to_dict(orient= 'records')

class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


with open("target.csv", 'w') as f_target:

    url = "http://127.0.0.1:9696/predict"
    for row in data:
        print('===============')
        response = requests.post(url, json = row).json() 
        print(f"prediction: {response['churn']}, 'Dir': {response['Dir']}")
        sleep(1)
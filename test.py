import json
import requests
import pandas as pd

url = "http://127.0.0.1:9696/predict"
data_path = "/home/godwin/Documents/Workflow/Customer-retention/data/raw_data/Churn.csv"
data = pd.read_csv(data_path)
data = data.to_dict()
output = requests.post(url, json=data).json()

print(output["Response"])

import os
import requests

child_directory = os.getcwd()
parent_directory = os.path.dirname(child_directory)

data_path = f'{parent_directory}/data/new data.csv'
url = "http://127.0.0.1:5080/predict"
response = requests.post(url, json = data_path).json()
print(response)
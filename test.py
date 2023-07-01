import requests

data_path = 'data/new data.csv'
url = "http://127.0.0.1:5080/predict"
response = requests.post(url, json = data_path).json()
print(response)
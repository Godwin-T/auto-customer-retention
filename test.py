import requests

url = "http://127.0.0.1:8000/process"

response = requests.get(url)
print(response)
print(response.json())

url = "http://127.0.0.1:8001/train"

response = requests.get(url)
print(response)
print(response.json())

url = "http://127.0.0.1:8002/deploy-auto"

response = requests.get(url)
print(response)
print(response.json())

url = "http://127.0.0.1:8002/predict"
response = requests.get(url)
print(response)
print(response.json())

import requests
from requests.exceptions import JSONDecodeError

urls = [
    "http://127.0.0.1:8000/process",
    "http://127.0.0.1:8001/train",
    "http://127.0.0.1:8002/deploy-auto",
    "http://127.0.0.1:8002/predict",
]

for url in urls:
    resp = requests.get(url)
    print(resp)
    try:
        print(resp.json())
    except JSONDecodeError:
        print("✗ failed to decode JSON, response text was:")
        print(resp.text)
    print()
import pandas as pd
import requests

url = "http://127.0.0.1:5050/predict"
data_path = "../raw_data/Telco-Customer-Churn.csv"

data_frame = pd.read_csv(data_path)
data_dict = data_frame.to_dict()

response = requests.post(url, json=data_dict).json()
ids = response["customerid"]
prediction = [int(i) for i in response["churn"]]
dicts = {"customerid": ids, "churn": prediction}
output_frame = pd.DataFrame(dicts)
# output_frame.to_csv("./Attrition.csv", index=False)
print(output_frame.shape)

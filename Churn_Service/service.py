import pandas as pd
import requests


def predict_churn(url: str, data_path):
    # Read the CSV file into a pandas DataFrame
    data_frame = pd.read_csv(data_path)

    # Convert the DataFrame to a dictionary
    data_dict = data_frame.to_dict()

    # Make a POST request to the specified URL with the data in JSON format
    response = requests.post(url, json=data_dict).json()

    # Extract customer IDs and churn predictions from the response
    ids = response["customerid"]
    prediction = [int(i) for i in response["churn"]]

    # Create a dictionary with customer IDs and churn predictions
    dicts = {"customerid": ids, "churn": prediction}

    # Create a new DataFrame from the dictionary
    output_frame = pd.DataFrame(dicts)

    # Return the DataFrame containing customer IDs and churn predictions
    return output_frame

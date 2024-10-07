import pandas as pd
import requests


import streamlit as st
from prefect import task, flow
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# from ChurnGuard.training_pipeline.constants import PREDICTION_URL
# from ChurnGuard.service import predict_churn
# from PromoGen.scripts.constant import API_KEY, COMPLETION_MODEL_NAME
# from PromoGen.scripts.utils import mail_generation

# chatllm = ChatOpenAI(temperature=0.5, openai_api_key=API_KEY)
# model = OpenAI(model=COMPLETION_MODEL_NAME, temperature=0.5, openai_api_key=API_KEY)


# def generate_emails():
#     st.subheader("Generate Promotional Mail")
#     mail_generation(chatllm)


def homepage():
    st.title("Customer Retention Service")
    st.write(
        """Welcome to our Customer Retention Service. This service specializes in predicting the\
             likelihood of customer churn for a telecommunications company. Additionally, it generates \
            promotional emails tailored to individual customer contexts, enhancing targeted engagement."""
    )

    st.write(
        """This service offers diverse usage options:\n 1. Generate promotional emails using pre-existing\
              contexts.\n 2. Engage with the chatbot to gather insights and create contexts for personalized promotional\
              emails.\n 3. Predict the likelihood of customer churn with advanced analytics."""
    )


# def generate_mails_page():
#     st.title("Generate promotional emails using pre-existing")
#     st.write("Welcome to the Mail Generation Service")
#     generate_emails()


# def churn_prediction():

#     st.subheader("Predict the likelihood of customer churn with advanced analytics.")
#     uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
#     if uploaded_file:
#         churn = predict_churn(PREDICTION_URL, uploaded_file)
#         csv_data = churn.to_csv(index=False)  # Convert DataFrame to CSV string
#         st.download_button(
#             label="Download CSV",
#             data=csv_data,
#             file_name="my_data.csv",
#             mime="text/csv",
# )

# def predict_churn_page():
#     st.title("Predict the likelihood of customer churn with advanced analytics.")
#     st.write("Welcome to the Churn Prediction Service")
#     churn_prediction()


def predict_churn(url: str, data_path):
    # Read the CSV file into a pandas DataFrame
    data_frame = pd.read_csv(data_path)

    # Convert the DataFrame to a dictionary
    data_dict = data_frame.to_dict()

    # Make a POST request to the specified URL with the data in JSON format
    response = requests.post(url, json=data_dict).json()

    # Extract customer IDs and churn predictions from the response
    # ids = response["customerid"]
    # prediction = [int(i) for i in response["churn"]]

    # # Create a dictionary with customer IDs and churn predictions
    # dicts = {"customerid": ids, "churn": prediction}

    # # Create a new DataFrame from the dictionary
    # output_frame = pd.DataFrame(dicts)

    # Return the DataFrame containing customer IDs and churn predictions
    return response["Response"]


url = "http://127.0.0.1:9696/predict"
data_path = "../ChurnGuard/data/churn-data/raw_data/Telco-Customer-Churn.csv"
if __name__ == "__main__":
    response = predict_churn(url, data_path)
    print(response)

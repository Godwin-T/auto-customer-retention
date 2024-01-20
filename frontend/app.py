import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from frontend.service import (
    homepage,
    generate_mails_page,
    chatbot_page,
    predict_churn_page,
    mail_revamp_page,
)


def main():
    st.sidebar.title("Navigation")
    page_options = ["Home", "Generate Mail", "Chatbot", "Predict Churn", "Revamp Mail"]
    selected_page = st.sidebar.radio("Select a service", page_options)

    if selected_page == "Home":
        homepage()
    if selected_page == "Generate Mail":
        generate_mails_page()
    elif selected_page == "Chatbot":
        chatbot_page()
    elif selected_page == "Predict Churn":
        predict_churn_page()
    elif selected_page == "Revamp Mail":
        mail_revamp_page()


# Run the Streamlit app
if __name__ == "__main__":
    main()

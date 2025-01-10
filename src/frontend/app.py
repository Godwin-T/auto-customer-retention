import os
import sqlite3
import requests
import streamlit as st
import pandas as pd
from src.frontend.HtmlTemplate import user_template, bot_template, chat_css
from src.backend.mail_blitz.utils import query_or_mail

# from src.backend.mail_blitz.churnreach import executives_crew, customer_crew
# from src.backend.mail_blitz.utils import llm_agent


def update_database(dataframe):

    db_path = "local.db"

    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        # Append new records to the 'users' table
        dataframe.to_sql("customers", conn, if_exists="append", index=False)
        conn.close()
    print("Database updated successfully")


def make_prediction(dataframe):

    url = "http://127.0.0.1:9696/predict"
    data = dataframe.to_dict()
    output = requests.post(url, json=data).json()
    print(output)


# Set the configuration for the Streamlit app
st.set_page_config(page_title="Customer Intel", page_icon="")
st.markdown(chat_css, unsafe_allow_html=True)

# Function to handle user input and chatbot response
def handle_userinput(user_query):

    # Iterate through the chat history (starting from the second message)
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            # Display user messages using the user template
            st.write(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)
        else:
            # Display bot messages using the bot template
            st.write(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)

    # Get the response from the chatbot using 'crew' module
    # user_mail_intent = query_or_mail(user_question)
    # if user_mail_intent == "Yes":
    #     response = customer_crew.kickoff({"query": user_question})
    # else:
    #     response = executives_crew.kickoff({"query": user_question})

    response = 121212  # llm_agent(st.session_state.chat_history, user_query)
    st.session_state.chat_history.append(response)
    # Display the bot's response using the bot template
    st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)


# Main function to run the Streamlit app
def main():
    # Initialize chat history in session state
    st.session_state.chat_history = []  # Initial message to indicate chat start

    # Capture user input from a chat input box

    user_question = st.chat_input("Enter your message:", key="chat_input")
    if user_question:  # Check if the user has entered a message
        st.session_state.chat_history.append(
            user_question
        )  # Append user message to chat history
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        excel_docs = st.file_uploader(
            "Upload your excel file to update database",
            accept_multiple_files=True,
            type={"csv"},
        )
        dataframe = pd.DataFrame()

        if st.button("Update Database"):
            with st.spinner("Processing"):

                for doc in excel_docs:
                    data = pd.read_csv(doc)
                    dataframe = pd.concat([dataframe, data], axis=1)

            print(dataframe.shape)
            update_database(dataframe)
            # get pdf text

            # raw_text = get_pdf_text(pdf_docs)

            # # get the text chunks
            # text_chunks = get_text_chunks(raw_text)

            # # create vector store
            # vectorstore = get_vectorstore(text_chunks)

            # # create conversation chain
            # st.session_state.conversation = get_conversation_chain(
            # vectorstore)
        if st.button("Make Prediction"):
            with st.spinner("Processing"):
                for doc in excel_docs:
                    data = pd.read_csv(doc)
                    dataframe = pd.concat([dataframe, data], axis=1)
                make_prediction(dataframe)


# Run the main function when the script is executed
if __name__ == "__main__":
    main()

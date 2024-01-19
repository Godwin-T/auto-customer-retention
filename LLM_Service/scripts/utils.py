import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.memory import ChatMessageHistory


def get_context():
    # Function to retrieve the context for mail generation
    context = "Generate a mail for one of our customers in a relationship informing them of a network upgrade coming to the area soon"
    return context


def mail_generation(model, context):
    # Function to generate a promotional mail based on a given context using a language model
    prompt = PromptTemplate(
        input_variables=["description"],
        template="We are a telecommunication company and we are trying to prevent customers from churning. Generate a promotional mail for one of our customer using information in {description} as context and it should have a subject relating to the context",
    )
    output = model.invoke(prompt.format(description=context))
    return output


def mail_revamp(model, mail, corrections):
    # Function to suggest corrections and revamp a generated mail
    prompt = PromptTemplate(
        input_variables=["mail", "corrections"],
        template="The {mail} you generated is fine but can these {corrections} be made to make it better",
    )
    output = model.invoke(prompt.format(mail=mail, corrections=corrections))
    return output


def chat_mode(model):
    # Function for interactive chat mode with a language model
    chat_history = ChatMessageHistory()

    # Initialize the chat messages in the session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Receive user input and add it to the chat history
    prompt = st.chat_input("Enter your message:", key="chat_input")
    if prompt:
        chat_history.add_user_message(prompt)
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate a response from the language model based on the chat history
        response = model(chat_history.messages).content
        chat_history.add_ai_message(response)

        # Display the assistant's response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

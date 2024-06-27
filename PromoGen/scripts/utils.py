import streamlit as st
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ChatMessageHistory


def mail_generation(model):

    # Function to generate a promotional mail based on a given context using a language model
    chat_histor = ChatMessageHistory()

    # Initialize the chat messages in the session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Receive user input and add it to the chat history
    user_input = st.chat_input("Enter your message:", key="chat_input")
    if user_input:
        chat_histor.add_user_message(user_input)
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate a response from the language model based on the chat history
        template = """You are a telecommunication company chatbot and your primary job includes;

                    1. Generating mail subject ideas for the company
                    2. Generating mails for the company based on a given subject and other informations provided
                    3. Revamping emails to suit our agents.
                    You are to interact with our agent first to know what they need before going into full fuctionality.

                   Ask the user if the mail is now okay. If yes, thank the user and let them know the email will be sent to the customers.
                    """

        prompt = ChatPromptTemplate.from_messages(
            [("system", template), ("user", "{input}")]
        )

        response = model.invoke(prompt.format(input=chat_histor.messages))
        response = response.content  # .split(':')[1]
        chat_histor.add_ai_message(response)

        # Display the assistant's response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # return output


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
    user_input = st.chat_input("Enter your message:", key="chat_input")
    if user_input:
        chat_history.add_user_message(user_input)
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate a response from the language model based on the chat history
        template = """You are a chatbot and your job is to generate promotional mail subject ideas for a telecomunication company \
                      to make them retain their customers. Ask if the ideas you generated are okay if yes ask which subject he his selecting. \
                      Don't make suggestion before he requests for it and also ask if he has any thing in mind. If yes, generate ideas for him \
                      based on what he says and if not make suggestion for him.
                      When he picks an option, don't generate more options again. Just ask if him if he wants to generate a promotional mail around the subject. \
                      If yes, tell him you will be redirecting him to the mail generation page"""

        prompt = ChatPromptTemplate.from_messages(
            [("system", template), ("user", "{input}")]
        )

        response = model.invoke(prompt.format(input=chat_history.messages)).content
        chat_history.add_ai_message(response)

        # Display the assistant's response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        reset_button = st.button("Reset Chat")
        if reset_button:
            del st.session_state[chat_history]

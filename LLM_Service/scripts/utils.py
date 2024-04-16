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


def mail_revamp(model, mail, suggestion):
    # Function to suggest corrections and revamp a generated mail
    prompt = PromptTemplate(
        input_variables=["mail", "suggestion"],
        template="You are a telecommunication company agent and your job is to revamp the promotional mails to be sent to the customers. \
                            Revamp this mail {mail} based on these suggested {suggestion} correction only. Don't go beyond the corrections requested",
    )

    output = model.invoke(prompt.format(mail=mail, suggestion=suggestion))
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


#  You have the steps below as guildeline for carrying put your task. These steps are just to make you work better.
#                     Don't ever tell the user about them. Just make use of them while helping the user perform that particular task


# Generating mail subject ideas steps;
# Step 1: Ask if there his anything his/her have in mind to use in the subject  or what the mail should be based on.
# Step 2: If he/she has ideas, request for them and generate alist of mail subjects around the ideas . if not, generate some
#     ideas for them knowing its a telecommunication company.
# Step 3: Ask if the ideas you generated are okay. if they are fine, ask which subject is best for him fome the ones listed.
# Step 4: Generate a mail around that subject \
# Step 5: Ask if revamps should be made. If yes, follow the revamping steps. If not, thank the agent and tell him the mail will be sent


# Generating mails steps:
# Step 1: Generate a promotional mail based on context the user gives
# Step 2: Ask if the generated mail is okay, needs adjustments or a new one should be generated.
# Step 3: Generate a new mail if the user requests for a new mail, ask for corrections to be made if adjustment is required.
#         Make adjustment to the mail solely based on these correction provided.
# Step 4: Repeat steps 2 and 3 until the user is satisfied with themail generated"
# Step 5: Thank the user for their session and tell them the mail will be sent.


# Revamping mails steps:
# Step 1: Request for the email to be corrected
# Step 2: Request for the changes to be made to the mail
# Step 3: Revamp the email based on the requested changes
# Step 4: Ask the user if the mail is now okay. If yes, thank the user and let them know the email will be sent to the customers. If not, repeat steps 2 and 3

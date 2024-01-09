import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.memory import ChatMessageHistory


def get_context():

    context = "Generate a mail for one of our customers in a relationship informing them of a network upgrade coming to the area soon"
    return context


def mail_generation(model, context):

    prompt = PromptTemplate(
        input_variables=["description"],
        template="We are a telecommunication company and we are trying to prevent customers from churning\
                                . Generate a promotional mail for one of our customer using information in {description} as \
                                    context and it should have a subject relating to the context",
    )
    output = model.invoke(prompt.format(description=context))
    return output


def mail_revamp(model, mail, corrections):

    prompt = PromptTemplate(
        input_variables=["mail" "corrections"],
        template="The {mail} you generated is fine but can these {corrections} be made to make it better",
    )
    output = model.invoke(prompt.format(mail=mail, corrections=corrections))
    return output


def chat_mode(model):

    chat_history = ChatMessageHistory()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Enter your message:", key="chat_input")
    if prompt:
        chat_history.add_user_message(prompt)
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = model(chat_history.messages).content
        chat_history.add_ai_message(response)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

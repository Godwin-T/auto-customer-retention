import streamlit as st

# CSS for the chat interface
chat_css = """
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    margin: 0 auto;
    max-width: 1000px;
}

.chat-message {
    display: flex;
    margin: 10px 0;
}

.chat-message.bot {
    justify-content: flex-start;
}

.chat-message.user {
    justify-content: flex-end;
}

.message {
    max-width: 60%;
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.1);
}

.messagebot {
    color: white;
    max-width: 85%;
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.1);

}

.messageuser {
    background-color: #0084ff;
    color: white;
    max-width: 60%;
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.1);
}
</style>
"""

# Chat template
bot_template = """
<div class="chat-message bot">
    <div class="messagebot">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="messageuser">{{MSG}}</div>
</div>
"""

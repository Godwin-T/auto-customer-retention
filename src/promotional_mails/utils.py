import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = "Your Key"
llm = ChatGroq(api_key=groq_key, model="llama3-70b-8192")


def instantiate_db(customer_data_path, prediction_data_path):

    db_path = "local.db"

    if os.path.exists(db_path):
        engine = create_engine(f"sqlite:///{db_path}")
    else:
        customerdf = pd.read_csv(customer_data_path)
        predictiondf = pd.read_csv(prediction_data_path)
        engine = create_engine("sqlite:///local.db")
        customerdf.to_sql("customers", engine, index=False)
        predictiondf.to_sql("customerchurn", engine, index=False)
    db = SQLDatabase(engine=engine)
    return db


# @tool("fetch_churn_table")
def fetch_table(tablename="customerchurn"):

    """
    Retrieves all records from the 'customer_churn' table in the MySQL database.
    """

    db_path = "local.db"
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {tablename}"
    dataf = pd.read_sql(query, conn)
    return dataf


def query_or_mail(user_query):

    prompt_template = "Is this '{query}' to generate a mail or asking a database questions? Repond with a Yes or No."
    prompt = PromptTemplate(template=prompt_template, input_variables=["query"])
    prompt = prompt.format(query=user_query)
    response = llm.invoke(prompt).content
    return response


def pandas_retriever():

    churn_table = fetch_table(tablename="customerchurn")
    customer_table = fetch_table(tablename="customers")
    pandas_agent = create_pandas_dataframe_agent(
        llm, [churn_table, customer_table], allow_dangerous_code=True
    )
    return pandas_agent


template = """You are a chatbot and your job is to generate is to respond to user queries.
                Ask if the answers is fine after all conversation"""

retriever_tool = pandas_retriever()
agent = AgentExecutor(agent=retriever_tool, tools=[])
response = retriever_tool.invoke("How many customers do we have?")
print(response)

# def llm_agent(history, input_text):

#     prompt = ChatPromptTemplate.from_messages([("system", template),
#                                                ("placeholder", "{history}"),
#                                                ("user", "{input_text}")])
#     prompt = prompt.format(history=history, input_text=input_text)
#     print("==================================================================")
#     response = retriever_tool.invoke("How many customers do we have?")
#     print(response)
#     response = response.content
#     return response

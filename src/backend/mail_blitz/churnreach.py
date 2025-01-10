import os
import sqlite3
import requests
import pandas as pd

from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process, LLM

from typing import List
from dotenv import load_dotenv

from src.backend.mail_blitz.utils import instantiate_db, fetch_table
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from langchain_core.tools import tool
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)


load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = "Your Key"

llm_ = ChatGroq(api_key=groq_key, model="llama3-70b-8192")
agent_llm = LLM(model="groq/llama3-8b-8192", api_key=groq_key)

customer_data_path = (
    "/home/godwin/Documents/Workflow/Customer-retention/data/raw_data/Churn.csv"
)
prediction_data_path = (
    "/home/godwin/Documents/Workflow/Customer-retention/prediction.csv"
)


def pandas_retriever():

    churn_table = fetch_table(tablename="customerchurn")
    customer_table = fetch_table(tablename="customera")
    pandas_agent = create_pandas_dataframe_agent(
        llm_, [churn_table, customer_table], allow_dangerous_code=True
    )
    return pandas_agent


db = instantiate_db(customer_data_path, prediction_data_path)
pandas_agent = pandas_retriever()


@tool("list_tables")
def list_tables() -> str:
    """List the available tables in the database"""
    return ListSQLDatabaseTool(db=db).invoke("")


@tool("tables_schema")
def tables_schema(tables: str) -> str:
    """
    Input is a comma-separated list of tables, output is the schema and sample rows
    for those tables. Be sure that the tables actually exist by calling `list_tables` first!
    Example Input: table1, table2, table3
    """
    tool = InfoSQLDatabaseTool(db=db)
    return tool.invoke(tables)


@tool("execute_sql")
def execute_sql(sql_query: str) -> str:
    """Execute a SQL query against the database. Returns the result"""
    return QuerySQLDataBaseTool(db=db).invoke(sql_query)


@tool("check_sql")
def check_sql(sql_query: str) -> str:
    """
    Use this tool to double check if your query is correct before executing it. Always use this
    tool before executing a query with `execute_sql`.
    """
    return QuerySQLCheckerTool(db=db, llm=llm_).invoke({"query": sql_query})


@tool("write_report")
def write_report(sql_query: str) -> str:
    """
    Use this tool to write report on the result provided by the analyst`.
    """
    return llm_.invoke({"query": sql_query})


@tool("make predictions")
def make_inference(input_data):
    """
    Use this tool to perform inference on data from a specified date range.
    Retrieves data for the given date range from a database, sends it to an inference
    endpoint for prediction, and prints the response.
    """
    data = pd.DataFrame(input_data)
    print(data.shape)
    data = input_data.to_dict()
    inference_endpoint = "https://retention.zapto.org/predict"

    response = requests.post(inference_endpoint, json=data).json()
    return response


@tool("Email draft generator")
def email_draft_generator(query: str):
    """
    Use this tool to write emails.

    Uses a language model to create write professional mails.
    """
    email_draft = llm_.invoke(
        f"Generate a professional mail to report this findings '{query}'."
    )
    return email_draft


@tool("retriever")
def retriver(query: dict) -> str:
    """
    Anwser questions asked by the user.
    """

    response = pandas_agent.invoke(input=query["description"])
    return response["output"]


@tool("make_churn_prediction")
def make_inference(query: str):
    """
    Use this tool to perform inference on data from a specified date range.

    Retrieves data for the given date range from a database with the input sql query, sends it to an inference
    endpoint for prediction, and prints the response.
    """

    conn = sqlite3.connect("local.db")
    print("========================================")
    print(query)
    # query = f"SELECT * FROM customers"
    dataf = pd.read_sql(query, conn)
    dataf_dicts = dataf.to_dict()
    inference_endpoint = "http://127.0.0.1:9696/predict"

    response = requests.post(inference_endpoint, json=dataf_dicts).json()
    return response


@tool("pull_prediction_data")
def pull_prediction_data():

    """
    Fetch the churn prediction table for analysis after the inference is complete.

    Returns:
        DataFrame: The retrieved churn table data.
    """
    dataf = fetch_table(tablename="customerchurn")
    return dataf


@tool("pull_customer_data")
def pull_customer_data():

    """
    Fetches the customer data from the database for making predictions.

    Returns:
        DataFrame: The customers data table.
    """
    dataf = fetch_table(tablename="customers")
    return dataf


churn_predictor = Agent(
    role="Customer manager",
    goal="Send data to a predition endpoint and get the prediction response",
    verbose=True,
    backstory=(
        "As a proactive customer manager, you specialize in facilitating seamless predictions to aid decision-making. "
        "You efficiently send data to prediction endpoints, and ensure the response is accurate and actionable, "
        "driving insights that enhance customer retention strategies."
    ),
    tools=[pull_customer_data, make_inference, pull_prediction_data],
    llm=agent_llm,
    allow_delegation=False,
)

database_administator = Agent(
    role="SQL Specialist",
    goal="Develop, optimize, and execute efficient SQL queries that provide comprehensive answers to queries",
    verbose=True,
    backstory=(
        "Driven by an unwavering curiosity for the depths of data, you excel at designing intricate queries tailored to "
        "fulfill analytical demands, enabling seamless and insightful data extraction."
    ),
    tools=[list_tables, tables_schema, execute_sql, check_sql],
    llm=agent_llm,
    allow_delegation=False,
)

data_analyst = Agent(
    role="Data Analyst",
    goal="Delve into structured data to extract actionable insights",
    verbose=True,
    backstory=(
        "Inspired by the potential of historical data, you merge curiosity and analytical acumen to identify trends "
        "and empower businesses with impactful, data-backed recommendations."
    ),
    llm=agent_llm,
    allow_delegation=False,
)

reporter = Agent(
    role="Insight Specialist/Reporter",
    goal="Compose detailed and compelling reports summarizing analytical findings to give comprehensive answer to inputs",
    verbose=True,
    memory=True,
    backstory=(
        "Equipped with a unique talent for distilling complex data into relatable stories, you transform raw analytics "
        "into captivating narratives that inform, engage, and inspire readers, making sophisticated insights easily digestible."
    ),
    llm=agent_llm,
    allow_delegation=False,
)

admin_secretary = Agent(
    role="Secratary",
    goal="Compose effective and contextually relevant email based on input to report research findngs",
    verbose=True,
    backstory=(
        "With a keen sense for impactful communication, you specialize in crafting concise and engaging email content. "
        "You consider the tone, purpose, and audience to produce drafts that resonate and achieve the intended response."
    ),
    tools=[email_draft_generator],
    llm=agent_llm,
    allow_delegation=False,
)

customer = Agent(
    role="Answer Architect",
    goal="Respond to this '{query}'",
    backstory=(
        """You thrive on clarity and precision, tailoring responses that directly address the query at hand while
            ensuring accuracy and relevance."""
    ),
    tools=[retriver],
    llm=agent_llm,
    allow_delegation=False,
    verbose=True,
)

# Research task
research = Task(
    description=(
        """
        Use your expertise to explore and analyze database tables. Identify available tables using `list_tables`,
        review their structure with `tables_schema`, execute targeted SQL queries with `execute_sql`,
        and ensure query accuracy using `check_sql`. Deliver a well-organized data for analysis.
        """
    ),
    expected_output="A well-organized tabular data retrieved from executed SQL queries together with the executed code",
    tools=[list_tables, tables_schema, execute_sql, check_sql],
    agent=database_administator,
)

# Analyst task
analyse = Task(
    description=(
        """
        You are an accomplished data analyst with a deep-rooted proficiency in using Python for data analysis.
        Your analyses are precise, thorough, and presented in an accessible format, maintaining clarity while detailing complex insights.
        Ensure your work is based strictly on the provided dataset and captures every nuance.
        Your final analysis should be comprehensive and structured for easy interpretation by the report writer.
        """
    ),
    expected_output="An in-depth and insightful analysis of the provided dataset in text format.",
    agent=data_analyst,
)

# Report task
report = Task(
    description=(
        """
        Your reputation as a skilled writer is built on your ability to communicate complex findings clearly and effectively.
        You excel at summarizing extensive analyses into concise bullet points that highlight the most critical aspects.
        Base your work on the comprehensive analysis provided by the analyst and present it in a structured, reader-friendly format.
        """
    ),
    expected_output="A detailed and concise report summarizing the analytical findings.",
    agent=reporter,
    tools=[write_report],
)

executive_mailing = Task(
    description=(
        """
        You are an office secratary known for crafting concise and engaging email to give office briefings based on the input. The input
        is the summary of findings gotten from a particular reseach.
        Your emails are clear, purposeful, and effectively communicate the research findings while maintaining an appropriate tone.
        The final output should be polished, professional, and ready for use.
        """
    ),
    expected_output="A well-crafted email reviewed and approved by the user.",
    agent=admin_secretary,
    tools=[email_draft_generator],
)

churn = Task(
    description=(
        """You are a question and answer personnel. Use the reponder tool to get your answers"""
    ),
    expected_output="A good and concise response to the question",
    agent=churn_predictor,
    tools=[retriver],
)

chr_pred = Task(
    description=(
        """You are a network guy. You send data to a prediction endpoint and get a response on the churn data that was predicted"""
    ),
    expected_output="A tabular data containing customer possibility of churning",
    agent=churn_predictor,
    tools=[make_inference, pull_prediction_data],
)


executives_crew = Crew(
    agents=[database_administator, data_analyst, reporter],
    tasks=[research, analyse, report],
    process=Process.sequential,
    memory=True,
    cache=True,
    max_rpm=100,
    planning=True,
    planning_llm=ChatGroq(api_key=groq_key, model="groq/llama3-8b-8192"),
)

customer_crew = Crew(
    agents=[churn_predictor],
    tasks=[churn],
    max_rpm=100,
    planning=True,
    # verbose=True,
    planning_llm=ChatGroq(api_key=groq_key, model="groq/llama3-8b-8192"),
)

prediction_crew = Crew(
    agents=[churn_predictor, data_analyst, reporter],
    tasks=[chr_pred, analyse, report],
    process=Process.sequential,
    memory=True,
    cache=True,
    verbose=True,
    max_rpm=100,
    planning=True,
    planning_llm=ChatGroq(api_key=groq_key, model="groq/llama3-8b-8192"),
)

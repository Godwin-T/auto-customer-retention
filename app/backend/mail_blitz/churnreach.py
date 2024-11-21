import os
import requests
import pandas as pd

from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process, LLM

from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase


from typing import List
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

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


def pull_data(data):

    db_path = "local.db"

    if os.path.exists(db_path):
        engine = create_engine(f"sqlite:///{db_path}")
    else:
        df = pd.read_csv(data)
        engine = create_engine("sqlite:///local.db")
        df.to_sql("customers", engine, index=False)
    db = SQLDatabase(engine=engine)
    return db


llm_ = ChatGroq(api_key=groq_key, model="llama3-70b-8192")
agent_llm = LLM(model="groq/llama3-8b-8192", api_key=groq_key)

data_path = "/home/godwin/Documents/Workflow/Customer-retention/data/raw_data/Churn.csv"
db = pull_data(data_path)


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


@tool("dataframe_retriever")
def dataframe_retriever(query, table) -> str:
    """Anwser questions asked by the use from the dataframe"""

    prompt_template = """You are an information retriever and a very good one at that. You are to retrieve substantial
                      information from the table can be used along side this '{query}' to genarate a good promotional mail."""
    prompt = PromptTemplate(template=prompt_template, input_variables=["query"])
    prompt = prompt.format(query=query)

    agent = create_pandas_dataframe_agent(llm_, table, verbose=True)
    response = agent.invoke(prompt)
    return response


@tool("fetch_churn_table")
def fetch_churn_table():

    """
    Retrieves all records from the 'customer_churn' table in the MySQL database.
    """

    username = os.getenv("DBUSERNAME")
    password = os.getenv("DBPASSWORD")
    hostname = os.getenv("HOSTNAME")
    dbname = os.getenv("DBNAME")

    tablename = "customer_churn"
    engine = create_engine(
        f"mysql+mysqlconnector://{username}:{password}@{hostname}/{dbname}"
    )
    query = f"SELECT * FROM {tablename}"
    dataf = pd.read_sql(query, engine)
    return dataf


def query_or_mail(user_query):

    prompt_template = "Is this '{query}' to generate a mail or asking a database questions? Repond with a Yes or No."
    prompt = PromptTemplate(template=prompt_template, input_variables=["query"])
    prompt = prompt.format(query=user_query)
    response = llm_.invoke(prompt).content
    return response


database_administator = Agent(
    role="SQL Specialist",
    goal="Develop, optimize, and execute efficient SQL queries that provide comprehensive answers to {query}",
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
    tools=[dataframe_retriever],
    llm=agent_llm,
    allow_delegation=True,
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

customer_care = Agent(
    role="Data Retriver",
    goal="Get data from a database and extracts information from it",
    backstory="",
    tools=[fetch_churn_table, dataframe_retriever],
    llm=agent_llm,
    allow_delegation=False,
)

public_relations = Agent(
    role="Promotional mail writer",
    goal="Write promational mails to be sent to customers in a telecommunication company based on the provided information",
    backstory="",
    llm=agent_llm,
    allow_delegation=False,
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
    expected_output="A well-organized tabular data retrieved from executed SQL queries.",
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

predictions_information = Task(
    description=(
        """
        You are a data analyst responsible for extracting and summarizing customer churn information to support the creation of personalized emails.
        The input consists of information of churning customers retrieved from customer churn records.
        Your role is to analyze this data, identify key insights, and deliver a concise, actionable summary that highlights the essential information
        for crafting targeted customer communications together with the information given.

        The summary should be clear, precise, and suitable for generating customer engagement strategies.
        """
    ),
    expected_output="A consice summary of the information that will be used to generate customer mails",
    agent=customer_care,
    tools=[fetch_churn_table, dataframe_retriever],
)

customer_mailing = Task(
    description=(
        """You are a customer faced promotional email writer know for for crafting concise and engaging email for the companies customers. You
        are to write a promtional mail to the company customers based on the information you are provided with.
        Your emails are clear, purposeful, and effectively communicate the intent of the company of its customers.
        The final output should be polished, professional, and ready for use."""
    ),
    expected_output="A well-crafted promotional email reviewed and approved by the user for the customers.",
    agent=public_relations,
)

executives_crew = Crew(
    agents=[database_administator, data_analyst, reporter, admin_secretary],
    tasks=[research, analyse, report, executive_mailing],
    process=Process.sequential,
    memory=True,
    cache=True,
    max_rpm=100,
    planning=True,
    planning_llm=ChatGroq(api_key=groq_key, model="groq/llama3-8b-8192"),
)

customer_crew = Crew(
    agents=[customer_care, public_relations],
    tasks=[predictions_information, customer_mailing],
    process=Process.sequential,
    memory=True,
    cache=True,
    max_rpm=100,
    planning=True,
    planning_llm=ChatGroq(api_key=groq_key, model="groq/llama3-8b-8192"),
)

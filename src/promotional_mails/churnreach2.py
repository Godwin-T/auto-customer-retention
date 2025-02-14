import os
from typing import Any

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase

from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_community.agent_toolkits import SQLDatabaseToolkit

load_dotenv()


def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


groq_key = os.getenv("GROQ_API_KEY")
chatgroq = ChatGroq(api_key=groq_key, model="llama3-70b-8192")

db = SQLDatabase("sqlite:///local.db")
toolkit = SQLDatabaseToolkit(db=db, llm=chatgroq)
tools = toolkit.get_tools()

list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

print(list_tables_tool.invoke(""))

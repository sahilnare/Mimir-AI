from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, AnyMessage

from langchain_community.utilities import SQLDatabase

import os
import re
from typing import Any, Annotated, Literal

from langchain_core.messages import ToolMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks, RunnablePassthrough
from langgraph.prebuilt import ToolNode

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

from langchain_core.tools import tool

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class AnalysisRequest(BaseModel):
    """A request for data analysis and insights."""
    data: str = Field(..., description="The data from SQL query results to analyze")
    query_intent: str = Field(..., description="The user's original query intent")

class SubmitFinalAnswer(BaseModel):
    """ Submit the final answers to the user based on the query results."""
    final_answer: str = Field(..., description="The final answer to the user.")

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\nPlease fix your mistakes.",
                tool_call_id=tc["id"]
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks:
    """Create a tool node with fallback to handle errors and surface them to the agent."""
    # Wrap the error handler function in a RunnablePassthrough
    error_handler = RunnablePassthrough(handle_tool_error)
    
    return ToolNode(tools).with_fallbacks(
        [error_handler],
        exception_key='error'
    )

@tool
def filtered_table_list() -> str:
    """List only the predetermined database tables."""
    allowed_tables = ["tracking_orders", "orders", 'order_details']
    return "\n".join(allowed_tables)


# list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables"

config = {"configurable": {"thread_id": "1"}}

def validate_query(query: str) -> bool:
    # Basic safety checks
    danger_patterns = [
        r'\bDROP\b',
        r'\bDELETE\b',
        r'\bTRUNCATE\b',
        r'\bUPDATE\b',
        r'\bINSERT\b',
        r'\bALTER\b',
        r'\bCREATE\b'
    ]
    
    for pattern in danger_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False
    return True

def create_db_query_tool(db):
    @tool
    def db_query_tool(query: str) -> str:
        """
        Execute a SQL query against the database and get back the result.
        If the query is not correct, an error message will be returned.
        If an error is returned, rewrite the query, check the query, and try again.
        """
        if not validate_query(query):
            return "Error: Query contains unsafe operations"
        
        result = db.run_no_throw(query)

        if result.startswith('[') and result.endswith(']'):
            try:
                # Clean up common Python objects in the string
                # Replace UUID objects
                result = re.sub(r'UUID\([\'"]([^\'"]*)[\'"]\)', r'\1', result)
                # Replace datetime.date objects
                result = re.sub(r'datetime\.date\((\d+), (\d+), (\d+)\)', r'\1-\2-\3', result)
                # Replace datetime.datetime objects if they exist
                result = re.sub(r'datetime\.datetime\((\d+), (\d+), (\d+), (\d+), (\d+)(?:, (\d+))?\)', 
                            lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}:{m.group(6) if m.group(6) else '00'}", 
                            result)
                # Replace None with empty string
                result = result.replace('None', '""')
                # print("\n\n\n")
                # print(f"RESULT {result}")
                # print("\n\n\n")
                
            except Exception as e:
                print('Error occurred during formatting:', e)
                # If any error occurs during formatting, return the original result
                pass

        if not result:
            return "Error: Query failed. Please rewrite your query and try again."
        
        return result
    return db_query_tool

def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls = [
                    {
                        "name": "filtered_table_list",
                        "args": {},
                        "id": "tool_call_1"
                    }
                ]
            )
        ]
    }

def create_model_check_query(db_query_tool):
    def model_check_query(state: State) -> dict[str, list[AIMessage]]:
        """
        Use this query to double check if your query is correct before executing it.
        """
        query_check_system = """You are a SQL expert with a strong attention to detail. Double check the PostgreSQL query for common mistakes, including:

        Using NOT IN with NULL values
        Using UNION when UNION ALL should have been used
        Using BETWEEN for exclusive ranges
        Data type mismatch in predicates
        Properly quoting identifiers with double quotes
        Using the correct number of arguments for functions
        Casting to the correct data type using PostgreSQL's type casting syntax (::)
        Using the proper columns for joins
        Using the correct case sensitivity for identifiers
        Using PostgreSQL-specific functions and operators correctly
        Proper handling of date/time data types
        Correct usage of DISTINCT vs DISTINCT ON
        Proper use of window functions and their frame clauses
        Correct schema qualification of objects

        If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query. You will call the appropriate tool to execute the query after running this check."""

        query_check_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=query_check_system),
            MessagesPlaceholder(variable_name="messages")
        ])

        query_check = query_check_prompt | ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(
            [db_query_tool], 
            tool_choice="required"
        )
        # Create a proper message list for the prompt
        last_message = state["messages"][-1]
        messages_for_prompt = [
            HumanMessage(content=last_message.content)
        ]
        
        return {
            "messages": [
                query_check.invoke({
                    "messages": messages_for_prompt
                })
            ]
        }
    return model_check_query

def query_gen_node(state: State):
    query_gen_system = """You are a SQL expert with a strong attention to detail.
    Given an input question, output a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer.
    DO NOT call any tool besides SubmitFinalAnswer to submit the final answer. Always remember not to submit any sensitive details like table names or raw queries as answer. Always try best to get an actual answer and while submitting response please don't mention user id of user in the answer.
    When generating the query:
    Output the SQL query that answers the input question without a tool call.
    LIMIT Condition should only be used for normal queries if the user is asking for a report do not add a limit condtion and Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 60 results using LIMIT 60.
    You can order the results by a relevant column using ORDER BY to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    When writing PostgreSQL queries:

    Use proper schema qualification when necessary
    Use double quotes for identifiers that need quoting
    Use PostgreSQL-specific casting syntax (::) when needed
    Be mindful of case sensitivity in identifiers
    Use appropriate PostgreSQL-specific functions and operators
    Handle date/time data types correctly
    Use DISTINCT ON when appropriate instead of plain DISTINCT
    When query is being made for some status try to use LIKE operator to get the similar resuls for similar statuses. For eg there maybe multiple statuses like RTO, RTO_DELIVERED, RTO_RETURNED etc. So use LIKE operator to get all the similar statuses.

    If you get an error while executing a query, rewrite the query and try again.
    If you get an empty result set, you should try to rewrite the query to get a non-empty result set.
    NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.
    If you have enough information to answer the input question, simply invoke the appropriate tool to submit the final answer to the user.
    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""

    # Node for a model to generate a query based on the question and schema
    query_gen_prompt = ChatPromptTemplate.from_messages(
            [("system", query_gen_system), ("placeholder", "{messages}")]
        )

    query_gen = query_gen_prompt | ChatOpenAI(model="gpt-4o", temperature=0).bind_tools([SubmitFinalAnswer])

    message = query_gen.invoke(state)

    # We are adding a check for SQL Queries which are not being executed.
    # print('HEYMAN')
    # print([keyword in message.content.upper() for keyword in ['SELECT', 'FROM']])
    # print(message.tool_calls)
    if any([keyword in message.content.upper() for keyword in ['SELECT', 'FROM']]):
        # print('HEYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY')
        return {"messages": [AIMessage(content=message.content)]}

    # Sometimes, the LLM will hallucinate and call the wrong tool. We need to catch this and return an error message.
    tool_message = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] == "SubmitFinalAnswer":
                final_answer = tc["args"].get("final_answer", "")
                if any(keyword in final_answer.upper() for keyword in ['SELECT', 'FROM']):
                    tool_message.append(
                        ToolMessage(
                            content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                            tool_call_id=tc["id"]
                        )
                    )
            elif tc["name"] != "SubmitFinalAnswer":
                tool_message.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                        tool_call_id=tc["id"]
                    )
                )
    else:
        tool_message = []
    return {"messages": [message] + tool_message}

def filter_messages(messages: list):
    # Filter last 5 messages for context
    return messages[-2:]

# def should_continue(state: State) -> Literal[END, "correct_query", "query_gen"]:
#     messages = state["messages"]
#     last_message = messages[-1]

#      # If we have tool calls and it's SubmitFinalAnswer, end the conversation
#     if getattr(last_message, "tool_calls", None):
#         if any(tc["name"] == "SubmitFinalAnswer" for tc in last_message.tool_calls):
#             return END
            
#     # If there's an error message, go back to query generation
#     if last_message.content.startswith("Error:"):
#         return "query_gen"
    
#     # If the message contains what looks like a SQL query, proceed to correct_query
#     if any(keyword in last_message.content.upper() for keyword in ['SELECT', 'FROM']):
#         # print('YESITSTRUE')
#         return "correct_query" 
    
#     # In any other case, continue with query generation
#     return "query_gen"
    
# def text2sql_agent(db):
#     toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model="gpt-4o"))
#     tools = toolkit.get_tools()
#     list_tables_tool = filtered_table_list
#     get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

#     db_query_tool = create_db_query_tool(db)

#     # print(list_tables_tool.invoke(""))

#     # print(get_schema_tool.invoke("orders"))

#     workflow = StateGraph(State)

#     # Add Nodes to the workflow
#     workflow.add_node("first_tool_call", first_tool_call)

#     workflow.add_node("list_tables_tool", create_tool_node_with_fallback([list_tables_tool]))

#     workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))

#     # Node for a model to choose the relevant tables based on the question and available tables
#     model_get_schema = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools([get_schema_tool])
#     workflow.add_node("model_get_schema", lambda state: {"messages": [model_get_schema.invoke(filter_messages(state["messages"]))]})    

#     workflow.add_node("query_gen", query_gen_node)

#     workflow.add_node("correct_query", create_model_check_query(db_query_tool))

#     workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))

#     # Specify the edges between the nodes
#     workflow.add_edge(START, "first_tool_call")
#     workflow.add_edge("first_tool_call", "list_tables_tool")
#     workflow.add_edge("list_tables_tool", "model_get_schema")
#     workflow.add_edge("model_get_schema", "get_schema_tool")
#     workflow.add_edge("get_schema_tool", "query_gen")
#     workflow.add_conditional_edges("query_gen", should_continue)
#     workflow.add_edge("correct_query", "execute_query")
#     workflow.add_edge("execute_query", "query_gen")

#     return workflow

    
def detect_intent(state: State) -> Literal["simple_query", "analysis_request"]:
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not user_messages:
        return "simple_query"
    
    last_user_message = user_messages[-1].content.lower()
 
    analysis_keywords = [
        "report", "insight", "analyze", "analysis", "trend", "pattern", 
        "summary", "overview", "breakdown", "statistics", "metrics", 
        "dashboard", "visualize", "visualization", "chart", "graph", 
        "compare", "comparison", "performance", "understand", "deep dive"
    ]

    for keyword in analysis_keywords:
        if keyword in last_user_message:
            return "analysis_request"
            
    return "simple_query"

def create_analysis_agent():

    analysis_system_prompt = """You are an expert data analyst who excels at interpreting SQL query results and providing insightful analysis.
    
    Your task is to:
    1. Examine the provided data (which comes from a SQL query)
    2. Identify key patterns, trends, and insights
    3. Provide a structured analysis with actionable recommendations
    4. Use statistical thinking to draw meaningful conclusions
    5. Relate your findings back to the user's original question
    
    Format your response as a clear, well-structured report with sections including:
    - Summary of Findings (1-2 sentences)
    - Key Insights (3-5 bullet points)
    - Detailed Analysis (2-3 paragraphs)
    - Recommendations (if applicable)
    
    IMPORTANT: Focus only on insights that can be directly derived from the data provided. Do not make up data or draw conclusions that aren't supported by the information available."""
    
    # print("\n\n\n")
    # print(HumanMessage(content=f"{analysis_system_prompt}\n\n""User Query: {query_intent}\n\nData from SQL Query: {data}"))
    # print("\n\n\n")
    analysis_prompt = ChatPromptTemplate.from_template(
        f"{analysis_system_prompt}\n\nUser Query: {{query_intent}}\n\nData from SQL Query: {{data}}"
    )
    
    analysis_chain = analysis_prompt | ChatOpenAI(model="o1-preview", temperature=1)
    
    return analysis_chain

def run_analysis(state: State) -> dict:
    
    # Get the most recent SQL query result
    tool_messages = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
    if not tool_messages:
        return {
            "messages": [
                AIMessage(content="I couldn't find any query results to analyze. Let me try to run a query first.")
            ]
        }

    last_query_result = tool_messages[-1].content

    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    original_query = user_messages[-1].content if user_messages else ""

    analysis_agent = create_analysis_agent()
    analysis_result = analysis_agent.invoke({
        "data": last_query_result,
        "query_intent": original_query
    })
    
    # Create a final answer with the analysis
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "SubmitFinalAnswer",
                        "args": {
                            "final_answer": analysis_result.content
                        },
                        "id": "analysis_result"
                    }
                ]
            )
        ]
    }

def format_results_for_analysis(state: State) -> dict:

    # # Get the most recent SQL query result
    tool_messages = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
    if not tool_messages:
        return {
            "messages": [
                AIMessage(content="I couldn't find any query results to analyze. Let me try to run a query first.")
            ]
        }
    
    # Get the last query result
    last_query_result = tool_messages[-1].content
    
    # Try to parse it as a table and convert to more analyzable format
    try:
        # Simple table parsing (adapt based on your actual output format)
        # lines = last_query_result.strip().split('\n')
        # print("\n\n")
        # print("\n\n")
        # print(f"LASTQ {last_query_result}")
        # print("\n\n")
        # print("\n\n")
        if len(last_query_result) <= 2:  # Not enough data for analysis
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "SubmitFinalAnswer",
                                "args": {
                                    "final_answer": "There's not enough data to provide a meaningful analysis. Here's the query result:\n\n" + last_query_result
                                },
                                "id": "no_analysis_needed"
                            }
                        ]
                    )
                ]
            }
        
        return {
            "messages": [
                AIMessage(content=f"Query results formatted for analysis:\n\n{last_query_result}")
            ]
        }
    except Exception as e:
        return {
            "messages": [
                AIMessage(content=f"Error formatting results for analysis: {str(e)}\n\nUsing raw results instead.")
            ]
        }

def should_continue_analytics(state: State) -> Literal[END, "correct_query", "query_gen", "format_for_analysis"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # If we have tool calls and it's SubmitFinalAnswer, end the conversation
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        if any(tc["name"] == "SubmitFinalAnswer" for tc in last_message.tool_calls):
            return END
            
    # If there's an error message, go back to query generation
    if isinstance(last_message, ToolMessage) and last_message.content.startswith("Error:"):
        return "query_gen"
    
    # If the message contains what looks like a SQL query, proceed to correct_query
    if isinstance(last_message, AIMessage) and any(keyword in (last_message.content or "").upper() for keyword in ['SELECT', 'FROM']):
        return "correct_query"
        
    # If we have query results and the original request was for analysis, go to analysis
    if isinstance(last_message, ToolMessage) and not last_message.content.startswith("Error:"):
        # Check if the original intent was for analysis
        if detect_intent(state) == "analysis_request":
            return "format_for_analysis"
    
    # In any other case, continue with query generation
    return "query_gen"

def text2sql_with_analysis_agent(db):

    toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model="gpt-4o"))
    tools = toolkit.get_tools()
    list_tables_tool = filtered_table_list 
    get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    db_query_tool = create_db_query_tool(db)

    workflow = StateGraph(State)
    
    workflow.add_node("first_tool_call", first_tool_call)
    workflow.add_node("list_tables_tool", create_tool_node_with_fallback([list_tables_tool]))
    workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))
    
    model_get_schema = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools([get_schema_tool])
    workflow.add_node("model_get_schema", lambda state: {"messages": [model_get_schema.invoke(filter_messages(state["messages"]))]})    
    
    workflow.add_node("query_gen", query_gen_node)
    workflow.add_node("correct_query", create_model_check_query(db_query_tool))
    workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))

    workflow.add_node("format_for_analysis", format_results_for_analysis)
    workflow.add_node("run_analysis", run_analysis)

    workflow.add_edge(START, "first_tool_call")
    workflow.add_edge("first_tool_call", "list_tables_tool")
    workflow.add_edge("list_tables_tool", "model_get_schema")
    workflow.add_edge("model_get_schema", "get_schema_tool")
    workflow.add_edge("get_schema_tool", "query_gen")

    workflow.add_conditional_edges("query_gen", should_continue_analytics)
    workflow.add_conditional_edges("execute_query", should_continue_analytics)

    workflow.add_edge("format_for_analysis", "run_analysis")
    workflow.add_edge("run_analysis", END)
    workflow.add_edge("correct_query", "execute_query")
    
    return workflow

# Usage example
# dbconn = SQLDatabase.from_uri("postgresql://openleaf:testpassword@localhost:63333/testdb")
# app = text2sql_with_analysis_agent(db=dbconn)

# messages = app.invoke({"messages": [(
#     "user","Give me the count of all delivered orders in Tier 2 Indian Cities"
# )]})
# dbconn = SQLDatabase.from_uri("postgresql://openleaf:testpassword@localhost:63333/testdb")
# app = text2sql_agent(db=dbconn)

# for event in app.stream(
#     {"messages": HumanMessage(content= """For this user id cd3a5daf-952f-4f28-9db2-9cbd529bd670 answer the following question.
#                     Get me the 10 recent orders""")}, config, stream_mode='values'
# ):
#     message = event["messages"][-1]
#     if isinstance(message, tuple):
#         print(message)
#     else:
#         message.pretty_print()

# json_str = messages["messages"][-1].tool_calls[0]["args"]["final_answer"]

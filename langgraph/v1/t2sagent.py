from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict

import getpass
import os
from IPython.display import Image, display

from langchain.chat_models import init_chat_model

from langchain import hub

from typing_extensions import Annotated

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

from langgraph.graph import StateGraph, START, END

from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langchain_core.messages import HumanMessage

from langgraph.prebuilt import create_react_agent

db = SQLDatabase.from_uri("postgresql://ai_agent:mimir123@localhost:63333/productiondb")

print(db.get_table_names())

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

os.environ['OPENAI_API_KEY'] = 'sk-proj-103m-testkey'

llm = init_chat_model("gpt-4o", model_provider="openai")

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

# assert len(query_prompt_template.messages) == 1
# query_prompt_template.messages[0].pretty_print()

class QueryOutput(TypedDict):
    """Generate SQL Query."""
    query: Annotated[str, ..., "Syntactically valid SQL Query."]


def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            'dialect': db.dialect,
            'top_k': 10,
            'table_info': db.get_table_info(['orders', 'tracking_orders', 'order_details']),
            'input': state['question'],
        }
    )
    
    prompt.messages[0].pretty_print()
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result['query']}

question = {"question": "Get the location and frequency of the top 10 RTO orders. I want results with most frequncy of RTO orders"}

query_gen = write_query(question)
query_gen = query_gen.get("query")
print(query_gen)

query = {"query": query_gen}

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDataBaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

print(execute_query(query))

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query and SQL Result, answer the user question.\n\n"
        f'Question: "{state["question"]}"\n'
        f'SQL Query: "{state["query"]}"\n'
        f'SQL Result: "{state["result"]}"'
    )

    response = llm.invoke(prompt)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)

graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()

# display(Image(graph.get_graph().draw_mermaid_png()))

for step in graph.stream(
    {"question": "Get destination locations and carriers of top 10 RTO orders and classify them into indian geographical regions"}, stream_mode="updates"
):
    print(step)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

print(tools)

prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

prompt_template.messages[0].pretty_print()

system_message = prompt_template.format(dialect=db.dialect, top_k=5)

agent_executor = create_react_agent(llm, tools, prompt=system_message)

for step in agent_executor.stream(
    {"messages": [{"role": "user", "content": question["question"]}]},
    stream_mode="values",
    ):
    step["messages"][-1].pretty_print()
    


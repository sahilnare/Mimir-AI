# Openleaf AI Server - Technical Documentation

## Overview

The Openleaf AI Server is a FastAPI-based application that provides a natural language interface to a PostgreSQL database. It uses LangGraph and LangChain to create an AI agent that can translate natural language questions into SQL queries, execute them against the database, and provide meaningful responses to users.

## Project Structure

```
Openleaf-AI-Server/
├── api.py                  # Main FastAPI application
├── configs/                # Configuration files
│   ├── config.py           # Configuration loader
│   ├── local.json          # Local environment config
│   ├── sandbox.json        # Sandbox environment config
│   └── production.json     # Production environment config
├── langgraph/              # LangGraph implementation
│   ├── v1/                 # Version 1 implementation
│   └── v2/                 # Version 2 implementation (current)
│       └── ai.py           # AI agent implementation
├── scripts/                # Utility scripts
│   └── tunnel.sh           # SSH tunnel setup script
├── ssh/                    # SSH keys for tunneling
├── Dockerfile              # Docker configuration
└── requirements.txt        # Python dependencies
```

## Technical Architecture

### 1. FastAPI Application (api.py)

The main application entry point that handles HTTP requests and responses.

#### Key Components:

- **FastAPI Setup**: Configures the FastAPI application with rate limiting, CORS middleware, and exception handlers.
- **Database Connection Management**: Uses an async context manager to establish and manage database connections.
- **Request Processing**: Processes incoming requests, creates AI agents, and formats responses.
- **Streaming Support**: Implements server-sent events (SSE) for streaming responses.
- **Logging and Monitoring**: Configures logging with rotation and S3 upload capabilities.

#### Key Functions:

- **`get_db()`**: Async context manager that establishes a database connection using configuration from the appropriate environment file.
- **`create_fresh_agent(db)`**: Creates a new instance of the LangGraph agent for each request.
- **`process_agent_response(response)`**: Extracts the final answer from the agent's response.
- **`stream_response(response)`**: Formats the agent's response for streaming.
- **`chat(request, query_request)`**: Main endpoint handler that processes user queries and returns responses.

#### API Endpoints:

- **`POST /ai/chat`**: Main endpoint for processing natural language queries.
- **`GET /ai/health`**: Health check endpoint.
- **`GET /ai/`**: Status check endpoint.

### 2. AI Agent (langgraph/v2/ai.py)

The core AI functionality that implements a LangGraph workflow for natural language to SQL conversion.

#### Key Components:

- **State Management**: Uses a TypedDict to manage the state of the conversation.
- **Tool Definitions**: Defines tools for database interaction, including query execution and schema retrieval.
- **Query Validation**: Implements safety checks to prevent dangerous SQL operations.
- **Workflow Definition**: Creates a state machine for processing queries through various stages.

#### Key Functions:

- **`text2sql_with_analysis_agent(db)`**: Main function that creates the LangGraph workflow.
- **`validate_query(query)`**: Validates SQL queries for safety.
- **`create_db_query_tool(db)`**: Creates a tool for executing SQL queries.
- **`query_gen_node(state)`**: Node that generates SQL queries from natural language.
- **`create_model_check_query(db_query_tool)`**: Node that validates and corrects SQL queries.
- **`detect_intent(state)`**: Determines if the user is requesting analysis or a simple query.
- **`run_analysis(state)`**: Analyzes query results and generates insights.
- **`should_continue_analytics(state)`**: Determines the next step in the workflow based on the current state.

#### Workflow Stages:

1. **Initialization**: Sets up the workflow and retrieves available tables.
2. **Schema Retrieval**: Gets the database schema for the relevant tables.
3. **Query Generation**: Generates a SQL query based on the user's question.
4. **Query Validation**: Validates and corrects the SQL query.
5. **Query Execution**: Executes the SQL query against the database.
6. **Result Analysis** (if requested): Analyzes the query results and generates insights.
7. **Response Generation**: Formats the final response for the user.

## Architectural Decisions and Rationale

### 1. Fresh Agent Instance for Each Request

```python
def create_fresh_agent(db):
    """Create a new instance of the agent for each request"""
    workflow = text2sql_with_analysis_agent(db)
    return workflow.compile()
```

**Decision**: Create a new LangGraph agent instance for each request rather than reusing a single instance.

**Rationale**:
- **Isolation**: Each request gets its own isolated agent, preventing state leakage between requests.
- **Resource Management**: Ensures resources are properly allocated and released after each request.
- **Scalability**: Allows for better horizontal scaling as each request is self-contained.
- **Reliability**: Reduces the risk of one request affecting another, improving overall system reliability.
- **Debugging**: Makes it easier to debug issues as each request has its own traceable agent instance.

### 2. Async Context Manager for Database Connections

```python
@asynccontextmanager
async def get_db():
    global db
    if db is None:
        app_config = load_config()
        db_config = app_config.get("db")
        openai_key = app_config.get("OPENAI_API_KEY")
        os.environ['OPENAI_API_KEY'] = openai_key
        db = SQLDatabase.from_uri(f"postgresql://{db_config.get('username')}:{db_config.get('password')}@{db_config.get('host')}:{db_config.get('port')}/{db_config.get('dbname')}")
    try:
        yield db
    except Exception as e:
        db = None
        raise e
    finally:
        pass
```

**Decision**: Use an async context manager for database connections.

**Rationale**:
- **Resource Management**: Ensures database connections are properly managed and released.
- **Error Handling**: Provides a clean way to handle exceptions and reset the connection if needed.
- **Lazy Initialization**: Only creates the database connection when it's actually needed.
- **Singleton Pattern**: Implements a singleton pattern for the database connection to avoid creating multiple connections.
- **Environment Configuration**: Sets up the OpenAI API key from the configuration, ensuring it's available for the AI agent.

### 3. LangGraph for Workflow Management

**Decision**: Use LangGraph to create a state machine for processing queries.

**Rationale**:
- **Modularity**: Allows for a modular approach to query processing, with each step handled by a separate node.
- **Flexibility**: Makes it easy to add, remove, or modify steps in the workflow.
- **Conditional Logic**: Supports conditional branching based on the state of the conversation.
- **Error Handling**: Provides a structured way to handle errors and retry failed steps.
- **Visualization**: Enables visualization of the workflow, making it easier to understand and debug.

### 4. Query Validation and Safety Checks

```python
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
```

**Decision**: Implement query validation to prevent dangerous SQL operations.

**Rationale**:
- **Security**: Prevents SQL injection and other security vulnerabilities.
- **Data Protection**: Ensures that data cannot be accidentally or maliciously modified or deleted.
- **Compliance**: Helps meet regulatory requirements for data protection.
- **Reliability**: Prevents accidental data loss or corruption.

### 5. Intent Detection for Analysis Requests

```python
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
```

**Decision**: Implement intent detection to differentiate between simple queries and analysis requests.

**Rationale**:
- **User Experience**: Provides a better user experience by offering different levels of response detail.
- **Resource Optimization**: Allocates resources appropriately based on the complexity of the request.
- **Specialized Processing**: Allows for specialized processing of analysis requests, such as generating insights and recommendations.
- **Flexibility**: Makes it easy to add more intent types in the future.

### 6. Streaming Support for Long-Running Queries

```python
async def stream_response(response) -> Generator[str, None, None]:
    """Stream the response."""
    try:
        for chunck in response:
            if chunck.get("messages"):
                last_message = chunck["messages"][-1]
                if hasattr(last_message, "tool_calls"):
                    for tool_call in last_message.tool_calls:
                        if tool_call["name"] == "SubmitFinalAnswer":
                            args = tool_call["args"]
                            if isinstance(args, str):
                                args = json.loads(args)
                            yield f"data: {args['final_answer']}\n\n"
                else:
                    yield f"data: {last_message.content}\n\n"
            await asyncio.sleep(0.1)
    except Exception as e:
        yield f"data: Error Occurred: {str(e)}\n\n"
    finally:
        yield "data: [DONE]\n\n"
```

**Decision**: Implement streaming support for long-running queries.

**Rationale**:
- **User Experience**: Provides real-time feedback to users during long-running queries.
- **Timeout Prevention**: Prevents timeouts for queries that take a long time to execute.
- **Progress Indication**: Allows for progress indication during query execution.
- **Error Handling**: Provides immediate feedback if an error occurs during query execution.

### 7. S3 Log Upload for Centralized Logging

```python
async def upload_logs_to_s3():
    """Uploads logs asynchronously to S3 every 60 seconds."""
    while True:
        try:
            if s3_client and os.path.exists(LOG_FILE):
                s3_client.upload_file(LOG_FILE, BUCKET_NAME, S3_LOG_PATH)
                logger.info(f"Uploaded logs to S3: {S3_LOG_PATH}")
        except NoCredentialsError:
            logger.error("AWS credentials not found. Unable to upload logs.")
        except Exception as e:
            logger.error(f"Error uploading logs to S3: {str(e)}")
        
        await asyncio.sleep(60)  # Upload every 60 seconds
```

**Decision**: Implement S3 log upload for centralized logging.

**Rationale**:
- **Centralized Logging**: Provides a centralized location for logs from multiple instances.
- **Retention**: Ensures logs are retained even if the local log file is rotated or deleted.
- **Analysis**: Enables analysis of logs across multiple instances and time periods.
- **Compliance**: Helps meet regulatory requirements for log retention.

## In-Depth Technical Implementation

### 1. LangGraph Workflow Implementation

The LangGraph workflow is implemented using a state machine approach, with each node representing a step in the workflow. The workflow is defined in the `text2sql_with_analysis_agent` function:

```python
def text2sql_with_analysis_agent(db):
    toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model="gpt-4o"))
    tools = toolkit.get_tools()
    list_tables_tool = filtered_table_list 
    get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    db_query_tool = create_db_query_tool(db)

    workflow = StateGraph(State)
    
    # Add nodes to the workflow
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

    # Add edges to the workflow
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
```

The workflow consists of the following steps:

1. **Initialization**: The `first_tool_call` node retrieves the list of available tables.
2. **Schema Retrieval**: The `model_get_schema` node determines which tables are relevant to the query, and the `get_schema_tool` node retrieves their schemas.
3. **Query Generation**: The `query_gen` node generates a SQL query based on the user's question and the database schema.
4. **Query Validation**: The `correct_query` node validates and corrects the SQL query.
5. **Query Execution**: The `execute_query` node executes the SQL query against the database.
6. **Result Analysis** (if requested): If the user requested analysis, the `format_for_analysis` and `run_analysis` nodes analyze the query results.
7. **Response Generation**: The agent generates a final response using the `SubmitFinalAnswer` tool.

The workflow uses conditional edges to determine the next step based on the current state:

```python
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
```

### 2. Query Generation and Execution

The query generation and execution process is implemented in several steps:

1. **Query Generation**: The `query_gen_node` function generates a SQL query based on the user's question and the database schema:

```python
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
    if any([keyword in message.content.upper() for keyword in ['SELECT', 'FROM']]):
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
```

2. **Query Validation**: The `create_model_check_query` function creates a node that validates and corrects the SQL query:

```python
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
```

3. **Query Execution**: The `create_db_query_tool` function creates a tool for executing SQL queries:

```python
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
                
            except Exception as e:
                print('Error occurred during formatting:', e)
                # If any error occurs during formatting, return the original result
                pass

        if not result:
            return "Error: Query failed. Please rewrite your query and try again."
        
        return result
    return db_query_tool
```

### 3. Result Analysis

The result analysis process is implemented in several steps:

1. **Intent Detection**: The `detect_intent` function determines if the user is requesting analysis or a simple query:

```python
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
```

2. **Analysis Agent Creation**: The `create_analysis_agent` function creates an agent for analyzing query results:

```python
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
    
    analysis_prompt = ChatPromptTemplate.from_template(
        f"{analysis_system_prompt}\n\nUser Query: {{query_intent}}\n\nData from SQL Query: {{data}}"
    )
    
    analysis_chain = analysis_prompt | ChatOpenAI(model="o1-preview", temperature=1)
    
    return analysis_chain
```

3. **Result Analysis**: The `run_analysis` function analyzes the query results and generates insights:

```python
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
```

### 4. FastAPI Request Processing

The FastAPI request processing is implemented in several steps:

1. **Request Validation**: The request is validated against the `QueryRequest` model:

```python
class QueryRequest(BaseModel):
    query: str
    user_id: str = "1ecf71e1-1748-425b-9071-56e925a391a3"
    is_customer: bool = False
    stream: bool = False
```

2. **Database Connection**: A database connection is established using the `get_db()` context manager:

```python
@asynccontextmanager
async def get_db():
    global db
    if db is None:
        app_config = load_config()
        db_config = app_config.get("db")
        openai_key = app_config.get("OPENAI_API_KEY")
        os.environ['OPENAI_API_KEY'] = openai_key
        db = SQLDatabase.from_uri(f"postgresql://{db_config.get('username')}:{db_config.get('password')}@{db_config.get('host')}:{db_config.get('port')}/{db_config.get('dbname')}")
    try:
        yield db
    except Exception as e:
        db = None
        raise e
    finally:
        pass
```

3. **Agent Creation**: A new LangGraph agent is created using `create_fresh_agent(db)`:

```python
def create_fresh_agent(db):
    """Create a new instance of the agent for each request"""
    workflow = text2sql_with_analysis_agent(db)
    return workflow.compile()
```

4. **Query Processing**: The agent processes the query through its workflow stages:

```python
agent_input_query = {
    "messages": [HumanMessage(content=f"For this {agent_prompt} answer the following question. {query_request.query}")]
}

if query_request.stream:
    # Convert the generator to an async generator for streaming
    async def async_stream():
        for chunck in agent.stream(agent_input_query):
            yield chunck

    # Returns response as a stream
    return StreamingResponse(stream_response(async_stream()), media_type="text/event-stream")
else:
    # Returns a final response after complete query execution
    response = agent.invoke(agent_input_query)
    result = process_agent_response(response)
    return QueryResponse(result=result)
```

5. **Response Processing**: The agent's response is processed by `process_agent_response()`:

```python
def process_agent_response(response) -> str:
    """Process the agent response."""
    if not response.get("messages"):
        return "No response from the agent."
    
    last_message = response["messages"][-1]
    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "SubmitFinalAnswer":
                args = tool_call["args"]
                if isinstance(args, str):
                    args = json.loads(args)
                return args["final_answer"]
    
    content = last_message.content
    if any(keyword in content.upper() for keyword in ['SELECT', 'FROM']):
        return "The query is being executed. Please wait for the response."
    
    return content
```

6. **Response Return**: The response is returned to the user as JSON or as a stream:

```python
async def stream_response(response) -> Generator[str, None, None]:
    """Stream the response."""
    try:
        for chunck in response:
            if chunck.get("messages"):
                last_message = chunck["messages"][-1]
                if hasattr(last_message, "tool_calls"):
                    for tool_call in last_message.tool_calls:
                        if tool_call["name"] == "SubmitFinalAnswer":
                            args = tool_call["args"]
                            if isinstance(args, str):
                                args = json.loads(args)
                            yield f"data: {args['final_answer']}\n\n"
                else:
                    yield f"data: {last_message.content}\n\n"
            await asyncio.sleep(0.1)
    except Exception as e:
        yield f"data: Error Occurred: {str(e)}\n\n"
    finally:
        yield "data: [DONE]\n\n"
```

## Configuration System

The application uses a flexible configuration system that loads environment-specific settings from JSON files.

### Configuration Files:

- **local.json**: For local development with SSH tunneling.
- **sandbox.json**: For sandbox environment testing.
- **production.json**: For production deployment.

### Configuration Structure:

```json
{
    "ssh": {
        "bastion_host": "test-url.ap-west-1.compute.amazonaws.com",
        "bastion_user": "ubuntu",
        "database_host": "sandbox-db.test-url.amazonaws.com",
        "database_port": "5432",
        "local_port": "63333"
    },
    "db": {
        "host": "localhost",
        "dbname": "testdb",
        "username": "openleaf",
        "password": "test",
        "port": 63333
    },
    "OPENAI_API_KEY": "sk-proj-103m-testkey",
    "aws": {
        "AWS_ACCESS_KEY_ID": "testkey",
        "AWS_SECRET_ACCESS_KEY": "testkey",
        "AWS_REGION": "us-east-1"
    }
}
```

### Configuration Loading:

The `load_config()` function in `configs/config.py` loads the appropriate configuration file based on the `APP_ENV` environment variable, defaulting to 'local' if not set:

```python
def load_config() -> Dict[str, Any]:
    """
    Load configuration based on APP_ENV environment variable.
    Defaults to 'local' if APP_ENV is not set.
    
    Returns:
        dict: Configuration dictionary containing all settings from the JSON file
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    env = os.getenv('APP_ENV', 'local')
    
    file_name = os.path.join('configs/', f'{env}.json')
    
    try:
        with open(file_name, 'r') as f:
            config_data = json.load(f)
            return config_data
            
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to read config file ({file_name}): {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse config file ({file_name}): {str(e)}")
```

## Database Access

The application connects to a PostgreSQL database in two ways:

### 1. Local Development:

- Uses SSH tunneling to connect to the RDS database.
- The tunnel is established by the `tunnel.sh` script:

```bash
#!/bin/sh

# Exit on any error
set -e

# Check if APP_ENV is set
if [ -z "${APP_ENV}" ]; then
    echo "No APP_ENV detected - running in local development mode with SSH tunneling..."
    
    # Start SSH tunnel in background with your local configuration
    echo "Setting up SSH tunnel to PostgreSQL RDS..."
    
    # Use sandbox configuration for local development
    ssh -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -i /root/.ssh/sandbox_key.pem \
        -f -N -L 63333:sandbox-db.test-url.amazonaws.com:5432 \
        ubuntu@test-url.ap-west-1.compute.amazonaws.com

    # Wait for tunnel to establish
    sleep 2

    # Verify tunnel is established
    if ! netstat -an | grep "LISTEN" | grep -q ":63333"; then
        echo "Failed to establish SSH tunnel on port 63333"
        exit 1
    fi

    echo "SSH tunnel successfully established on port 63333"
    echo "Database accessible at localhost:63333"

else
    echo "Running in ${APP_ENV} environment - no SSH tunnel needed..."
    echo "Application will use direct RDS connection via ECS/EC2 network"
fi

# Start the main application
echo "Starting application..."
exec uvicorn api:app --host 0.0.0.0 --port 5000
```

- Database is accessed via localhost:63333.

### 2. Production/Sandbox:

- Direct connection to the RDS database.
- No SSH tunnel needed.
- Uses environment-specific configuration.

## Running Locally

### Prerequisites:

- Python 3.12
- Docker (optional)
- Access to the RDS database (via SSH key)

### Setup Steps:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Openleaf-AI-Server
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the environment**:
   - Ensure the `configs/local.json` file has the correct credentials.
   - Set the `APP_ENV` environment variable to "local" (or leave unset).

5. **Run the application**:
   ```bash
   python api.py
   ```
   Or with uvicorn directly:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 5000
   ```

### Using Docker:

1. **Build the Docker image**:
   ```bash
   docker build -t openleaf-ai-server .
   ```

2. **Run the container**:
   ```bash
   docker run -p 5000:5000 openleaf-ai-server
   ```

## API Usage

### Chat Endpoint

```
POST /ai/chat
```

**Request Body**:
```json
{
  "query": "How many orders were delivered last month?",
  "user_id": "1ecf71e1-1748-425b-9071-56e925a391a3",
  "is_customer": false,
  "stream": false
}
```

**Response**:
```json
{
  "result": "Last month, there were 1,234 delivered orders."
}
```

### Streaming Response

To receive a streaming response, set the `stream` parameter to `true` in the request body. The response will be returned as a server-sent events (SSE) stream.

## Troubleshooting

### Common Issues:

1. **Database Connection Issues**:
   - Check if the SSH tunnel is established (port 63333).
   - Verify database credentials in the configuration.
   - Ensure the RDS instance is accessible.

2. **Rate Limiting**:
   - The application has rate limits (5 requests per minute).
   - Check the response headers for retry information.

3. **Logging**:
   - Logs are written to server.log.
   - Logs are uploaded to S3 every 60 seconds.
   - Check the logs for detailed error information.

## Security Considerations

- API keys and credentials are stored in configuration files.
- SSH tunneling provides secure database access.
- Rate limiting prevents abuse.
- Query validation prevents dangerous SQL operations.

## Conclusion

The Openleaf AI Server provides a powerful natural language interface to a PostgreSQL database. It uses modern AI techniques to understand user queries, generate SQL, and provide meaningful responses. The application is designed to be flexible, secure, and easy to deploy in various environments. 
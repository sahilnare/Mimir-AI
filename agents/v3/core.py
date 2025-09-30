import os
import re
import json
import asyncio
from typing import Dict, List, Literal, Annotated, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from agents.v3.vector_store import VectorStoreManager
from langgraph.prebuilt import ToolNode

from langchain_core.messages import ToolMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase



class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    similar_questions: List[Dict]
    relevant_tables: List[str]
    schema_docs: Dict[str, str]
    is_sql_query: bool


class SubmitFinalAnswer(BaseModel):
    """Submit the final answers to the user based on the query results."""
    final_answer: str = Field(..., description="The final answer to the user.")


class SQLAgent:
    def __init__(self, db_uri: str, config: Dict, vectorstore: VectorStoreManager):
        """
        Initialize SQL Agent with database connection and services
        
        Args:
            db_uri: Database connection URI
        """
        self.db = SQLDatabase.from_uri(db_uri)
        self.db_name = db_uri.split('/')[-1].split('?')[0]
        self.vectorstore = vectorstore
        self.api_key = config.get("GOOGLE_API_KEY")
        
        # Load schema documentation
        self.schema_docs = self._load_schema_docs(config.get("schema_docs_path"))
        self.table_names = self._get_table_names()
        
        # Create tools
        self.db_query_tool = self._create_db_query_tool()
        
        # Build and compile workflow
        self.app = self._build_workflow()
        
    def _load_schema_docs(self, path: str) -> Dict[str, str]:
        """Load schema documentation from JSON file"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Schema docs loading failed: {e}")
            return {}
        
    def _get_table_names(self):
        """Extract all table names from schema"""
        all_tables = []
        for db in self.schema_docs.get('databases', []):
            if db.get('database_name') == self.db_name:
                all_tables = [table['table_name'] for table in db.get('tables', [])]
                break
        return all_tables
    
    def _create_db_query_tool(self):
        """Create database query tool with safety validation"""
        @tool
        def db_query_tool(query: str) -> str:
            """Execute a SQL query against the database and get back the result."""
            print("query db")
            if not self._validate_query(query):
                return "Error: Query contains unsafe operations"
            
            result = self.db.run_no_throw(query)
            print(f"Result: {result}")
            
            if result.startswith('[') and result.endswith(']'):
                try:
                    result = re.sub(r'UUID\([\'"]([^\'"]*)[\'"]\)', r'\1', result)
                    result = re.sub(r'datetime\.date\((\d+), (\d+), (\d+)\)', r'\1-\2-\3', result)
                    result = result.replace('None', '""')
                except Exception as e:
                    print(f'Error formatting result: {e}')
            
            return result or "Error: Query failed. Please rewrite your query and try again."
        
        return db_query_tool
    
    def _validate_query(self, query: str) -> bool:
        """Validate SQL query for safety"""
        danger_patterns = [
            r'\bDROP\b', r'\bDELETE\b', r'\bTRUNCATE\b', 
            r'\bUPDATE\b', r'\bINSERT\b', r'\bALTER\b', r'\bCREATE\b'
        ]
        return not any(re.search(pattern, query, re.IGNORECASE) for pattern in danger_patterns)
    
    def _extract_table_names_from_sql(self, sql_query: str) -> List[str]:
        """Extract table names from SQL query by matching against schema tables"""
        # Normalize SQL query for matching
        sql_lower = sql_query.lower()
        
        # Find which tables from schema are referenced in the query
        referenced_tables = []
        for table_name in self.table_names:
            pattern = r'\b' + re.escape(table_name.lower()) + r'\b'
            if re.search(pattern, sql_lower):
                referenced_tables.append(table_name)
        
        return referenced_tables
    
    def _detect_sql_intent(self, query: str) -> bool:
        """Detect if query is asking for data (SQL intent) vs general question"""
        sql_keywords = [
            'show', 'get', 'find', 'list', 'count', 'how many', 'total', 
            'sum', 'average', 'recent', 'latest', 'orders', 'customers',
            'report', 'data', 'records', 'table', 'database'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in sql_keywords)
    
    
    async def _search_similar_node(self, state: State) -> Dict:
        """Search for similar questions and extract relevant tables"""
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        if not user_messages:
            return {
                "similar_questions": [],
                "relevant_tables": [],
                "is_sql_query": False
            }
        
        user_query = user_messages[-1].content
        is_sql_query = self._detect_sql_intent(user_query)
        
        if not is_sql_query:
            return {
                "similar_questions": [],
                "relevant_tables": [],
                "is_sql_query": False
            }
        print(user_query.split(":")[1].strip())
        # Async search for similar questions
        similar_questions = await self.vectorstore.search_similar_questions(user_query.split(":")[1].strip(), {"db_name": self.db_name}, k=3)
        # Extract table names from similar SQL queries
        relevant_tables = []
        for q in similar_questions:
            if q["table_names"]:
                tables = q["table_names"]
            else:
                tables = self._extract_table_names_from_sql(q['sql_query'])
            relevant_tables.extend(tables)
        
        relevant_tables = list(dict.fromkeys(relevant_tables))  # Remove duplicates
        print(relevant_tables)
        
        return {
            "similar_questions": similar_questions,
            "relevant_tables": relevant_tables,
            "is_sql_query": is_sql_query
        }
    
    async def _general_response_node(self, state: State) -> Dict:
        """Handle general non-SQL questions"""
        if state.get("is_sql_query", False):
            return {"messages": []}
        
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        if not user_messages:
            return {"messages": []}
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=self.api_key,
            temperature=0.7
        ).bind_tools([SubmitFinalAnswer])
        
        system_msg = """You are a helpful assistant. Answer the user's question directly and conversationally. 
        Use the SubmitFinalAnswer tool to provide your response."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("placeholder", "{messages}")
        ])
        
        chain = prompt | llm
        response = await chain.ainvoke(state)
        
        return {"messages": [response]}
    
    def _get_schema_docs_node(self, state: State) -> Dict:
        """Get schema documentation based on relevant tables"""
        if not state.get("is_sql_query", False):
            return {"schema_docs": {}}
        
        if state.get("relevant_tables"):
            # Filter schema for relevant tables only
            schema_docs = {}
            for db in self.schema_docs.get('databases', []):
                if db.get('database_name') == self.db_name:
                    for table in db.get('tables', []):
                        if table['table_name'] in state["relevant_tables"]:
                            schema_docs[table['table_name']] = table
                    break
        else:
            # Return all tables if no relevant tables found
            schema_docs = {}
            for db in self.schema_docs.get('databases', []):
                if db.get('database_name') == self.db_name:
                    for table in db.get('tables', []):
                        schema_docs[table['table_name']] = table
                    break
        return {"schema_docs": schema_docs}
    
    async def _query_gen_node(self, state: State) -> Dict:
        """Generate SQL query with enhanced context"""
        if not state.get("is_sql_query", False):
            return {"messages": []}
        
        # Check if we just got results from execute_query
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], ToolMessage):
            # We have query results, now generate final answer
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=self.api_key,
                temperature=0.3
            ).bind_tools([SubmitFinalAnswer], tool_choice="SubmitFinalAnswer")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Based on the query results, provide a clear, concise answer to the user's question. Use the SubmitFinalAnswer tool."),
                MessagesPlaceholder(variable_name="messages")
            ])
            
            chain = prompt | llm
            response = await chain.ainvoke(state)
            return {"messages": [response]}
        
        # Generate SQL query
        schema_context = ""
        if state.get("schema_docs"):
            schema_context = "Database Schema:\n"
            for table_name, table_info in state["schema_docs"].items():
                schema_context += f"\n## Table: {table_name}\n"
                schema_context += f"Description: {table_info.get('description', 'N/A')}\n"
                schema_context += "Columns:\n"
                
                for col in table_info.get('columns', []):
                    col_name = col.get('column_name', 'unknown')
                    col_type = col.get('data_type', 'unknown')
                    col_desc = col.get('description', '')
                    schema_context += f"  - {col_name} ({col_type}): {col_desc}\n"
                
                schema_context += "\n"
        
        similar_context = ""
        if state.get("similar_questions"):
            similar_context = "\nSimilar Questions and Queries for Reference:\n"
            for i, q in enumerate(state["similar_questions"][:3], 1):
                similar_context += f"{i}. Question: {q['question']}\n   SQL: {q['sql_query']}\n\n"
        
        system_prompt = f"""You are a SQL expert. Generate a syntactically correct PostgreSQL query.

    {schema_context}

    {similar_context}

    Guidelines:
    - Use schema documentation to understand table structures
    - Reference similar queries if helpful
    - LIMIT to 60 results unless specified otherwise
    - Use ORDER BY for meaningful ordering
    - Use LIKE for status matching (e.g., 'RTO%')
    - You MUST call the db_query_tool to execute your query
    - NEVER make DML statements (INSERT, UPDATE, DELETE, DROP, etc.)
    """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=self.api_key,
            temperature=0.3
        ).bind_tools([self.db_query_tool], tool_config={
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": ["db_query_tool"]
            }
        })
        
        chain = prompt | llm
        response = await chain.ainvoke(state)
        print(f"Generated tool calls: {response.tool_calls}")
        
        return {"messages": [response]}

    
    async def _correct_query_node(self, state: State) -> Dict:
        """Double check and correct SQL query before execution"""
        system_prompt = """You are a SQL expert. Double check this PostgreSQL query for mistakes:
        - NOT IN with NULL values
        - UNION vs UNION ALL usage  
        - Data type mismatches
        - Proper identifier quoting
        - PostgreSQL-specific casting (::)
        - Proper joins and references
        - Date/time handling
        
        If there are mistakes, rewrite the query. If correct, reproduce the original.
        You must call db_query_tool to execute the query."""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=self.api_key,
            temperature=0
        ).bind_tools([self.db_query_tool], tool_config={
        "function_calling_config": {
            "mode": "ANY",
            "allowed_function_names": ["db_query_tool"]
        }
    })
        
        chain = prompt | llm
        last_message = state["messages"][-1]
        response = await chain.ainvoke({"messages": [HumanMessage(content=last_message.content)]})
        
        return {"messages": [response]}
    
    def _should_continue(self, state: State) -> Literal[END, "execute_query", "query_gen", "general_response"]:
        """Determine next step in workflow"""
        if not state.get("is_sql_query", False):
            return "general_response"
        
        messages = state["messages"]
        if not messages:
            return "query_gen"
        
        last_message = messages[-1]
        
        # Check If SubmitFinalAnswer was called
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            if any(tc["name"] == "SubmitFinalAnswer" for tc in last_message.tool_calls):
                return END
            # If db_query_tool was called, execute it
            if any(tc["name"] == "db_query_tool" for tc in last_message.tool_calls):
                return "execute_query"
        
        # If we just got tool results, go back to query_gen to format answer
        if isinstance(last_message, ToolMessage):
            if last_message.content.startswith("Error:"):
                # On error, regenerate query
                return "query_gen"
            else:
                # On success, format the answer
                return "query_gen"
        
        # Default: generate query
        return "query_gen"
    
    def _build_workflow(self) -> StateGraph:
        """Build and compile the LangGraph workflow"""
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("search_similar", self._search_similar_node)
        workflow.add_node("get_schema_docs", self._get_schema_docs_node)
        workflow.add_node("general_response", self._general_response_node)
        workflow.add_node("query_gen", self._query_gen_node)
        workflow.add_node("execute_query", ToolNode([self.db_query_tool]))
        
        # Define edges
        workflow.add_edge(START, "search_similar")
        workflow.add_edge("search_similar", "get_schema_docs")
        workflow.add_edge("get_schema_docs", "query_gen")
        workflow.add_conditional_edges("query_gen", self._should_continue)
        workflow.add_edge("general_response", END)
        workflow.add_edge("execute_query", "query_gen")
        
        # Compile workflow
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
        
    async def get_response(self, query: str, config: Dict[str, Any] = None) -> str:
        """Get a single response (non-streaming) - async version"""
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        result = await self.app.ainvoke({
            "messages": [HumanMessage(content=query)],
            "similar_questions": [],
            "relevant_tables": [],
            "schema_docs": {},
            "is_sql_query": False
        }, config)
        
        # Extract final answer
        for message in reversed(result["messages"]):
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    if tc["name"] == "SubmitFinalAnswer":
                        return tc["args"]["final_answer"]
            elif isinstance(message, ToolMessage) and not message.content.startswith("Error:"):
                return message.content
        
        return "I couldn't process your request. Please try again."
    
    async def stream_response(self, query: str, config: Dict[str, Any] = None):
        """Get streaming response - async version"""
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "similar_questions": [],
            "relevant_tables": [],
            "schema_docs": {},
            "is_sql_query": False
        }
        
        async for event in self.app.astream(initial_state, config, stream_mode="values"):
            yield event
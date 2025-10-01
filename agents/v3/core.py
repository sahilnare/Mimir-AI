import os
import re
import json
import asyncio
from typing import Dict, List, Any, AsyncGenerator, Optional
from dataclasses import dataclass, field
from sqlalchemy import text

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from agents.v3.vector_store import VectorStoreManager


@dataclass
class ConversationState:
    messages: List[Any] = field(default_factory=list)
    similar_questions: List[Dict] = field(default_factory=list)
    relevant_tables: List[str] = field(default_factory=list)
    schema_docs: Dict[str, Any] = field(default_factory=dict)
    is_sql_query: bool = False
    query_result: Optional[str] = None


class SQLAgent:
    def __init__(self, db_uri: str, config: Dict, vectorstore: VectorStoreManager):
        self.db = SQLDatabase.from_uri(db_uri)
        self.db_name = db_uri.split('/')[-1].split('?')[0]
        self.vectorstore = vectorstore
        self.api_key = config.get("GOOGLE_API_KEY")
        
        # Load schema documentation
        self.schema_docs = self._load_schema_docs(config.get("schema_docs_path"))
        self.table_names = self._get_table_names()
        
        # Initialize LLMs
        self.query_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=self.api_key,
            temperature=0.3
        )
        
        self.answer_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=self.api_key,
            temperature=0.3,
            streaming=True
        )
        
        # Conversation history
        self.conversations: Dict[str, ConversationState] = {}
    
    def _load_schema_docs(self, path: str) -> Dict[str, str]:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Schema docs loading failed: {e}")
            return {}
    
    def _get_table_names(self) -> List[str]:
        all_tables = []
        for db in self.schema_docs.get('databases', []):
            if db.get('database_name') == self.db_name:
                all_tables = [table['table_name'] for table in db.get('tables', [])]
                break
        return all_tables
    
    def _validate_query(self, query: str) -> bool:
        danger_patterns = [
            r'\bDROP\b', r'\bDELETE\b', r'\bTRUNCATE\b', 
            r'\bUPDATE\b', r'\bINSERT\b', r'\bALTER\b', r'\bCREATE\b'
        ]
        return (
            "f17d6407-9d6b-45f5-854c-30a43b4b9615" in query 
            and not any(re.search(pattern, query, re.IGNORECASE) for pattern in danger_patterns)
        )    
    
    def _execute_query(self, query: str) -> str:
        """Execute SQL query with safety checks"""
        if not self._validate_query(query):
            return "Error: Query contains unsafe operations"
        
        try:
            with self.db._engine.connect() as conn:
                result_proxy = conn.execute(text(query))
                all_columns = list(result_proxy.keys())

                # Priority columns if available
                preferred_columns = [
                    "order_id", 
                    "customer_city", 
                    "customer_state", 
                    "warehouse_name", 
                    "price_of_shipment"
                ]

                # Pick preferred columns if present, else fallback to first 5
                columns = [col for col in preferred_columns if col in all_columns]
                if not columns:
                    columns = all_columns[:5]

                rows = result_proxy.fetchmany(5)

                limited_rows = []
                for row in rows:
                    row_dict = dict(zip(all_columns, row))
                    limited_row = {k: row_dict[k] for k in columns if k in row_dict}
                    limited_rows.append(limited_row)

                formatted_result = f"Columns: {columns}\n\nData:\n"
                for i, row in enumerate(limited_rows, 1):
                    formatted_result += f"Row {i}: {row}\n"
                
                return formatted_result
        except Exception as e:
            return f"Error: Query execution failed - {str(e)}"
    
    def _extract_table_names(self, sql_query: str) -> List[str]:
        sql_lower = sql_query.lower()
        referenced_tables = []
        for table_name in self.table_names:
            pattern = r'\b' + re.escape(table_name.lower()) + r'\b'
            if re.search(pattern, sql_lower):
                referenced_tables.append(table_name)
        return referenced_tables
    
    def _detect_sql_intent(self, query: str) -> bool:
        sql_keywords = [
            'show', 'get', 'find', 'list', 'count', 'how many', 'total', 'shippment', 
            'sum', 'average', 'recent', 'latest', 'orders', 'customers', 'rto',
            'report', 'data', 'records', 'table', 'database', "delivery"
        ]
        return any(keyword in query.lower() for keyword in sql_keywords)
    
    async def _search_similar(self, state: ConversationState, query: str):
        """Step 1: Search for similar questions"""
        state.is_sql_query = self._detect_sql_intent(query)
        
        if not state.is_sql_query:
            return
        
        print(f"Searching similar questions for: {query}")
        state.similar_questions = await self.vectorstore.search_similar_questions(
            query, 
            {"db_name": self.db_name}, 
            k=3
        )
        
        # Extract relevant tables
        for q in state.similar_questions:
            tables = q.get("table_names") or self._extract_table_names(q['sql_query'])
            state.relevant_tables.extend(tables)
        
        state.relevant_tables = list(dict.fromkeys(state.relevant_tables))
        print(f"Relevant tables: {state.relevant_tables}")
    
    def _get_schema_context(self, state: ConversationState) -> str:
        """Step 2: Build schema context"""
        if not state.is_sql_query:
            return ""
        
        # Get relevant schema docs
        for db in self.schema_docs.get('databases', []):
            if db.get('database_name') == self.db_name:
                for table in db.get('tables', []):
                    if not state.relevant_tables or table['table_name'] in state.relevant_tables:
                        state.schema_docs[table['table_name']] = table
                break
        
        if not state.schema_docs:
            return ""
        
        schema_context = "Database Schema:\n"
        for table_name, table_info in state.schema_docs.items():
            schema_context += f"\n## Table: {table_name}\n"
            schema_context += f"Description: {table_info.get('description', 'N/A')}\n"
            schema_context += "Columns:\n"
            
            for col in table_info.get('columns', []):
                col_name = col.get('column_name', 'unknown')
                col_type = col.get('data_type', 'unknown')
                col_desc = col.get('description', '')
                schema_context += f"  - {col_name} ({col_type}): {col_desc}\n"
        
        return schema_context
    
    def _get_similar_context(self, state: ConversationState) -> str:
        """Build similar questions context"""
        if not state.similar_questions:
            return ""
        
        similar_context = "\nSimilar Questions and Queries for Reference:\n"
        for i, q in enumerate(state.similar_questions[:3], 1):
            similar_context += f"{i}. Question: {q['question']}\n   SQL: {q['sql_query']}\n\n"
        
        return similar_context
    
    async def _generate_sql(self, state: ConversationState, query: str) -> Optional[str]:
        """Step 3: Generate SQL query"""
        schema_context = self._get_schema_context(state)
        similar_context = self._get_similar_context(state)
        
        system_prompt = f"""You are a SQL expert. Generate a syntactically correct PostgreSQL query.

{schema_context}

{similar_context}

Guidelines:
- User id is f17d6407-9d6b-45f5-854c-30a43b4b9615
- Always filter by orders.user_id in your generated SQL query
- Use schema documentation to understand table structures
- Reference similar queries if helpful
- Use ORDER BY for meaningful ordering
- Use LIKE for status matching (e.g., 'RTO%')
- NEVER make DML statements (INSERT, UPDATE, DELETE, DROP, etc.)
- Always add LIMIT 60 to your queries

Output ONLY the SQL query, nothing else."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])
        
        chain = prompt | self.query_llm
        response = await chain.ainvoke({"query": query})
        
        # Extract SQL from response
        sql_query = response.content.strip()
        # Remove markdown code blocks if present
        sql_query = re.sub(r'```sql\s*|\s*```', '', sql_query).strip()
        
        print(f"Generated SQL: {sql_query}")
        return sql_query
    
    async def _stream_answer(self, state: ConversationState, query: str) -> AsyncGenerator[str, None]:
        """Step 4: Stream the formatted answer"""
        if state.is_sql_query and state.query_result:
            # Format SQL results as table
            system_prompt = """Based on the query results, provide a clear answer formatted as a markdown table.

FORMATTING RULES:
1. The query results output includes column names and row data
2. Create a markdown table with the column names as headers
3. Each row of data should be a table row
4. Format the data clearly and readably
5. Add a brief summary before the table explaining what the data shows

Example format:
Here are the order details:

| Column1 | Column2 | Column3 |
|---------|---------|---------|
| Value1  | Value2  | Value3  |
| Value4  | Value5  | Value6  |"""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Original question: {query}\n\nQuery results:\n{results}")
            ])
        else:
            # General response
            system_prompt = "You are a helpful assistant. Answer the user's question directly and conversationally."
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{query}")
            ])
        
        chain = prompt | self.answer_llm
        
        input_data = {"query": query}
        if state.query_result:
            input_data["results"] = state.query_result
        
        async for chunk in chain.astream(input_data):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
    
    async def stream_response(
        self, 
        query: str, 
        thread_id: str = "default"
    ) -> AsyncGenerator[str, None]:
        """Main entry point: Stream response for a query"""
        
        # Get or create conversation state
        if thread_id not in self.conversations:
            self.conversations[thread_id] = ConversationState()
        
        state = self.conversations[thread_id]
        state.messages.append(HumanMessage(content=query))
        
        # Step 1: Search similar questions
        await self._search_similar(state, query)
        
        # Step 2 & 3: Generate and execute SQL if needed
        if state.is_sql_query:
            sql_query = await self._generate_sql(state, query)
            
            if sql_query:
                state.query_result = self._execute_query(sql_query)
                
                # Check for errors
                if state.query_result.startswith("Error:"):
                    yield f"I encountered an error executing the query: {state.query_result}\n"
                    return
        
        # Step 4: Stream the answer
        async for chunk in self._stream_answer(state, query):
            yield chunk
    
    async def get_response(
        self, 
        query: str, 
        thread_id: str = "default"
    ) -> str:
        """Get complete response (non-streaming)"""
        response_parts = []
        async for chunk in self.stream_response(query, thread_id):
            response_parts.append(chunk)
        return "".join(response_parts)
    
    def clear_conversation(self, thread_id: str = "default"):
        """Clear conversation history"""
        if thread_id in self.conversations:
            del self.conversations[thread_id]

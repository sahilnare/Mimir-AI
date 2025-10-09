import os
import re
import json
import asyncio
import base64
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from io import BytesIO
from typing import Dict, List, Any, AsyncGenerator, Optional, Tuple
from dataclasses import dataclass, field
from sqlalchemy import text

from langchain_core.messages import HumanMessage
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
    full_dataframe: Optional[pd.DataFrame] = None
    last_sql_query: Optional[str] = None
    needs_chart: bool = False
    chart_data: Optional[Dict] = None


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
            temperature=0.4,
            streaming=False
        )
        
        self.answer_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=self.api_key,
            temperature=0.3,
            streaming=True
        )
        
        self.chart_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=self.api_key,
            temperature=0.4,
            streaming=False,
            max_output_tokens=1024
        )
        self.config_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            api_key=self.api_key,
            temperature=0.2,
            streaming=False,
            max_output_tokens=1024
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
    
    def _validate_query(self, query: str, user_id: str) -> bool:
        danger_patterns = [
            r'\bDROP\b', r'\bDELETE\b', r'\bTRUNCATE\b', 
            r'\bUPDATE\b', r'\bINSERT\b', r'\bALTER\b', r'\bCREATE\b'
        ]
        return (
            user_id in query 
            and not any(re.search(pattern, query, re.IGNORECASE) for pattern in danger_patterns)
        )    
    
    def _execute_query(self, query: str, user_id: str) -> Tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Execute SQL query and return preview string, full DataFrame, and preview DataFrame"""
        if not self._validate_query(query, user_id):
            return "Error: Query contains unsafe operations", None, None

        try:
            with self.db._engine.connect() as conn:
                result_proxy = conn.execute(text(query))
                all_columns = list(result_proxy.keys())
                
                # Fetch ALL rows for DataFrame
                all_rows = result_proxy.fetchall()
                
                print(all_rows)
                # Create full DataFrame
                full_df = None
                if all_rows and len(all_rows) > 5:
                    full_df = pd.DataFrame(all_rows, columns=all_columns)
  
                # Priority columns for preview
                preferred_columns = [
                    "order_id",
                    "customer_city",
                    "customer_state",
                    "warehouse_name",
                    "price_of_shipment"
                ]
                
                # Pick preferred columns if present, else fallback to first 10
                preview_columns = [col for col in preferred_columns if col in all_columns]
                if not preview_columns or len(preview_columns) < 3:
                    preview_columns = all_columns[:10]
                
                # Create preview DataFrame (first 10 rows with preview columns)
                preview_rows = all_rows[:10]
                if preview_rows:
                    preview_df = pd.DataFrame(preview_rows, columns=all_columns)[preview_columns]
                else:
                    preview_df = pd.DataFrame(columns=preview_columns)
                
                # Create preview string (first 5 rows only)
                limited_rows = []
                for row in preview_rows:
                    row_dict = dict(zip(all_columns, row))
                    limited_row = {k: row_dict[k] for k in preview_columns if k in row_dict}
                    limited_rows.append(limited_row)
                
                formatted_result = f"Columns: {preview_columns}\n\nData (showing {len(preview_rows)} of {len(all_rows)} rows):\n"
                for i, row in enumerate(limited_rows, 1):
                    formatted_result += f"Row {i}: {row}\n"
                
                return formatted_result, preview_df, full_df
                
        except Exception as e:
            return f"Error: Query execution failed - {str(e)}", None, None
    
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
            'report', 'data', 'records', 'table', 'database', "delivery", "which"
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
    
    async def _generate_sql(self, state: ConversationState, query: str, user_id: str) -> Optional[str]:
        """Step 3: Generate SQL query"""
        schema_context = self._get_schema_context(state)
        similar_context = self._get_similar_context(state)
        
        system_prompt = f"""You are a SQL expert. Generate a syntactically correct PostgreSQL query.

{schema_context}

{similar_context}

Guidelines:
- User id is {user_id}
- Always filter by orders.user_id in your generated SQL query
- Use schema documentation to understand table structures
- Reference similar queries if helpful
- Use ORDER BY for meaningful ordering
- Use LIKE for status matching (e.g., 'RTO%')
- NEVER make DML statements (INSERT, UPDATE, DELETE, DROP, etc.)
- Always add LIMIT 60 to your queries

Output ONLY the SQL query, nothing else."""
        
        system_prompt = system_prompt.replace("{", "{{").replace("}", "}}")
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
    
    async def _decide_chart_need(self, preview_df: DataFrame, query: str) -> bool:
        """Decide if the query results need a chart visualization"""
        if not preview_df is not None or len(preview_df) == 0:
            return False
        
        df = preview_df
        
        # Simple heuristics first
        # Single value results don't need charts
        if len(df) == 1 and len(df.columns) <= 2:
            return False
        
        # Count queries with single result don't need charts
        if len(df) == 1 and any(col.lower() in ['count', 'total', 'sum'] for col in df.columns):
            return False
                
        # Ask LLM for decision
        decision_prompt = f"""Analyze if this query result needs a chart visualization.

User Question: {query}

Data Info:
- Rows: {len(df)}
- Columns: {list(df.columns)}
- Sample data: {df.head(3).to_dict('records')}

Decision Rules:
- Simple counts/single values: NO chart
- Time series data: YES chart
- Comparisons between categories: YES chart
- Multiple numeric values to compare: YES chart
- List of records without aggregation: NO chart

Respond with ONLY "YES" or "NO"."""
        
        try:
            response = await self.chart_llm.ainvoke([HumanMessage(content=decision_prompt)])
            decision = response.content.strip().upper()
            return "YES" in decision
        except Exception as e:
            print(f"Chart decision error: {e}")
            return False
    
    async def _generate_chart_config(self, df: DataFrame, query: str) -> Optional[Dict]:
        """Generate chart configuration"""
        
        config_prompt = f"""Generate chart configuration for this data.

User Question: {query}

Data Info:
- Columns: {list(df.columns)}
- Data types: {df.dtypes.to_dict()}
- Sample: {df.head(3).to_dict('records')}

Return JSON with this structure:
{{
    "chart_type": "bar|line|pie|scatter",
    "x_column": "exact_column_name",
    "y_column": "exact_column_name",
    "title": "Chart title",
    "xlabel": "X-axis label",
    "ylabel": "Y-axis label"
}}

Guidelines:
- Use exact column names from the data
- bar: for categories comparison
- line: for **time series** or trends
- pie: for **proportions** (max 10 categories), **breakdown** of orders
- Choose most appropriate chart type

Output ONLY valid JSON."""
        
        try:
            response = await self.config_llm.ainvoke([HumanMessage(content=config_prompt)])
            config_str = response.content.strip()
            print(f"config:{config_str}")
            
            # Extract JSON from response
            config_str = re.sub(r'```json\s*|\s*```', '', config_str).strip()
            config = json.loads(config_str)
            
            # Validate columns exist
            if config['x_column'] not in df.columns or config['y_column'] not in df.columns:
                print(f"Invalid columns in config: {config}")
                return None
            
            return config
        except Exception as e:
            print(f"Chart config generation error: {e}")
            return None
    
    def _create_chart(self, df: DataFrame, config: Dict) -> Optional[str]:
        """Create chart and return base64 encoded image"""
        try:
            chart_type = config.get('chart_type', 'bar')
            x_col = config['x_column']
            y_col = config['y_column']
            
            # Make a copy to avoid modifying original
            df_plot = df.copy()
            
            # Convert datetime columns to string for better display
            if pd.api.types.is_datetime64_any_dtype(df_plot[x_col]):
                df_plot[x_col] = df_plot[x_col].dt.strftime('%Y-%m-%d')
            
            # Ensure y column is numeric
            df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors='coerce')
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create chart based on type
            if chart_type == 'bar':
                ax.bar(df_plot[x_col], df_plot[y_col])
                plt.xticks(rotation=45, ha='right')
            elif chart_type == 'line':
                ax.plot(df_plot[x_col], df_plot[y_col], marker='o', linewidth=2)
                plt.xticks(rotation=45, ha='right')
            elif chart_type == 'pie':
                ax.pie(df_plot[y_col], labels=df_plot[x_col], autopct='%1.1f%%')
                ax.set_ylabel('')
            elif chart_type == 'scatter':
                ax.scatter(df_plot[x_col], df_plot[y_col])
                plt.xticks(rotation=45, ha='right')
            else:
                ax.bar(df_plot[x_col], df_plot[y_col])
                plt.xticks(rotation=45, ha='right')
            
            # Apply labels
            ax.set_title(config.get('title', 'Chart'), fontsize=14, fontweight='bold')
            if chart_type != 'pie':
                ax.set_xlabel(config.get('xlabel', x_col), fontsize=11)
                ax.set_ylabel(config.get('ylabel', y_col), fontsize=11)
            
            # Style improvements
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            # Save to buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            
            # Encode as base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            buffer.close()
            
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            print(f"Chart creation error: {e}")
            return None
    
    async def _stream_answer(self, state: ConversationState, query: str) -> AsyncGenerator[str, None]:
        """Step 4: Stream the formatted answer"""
        if state.is_sql_query and state.query_result:
            # Format SQL results as table
            system_prompt = """You are given query results that include column names and row data.  
Your task is to present the results in a clear and readable format.  

### Formatting Rules:
1. If the query results contain multiple rows:  
   - Create a **markdown table**.  
   - Use the column names as headers.  
   - Display each row of data as a row in the table.  

2. If the query results contain only a single row or a single value:  
   - Do not use a table.  
   - Instead, provide a short, natural-language description of the result.  

3. Ensure the formatting is clean and easy to read.  

4. When a table is presented, include a brief summary (before or after the table) explaining what the data shows.

5. The input data you get is only a preview portion of the full data (The first 10 rows or less)
"""
            
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

        user_id = thread_id.split("_")[1]
        
        state = self.conversations[thread_id]
        state.messages.append(HumanMessage(content=query))
        
        # Step 1: Search similar questions
        await self._search_similar(state, query)
        
        # Step 2 & 3: Generate and execute SQL if needed
        if state.is_sql_query:
            sql_query = await self._generate_sql(state, query, user_id)
            state.last_sql_query = sql_query
            
            if sql_query:
                preview_result, preview_df, full_df = self._execute_query(sql_query, user_id)
                state.query_result = preview_result
                state.full_dataframe = full_df
                print(preview_df)
                # Check for errors
                if state.query_result.startswith("Error:"):
                    yield f"I encountered an error executing the query: {state.query_result}\n"
                    return
        
        # Step 4: Stream the answer
        async for chunk in self._stream_answer(state, query):
            yield chunk
        
        # Step 5: Generate chart if needed
        if state.is_sql_query and preview_df is not None:
            needs_chart = await self._decide_chart_need(preview_df, query)
            print(needs_chart)
            
            if needs_chart:
                yield "\n\n📊 Generating visualization...\n"
                
                chart_config = await self._generate_chart_config(preview_df, query)
                print(chart_config)

                if chart_config:
                    chart_image = self._create_chart(preview_df, chart_config)
                    if chart_image:
                        # Return JSON-like object at the end
                        state.chart_data = chart_image
    
    def get_last_dataframe(self, thread_id: str = "default") -> Optional[pd.DataFrame]:
        """Get the full DataFrame from the last query in this conversation thread"""
        if thread_id in self.conversations:
            df = self.conversations[thread_id].full_dataframe
            self.conversations[thread_id].full_dataframe = None
            return df
        return None
    
    def get_last_chart_data(self, thread_id: str = "default") -> Optional[Dict]:
        """Get the chart configuration from the last query"""
        if thread_id in self.conversations:
            chart_data = self.conversations[thread_id].chart_data
            self.conversations[thread_id].chart_data = None
            return chart_data
        return None
    
    def clear_conversation(self, thread_id: str = "default"):
        """Clear conversation history"""
        if thread_id in self.conversations:
            del self.conversations[thread_id]
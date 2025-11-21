import os
import re
import json
import asyncio
import base64
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from io import BytesIO
from typing import Dict, List, Any, AsyncGenerator, Optional, Tuple, Literal
from dataclasses import dataclass, field
from sqlalchemy import text
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from agents.v3.vector_store import VectorStoreManager


class QueryIntent(str, Enum):
    """Query intent types"""
    GENERAL_KNOWLEDGE = "general_knowledge"  # General questions, greetings, how-tos
    DATA_QUERY = "data_query"  # Needs SQL execution
    RECOMMENDATION = "recommendation"  # Strategic advice based on data analysis
    AMBIGUOUS = "ambiguous"  # Unclear intent


@dataclass
class ConversationState:
    messages: List[Any] = field(default_factory=list)
    similar_questions: List[Dict] = field(default_factory=list)
    relevant_tables: List[str] = field(default_factory=list)
    schema_docs: Dict[str, Any] = field(default_factory=dict)
    query_intent: Optional[QueryIntent] = None
    query_result: Optional[str] = None
    full_dataframe: Optional[pd.DataFrame] = None
    last_sql_query: Optional[str] = None
    needs_chart: bool = False
    chart_data: Optional[Dict] = None
    kpi_results: Dict[str, Any] = field(default_factory=dict)


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
            model="gemini-2.5-pro",
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
            temperature=0.3,
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
        
        self.intent_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            api_key=self.api_key,
            temperature=0.1,
            streaming=False,
            max_output_tokens=512
        )
        
        # Conversation history
        self.conversations: Dict[str, ConversationState] = {}
        self.current_user_id = None
    
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
        """Execute SQL query and return preview string, preview DataFrame, and full DataFrame"""
        if not self._validate_query(query, user_id):
            return "Error: Query contains unsafe operations", None, None

        try:
            with self.db._engine.connect() as conn:
                result_proxy = conn.execute(text(query))
                all_columns = list(result_proxy.keys())
                
                # Fetch ALL rows for DataFrame
                all_rows = result_proxy.fetchall()
                
                print(f"Query returned {len(all_rows)} rows")
                
                # Create full DataFrame
                full_df = None
                if all_rows and len(all_rows) > 5:
                    full_df = pd.DataFrame(all_rows, columns=all_columns)
  
                # Priority columns for preview
                preferred_columns = [
                    "order_id", "customer_city", "customer_state",
                    "warehouse_name", "price_of_shipment", "current_order_status"
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
                
                # Create preview string (first 10 rows only)
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
    
    async def _detect_query_intent(self, query: str) -> QueryIntent:
        """Intelligently detect query intent using LLM"""
        
        intent_prompt = f"""Analyze the user's query and classify it into ONE of these categories:

**Categories:**
1. **general_knowledge**: Greetings, small talk, questions about how things work, definitions, explanations that don't need data
   Examples: "Hi", "What is RTO?", "How does shipping work?", "Explain COD"

2. **data_query**: Questions that require fetching data from database
   Examples: "Show me orders from Mumbai", "How many RTO orders?", "List delivered shipments", "Which carrier has most delays?"

3. **recommendation**: Questions asking for strategic advice, suggestions, or optimization strategies
   Examples: "How to reduce RTO?", "What can I do to improve delivery rates?", "Suggest ways to optimize shipping costs", "How to handle NDR better?"

4. **ambiguous**: Unclear or mixed intent

**User Query:** "{query}"

**Analysis Guidelines:**
- If asking "reccommend me", "how to", "ways to", "suggestions", "what should I do" → likely **recommendation**
- If asking "show", "list", "how many", "which", "get", "find" → likely **data_query**
- If general conversation or asking definitions → **general_knowledge**
- Recommendations often need data analysis FIRST, then strategic advice

Respond with ONLY ONE word: general_knowledge, data_query, recommendation, or ambiguous"""
        
        try:
            response = await self.intent_llm.ainvoke([HumanMessage(content=intent_prompt)])
            intent_str = response.content.strip().lower()
            
            # Map response to enum
            if "general_knowledge" in intent_str:
                return QueryIntent.GENERAL_KNOWLEDGE
            elif "data_query" in intent_str:
                return QueryIntent.DATA_QUERY
            elif "recommendation" in intent_str:
                return QueryIntent.RECOMMENDATION
            else:
                return QueryIntent.AMBIGUOUS
                
        except Exception as e:
            print(f"Intent detection error: {e}")
            return QueryIntent.AMBIGUOUS
        
    def _register_kpi_tools(self, user_id: str):
        """Create KPI calculation tools For recommendation agent"""

        @tool
        def calculate_rto_rate(days: int = 50) -> Dict[str, Any]:
            """
            Calculate RTO (Return to Origin) rate for the user's orders.
            RTO statuses include: RTO, RTO IN TRANSIT, RTO DELIVERED, RTO OFD, RTO INITIATED, RTO NDR

            Args:
              days: int default: 50
            """
            print("calculate_rto_rate tool is called!")
            query = f"""
            SELECT 
                COUNT(*) FILTER (
                    WHERE t.current_order_status IN (
                        'RTO', 'RTO IN TRANSIT', 'RTO DELIVERED', 
                        'RTO OFD', 'RTO INITIATED', 'RTO NDR'
                    )
                ) as rto_count,
                COUNT(*) as total_orders,
                ROUND(
                    100.0 * COUNT(*) FILTER (
                        WHERE t.current_order_status IN (
                            'RTO', 'RTO IN TRANSIT', 'RTO DELIVERED', 
                            'RTO OFD', 'RTO INITIATED', 'RTO NDR'
                        )
                    ) / NULLIF(COUNT(*), 0), 
                    2
                ) as rto_rate
            FROM orders o
            INNER JOIN tracking_orders t ON o.order_id = t.order_id
            WHERE o.user_id = '{user_id}'
                AND t.current_order_status != 'CANCELLED'
                AND t.ordered_date IS NOT NULL
                AND t.ordered_date >= CURRENT_DATE - INTERVAL '{days} days'
            """
            preview, preview_df, full_df = self._execute_query(query, user_id)
            
            if preview.startswith("Error:"):
                return {"error": preview}
            
            if preview_df is not None and len(preview_df) > 0:
                row = preview_df.iloc[0]
                rto_rate = row['rto_rate']
                if pd.isna(rto_rate) or rto_rate is None:
                    rto_rate = 0.0
                
                return {
                    "rto_count": int(row['rto_count']) if not pd.isna(row['rto_count']) else 0,
                    "total_orders": int(row['total_orders']) if not pd.isna(row['total_orders']) else 0,
                    "rto_rate": float(rto_rate),
                    "period_days": days
                }
            return {"error": "No data found"}

        @tool
        def get_delivery_performance(days: int = 50) -> Dict[str, Any]:
            """
            Get delivery performance metrics.
            Excludes CANCELLED orders from calculations.

            Args:
              days: int default: 50
            """
            print("get_delivery_performance tool is called!")
            query = f"""
            SELECT 
                COUNT(*) FILTER (WHERE t.current_order_status = 'DELIVERED') as delivered_count,
                COUNT(*) FILTER (WHERE t.current_order_status IN ('IN TRANSIT', 'OUT FOR DELIVERY')) as in_transit_count,
                COUNT(*) FILTER (WHERE t.current_order_status = 'UNDELIVERED') as undelivered_count,
                COUNT(*) FILTER (WHERE t.current_order_status = 'DELAYED') as delayed_count,
                COUNT(*) FILTER (WHERE t.sla_breach = TRUE) as sla_breach_count,
                COUNT(*) as total_orders,
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE t.current_order_status = 'DELIVERED') / NULLIF(COUNT(*), 0), 
                    2
                ) as delivery_rate,
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE t.sla_breach = TRUE) / NULLIF(COUNT(*), 0), 
                    2
                ) as sla_breach_rate
            FROM orders o
            INNER JOIN tracking_orders t ON o.order_id = t.order_id
            WHERE o.user_id = '{user_id}'
                AND t.current_order_status != 'CANCELLED'
                AND t.ordered_date IS NOT NULL
                AND t.ordered_date >= CURRENT_DATE - INTERVAL '{days} days'
            """
            preview, preview_df, full_df = self._execute_query(query, user_id)
            
            if preview.startswith("Error:"):
                return {"error": preview}
            
            if preview_df is not None and len(preview_df) > 0:
                row = preview_df.iloc[0]
                delivery_rate = row['delivery_rate']
                sla_breach_rate = row['sla_breach_rate']
                if pd.isna(delivery_rate) or delivery_rate is None:
                    delivery_rate = 0.0
                if pd.isna(sla_breach_rate) or sla_breach_rate is None:
                    sla_breach_rate = 0.0
                    
                return {
                    "delivered_count": int(row['delivered_count']) if not pd.isna(row['delivered_count']) else 0,
                    "in_transit_count": int(row['in_transit_count']) if not pd.isna(row['in_transit_count']) else 0,
                    "undelivered_count": int(row['undelivered_count']) if not pd.isna(row['undelivered_count']) else 0,
                    "delayed_count": int(row['delayed_count']) if not pd.isna(row['delayed_count']) else 0,
                    "sla_breach_count": int(row['sla_breach_count']) if not pd.isna(row['sla_breach_count']) else 0,
                    "total_orders": int(row['total_orders']) if not pd.isna(row['total_orders']) else 0,
                    "delivery_rate": float(delivery_rate),
                    "sla_breach_rate": float(sla_breach_rate),
                    "period_days": days
                }
            return {"error": "No data found"}

        @tool
        def analyze_ndr_reasons(days: int = 50, limit: int = 10) -> Dict[str, Any]:
            """
            Analyze top reasons for Non-Delivery Reports (NDR).
            Only includes orders that actually have NDR records.

            Args:
              days: int default: 50
              limit: int default: 10
            """
            print("analyze_ndr_reasons tool is called!")
            query = f"""
            SELECT 
                n.current_ndr_reason,
                COUNT(*) as ndr_count,
                COUNT(DISTINCT n.track_id) as affected_shipments
            FROM orders o
            INNER JOIN tracking_orders t ON o.order_id = t.order_id
            INNER JOIN ndr_orders n ON t.track_id = n.track_id
            WHERE o.user_id = '{user_id}'
                AND t.ordered_date IS NOT NULL
                AND t.ordered_date >= CURRENT_DATE - INTERVAL '{days} days'
                AND n.current_ndr_reason IS NOT NULL
                AND n.current_ndr_reason != ''
            GROUP BY n.current_ndr_reason
            ORDER BY ndr_count DESC
            LIMIT {limit}
            """
            preview, preview_df, full_df = self._execute_query(query, user_id)
            
            if preview.startswith("Error:"):
                return {"error": preview}
            
            if preview_df is not None and len(preview_df) > 0:
                reasons = []
                for _, row in preview_df.iterrows():
                    reasons.append({
                        "reason": row['current_ndr_reason'],
                        "count": int(row['ndr_count']),
                        "affected_shipments": int(row['affected_shipments'])
                    })
                return {
                    "top_ndr_reasons": reasons,
                    "total_ndr_types": len(reasons),
                    "period_days": days
                }
            return {"error": "No NDR data found"}

        @tool
        def compare_carrier_performance(days: int = 50) -> Dict[str, Any]:
            """
            Compare performance across different carriers.
            Excludes CANCELLED orders and includes SLA breach metrics.

            Args:
              days: int default: 50
            """
            print("compare_carrier_performance tool is called!")
            query = f"""
            SELECT 
                t.carrier_name,
                COUNT(*) as total_shipments,
                COUNT(*) FILTER (WHERE t.current_order_status = 'DELIVERED') as delivered,
                COUNT(*) FILTER (
                    WHERE t.current_order_status IN (
                        'RTO', 'RTO IN TRANSIT', 'RTO DELIVERED', 
                        'RTO OFD', 'RTO INITIATED', 'RTO NDR'
                    )
                ) as rto,
                COUNT(*) FILTER (WHERE t.current_order_status = 'UNDELIVERED') as undelivered,
                COUNT(*) FILTER (WHERE t.sla_breach = TRUE) as sla_breaches,
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE t.current_order_status = 'DELIVERED') / NULLIF(COUNT(*), 0), 
                    2
                ) as delivery_rate,
                ROUND(
                    100.0 * COUNT(*) FILTER (
                        WHERE t.current_order_status IN (
                            'RTO', 'RTO IN TRANSIT', 'RTO DELIVERED', 
                            'RTO OFD', 'RTO INITIATED', 'RTO NDR'
                        )
                    ) / NULLIF(COUNT(*), 0), 
                    2
                ) as rto_rate,
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE t.sla_breach = TRUE) / NULLIF(COUNT(*), 0), 
                    2
                ) as sla_breach_rate
            FROM orders o
            INNER JOIN tracking_orders t ON o.order_id = t.order_id
            WHERE o.user_id = '{user_id}'
                AND t.current_order_status != 'CANCELLED'
                AND t.ordered_date IS NOT NULL
                AND t.ordered_date >= CURRENT_DATE - INTERVAL '{days} days'
                AND t.carrier_name IS NOT NULL
            GROUP BY t.carrier_name
            HAVING COUNT(*) > 0
            ORDER BY total_shipments DESC
            LIMIT 10
            """
            preview, preview_df, _ = self._execute_query(query, user_id)
            
            if preview.startswith("Error:"):
                return {"error": preview}
            
            if preview_df is not None and len(preview_df) > 0:
                carriers = []
                for _, row in preview_df.iterrows():
                    delivery_rate = row['delivery_rate']
                    rto_rate = row['rto_rate']
                    sla_breach_rate = row['sla_breach_rate']
                    if pd.isna(delivery_rate): delivery_rate = 0.0
                    if pd.isna(rto_rate): rto_rate = 0.0
                    if pd.isna(sla_breach_rate): sla_breach_rate = 0.0
                    
                    carriers.append({
                        "carrier_name": row['carrier_name'],
                        "total_shipments": int(row['total_shipments']),
                        "delivered": int(row['delivered']),
                        "rto": int(row['rto']),
                        "undelivered": int(row['undelivered']),
                        "sla_breaches": int(row['sla_breaches']),
                        "delivery_rate": float(delivery_rate),
                        "rto_rate": float(rto_rate),
                        "sla_breach_rate": float(sla_breach_rate)
                    })
                return {
                    "carriers": carriers,
                    "period_days": days
                }
            return {"error": "No carrier data found"}
        
        @tool
        def analyze_order_types_and_modes(days: int = 50) -> Dict[str, Any]:
            """
            Analyze order distribution by payment type (COD vs PREPAID) and shipping mode (SURFACE vs EXPRESS).
            Uses order_details table for accurate payment and shipping information.

            Args:
               days: int default: 50
            """
            print("analyze_order_types_and_modes tool is called!")
            query = f"""
            SELECT 
                od.order_type,
                od.order_mode,
                COUNT(*) as order_count,
                COUNT(*) FILTER (WHERE t.current_order_status = 'DELIVERED') as delivered_count,
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE t.current_order_status = 'DELIVERED') / NULLIF(COUNT(*), 0), 
                    2
                ) as delivery_rate
            FROM orders o
            INNER JOIN tracking_orders t ON o.order_id = t.order_id
            INNER JOIN order_details od ON o.order_id = od.order_id
            WHERE o.user_id = '{user_id}'
                AND t.current_order_status != 'CANCELLED'
                AND t.ordered_date IS NOT NULL
                AND t.ordered_date >= CURRENT_DATE - INTERVAL '{days} days'
            GROUP BY od.order_type, od.order_mode
            ORDER BY order_count DESC
            """
            preview, preview_df, _ = self._execute_query(query, user_id)
            
            if preview.startswith("Error:"):
                return {"error": preview}
            
            if preview_df is not None and len(preview_df) > 0:
                segments = []
                for _, row in preview_df.iterrows():
                    delivery_rate = row['delivery_rate']
                    if pd.isna(delivery_rate): delivery_rate = 0.0
                    
                    segments.append({
                        "order_type": row['order_type'],
                        "order_mode": row['order_mode'],
                        "order_count": int(row['order_count']),
                        "delivered_count": int(row['delivered_count']),
                        "delivery_rate": float(delivery_rate)
                    })
                return {
                    "segments": segments,
                    "period_days": days
                }
            return {"error": "No order type data found"}
        
        @tool
        def execute_custom_query(sql_query: str) -> str:
            """
            Execute a custom SQL query when no predefined KPI tool fits the user's question.
            
            CRITICAL: The SQL query MUST include a WHERE clause filtering by user_id.
            Example: WHERE o.user_id = '{user_id}' or WHERE orders.user_id = '{user_id}'
            
            Args:
                sql_query: The SQL query to execute (MUST include user_id filter for security)
            
            Returns:
                String with query results or error if user_id filter is missing
            """
            print("execute_custom_query tool is called!")
            # Validate that user_id is in the query
            if user_id not in sql_query:
                return f"Error: SQL query must include user_id filter (user_id = '{user_id}'). Please regenerate the query with proper filtering."
            
            preview, _, _ = self._execute_query(sql_query, user_id)
            return preview
            
        return [
                calculate_rto_rate,
                get_delivery_performance,
                analyze_ndr_reasons,
                compare_carrier_performance,
                analyze_order_types_and_modes,
                execute_custom_query
            ]
    
    async def _search_similar(self, state: ConversationState, query: str):
        """Search for similar questions in vector store"""
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
        """Build schema context"""
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
    
    def _build_tools_description(self, tools: List) -> str:
        """
        Build a description of available tools.
        
        Args:
            tools: List of tool objects
            
        Returns:
            Formatted string describing all available tools
        """
        descriptions = []
        
        for tool in tools:
            # Basic format: tool_name: description
            descriptions.append(f"• {tool.name}: {tool.description}")
        
        return "\n".join(descriptions)
    
    async def _handle_data_query(
        self, 
        query: str, 
        user_id: str,
        state: ConversationState
    ) -> Tuple[Optional[str], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Generate SQL and execute directly without agent framework"""
        
        schema_context = self._get_schema_context(state)
        similar_context = ""
        if state.similar_questions:
            similar_context = "\nSimilar Questions for Reference that may help you:\n"
            for i, q in enumerate(state.similar_questions[:3], 1):
                similar_context += f"{i}. Q: {q['question']}\n   SQL: {q['sql_query']}\n\n"
        
        sql_prompt = f"""You are SQL expert. Generate a syntactically correct PostgreSQL query for this user question

User Question: {query}

{schema_context}

{similar_context}

Guidelines:
- User id is {user_id}
- Always filter by orders.user_id in your generated SQL query
- Use schema documentation to understand table structures
- Reference similar queries if helpful
- Use ordered_date column to sort/filter orders by date (Don't use last_updated_date unless it's explicitly mentionned by the user)
- Use ORDER BY for meaningful ordering
- Use LIKE for status matching (e.g., 'RTO%')
- NEVER make DML statements (INSERT, UPDATE, DELETE, DROP, etc.)
- If the question asks or requires "comparison", "analysis", "breakdown", "distribution", "categorisation" use GROUP BY with aggregations
- Common aggregation patterns:
  * Counts: COUNT(*), COUNT(DISTINCT column)
  * Percentages: ROUND(100.0 * COUNT(*) FILTER (WHERE condition) / NULLIF(COUNT(*), 0), 2)
  * Sums/Averages: SUM(column), AVG(column)
- Always add LIMIT 60 to your queries

Output ONLY the SQL query, nothing else."""
        
        try:
            response = await self.query_llm.ainvoke([HumanMessage(content=sql_prompt)])
            sql_query = response.content.strip()
            
            # Clean SQL
            sql_query = re.sub(r'```sql\s*|\s*```', '', sql_query).strip()

            print(f"Generated Query: {sql_query}")
            
            # Execute
            preview, preview_df, full_df = self._execute_query(sql_query, user_id)
            
            return preview, preview_df, full_df
            
        except Exception as e:
            print(f"Data query error: {e}")
            return f"Error: Could not process query - {str(e)}", None, None
        
    async def _handle_recommendation_query(
        self, query: str, user_id: str, state: ConversationState
    ) -> Tuple[Optional[str], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Use LangGraph agent with tools to handle data queries"""
        # Create tools
        tools = self._register_kpi_tools(user_id)
        
        # Create agent
        agent = create_react_agent(self.query_llm, tools)
        
        schema_context = self._get_schema_context(state)
        
        # Dynamically build tool descriptions
        tools_description = self._build_tools_description(tools)
        
        # Agent prompt
        agent_prompt = f"""You are a data analyst assistant with access to tools for calculating KPIs and executing SQL queries.

    User Question: {query}

    {schema_context}

    **CRITICAL SECURITY RULES:**
    1. ALWAYS filter by '{user_id}' in ALL SQL queries
    2. When using execute_custom_query tool, the SQL MUST be a POSTGRES SQL and include: WHERE orders.user_id = '{user_id}'

    **AVAILABLE TOOLS**
    {tools_description}

    **Instructions:**
    1. Analyze the question and decide which tool(s) to use
    2. If there's a specific KPI tool that fits, use it
    3. If no specific tool fits, use execute_custom_query with a custom SQL query that INCLUDES user_id filter
    4. The predefined KPI tools already handle user_id filtering automatically
    5. Use AT LEAST 2 tools to gather comprehensive data if no data fallback to use execute_custom_query tool
    6. Use Interval past 50 days by setting days param to 50
    7. Return the tool results clearly

    Choose the most appropriate tool(s) and execute them to gather the necessary data to provide informed recommendations."""
        
        try:
            # Run agent
            result = await agent.ainvoke({"messages": [HumanMessage(content=agent_prompt)]})
            
            # Extract results from agent messages
            agent_response = result['messages'][-1].content
            
            # Store KPI results by looking for ToolMessage objects
            from langchain_core.messages import ToolMessage
            
            for msg in result['messages']:
                if isinstance(msg, ToolMessage):
                    tool_name = msg.name
                    tool_content = msg.content
                    
                    try:
                        if isinstance(tool_content, str):
                            state.kpi_results[tool_name] = json.loads(tool_content)
                        else:
                            state.kpi_results[tool_name] = tool_content
                    except json.JSONDecodeError:
                        state.kpi_results[tool_name] = tool_content
                                    
            return agent_response
            
        except Exception as e:
            print(f"Agent execution error: {e}")
            return f"Error: Could not process query - {str(e)}", None, None
    
    async def _decide_chart_need(self, preview_df: DataFrame, query: str) -> bool:
        """Decide if the query results need a chart visualization"""
        if preview_df is None or len(preview_df) == 0:
            return False
        
        df = preview_df
        
        # Simple heuristics first
        if len(df) == 1 and len(df.columns) <= 2:
            return False
        
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
            print(f"DECISION: {decision}")
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
- line: for time series or trends
- pie: for proportions (max 10 categories)
- Choose most appropriate chart type

Output ONLY valid JSON."""
        
        try:
            response = await self.config_llm.ainvoke([HumanMessage(content=config_prompt)])
            config_str = response.content.strip()
            
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
            
            df_plot = df.copy()
            
            if pd.api.types.is_datetime64_any_dtype(df_plot[x_col]):
                df_plot[x_col] = df_plot[x_col].dt.strftime('%Y-%m-%d')
            
            df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors='coerce')
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
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
            
            ax.set_title(config.get('title', 'Chart'), fontsize=14, fontweight='bold')
            if chart_type != 'pie':
                ax.set_xlabel(config.get('xlabel', x_col), fontsize=11)
                ax.set_ylabel(config.get('ylabel', y_col), fontsize=11)
            
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            buffer.close()
            
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            print(f"Chart creation error: {e}")
            return None
    
    async def _stream_answer(self, state: ConversationState, query: str) -> AsyncGenerator[str, None]:
        """Stream the formatted answer based on intent"""
        
        if state.query_intent == QueryIntent.GENERAL_KNOWLEDGE:
            # Simple conversational response
            system_prompt = "You are a helpful assistant. Answer the user's question directly and conversationally."
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{query}")
            ])
            
            chain = prompt | self.answer_llm
            async for chunk in chain.astream({"query": query}):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
        
        elif state.query_intent == QueryIntent.DATA_QUERY:
            # Format data results
            if state.query_result:
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

5. The input data you get is only a preview portion of the full data (The first 10 rows or less)"""
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "Original question: {query}\n\nQuery results:\n{results}")
                ])
                
                chain = prompt | self.answer_llm
                async for chunk in chain.astream({"query": query, "results": state.query_result}):
                    if hasattr(chunk, 'content') and chunk.content:
                        yield chunk.content
        
        elif state.query_intent == QueryIntent.RECOMMENDATION:
            # Provide strategic recommendations based on KPI data
            kpi_context = ""
            if state.kpi_results:
                kpi_context = "\n\nRelevant KPI Data:\n"
                for tool_name, result in state.kpi_results.items():
                    kpi_context += f"- {tool_name}: {json.dumps(result, indent=2)}\n"
                print(f"KPI CONTEXT: {kpi_context}")
            
            system_prompt = f"""You are a logistics and e-commerce expert providing strategic recommendations.

The user has asked for advice/recommendations. Use the KPI data provided to give actionable insights.

Guidelines:
1. Start with a brief analysis of their current situation based on KPIs don't execeed 5 lines
2. Provide 3-5 specific, actionable recommendations
3. Explain the reasoning behind each recommendation
4. Prioritize recommendations by impact
5. Use data to support your recommendations

{kpi_context}

Be concise but thorough. Focus on practical steps they can take."""
            
            system_prompt = system_prompt.replace("{", "{{").replace("}", "}}")
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{query}")
            ])
            
            chain = prompt | self.answer_llm
            async for chunk in chain.astream({"query": query}):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
        
        else:  # AMBIGUOUS
            # Try to clarify or provide best-effort response
            system_prompt = "You are a helpful assistant. Answer the question as best you can."
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{query}")
            ])
            
            chain = prompt | self.answer_llm
            async for chunk in chain.astream({"query": query}):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
    
    async def stream_response(
        self, 
        query: str, 
        thread_id: str = "000000000000000000000000000"
    ) -> AsyncGenerator[str, None]:
        """Main entry point: Stream response for a query"""
        
        # Get or create conversation state
        if thread_id not in self.conversations:
            self.conversations[thread_id] = ConversationState()

        user_id = thread_id.split("_")[1]
        self.current_user_id = user_id
        
        state = self.conversations[thread_id]
        state.messages.append(HumanMessage(content=query))
        
        # Step 1: Detect intent
        state.query_intent = await self._detect_query_intent(query)
        print(f"Detected intent: {state.query_intent}")
        
        # Step 2: Handle based on intent
        if state.query_intent == QueryIntent.GENERAL_KNOWLEDGE:
            # No data fetching needed, just stream answer
            async for chunk in self._stream_answer(state, query):
                yield chunk
            return
        
        elif state.query_intent == QueryIntent.DATA_QUERY:
            # Search similar questions
            await self._search_similar(state, query)
            
            # Use agent to handle data query
            result, preview_df, full_df = await self._handle_data_query(
                query, user_id, state
            )
            
            state.query_result = result
            state.full_dataframe = full_df
            
            # Check for errors
            if state.query_result and state.query_result.startswith("Error:"):
                yield f"I encountered an error: {state.query_result}\n"
                return
            
            # Stream the answer
            async for chunk in self._stream_answer(state, query):
                yield chunk

            # Generate chart if needed
            if preview_df is not None and len(preview_df) > 0:
                needs_chart = await self._decide_chart_need(preview_df, query)
                
                if needs_chart:
                    yield "\n\n📊 Generating visualization...\n"
                    
                    chart_config = await self._generate_chart_config(preview_df, query)
                    
                    if chart_config:
                        chart_image = self._create_chart(preview_df, chart_config)
                        if chart_image:
                            state.chart_data = chart_image
        
        elif state.query_intent == QueryIntent.RECOMMENDATION:
                        
            # Use agent to gather KPIs
            result = await self._handle_recommendation_query(
                query, user_id, state
            )
            
            # Store the gathered data
            if result and not result.startswith("Error:"):
                state.query_result = result
            
            # Now stream strategic recommendations
            async for chunk in self._stream_answer(state, query):
                yield chunk
        
        else:  # AMBIGUOUS
            # Try best-effort approach
            yield "I'm not entirely sure what you're asking for. Let me try to help:\n\n"
            async for chunk in self._stream_answer(state, query):
                yield chunk
    
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
    
    def get_conversation_state(self, thread_id: str = "default") -> Optional[ConversationState]:
        """Get the conversation state for debugging/analysis"""
        return self.conversations.get(thread_id)
    
    def clear_conversation(self, thread_id: str = "default"):
        """Clear conversation history"""
        if thread_id in self.conversations:
            del self.conversations[thread_id]
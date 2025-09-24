from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    Float,
    insert,
    inspect,
    text,
)

from sqlalchemy.exc import SQLAlchemyError

import re

from smolagents import tool, CodeAgent, HfApiModel

engine = create_engine(
    f"postgresql://openleaf:testpassword@localhost:63333/testdb"
)

inspector = inspect(engine)

print("Available tables:", inspector.get_table_names())

# columns_info = [(col["name"], col["type"]) for col in inspector.get_columns("tracking_orders")]

# table_description = "Columns:\n" + "\n".join([f"  - {name}: {col_type}" for name, col_type in columns_info])

# print(table_description)

# updated_description = """Allows you to perform SQL queries on the table. Beware that this tool's output is a string representation of the execution output.
# It can use the following tables:"""

# inspector = inspect(engine)
# for table in ["tracking_orders", "orders", 'order_details']:
#     columns_info = [(col["name"], col["type"]) for col in inspector.get_columns(table)]

#     table_description = f"Table '{table}':\n"

#     table_description += "Columns:\n" + "\n".join([f"  - {name}: {col_type}" for name, col_type in columns_info])
#     updated_description += "\n\n" + table_description

# print(updated_description)

def get_table_schema():
    inspector = inspect(engine)
    schema = {}
    
    for table_name in ['orders', 'tracking_orders', 'order_details']:
        # Get columns and their info
        columns = inspector.get_columns(table_name)
        # Get foreign keys
        foreign_keys = inspector.get_foreign_keys(table_name)
        # Get primary key constraints
        pk_constraint = inspector.get_pk_constraint(table_name)
        
        schema[table_name] = {
            'columns': {col['name']: {
                'type': str(col['type']),
                'nullable': col.get('nullable', True),
                'default': str(col.get('default', 'None')),
                'is_primary_key': col['name'] in pk_constraint['constrained_columns'] if pk_constraint else False
            } for col in columns},
            'foreign_keys': foreign_keys,
            'primary_key_columns': pk_constraint['constrained_columns'] if pk_constraint else []
        }
    return schema

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

@tool
def sql_engine(query: str) -> str:
    """
    Executes PostGres SQL queries on the database tables.
    Common queries you can perform:
    - Filtering: WHERE column = value
    - Ordering: ORDER BY column ASC/DESC
    - Limiting results: LIMIT n
    - Joining tables: JOIN table ON conditions
    - Aggregations: COUNT(*), SUM(column), AVG(column)
    
    Available tables and their relations:
    - orders: Contains main order information
    - tracking_orders: Has shipping and status details
    - order_details: Contains line item details
    
    Key relationships:
    - orders.order_id links to tracking_orders.order_id
    - orders.order_id links to order_details.order_id
    
    Important columns and their meanings:
    orders:
    - order_date: When order was placed
    - customer_id: Unique identifier for customer
    
    tracking_orders:
    - current_order_status: Current status (e.g., 'RTO', 'Delivered')
    - carrier: Shipping carrier name
    
    order_details:
    - quantity: Number of items
    - price: Price per item

    Example queries:
    1. Find all RTO orders:
    SELECT * FROM tracking_orders WHERE current_order_status = 'RTO'
    
    2. Get orders with their tracking status:
    SELECT o.order_id, o.order_date, t.current_order_status 
    FROM orders o 
    JOIN tracking_orders t ON o.order_id = t.order_id

    3. Get total value by status:
      SELECT od.invoice_value AS income                                                                                                                                                      
        FROM tracking_orders t                                                                                                                                                                 
        JOIN order_details od ON t.order_id = od.order_id                                                                                                                                      
        WHERE t.current_order_status = 'DELIVERED'                                                                                                                                             
        ORDER BY t.last_updated_date DESC, t.last_updated_time DESC                                                                                                                            
        LIMIT 10   
    
    Args:
        query: SQL query to execute (must be valid PostgreSQL syntax)
    Returns:
        String representation of results
    """
    if not validate_query(query):
        return "Error: Query contains unsafe operations"
    
    try:
        output = ""
        with engine.connect() as connection:
            result = connection.execute(text(query))
            columns = result.keys()
            
            output = "|" + "|".join(str(col) for col in columns) + "|\n"
            output += "|" + "|".join("-" * len(str(col)) for col in columns) + "|\n"
            
            for row in result:
                output += "|" + "|".join(str(val) for val in row) + "|\n"
        return output
    except SQLAlchemyError as e:
        return f"Error executing query: {str(e)}"

@tool
def chat(message: str) -> str:
    """
    A simple chat tool that answers plesentries and basic non data related questions.
    Do NOT use this tool for data related questions but use only for general chat.
    Example:
    If user asks Hi, Hello etc, Respond with appropriate answer like "Hi I am Openleaf Bot. I am here to answer all your logistics and order related questions"
    If the user asks SQL related question or data related question, use SQL Engine tool to answer the question.
    
    Args:
        message: The message to respond to
    Returns:
        The answer
    """

    responses = {
        "hello": "Hello! How can I help you today?",
        "hi": "Hi there! What can I do for you?",
        "how are you": "I'm doing well, thank you for asking! How can I assist you?",
        "bye": "Goodbye! Have a great day!",
        "default": "I'm here to help! Would you like to know something about the data?"
    }
    
    # Convert to lowercase and find closest match
    msg = message.lower().strip()
    return responses.get(msg, responses["default"])

sql_engine.description += f"\n\nDetailed Schema:\n{get_table_schema()}"

print(sql_engine.description)

agent = CodeAgent(tools=[chat, sql_engine], model=HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct", token= "test"),)

agent.run("Get destination locations and frequency of top 20 RTO orders and classify them into indian geographical regions.")
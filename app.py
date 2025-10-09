import streamlit as st
import requests
from typing import Generator
from datetime import datetime

API_BASE_URL = "http://localhost:5000/ai"

st.set_page_config(
    page_title="SQL Agent Chat",
    page_icon="🤖",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    st.session_state.user_id = "f17d6407-9d6b-45f5-854c-30a43b4b9615"

if "is_customer" not in st.session_state:
    st.session_state.is_customer = False

def get_thread_id() -> str:
    """Generate thread ID based on user type."""
    return f"user_{st.session_state.user_id}"

def stream_chat_response(query: str, user_id: str, is_customer: bool) -> Generator[str, None, None]:
    """Stream plain text response from the API (no SSE)."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "query": query,
                "user_id": user_id,
                "is_customer": is_customer            
                },
            stream=True,
            timeout=100
        )

        if response.status_code == 200:
            for chunk in response.iter_content(decode_unicode=True, chunk_size=1):
                if chunk:
                    yield chunk
        else:
            yield f"Error: API returned status {response.status_code}"

    except requests.exceptions.RequestException as e:
        yield f"Error connecting to API: {str(e)}"


def check_data_available() -> tuple[bool, int]:
    """Check if data is available for download."""
    try:
        thread_id = get_thread_id()
        response = requests.get(
            f"{API_BASE_URL}/chat/{thread_id}/check",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("has_data", False), data.get("row_count", 0)
        
        return False, 0
            
    except requests.exceptions.RequestException:
        return False, 0


def download_excel() -> tuple[bytes, str]:
    """Download Excel file for current thread."""
    try:
        thread_id = get_thread_id()
        
        response = requests.get(
            f"{API_BASE_URL}/chat/{thread_id}/download",
            timeout=30
        )
        
        if response.status_code == 200:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"query_results_{timestamp}.xlsx"
            return response.content, filename
        
        return None, None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Download error: {str(e)}")
        return None, None


def download_chart() -> bytes:
    """Download chart image for current thread."""
    try:
        thread_id = get_thread_id()
        
        response = requests.get(
            f"{API_BASE_URL}/chat/{thread_id}/download/chart",
            timeout=30
        )
        
        if response.status_code == 200:
            return response.content
        
        return None
            
    except requests.exceptions.RequestException:
        return None


def add_question_pair(question: str, sql_query: str, db_name: str) -> dict:
    """Add a question-SQL pair to the vector store."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/vectorstore/add-pair",
            params={
                "question": question,
                "sql_query": sql_query,
                "db_name": db_name
            },
            timeout=30
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}

with st.sidebar:
    st.title("⚙️ Settings")

    # User ID input
    st.session_state.user_id = st.text_input(
        "User/Order ID",
        value=st.session_state.user_id,
        help="Enter the user ID or order ID"
    )

    # Toggle for customer/user mode
    st.session_state.is_customer = st.checkbox(
        "Customer Mode (Order ID)",
        value=st.session_state.is_customer,
        help="Check if the ID above is an order ID, uncheck for user ID"
    )

    st.divider()

    # Add Question-SQL Pair Section
    st.subheader("📝 Add Training Data")

    with st.expander("Add Question-SQL Pair"):
        with st.form("add_pair_form"):
            st.write("Add a new question-SQL pair to improve the agent")

            example_question = st.text_area(
                "Example Question",
                placeholder="e.g., Show me all my orders",
                help="Natural language question"
            )

            example_sql = st.text_area(
                "SQL Query",
                placeholder="e.g., SELECT * FROM orders WHERE user_id = '...'",
                help="The corresponding SQL query"
            )

            db_name = st.text_input(
                "Database Name",
                value="shypmax_tracking",
                help="Name of the database this query applies to"
            )

            submit_pair = st.form_submit_button("Add Pair")

            if submit_pair:
                if example_question and example_sql and db_name:
                    with st.spinner("Adding to vector store..."):
                        result = add_question_pair(example_question, example_sql, db_name)
                        if result.get("status") == "success":
                            st.success(f"✅ Added successfully! ID: {result.get('document_id', 'N/A')}")
                        else:
                            st.error(f"❌ Failed: {result.get('message', 'Unknown error')}")
                else:
                    st.warning("⚠️ Please fill in all fields")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


st.title("AI SQL Agent")
st.caption("Ask questions about your data in natural language")

# Show current context
context_type = "Order ID" if st.session_state.is_customer else "User ID"
st.info(f"**Current Context:** {context_type} = `{st.session_state.user_id}`")

# Show chat history with stored charts and download buttons
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show chart if this message has chart data
        if message["role"] == "assistant" and message.get("chart_data"):
            st.image(message["chart_data"])
        
        # Show download button if this message has Excel data
        if message["role"] == "assistant" and message.get("has_data"):
            st.write("For inspecting the full data, please download the excel file below")
            
            excel_data = message.get("excel_data")
            filename = message.get("filename")
            
            if excel_data and filename:
                st.download_button(
                    label=f"📥 Download Excel",
                    data=excel_data,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_{idx}",
                )

# Chat input
if prompt := st.chat_input("Ask a question about your data..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for chunk in stream_chat_response(
            prompt,
            st.session_state.user_id,
            st.session_state.is_customer
        ):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

        # Prepare assistant message
        assistant_message = {
            "role": "assistant",
            "content": full_response,
            "has_data": False,
            "chart_data": None
        }
        
        # Download and store chart if available
        chart_data = download_chart()
        if chart_data:
            assistant_message["chart_data"] = chart_data
            st.image(chart_data)
        
        # Check if Excel data is available
        has_data, row_count = check_data_available()
        
        if has_data and row_count > 0:
            excel_data, filename = download_excel()
            
            if excel_data and filename:
                # Store excel data in the message
                assistant_message["has_data"] = True
                assistant_message["excel_data"] = excel_data
                assistant_message["filename"] = filename
                assistant_message["row_count"] = row_count
                
                # Show download button immediately
                st.write("For inspecting the full data, please download the excel file below")
                st.download_button(
                    label=f"📥 Download Excel",
                    data=excel_data,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_latest",
                )

        # Save assistant message with all data
        st.session_state.messages.append(assistant_message)


st.divider()
st.caption("💡 Tip: Use the sidebar to add example questions and improve the agent's responses")

with st.expander("📋 Example Queries"):
    st.markdown("""
    **Try asking:**
    - Show me all my orders
    - How many orders do I have?
    - What's the status of my latest order?
    - Show orders that are in transit
    - List all RTO orders
    - Count orders by status
    """)
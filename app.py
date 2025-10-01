import streamlit as st
import requests
from typing import Generator

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

def stream_chat_response(query: str, user_id: str, is_customer: bool) -> Generator[str, None, None]:
    """Stream plain text response from the API (no SSE)."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "query": query,
                "user_id": user_id,
                "is_customer": is_customer,
                "stream": True
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


st.title("🤖 SQL Agent Chat")
st.caption("Ask questions about your data in natural language")

# Show current context
context_type = "Order ID" if st.session_state.is_customer else "User ID"
st.info(f"**Current Context:** {context_type} = `{st.session_state.user_id}`")

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_response})


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

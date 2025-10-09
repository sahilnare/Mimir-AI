from contextlib import asynccontextmanager
import base64
import asyncio
import json
import logging
from typing import AsyncGenerator, Optional
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, HTTPException, Request
from io import BytesIO
import pandas as pd
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import uvicorn

from configs.config import load_config
from agents.v3.core import SQLAgent
from agents.v3.vector_store import VectorStoreManager

# Configure Logging
LOG_FILE = "server.log"
logger = logging.getLogger("FastAPI-Logger")
logger.setLevel(logging.DEBUG)

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])

# Global variables for services
sql_agent: Optional[SQLAgent] = None
vectorstore_manager: Optional[VectorStoreManager] = None
app_config = None

class QueryRequest(BaseModel):
    query: str
    user_id: str = "f17d6407-9d6b-45f5-854c-30a43b4b9615"
    is_customer: bool = False

class QueryResponse(BaseModel):
    result: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global sql_agent, vectorstore_manager, app_config
    
    try:
        # Load configuration
        app_config = load_config()
        db_config = app_config.get("db")
        
        # Initialize VectorStoreManager
        logger.info("Initializing VectorStoreManager...")
        vectorstore_manager = VectorStoreManager(app_config)
        await vectorstore_manager.initialize()
        logger.info("VectorStoreManager initialized successfully")
        
        # Create database URI
        db_uri = f"postgresql://{db_config.get('username')}:{db_config.get('password')}@{db_config.get('host')}:{db_config.get('port')}/{db_config.get('dbname')}"
        
        # Initialize SQLAgent
        logger.info("Initializing SQLAgent...")
        sql_agent = SQLAgent(
            db_uri=db_uri,
            config=app_config,
            vectorstore=vectorstore_manager
        )
        logger.info("SQLAgent initialized successfully")
        
        logger.info("Application startup complete")
        yield
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Starting application shutdown...")
        if vectorstore_manager:
            await vectorstore_manager.close()
        logger.info("Application shutdown complete")

app = FastAPI(
    title="SQL Query Agent API", 
    description="API to query data using SQL Agent with vector similarity search",
    docs_url="/ai/docs",
    redoc_url="/ai/redoc",
    openapi_url="/ai/openapi.json",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Timeout"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests"""
    forwarded_host = request.headers.get("X-Forwarded-Host", "N/A")
    forwarded_proto = request.headers.get("X-Forwarded-Proto", "N/A")
    forwarded_for = request.headers.get("X-Forwarded-For", "N/A")
    
    logger.info(f"Request: {request.method} {request.url}")
    logger.info(f"Headers - Host: {forwarded_host}, Proto: {forwarded_proto}, For: {forwarded_for}")
    
    response = await call_next(request)
    return response

async def stream_response(query: str, thread_id: str) -> AsyncGenerator[str, None]:
    """Stream the agent response in real-time from LLM"""
    try:
        async for chunk in sql_agent.stream_response(query, thread_id=thread_id):
            if chunk:
                yield chunk
    except Exception as e:
        logger.error(f"Error in stream_response: {e}")
        yield f"[Error] {str(e)}"


@app.post("/ai/chat", tags=["AI"])
@limiter.limit("5/minute")
async def chat(request: Request, query_request: QueryRequest):
    """
    Execute a query against the database and return AI-generated response.
    Supports streaming text and chart visualization.
    """
    if not sql_agent:
        raise HTTPException(status_code=500, detail="SQL Agent not initialized")

    if not query_request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Handle greetings for very short queries
    if len(query_request.query) <= 3:
        greetings = ["hello", "hi", "hey", "greetings", "wassup", "what's up", "howdy"]
        if any(greeting in query_request.query.lower() for greeting in greetings):
            greeting_response = "Hello! How can I help you with your data queries today?"
            async def stream_greeting():
                for word in greeting_response.split():
                    yield word + " "
                    await asyncio.sleep(0.05)
            return StreamingResponse(stream_greeting(), media_type="text/plain")
    
    # Generate thread_id for this user
    thread_id = f"user_{query_request.user_id}"
    return StreamingResponse(
            stream_response(query_request.query, thread_id),
            media_type="text/plain"
        )
 
@app.get("/ai/chat/{thread_id}/check/excel", tags=["AI"])
async def check_query_results(thread_id: str):
    """
    Check if query results are available for download.
    Returns: {"has_data": bool, "row_count": int}
    """
    if not sql_agent:
        raise HTTPException(status_code=500, detail="SQL Agent not initialized")
    
    df = sql_agent.get_last_dataframe(thread_id)
    has_data = df is not None and not df.empty
    row_count = len(df) if has_data else 0
    
    return {
        "has_data": has_data,
        "row_count": row_count
    }

@app.get("/ai/chat/{thread_id}/download/excel", tags=["AI"])
async def download_query_results(thread_id: str):
    """
    Download Excel file for the last query results in this conversation thread.
    """
    if not sql_agent:
        raise HTTPException(status_code=500, detail="SQL Agent not initialized")
    
    # Get the DataFrame from the conversation
    df = sql_agent.get_last_dataframe(thread_id)
    
    if df is None or df.empty:
        raise HTTPException(
            status_code=404, 
            detail="No query results available for this conversation"
        )
    
    # Convert DataFrame to Excel
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Query Results')
        
        output.seek(0)
        
        # Generate filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_results_{thread_id}_{timestamp}.xlsx"
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating Excel: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate Excel: {str(e)}")

@app.get("/ai/chat/{thread_id}/check/chart", tags=["AI"])
async def check_chart_data(thread_id: str):
    """
    Check if query results are available for download.
    Returns: {"has_data": bool, "row_count": int}
    """
    if not sql_agent:
        raise HTTPException(status_code=500, detail="SQL Agent not initialized")
    
    data = sql_agent.get_last_chart_data(thread_id)
    has_data = data is not None
  
    return {
        "has_data": has_data
    }

@app.get("/ai/chat/{thread_id}/download/chart", tags=["AI"])
async def download_chart(thread_id: str):
    """
    Return the chart as a PNG image for the given thread_id.
    """
    if not sql_agent:
        raise HTTPException(status_code=500, detail="SQL Agent not initialized")
    
    chart_base64 = sql_agent.get_last_chart_data(thread_id)
    if not chart_base64:
        raise HTTPException(status_code=404, detail="No chart found for this thread")
    
    # Remove the data:image/png;base64, prefix if present
    if chart_base64.startswith("data:image/png;base64,"):
        chart_base64 = chart_base64.split(",", 1)[1]
    
    chart_bytes = base64.b64decode(chart_base64)
    
    return Response(content=chart_bytes, media_type="image/png")

@app.delete("/ai/chat/{thread_id}", tags=["AI"])
async def clear_conversation(thread_id: str):
    """
    Clear conversation history and cached results for a thread.
    """
    if not sql_agent:
        raise HTTPException(status_code=500, detail="SQL Agent not initialized")
    
    try:
        sql_agent.clear_conversation(thread_id)
        return {"status": "success", "message": f"Conversation {thread_id} cleared"}
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/ai/health")
@limiter.exempt
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "sql_agent": sql_agent is not None,
        "vectorstore": vectorstore_manager is not None
    }
    return status

@app.get("/ai/")
@limiter.exempt
async def root():
    """Root endpoint"""
    return {"status": "SQL Agent API is running"}

# Utility endpoints for managing vector store
@app.post("/ai/vectorstore/add-pair", tags=["VectorStore"])
@limiter.limit("10/minute") 
async def add_question_pair(request: Request, question: str, sql_query: str, db_name: str):
    """Add a single question-SQL pair to vector store"""
    try:
        if not vectorstore_manager:
            raise HTTPException(status_code=500, detail="VectorStore not initialized")
        
        doc_id = await vectorstore_manager.add_question_query_pair(
            question=question,
            sql_query=sql_query, 
            db_name=db_name
        )
        
        if doc_id:
            return {"status": "success", "document_id": doc_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to add question pair")
            
    except Exception as e:
        logger.error(f"Error adding question pair: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/vectorstore/add-from-json", tags=["VectorStore"])
@limiter.limit("2/hour")
async def add_from_json(request: Request, json_file_path: str):
    """Add question-SQL pairs from JSON file"""
    try:
        if not vectorstore_manager:
            raise HTTPException(status_code=500, detail="VectorStore not initialized")
        
        results = await vectorstore_manager.add_questions_from_json(json_file_path)
        return results
        
    except Exception as e:
        logger.error(f"Error adding from JSON: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
from contextlib import asynccontextmanager
import os
import asyncio
import json
import logging
from typing import AsyncGenerator, Optional
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
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
    stream: bool = False

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

async def stream_response(query: str, config: dict) -> AsyncGenerator[str, None]:
    """Stream the agent response"""
    try:
        for event in sql_agent.stream_response(query, config):
            async for event in sql_agent.stream_response(query, config):
                # Extract meaningful content from the event
                if event.get("messages"):
                    last_message = event["messages"][-1]
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            if tool_call["name"] == "SubmitFinalAnswer":
                                args = tool_call["args"]
                                if isinstance(args, str):
                                    args = json.loads(args)
                                yield f"data: {args['final_answer']}\n\n"
                                return
                    elif hasattr(last_message, 'content') and last_message.content:
                        yield f"data: {last_message.content}\n\n"
                await asyncio.sleep(0.1)
    except Exception as e:
        yield f"data: Error: {str(e)}\n\n"
    finally:
        yield "data: [DONE]\n\n"

@app.exception_handler(RateLimitExceeded)
async def rate_limited_handler(request: Request, exc: RateLimitExceeded):
    retry_after = getattr(exc, 'retry_after', 60)
    return JSONResponse(
        content={
            "detail": "Rate Limit Exceeded. Please try again later.",
            "retry_after_seconds": retry_after
        }, 
        status_code=429, 
        headers={"Retry-After": str(retry_after)}
    )

@app.post("/ai/chat", tags=["AI"])
@limiter.limit("5/minute")
async def chat(request: Request, query_request: QueryRequest):
    """
    Execute a query against the database and return AI-generated response
    """
    try:
        if not sql_agent:
            raise HTTPException(status_code=500, detail="SQL Agent not initialized")
        
        # Handle greetings for very short queries
        if len(query_request.query) <= 3:
            greetings = ["hello", "hi", "hey", "greetings", "wassup", "what's up", "howdy"]
            if any(greeting in query_request.query.lower() for greeting in greetings):
                return QueryResponse(result="Hello! How can I help you with your data queries today?")
        
        if not query_request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Build context for the query
        context_prefix = f'user id {query_request.user_id}'
        if query_request.is_customer:
            context_prefix = f'order id {query_request.user_id}'
        
        full_query = f"For this {context_prefix} answer the following question: {query_request.query}"
        
        # Generate unique config for this request
        config = {"configurable": {"thread_id": f"thread_{query_request.user_id}_{id(request)}"}}
        
        if query_request.stream:
            return StreamingResponse(
                stream_response(full_query, config), 
                media_type="text/event-stream"
            )
        else:
            result = await sql_agent.get_response(full_query, config)
            return QueryResponse(result=result)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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
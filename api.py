from copy import deepcopy
import os
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Generator, Optional
from langchain.schema import HumanMessage
from langchain_community.utilities import SQLDatabase
import asyncio
from contextlib import asynccontextmanager
import json

import uvicorn
from configs.config import load_config
from langgraph.v2.ai import text2sql_with_analysis_agent
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import os
import boto3
import asyncio
from logging.handlers import RotatingFileHandler
from botocore.exceptions import NoCredentialsError
from datetime import datetime
from botocore.config import Config
from fastapi.middleware.cors import CORSMiddleware

# os.environ["APP_ENV"] = "sandbox"
# Code to upload logs to S3
BUCKET_NAME = "openleaf-stg-alb"
LOG_FILE = "server.log"
S3_LOG_PATH = f"logs/{datetime.utcnow().strftime('%Y-%m-%d')}/server.log"

configs = load_config()
aws_config = configs.get("aws")
s3_client = None
aws_access_key = aws_config["AWS_ACCESS_KEY_ID"].strip()
aws_secret_key = aws_config["AWS_SECRET_ACCESS_KEY"].strip()

s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_config["AWS_REGION"],
    config=Config(signature_version="s3v4")
)

# Configure Logging
logger = logging.getLogger("FastAPI-Logger")
logger.setLevel(logging.DEBUG)

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])

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

async def start_log_uploader():
    asyncio.create_task(upload_logs_to_s3())

# class PrefixMiddleware(BaseHTTPMiddleware):
#     async def dispatch(self, request: Request, call_next):
#         prefix = request.headers.get("X-Forwarded-Prefix", "").rstrip("/")
#         print(f"Prefix: {prefix}")
#         request.scope["root_path"] = prefix
#         return await call_next(request)
    
class QueryRequest(BaseModel):
    query: str
    user_id: str = "1ecf71e1-1748-425b-9071-56e925a391a3"
    is_customer: bool = False
    stream: bool = False

class QueryResponse(BaseModel):
    result: str

app = FastAPI(
    title="SQL Query Agent API", 
    description="API to query answer data Based questions using SQL Agent",
    docs_url="/ai/docs",
    redoc_url="/ai/redoc",
    openapi_url="/ai/openapi.json",
    # root_path="/ai"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # List of allowed origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # List of allowed methods
    allow_headers=["Authorization", "Content-Type", "Timeout"],  # List of allowed headers
)

app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)
# app.add_middleware(PrefixMiddleware)

db = None

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Log request details, including ALB headers
    forwarded_host = request.headers.get("X-Forwarded-Host", "N/A")
    forwarded_proto = request.headers.get("X-Forwarded-Proto", "N/A")
    forwarded_port = request.headers.get("X-Forwarded-Port", "N/A")
    forwarded_for = request.headers.get("X-Forwarded-For", "N/A")
    
    logger.info(f"Incoming Request: {request.method} {request.url} | Headers: {dict(request.headers)}")
    logger.info(f"ALB Routing Info: Host: {forwarded_host}, Proto: {forwarded_proto}, Port: {forwarded_port}, For: {forwarded_for}")

    logger.warning(f"Incoming Request: {request.method} {request.url} | Headers: {dict(request.headers)}")
    logger.warning(f"ALB Routing Info: Host: {forwarded_host}, Proto: {forwarded_proto}, Port: {forwarded_port}, For: {forwarded_for}")
    
    response = await call_next(request)
    return response

# @app.middleware("http")
# async def strip_ai_prefix(request: Request, call_next):
#     if request.url.path.startswith("/ai"):
#         # Remove the '/ai' prefix from the URL path
#         request.scope["path"] = request.url.path[len("/ai"):]
#         logger.warning(f"Incoming Request: {request.method} {request.url} | Headers: {dict(request.headers)}")

#     return await call_next(request)

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

# router = APIRouter(
#     prefix="/ai",
#     tags=["AI"]
# )

# def add_route_prefix(route: str) -> str:
#     return f"/ai{route}"


def create_fresh_agent(db):
    """Create a new instance of the agent for each request"""
    workflow = text2sql_with_analysis_agent(db)
    return workflow.compile()

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


@app.exception_handler(RateLimitExceeded)
async def rate_limited_handler(request: Request, exc: RateLimitExceeded):
    retry_after = int(exc.description.split(" ")[-1])
    response_body = {
        "detail": "Rate Limit Exceeded. Please try again later.",
        "retry_after_seconds": retry_after
    }
    return JSONResponse(content=response_body, status_code=429, headers={"Retry-After": str(retry_after)})

@app.post("/ai/chat", tags=["AI"])
@limiter.limit("5/minute")
async def chat(request: Request, query_request: QueryRequest):
    """
    Executes a query against DB based on prompt and then use the result to answer the questions.
    Parameters:
    - query: str: The query to execute
    - stream: bool: Stream the response based on Boolean Flag
    """
    try:
        if len(query_request.query) <= 3:
            greetings = ["hello", "hi", "hey", "greetings", "wassup", "what's up", "howdy", "good morning", "good afternoon", "good evening", "good night"]
            for keyword in greetings:
                if keyword in query_request.query.lower():
                    return QueryResponse(result="Hello! How can I help you today?")
        
        if query_request.query != "":
            async with get_db() as db:
                agent = create_fresh_agent(db)

                agent_prompt = f'user id {query_request.user_id}'

                if query_request.is_customer:
                    agent_prompt = f'order id {query_request.user_id}'

                agent_input_query = {
                    "messages": [HumanMessage(content=f"For this {agent_prompt} answer the following question. {query_request.query}")]
                }

                # For future purposes wrt Persistent Memory
                # config = {"configurable": {"thread_id": f"thread_{request.user_id}_{asyncio.current_task().get_name()}"}}

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
                    # print(response)
                    result = process_agent_response(response)
                    return QueryResponse(result=result)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai/health")
@limiter.exempt
async def ecs_health():
    return {"status": "Service Healthy!"}

@app.get("/ai/")
@limiter.exempt
async def ecs_status_check():
    return {"status": "Ok"}

@app.on_event("startup")
async def startup_event():
    """Startup event."""
    logger.info("Application startup complete")
    logger.info(f"Available routes:")
    for route in app.routes:
        logger.info(f"{route.methods} {route.path}")
    await start_log_uploader()
    pass

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event."""
    global db
    if db:
        db = None

# app.include_router(router)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=5000)

                                         
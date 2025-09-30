from typing import List, Dict, Any, Optional, Tuple
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
import chromadb
from chromadb.config import Settings
import asyncio
import re 
import logging
import uuid
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages vector store operations and knowledge graph functionality"""
    
    def __init__(self, config):
        self.config = config
        self.embeddings = None
        self.vectorstore = None
        self.chroma_client = None
        self.collection = None
        
    async def initialize(self):
        """Initialize the vector store and embeddings"""
        try:
            # Initialize embeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=self.config.get("GOOGLE_API_KEY"),
                model="gemini-embedding-001",
                task_type="retrieval_query" 
            )
            
            # Ensure the database directory exists
            db_path = Path(self.config.get('chroma_db_path'))
            db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            collection_name = self.config.get("collection_name")
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
                logger.info(f"Loaded existing collection: {collection_name}")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Question -> SQL pairs"}
                )
                logger.info(f"Created new collection: {collection_name}")
            
            # Initialize Langchain ChromaDB wrapper
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
            
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add multiple documents to the vector store"""
        try:
            # Generate unique IDs for documents
            doc_ids = [str(uuid.uuid4()) for _ in documents]
            await self.vectorstore.aadd_documents(documents=documents, ids=doc_ids)
            
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return []
    
    async def add_question_query_pair(
        self,
        question: str,
        sql_query: str,
        db_name: str,
        table_names: Optional[List[str]] = None,
        tag: Optional[List[str]] = None
    ) -> Optional[str]:
        """Add a single question-query pair to the vector store"""
        try:
            # Create metadata
            metadata = {
                'sql_query': sql_query,
                'db_name': db_name,
                'table_names': ",".join(table_names) if table_names else '',
                'tag': tag or '',
                'created_at': datetime.now().isoformat()
            }
            
            # Create document
            document = Document(
                page_content=question,
                metadata=metadata
            )
            
            # Add to vector store
            doc_ids = await self.add_documents([document])
            
            if doc_ids:
                logger.info(f"Added question-query pair with ID: {doc_ids[0]}")
                return doc_ids[0]
            else:
                logger.error("Failed to add question-query pair")
                return None
                
        except Exception as e:
            logger.error(f"Error adding question-query pair: {e}")
            return None
    
    async def add_questions_from_json(
        self,
        json_file_path: str,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Add question-query pairs from JSON file in batches"""
        try:
            import json
            
            # Read JSON file
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate JSON structure
            if not isinstance(data, list):
                raise ValueError("JSON file should contain a list of question-query pairs")
            
            results = {
                'total_processed': 0,
                'successful': 0,
                'failed': 0,
                'errors': []
            }
            
            # Process in batches
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                documents = []
                
                for item in batch:
                    try:
                        # Validate required fields
                        if 'question' not in item or 'sql_query' not in item or 'db_name' not in item:
                            results['failed'] += 1
                            results['errors'].append(f"Missing required fields in item: {item}")
                            continue
                        
                        # Create metadata
                        metadata = {
                            'sql_query': item['sql_query'],
                            'db_name': item['db_name'],
                            'table_names': ",".join(item.get('table_names', [])) if item.get('table_names') else '',
                            'tag': item.get('tag', ''),
                            'created_at': datetime.now().isoformat(),
                        }
                        
                        # Create document
                        document = Document(
                            page_content=item['question'],
                            metadata=metadata
                        )
                        
                        documents.append(document)
                        results['total_processed'] += 1
                        
                    except Exception as e:
                        results['failed'] += 1
                        results['errors'].append(f"Error processing item {item}: {str(e)}")
                
                # Add batch to vector store
                if documents:
                    doc_ids = await self.add_documents(documents)
                    results['successful'] += len(doc_ids)
                    logger.info(f"Added batch of {len(doc_ids)} documents")
            
            logger.info(f"Finished processing JSON file. Results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error adding questions from JSON: {e}")
            return {
                'total_processed': 0,
                'successful': 0,
                'failed': 0,
                'errors': [str(e)]
            }
    
    async def search_similar_questions(
        self, 
        query: str, 
        filter: Dict,
        k: int = 5,
    ) -> List[dict]:
        """Search for similar questions in vector store with relevance score"""
        if not self.vectorstore:
            return []
        
        try:
            # Add where filter for db_name and k parameter
            search_kwargs = {
                "k": k,
                "filter": filter
            }
            
            # Use LangChain's similarity_search_with_relevance_scores
            results = await asyncio.to_thread(
                self.vectorstore.similarity_search_with_relevance_scores,
                query,
                **search_kwargs
            )
            
            # Filter and format results
            similar_questions = []
            for doc, score in results:
                if score > 0.7:
                    similar_questions.append({
                        'question': doc.page_content,
                        'sql_query': doc.metadata.get('sql_query', ''),
                        'table_names': doc.metadata.get('table_names', '').split(",") if doc.metadata.get('table_names') else []
                    })
            
            return similar_questions
            
        except Exception as e:
            logger.error(f"Error in similarity search with scores: {e}")
            return []
    
    async def close(self):
        """Close the vector store connection"""
        try:
            self.vectorstore = None
            self.chroma_client = None
            self.collection = None
            logger.info("Vector store closed successfully")
        except Exception as e:
            logger.error(f"Error closing vector store: {e}")
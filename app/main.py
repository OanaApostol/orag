"""FastAPI application for RAG-powered Help Center chatbot."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

from app.config import get_settings
from app.data_processor import DataProcessor
from app.vector_store import VectorStore
from app.rag_engine import RAGEngine


# Global instances
vector_store = None
rag_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Initializes the vector store and indexes documents on startup.
    """
    global vector_store, rag_engine
    
    print("🚀 Starting RAG application...")
    
    # Initialize components
    vector_store = VectorStore()
    vector_store.initialize_index(force_recreate=False)
    
    # Check if index needs to be populated
    stats = vector_store.get_index_stats()
    if stats.get('total_vectors', 0) == 0:
        print("📚 Index is empty. Loading and indexing documents...")
        
        # Process and index documents
        processor = DataProcessor()
        documents = processor.load_articles()
        
        print(f"📊 Document stats: {processor.get_stats(documents)}")
        
        vector_store.index_documents(documents)
        print("✅ Documents indexed successfully")
    else:
        print(f"✅ Using existing index with {stats['total_vectors']} vectors")
    
    # Initialize RAG engine
    rag_engine = RAGEngine(vector_store)
    
    print("✨ RAG application ready!")
    
    yield
    
    print("👋 Shutting down RAG application...")


# Initialize FastAPI app
app = FastAPI(
    title="Typeform Help Center RAG Chatbot",
    description="AI-powered chatbot for Typeform Help Center using RAG",
    version="0.1.0",
    lifespan=lifespan
)


# Request/Response models
class QuestionRequest(BaseModel):
    """Request model for asking a question."""
    question: str = Field(..., description="User's question", min_length=1)
    top_k: Optional[int] = Field(None, description="Number of documents to retrieve", ge=1, le=10)


class Source(BaseModel):
    """Source information for a response."""
    title: str
    url: str
    relevance_score: float


class QuestionResponse(BaseModel):
    """Response model for question answers."""
    answer: str
    sources: List[Source]
    retrieved_chunks: int
    confidence: float
    is_fallback: Optional[bool] = False
    quality_score: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
    index_stats: Dict


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Typeform Help Center RAG Chatbot API",
        "version": "0.1.0",
        "endpoints": {
            "POST /ask_question": "Ask a question about Typeform Help Center",
            "GET /health": "Check API health and index status",
            "GET /docs": "API documentation (Swagger UI)",
            "GET /chat": "Interactive chat interface"
        }
    }


@app.get("/chat", tags=["Chat"])
async def chat_interface():
    """Serve the interactive chat interface."""
    return FileResponse("chat.html")


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint to verify the API and vector store are functioning.
    """
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        stats = vector_store.get_index_stats()
        
        return {
            "status": "healthy",
            "message": "RAG application is running",
            "index_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.post("/ask_question", response_model=QuestionResponse, tags=["Chat"])
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get an AI-generated response based on Help Center articles.
    
    This endpoint:
    1. Retrieves relevant Help Center content using semantic search
    2. Generates a conversational response using an LLM
    3. Returns the answer with source citations
    
    Args:
        request: QuestionRequest with the user's question
    
    Returns:
        QuestionResponse with answer, sources, and metadata
    """
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")
        
        # Generate response
        result = rag_engine.generate_response(
            query=request.question,
            top_k=request.top_k
        )
        
        return QuestionResponse(
            answer=result['answer'],
            sources=[Source(**source) for source in result['sources']],
            retrieved_chunks=result['retrieved_chunks'],
            confidence=result['confidence'],
            is_fallback=result.get('is_fallback', False),
            quality_score=result.get('quality_score', None)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.get("/stats", tags=["General"])
async def get_stats():
    """
    Get statistics about the indexed documents and vector store.
    """
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        index_stats = vector_store.get_index_stats()
        
        return {
            "index_stats": index_stats,
            "configuration": {
                "embedding_model": get_settings().embedding_model,
                "llm_model": get_settings().llm_model,
                "chunk_size": get_settings().chunk_size,
                "chunk_overlap": get_settings().chunk_overlap,
                "top_k_results": get_settings().top_k_results
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


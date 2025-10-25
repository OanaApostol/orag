"""FastAPI application for RAG-powered Help Center chatbot."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
from functools import wraps

from app.config import get_settings
from app.data_processor import DataProcessor
from app.vector_store import VectorStore
from app.rag_engine import RAGEngine


# Application constants
APP_VERSION = "0.1.0"
APP_TITLE = "Typeform Help Center RAG Chatbot"
APP_DESCRIPTION = "AI-powered chatbot for Typeform Help Center using RAG"

# Global instances
vector_store = None
rag_engine = None


def require_components(*components):
    """Decorator to check if required components are initialized."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for component_name in components:
                component = globals().get(component_name)
                if not component:
                    raise HTTPException(
                        status_code=503,
                        detail=f"{component_name} not initialized"
                    )
            return await func(*args, **kwargs)
        return wrapper
    return decorator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global vector_store, rag_engine
    
    print("🚀 Starting RAG application...")
    
    # Initialize vector store
    vector_store = VectorStore()
    vector_store.initialize_index(force_recreate=True)
    
    # Check and populate index if needed
    if not vector_store.get_index_stats().get('total_vectors'):
        print("📚 Index is empty. Loading and indexing documents...")
        processor = DataProcessor()
        documents = processor.load_articles()
        print(f"📊 Document stats: {processor.get_stats(documents)}")
        vector_store.index_documents(documents)
        print("✅ Documents indexed successfully")
    else:
        stats = vector_store.get_index_stats()
        print(f"✅ Using existing index with {stats['total_vectors']} vectors")
    
    # Initialize RAG engine
    rag_engine = RAGEngine(vector_store)
    print("✨ RAG application ready!")
    
    yield
    print("👋 Shutting down RAG application...")


# Initialize FastAPI app
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan
)


# Request/Response models
class QuestionRequest(BaseModel):
    question: str = Field(..., description="User's question", min_length=1)
    top_k: Optional[int] = Field(None, description="Number of documents to retrieve", ge=1, le=10)


class Source(BaseModel):
    title: str
    url: str
    relevance_score: float


class QuestionResponse(BaseModel):
    answer: str
    sources: List[Source]
    retrieved_chunks: int
    confidence: float
    is_fallback: Optional[bool] = False
    quality_score: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    message: str
    index_stats: Dict


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": f"{APP_TITLE} API",
        "version": APP_VERSION,
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
@require_components("vector_store")
async def health_check():
    """Health check endpoint to verify the API and vector store are functioning."""
    try:
        return {
            "status": "healthy",
            "message": "RAG application is running",
            "index_stats": vector_store.get_index_stats()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.post("/ask_question", response_model=QuestionResponse, tags=["Chat"])
@require_components("rag_engine")
async def ask_question(request: QuestionRequest):
    """Ask a question and get an AI-generated response based on Help Center articles."""
    try:
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
            quality_score=result.get('quality_score')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.get("/stats", tags=["General"])
@require_components("vector_store")
async def get_stats():
    """Get statistics about the indexed documents and vector store."""
    try:
        settings = get_settings()
        return {
            "index_stats": vector_store.get_index_stats(),
            "configuration": {
                "embedding_model": settings.embedding_model,
                "llm_model": settings.llm_model,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "top_k_results": settings.top_k_results
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
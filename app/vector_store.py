"""Vector store management using Pinecone for semantic search."""

from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from app.config import get_settings
from app.data_processor import Document
import time


class VectorStore:
    """
    Manages vector embeddings and semantic search using Pinecone.
    
    Design Decisions:
    - Uses OpenAI's text-embedding-3-large for cost-effective, high-quality embeddings
    - Pinecone serverless for scalable vector storage
    - Batched embedding generation for efficiency
    - Metadata filtering support for better retrieval
    """
    
    EMBEDDING_BATCH_SIZE = 100
    VECTOR_BATCH_SIZE = 100
    PINECONE_REGION = 'us-east-1'
    PINECONE_CLOUD = 'aws'
    INIT_WAIT_TIME = 1  # seconds
    
    # Default metadata values for search results
    METADATA_DEFAULTS = {
        'title': '',
        'url': '',
        'article_id': '',
        'chunk_index': '',
        'chunk_type': 'general',
        'topics': '',
        'difficulty': 'beginner',
        'quality_score': '0.0',
        'has_code': 'false',
        'has_steps': 'false'
    }
    
    def __init__(self, settings=None):
        """Initialize vector store with Pinecone and OpenAI clients."""
        self.settings = settings or get_settings()
        self.pc = Pinecone(api_key=self.settings.pinecone_api_key)
        self.openai_client = OpenAI(api_key=self.settings.openai_api_key)
        self.index = None
    
    def initialize_index(self, force_recreate: bool = False):
        """
        Initialize or create Pinecone index.
        
        Args:
            force_recreate: If True, delete and recreate the index
        """
        index_name = self.settings.pinecone_index_name
        existing_indexes = self._get_existing_indexes()
        
        if force_recreate and index_name in existing_indexes:
            print(f"Deleting existing index: {index_name}")
            self.pc.delete_index(index_name)
            time.sleep(self.INIT_WAIT_TIME)
        
        if index_name not in existing_indexes:
            self._create_index(index_name)
        
        self.index = self.pc.Index(index_name)
        print(f"Connected to index: {index_name}")
    
    def _get_existing_indexes(self) -> List[str]:
        """Get list of existing index names, safely handling errors."""
        try:
            return [idx.name for idx in self.pc.list_indexes()]
        except Exception as e:
            print(f"Warning: Could not list existing indexes: {e}")
            return []
    
    def _create_index(self, index_name: str):
        """Create a new Pinecone index."""
        print(f"Creating new index: {index_name}")
        self.pc.create_index(
            name=index_name,
            dimension=self.settings.embedding_dimension,
            metric='cosine',
            spec=ServerlessSpec(cloud=self.PINECONE_CLOUD, region=self.PINECONE_REGION)
        )
        time.sleep(self.INIT_WAIT_TIME)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for i in range(0, len(texts), self.EMBEDDING_BATCH_SIZE):
            batch = texts[i:i + self.EMBEDDING_BATCH_SIZE]
            response = self.openai_client.embeddings.create(
                model=self.settings.embedding_model,
                input=batch
            )
            embeddings.extend([item.embedding for item in response.data])
        return embeddings
    
    def index_documents(self, documents: List[Document]):
        """
        Index documents into Pinecone.
        
        Args:
            documents: List of Document objects to index
        """
        if not self.index:
            raise ValueError("Index not initialized. Call initialize_index() first.")
        
        print(f"Generating embeddings for {len(documents)} documents...")
        
        texts = [doc.content for doc in documents]
        embeddings = self.generate_embeddings(texts)
        
        vectors = [
            {
                'id': doc.chunk_id,
                'values': embedding,
                'metadata': self._build_metadata(doc)
            }
            for doc, embedding in zip(documents, embeddings)
        ]
        
        for i in range(0, len(vectors), self.VECTOR_BATCH_SIZE):
            self.index.upsert(vectors=vectors[i:i + self.VECTOR_BATCH_SIZE])
        
        print(f"Successfully indexed {len(documents)} documents")
    
    def _build_metadata(self, doc: Document) -> Dict:
        """Build metadata dictionary for a document."""
        metadata = {**doc.metadata, 'content': doc.content}
        
        if hasattr(doc, 'quality_score') and doc.quality_score is not None:
            metadata['quality_score'] = str(doc.quality_score)
        
        if hasattr(doc, 'chunk_type') and doc.chunk_type is not None:
            metadata['chunk_type'] = doc.chunk_type
        
        return metadata
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Search for similar documents using semantic search.
        
        Args:
            query: User query string
            top_k: Number of results to return (defaults to settings)
        
        Returns:
            List of matching documents with scores and metadata
        """
        if not self.index:
            raise ValueError("Index not initialized. Call initialize_index() first.")
        
        top_k = top_k or self.settings.top_k_results
        query_embedding = self.generate_embeddings([query])[0]
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return [
            {
                'id': match.id,
                'score': match.score,
                'content': match.metadata.get('content', ''),
                'metadata': {
                    key: match.metadata.get(key, default)
                    for key, default in self.METADATA_DEFAULTS.items()
                }
            }
            for match in results.matches
        ]
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the current index."""
        if not self.index:
            return {'error': 'Index not initialized'}
        
        stats = self.index.describe_index_stats()
        return {
            'total_vectors': stats.total_vector_count,
            'dimension': stats.dimension,
            'index_fullness': stats.index_fullness
        }
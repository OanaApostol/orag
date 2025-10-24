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
    - Uses OpenAI's text-embedding-3-small for cost-effective, high-quality embeddings
    - Pinecone serverless for scalable vector storage
    - Batched embedding generation for efficiency
    - Metadata filtering support for better retrieval
    """
    
    def __init__(self):
        """Initialize vector store with Pinecone and OpenAI clients."""
        self.settings = get_settings()
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.settings.pinecone_api_key)
        
        # Initialize OpenAI for embeddings
        self.openai_client = OpenAI(api_key=self.settings.openai_api_key)
        
        self.index = None
    
    def initialize_index(self, force_recreate: bool = False):
        """
        Initialize or create Pinecone index.
        
        Args:
            force_recreate: If True, delete and recreate the index
        """
        index_name = self.settings.pinecone_index_name
        
        try:
            # Check if index exists
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if force_recreate and index_name in existing_indexes:
                print(f"Deleting existing index: {index_name}")
                self.pc.delete_index(index_name)
                existing_indexes.remove(index_name)
                time.sleep(1)  # Wait for deletion to complete
        except Exception as e:
            print(f"Warning: Could not list existing indexes: {e}")
            existing_indexes = []
        
        # Create index if it doesn't exist
        if index_name not in existing_indexes:
            print(f"Creating new index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=self.settings.embedding_dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            # Wait for index to be ready
            time.sleep(1)
        
        # Connect to index
        self.index = self.pc.Index(index_name)
        print(f"Connected to index: {index_name}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        # Process in batches to avoid rate limits
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.openai_client.embeddings.create(
                model=self.settings.embedding_model,
                input=batch
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
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
        
        # Generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = self.generate_embeddings(texts)
        
        # Prepare vectors for upsert
        vectors = []
        for doc, embedding in zip(documents, embeddings):
            # Enhanced metadata with quality score and chunk type
            enhanced_metadata = {
                **doc.metadata,
                'content': doc.content  # Store content in metadata for retrieval
            }
            
            # Add quality score and chunk type if available
            if hasattr(doc, 'quality_score') and doc.quality_score is not None:
                enhanced_metadata['quality_score'] = str(doc.quality_score)
            
            if hasattr(doc, 'chunk_type') and doc.chunk_type is not None:
                enhanced_metadata['chunk_type'] = doc.chunk_type
            
            vectors.append({
                'id': doc.chunk_id,
                'values': embedding,
                'metadata': enhanced_metadata
            })
        
        # Upsert to Pinecone in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        print(f"Successfully indexed {len(documents)} documents")
    
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
        
        if top_k is None:
            top_k = self.settings.top_k_results
        
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results with enhanced metadata
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                'id': match.id,
                'score': match.score,
                'content': match.metadata.get('content', ''),
                'metadata': {
                    'title': match.metadata.get('title', ''),
                    'url': match.metadata.get('url', ''),
                    'article_id': match.metadata.get('article_id', ''),
                    'chunk_index': match.metadata.get('chunk_index', ''),
                    'chunk_type': match.metadata.get('chunk_type', 'general'),
                    'topics': match.metadata.get('topics', ''),
                    'difficulty': match.metadata.get('difficulty', 'beginner'),
                    'quality_score': match.metadata.get('quality_score', '0.0'),
                    'has_code': match.metadata.get('has_code', 'false'),
                    'has_steps': match.metadata.get('has_steps', 'false')
                }
            })
        
        return formatted_results
    
    def get_index_stats(self) -> Dict:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary with index statistics
        """
        if not self.index:
            return {'error': 'Index not initialized'}
        
        stats = self.index.describe_index_stats()
        return {
            'total_vectors': stats.total_vector_count,
            'dimension': stats.dimension,
            'index_fullness': stats.index_fullness
        }


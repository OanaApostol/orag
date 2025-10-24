"""Configuration settings for the RAG application."""

import yaml
from pathlib import Path
from functools import lru_cache


class Settings:
    """Application settings loaded from YAML configuration file."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize settings from YAML file."""
        self.config_path = Path(config_path)
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # OpenAI Configuration
        self.openai_api_key = config['openai']['api_key']
        
        # Pinecone Configuration
        self.pinecone_api_key = config['pinecone']['api_key']
        self.pinecone_index_name = config['pinecone'].get('index_name', 'typeform-help-center')
        
        # Embedding Model Configuration
        self.embedding_model = config['embedding']['model']
        self.embedding_dimension = config['embedding']['dimension']
        
        # LLM Configuration
        self.llm_model = config['llm']['model']
        self.llm_temperature = config['llm']['temperature']
        self.max_tokens = config['llm']['max_tokens']
        
        # Retrieval Configuration
        self.top_k_results = config['retrieval']['top_k_results']
        self.chunk_size = config['retrieval']['chunk_size']
        self.chunk_overlap = config['retrieval']['chunk_overlap']
        
        # Enhanced Chunking Configuration
        self.use_semantic_chunking = config['chunking']['use_semantic_chunking']
        self.semantic_threshold_percentile = config['chunking']['semantic_threshold_percentile']
        self.min_chunk_quality_score = config['chunking']['min_chunk_quality_score']
        self.enable_content_aware_chunking = config['chunking']['enable_content_aware_chunking']
        self.dynamic_chunk_sizing = config['chunking']['dynamic_chunk_sizing']


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


"""Configuration settings for the RAG application."""

import yaml
from pathlib import Path
from functools import lru_cache


class Settings:
    """Application settings loaded from YAML configuration file."""
    
    # Mapping from YAML path to attribute name
    CONFIG_MAPPING = {
        ('openai', 'api_key'): 'openai_api_key',
        ('pinecone', 'api_key'): 'pinecone_api_key',
        ('pinecone', 'index_name'): 'pinecone_index_name',
        ('embedding', 'model'): 'embedding_model',
        ('embedding', 'dimension'): 'embedding_dimension',
        ('llm', 'model'): 'llm_model',
        ('llm', 'temperature'): 'llm_temperature',
        ('llm', 'max_tokens'): 'max_tokens',
        ('retrieval', 'top_k_results'): 'top_k_results',
        ('retrieval', 'chunk_size'): 'chunk_size',
        ('retrieval', 'chunk_overlap'): 'chunk_overlap',
        ('chunking', 'use_semantic_chunking'): 'use_semantic_chunking',
        ('chunking', 'semantic_threshold_percentile'): 'semantic_threshold_percentile',
        ('chunking', 'min_chunk_quality_score'): 'min_chunk_quality_score',
        ('chunking', 'enable_content_aware_chunking'): 'enable_content_aware_chunking',
        ('chunking', 'dynamic_chunk_sizing'): 'dynamic_chunk_sizing',
    }
    
    # Default values for optional keys
    DEFAULTS = {
        ('pinecone', 'index_name'): 'typeform-help-center',
    }
    
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
        
        if not config:
            raise ValueError(f"Configuration file is empty: {self.config_path}")
        
        # Load all configured attributes
        for (section, key), attr_name in self.CONFIG_MAPPING.items():
            if section not in config:
                raise KeyError(f"Missing configuration section: [{section}]")
            
            default = self.DEFAULTS.get((section, key))
            value = config[section].get(key, default)
            
            if value is None:
                raise KeyError(f"Missing required key: [{section}][{key}]")
            
            setattr(self, attr_name, value)


@lru_cache()
def get_settings(config_path: str = "config.yaml") -> Settings:
    """Get cached settings instance."""
    return Settings(config_path)
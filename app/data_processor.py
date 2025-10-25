"""Enhanced data ingestion and preprocessing module for Help Center articles."""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from app.config import get_settings


# Module-level constants for content detection and processing
CONTENT_PATTERNS = {
    'steps': [r'step\s+\d+', r'^\d+\.', r'how to', r'follow these steps'],
    'faq': [r'q:', r'q\)', r'question:', r'answer:', r'what (is|are)', r'why'],
    'tutorial': [r'tutorial', r'guide', r'introduction', r'getting started'],
}

OVERLAP_PERCENTAGES = {
    'steps': 0.15,
    'faq': 0.2,
    'tutorial': 0.25,
    'default': 0.2,
}

TOPIC_PATTERNS = {
    'integration': r'(integration|webhook|api|zapier)',
    'form_design': r'(form design|field|question)',
    'payment': r'(payment|stripe|billing|checkout)',
    'collaboration': r'(collaboration|team|workspace|share)',
    'reports': r'(report|analytics|result)',
    'account': r'(account|profile|settings|password)',
}

STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
    'to', 'was', 'will', 'with', 'you', 'your', 'i', 'me', 'this', 'or'
}


@dataclass
class Document:
    """Represents a document chunk with metadata."""
    content: str
    metadata: Dict[str, str]
    chunk_id: str
    quality_score: Optional[float] = None
    chunk_type: Optional[str] = None


class DataProcessor:
    """
    Enhanced data processor with semantic chunking and content-aware strategies.
    
    Design Decisions:
    - Uses semantic chunking for better coherence when enabled
    - Content-type aware chunking for different article structures
    - Dynamic chunk sizing based on content characteristics
    - Quality validation to filter out poor chunks
    - Rich metadata for better retrieval
    """
    
    def __init__(self, settings=None):
        """Initialize the data processor with configuration."""
        self.settings = settings or get_settings()
        
        # Initialize semantic chunker if enabled
        if self.settings.use_semantic_chunking:
            try:
                self.embeddings = OpenAIEmbeddings(model=self.settings.embedding_model)
                # Use RecursiveCharacterTextSplitter with enhanced separators for semantic-like chunking
                self.semantic_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
                )
            except Exception as e:
                print(f"Warning: Could not initialize semantic chunker: {e}")
                print("Falling back to recursive chunking")
                self.semantic_splitter = None
        else:
            self.semantic_splitter = None
        
        # Enhanced recursive splitter with better separators
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " - ", " ", ""]
        )
    
    def load_articles(self, articles: List[Dict[str, str]] = None) -> List[Document]:
        """
        Load and preprocess Help Center articles with enhanced chunking.
        Can load from provided articles or HTML files in data folder.
        
        Args:
            articles: List of article dictionaries with 'title', 'url', and 'content'
                    If None, loads from HTML files in data folder
        
        Returns:
            List of Document objects with chunks and metadata
        """
        documents = []
        
        if articles is None:
            articles = self._load_html_articles()
        
        for idx, article in enumerate(articles):
            cleaned_content = self._enhanced_text_cleaning(article['content'])
            chunking_strategy = self._detect_content_type(cleaned_content)
            
            # Create chunks using appropriate strategy
            if chunking_strategy == 'step_by_step':
                chunks = self._chunk_step_by_step(cleaned_content)
            elif chunking_strategy == 'faq':
                chunks = self._chunk_faq_format(cleaned_content)
            elif chunking_strategy == 'semantic' and self.semantic_splitter:
                try:
                    chunks = self.semantic_splitter.split_text(cleaned_content)
                except Exception as e:
                    print(f"Semantic chunking failed, falling back to recursive: {e}")
                    chunks = self._chunk_with_enhanced_recursive(cleaned_content)
            else:
                chunks = self._chunk_with_enhanced_recursive(cleaned_content)
            
            # Create Document objects with enhanced metadata
            for chunk_idx, chunk in enumerate(chunks):
                quality_score = self._calculate_chunk_quality(chunk)
                
                if quality_score < self.settings.min_chunk_quality_score:
                    continue
                
                chunk_type = self._classify_chunk_type(chunk)
                
                doc = Document(
                    content=chunk,
                    metadata=self._create_enhanced_metadata(chunk, article, chunk_idx, len(chunks), quality_score),
                    chunk_id=f"article_{idx}_chunk_{chunk_idx}",
                    quality_score=quality_score,
                    chunk_type=chunk_type
                )
                documents.append(doc)
        
        return documents

    def _load_html_articles(self) -> List[Dict[str, str]]:
        """Load articles from HTML files in data folder."""
        import os
        from bs4 import BeautifulSoup
        
        articles = []
        data_folder = "data"
        
        for filename in os.listdir(data_folder):
            if filename.endswith('.html'):
                filepath = os.path.join(data_folder, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        soup = BeautifulSoup(file.read(), 'html.parser')
                        
                        title = soup.find('title').text.replace(' – Help Center', '')
                        content_div = soup.find('div', class_='article-body') or soup.find('main')
                        
                        if content_div:
                            content = content_div.get_text(separator='\n', strip=True)
                            
                            articles.append({
                                'title': title,
                                'url': f"https://help.typeform.com/hc/en-us/articles/{filename}",
                                'content': content
                            })
                            
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
        
        return articles
    
    def _detect_content_type(self, content: str) -> str:
        """Detect content type and return appropriate chunking strategy."""
        if not self.settings.enable_content_aware_chunking:
            return 'semantic' if self.semantic_splitter else 'recursive'
        
        if self._has_pattern(content, 'steps'):
            return 'step_by_step'
        if self._has_pattern(content, 'faq'):
            return 'faq'
        if self._has_pattern(content, 'tutorial'):
            return 'tutorial'
        
        # Default to semantic for complex content
        if len(content) > 2000 and self.semantic_splitter:
            return 'semantic'
        
        return 'recursive'
    
    def _has_pattern(self, text: str, pattern_type: str) -> bool:
        """Check if text matches patterns of given type."""
        if pattern_type not in CONTENT_PATTERNS:
            return False
        
        patterns = CONTENT_PATTERNS[pattern_type]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE if pattern_type != 'steps' else 0):
                return True
        return False
    
    def _chunk_with_enhanced_recursive(self, content: str) -> List[str]:
        """Enhanced recursive chunking with dynamic sizing."""
        if self.settings.dynamic_chunk_sizing:
            optimal_size = self._calculate_optimal_chunk_size(content)
            optimal_overlap = int(optimal_size * OVERLAP_PERCENTAGES.get('tutorial', 0.25))
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=optimal_size,
                chunk_overlap=optimal_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " - ", " ", ""],
                length_function=len
            )
            return splitter.split_text(content)
        else:
            return self.recursive_splitter.split_text(content)
    
    def _chunk_step_by_step(self, content: str) -> List[str]:
        """Chunk step-by-step guides to preserve complete steps."""
        steps = re.split(r'\n(?=\d+\.)', content)
        chunks = []
        current_chunk = ""
        
        for step in steps:
            if len(current_chunk + step) <= self.settings.chunk_size:
                current_chunk += step
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = step
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_faq_format(self, content: str) -> List[str]:
        """Chunk FAQ format to preserve question-answer pairs."""
        qa_pairs = re.split(r'\n(?=[A-Z][^.!?]*\?)', content)
        return [qa.strip() for qa in qa_pairs if qa.strip()]
    
    def _calculate_optimal_chunk_size(self, content: str) -> int:
        """Calculate optimal chunk size based on content characteristics."""
        base_size = self.settings.chunk_size
        sentence_count = len(re.findall(r'[.!?]+', content))
        avg_sentence_length = len(content) / max(sentence_count, 1)
        
        if avg_sentence_length > 100:
            return int(base_size * 0.7)
        elif avg_sentence_length < 50:
            return int(base_size * 1.2)
        return base_size
    
    def _calculate_chunk_quality(self, chunk: str) -> float:
        """Calculate quality score for a chunk."""
        score = 0.0
        
        # Length appropriateness (0-0.3)
        score += min(len(chunk) / self.settings.chunk_size, 1.0) * 0.3
        
        # Completeness (0-0.3)
        if self._is_complete_sentence(chunk):
            score += 0.3
        
        # Context richness (0-0.2)
        if self._has_sufficient_context(chunk):
            score += 0.2
        
        # Information density (0-0.2)
        if self._has_high_information_density(chunk):
            score += 0.2
        
        return min(score, 1.0)
    
    def _create_enhanced_metadata(self, chunk: str, article: dict, chunk_idx: int, total_chunks: int, quality_score: float) -> Dict[str, str]:
        """Create rich metadata for better retrieval."""
        chunk_type = self._classify_chunk_type(chunk)
        key_topics = self._extract_topics(chunk)
        difficulty_level = self._assess_difficulty(chunk)
        
        return {
            'title': article['title'],
            'url': article['url'],
            'article_id': str(article.get('id', '')),
            'chunk_index': str(chunk_idx),
            'total_chunks': str(total_chunks),
            'chunk_type': chunk_type,
            'topics': ','.join(key_topics),
            'difficulty': difficulty_level,
            'word_count': str(len(chunk.split())),
            'char_count': str(len(chunk)),
            'has_code': 'true' if self._has_pattern(chunk, 'code') else 'false',
            'has_steps': 'true' if self._has_pattern(chunk, 'steps') else 'false',
            'quality_score': str(quality_score)
        }
    
    def _classify_chunk_type(self, chunk: str) -> str:
        """Classify chunk type for better retrieval."""
        if re.search(r'\d+\.\s+[A-Z]', chunk):
            return 'step_by_step'
        elif re.search(r'\?\s*$', chunk.strip()):
            return 'question'
        elif re.search(r'^[A-Z][^.!?]*:', chunk):
            return 'definition'
        elif re.search(r'(best practice|tip|recommendation)', chunk.lower()):
            return 'advice'
        elif re.search(r'(error|problem|issue|troubleshoot)', chunk.lower()):
            return 'troubleshooting'
        else:
            return 'general'
    
    def _extract_topics(self, chunk: str) -> List[str]:
        """Extract key topics from chunk."""
        topics = []
        for topic, pattern in TOPIC_PATTERNS.items():
            if re.search(pattern, chunk.lower()):
                topics.append(topic)
        return topics
    
    def _assess_difficulty(self, chunk: str) -> str:
        """Assess difficulty level of chunk."""
        technical_terms = len(re.findall(r'\b(api|webhook|integration|configuration|authentication)\b', chunk.lower()))
        avg_sentence_length = len(chunk.split()) / max(len(re.findall(r'[.!?]+', chunk)), 1)
        
        if technical_terms > 3 or avg_sentence_length > 25:
            return 'advanced'
        elif technical_terms > 1 or avg_sentence_length > 15:
            return 'intermediate'
        else:
            return 'beginner'
    
    def _is_complete_sentence(self, chunk: str) -> bool:
        """Check if chunk contains complete sentences."""
        sentences = re.split(r'[.!?]+', chunk)
        return len(sentences) >= 1 and any(s.strip() for s in sentences)
    
    def _has_sufficient_context(self, chunk: str) -> bool:
        """Check if chunk has sufficient context."""
        sentences = re.findall(r'[.!?]+', chunk)
        return len(sentences) >= 2 or len(chunk) > 100
    
    def _has_high_information_density(self, chunk: str) -> bool:
        """Check if chunk has high information density."""
        words = chunk.lower().split()
        meaningful_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        return len(meaningful_words) / max(len(words), 1) > 0.6
    
    def _enhanced_text_cleaning(self, text: str) -> str:
        """Enhanced text cleaning that preserves structure."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\*\*(.*?)\*\*', r'**\1**', text)
        text = re.sub(r'\*(.*?)\*', r'*\1*', text)
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"\*\n]', '', text)
        text = re.sub(r'^[\s]*[-*]\s+', '• ', text, flags=re.MULTILINE)
        return text.strip()
    
    def get_stats(self, documents: List[Document]) -> Dict:
        """Get statistics about the processed documents."""
        total_chunks = len(documents)
        avg_chunk_length = sum(len(doc.content) for doc in documents) / total_chunks if total_chunks > 0 else 0
        unique_articles = len(set(doc.metadata['article_id'] for doc in documents))
        
        chunk_types = {}
        quality_scores = []
        
        for doc in documents:
            chunk_type = doc.chunk_type or 'unknown'
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            if doc.quality_score is not None:
                quality_scores.append(doc.quality_score)
        
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            'total_chunks': total_chunks,
            'average_chunk_length': round(avg_chunk_length, 2),
            'unique_articles': unique_articles,
            'average_quality_score': round(avg_quality_score, 3),
            'chunk_types': chunk_types,
            'quality_score_range': {
                'min': round(min(quality_scores), 3) if quality_scores else 0,
                'max': round(max(quality_scores), 3) if quality_scores else 0
            }
        }


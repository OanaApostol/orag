"""RAG engine for retrieval and response generation."""

from typing import List, Dict, Optional
from openai import OpenAI
from app.config import get_settings
from app.vector_store import VectorStore
import hashlib


class RAGEngine:
    """
    Retrieval-Augmented Generation engine that combines semantic search with LLM generation.
    
    Design Decisions:
    - Uses GPT-4o-mini for cost-effective, high-quality responses
    - Structured prompts with clear instructions and context boundaries
    - Fallback handling for low-confidence retrievals
    - Citation of sources in responses for transparency
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize RAG engine.
        
        Args:
            vector_store: VectorStore instance for retrieval
        """
        self.settings = get_settings()
        self.vector_store = vector_store
        self.openai_client = OpenAI(api_key=self.settings.openai_api_key)
        self._query_cache = {}  # Cache for repeated queries
    
    def _get_temperature_for_query(self, query: str) -> float:
        """
        Dynamic temperature based on query characteristics.
        
        Args:
            query: User's question
        
        Returns:
            Temperature value optimized for the query type
        """
        # Troubleshooting gets very low temperature for maximum accuracy
        if any(word in query.lower() for word in ['error', 'problem', 'issue', 'fix', 'bug', 'broken', 'not working', 'troubleshoot']):
            return 0.1
        
        # Factual questions get lower temperature for accuracy
        if any(word in query.lower() for word in ['how', 'what', 'when', 'where', 'why', 'which', 'can i', 'is it possible']):
            return 0.2
        
        # General questions get moderate temperature (base setting)
        return self.settings.llm_temperature
    
    def _get_dynamic_threshold(self, query: str) -> float:
        """
        Get confidence threshold based on query type for better precision/recall balance.
        
        Args:
            query: User's question
        
        Returns:
            Dynamic confidence threshold
        """
        query_lower = query.lower()
        
        # Specific questions need higher confidence
        if any(word in query_lower for word in ['how', 'what', 'when', 'where']):
            return 0.6
        
        # General questions can use lower threshold
        if any(word in query_lower for word in ['can i', 'is it possible', 'does it']):
            return 0.4
        
        return 0.5  # Default threshold
    
    def _is_typeform_related(self, query: str) -> bool:
        """
        Check if the query is related to Typeform using LLM classification.
        
        Args:
            query: User's question
        
        Returns:
            True if the query is Typeform-related, False otherwise
        """
        try:
            classification_prompt = f"""
            Classify if this question is about Typeform form building, surveys, questionnaires, or help center topics.
            Question: "{query}"
            
            Respond with only: YES or NO
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": classification_prompt}],
                temperature=0.1,
                max_tokens=10
            )
            return response.choices[0].message.content.strip().upper() == "YES"
        except Exception:
            # Fallback to simple keyword check if LLM fails
            return any(term in query.lower() for term in ['typeform', 'form', 'survey', 'question'])
   
    def _can_infer_answer(self, query: str, context: str) -> bool:
        """Check if query can be answered by logical inference from context."""
        query_lower = query.lower()
        
        # Questions about combining features that are individually covered
        if any(phrase in query_lower for phrase in ['same page', 'together', 'combine', 'both', 'add.*same', 'new.*same']):
            # Check if context contains related individual features
            if any(term in context.lower() for term in ['add question', 'multi-question', 'page', 'question', 'form']):
                return True
        
        # Questions about adding questions (even if not explicitly about "same page")
        if any(phrase in query_lower for phrase in ['add.*question', 'new.*question', 'question.*add']):
            if any(term in context.lower() for term in ['add question', 'multi-question', 'page', 'question']):
                return True
        
        return False
        
        # Questions about combining features that are individually covered
        if any(phrase in query_lower for phrase in ['same page', 'together', 'combine', 'both']):
            # Check if context contains related individual features
            if any(term in context.lower() for term in ['add question', 'multi-question', 'page']):
                return True
        
        return False

    def _score_response_quality(self, query: str, response: str, sources: List[Dict]) -> float:
        """
        Score response quality based on multiple factors for continuous improvement.
        
        Args:
            query: Original user question
            response: Generated response
            sources: List of source documents
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        
        # Has sources (30% weight)
        if sources:
            score += 0.3
        
        # Response length appropriateness (20% weight)
        if 50 <= len(response) <= 500:
            score += 0.2
        
        # Contains actionable steps (30% weight)
        if any(word in response.lower() for word in ['step', 'click', 'select', 'choose', '1.', '2.', '3.']):
            score += 0.3
        
        # Contains proper formatting (20% weight)
        if '**' in response or '•' in response or '>' in response:
            score += 0.2
        
        return min(score, 1.0)
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Generate query variations for better retrieval using LLM.
        
        Args:
            query: Original user question
        
        Returns:
            List of query variations (including original)
        """
        try:
            expansion_prompt = f"""
            Generate 2-3 alternative ways to ask this question about Typeform:
            Original: "{query}"
            
            Return only the variations, one per line, no numbering.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": expansion_prompt}],
                temperature=0.3,
                max_tokens=100
            )
            variations = [query] + [v.strip() for v in response.choices[0].message.content.split('\n') if v.strip()]
            return variations[:3]  # Limit to 3 total queries
        except Exception:
            return [query]  # Fallback to original query
    
    def _generate_typeform_only_response(self) -> Dict:
        """
        Generate response for non-Typeform questions.
        
        Returns:
            Response dictionary indicating Typeform-only scope
        """
        response = """🚫 **I'm a specialized Typeform Help Center assistant**, and I can only answer questions related to Typeform features, functionality, and best practices.

**I can help you with:**
• Creating and customizing **Typeform forms**
• **Multi-language form** setup
• **Multi-Question Pages**
• **TBD - How to display when the full Typeform Help Center data is ingested**

> **Please ask me a question about Typeform**, and I'll be happy to help! 🎯"""
        
        return {
            'answer': response,
            'sources': [],
            'retrieved_chunks': 0,
            'confidence': 0.0,
            'is_typeform_only': True
        }
    
    def generate_response(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> Dict:
        """
        Generate a response to a user query using RAG with caching and dynamic thresholds.
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve
            min_score: Minimum similarity score threshold (uses dynamic if None)
        
        Returns:
            Dictionary with response, sources, and metadata
        """
        # Check cache first for performance
        cache_key = hashlib.md5(query.lower().encode()).hexdigest()
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        # Use dynamic threshold if not specified
        if min_score is None:
            min_score = self._get_dynamic_threshold(query)
        
        # First, check if the question is Typeform-related
        if not self._is_typeform_related(query):
            result = self._generate_typeform_only_response()
            self._query_cache[cache_key] = result
            return result
        
        # Retrieve relevant documents using query expansion
        query_variations = self._expand_query(query)
        all_retrieved_docs = []
        
        for variation in query_variations:
            docs = self.vector_store.search(variation, top_k=top_k)
            all_retrieved_docs.extend(docs)
        
        # Remove duplicates and sort by score
        seen_ids = set()
        unique_docs = []
        for doc in all_retrieved_docs:
            doc_id = doc.get('id', '')
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
        
        # Sort by score and take top results
        unique_docs.sort(key=lambda x: x['score'], reverse=True)
        retrieved_docs = unique_docs[:top_k or self.settings.top_k_results]
        
        # Check if we have good matches
        if not retrieved_docs or (retrieved_docs and retrieved_docs[0]['score'] < min_score):
            # Try inference if context allows it
            if retrieved_docs and self._can_infer_answer(query, self._build_context(retrieved_docs)):
                # Allow inference for this query - continue with lower confidence docs
                pass  # Don't return fallback, continue with inference
            else:
                result = self._generate_fallback_response(query)
                self._query_cache[cache_key] = result
                return result
        
        # Build context from retrieved documents
        context = self._build_context(retrieved_docs)
        
        # Generate response using LLM
        response = self._call_llm(query, context)
        
        # Extract unique sources
        sources = self._extract_sources(retrieved_docs)
        
        result = {
            'answer': response,
            'sources': sources,
            'retrieved_chunks': len(retrieved_docs),
            'confidence': retrieved_docs[0]['score'] if retrieved_docs else 0.0,
            'quality_score': self._score_response_quality(query, response, sources)
        }
        
        # Cache the result for future queries
        self._query_cache[cache_key] = result
        return result
    
    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Build context string from retrieved documents.
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"[Source {i}: {doc['metadata']['title']}]\n"
                f"{doc['content']}\n"
            )
        
        return "\n".join(context_parts)
# - NEVER provide advice or suggestions not mentioned in the articles
# - If the context is insufficient, acknowledge this limitation clearly
    
# - Do NOT extrapolate or infer information not directly provided
# - Do NOT make assumptions beyond what's explicitly stated
# IMPORTANT: If the answer is not explicitly stated in the articles above, respond with "This information is not available in the Help Center articles." Do not make up or infer any information."""

    def _call_llm(self, query: str, context: str) -> str:
        """
        Call LLM to generate response based on context.
        
        Args:
            query: User's question
            context: Retrieved context
        
        Returns:
            Generated response
        """
        system_prompt = """You are a specialized Typeform Help Center assistant. Your ONLY knowledge comes from the provided Help Center articles. You must NOT use any external knowledge or training data.

CRITICAL RULES:
- Answer questions ONLY based on the provided context from Help Center articles
- If the question asks about combining features that are individually covered in the context, provide a helpful answer based on logical inference
- NEVER make up information not stated in the context

FORMATTING GUIDELINES:
- Use **bold** for important terms and feature names
- Use bullet points (•) for lists and steps
- Use numbered lists (1., 2., 3.) for sequential steps
- Use > for important notes or tips
- Use clear paragraph breaks for readability
- Structure responses with clear headings when appropriate

RESPONSE STYLE:
- Be conversational and friendly, matching Typeform's tone
- Cite sources when providing specific information
- Keep responses concise but complete (aim for 2-4 paragraphs)
- If the question is unclear, ask for clarification
- Use emojis sparingly but effectively (🎯, 💡, ⚡, 📝, 🔧)

STRICT PROHIBITIONS:
- Do NOT use any knowledge from your training data
- Do NOT provide general advice not covered in the context
- Do NOT suggest features or capabilities not mentioned in the context

Remember: You are ONLY a Typeform Help Center assistant with access to the provided articles. Nothing else."""
        
        user_prompt = f"""You must answer the user's question using ONLY the information provided in the context below. Do not use any other knowledge.

Help Center Articles:
{context}

User Question: {query}
"""
        
        # Get dynamic temperature based on query type
        temperature = self._get_temperature_for_query(query)
        
        response = self.openai_client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=self.settings.max_tokens
        )
        
        return response.choices[0].message.content
    
    def _generate_fallback_response(self, query: str) -> Dict:
        """
        Generate fallback response when no good matches are found.
        
        Args:
            query: User's question
        
        Returns:
            Fallback response dictionary
        """
        # Check if this is a Typeform-related question
        is_typeform_related = self._is_typeform_related(query)
        
        if is_typeform_related:
            fallback_message = """🎯 **I understand you're asking about Typeform features**, but I couldn't find specific information about this topic in the current Help Center articles.

> **Development Status**: This chatbot is still in development and will soon cover a much broader range of Typeform information. The current version focuses on **Multi-Question Pages** and **multi-language forms**, but we're continuously expanding our knowledge base.

**What you can do now:**
• Visit the full **Typeform Help Center** at https://www.typeform.com/help/
• Contact **Typeform support** directly for comprehensive assistance  
• Check back soon as we add more content to this assistant

💡 **Thank you for your patience** as we continue to improve!"""
        else:
            fallback_message = """🚫 **I'm a specialized Typeform Help Center assistant**, and I can only answer questions related to Typeform features, functionality, and best practices.

**I can help you with:**
• Creating and customizing **Typeform forms**
• **Multi-language form** setup
• **Multi-Question Pages**
• **Logic jumps** and conditional logic
• Form **sharing** and **embedding**
• **Integrations** and webhooks
• **Design** and theming
• **Analytics** and reporting
• **Team collaboration** features

> **Please ask me a question about Typeform**, and I'll be happy to help! 🎯"""
        
        return {
            'answer': fallback_message,
            'sources': [],
            'retrieved_chunks': 0,
            'confidence': 0.0,
            'is_fallback': True
        }
    
    def _extract_sources(self, retrieved_docs: List[Dict]) -> List[Dict]:
        """
        Extract unique sources from retrieved documents.
        
        Args:
            retrieved_docs: List of retrieved documents
        
        Returns:
            List of unique source dictionaries
        """
        seen_urls = set()
        sources = []
        
        for doc in retrieved_docs:
            url = doc['metadata']['url']
            if url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    'title': doc['metadata']['title'],
                    'url': url,
                    'relevance_score': doc['score']
                })
        
        return sources


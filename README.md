# Typeform Help Center RAG Chatbot

An AI-powered chatbot for Typeform's Help Center using Retrieval-Augmented Generation (RAG) to provide accurate, context-aware responses to user questions.

## 🎯 Project Overview

This project implements an end-to-end RAG solution that:
- ✅ Loads and preprocesses Help Center articles from HTML files
- ✅ Chunks and embeds content using OpenAI's `text-embedding-3-large` model
- ✅ Stores embeddings in Pinecone vector database for semantic search
- ✅ Retrieves relevant context for user queries with dynamic thresholds
- ✅ Generates responses using OpenAI's GPT-4o-mini model
- ✅ Exposes a REST API using FastAPI
- ✅ Includes evaluation framework
- ✅ Containerized with Docker for easy deployment

## 🏗️ Architecture

```
User Query → FastAPI Endpoint → Query Expansion → Vector Search (Pinecone) → Context Retrieval
                                                                ↓
User ← Response Generation (GPT-4o-mini) ← Prompt + Context ←┘
```

### Key Components

1. **Data Processor** (`app/data_processor.py`)
   - Loads and cleans Help Center articles from HTML files
   - Implements multi-strategy chunking (semantic, recursive, content-aware)
   - Preserves metadata for better retrieval and quality scoring
   - Classifies chunks by type (step-by-step, question, definition, advice, troubleshooting, general)

2. **Vector Store** (`app/vector_store.py`)
   - Manages Pinecone integration with `text-embedding-3-large` (3072 dimensions)
   - Generates embeddings in batches (100 texts per batch) for efficiency during indexing
   - Processes individual search queries through single embedding generation
   - Handles semantic search queries with metadata filtering

3. **RAG Engine** (`app/rag_engine.py`)
   - Orchestrates retrieval and generation with query expansion (2-3 variations)
   - Implements dynamic temperature based on query type
   - Implements dynamic similarity thresholds per query classification
   - Handles fallback scenarios and response quality scoring
   - Caches responses for repeated queries

4. **FastAPI Application** (`app/main.py`)
   - REST API with `/ask_question` endpoint
   - Health checks, statistics, and interactive documentation at `/docs`
   - Web-based chat interface at `/chat`
   - Automatic index initialization with `force_recreate=True`

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key
- Pinecone API key (free tier is sufficient)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd orag
   ```

2. **Configure API keys in `config.yaml`**
   ```yaml
   openai:
     api_key: "your_openai_api_key_here"
   
   pinecone:
     api_key: "your_pinecone_api_key_here"
   ```

3. **Add Help Center articles to `data/` folder**
   - Download Help Center articles as HTML files
   - Place them in the `data/` directory
   - See [Data Management](#-data-management) section for details

4. **Run with Docker**
   ```bash
   docker-compose up --build
   ```

The API will be available at `http://localhost:8000`

### Testing the API

**Using curl:**
```bash
curl -X POST "http://localhost:8000/ask_question" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I create a form in multiple languages?"}'
```

**Using the interactive docs:**
Visit `http://localhost:8000/docs` for Swagger UI

**Interactive chat interface:**
Visit `http://localhost:8000/chat` for web-based chat

**Health check:**
```bash
curl http://localhost:8000/health
```

**Statistics:**
```bash
curl http://localhost:8000/stats
```

## 📚 Data Management

This POC focuses on manual data management for simplicity and control.

### Adding Help Center Articles

1. Navigate to a Help Center article in your browser
2. Save the page as HTML (File → Save Page As → Web Page, Complete)
3. Place the `.html` file into the `data/` folder

### Expected Folder Structure

```
data/
├── Article-Title-1 – Help Center.html
├── Article-Title-1 – Help Center_files/     # Assets folder (auto-created)
│   ├── images/
│   ├── stylesheets/
│   └── scripts/
├── Article-Title-2 – Help Center.html
└── Article-Title-2 – Help Center_files/
```

### Updating Articles

1. Replace the existing `.html` file in `data/` with the updated one
2. Restart the application to reindex:
   ```bash
   docker-compose restart
   ```

**Note:** The index is recreated on every startup (`force_recreate=True` in `app/main.py`). Articles are processed and embedded each time the service starts.

## 🎨 Prompt Design

### System Prompt

The RAG engine uses a carefully engineered system prompt to ensure accurate, contextual responses:

```
You are a specialized Typeform Help Center assistant. Your ONLY knowledge comes from 
the provided Help Center articles. You must NOT use any external knowledge or training data.

CRITICAL RULES:
- Answer questions ONLY based on the provided context from Help Center articles
- If the question asks about combining features that are individually covered in the context, 
  provide a helpful answer based on logical inference
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

Remember: You are ONLY a Typeform Help Center assistant with access to the provided articles. 
Nothing else.
```

### Prompt Engineering Techniques

1. **Role Definition:** Establishes clear boundaries as Typeform Help Center assistant
2. **Strict Context Adherence:** Forces responses based only on provided context
3. **Logical Inference:** Allows combining information when features are individually covered
4. **Anti-Hallucination Measures:** Multiple prohibitions against external knowledge
5. **Formatting Guidelines:** Ensures consistent, readable output
6. **Conversational Tone:** Matches Typeform's brand voice

## 📊 Design Decisions

### 1. Enhanced Chunking Strategy

**Choice:** Multi-strategy chunking with semantic awareness and content-type detection

**Core Features:**
- **Semantic Chunking:** Uses OpenAI embeddings to identify natural breakpoints
- **Content-Aware Chunking:** Different strategies for step-by-step guides, FAQs, tutorials
- **Dynamic Sizing:** Adjusts chunk size based on content density and complexity
- **Quality Validation:** Filters out low-quality chunks using scoring algorithm
- **Rich Metadata:** Extracts topics, difficulty levels, and content characteristics

**Chunking Strategies:**
- **Semantic:** For complex content >2000 characters (uses embedding similarity)
- **Step-by-Step:** Preserves complete numbered steps
- **FAQ:** Maintains question-answer pairs
- **Recursive:** Fallback with enhanced separators and dynamic sizing

**Quality Scoring (0-1 scale):**
- Length appropriateness (30%)
- Sentence completeness (30%)
- Context richness (20%)
- Information density (20%)

**Trade-offs:**
- ✅ Better semantic coherence and context preservation
- ✅ Content-type specific optimization
- ✅ Quality filtering reduces noise
- ✅ Rich metadata improves retrieval precision
- ❌ Higher computational cost for semantic chunking
- ❌ More complex configuration options

### 2. Embedding Model

**Choice:** OpenAI's `text-embedding-3-large`

**Reasoning:**
- **Domain-specific accuracy:** The Typeform Help Center contains specialised terminology that requires high-dimensional semantic understanding
- **Similar questions, distinct meanings:** Better distinguishes between similar words or phrases with different meanings
- **Technical content handling:** The 3072-dimensional representation captures nuanced differences in technical tutorials and API documentation
- **Validated through testing:** `text-embedding-3-small` delivered insufficient answer quality
- **Semantic chunking dependency:** Relies on high-quality embeddings for accurate chunk boundary detection
- **Customer support use case:** Prioritizes answer accuracy and reliability over cost

**Model Comparison:**
- `text-embedding-3-small` (1536 dims): Lower cost but insufficient for technical content (tested and rejected)
- `text-embedding-3-large` (3072 dims): Superior semantic precision justified by improved answer quality ✓ CHOSEN

**Embedding Generation:**
- During indexing: Embeddings generated in batches of 100 for efficiency
- During search: Individual query embedding generated per search

### 3. Vector Database

**Choice:** Pinecone

**Advantages:**
- **Managed service:** No infrastructure overhead
- **Serverless tier:** Perfect for prototyping
- **Fast queries:** Optimized for similarity search
- **Metadata filtering:** Supports filtering by article, chunk type, etc.
- **Free tier sufficient:** 100K vectors for free

**Indexing Process:**
1. Data Ingestion: HTML → BeautifulSoup → Clean text
2. Content Detection: FAQ vs. step-by-step vs. tutorial
3. Chunking: Different strategies per content type
4. Quality Filtering: Remove chunks below 0.3 score
5. Embedding: Convert to 3072-dim vectors (batched)
6. Metadata Enrichment: Add topics, difficulty, chunk type
7. Pinecone Upload: Batch upsert with metadata

**Search Process:**
```
User Query
    ↓
[Classification] ← Is it Typeform-related? (LLM check)
    ↓ (Yes)
[Query Expansion] ← Generate 2-3 variations (LLM)
    ↓
[Dynamic Threshold] ← Set confidence threshold based on query type
    ↓
[Vector Embedding] ← Convert query to 3072-dim vector
    ↓
[Pinecone Search] ← Find top-3 similar chunks per variation
    ↓
[Deduplication] ← Remove duplicate chunks
    ↓
[Filtering] ← Keep only chunks above threshold
    ↓
[Context Building] ← Combine chunks into single context
    ↓
[Inference Check] ← Can answer be inferred from available context?
    ↓
[LLM Generation] ← Generate response with dynamic temperature
    ↓
[Quality Scoring] ← Score response on multiple factors
    ↓
[Caching] ← Store for future identical queries
    ↓
Response to User
```

**Key Features:**
- **Semantic chunking** preserves content boundaries
- **Query expansion** catches paraphrased questions
- **Dynamic thresholds** adjust precision per query type
- **Source attribution** with relevance scores
- **Query caching** eliminates redundant searches
- **Inference checking** allows combining information when justified

### 4. Dynamic Retrieval Configuration

**Choice:** Query-type-specific thresholds for optimal precision/recall balance

**Threshold Values:**
- **"How/What/When/Where" questions:** 0.45 threshold (higher precision)
- **"Can I/Is it possible" questions:** 0.3 threshold (higher recall)
- **General questions:** 0.5 threshold (balanced)

**Rationale:**
- Different question types require different confidence levels
- Specific questions benefit from higher precision
- General questions can accept lower threshold for broader recall

### 5. Dynamic Temperature Strategy

**Choice:** Context-aware temperature based on query type

**Temperature Values:**
- **Error/Troubleshooting queries:** 0.2 (focused, factual)
- **Factual questions (How/What/When/Where/Why/Which/Can I):** 0.3 (structured, clear)
- **General questions:** 0.25 (balanced from config default)

**Rationale:**
- Troubleshooting requires maximum accuracy to avoid incorrect solutions
- Factual questions benefit from structured, predictable outputs
- General questions balance accuracy with natural conversational flow

### 6. Language Model

**Choice:** GPT-4o-mini with Dynamic Temperature

**Reasoning:**
- **Cost-effective:** $0.15/1M input tokens (vs $5/1M for GPT-4)
- **Fast:** Lower latency than GPT-4
- **Sufficient capability:** Excellent for Q&A tasks
- **JSON mode support:** Structured outputs when needed

## 📊 Evaluation Framework

The project includes a comprehensive evaluation framework in the `evaluation/` folder:

### Evaluation Metrics

**Retrieval Quality:**
- Precision@k, Recall@k, MRR, NDCG@k

**Generation Quality:**
- BLEU Score, ROUGE-L, Semantic Similarity, Quality Score

**System Performance:**
- Response Time, Confidence Score, Role Adherence

### Running Evaluation

```bash
# Navigate to evaluation directory
cd evaluation

# Run complete evaluation example
python evaluate_rag.py --example

# Run evaluation with custom test data
python evaluate_rag.py --test_data test_questions.json --output evaluation_report.json
```

See `evaluation/EVALUATION_README.md` for detailed documentation.

## 🔧 Configuration

Configuration is managed through the `config.yaml` file:

```yaml
openai:
  api_key: "your_openai_api_key"

pinecone:
  api_key: "your_pinecone_api_key"

embedding:
  model: "text-embedding-3-large"
  dimension: 3072

llm:
  model: "gpt-4o-mini"
  temperature: 0.25
  max_tokens: 500

retrieval:
  top_k_results: 3
  chunk_size: 800
  chunk_overlap: 200

chunking:
  use_semantic_chunking: true
  semantic_threshold_percentile: 95
  min_chunk_quality_score: 0.3
  enable_content_aware_chunking: true
  dynamic_chunk_sizing: true
```

### Key Settings

| Setting | Value | Description |
|---------|-------|-------------|
| `embedding.model` | `text-embedding-3-large` | High-quality embeddings with 3072 dimensions |
| `embedding.dimension` | `3072` | Vector dimension for Pinecone index |
| `llm.model` | `gpt-4o-mini` | Cost-effective, high-quality language model |
| `llm.temperature` | `0.25` | Base temperature (overridden by query type) |
| `retrieval.top_k_results` | `3` | Number of chunks to retrieve |
| `retrieval.chunk_size` | `800` | Chunk size in characters |
| `retrieval.chunk_overlap` | `200` | Chunk overlap in characters |

## 📈 Performance Metrics

Based on evaluation results:

**Retrieval Performance:**
- Precision@1: 0.75 (75% of top results are relevant)
- MRR: 0.75 (relevant results found quickly)
- NDCG@1: 0.75 (good ranking quality)

**Generation Performance:**
- BLEU Score: 0.16 (moderate word overlap with ground truth)
- ROUGE-L: 0.15 (decent sentence structure similarity)
- Semantic Similarity: 0.14 (good semantic alignment)
- Quality Score: 0.17 (composite quality metric)

**System Performance:**
- Average Response Time: ~6 seconds
- Average Confidence: 0.38
- Role Adherence: 0.85+ (excellent adherence to Typeform Help Center role)

## 🛠️ Development

### Project Structure

```
orag/
├── app/                         # Main application code
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── data_processor.py       # HTML processing and chunking
│   ├── main.py                 # FastAPI application
│   ├── rag_engine.py           # RAG orchestration
│   └── vector_store.py         # Pinecone integration
├── data/                        # Help Center articles (HTML files)
├── evaluation/                  # Evaluation framework
│   ├── evaluate_rag.py         # Main evaluation script
│   ├── EVALUATION_README.md    # Evaluation documentation
│   ├── evaluation_report.json  # Latest evaluation results
│   └── test_questions.json     # Sample test questions
├── config.yaml                 # Configuration file
├── docker-compose.yml          # Docker setup
├── Dockerfile                  # Container definition
├── pyproject.toml              # Python dependencies (uv)
├── chat.html                   # Web-based chat interface
├── uv.lock                     # Locked dependencies
└── README.md                   # This file
```

### Dependencies

Key dependencies managed via `pyproject.toml`:
- **FastAPI** (0.104.1) - Web framework
- **OpenAI** (≥1.6.1) - LLM and embeddings
- **Pinecone** (3.0.0) - Vector database
- **LangChain** (0.1.0) - LLM framework
- **BeautifulSoup4** (4.12.2) - HTML parsing
- **PyYAML** (6.0.1) - Configuration management

### Local Development

```bash
# Install dependencies
uv sync

# Run locally (requires API keys in config.yaml)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Considerations

- **API Key Security:** Move API keys to environment variables, not config.yaml
- **Index Management:** Implement index backup/restore strategy
- **Monitoring:** Add logging and metrics collection
- **Scaling:** Consider horizontal scaling for high traffic

## 📝 API Reference

### Endpoints

**POST `/ask_question`**
- **Description:** Ask a question to the RAG chatbot
- **Request Body:** `{"question": "string", "top_k": int (optional)}`
- **Response:** `{"answer": "string", "sources": [...], "retrieved_chunks": int, "confidence": float, "is_fallback": bool, "quality_score": float}`

**GET `/health`**
- **Description:** Health check endpoint
- **Response:** `{"status": "string", "message": "string", "index_stats": {...}}`

**GET `/stats`**
- **Description:** System statistics
- **Response:** Statistics about indexed documents and vector store

**GET `/chat`**
- **Description:** Interactive chat interface
- **Response:** HTML page with web-based chat interface

**GET `/docs`**
- **Description:** Interactive API documentation (Swagger UI)

## 🐛 Known Limitations & Future Improvements

### Current Limitations

1. **Limited Content:** Only 2 Help Center articles
   - Solution: Add web scraping to ingest full Help Center

2. **No Conversation History:** Each query is independent
   - Solution: Implement conversation memory with session management

3. **English Only:** No multi-language support
   - Solution: Add language detection and translation

4. **No User Personalization:** Same responses for all users
   - Solution: Incorporate user context and preferences

5. **Index Recreation on Startup:** Index always rebuilt on service restart
   - Solution: Implement conditional index rebuild based on content changes

### Proposed Improvements

#### Short-term (1-2 weeks)
- [ ] Implement selective index updates (only changed content)
- [ ] Add response caching layer (Redis)
- [ ] Improve error messages and logging
- [ ] Add query preprocessing (spell check, intent classification)

#### Medium-term (1-2 months)
- [ ] Implement feedback collection mechanism
- [ ] Add analytics dashboard
- [ ] Support file upload for custom articles
- [ ] Implement A/B testing framework

#### Long-term (3+ months)
- [ ] Multi-modal support (images, videos)
- [ ] Fine-tune embedding model on Typeform content
- [ ] Implement hybrid search (semantic + keyword)
- [ ] Build feedback loop for continuous improvement

## 💰 Cost Estimation

Based on typical usage:

**Per 1,000 queries:**
- Embeddings (query): $0.002
- Vector search: Free (up to 100K vectors)
- LLM generation: $0.15-0.30 (varies by response length)
- **Total: ~$0.15-0.30 per 1,000 queries**

**Monthly costs (10K queries/month):**
- OpenAI: ~$2-3
- Pinecone: Free (or $70 for paid tier if needed)
- Infrastructure: Varies by hosting choice

## 🤝 Contributing

This is a prototype project. For production use, consider:
- Adding authentication and rate limiting
- Implementing comprehensive error handling
- Adding extensive test coverage
- Setting up CI/CD pipeline
- Implementing monitoring and alerting

## 📝 License

This project was created as a technical case study.

## 📧 Questions?

For questions or clarifications, please reach out to Oana :) 

---

**Built with:** FastAPI, OpenAI, Pinecone, LangChain, Docker


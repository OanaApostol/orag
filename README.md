# Typeform Help Center RAG Chatbot

An AI-powered chatbot for Typeform's Help Center using Retrieval-Augmented Generation (RAG) to provide accurate, context-aware responses to user questions.

## 🎯 Project Overview

This project implements an end-to-end RAG solution that:
- ✅ Loads and preprocesses Help Center articles
- ✅ Chunks and embeds content for semantic search
- ✅ Stores embeddings in Pinecone vector database
- ✅ Retrieves relevant context for user queries
- ✅ Generates accurate responses using OpenAI's GPT models
- ✅ Exposes a REST API using FastAPI
- ✅ Containerized with Docker for easy deployment

## 🏗️ Architecture

```
User Query → FastAPI Endpoint → Vector Search (Pinecone) → Context Retrieval
                                                                ↓
User ← Response Generation (GPT-4o-mini) ← Prompt + Context ←┘
```

### Key Components

1. **Data Processor** (`app/data_processor.py`)
   - Loads and cleans Help Center articles
   - Implements semantic chunking strategy
   - Preserves metadata for better retrieval

2. **Vector Store** (`app/vector_store.py`)
   - Manages Pinecone integration
   - Generates embeddings using OpenAI
   - Handles semantic search queries

3. **RAG Engine** (`app/rag_engine.py`)
   - Orchestrates retrieval and generation
   - Implements prompt engineering
   - Handles fallback scenarios

4. **FastAPI Application** (`app/main.py`)
   - REST API with `/ask_question` endpoint
   - Health checks and statistics
   - Automatic index initialization

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key
- Pinecone API key (free tier is sufficient)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RAG
   ```

2. **Create environment file**
   ```bash
   cp env.example .env
   ```

3. **Add your API keys to `.env`**
   ```bash
   # Required
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=your_pinecone_environment_here
   
   # Optional (defaults provided)
   PINECONE_INDEX_NAME=typeform-help-center
   EMBEDDING_MODEL=text-embedding-3-large
   LLM_MODEL=gpt-4o-mini
   ```

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

**Health check:**
```bash
curl http://localhost:8000/health
```

## 📚 Data Management

This POC does not focus on automated data freshness. Articles are managed manually.

### Adding Help Center Articles

1. Download the Help Center article as HTML (File → Save Page As → Web Page, Complete)
2. Place the `.html` file into the `data/` folder in this repository

### Expected Folder Structure

```
data/
├── Article-Title-1 – Help Center.html
├── Article-Title-1 – Help Center_files/     # Assets folder (auto-created by the browser)
│   ├── images/
│   ├── stylesheets/
│   └── scripts/
├── Article-Title-2 – Help Center.html
└── Article-Title-2 – Help Center_files/
```

### Updating Articles

1. Replace the existing `.html` file in `data/` with the updated one
2. Restart the app to reindex

Note: When you add or update files in `data/`, you must re-embed to reflect changes in search.
The app only rebuilds when the Pinecone index is empty. To apply changes - temporarily set `force_recreate=True` in `app/main.py` and restart

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

**Reasoning**

- **Domain-specific accuracy:** The Typeform Help Center contains specialised terminology and concepts that require high-dimensional semantic understanding.

- **Similar questions, distinct meanings:** The larger model better distinguishes between similar words or phrases with different meanings (e.g., multiple questions in different languages on the same page).

- **Technical content handling:** The 3072-dimensional representation captures nuanced differences in technical tutorials, API documentation, and integration guides.

- **Validated through testing:** text-embedding-3-small delivered insufficient answer quality.

- **Semantic chunking dependency:** The FAQ and semantic chunking strategies rely on high-quality embeddings for accurate chunk boundary detection.

- **Customer support use case:** Prioritises answer accuracy and reliability over cost in production deployments.

**Model Comparison:**
- `text-embedding-3-small` (1536 dims): Lower cost but insufficient for technical Help Center content (tested and rejected)
- `text-embedding-3-large` (3072 dims): Superior semantic precision justified by improved answer quality ✓ CHOSEN

### 3. Vector Database

**Choice:** Pinecone

**Advantages:**
- **Managed service:** No infrastructure overhead
- **Serverless tier:** Perfect for prototyping
- **Fast queries:** Optimized for similarity search
- **Metadata filtering:** Supports filtering by article, date, etc.
- **Free tier sufficient:** 100K vectors for free

**Indexing Process**
Data Ingestion: HTML → BeautifulSoup → Clean text
Content Detection: FAQ vs. step-by-step vs. tutorial
Chunking: Different strategies per content type
Quality Filtering: Remove chunks below 0.3 score
Embedding: Convert to 3072-dim vectors
Metadata Enrichment: Add topics, difficulty, etc.
Pinecone Upload: Batch upsert with metadata

**Search Process**
```User Query
    ↓
[Classification] ← Is it Typeform-related? (LLM check)
    ↓ (Yes)
[Query Expansion] ← Generate 2-3 variations (LLM)
    ↓
[Dynamic Threshold] ← Set confidence threshold based on query type
    ↓
[Vector Embedding] ← Convert 3 queries to 3072-dim vectors
    ↓
[Pinecone Search] ← Find top-3 similar chunks for each variation
    ↓
[Deduplication] ← Remove duplicate chunks
    ↓
[Filtering] ← Keep only chunks above threshold
    ↓
[Context Building] ← Combine chunks into single context
    ↓
[LLM Generation] ← Generate response with dynamic temperature
    ↓
[Quality Scoring] ← Score response on multiple factors
    ↓
[Caching] ← Store for future identical queries
    ↓
Response to User```

**Key Features**

- **Semantic chunking** preserves content boundaries
- **Query expansion** catches paraphrased questions
- **Dynamic thresholds** adjust precision per query type
- **Source attribution** with relevance scores
- **Query caching** eliminates redundant searches

### 4. Language Model

**Choice:** GPT-4o-mini with Dynamic Temperature

**Reasoning:**
- **Cost-effective:** $0.15/1M input tokens (vs $5/1M for GPT-4)
- **Fast:** Lower latency than GPT-4
- **Sufficient capability:** Excellent for Q&A tasks
- **JSON mode support:** Structured outputs when needed

**Dynamic Temperature Strategy:**
- **Troubleshooting queries:** 0.2 (maximum accuracy for error resolution)
- **Factual questions:** 0.3 (high accuracy for how-to and what-is questions)
- **General questions:** 0.4 (balanced accuracy and natural conversation)

**Prompt Design:**
- System prompt establishes role and guidelines
- Clear instructions to avoid hallucination
- Emphasis on citing sources
- Conversational tone matching Typeform's brand

### 5. Retrieval Configuration

**Choice:** Top-3 results with dynamic similarity thresholds

**Reasoning:**
- **Top-3** provides enough context without overwhelming the LLM
- **Dynamic thresholds** (0.3-0.6) adjust based on query type for better recall/precision balance
  - Troubleshooting queries: 0.2 temperature for high accuracy
  - Specific questions (how/what/when/where): 0.4 similarity threshold
  - General questions (can i/is it possible/does it): 0.3 similarity threshold
  - Default: 0.5 similarity threshold
- Fallback response when no good matches found
- Tuned through testing with sample queries

## 🔍 Evaluation Strategy

### Quality Metrics

1. **Retrieval Metrics**
   - **Relevance Score:** Cosine similarity of retrieved chunks (target: >0.6 for specific questions, >0.4 for general)
   - **Coverage:** % of queries with at least one relevant result (target: >90%)
   - **Precision@3:** Are top-3 results relevant? (target: >80%)

2. **Response Quality Metrics**
   - **Accuracy:** Does answer match article content? (manual evaluation)
   - **Completeness:** Does answer fully address the question? (1-5 scale)
   - **Hallucination Rate:** % of responses with unsupported claims (target: <5%)
   - **Citation Accuracy:** Do citations match content? (target: 100%)

3. **User Experience Metrics**
   - **Response Time:** End-to-end latency (target: <3 seconds)
   - **Fallback Rate:** % of queries triggering fallback (target: <10%)
   - **Confidence Score:** Average retrieval confidence (target: >0.75)

### Evaluation Approach

**Phase 1: Offline Evaluation**
- Created test set of 20 representative questions (see `EVALUATION.md`)
- Manual review of answers for accuracy and relevance
- Measured retrieval and response quality metrics

**Phase 2: Stress Testing**
- Edge cases: ambiguous questions, out-of-scope queries
- Adversarial examples: misleading questions, prompt injections
- Robustness: typos, different phrasings

**Phase 3: A/B Testing (Production)**
- Compare against keyword search baseline
- Measure user satisfaction (thumbs up/down)
- Track query reformulation rate

### Observability in Production

For production deployment, implement:

1. **Logging & Monitoring**
   - Log all queries and responses
   - Track latency, error rates, fallback rates
   - Monitor API usage and costs

2. **Quality Tracking**
   - User feedback collection (thumbs up/down)
   - Manual review of flagged responses
   - Regular review of low-confidence queries

3. **Continuous Evaluation**
   - Automated tests on curated question set
   - Weekly manual review of random samples
   - Track metrics trends over time

4. **Feedback Loop**
   - Collect user corrections/feedback
   - Use feedback to improve prompts and chunking
   - Retrain or fine-tune models as needed

## 📈 Performance & Scalability

### Current Performance
- **Embedding generation:** ~1-2 seconds for query
- **Vector search:** <100ms in Pinecone
- **LLM generation:** ~2-3 seconds
- **Total latency:** ~3-5 seconds

### Scaling Considerations

**For 1K users/day:**
- Current setup is sufficient
- Free tier of Pinecone handles this easily
- Cost: ~$0.50-1/day (mainly LLM calls)

**For 10K+ users/day:**
- Consider caching frequent queries (Redis)
- Implement rate limiting and authentication
- Upgrade to Pinecone paid tier
- Use GPT-4o-mini batch API for lower costs
- Consider streaming responses for better UX

**For production:**
- Add CDN for API caching
- Implement request queuing for high load
- Use connection pooling for Pinecone
- Add redundancy and failover
- Monitor and auto-scale based on traffic

## 🔧 Configuration

Configuration is managed through the `config.yaml` file:

```yaml
# Typeform RAG Chatbot Configuration

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

| Setting | Description | Default |
|---------|-------------|---------|
| `openai.api_key` | OpenAI API key | **Required** |
| `pinecone.api_key` | Pinecone API key | **Required** |
| `embedding.model` | OpenAI embedding model | `text-embedding-3-large` |
| `embedding.dimension` | Embedding dimensions | `3072` |
| `llm.model` | OpenAI LLM model | `gpt-4o-mini` |
| `llm.temperature` | Base LLM temperature (overridden by query type) | `0.25` |
| `llm.max_tokens` | Max tokens in response | `500` |
| `retrieval.top_k_results` | Number of chunks to retrieve | `3` |
| `retrieval.chunk_size` | Chunk size in characters | `800` |
| `retrieval.chunk_overlap` | Chunk overlap in characters | `200` |
| `chunking.use_semantic_chunking` | Enable semantic chunking | `true` |
| `chunking.semantic_threshold_percentile` | Semantic similarity threshold | `95` |
| `chunking.min_chunk_quality_score` | Minimum chunk quality score | `0.3` |
| `chunking.enable_content_aware_chunking` | Enable content-type detection | `true` |
| `chunking.dynamic_chunk_sizing` | Enable dynamic chunk sizing | `true` |

## 🧪 Testing

### Run test queries
```bash
docker-compose exec rag-api python scripts/test_query.py
```

### Test enhanced chunking
```bash
docker-compose exec rag-api python scripts/test_enhanced_chunking.py
```

### Test dynamic temperature
```bash
docker-compose exec rag-api python scripts/test_dynamic_temperature.py
```

### Manually setup index
```bash
docker-compose exec rag-api python scripts/setup_index.py
```

## 🐛 Known Limitations & Future Improvements

### Current Limitations

1. **Limited Content:** Only 2 Help Center articles
   - Solution: Add web scraping to ingest full Help Center

2. **No Conversation History:** Each query is independent
   - Solution: Implement conversation memory with session management

3. **English Only:** No multi-language support
   - Solution: Add language detection and translation

4. **Basic Error Handling:** Limited fallback options
   - Solution: Implement more sophisticated error recovery

5. **No User Personalization:** Same responses for all users
   - Solution: Incorporate user context and preferences

### Proposed Improvements

#### Short-term (1-2 weeks)
- [ ] Add conversation history tracking
- [ ] Implement response caching
- [ ] Add more comprehensive test suite
- [ ] Improve error messages and logging
- [ ] Add query preprocessing (spell check, intent classification)

#### Medium-term (1-2 months)
- [ ] Implement feedback collection mechanism
- [ ] Add analytics dashboard
- [ ] Support file upload for custom articles
- [ ] Implement A/B testing framework
- [ ] Add query reformulation suggestions

#### Long-term (3+ months)
- [ ] Multi-modal support (images, videos)
- [ ] Fine-tune embedding model on Typeform content
- [ ] Implement hybrid search (semantic + keyword)
- [ ] Add support for complex reasoning chains
- [ ] Build feedback loop for continuous improvement

### Product Ideation

**Enhanced Features:**
1. **Proactive Suggestions:** Suggest related articles before user asks
2. **Visual Tutorials:** Return embedded GIFs/videos from articles
3. **Contextual Help:** Integrate directly into Typeform builder UI
4. **Community Answers:** Incorporate community forum discussions
5. **Personalized Responses:** Tailor based on user's form type and complexity

**Integration Opportunities:**
- Slack/Discord bot for team support
- Chrome extension for in-context help
- Typeform builder sidebar integration
- Email response automation for support team

## 🏗️ Project Structure

```
RAG/
├── app/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── data_processor.py      # Data ingestion and chunking
│   ├── vector_store.py        # Pinecone integration
│   ├── rag_engine.py          # RAG orchestration
│   └── main.py                # FastAPI application
├── data/
│   ├── __init__.py
│   └── help_articles.py       # Help Center content
├── scripts/
│   ├── setup_index.py         # Index initialization script
│   └── test_query.py          # Testing script
├── notebooks/                  # Jupyter notebooks for exploration
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── requirements.txt            # Python dependencies
├── env.example                 # Example environment variables
├── .gitignore
├── README.md                   # This file
└── EVALUATION.md               # Detailed evaluation results

```

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

For questions or clarifications, please reach out to the Typeform Data Science team.

---

**Built with:** FastAPI, OpenAI, Pinecone, LangChain, Docker


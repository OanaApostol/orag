# RAG Evaluation Script

A streamlined, lightweight evaluation script for measuring RAG chatbot performance without modifying the existing codebase.

## 📁 **Files**

- `evaluate_rag.py` - Main evaluation script
- `test_questions.json` - Sample test data
- `evaluation_report.json` - Latest evaluation results
- `EVALUATION_README.md` - This documentation

## 🎯 **Features**

### **Retrieval Metrics**
- **Precision@k** - Accuracy of top-k retrieved results
- **Recall@k** - Coverage of relevant results in top-k
- **MRR** - Mean Reciprocal Rank (ranking quality)
- **NDCG@k** - Normalized Discounted Cumulative Gain

### **Generation Metrics**
- **BLEU Score** - N-gram overlap with ground truth
- **ROUGE-L** - Longest common subsequence F1
- **Semantic Similarity** - Word overlap similarity
- **Quality Score** - Composite quality metric

### **System Metrics**
- **Response Time** - End-to-end latency
- **Confidence Score** - RAG system confidence

## 🚀 **Quick Start**

### **Option 1: Run Complete Example**
```bash
# Navigate to evaluation directory
cd evaluation
python evaluate_rag.py --example
```

### **Option 2: Step by Step**
```bash
# Navigate to evaluation directory
cd evaluation

# 1. Create sample test data
python evaluate_rag.py --create_sample

# 2. Run evaluation
python evaluate_rag.py --test_data test_questions.json --output evaluation_report.json
```

### **3. View Results**
The script outputs a summary to console and saves detailed results to JSON.

## 📊 **Sample Results**

```
============================================================
RAG EVALUATION SUMMARY
============================================================
Total Questions: 5
Average Response Time: 6.59s
Average Confidence: 0.377

RETRIEVAL METRICS:
  precision@1: 0.000 ± 0.000
  recall@1: 0.000 ± 0.000
  ndcg@1: 0.000 ± 0.000
  precision@3: 0.000 ± 0.000
  recall@3: 0.000 ± 0.000
  ndcg@3: 0.000 ± 0.000
  precision@5: 0.000 ± 0.000
  recall@5: 0.000 ± 0.000
  ndcg@5: 0.000 ± 0.000
  mrr: 0.000 ± 0.000

GENERATION METRICS:
  bleu: 0.134 ± 0.038
  rouge_l: 0.132 ± 0.036
  semantic_sim: 0.117 ± 0.044
  quality: 0.153 ± 0.022
```

## 📝 **Test Data Format**

Create a JSON file with test questions and ground truth answers:

```json
[
  {
    "question": "How do I create a form in multiple languages?",
    "ground_truth": "To create a multilingual form, go to Settings > Languages and select the languages you want to support.",
    "relevant_chunks": [
      "To create a multilingual form, go to Settings > Languages",
      "Select the languages you want to support",
      "Translate your questions and options for each language"
    ]
  }
]
```

## 🔧 **Usage Options**

```bash
# Run complete example (recommended)
python evaluate_rag.py --example

# Basic evaluation
python evaluate_rag.py

# Custom test data
python evaluate_rag.py --test_data my_questions.json

# Custom output file
python evaluate_rag.py --output my_report.json

# Create sample data only
python evaluate_rag.py --create_sample

# Help
python evaluate_rag.py --help
```

## 📈 **Understanding Metrics**

### **Retrieval Quality**
- **Precision@k**: Of the top-k results, how many are relevant?
- **Recall@k**: Of all relevant results, how many are in top-k?
- **MRR**: How quickly do we find the first relevant result?
- **NDCG@k**: How well-ranked are the results?

### **Generation Quality**
- **BLEU**: Word-level similarity to ground truth
- **ROUGE-L**: Sentence structure similarity
- **Semantic Similarity**: Meaning overlap
- **Quality Score**: Composite metric (0-1 scale)

### **Performance**
- **Response Time**: Total latency (embedding + search + generation)
- **Confidence**: RAG system's confidence in the answer
- **Error Rate**: Percentage of failed responses

## 🎯 **Interpretation Guidelines**

### **Good Performance**
- **Precision@3 > 0.7**: Most top-3 results are relevant
- **Recall@3 > 0.6**: Most relevant content is retrieved
- **MRR > 0.5**: First relevant result appears early
- **BLEU > 0.3**: Generated answers match ground truth well
- **Response Time < 5s**: Fast enough for real-time use

### **Areas for Improvement**
- **Low Precision**: Retrieval threshold too low
- **Low Recall**: Retrieval threshold too high
- **Low BLEU**: Generation quality needs improvement
- **High Response Time**: Optimization needed

## 🔍 **Troubleshooting**

### **Common Issues**

1. **All retrieval metrics = 0**
   - Check if Pinecone index has data
   - Verify embedding model configuration
   - Ensure test questions are Typeform-related

2. **Low generation metrics**
   - Review ground truth quality
   - Check if retrieved context is relevant
   - Verify LLM temperature settings

3. **High response times**
   - Check API rate limits
   - Verify network connectivity
   - Consider caching strategies

### **Debug Mode**
Add `--verbose` flag for detailed logging:
```bash
python evaluate_rag.py --test_data test_questions.json --verbose
```

## 📋 **Requirements**

- Python 3.8+
- Existing RAG codebase (app/ directory)
- OpenAI API key configured
- Pinecone API key configured

**No additional dependencies required** - uses only standard library and existing RAG components.

## 🎨 **Customization**

### **Adding New Metrics**
Edit `evaluate_rag.py` and add new methods to `RAGEvaluator` class:

```python
def calculate_custom_metric(self, reference: str, candidate: str) -> float:
    # Your custom metric implementation
    return score

def evaluate_generation(self, ground_truth: str, predicted: str) -> Dict[str, float]:
    # Add to existing metrics
    return {
        **existing_metrics,
        'custom_metric': self.calculate_custom_metric(ground_truth, predicted)
    }
```

### **Modifying Thresholds**
Adjust relevance thresholds in `evaluate_retrieval()`:
```python
if sim > 0.3:  # Change this threshold
    relevant_count += 1
```

## 📊 **Report Structure**

The generated JSON report contains:

```json
{
  "summary": {
    "total_questions": 5,
    "avg_response_time": 8.66,
    "avg_confidence": 0.489
  },
  "retrieval_metrics": {
    "precision@1": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
    "recall@1": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
    "mrr": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
    "ndcg@1": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
  },
  "generation_metrics": {
    "bleu": {"mean": 0.074, "std": 0.027, "min": 0.0, "max": 0.1},
    "rouge_l": {"mean": 0.124, "std": 0.044, "min": 0.0, "max": 0.2},
    "semantic_similarity": {"mean": 0.122, "std": 0.037, "min": 0.0, "max": 0.2},
    "quality_score": {"mean": 0.123, "std": 0.016, "min": 0.1, "max": 0.15}
  },
  "detailed_results": [
    {
      "question": "...",
      "ground_truth": "...",
      "predicted_answer": "...",
      "confidence": 0.489,
      "response_time": 8.66,
      "retrieval_scores": {...},
      "generation_scores": {...}
    }
  ]
}
```

## 🚀 **Next Steps**

1. **Expand Test Dataset**: Add more diverse questions
2. **A/B Testing**: Compare different configurations
3. **Continuous Evaluation**: Run regularly to track performance
4. **Benchmarking**: Compare against other RAG systems
5. **Optimization**: Use results to improve retrieval/generation

---

**Built for**: Typeform RAG Chatbot Evaluation  
**Dependencies**: None (uses existing codebase)  
**Output**: JSON report + console summary

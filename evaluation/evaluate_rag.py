#!/usr/bin/env python3
"""
Optimized RAG Evaluation Script

A streamlined evaluation script that measures RAG performance with minimal code overhead.
Combines evaluation and example functionality in a single, efficient script.

Usage:
    python evaluate_rag.py --create_sample
    python evaluate_rag.py --test_data test_questions.json
    python evaluate_rag.py --example
"""

import json
import argparse
import time
import sys
import math
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import logging

# Import RAG components
try:
    import sys
    from pathlib import Path
    # Add parent directory to path to import app modules
    sys.path.append(str(Path(__file__).parent.parent))
    
    from app.config import get_settings
    from app.data_processor import DataProcessor
    from app.vector_store import VectorStore
    from app.rag_engine import RAGEngine
except ImportError as e:
    print(f"Could not import RAG components: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    question: str
    ground_truth: str
    predicted_answer: str
    retrieval_scores: Dict[str, float]
    generation_scores: Dict[str, float]
    response_time: float
    confidence: float


class RAGEvaluator:
    """Streamlined RAG evaluation with optimized metrics calculation"""
    
    def __init__(self, config_path: str = None):
        """Initialize evaluator with RAG components"""
        logger.info("Initializing RAG components...")
        
        # Use config from parent directory if not specified
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "config.yaml")
        
        self.settings = get_settings(config_path)
        self.data_processor = DataProcessor(self.settings)
        self.vector_store = VectorStore(self.settings)
        self.vector_store.initialize_index(force_recreate=False)
        self.rag_engine = RAGEngine(self.vector_store, self.settings)
        logger.info("Evaluation setup complete!")
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.sub(r'[^\w\s]', ' ', text.lower()).split()
    
    def calculate_metrics(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate all generation metrics in one pass"""
        ref_tokens = self.tokenize(reference)
        cand_tokens = self.tokenize(candidate)
        
        if not ref_tokens or not cand_tokens:
            return {'bleu': 0.0, 'rouge_l': 0.0, 'semantic_sim': 0.0, 'quality': 0.0}
        
        # BLEU (simplified 1-2 gram)
        ref_set = set(ref_tokens)
        cand_set = set(cand_tokens)
        overlap = len(ref_set & cand_set)
        bleu = overlap / len(cand_set) if cand_set else 0.0
        
        # ROUGE-L (LCS approximation)
        lcs = self._lcs_length(ref_tokens, cand_tokens)
        rouge_l = 2 * lcs / (len(ref_tokens) + len(cand_tokens)) if ref_tokens and cand_tokens else 0.0
        
        # Semantic similarity (Jaccard)
        semantic_sim = overlap / len(ref_set | cand_set) if (ref_set | cand_set) else 0.0
        
        # Quality score (composite)
        length_ratio = min(len(ref_tokens), len(cand_tokens)) / max(len(ref_tokens), len(cand_tokens))
        quality = (bleu * 0.3 + rouge_l * 0.3 + semantic_sim * 0.2 + length_ratio * 0.2)
        
        return {
            'bleu': bleu,
            'rouge_l': rouge_l,
            'semantic_sim': semantic_sim,
            'quality': quality
        }
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate LCS length efficiently"""
        m, n = len(seq1), len(seq2)
        if m == 0 or n == 0:
            return 0
        
        # Use only two rows for space efficiency
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev, curr = curr, prev
        
        return prev[n]
    
    def evaluate_retrieval(self, retrieved_chunks: List[Dict], ground_truth_chunks: List[str]) -> Dict[str, float]:
        """Streamlined retrieval evaluation using actual Pinecone scores"""
        if not retrieved_chunks:
            return {f'{metric}@{k}': 0.0 for metric in ['precision', 'recall', 'ndcg'] for k in [1, 3, 5]} | {'mrr': 0.0}
        
        # Use actual Pinecone similarity scores instead of recalculating
        similarities = [chunk.get('relevance_score', chunk.get('score', 0.0)) for chunk in retrieved_chunks]
        
        # Calculate all metrics at once
        metrics = {}
        for k in [1, 3, 5]:
            top_k_sims = similarities[:k]
            relevant_count = sum(1 for sim in top_k_sims if sim > 0.5)
            
            metrics[f'precision@{k}'] = relevant_count / k if k > 0 else 0.0
            metrics[f'recall@{k}'] = relevant_count / len(ground_truth_chunks) if ground_truth_chunks else 0.0
            
            # NDCG
            dcg = sum((1 if sim > 0.5 else 0) / math.log2(i + 2) for i, sim in enumerate(top_k_sims))
            idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(ground_truth_chunks))))
            metrics[f'ndcg@{k}'] = dcg / idcg if idcg > 0 else 0.0
        
        # MRR
        mrr = 0.0
        for i, sim in enumerate(similarities):
            if sim > 0.5:
                mrr = 1.0 / (i + 1)
                break
        metrics['mrr'] = mrr
        
        return metrics
    
    def evaluate_single(self, question_data: Dict) -> EvaluationResult:
        """Evaluate a single question efficiently"""
        question = question_data['question']
        ground_truth = question_data['ground_truth']
        ground_truth_chunks = question_data.get('relevant_chunks', [ground_truth])
        
        logger.info(f"Evaluating: {question[:50]}...")
        
        start_time = time.time()
        
        try:
            result = self.rag_engine.generate_response(question)
            predicted_answer = result['answer']
            retrieved_chunks = result.get('sources', [])
            confidence = result.get('confidence', 0.0)
        except Exception as e:
            logger.error(f"Error: {e}")
            predicted_answer = "Error generating response"
            retrieved_chunks = []
            confidence = 0.0
        
        response_time = time.time() - start_time
        
        # Calculate all metrics
        retrieval_scores = self.evaluate_retrieval(retrieved_chunks, ground_truth_chunks)
        generation_scores = self.calculate_metrics(ground_truth, predicted_answer)
        
        return EvaluationResult(
            question=question,
            ground_truth=ground_truth,
            predicted_answer=predicted_answer,
            retrieval_scores=retrieval_scores,
            generation_scores=generation_scores,
            response_time=response_time,
            confidence=confidence
        )
    
    def evaluate_all(self, test_data: List[Dict]) -> List[EvaluationResult]:
        """Evaluate all questions"""
        results = []
        for i, question_data in enumerate(test_data):
            logger.info(f"Progress: {i+1}/{len(test_data)}")
            results.append(self.evaluate_single(question_data))
        return results
    
    def generate_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate concise report"""
        if not results:
            return {'error': 'No results to report'}
        
        # Aggregate metrics efficiently
        all_retrieval = {}
        all_generation = {}
        response_times = []
        confidences = []
        
        for result in results:
            for key, value in result.retrieval_scores.items():
                all_retrieval.setdefault(key, []).append(value)
            for key, value in result.generation_scores.items():
                all_generation.setdefault(key, []).append(value)
            response_times.append(result.response_time)
            confidences.append(result.confidence)
        
        def stats(values):
            if not values:
                return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return {
                'mean': mean,
                'std': math.sqrt(variance),
                'min': min(values),
                'max': max(values)
            }
        
        return {
            'summary': {
                'total_questions': len(results),
                'avg_response_time': stats(response_times)['mean'],
                'avg_confidence': stats(confidences)['mean']
            },
            'retrieval_metrics': {k: stats(v) for k, v in all_retrieval.items()},
            'generation_metrics': {k: stats(v) for k, v in all_generation.items()},
            'detailed_results': [
                {
                    'question': r.question,
                    'ground_truth': r.ground_truth,
                    'predicted_answer': r.predicted_answer,
                    'confidence': r.confidence,
                    'response_time': r.response_time,
                    'retrieval_scores': r.retrieval_scores,
                    'generation_scores': r.generation_scores
                }
                for r in results
            ]
        }


def create_sample_data():
    """Create sample test data"""
    sample_data = [
        {
            "question": "How do I create a form in multiple languages?",
            "ground_truth": "To create a multilingual form, go to Settings > Languages and select the languages you want to support. You can then translate your questions and options for each language.",
            "relevant_chunks": [
                "To create a multilingual form, go to Settings > Languages",
                "Select the languages you want to support",
                "Translate your questions and options for each language"
            ]
        },
        {
            "question": "What is a Multi-Question Page?",
            "ground_truth": "A Multi-Question Page allows you to group multiple questions on a single page, making your form more compact and easier to navigate for respondents.",
            "relevant_chunks": [
                "Multi-Question Page allows you to group multiple questions",
                "Group multiple questions on a single page",
                "Makes your form more compact and easier to navigate"
            ]
        },
        {
            "question": "How do I integrate Typeform with Zapier?",
            "ground_truth": "To integrate Typeform with Zapier, go to the Integrations section in your form settings, search for Zapier, and follow the connection process to link your accounts.",
            "relevant_chunks": [
                "Go to the Integrations section in your form settings",
                "Search for Zapier in integrations",
                "Follow the connection process to link accounts"
            ]
        },
        {
            "question": "Can I edit the AI translation?",
            "ground_truth": "Yes, you can edit AI translations by going to the translation interface and manually adjusting any automatically translated content.",
            "relevant_chunks": [
                "You can edit AI translations",
                "Go to the translation interface",
                "Manually adjust automatically translated content"
            ]
        },
        {
            "question": "What languages are available in the translations settings?",
            "ground_truth": "Typeform supports over 50 languages including English, Spanish, French, German, Italian, Portuguese, Dutch, and many others in the translation settings.",
            "relevant_chunks": [
                "Typeform supports over 50 languages",
                "English, Spanish, French, German, Italian, Portuguese, Dutch",
                "Many other languages in translation settings"
            ]
        }
    ]
    
    with open('test_questions.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("✅ Created sample test data: evaluation/test_questions.json")


def run_example():
    """Run complete evaluation example"""
    print("🚀 RAG Evaluation Example")
    print("=" * 50)
    
    # Create sample data
    print("\n1. Creating sample test data...")
    create_sample_data()
    
    # Run evaluation
    print("\n2. Running evaluation...")
    config_path = str(Path(__file__).parent.parent / "config.yaml")
    evaluator = RAGEvaluator(config_path)
    
    with open('test_questions.json', 'r') as f:
        test_data = json.load(f)
    
    logger.info("Starting evaluation...")
    results = evaluator.evaluate_all(test_data)
    
    # Generate and display report
    print("\n3. Generating report...")
    report = evaluator.generate_report(results)
    
    # Save report
    with open('example_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Display summary
    print("\n" + "="*60)
    print("RAG EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Questions: {report['summary']['total_questions']}")
    print(f"Average Response Time: {report['summary']['avg_response_time']:.2f}s")
    print(f"Average Confidence: {report['summary']['avg_confidence']:.3f}")
    
    print("\nRETRIEVAL METRICS:")
    for metric, stats in report['retrieval_metrics'].items():
        print(f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    print("\nGENERATION METRICS:")
    for metric, stats in report['generation_metrics'].items():
        print(f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    print(f"\n✨ Detailed report saved to: example_report.json")


def main():
    parser = argparse.ArgumentParser(description='Optimized RAG evaluation script')
    parser.add_argument('--config', default='../config.yaml', help='Path to config file')
    parser.add_argument('--test_data', default='test_questions.json', help='Path to test data JSON file')
    parser.add_argument('--output', default='evaluation_report.json', help='Output report file')
    parser.add_argument('--create_sample', action='store_true', help='Create sample test data')
    parser.add_argument('--example', action='store_true', help='Run complete example')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data()
        return
    
    if args.example:
        run_example()
        return
    
    # Regular evaluation
    if not Path(args.test_data).exists():
        print(f"Test data file {args.test_data} not found.")
        print("Use --create_sample to create sample test data, or --example to run complete example.")
        return
    
    # Run evaluation
    evaluator = RAGEvaluator(args.config)
    
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    
    logger.info("Starting evaluation...")
    results = evaluator.evaluate_all(test_data)
    
    # Generate report
    logger.info("Generating report...")
    report = evaluator.generate_report(results)
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("RAG EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Questions: {report['summary']['total_questions']}")
    print(f"Average Response Time: {report['summary']['avg_response_time']:.2f}s")
    print(f"Average Confidence: {report['summary']['avg_confidence']:.3f}")
    
    print("\nRETRIEVAL METRICS:")
    for metric, stats in report['retrieval_metrics'].items():
        print(f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    print("\nGENERATION METRICS:")
    for metric, stats in report['generation_metrics'].items():
        print(f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    print(f"\nDetailed report saved to: {args.output}")


if __name__ == "__main__":
    main()
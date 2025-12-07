#!/usr/bin/env python3
"""
🎯 MULTIHOP-RAG EVALUATION SYSTEM
Version: 1.0 (Production)
Model: Ollama llama3.1:8b
Dataset: MultiHop-RAG (2556 queries)

USAGE:
- Test mode (fast): python run_everything.py
- Full mode (paper): Set TEST_MODE = False
"""

import json
import requests
from pathlib import Path
from collections import defaultdict
import time
import re
# ==================== CONFIGURATION ====================
TEST_MODE = False          # Set to False for full evaluation (2556 queries)
MODEL_NAME = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"
BASELINE_ACC = 0.56       # GPT-4 baseline from Tang et al. (2024)
# ======================================================

class MultiHopEvaluator:
    def __init__(self, model=MODEL_NAME):
        self.model = model
        self.ollama_url = OLLAMA_URL
        print(f"✅ Initialized evaluator with {self.model}")
        
    def load_data(self):
        """Load MultiHop-RAG dataset"""
        print("\n📂 Loading MultiHop-RAG dataset...")
        
        # Load queries
        with open("data/multihoprag.json") as f:
            data = json.load(f)
            self.queries = list(data.values()) if isinstance(data, dict) else data
            
            # Apply test mode filter
            if TEST_MODE:
                self.queries = self.queries[:100]
                print(f"  🔬 TEST MODE: {len(self.queries)} queries")
            else:
                print(f"  📊 FULL MODE: {len(self.queries)} queries")
        
        # Load corpus
        with open("data/corpus.json") as f:
            corpus = json.load(f)
            self.corpus = [{"id": k, **v} for k, v in corpus.items()] if isinstance(corpus, dict) else corpus
        
        print(f"📚 {len(self.corpus)} documents loaded")
        return self
    
    def retrieve(self, query: str, k: int = 4):
        """Hybrid retrieval: keyword + entity matching"""
        # Extract key terms (proper nouns, numbers, long words)
        words = [w.lower() for w in query.split() if len(w) > 3]
        entities = re.findall(r'\b[A-Z][a-zA-Z]+\b', query)
        key_terms = words + [e.lower() for e in entities]
        
        # Score documents
        scored = []
        for doc in self.corpus:
            text = doc.get('text', '').lower()
            title = doc.get('title', '').lower()
            
            # Term frequency scoring
            text_score = sum(1 for t in key_terms if t in text)
            title_score = sum(2 for t in key_terms if t in title)
            
            # Entity bonus (exact phrase match)
            entity_score = 0
            for entity in entities:
                if entity.lower() in text or entity.lower() in title:
                    entity_score += 5
            
            total_score = text_score + title_score + entity_score
            
            if total_score > 0:
                scored.append((doc, total_score))
        
        # Fallback: return top-K by any match
        if not scored:
            print(f"  ⚠️ Weak match for: {query[:50]}... (fallback)")
            return self.corpus[:k]
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:k]]
    
    def generate(self, query: str, evidence: list) -> str:
        """Generate answer using Ollama"""
        # Format evidence
        evidence_str = "\n\n".join([
            f"[Doc {i+1}] {doc.get('text', '')[:500]}..."
            for i, doc in enumerate(evidence)
        ])
        
        # Prompt that forces short answers
        prompt = f"""Documents:\n{evidence_str}\n\nQuestion: {query}\nAnswer (1-5 words): """
        
        try:
            response = requests.post(self.ollama_url, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 15}
            }, timeout=60)
            
            raw = response.json()['response'].strip()
            
            # Clean answer
            answer = raw.split('.')[0].strip()
            answer = answer.split('\n')[0].strip()
            
            # If too long, extract first entity
            if len(answer.split()) > 5:
                entities = re.findall(r'\b[A-Z][a-zA-Z]+\b', answer)
                if entities:
                    answer = " ".join(entities[:3])
            
            return answer
            
        except Exception as e:
            print(f"  ⚠️ Generation error: {e}")
            return "Insufficient Information"
    
    def match(self, pred: str, exp: str) -> bool:
        """Flexible answer matching"""
        pred = pred.strip().lower()
        exp = exp.strip().lower()
        
        # Exact match
        if pred == exp:
            return True
        
        # Yes/No
        if exp in ['yes', 'no'] and pred in ['yes', 'no']:
            return pred == exp
        
        # Substring match
        if exp in pred or pred in exp:
            return True
        
        # Overlap > 50%
        pred_words = set(pred.split())
        exp_words = set(exp.split())
        if pred_words and exp_words:
            overlap = len(pred_words & exp_words) / len(exp_words)
            return overlap > 0.5
        
        return False
    
    def evaluate(self):
        """Run full evaluation"""
        print(f"\n🎯 Evaluating {len(self.queries)} queries...")
        
        correct = 0
        type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for i, q in enumerate(self.queries, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(self.queries)}")
            
            # Retrieve and generate
            retrieved = self.retrieve(q['query'], k=4)
            answer = self.generate(q['query'], retrieved)
            
            # Check correctness
            is_correct = self.match(answer, q['answer'])
            if is_correct:
                correct += 1
            
            # Track per-type
            qtype = q.get('type', 'unknown')
            type_stats[qtype]['total'] += 1
            if is_correct:
                type_stats[qtype]['correct'] += 1
            
            # Brief pause for stability
            time.sleep(0.02)
        
        # Calculate metrics
        accuracy = correct / len(self.queries)
        improvement = ((accuracy / BASELINE_ACC) - 1) * 100
        
        # Display results
        print("\n" + "="*70)
        print("🏆 FINAL EVALUATION RESULTS")
        print("="*70)
        print(f"📊 Accuracy: {accuracy:.2%} ({correct}/{len(self.queries)})")
        print(f"📈 vs GPT-4 Baseline: +{improvement:+.1f}% relative improvement")
        print(f"🤖 Model: {self.model}")
        print(f"📋 Mode: {'TEST (100 queries)' if TEST_MODE else 'FULL (2556 queries)'}")
        
        # Per-type performance
        if len(type_stats) > 1:
            print(f"\n📋 Per-Type Performance:")
            for qtype, stats in sorted(type_stats.items()):
                if stats['total'] > 0:
                    acc = stats['correct'] / stats['total']
                    print(f"  {qtype:<12}: {acc:.1%} ({stats['correct']}/{stats['total']})")
        
        # Save results
        Path("results").mkdir(exist_ok=True)
        output_file = "results/test_results.json" if TEST_MODE else "results/full_results.json"
        
        with open(output_file, "w") as f:
            json.dump({
                "metadata": {
                    "model": self.model,
                    "mode": "test" if TEST_MODE else "full",
                    "queries_evaluated": len(self.queries),
                    "baseline_accuracy": BASELINE_ACC
                },
                "metrics": {
                    "accuracy": accuracy,
                    "correct_count": correct,
                    "relative_improvement_percent": improvement
                },
                "per_type_accuracy": {k: v['correct']/v['total'] for k,v in type_stats.items()},
                "configuration": {
                    "retriever": "hybrid_keyword_entity",
                    "generator": "forced_short_answers",
                    "k_retrieved": 4
                }
            }, f, indent=2)
        
        print(f"\n💾 Results saved to {output_file}")
        return accuracy, improvement

def main():
    """Main execution"""
    print("="*70)
    print("🚀 MULTIHOP-RAG EVALUATION SYSTEM")
    print("="*70)
    
    evaluator = MultiHopEvaluator()
    evaluator.load_data().evaluate()

if __name__ == "__main__":
    main()

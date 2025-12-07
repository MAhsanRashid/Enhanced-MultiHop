# Enhanced MultiHop-RAG

Open-source implementation of multi-hop retrieval-augmented generation.

## Results
- **Accuracy**: 33.3% on full MultiHop-RAG dataset
- **Cost**: $0 (vs $76.68 for GPT-4 baseline)
- **Model**: LLaMA 3.1 8B
- **Novelty**: Graph-enhanced retrieval, confidence-aware generation, zero-cost deployment

## Usage
```bash
# Full evaluation
python src/evaluation.py

# Test on 100 queries
TEST_MODE=True python src/evaluation.py

| Metric      | Baseline (GPT-4) | Ours         | Advantage         |
| ----------- | ---------------- | ------------ | ----------------- |
| Accuracy    | 56.0%            | 33.3%        | Deployable & free |
| Cost        | \$0.03/query     | \$0.00/query | 100% savings      |
| Latency     | 2.1s             | 1.8s         | 14% faster        |
| Open Source | No               | Yes          | Reproducible      |

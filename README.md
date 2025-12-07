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

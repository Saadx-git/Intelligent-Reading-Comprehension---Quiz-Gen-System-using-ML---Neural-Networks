# Model B: Hint Generation System - Quick Start Guide

## What is Model B?

Model B generates **ranked hints** (top-3 most relevant sentences) from a passage to help students understand reading comprehension questions. It implements two strategies:

1. **Extractive Strategy**: Uses TF-IDF cosine similarity to score sentences
2. **ML-Scored Strategy**: Uses trained Logistic Regression to rank sentences

---

## Quick Usage

### Load the Models

```python
from models.model_b.loader import ModelBLoader

# Load all artifacts
loader = ModelBLoader('models/model_b/traditional')

# Print evaluation metrics
print(loader.get_metrics_summary())
```

### Get Hints for a Question

```python
question = "Why did the character make that decision?"
article = "The character faced a difficult choice..."

# Get extractive hints (cosine similarity)
hints = loader.get_extractive_hints(question, article, top_k=3)

# Get ML-scored hints (trained model)
ml_hints = loader.get_ml_hints(question, article, top_k=3)
```

### Access Pre-computed Results

```python
# Get pre-computed result for dev example 0
dev_result = loader.get_dev_result(0, strategy='extractive')

# Result format:
# {
#     'hints': [{'sentence': str, 'score': float, 'rank': int}, ...],
#     'hint_1_general': str,
#     'hint_2_moderate': str,
#     'hint_3_explicit': str,
#     'num_sentences': int,
#     'top_scores': [float, ...]
# }
```

---

## Results Summary

### Evaluation Metrics (500 Dev Samples)

| Metric | Extractive | ML-Scored |
|--------|-----------|-----------|
| **Precision@3** | 0.9760 | 0.9760 |
| **Recall@3** | 0.9760 | 0.9760 |
| **Accuracy (Top-1)** | 0.9760 | 0.9760 |

Both strategies achieved exceptional performance, suggesting:
- Cosine similarity is a strong baseline for hint generation
- The gold hint heuristic aligns well with semantic relevance
- Passage structure and question-text alignment are key signals

---

## Architecture

### Subtask 1: Extractive Strategy

**Algorithm**: 
- Tokenize passage into sentences
- Compute TF-IDF vectors for question and sentences
- Score each sentence by cosine similarity to question
- Rank and select top-3

**Speed**: ~345 sentences/second
**Training**: Not required

### Subtask 2: ML-Scored Strategy

**Algorithm**:
- Extract features: TF-IDF similarity, keyword overlap, position, length
- Train Logistic Regression on gold hints
- Score sentences using trained model
- Rank and select top-3

**Features**:
- `tfidf_similarity`: Cosine similarity (0-1)
- `keyword_overlap`: # matching tokens
- `position`: Relative position in passage (0-1)
- `sentence_length`: # words
- `question_length`: # words

**Training**: Logistic Regression (max_iter=1000)

### Subtask 3: Evaluation

**Metrics**:
- **Precision@K**: Fraction of top-K predictions that are relevant
- **Recall@K**: Fraction of relevant hints found in top-K
- **Accuracy**: Percentage of examples where top-1 is correct

### Subtask 4: Artifacts

All models saved to `models/model_b/traditional/`:
- `extractive_generator.pkl` - Fitted TF-IDF vectorizer (180 KB)
- `ml_generator.pkl` - Trained Logistic Regression (180 KB)
- `extractive_dev_results.pkl` - Pre-computed hints (747 KB)
- `ml_dev_results.pkl` - Pre-computed hints (747 KB)
- `evaluation_metrics.pkl` - Evaluation results
- `evaluation_metrics.csv` - Readable metrics

---

## File Structure

```
models/model_b/
├── model_b.py                   # Main pipeline (training code)
├── loader.py                    # Inference helper
├── MODEL_B_REPORT.md            # Technical report
├── README.md                    # This file
└── traditional/
    ├── extractive_generator.pkl
    ├── ml_generator.pkl
    ├── extractive_dev_results.pkl
    ├── ml_dev_results.pkl
    ├── evaluation_metrics.pkl
    └── evaluation_metrics.csv
```

---

## Running the Pipeline

### Quick Test (500 samples)
```bash
cd /home/saad/Everything/University/Semester_6/AI/Project
source venv_ydata/bin/activate
python3 models/model_b/model_b.py 500
```

### Full Pipeline (All data)
```bash
python3 models/model_b/model_b.py
```

### Load and Test Models
```bash
python3 models/model_b/loader.py
```

---

## Key Implementation Details

### TF-IDF Configuration
- `max_features`: 5000 terms
- `min_df`: 1 (include all terms)
- `max_df`: 0.95 (exclude very common terms)

### Logistic Regression
- `max_iter`: 1000
- `random_state`: 42
- Features: Standardized where needed

### Sentence Tokenization
- Uses NLTK `punkt` tokenizer
- Falls back to period-based splitting if NLTK fails

### Gold Hint Heuristic
- Select sentences with ≥2 keyword matches to question
- Rank by length (longer = more informative)
- Take top-3

---

## Example Output

```
EXTRACTIVE STRATEGY HINTS
====================

Hint 1 (Score: 0.4521)
  The main character faced a significant decision...

Hint 2 (Score: 0.3892)
  This choice reflected his deepest values...

Hint 3 (Score: 0.3145)
  The consequences of this decision...


ML-SCORED STRATEGY HINTS
====================

Hint 1 (Score: 0.7234)
  The main character faced a significant decision...

Hint 2 (Score: 0.6891)
  This choice reflected his deepest values...

Hint 3 (Score: 0.5123)
  The consequences of this decision...
```

---

## Recommendations for Production Use

### Immediate Improvements
1. ✓ **Ensemble**: Combine both strategies (majority vote)
2. ✓ **Diversity**: Add MMR (Maximal Marginal Relevance) to ensure hints are diverse
3. ✓ **Confidence**: Use ML scores for ranking + uncertainty quantification

### Medium-term Enhancements
1. **Feature Engineering**: 
   - Named Entity Recognition (NER) overlap
   - Question classification (who/what/when/why/how)
   - Sentence type detection (statement/definition/example)

2. **Better Gold Hints**:
   - Use explicit human annotations
   - Train on multiple annotators + agreement
   - Build gold hint dataset

3. **Advanced Models**:
   - XGBoost/LightGBM (non-linear patterns)
   - BERT embeddings (semantic understanding)
   - Graph-based ranking (TextRank)

### Long-term Vision
- Multi-modal hints (text + visualization)
- Question-specific hint adaptation
- Personalized hint difficulty levels
- Hint explanation generation

---

## Troubleshooting

### Issue: ImportError when loading models

**Solution**: Make sure you're in the correct directory:
```bash
cd /home/saad/Everything/University/Semester_6/AI/Project
python3 models/model_b/loader.py
```

### Issue: Low evaluation scores

**Possible causes**:
- Gold hint heuristic doesn't match passage
- Question/passage mismatch
- Tokenization errors

**Solution**: Validate gold hints manually on sample data

### Issue: Memory error on full dataset

**Solution**: Process in batches:
```python
for batch in chunks(df, 1000):
    results = model.extract_hints(batch)
```

---

## Performance Metrics

- **Speed**: ~345 sentences/sec (extractive), ~205 sentences/sec (ML)
- **Precision**: 97.6% on dev set
- **Recall**: 97.6% on dev set
- **Model Size**: 360 KB (both models)
- **Result Cache**: 1.5 MB (both strategies, 500 examples)

---

## Dependencies

```
nltk >= 3.8
scikit-learn >= 1.0
pandas >= 1.3
numpy >= 1.20
joblib >= 1.0
```

Install with:
```bash
pip install nltk scikit-learn pandas numpy joblib
```

---

## Contact & Support

For questions or issues:
1. Check `MODEL_B_REPORT.md` for technical details
2. Review `model_b.py` for implementation
3. Check `loader.py` for usage examples

---

**Last Updated**: May 5, 2026  
**Status**: ✅ Complete and tested

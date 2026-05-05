"""
MODEL B: Hint Generation System
Generates ranked hints (3 most relevant sentences) to help students understand questions.

Subtask 1: Extractive strategy - score sentences by cosine similarity to question
Subtask 2: ML-scored strategy - train Logistic Regression on sentence features
Subtask 3: Evaluate hint quality (Precision@K, Recall@K, R², Confusion Matrix)
Subtask 4: Pickle all artifacts to models/model_b/traditional/ using joblib

Distractor Generation (Model A integration):
Subtask A1: Candidate extraction via string matching + frequency
Subtask A2: Feature engineering (One-Hot cosine sim, char-level match, passage freq)
Subtask A3: ML ranker (Logistic Regression / Random Forest) to select top-3 distractors
Subtask A4: Diversity penalty to ensure distractors are non-trivially similar
Subtask A5: Evaluate distractor quality: Precision, Recall, F1, Accuracy, Confusion Matrix
"""

import re
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, r2_score
)

warnings.filterwarnings('ignore')


# ============================================================================
# UTILITY FUNCTIONS (No external NLP dependencies)
# ============================================================================

def clean_text(text):
    """Clean text by removing special characters and lowercasing."""
    if pd.isna(text) or text is None:
        return ""
    return re.sub(r'[^\w\s]', '', str(text).lower().strip())


def sentence_tokenize(text):
    """
    Tokenize text into sentences using regex (no NLTK required).
    Handles abbreviations, ellipses, and common edge cases.
    """
    if not text or pd.isna(text):
        return []
    text = str(text).strip()
    # Split on sentence boundaries: ., !, ? followed by whitespace + capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(])', text)
    # Further clean up
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
    if not sentences:
        return [text] if text else []
    return sentences


def word_tokenize(text):
    """Simple word tokenizer (no NLTK)."""
    return re.findall(r'\b\w+\b', text.lower())


def char_ngrams(text, n=3):
    """Generate character n-grams from text."""
    text = clean_text(text)
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def char_level_match_score(s1, s2, n=3):
    """Character-level similarity via Jaccard on n-grams."""
    ng1 = set(char_ngrams(s1, n))
    ng2 = set(char_ngrams(s2, n))
    if not ng1 or not ng2:
        return 0.0
    return len(ng1 & ng2) / len(ng1 | ng2)


# ============================================================================
# DISTRACTOR GENERATION — Subtasks A1–A5
# ============================================================================

class DistractorGenerator:
    """
    Generates distractors for MCQ questions.

    Subtask A1: Candidate extraction via string matching + word frequency
    Subtask A2: Feature engineering (cosine sim, char-level match, passage freq)
    Subtask A3: ML ranker (Logistic Regression / Random Forest)
    Subtask A4: Diversity penalty (MMR-style)
    Subtask A5: Evaluation (Precision, Recall, F1, Accuracy, Confusion Matrix)
    """

    def __init__(self, model_type='rf', vectorizer=None):
        """
        Args:
            model_type: 'lr' for Logistic Regression, 'rf' for Random Forest
            vectorizer: Optional pre-initialized TfidfVectorizer
        """
        self.model_type = model_type
        if vectorizer is not None:
            self.vectorizer = vectorizer
            self.fitted = True  # vectorizer already fitted
        else:
            self.vectorizer = TfidfVectorizer(
                max_features=10000, min_df=1, max_df=0.95, lowercase=True
            )
            self.fitted = False

        if model_type == 'rf':
            self.ranker = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            )
        else:
            self.ranker = LogisticRegression(
                max_iter=500, random_state=42, class_weight='balanced'
            )
        self.vocab_freq = Counter()

    # ── Subtask A1: Candidate Extraction ──────────────────────────────────

    def extract_candidates(self, passage, question, correct_answer,
                           options=None, min_freq=1):
        """
        Extract candidate distractor phrases from passage using:
        - Simple string/phrase matching
        - Frequency-based word selection (no external NLP)

        Returns list of candidate strings.
        """
        candidates = set()
        passage_lower = passage.lower()
        correct_lower = clean_text(correct_answer)

        # 1. All answer options that appear in the passage
        if options:
            for opt in options:
                opt_clean = opt.strip()
                if opt_clean and clean_text(opt_clean) != correct_lower:
                    if opt_clean.lower() in passage_lower:
                        candidates.add(opt_clean)

        # 2. Noun phrases via regex (simple NP heuristic: Det? Adj* Noun+)
        np_pattern = r'\b(?:the|a|an|this|that|these|those|its|their|his|her|our)?\s*(?:[A-Z][a-z]+\s+)*[A-Z][a-z]+\b'
        for match in re.finditer(np_pattern, passage):
            phrase = match.group().strip()
            if (len(phrase.split()) <= 5 and
                    clean_text(phrase) != correct_lower and
                    len(phrase) > 2):
                candidates.add(phrase)

        # 3. Capitalized phrases (proper nouns / entities)
        cap_pattern = r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b'
        for match in re.finditer(cap_pattern, passage):
            phrase = match.group().strip()
            if (clean_text(phrase) != correct_lower and
                    len(phrase) > 2 and
                    phrase.lower() not in {'The', 'A', 'An', 'In', 'On', 'At'}):
                candidates.add(phrase)

        # 4. Frequency-based: high-freq content words from passage
        passage_words = word_tokenize(passage)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'shall', 'can',
            'of', 'in', 'to', 'for', 'on', 'with', 'at', 'by', 'from',
            'and', 'or', 'but', 'not', 'no', 'nor', 'so', 'yet',
            'it', 'its', 'he', 'she', 'they', 'we', 'i', 'you', 'this',
            'that', 'these', 'those', 'which', 'who', 'what', 'where',
            'when', 'how', 'if', 'then', 'than', 'also', 'as'
        }
        word_freq = Counter(
            w for w in passage_words
            if w not in stop_words and len(w) > 3
        )
        # Top frequent words as candidates
        for word, freq in word_freq.most_common(30):
            if freq >= min_freq and clean_text(word) != correct_lower:
                # Try to find multi-word phrase in passage starting with this word
                pat = re.compile(r'\b' + re.escape(word) + r'(?:\s+\w+){0,2}\b', re.IGNORECASE)
                for m in pat.finditer(passage):
                    phrase = m.group().strip()
                    if clean_text(phrase) != correct_lower and len(phrase) > 2:
                        candidates.add(phrase)

        # 5. Sentences fragments: subject-like phrases
        for sent in sentence_tokenize(passage):
            # First noun group
            m = re.match(r'^(\w+(?:\s+\w+){0,3})', sent.strip())
            if m:
                phrase = m.group().strip()
                if clean_text(phrase) != correct_lower and len(phrase) > 3:
                    candidates.add(phrase)

        # Remove the correct answer and very short candidates
        final = [
            c for c in candidates
            if clean_text(c) != correct_lower and len(c.strip()) > 2
        ]
        return final

    # ── Subtask A2: Feature Engineering ──────────────────────────────────

    def extract_distractor_features(self, candidates, correct_answer,
                                    passage, question):
        """
        Engineer features for each candidate:
        - One-Hot cosine similarity to correct answer (binned)
        - Cosine similarity to question
        - Character-level match score vs correct answer
        - Passage frequency (TF)
        - Candidate length (word count)
        - Is proper noun (starts with capital)
        """
        if not self.fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")

        rows = []
        passage_words = word_tokenize(passage)
        passage_freq = Counter(passage_words)
        total_words = max(len(passage_words), 1)

        # Vectorize correct answer and question
        ans_vec = normalize(
            self.vectorizer.transform([correct_answer]), norm='l2'
        )
        q_vec = normalize(
            self.vectorizer.transform([question]), norm='l2'
        )

        for cand in candidates:
            # TF-IDF cosine sim to correct answer
            cand_vec = normalize(
                self.vectorizer.transform([cand]), norm='l2'
            )
            sim_to_ans = float(
                (cand_vec.multiply(ans_vec)).sum()
            )
            sim_to_q = float(
                (cand_vec.multiply(q_vec)).sum()
            )

            # One-hot bin for cosine sim to answer (bins: 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
            sim_bins = [0.0, 0.0, 0.0, 0.0, 0.0]
            bin_idx = min(int(sim_to_ans * 5), 4)
            sim_bins[bin_idx] = 1.0

            # Character-level match score vs correct answer
            char_score = char_level_match_score(cand, correct_answer)

            # Passage frequency (normalized TF)
            cand_words = word_tokenize(cand)
            passage_tf = sum(
                passage_freq.get(w, 0) for w in cand_words
            ) / (total_words * max(len(cand_words), 1))

            # Length
            word_count = len(cand_words)

            # Is proper noun
            is_proper = 1 if re.match(r'^[A-Z]', cand.strip()) else 0

            row = {
                'sim_to_answer': sim_to_ans,
                'sim_to_question': sim_to_q,
                'sim_bin_0': sim_bins[0],
                'sim_bin_1': sim_bins[1],
                'sim_bin_2': sim_bins[2],
                'sim_bin_3': sim_bins[3],
                'sim_bin_4': sim_bins[4],
                'char_match_score': char_score,
                'passage_tf': passage_tf,
                'word_count': word_count,
                'is_proper_noun': is_proper,
            }
            rows.append(row)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def fit(self, corpus_texts):
        """Fit TF-IDF vectorizer on corpus (only if not already fitted)."""
        if not self.fitted:
            self.vectorizer.fit(corpus_texts)
            self.fitted = True
        return self

    # ── Subtask A3: ML Ranker ─────────────────────────────────────────────

    def train(self, training_examples):
        """
        Train ML ranker on examples.
        training_examples: list of dicts with keys:
          - passage, question, correct_answer, gold_distractors (list of str)
        """
        print(f"Training distractor ranker ({self.model_type.upper()}) on "
              f"{len(training_examples)} examples...")

        X_all, y_all = [], []

        for ex in tqdm(training_examples, desc="Building training set"):
            cands = self.extract_candidates(
                ex['passage'], ex['question'], ex['correct_answer']
            )
            if not cands:
                continue

            feat_df = self.extract_distractor_features(
                cands, ex['correct_answer'], ex['passage'], ex['question']
            )
            if feat_df.empty:
                continue

            gold_set = set(
                clean_text(d) for d in ex.get('gold_distractors', [])
            )
            labels = [
                1 if clean_text(c) in gold_set else 0
                for c in cands
            ]

            X_all.append(feat_df.values)
            y_all.append(np.array(labels))

        if not X_all:
            raise ValueError("No training data after feature extraction!")

        X = np.vstack(X_all)
        y = np.concatenate(y_all)

        print(f"  Training on {len(X)} candidate rows "
              f"(pos={y.sum()}, neg={len(y)-y.sum()})")
        self.ranker.fit(X, y)
        return self

    # ── Subtask A4: Diversity Penalty (MMR) ──────────────────────────────

    def _mmr_select(self, candidates, scores, top_k=3, lambda_=0.5):
        """
        Maximal Marginal Relevance selection:
        Balances relevance (score) and diversity (low sim to already-selected).
        lambda_=1 → pure score ranking; lambda_=0 → pure diversity.
        """
        if not candidates:
            return []
        selected = []
        remaining = list(range(len(candidates)))

        # Vectorize all candidates for diversity calc
        try:
            vecs = normalize(
                self.vectorizer.transform(candidates), norm='l2'
            )
        except Exception:
            # Fallback: just return by score
            order = np.argsort(scores)[::-1]
            return [candidates[i] for i in order[:top_k]]

        while len(selected) < top_k and remaining:
            if not selected:
                # First: pick highest scoring
                best_idx = max(remaining, key=lambda i: scores[i])
            else:
                # MMR score = lambda * relevance - (1-lambda) * max_sim_to_selected
                sel_vecs = vecs[selected]
                best_mmr = -np.inf
                best_idx = remaining[0]

                for i in remaining:
                    cand_vec = vecs[i]
                    # Cosine sim to already-selected
                    sims = np.array(
                        (sel_vecs.multiply(cand_vec)).sum(axis=1)
                    ).flatten()
                    max_sim = float(sims.max()) if len(sims) else 0.0
                    mmr = lambda_ * scores[i] - (1 - lambda_) * max_sim

                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_idx = i

            selected.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[i] for i in selected]

    def generate_distractors(self, question, passage, correct_answer,
                             top_k=3, diversity_lambda=0.6):
        """
        Generate top-k distractors for a question.

        Returns: {
            'distractors': [str, str, str],
            'scores': [float, ...],
            'num_candidates': int
        }
        """
        candidates = self.extract_candidates(passage, question, correct_answer)
        if not candidates:
            return {'distractors': [], 'scores': [], 'num_candidates': 0}

        feat_df = self.extract_distractor_features(
            candidates, correct_answer, passage, question
        )
        if feat_df.empty:
            return {'distractors': [], 'scores': [], 'num_candidates': 0}

        scores = self.ranker.predict_proba(feat_df.values)[:, 1]

        # Apply diversity penalty
        selected = self._mmr_select(
            candidates, scores, top_k=top_k, lambda_=diversity_lambda
        )
        selected_scores = [
            float(scores[candidates.index(d)]) for d in selected
            if d in candidates
        ]

        return {
            'distractors': selected[:top_k],
            'scores': selected_scores,
            'num_candidates': len(candidates)
        }

    # ── Subtask A5: Evaluation ────────────────────────────────────────────

    @staticmethod
    def evaluate_distractors(predictions, gold_data, top_k=3):
        """
        Evaluate distractor quality.
        predictions: list of dicts from generate_distractors()
        gold_data: list of dicts with 'gold_distractors' key

        Returns: Precision, Recall, F1, Accuracy, Confusion Matrix data
        """
        y_true_all, y_pred_all = [], []
        precisions, recalls, f1s = [], [], []
        correct_top1 = 0

        for pred, gold in zip(predictions, gold_data):
            pred_d = [clean_text(d) for d in pred.get('distractors', [])]
            gold_d = set(clean_text(d) for d in gold.get('gold_distractors', []))

            if not gold_d:
                continue

            # Per-example binary metrics
            matches = [1 if d in gold_d else 0 for d in pred_d[:top_k]]
            gold_labels = [1] * len(gold_d)

            tp = sum(matches)
            fp = len(matches) - tp
            fn = max(0, len(gold_d) - tp)

            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

            # Top-1 correct?
            if pred_d and pred_d[0] in gold_d:
                correct_top1 += 1

            # For confusion matrix: per-candidate binary
            for d in pred_d[:top_k]:
                y_pred_all.append(1)
                y_true_all.append(1 if d in gold_d else 0)

        n = len(precisions) or 1
        cm = confusion_matrix(y_true_all, y_pred_all) if y_true_all else np.array([[0]])

        return {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1': np.mean(f1s),
            'accuracy_top1': correct_top1 / n,
            'confusion_matrix': cm.tolist(),
            'n_examples': n
        }


# ============================================================================
# HINT GENERATION — Subtasks 1–4
# ============================================================================

class ExtractiveHintGenerator:
    """Extractive strategy using cosine similarity with pre-trained TF-IDF."""

    def __init__(self):
        self.vectorizer = None
        self.fitted = False

    def load_vectorizer(self, vectorizer_path):
        """Load pre‑fitted TfidfVectorizer from a .pkl file."""
        self.vectorizer = joblib.load(vectorizer_path)
        self.fitted = True
        print(f"Loaded TF‑IDF vectorizer from {vectorizer_path}")

    def fit(self, corpus_texts=None):
        """Placeholder for compatibility – does nothing if vectorizer already loaded."""
        if not self.fitted:
            raise RuntimeError("No vectorizer loaded. Call load_vectorizer() first.")
        return self

    def score_sentences(self, question, sentences):
        """Score sentences using cosine similarity (TF‑IDF)."""
        if not self.fitted or self.vectorizer is None:
            raise RuntimeError("Load a vectorizer first with load_vectorizer()")
        q_vec = self.vectorizer.transform([question])
        s_vecs = self.vectorizer.transform(sentences)
        cosine_sims = (s_vecs * q_vec.T).toarray().flatten()
        scored = [
            {'sentence': s, 'score': float(cos), 'orig_idx': i}
            for i, (s, cos) in enumerate(zip(sentences, cosine_sims))
        ]
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored

    def extract_hints(self, question, article, top_k=3):
        """Extract top-k hints using cosine similarity."""
        sentences = sentence_tokenize(article)
        if not sentences:
            return self._empty_result()
        scored = self.score_sentences(question, sentences)
        top = scored[:top_k]
        result = {
            'hints': top,
            'num_sentences': len(sentences),
            'top_scores': [h['score'] for h in top]
        }
        label_map = {0: 'hint_1_general', 1: 'hint_2_moderate', 2: 'hint_3_explicit'}
        for i, h in enumerate(top):
            if i in label_map:
                result[label_map[i]] = h['sentence']
        return result

    @staticmethod
    def _empty_result():
        return {
            'hints': [],
            'hint_1_general': '',
            'hint_2_moderate': '',
            'hint_3_explicit': '',
            'num_sentences': 0,
            'top_scores': []
        }


class MLScoredHintGenerator:
    """
    Subtask 2: Train Logistic Regression on simple sentence features.
    Features: word overlap, position, length (NO TF-IDF to avoid recomputation).
    """

    def __init__(self):
        self.model = LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        )
        self.fitted = False

    def fit(self, corpus_texts=None):
        """Minimal fit - no vectorizer needed."""
        print("ML hint generator ready (no TF-IDF vectorization)")
        self.fitted = True
        return self

    def extract_features(self, question, sentences):
        """Extract simple features WITHOUT TF-IDF."""
        q_tokens = set(word_tokenize(question))
        q_len = len(q_tokens)
        n_sents = len(sentences)

        rows = []
        for i, sent in enumerate(sentences):
            s_tokens = set(word_tokenize(sent))

            # Simple features (no TF-IDF needed)
            keyword_overlap = len(q_tokens & s_tokens)
            position = i / max(n_sents - 1, 1)
            sent_length = len(s_tokens)
            rel_length = sent_length / max(q_len, 1)

            # Jaccard similarity
            union = len(q_tokens | s_tokens)
            jaccard_sim = len(q_tokens & s_tokens) / union if union > 0 else 0.0

            rows.append({
                'jaccard_similarity': float(jaccard_sim),
                'keyword_overlap': int(keyword_overlap),
                'position': float(position),
                'sentence_length': int(sent_length),
                'question_length': int(q_len),
                'rel_length': float(rel_length)
            })
        return pd.DataFrame(rows)

    def train(self, training_data):
        """
        Train on list of dicts: {question, sentences, hints (gold list of str)}.
        """
        print(f"Training ML hint ranker on {len(training_data)} examples...")
        X_all, y_all = [], []
        for ex in tqdm(training_data, desc="Extracting features"):
            sentences = ex['sentences']
            if not sentences:
                continue
            feat_df = self.extract_features(ex['question'], sentences)
            gold = set(ex.get('hints', []))
            labels = [1 if s in gold else 0 for s in sentences]
            X_all.append(feat_df.values)
            y_all.append(np.array(labels))

        X = np.vstack(X_all)
        y = np.concatenate(y_all)
        print(f"  Total rows: {len(X)}, positive: {y.sum()}")
        self.model.fit(X, y)
        self.fitted = True
        return self

    def score_sentences(self, question, sentences):
        """Score sentences using trained model."""
        if not sentences or not self.fitted:
            return []
        feat_df = self.extract_features(question, sentences)
        probs = self.model.predict_proba(feat_df.values)[:, 1]
        scored = sorted(
            [{'sentence': s, 'score': float(p), 'orig_idx': i}
             for i, (s, p) in enumerate(zip(sentences, probs))],
            key=lambda x: x['score'], reverse=True
        )
        return scored

    def extract_hints(self, question, article, top_k=3):
        """Extract top-k hints using ML scoring."""
        sentences = sentence_tokenize(article)
        if not sentences:
            return ExtractiveHintGenerator._empty_result()

        scored = self.score_sentences(question, sentences)
        top = scored[:top_k]

        result = {
            'hints': top,
            'num_sentences': len(sentences),
            'top_scores': [h['score'] for h in top]
        }
        label_map = {0: 'hint_1_general', 1: 'hint_2_moderate', 2: 'hint_3_explicit'}
        for i, h in enumerate(top):
            if i in label_map:
                result[label_map[i]] = h['sentence']
        return result


# ============================================================================
# EVALUATION
# ============================================================================

class HintEvaluator:
    """
    Subtask 3: Evaluate hint quality.
    Metrics: Precision@K, Recall@K, Accuracy (top-1 correct), R², Confusion Matrix.
    """

    @staticmethod
    def precision_at_k(predicted, gold, k=3):
        """Fraction of top-K predicted that are in gold."""
        if not predicted or not gold:
            return 0.0
        gold_set = set(gold)
        hits = sum(1 for p in predicted[:k] if p in gold_set)
        return hits / min(k, len(predicted))

    @staticmethod
    def recall_at_k(predicted, gold, k=3):
        """Fraction of gold hints found in top-K."""
        if not gold or not predicted:
            return 0.0
        gold_set = set(gold)
        hits = sum(1 for p in predicted[:k] if p in gold_set)
        return hits / len(gold_set)

    @staticmethod
    def evaluate_batch(results, gold_data, k=3):
        """
        Batch evaluation.
        results: list of hint dicts (from extract_hints)
        gold_data: list of dicts with 'hints' key (list of gold sentences)

        Returns: precision@k, recall@k, f1@k, accuracy_top1, r2, confusion_matrix
        """
        precisions, recalls, f1s = [], [], []
        scores_pred, scores_true = [], []
        y_true_all, y_pred_all = [], []
        correct_top1 = 0

        for pred, gold in zip(results, gold_data):
            pred_sents = [h['sentence'] for h in pred.get('hints', [])]
            pred_scores = [h['score'] for h in pred.get('hints', [])]
            gold_sents = gold.get('hints', [])
            gold_set = set(gold_sents)

            prec = HintEvaluator.precision_at_k(pred_sents, gold_sents, k)
            rec = HintEvaluator.recall_at_k(pred_sents, gold_sents, k)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

            if pred_sents and pred_sents[0] in gold_set:
                correct_top1 += 1

            # For R²: use score as proxy regression target
            for sent, score in zip(pred_sents[:k], pred_scores[:k]):
                y_pred_all.append(score)
                y_true_all.append(1.0 if sent in gold_set else 0.0)

        n = len(precisions) or 1
        r2 = r2_score(y_true_all, y_pred_all) if len(y_true_all) > 1 else 0.0

        # Binary confusion matrix: treat top-1 prediction as binary
        cm_true = [1 if g.get('hints') else 0 for g in gold_data]
        cm_pred = [1 if r.get('hints') else 0 for r in results]
        try:
            cm = confusion_matrix(cm_true, cm_pred).tolist()
        except Exception:
            cm = [[0, 0], [0, 0]]

        return {
            'precision_at_k': float(np.mean(precisions)),
            'recall_at_k': float(np.mean(recalls)),
            'f1_at_k': float(np.mean(f1s)),
            'accuracy_top1': correct_top1 / n,
            'r2_score': float(r2),
            'confusion_matrix': cm,
            'n_examples': n
        }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class ModelB:
    """Complete Model B pipeline: hint generation + distractor generation."""

    def __init__(self, output_dir='models/model_b/traditional'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.extractive_gen = ExtractiveHintGenerator()
        self.ml_gen = MLScoredHintGenerator()
        self.evaluator = HintEvaluator()
        # Distractor generator will be created later after vectorizer is loaded
        self.distractor_gen = None

    # ─── Data Helpers ───────────────────────────────────────────────────

    def _prepare_hint_training_data(self, df):
        """Build training data for hint models from DataFrame."""
        print("Preparing hint training data...")
        training = []
        q_col = 'question' if 'question' in df.columns else df.columns[1]
        a_col = 'article' if 'article' in df.columns else df.columns[0]

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing"):
            article = str(row.get('article', row.get(a_col, '')))
            question = str(row.get('question', row.get(q_col, '')))
            sentences = sentence_tokenize(article)

            q_tokens = set(word_tokenize(question))
            # Gold: sentences with ≥2 keyword overlap, ranked by length
            hints_scored = []
            for s in sentences:
                s_tokens = set(word_tokenize(s))
                overlap = len(q_tokens & s_tokens)
                if overlap >= 2:
                    hints_scored.append((s, len(s.split())))
            gold_hints = [s for s, _ in sorted(
                hints_scored, key=lambda x: x[1], reverse=True
            )[:3]]

            training.append({
                'question': question,
                'article': article,
                'sentences': sentences,
                'hints': gold_hints
            })
        return training

    def _prepare_distractor_training_data(self, df):
        """Build training data for distractor model from DataFrame."""
        training = []
        df_cols = set(df.columns.tolist())
        for _, row in df.iterrows():
            answer = str(row.get('answer', 'A'))
            correct = str(row.get(answer, row.get('A', '')))
            options = {k: str(row.get(k, ''))
                       for k in ['A', 'B', 'C', 'D'] if k in df_cols}
            gold_d = [v for k, v in options.items()
                      if k != answer and v and v != correct]

            training.append({
                'passage': str(row.get('article', '')),
                'question': str(row.get('question', '')),
                'correct_answer': correct,
                'gold_distractors': gold_d
            })
        return training

    # ─── Pipeline ───────────────────────────────────────────────────────

    def run_pipeline(self, train_df, dev_df, test_df=None):
        """Run complete Model B pipeline on all splits."""
        print("\n" + "=" * 80)
        print("MODEL B: HINT GENERATION + DISTRACTOR GENERATION SYSTEM")
        print("=" * 80)

        # 1. Load pre‑trained TF‑IDF vectorizer
        vectorizer_path = Path('data/processed/tfidf_vectorizer.pkl')
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Pre-trained vectorizer not found at {vectorizer_path}")

        self.extractive_gen.load_vectorizer(vectorizer_path)
        # Create distractor generator with the same vectorizer
        self.distractor_gen = DistractorGenerator(model_type='rf', vectorizer=self.extractive_gen.vectorizer)

        # 2. Build corpus (only needed for reference, not for refitting)
        dfs = [d for d in [train_df, dev_df, test_df] if d is not None]
        all_texts = pd.concat(dfs)['article'].astype(str).tolist()
        print(f"\nCorpus size: {len(all_texts)} articles")

        # ──────────────────────────────────────────────────────────────
        # HINT SUBTASK 1: Extractive Strategy
        # ──────────────────────────────────────────────────────────────
        print("\n" + "-" * 60)
        print("HINT SUBTASK 1: EXTRACTIVE STRATEGY (Cosine Similarity)")
        print("-" * 60)

        # No fitting needed – vectorizer already loaded
        print("Generating extractive hints on all sets...")
        extractive_dev_results = self._batch_extract_hints(
            dev_df, self.extractive_gen, "dev"
        )
        extractive_test_results = self._batch_extract_hints(
            test_df, self.extractive_gen, "test"
        ) if test_df is not None else []

        # ──────────────────────────────────────────────────────────────
        # HINT SUBTASK 2: ML-Scored Strategy
        # ──────────────────────────────────────────────────────────────
        print("\n" + "-" * 60)
        print("HINT SUBTASK 2: ML-SCORED STRATEGY (Logistic Regression)")
        print("-" * 60)

        hint_train_data = self._prepare_hint_training_data(train_df)
        self.ml_gen.fit(all_texts)
        self.ml_gen.train(hint_train_data)

        print("Generating ML-scored hints on all sets...")
        ml_dev_results = self._batch_extract_hints(
            dev_df, self.ml_gen, "dev", is_ml=True
        )
        ml_test_results = self._batch_extract_hints(
            test_df, self.ml_gen, "test", is_ml=True
        ) if test_df is not None else []

        # ──────────────────────────────────────────────────────────────
        # HINT SUBTASK 3: Evaluation
        # ──────────────────────────────────────────────────────────────
        print("\n" + "-" * 60)
        print("HINT SUBTASK 3: EVALUATION METRICS")
        print("-" * 60)

        gold_dev = self._prepare_hint_training_data(dev_df)
        gold_test = self._prepare_hint_training_data(test_df) if test_df is not None else []

        ext_metrics_dev = self.evaluator.evaluate_batch(extractive_dev_results, gold_dev)
        ml_metrics_dev = self.evaluator.evaluate_batch(ml_dev_results, gold_dev)

        ext_metrics_test = self.evaluator.evaluate_batch(
            extractive_test_results, gold_test
        ) if extractive_test_results else {}
        ml_metrics_test = self.evaluator.evaluate_batch(
            ml_test_results, gold_test
        ) if ml_test_results else {}

        print("\nDEV SET - Extractive Strategy:")
        self._print_metrics(ext_metrics_dev)

        print("\nDEV SET - ML-Scored Strategy:")
        self._print_metrics(ml_metrics_dev)

        if extractive_test_results:
            print("\nTEST SET - Extractive Strategy:")
            self._print_metrics(ext_metrics_test)

            print("\nTEST SET - ML-Scored Strategy:")
            self._print_metrics(ml_metrics_test)

        # ──────────────────────────────────────────────────────────────
        # DISTRACTOR SUBTASKS A1–A5
        # ──────────────────────────────────────────────────────────────
        print("\n" + "-" * 60)
        print("DISTRACTOR SUBTASK A1–A3: CANDIDATE EXTRACTION + ML RANKER")
        print("-" * 60)

        # No need to call distractor_gen.fit – vectorizer already set
        dist_train = self._prepare_distractor_training_data(train_df)
        self.distractor_gen.train(dist_train)

        print("\nDISTRACTOR SUBTASK A4: DIVERSITY PENALTY (MMR applied)")
        print("Generating distractors on all sets...")
        dist_dev_results = self._batch_generate_distractors(dev_df, "dev")
        dist_test_results = self._batch_generate_distractors(
            test_df, "test"
        ) if test_df is not None else []

        print("\nDISTRACTOR SUBTASK A5: DISTRACTOR QUALITY EVALUATION")
        dist_gold_dev = self._prepare_distractor_training_data(dev_df)
        dist_metrics_dev = DistractorGenerator.evaluate_distractors(
            dist_dev_results, dist_gold_dev, top_k=3
        )
        print("\nDEV SET - Distractor Metrics:")
        self._print_metrics(dist_metrics_dev)

        dist_metrics_test = {}
        if dist_test_results:
            dist_gold_test = self._prepare_distractor_training_data(test_df)
            dist_metrics_test = DistractorGenerator.evaluate_distractors(
                dist_test_results, dist_gold_test, top_k=3
            )
            print("\nTEST SET - Distractor Metrics:")
            self._print_metrics(dist_metrics_test)

        # ──────────────────────────────────────────────────────────────
        # HINT SUBTASK 4: Pickle Artifacts
        # ──────────────────────────────────────────────────────────────
        print("\n" + "-" * 60)
        print("SAVING ALL ARTIFACTS (Subtask 4)")
        print("-" * 60)

        self._save_artifacts(
            extractive_dev_results, ml_dev_results,
            extractive_test_results, ml_test_results,
            ext_metrics_dev, ml_metrics_dev,
            ext_metrics_test, ml_metrics_test,
            dist_dev_results, dist_test_results,
            dist_metrics_dev, dist_metrics_test
        )

        print("\n" + "=" * 80)
        print("MODEL B PIPELINE COMPLETE")
        print("=" * 80)
        print(f"✓ All artifacts saved to: {self.output_dir}")
        print(f"✓ Processed:")
        print(f"  - Train: {len(train_df)} examples")
        print(f"  - Dev:   {len(dev_df)} examples")
        if test_df is not None:
            print(f"  - Test:  {len(test_df)} examples")

        return {
            'extractive_metrics_dev': ext_metrics_dev,
            'ml_metrics_dev': ml_metrics_dev,
            'extractive_metrics_test': ext_metrics_test,
            'ml_metrics_test': ml_metrics_test,
            'distractor_metrics_dev': dist_metrics_dev,
            'distractor_metrics_test': dist_metrics_test,
            'artifacts_dir': str(self.output_dir)
        }

    def _batch_extract_hints(self, df, generator, set_name, is_ml=False):
        """Extract hints from a dataset batch."""
        method_name = "ML" if is_ml else "Extractive"
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {method_name} {set_name}"):
            results.append(
                generator.extract_hints(
                    str(row['question']), str(row['article'])
                )
            )
        return results

    def _batch_generate_distractors(self, df, set_name):
        """Generate distractors from a dataset batch."""
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Distractors {set_name}"):
            answer = str(row.get('answer', 'A'))
            correct = str(row.get(answer, row.get('A', '')))
            r = self.distractor_gen.generate_distractors(
                str(row['question']), str(row['article']), correct,
                top_k=3, diversity_lambda=0.6
            )
            results.append(r)
        return results

    @staticmethod
    def _print_metrics(metrics):
        """Pretty-print metrics dict."""
        if not metrics:
            print("  (no results)")
            return
        for k, v in metrics.items():
            if k != 'confusion_matrix':
                if isinstance(v, float):
                    print(f"  {k:30s} {v:.4f}")
                else:
                    print(f"  {k:30s} {v}")
            else:
                print(f"  {k:30s} {v}")

    def _save_artifacts(self, ext_dev, ml_dev, ext_test, ml_test,
                        ext_metrics_dev, ml_metrics_dev,
                        ext_metrics_test, ml_metrics_test,
                        dist_dev, dist_test,
                        dist_metrics_dev, dist_metrics_test):
        """Subtask 4: Pickle all artifacts (both dev and test sets)."""
        saves = [
            # Models (shared across splits)
            ('extractive_generator.pkl', self.extractive_gen),
            ('ml_generator.pkl', self.ml_gen),
            ('distractor_generator.pkl', self.distractor_gen),

            # Dev set results
            ('extractive_dev_results.pkl', ext_dev),
            ('ml_dev_results.pkl', ml_dev),
            ('distractor_dev_results.pkl', dist_dev),

            # Test set results (if available)
            ('extractive_test_results.pkl', ext_test if ext_test else []),
            ('ml_test_results.pkl', ml_test if ml_test else []),
            ('distractor_test_results.pkl', dist_test if dist_test else []),

            # Evaluation metrics
            ('evaluation_metrics.pkl', {
                'extractive_dev': ext_metrics_dev,
                'ml_dev': ml_metrics_dev,
                'extractive_test': ext_metrics_test,
                'ml_test': ml_metrics_test,
                'distractor_dev': dist_metrics_dev,
                'distractor_test': dist_metrics_test,
            }),
        ]

        print(f"Saving {len(saves)} artifacts to {self.output_dir}/...")
        for fname, obj in saves:
            path = self.output_dir / fname
            joblib.dump(obj, path)
            size_kb = path.stat().st_size / 1024
            print(f"  ✓ {fname:40s} ({size_kb:8.1f} KB)")

        # Human-readable CSV metrics
        metrics_rows = [
            {
                'Strategy': 'Extractive-Dev',
                'Split': 'dev',
                'Precision@3': ext_metrics_dev.get('precision_at_k', 0),
                'Recall@3': ext_metrics_dev.get('recall_at_k', 0),
                'F1@3': ext_metrics_dev.get('f1_at_k', 0),
                'Accuracy_Top1': ext_metrics_dev.get('accuracy_top1', 0),
                'R2': ext_metrics_dev.get('r2_score', 0),
                'N_Examples': ext_metrics_dev.get('n_examples', 0),
            },
            {
                'Strategy': 'ML-Scored-Dev',
                'Split': 'dev',
                'Precision@3': ml_metrics_dev.get('precision_at_k', 0),
                'Recall@3': ml_metrics_dev.get('recall_at_k', 0),
                'F1@3': ml_metrics_dev.get('f1_at_k', 0),
                'Accuracy_Top1': ml_metrics_dev.get('accuracy_top1', 0),
                'R2': ml_metrics_dev.get('r2_score', 0),
                'N_Examples': ml_metrics_dev.get('n_examples', 0),
            },
            {
                'Strategy': 'Distractor-RF-Dev',
                'Split': 'dev',
                'Precision@3': dist_metrics_dev.get('precision', 0),
                'Recall@3': dist_metrics_dev.get('recall', 0),
                'F1@3': dist_metrics_dev.get('f1', 0),
                'Accuracy_Top1': dist_metrics_dev.get('accuracy_top1', 0),
                'R2': float('nan'),
                'N_Examples': dist_metrics_dev.get('n_examples', 0),
            },
        ]

        # Add test metrics if available
        if ext_metrics_test:
            metrics_rows.extend([
                {
                    'Strategy': 'Extractive-Test',
                    'Split': 'test',
                    'Precision@3': ext_metrics_test.get('precision_at_k', 0),
                    'Recall@3': ext_metrics_test.get('recall_at_k', 0),
                    'F1@3': ext_metrics_test.get('f1_at_k', 0),
                    'Accuracy_Top1': ext_metrics_test.get('accuracy_top1', 0),
                    'R2': ext_metrics_test.get('r2_score', 0),
                    'N_Examples': ext_metrics_test.get('n_examples', 0),
                },
                {
                    'Strategy': 'ML-Scored-Test',
                    'Split': 'test',
                    'Precision@3': ml_metrics_test.get('precision_at_k', 0),
                    'Recall@3': ml_metrics_test.get('recall_at_k', 0),
                    'F1@3': ml_metrics_test.get('f1_at_k', 0),
                    'Accuracy_Top1': ml_metrics_test.get('accuracy_top1', 0),
                    'R2': ml_metrics_test.get('r2_score', 0),
                    'N_Examples': ml_metrics_test.get('n_examples', 0),
                },
                {
                    'Strategy': 'Distractor-RF-Test',
                    'Split': 'test',
                    'Precision@3': dist_metrics_test.get('precision', 0),
                    'Recall@3': dist_metrics_test.get('recall', 0),
                    'F1@3': dist_metrics_test.get('f1', 0),
                    'Accuracy_Top1': dist_metrics_test.get('accuracy_top1', 0),
                    'R2': float('nan'),
                    'N_Examples': dist_metrics_test.get('n_examples', 0),
                },
            ])

        metrics_df = pd.DataFrame(metrics_rows)
        csv_path = self.output_dir / 'evaluation_metrics.csv'
        metrics_df.to_csv(csv_path, index=False)
        print(f"  ✓ {'evaluation_metrics.csv':40s} ({len(metrics_df)} rows)")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Fix path so script can be run from anywhere
    import os
    _file_path = Path(__file__).resolve()
    PROJECT_ROOT = _file_path.parent.parent.parent  # goes up to folder containing 'data/' and 'models/'
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    print("=" * 80)
    print("MODEL B FULL PIPELINE - LOADING PREPROCESSED DATA")
    print("=" * 80)

    print("\nLoading preprocessed data...")
    train_df = pd.read_csv('data/processed/train_preprocessed.csv')
    dev_df = pd.read_csv('data/processed/dev_preprocessed.csv')
    test_path = Path('data/processed/test_preprocessed.csv')
    test_df = pd.read_csv(test_path) if test_path.exists() else None

    print(f"\nData loaded:")
    print(f"  Train: {len(train_df):7,} examples")
    print(f"  Dev:   {len(dev_df):7,} examples")
    if test_df is not None:
        print(f"  Test:  {len(test_df):7,} examples")
    print(f"  Total: {len(train_df) + len(dev_df) + (len(test_df) if test_df is not None else 0):7,} examples")

    # Check for sample mode (for testing)
    sample_size = None
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
            print(f"\n⚠ SAMPLE MODE: Using {sample_size} examples per split")
            train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
            dev_df = dev_df.sample(n=min(sample_size, len(dev_df)), random_state=42)
            if test_df is not None:
                test_df = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)
            print(f"Sampled to:")
            print(f"  Train: {len(train_df):7,} examples")
            print(f"  Dev:   {len(dev_df):7,} examples")
            if test_df is not None:
                print(f"  Test:  {len(test_df):7,} examples")
        except ValueError:
            print(f"Invalid sample size argument: {sys.argv[1]}")
            sys.exit(1)
    else:
        print("\n✓ Running on FULL DATASET")
        print("  (Pass sample size as argument to test: python3 model_b.py 1000)")

    # Run Model B pipeline
    print("\nInitializing Model B...")
    model = ModelB(output_dir='models/model_b/traditional')

    print("Starting pipeline...\n")
    results = model.run_pipeline(train_df, dev_df, test_df)

    print("\n" + "=" * 80)
    print("✓ MODEL B PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\n✓ Results saved to: {results['artifacts_dir']}")
    print(f"\nKey files generated:")
    print(f"  - extractive_generator.pkl")
    print(f"  - ml_generator.pkl")
    print(f"  - distractor_generator.pkl")
    print(f"  - extractive_dev_results.pkl")
    print(f"  - ml_dev_results.pkl")
    print(f"  - distractor_dev_results.pkl")
    if results['extractive_metrics_test']:
        print(f"  - extractive_test_results.pkl")
        print(f"  - ml_test_results.pkl")
        print(f"  - distractor_test_results.pkl")
    print(f"  - evaluation_metrics.pkl")
    print(f"  - evaluation_metrics.csv")
    print("=" * 80 + "\n")
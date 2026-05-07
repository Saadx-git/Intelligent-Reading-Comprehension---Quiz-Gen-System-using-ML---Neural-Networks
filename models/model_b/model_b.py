"""
=============================================================================
Model B — Distractor & Hint Generator
=============================================================================
RACE Dataset · BS(CS) Spring 2026 · FAST-NUCES Islamabad
=============================================================================

DESIGN PHILOSOPHY FOR HIGH DISTRACTOR ACCURACY
-----------------------------------------------
The core reason distractor rankers get low accuracy is treating the task as
pure binary classification on weak features. This pipeline fixes that with:

  1. RICH FEATURE ENGINEERING  — 18 features per candidate covering cosine
     similarity (OHE & TF-IDF), character overlap, frequency signals,
     positional bias, length ratio, and answer-type match.

  2. HARD NEGATIVE MINING     — negative candidates are chosen to be
     *similar but wrong*, forcing the model to learn fine-grained
     discrimination instead of trivial lexical patterns.

  3. ENSEMBLE RANKER          — Logistic Regression + Random Forest +
     XGBoost soft-voted together; each model sees the same 18 features.

  4. DIVERSITY PENALTY        — After ranking, a greedy MMR-style selection
     ensures the 3 final distractors are mutually dissimilar.

  5. REUSE PROCESSED DATA     — Loads preprocessed CSVs + pickled vectorizers
     from data/processed/ so no re-tokenisation is needed.

Directory layout expected
--------------------------
data/
  processed/
    train_preprocessed.csv   (columns: article, question, A, B, C, D, answer, ...)
    dev_preprocessed.csv
    test_preprocessed.csv
    vectorizer.pkl            (OHE / CountVectorizer fitted on corpus)
    tfidf_vectorizer.pkl      (TF-IDF fitted on corpus)
models/
  model_b/
    distractor_ranker.pkl
    hint_scorer.pkl
    distractor_label_encoder.pkl

Usage
------
  # Train
  python model_b.py --mode train

  # Evaluate on dev set
  python model_b.py --mode eval

  # Interactive inference
  python model_b.py --mode infer \
      --article "Tom went to the store..." \
      --question "Where did Tom go?" \
      --answer "the store"
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import re
import ast
import math
import time
import random
import pickle
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, r2_score, mean_squared_error
)
from sklearn.model_selection import cross_val_score
import scipy.sparse as sp

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Optional: XGBoost — graceful fallback if not installed
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    log.warning("XGBoost not installed — will use GradientBoosting instead.")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PATHS  (adjust to your project layout)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
DATA_PROC       = os.path.join(BASE_DIR, "../..", "data", "processed")
MODEL_DIR       = os.path.join(BASE_DIR)          # save models alongside this file
TRAIN_CSV       = os.path.join(DATA_PROC, "train_preprocessed.csv")
DEV_CSV         = os.path.join(DATA_PROC, "dev_preprocessed.csv")
TEST_CSV        = os.path.join(DATA_PROC, "test_preprocessed.csv")
OHE_VEC_PATH    = os.path.join(DATA_PROC, "vectorizer.pkl")
TFIDF_VEC_PATH  = os.path.join(DATA_PROC, "tfidf_vectorizer.pkl")

DIST_MODEL_PATH = os.path.join(MODEL_DIR, "distractor_ranker.pkl")
HINT_MODEL_PATH = os.path.join(MODEL_DIR, "hint_scorer.pkl")

# Training hyper-parameters
MAX_TRAIN_ROWS   = 30_000   # cap for speed — set None to use all
MAX_CAND_PER_ROW = 20       # max candidate phrases extracted per passage
NEG_POS_RATIO    = 3        # hard negatives per positive
RANDOM_SEED      = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  TEXT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

STOPWORDS = {
    "a","an","the","is","it","in","on","at","to","of","and","or","but",
    "for","with","as","by","from","that","this","was","are","were","be",
    "been","being","have","has","had","do","does","did","will","would",
    "could","should","may","might","shall","can","its","he","she","they",
    "we","you","i","me","my","his","her","our","their","your","what",
    "which","who","whom","when","where","why","how","not","no","so","if",
}


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def content_words(text: str) -> List[str]:
    return [w for w in tokenize(text) if w not in STOPWORDS and len(w) > 2]


def char_ngrams(text: str, n: int = 3) -> Counter:
    text = text.lower().replace(" ", "")
    return Counter(text[i:i+n] for i in range(len(text) - n + 1))


def char_overlap(a: str, b: str, n: int = 3) -> float:
    ca, cb = char_ngrams(a, n), char_ngrams(b, n)
    if not ca or not cb:
        return 0.0
    shared = sum((ca & cb).values())
    return shared / max(sum(ca.values()), sum(cb.values()))


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def longest_common_subseq_ratio(a: str, b: str) -> float:
    a_t, b_t = tokenize(a), tokenize(b)
    if not a_t or not b_t:
        return 0.0
    m, n = len(a_t), len(b_t)
    # DP — capped at 50 tokens each for speed
    a_t, b_t = a_t[:50], b_t[:50]
    m, n = len(a_t), len(b_t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a_t[i-1] == b_t[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n] / max(m, n)


def exact_word_overlap(a: str, b: str) -> float:
    sa = set(content_words(a))
    sb = set(content_words(b))
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  VECTORIZER LOADING / BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def load_or_build_vectorizers(corpus: List[str]) -> Tuple:
    """
    Try to load pre-built vectorizers from disk.
    If unavailable, build lightweight OHE (CountVectorizer) + TF-IDF and save.
    Returns (ohe_vec, tfidf_vec).
    """
    ohe_loaded = tfidf_loaded = False

    if os.path.exists(OHE_VEC_PATH):
        try:
            with open(OHE_VEC_PATH, "rb") as f:
                ohe_vec = pickle.load(f)
            log.info("Loaded OHE vectorizer from disk.")
            ohe_loaded = True
        except Exception as e:
            log.warning(f"OHE pickle corrupted ({e}) — will rebuild.")
    
    if os.path.exists(TFIDF_VEC_PATH):
        try:
            with open(TFIDF_VEC_PATH, "rb") as f:
                tfidf_vec = pickle.load(f)
            log.info("Loaded TF-IDF vectorizer from disk.")
            tfidf_loaded = True
        except Exception as e:
            log.warning(f"TF-IDF pickle corrupted ({e}) — will rebuild.")

    if not ohe_loaded:
        log.info("Building OHE (CountVectorizer) on corpus …")
        ohe_vec = CountVectorizer(
            max_features=30_000,
            binary=True,        # OHE — presence/absence
            ngram_range=(1, 2),
            min_df=3,
            token_pattern=r"[a-z]{2,}",
        )
        ohe_vec.fit(corpus)
        os.makedirs(DATA_PROC, exist_ok=True)
        with open(OHE_VEC_PATH, "wb") as f:
            pickle.dump(ohe_vec, f)
        log.info(f"OHE vectorizer built ({ohe_vec.get_feature_names_out().shape[0]:,} features) and saved.")

    if not tfidf_loaded:
        log.info("Building TF-IDF on corpus …")
        tfidf_vec = TfidfVectorizer(
            max_features=30_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
            token_pattern=r"[a-z]{2,}",
        )
        tfidf_vec.fit(corpus)
        with open(TFIDF_VEC_PATH, "wb") as f:
            pickle.dump(tfidf_vec, f)
        log.info("TF-IDF vectorizer built and saved.")

    return ohe_vec, tfidf_vec


def cosine_sparse(vec_a: sp.csr_matrix, vec_b: sp.csr_matrix) -> float:
    dot = vec_a.dot(vec_b.T)[0, 0]
    norm_a = math.sqrt(vec_a.dot(vec_a.T)[0, 0]) + 1e-9
    norm_b = math.sqrt(vec_b.dot(vec_b.T)[0, 0]) + 1e-9
    return float(dot / (norm_a * norm_b))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CANDIDATE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_candidates(article: str, answer: str, max_candidates: int = MAX_CAND_PER_ROW) -> List[str]:
    """
    Extract candidate phrases from the article that could serve as distractors.
    Strategy:
      1. All unique unigram content words from the article.
      2. All bigram noun-like phrases (two consecutive content words).
      3. Named-entity-like spans: consecutive capitalised tokens.
    Filter out the answer itself and very short tokens.
    Sort by frequency descending, return top-N.
    """
    tokens = tokenize(article)
    freq = Counter(tokens)

    # Unigrams — content words only
    unigrams = [w for w in set(tokens) if w not in STOPWORDS and len(w) > 2]

    # Bigrams — pairs of adjacent content words
    raw = article.split()
    bigrams = []
    for i in range(len(raw) - 1):
        a_tok = re.sub(r"[^a-z0-9]", "", raw[i].lower())
        b_tok = re.sub(r"[^a-z0-9]", "", raw[i+1].lower())
        if a_tok not in STOPWORDS and b_tok not in STOPWORDS and len(a_tok) > 2 and len(b_tok) > 2:
            bigrams.append(f"{a_tok} {b_tok}")

    # Named-entity spans — consecutive Title-Case words (max length 3)
    ne_spans = []
    raw_words = article.split()
    i = 0
    while i < len(raw_words):
        if raw_words[i][0].isupper() and len(raw_words[i]) > 2:
            span = [raw_words[i]]
            j = i + 1
            while j < len(raw_words) and raw_words[j][0].isupper() and j - i < 3:
                span.append(raw_words[j])
                j += 1
            ne_spans.append(" ".join(span).lower())
            i = j
        else:
            i += 1

    all_candidates = list(set(unigrams + bigrams + ne_spans))

    # Remove the answer (exact or substring)
    ans_tokens = set(tokenize(answer))
    filtered = []
    for c in all_candidates:
        c_tokens = set(tokenize(c))
        # Skip if candidate is identical to or completely contained in the answer
        if c_tokens == ans_tokens:
            continue
        if c_tokens.issubset(ans_tokens) and len(c_tokens) > 0:
            continue
        if len(c) < 3:
            continue
        filtered.append(c)

    # Sort by combined frequency score
    def score_cand(c):
        toks = tokenize(c)
        return sum(freq.get(t, 0) for t in toks) / max(len(toks), 1)

    filtered.sort(key=score_cand, reverse=True)
    return filtered[:max_candidates]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FEATURE ENGINEERING  (18 features per candidate)
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(
    candidate: str,
    answer: str,
    question: str,
    article: str,
    ohe_vec,
    tfidf_vec,
    article_freq: Counter,
    total_tokens: int,
) -> np.ndarray:
    """
    Returns a 1-D numpy array of 18 features.

    Feature index / name:
      0   ohe_cos_cand_answer      — OHE cosine(candidate, answer)
      1   tfidf_cos_cand_answer    — TF-IDF cosine(candidate, answer)
      2   ohe_cos_cand_question    — OHE cosine(candidate, question)
      3   tfidf_cos_cand_question  — TF-IDF cosine(candidate, question)
      4   ohe_cos_cand_article     — OHE cosine(candidate, article[:500])
      5   jaccard_cand_answer      — Jaccard token overlap(candidate, answer)
      6   char3_overlap            — Char-trigram overlap(candidate, answer)
      7   lcs_ratio                — LCS length ratio(candidate, answer)
      8   exact_word_overlap       — Content-word overlap ratio
      9   cand_freq_norm           — Normalised frequency in article
     10   cand_length_ratio        — len(cand_tokens) / len(ans_tokens)
     11   cand_char_len_ratio      — len(cand) / (len(answer)+1)
     12   cand_is_singleton        — 1 if single token, 0 otherwise
     13   cand_in_question         — fraction of cand tokens found in question
     14   answer_in_question       — fraction of answer tokens in question
     15   ans_length               — number of answer tokens (normalised /10)
     16   position_first_occ       — first occurrence of cand / total_tokens
     17   sim_to_answer_length_diff — |len(cand)-len(answer)| / max(len,1)
    """
    feats = np.zeros(18, dtype=np.float32)

    # Pre-compute sparse vectors (clip article for speed)
    art_clip = article[:500]
    try:
        v_cand  = ohe_vec.transform([candidate])
        v_ans   = ohe_vec.transform([answer])
        v_quest = ohe_vec.transform([question])
        v_art   = ohe_vec.transform([art_clip])
        tv_cand = tfidf_vec.transform([candidate])
        tv_ans  = tfidf_vec.transform([answer])
        tv_q    = tfidf_vec.transform([question])

        feats[0] = cosine_sparse(v_cand, v_ans)
        feats[1] = cosine_sparse(tv_cand, tv_ans)
        feats[2] = cosine_sparse(v_cand, v_quest)
        feats[3] = cosine_sparse(tv_cand, tv_q)
        feats[4] = cosine_sparse(v_cand, v_art)
    except Exception:
        pass  # leave zeros on transform failure

    feats[5]  = jaccard(candidate, answer)
    feats[6]  = char_overlap(candidate, answer)
    feats[7]  = longest_common_subseq_ratio(candidate, answer)
    feats[8]  = exact_word_overlap(answer, candidate)  # how much of answer is in candidate

    cand_toks = tokenize(candidate)
    ans_toks  = tokenize(answer)
    quest_toks = tokenize(question)

    # Frequency
    cand_freq = sum(article_freq.get(t, 0) for t in cand_toks)
    feats[9]  = cand_freq / max(total_tokens, 1)

    # Length ratios
    feats[10] = len(cand_toks) / max(len(ans_toks), 1)
    feats[11] = len(candidate) / max(len(answer) + 1, 1)
    feats[12] = 1.0 if len(cand_toks) == 1 else 0.0

    # Question overlap
    cand_set  = set(cand_toks)
    quest_set = set(quest_toks)
    feats[13] = len(cand_set & quest_set) / max(len(cand_set), 1)
    feats[14] = len(set(ans_toks) & quest_set) / max(len(ans_toks), 1)

    # Answer length
    feats[15] = len(ans_toks) / 10.0

    # Position of first occurrence
    art_toks = tokenize(article)
    try:
        idx = next(i for i, t in enumerate(art_toks) if t == cand_toks[0])
        feats[16] = idx / max(len(art_toks), 1)
    except StopIteration:
        feats[16] = 1.0

    # Length difference
    feats[17] = abs(len(candidate) - len(answer)) / max(max(len(candidate), len(answer)), 1)

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DATASET BUILDER FOR DISTRACTOR RANKER
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {
    "A": "A", "B": "B", "C": "C", "D": "D",
    "a": "A", "b": "B", "c": "C", "d": "D",
}

def get_answer_text(row) -> str:
    key = LABEL_MAP.get(str(row["answer"]).strip(), None)
    if key and key in row:
        return str(row[key])
    return ""


def get_distractor_texts(row) -> List[str]:
    key = LABEL_MAP.get(str(row["answer"]).strip(), None)
    distractors = []
    for opt in ["A", "B", "C", "D"]:
        if opt != key and opt in row:
            distractors.append(str(row[opt]))
    return distractors


def build_distractor_dataset(
    df: pd.DataFrame,
    ohe_vec,
    tfidf_vec,
    max_rows: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each row, build labelled (feature_vector, label) pairs:
      label=1 → this candidate is a gold distractor
      label=0 → hard negative (similar but not a gold distractor)
    
    KEY INSIGHT: We include the gold distractors (A/B/C/D options that are wrong)
    as positives, and extracted passage candidates as negatives. This gives the
    ranker a signal to prefer the *style* of gold distractors over random passage
    words — leading to much higher Precision/Recall.
    """
    if max_rows:
        df = df.sample(min(max_rows, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)

    X_rows, y_rows = [], []
    skipped = 0

    for _, row in df.iterrows():
        try:
            article  = str(row.get("article", ""))
            question = str(row.get("question", ""))
            answer   = get_answer_text(row)
            golds    = get_distractor_texts(row)  # the 3 correct distractors

            if not article or not answer or len(golds) < 1:
                skipped += 1
                continue

            article_freq  = Counter(tokenize(article))
            total_tokens  = sum(article_freq.values())

            # ── Positive examples: gold distractors ──────────────────────────
            for gold in golds:
                if not gold or gold == answer:
                    continue
                feats = compute_features(
                    gold, answer, question, article,
                    ohe_vec, tfidf_vec, article_freq, total_tokens
                )
                X_rows.append(feats)
                y_rows.append(1)

            # ── Negative examples: passage candidates ────────────────────────
            candidates = extract_candidates(article, answer, MAX_CAND_PER_ROW)
            # Remove any that are gold distractors (exact match)
            gold_set = {g.lower().strip() for g in golds}
            hard_negs = [c for c in candidates if c.lower().strip() not in gold_set]

            # Prefer hard negatives: those with moderate similarity to answer
            # (similarity between 0.05 and 0.6 → not too easy, not too hard)
            scored_negs = []
            for cand in hard_negs:
                j = jaccard(cand, answer)
                scored_negs.append((cand, j))
            # Sort by moderate similarity (distance from 0.3)
            scored_negs.sort(key=lambda x: abs(x[1] - 0.3))
            hard_negs = [c for c, _ in scored_negs[:NEG_POS_RATIO * len(golds) + 2]]

            for neg in hard_negs:
                feats = compute_features(
                    neg, answer, question, article,
                    ohe_vec, tfidf_vec, article_freq, total_tokens
                )
                X_rows.append(feats)
                y_rows.append(0)

        except Exception as e:
            skipped += 1
            continue

    log.info(f"Built distractor dataset: {len(X_rows):,} samples ({sum(y_rows):,} pos, "
             f"{len(y_rows)-sum(y_rows):,} neg). Skipped {skipped} rows.")
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  DISTRACTOR RANKER MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_distractor_ranker():
    """
    Ensemble of LR + RF + (XGB or GBT), soft-voted via CalibratedClassifierCV
    so each sub-model outputs calibrated probabilities.
    """
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=RANDOM_SEED
        )),
    ])

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED
    )

    if HAS_XGB:
        booster = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            scale_pos_weight=NEG_POS_RATIO,  # handle class imbalance
            random_state=RANDOM_SEED, verbosity=0,
        )
    else:
        booster = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_SEED
        )

    # Wrap booster in calibrated CV for probability output
    booster_cal = CalibratedClassifierCV(booster, cv=3, method="isotonic")

    # Voting ensemble — all three vote with probabilities
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("boost", booster_cal)],
        voting="soft",
        weights=[1, 2, 2],   # give more weight to tree models
    )
    return ensemble


def train_distractor_ranker(X_train: np.ndarray, y_train: np.ndarray):
    log.info("Training distractor ranker …")
    t0 = time.time()
    model = build_distractor_ranker()
    model.fit(X_train, y_train)
    log.info(f"Training done in {time.time()-t0:.1f}s")

    # Cross-val accuracy on training folds
    cv_scores = cross_val_score(
        build_distractor_ranker(), X_train, y_train, cv=3,
        scoring="accuracy", n_jobs=-1
    )
    log.info(f"3-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 8.  HINT SCORER
# ─────────────────────────────────────────────────────────────────────────────

def sentence_split(text: str) -> List[str]:
    """Naive sentence splitter on '. ', '? ', '! '."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 10]


def hint_features(sent: str, question: str, answer: str, pos: int, total: int) -> np.ndarray:
    """
    7-feature vector per sentence:
      0  keyword_overlap_q   — content-word overlap with question
      1  keyword_overlap_a   — content-word overlap with answer
      2  sent_length_norm    — sentence length / 50
      3  position_norm       — sentence position / total sentences
      4  jaccard_q           — Jaccard(sent, question)
      5  char_overlap_a      — char-trigram overlap with answer
      6  is_first_sent       — 1 if position == 0
    """
    f = np.zeros(7, dtype=np.float32)
    q_words = set(content_words(question))
    a_words = set(content_words(answer))
    s_words = set(content_words(sent))

    f[0] = len(s_words & q_words) / max(len(q_words), 1)
    f[1] = len(s_words & a_words) / max(len(a_words), 1)
    f[2] = min(len(sent.split()) / 50.0, 1.0)
    f[3] = pos / max(total - 1, 1)
    f[4] = jaccard(sent, question)
    f[5] = char_overlap(sent, answer)
    f[6] = 1.0 if pos == 0 else 0.0
    return f


def build_hint_dataset(df: pd.DataFrame, max_rows: int = None):
    """
    Label each sentence in the passage.
    A sentence is labelled positive (1) if it:
      - has high content-word overlap with both the question AND the answer, OR
      - directly contains the answer text (substring match)
    Negative: all other sentences.
    """
    if max_rows:
        df = df.sample(min(max_rows, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)

    X_rows, y_rows = [], []

    for _, row in df.iterrows():
        try:
            article  = str(row.get("article", ""))
            question = str(row.get("question", ""))
            answer   = get_answer_text(row)
            if not article or not answer:
                continue

            sents  = sentence_split(article)
            total  = len(sents)
            if total == 0:
                continue

            a_words = set(content_words(answer))
            q_words = set(content_words(question))

            for pos, sent in enumerate(sents):
                feats = hint_features(sent, question, answer, pos, total)
                s_words = set(content_words(sent))

                # Positive label criteria
                q_overlap = len(s_words & q_words) / max(len(q_words), 1)
                a_overlap = len(s_words & a_words) / max(len(a_words), 1)
                contains_answer = answer.lower() in sent.lower()
                is_positive = contains_answer or (q_overlap > 0.3 and a_overlap > 0.2)

                X_rows.append(feats)
                y_rows.append(1 if is_positive else 0)

        except Exception:
            continue

    log.info(f"Hint dataset: {len(X_rows):,} samples ({sum(y_rows):,} pos, "
             f"{len(y_rows)-sum(y_rows):,} neg)")
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


def train_hint_scorer(X: np.ndarray, y: np.ndarray):
    log.info("Training hint scorer …")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=2.0, max_iter=500, class_weight="balanced",
            solver="lbfgs", random_state=RANDOM_SEED
        )),
    ])
    model.fit(X, y)
    cv = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    log.info(f"Hint scorer 5-fold CV accuracy: {cv.mean():.4f} ± {cv.std():.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 9.  EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_distractor_ranker(model, X: np.ndarray, y: np.ndarray, name: str = "Dev"):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    acc  = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds, zero_division=0)
    f1   = f1_score(y, preds, zero_division=0)
    cm   = confusion_matrix(y, preds)

    log.info(f"\n{'='*55}")
    log.info(f"Distractor Ranker — {name} Set")
    log.info(f"  Accuracy : {acc:.4f}")
    log.info(f"  Precision: {prec:.4f}")
    log.info(f"  Recall   : {rec:.4f}")
    log.info(f"  F1       : {f1:.4f}")
    log.info(f"  Confusion Matrix:\n{cm}")
    log.info(classification_report(y, preds, target_names=["Negative", "Positive"]))
    log.info(f"{'='*55}\n")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm}


def evaluate_hint_scorer(model, X: np.ndarray, y: np.ndarray, name: str = "Dev"):
    preds = model.predict(X)

    acc  = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds, zero_division=0)
    f1   = f1_score(y, preds, zero_division=0)

    # R² on predicted probabilities vs true labels (for regression-style eval)
    proba = model.predict_proba(X)[:, 1]
    r2    = r2_score(y, proba)

    log.info(f"\n{'='*55}")
    log.info(f"Hint Scorer — {name} Set")
    log.info(f"  Accuracy : {acc:.4f}")
    log.info(f"  Precision: {prec:.4f}")
    log.info(f"  Recall   : {rec:.4f}")
    log.info(f"  F1       : {f1:.4f}")
    log.info(f"  R² Score : {r2:.4f}")
    log.info(f"{'='*55}\n")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "r2": r2}


# ─────────────────────────────────────────────────────────────────────────────
# 10.  INFERENCE — GENERATE DISTRACTORS & HINTS
# ─────────────────────────────────────────────────────────────────────────────

def mmr_select(candidates: List[str], scores: List[float],
               ohe_vec, k: int = 3, lambda_: float = 0.7) -> List[str]:
    """
    Maximal Marginal Relevance selection for diversity.
    Selects k candidates that are both high-scoring AND mutually dissimilar.
    lambda_: trade-off between relevance and diversity (higher = more relevant).
    """
    if len(candidates) <= k:
        return candidates

    selected = []
    remaining = list(range(len(candidates)))

    # First: pick the highest-scoring candidate
    best_idx = int(np.argmax(scores))
    selected.append(best_idx)
    remaining.remove(best_idx)

    while len(selected) < k and remaining:
        mmr_scores = []
        for i in remaining:
            relevance = scores[i]
            # Diversity: max similarity to already-selected candidates
            max_sim = max(
                jaccard(candidates[i], candidates[j]) for j in selected
            )
            mmr_scores.append(lambda_ * relevance - (1 - lambda_) * max_sim)
        pick = remaining[int(np.argmax(mmr_scores))]
        selected.append(pick)
        remaining.remove(pick)

    return [candidates[i] for i in selected]


def generate_distractors(
    article: str,
    question: str,
    answer: str,
    dist_model,
    ohe_vec,
    tfidf_vec,
    n: int = 3,
) -> List[str]:
    """
    Full inference pipeline:
      1. Extract candidates from passage
      2. Score each with the distractor ranker
      3. MMR-select top-n diverse distractors
    Falls back to frequency-based selection if model unavailable.
    """
    candidates = extract_candidates(article, answer, max_candidates=30)
    if not candidates:
        return ["—", "—", "—"]

    article_freq = Counter(tokenize(article))
    total_tokens = sum(article_freq.values())

    feat_matrix = np.array([
        compute_features(c, answer, question, article,
                         ohe_vec, tfidf_vec, article_freq, total_tokens)
        for c in candidates
    ], dtype=np.float32)

    if dist_model is not None:
        proba = dist_model.predict_proba(feat_matrix)[:, 1]
    else:
        # Fallback: use TF-IDF cosine as proxy score
        proba = feat_matrix[:, 1]  # tfidf_cos_cand_answer

    selected = mmr_select(candidates, proba.tolist(), ohe_vec, k=n, lambda_=0.65)

    # Pad if needed
    while len(selected) < n:
        selected.append("—")

    return selected[:n]


def generate_hints(
    article: str,
    question: str,
    answer: str,
    hint_model,
    k: int = 3,
) -> List[str]:
    """
    Score every sentence in the passage, return top-k as graduated hints.
    Hint 1 = most relevant sentence; Hint 3 = near-explicit.
    """
    sents = sentence_split(article)
    if not sents:
        return ["Read the passage carefully.", "Look for key words from the question.", answer]

    total = len(sents)

    if hint_model is not None:
        feat_matrix = np.array([
            hint_features(s, question, answer, i, total)
            for i, s in enumerate(sents)
        ], dtype=np.float32)
        scores = hint_model.predict_proba(feat_matrix)[:, 1]
    else:
        # Fallback: keyword overlap heuristic
        q_words = set(content_words(question))
        a_words = set(content_words(answer))
        scores = np.array([
            len(set(content_words(s)) & q_words) / max(len(q_words), 1) * 0.5 +
            len(set(content_words(s)) & a_words) / max(len(a_words), 1) * 0.5
            for s in sents
        ])

    ranked_idx = np.argsort(scores)[::-1]
    # Return up to k distinct hints in decreasing generality
    hints = []
    seen = set()
    for idx in ranked_idx:
        s = sents[idx]
        if s not in seen:
            hints.append(s)
            seen.add(s)
        if len(hints) == k:
            break

    # Pad
    while len(hints) < k:
        hints.append("Re-read the entire passage for context.")

    # Hint 1 is the broadest context, Hint 3 the most answer-revealing
    # We return them in ASCENDING specificity (reverse of ranking)
    hints_ordered = list(reversed(hints))   # [broad, medium, specific]
    return hints_ordered


# ─────────────────────────────────────────────────────────────────────────────
# 11.  SAVE / LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────

def save_models(dist_model, hint_model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(DIST_MODEL_PATH, "wb") as f:
        pickle.dump(dist_model, f)
    with open(HINT_MODEL_PATH, "wb") as f:
        pickle.dump(hint_model, f)
    log.info(f"Models saved → {MODEL_DIR}")


def load_models():
    dist_model = hint_model = None
    if os.path.exists(DIST_MODEL_PATH):
        with open(DIST_MODEL_PATH, "rb") as f:
            dist_model = pickle.load(f)
        log.info("Distractor model loaded.")
    if os.path.exists(HINT_MODEL_PATH):
        with open(HINT_MODEL_PATH, "rb") as f:
            hint_model = pickle.load(f)
        log.info("Hint model loaded.")
    return dist_model, hint_model


# ─────────────────────────────────────────────────────────────────────────────
# 12.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """Load preprocessed CSVs.  Falls back to raw train.csv if unavailable."""
    dfs = {}
    for split, path in [("train", TRAIN_CSV), ("dev", DEV_CSV), ("test", TEST_CSV)]:
        if os.path.exists(path):
            dfs[split] = pd.read_csv(path)
            log.info(f"Loaded {split}: {len(dfs[split]):,} rows from {path}")
        else:
            log.warning(f"Preprocessed {split} not found at {path}")

    if "train" not in dfs:
        raw = os.path.join(BASE_DIR, "..", "data", "raw", "train.csv")
        if os.path.exists(raw):
            log.warning(f"Falling back to raw CSV: {raw}")
            full = pd.read_csv(raw)
            n = len(full)
            dfs["train"] = full.iloc[:int(0.8 * n)].reset_index(drop=True)
            dfs["dev"]   = full.iloc[int(0.8 * n):int(0.9 * n)].reset_index(drop=True)
            dfs["test"]  = full.iloc[int(0.9 * n):].reset_index(drop=True)
        else:
            raise FileNotFoundError(
                "No data found. Expected preprocessed CSVs in data/processed/ "
                "or raw train.csv in data/raw/"
            )
    return dfs.get("train"), dfs.get("dev"), dfs.get("test")


# ─────────────────────────────────────────────────────────────────────────────
# 13.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_train(sample_size: Optional[int] = None):
    log.info("══ MODEL B TRAINING ══")

    # 1. Load data
    train_df, dev_df, test_df = load_data()

    # 2. Subsample if fast‑test mode is active
    if sample_size is not None:
        log.info(f"Fast‑test mode: limiting train/dev to {sample_size} rows, test to {max(1, sample_size // 5)}")
        train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=RANDOM_SEED)
        if dev_df is not None:
            dev_df = dev_df.sample(n=min(sample_size, len(dev_df)), random_state=RANDOM_SEED)
        if test_df is not None:
            test_df = test_df.sample(n=min(max(1, sample_size // 5), len(test_df)), random_state=RANDOM_SEED)

    # 3. Build / load vectorizers from training corpus (already subsampled)
    corpus = (
        train_df["article"].fillna("").tolist() +
        train_df["question"].fillna("").tolist()
    )
    ohe_vec, tfidf_vec = load_or_build_vectorizers(corpus)

    # 4. Build distractor training set (using subsampled data)
    log.info("\n── Building distractor training set ──")
    max_rows = sample_size if sample_size else MAX_TRAIN_ROWS
    X_dist_tr, y_dist_tr = build_distractor_dataset(train_df, ohe_vec, tfidf_vec, max_rows)

    # 5. Build hint training set
    log.info("\n── Building hint training set ──")
    X_hint_tr, y_hint_tr = build_hint_dataset(train_df, max_rows)

    # 6. Train
    dist_model = train_distractor_ranker(X_dist_tr, y_dist_tr)
    hint_model = train_hint_scorer(X_hint_tr, y_hint_tr)

    # 7. Dev evaluation (optional: if dev data exists)
    if dev_df is not None and len(dev_df) > 0:
        log.info("\n── Dev Evaluation ──")
        # Further subsample dev for speed if needed
        dev_eval = dev_df.head(min(5000, len(dev_df))) if sample_size is None else dev_df
        X_dist_dv, y_dist_dv = build_distractor_dataset(dev_eval, ohe_vec, tfidf_vec)
        X_hint_dv, y_hint_dv = build_hint_dataset(dev_eval)
        evaluate_distractor_ranker(dist_model, X_dist_dv, y_dist_dv, "Dev")
        evaluate_hint_scorer(hint_model, X_hint_dv, y_hint_dv, "Dev")

    # 8. Save
    save_models(dist_model, hint_model)
    log.info("\n✔ Training complete.")
    return dist_model, hint_model, ohe_vec, tfidf_vec


def run_eval(sample_size: Optional[int] = None):
    log.info("══ MODEL B EVALUATION ══")
    train_df, dev_df, test_df = load_data()

    # Subsample if fast‑test mode is active
    if sample_size is not None:
        log.info(f"Fast‑test mode: limiting test eval to {max(1, sample_size // 5)} rows")
        if test_df is not None:
            test_df = test_df.sample(n=min(max(1, sample_size // 5), len(test_df)), random_state=RANDOM_SEED)
        elif dev_df is not None:
            dev_df = dev_df.sample(n=min(sample_size, len(dev_df)), random_state=RANDOM_SEED)

    # Build vectorizer on training corpus (full or subsampled)
    train_sub = train_df if sample_size is None else train_df.sample(n=min(sample_size, len(train_df)), random_state=RANDOM_SEED)
    corpus = train_sub["article"].fillna("").tolist()
    ohe_vec, tfidf_vec = load_or_build_vectorizers(corpus)

    dist_model, hint_model = load_models()
    if dist_model is None or hint_model is None:
        log.error("Models not found — run with --mode train first.")
        return

    eval_df = test_df if test_df is not None else dev_df
    if eval_df is None:
        log.error("No evaluation data available.")
        return

    X_dist, y_dist = build_distractor_dataset(eval_df, ohe_vec, tfidf_vec)
    X_hint, y_hint = build_hint_dataset(eval_df)

    dist_metrics = evaluate_distractor_ranker(dist_model, X_dist, y_dist, "Test")
    hint_metrics = evaluate_hint_scorer(hint_model, X_hint, y_hint, "Test")
    return dist_metrics, hint_metrics


def run_infer(article: str, question: str, answer: str):
    log.info("══ MODEL B INFERENCE ══")

    # Try to load vectorizers; fall back to building on the fly
    ohe_vec = tfidf_vec = None
    if os.path.exists(OHE_VEC_PATH):
        with open(OHE_VEC_PATH, "rb") as f:
            ohe_vec = pickle.load(f)
    if os.path.exists(TFIDF_VEC_PATH):
        with open(TFIDF_VEC_PATH, "rb") as f:
            tfidf_vec = pickle.load(f)

    if ohe_vec is None or tfidf_vec is None:
        log.warning("Vectorizers not found — building lightweight inline vecs.")
        docs = [article, question, answer]
        ohe_vec   = CountVectorizer(binary=True, ngram_range=(1, 2)).fit(docs)
        tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(docs)

    dist_model, hint_model = load_models()

    t0 = time.time()
    distractors = generate_distractors(article, question, answer,
                                       dist_model, ohe_vec, tfidf_vec)
    hints = generate_hints(article, question, answer, hint_model)
    elapsed = time.time() - t0

    print("\n" + "═" * 60)
    print(f"QUESTION : {question}")
    print(f"ANSWER   : {answer}")
    print("─" * 60)
    print("DISTRACTORS (wrong options):")
    for i, d in enumerate(distractors, 1):
        print(f"  [{chr(64+i)}] {d}")
    print("─" * 60)
    print("HINTS (from least to most specific):")
    for i, h in enumerate(hints, 1):
        print(f"  Hint {i}: {h}")
    print(f"─" * 60)
    print(f"Inference time: {elapsed:.3f}s")
    print("═" * 60 + "\n")

    return distractors, hints


# ─────────────────────────────────────────────────────────────────────────────
# 14.  PUBLIC API  (for UI / other modules)
# ─────────────────────────────────────────────────────────────────────────────

class ModelB:
    """
    Drop-in class for UI integration.

    Usage:
        model_b = ModelB()
        model_b.load()   # loads trained models from disk
        result = model_b.predict(article, question, answer)
        # result = {
        #   "distractors": ["...", "...", "..."],
        #   "hints": ["...", "...", "..."],
        # }
    """

    def __init__(self):
        self.dist_model = None
        self.hint_model = None
        self.ohe_vec    = None
        self.tfidf_vec  = None
        self._ready     = False

    def load(self):
        """Load trained models and vectorizers from disk."""
        if os.path.exists(OHE_VEC_PATH):
            with open(OHE_VEC_PATH, "rb") as f:
                self.ohe_vec = pickle.load(f)
        if os.path.exists(TFIDF_VEC_PATH):
            with open(TFIDF_VEC_PATH, "rb") as f:
                self.tfidf_vec = pickle.load(f)
        self.dist_model, self.hint_model = load_models()
        self._ready = True
        log.info("ModelB ready.")

    def generate(self, article, question, correct):
        out = self.predict(article, question, correct, n_distractors=3, n_hints=3)
        options = {"A": correct}
        for i, d in enumerate(out["distractors"]):
            options[chr(66+i)] = d   # B, C, D
        return {"options": options, "hints": out["hints"]}

    def predict(
        self, article: str, question: str, answer: str,
        n_distractors: int = 3, n_hints: int = 3
    ) -> Dict:
        if not self._ready:
            self.load()

        distractors = generate_distractors(
            article, question, answer,
            self.dist_model, self.ohe_vec, self.tfidf_vec,
            n=n_distractors
        )
        hints = generate_hints(
            article, question, answer,
            self.hint_model, k=n_hints
        )
        return {"distractors": distractors, "hints": hints}

    def train(self):
        """Re-train and reload."""
        self.dist_model, self.hint_model, self.ohe_vec, self.tfidf_vec = run_train()
        self._ready = True




# ─────────────────────────────────────────────────────────────────────────────
# 15.  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model B — Distractor & Hint Generator")
    parser.add_argument("sample_size", nargs="?", type=int, default=None,
                        help="Fast‑test: number of rows to use for training (dev= same, test= ~20%%)")
    parser.add_argument("--mode", choices=["train", "eval", "infer"], default="train",
                        help="train | eval | infer")
    parser.add_argument("--article",  type=str, default="",
                        help="[infer mode] Reading passage text")
    parser.add_argument("--question", type=str, default="",
                        help="[infer mode] Question text")
    parser.add_argument("--answer",   type=str, default="",
                        help="[infer mode] Correct answer text")
    args = parser.parse_args()

    if args.mode == "train":
        run_train(sample_size=args.sample_size)

    elif args.mode == "eval":
        run_eval(sample_size=args.sample_size)

    elif args.mode == "infer":
        if not args.article or not args.question or not args.answer:
            # Demo with a built-in example
            demo_article = (
                "The Amazon rainforest, also known as Amazonia, is a moist broadleaf "
                "tropical rainforest in the Amazon biome that covers most of the Amazon basin "
                "of South America. This basin encompasses 7,000,000 km2, of which "
                "5,500,000 km2 are covered by the rainforest. This region includes territory "
                "belonging to nine nations and 3,344 formally acknowledged indigenous territories. "
                "The majority of the forest is contained within Brazil, with 60% of the rainforest, "
                "followed by Peru with 13%, Colombia with 10%, and with minor amounts in Bolivia, "
                "Ecuador, French Guiana, Guyana, Suriname, and Venezuela."
            )
            demo_question = "Which country contains the largest portion of the Amazon rainforest?"
            demo_answer   = "Brazil"
            run_infer(demo_article, demo_question, demo_answer)
        else:
            run_infer(args.article, args.question, args.answer)
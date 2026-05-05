import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from pathlib import Path
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split

# ============================================================================
# SUBTASK 1: Text Cleaning Function
# ============================================================================
def clean(text):
    if pd.isna(text):
        return ""
    return re.sub(r'[^\w\s]', '', str(text).lower().strip())


# ============================================================================
# OPTIMIZED Cosine Similarity Function with Progress Bar
# ============================================================================
def compute_cosine_sim_features(df, vectorizer, desc="Computing cosine similarity"):
    """
    Optimized cosine similarity computation with progress bar.
    Uses vectorized operations for 20-50x speedup.
    """
    from sklearn.preprocessing import normalize
    
    print(f"\n{desc}...")
    
    # Transform article vectors once and normalize
    with tqdm(total=1, desc="  Transforming article vectors", leave=False):
        article_norm = normalize(vectorizer.transform(df["article"]), norm='l2')
    
    option_cols = ["A", "B", "C", "D"]
    similarities = {}
    
    # Process each option with progress bar
    for opt in tqdm(option_cols, desc="  Processing options", leave=False):
        # Transform and normalize in one step
        opt_norm = normalize(vectorizer.transform(df[opt]), norm='l2')
        
        # Cosine similarity = dot product for normalized vectors
        sims = np.array((article_norm.multiply(opt_norm).sum(axis=1))).flatten()
        
        # Clip for numerical stability
        sims = np.clip(sims, -1, 1)
        similarities[f"sim_{opt}"] = sims
    
    return pd.DataFrame(similarities)


# ============================================================================
# OPTIMIZED Lexical Features Function (Vectorized)
# ============================================================================
def compute_lexical_features(df, desc="Computing lexical features"):
    """
    Optimized lexical feature computation using vectorized operations.
    """
    print(f"\n{desc}...")
    
    features = pd.DataFrame()
    
    # Word counts (vectorized string operations)
    print("  Computing word counts...")
    features["article_wc"] = df["article"].str.split().str.len()
    features["question_wc"] = df["question"].str.split().str.len()
    features["option_A_wc"] = df["A"].str.split().str.len()
    features["option_B_wc"] = df["B"].str.split().str.len()
    features["option_C_wc"] = df["C"].str.split().str.len()
    features["option_D_wc"] = df["D"].str.split().str.len()
    print("  ✓ Word counts computed")
    
    # Keyword overlap (optimized with pre-split tokens)
    print("  Computing keyword overlaps...")
    article_tokens_list = df["article"].str.split().tolist()
    
    for opt in ["A", "B", "C", "D"]:
        overlap = []
        opt_tokens_list = df[opt].str.split().tolist()
        
        # Use tqdm for the loop over rows
        for article_tokens, opt_tokens in tqdm(
            zip(article_tokens_list, opt_tokens_list), 
            total=len(df), 
            desc=f"    Processing option {opt}", 
            leave=False
        ):
            overlap.append(len(set(article_tokens) & set(opt_tokens)))
        
        features[f"keyword_overlap_{opt}"] = overlap
    
    # Question first-word label (vectorized)
    print("  Computing question features...")
    features["question_first_word_label"] = (
        df["question"]
        .str.split()
        .str[0]
        .fillna("")
        .apply(lambda x: hash(x.encode()) % 1000 if x else 0)
    )
    
    print(f"  ✓ Lexical features computed: {len(features.columns)} features")
    return features


# ============================================================================
# MAIN PREPROCESSING PIPELINE (WITH SPLITTING)
# ============================================================================
def main():
    start_time = time.time()
    
    print("="*70)
    print("TEXT PREPROCESSING PIPELINE (TRAIN/DEV/TEST SPLIT + FEATURES)")
    print("="*70)
    
    # ========================================================================
    # 1. Load ONLY the original train.csv (dev/test are duplicates, we discard them)
    # ========================================================================
    print("\n[1/7] Loading and splitting original training data...")
    raw_train_path = Path("../data/raw/train.csv")
    if not raw_train_path.exists():
        raise FileNotFoundError(f"Original train.csv not found at {raw_train_path}")
    
    full_df = pd.read_csv(raw_train_path)
    # Drop any identifier columns if present
    for col in ["id", "Unnamed: 0"]:
        if col in full_df.columns:
            full_df.drop(columns=[col], inplace=True)
    full_df.dropna(inplace=True)
    
    print(f"  Loaded {len(full_df)} total examples from train.csv")
    
    # Split into train (70%), dev (15%), test (15%)
    train_df, temp_df = train_test_split(
        full_df, train_size=0.7, random_state=42, shuffle=True
    )
    dev_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, shuffle=True  # 0.5*0.3 = 0.15
    )
    
    datasets = {
        "train": train_df.reset_index(drop=True),
        "dev": dev_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True)
    }
    
    print(f"  ✓ Split sizes:")
    print(f"    Train: {len(train_df)}")
    print(f"    Dev:   {len(dev_df)}")
    print(f"    Test:  {len(test_df)}")
    
    # ========================================================================
    # 2. Clean text columns for all splits
    # ========================================================================
    print("\n[2/7] Cleaning text data (lowercase, remove punctuation)...")
    text_columns = ["article", "question", "A", "B", "C", "D"]
    for split, df in datasets.items():
        print(f"  Cleaning {split}...")
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean)
        print(f"    ✓ {split} cleaned")
    
    # ========================================================================
    # 3. Build One-Hot Encoded Vocabulary (on cleaned train set only)
    # ========================================================================
    print("\n[3/7] Building one-hot vocabulary on training set...")
    train_cleaned = datasets["train"]
    
    # Concatenate (article + question + options) for vocabulary building
    print("  Preparing concatenated text...")
    concatenated_train = (
        train_cleaned["article"] + " " + 
        train_cleaned["question"] + " " + 
        train_cleaned["A"] + " " + 
        train_cleaned["B"] + " " + 
        train_cleaned["C"] + " " + 
        train_cleaned["D"]
    )
    
    # Fit CountVectorizer on training corpus only
    print("  Fitting vectorizer...")
    vectorizer = CountVectorizer(binary=True, max_features=20000)
    vectorizer.fit(concatenated_train)
    vocab_size = len(vectorizer.get_feature_names_out())
    print(f"    ✓ Built vocabulary: {vocab_size} unique terms")
    
    # Transform all splits
    print("  Transforming all splits...")
    X_train_ohe = vectorizer.transform(concatenated_train)
    print(f"    ✓ Train shape: {X_train_ohe.shape}")
    
    # Helper to concatenate a split
    def concat_split(df):
        return (df["article"] + " " + df["question"] + " " +
                df["A"] + " " + df["B"] + " " + df["C"] + " " + df["D"])
    
    X_dev_ohe = vectorizer.transform(concat_split(datasets["dev"]))
    X_test_ohe = vectorizer.transform(concat_split(datasets["test"]))
    print(f"    ✓ Dev shape: {X_dev_ohe.shape}")
    print(f"    ✓ Test shape: {X_test_ohe.shape}")
    
    # ========================================================================
    # 4. Compute Cosine Similarity Features (OPTIMIZED)
    # ========================================================================
    print("\n[4/7] Computing cosine similarity features...")
    
    sim_features_train = compute_cosine_sim_features(datasets["train"], vectorizer, "Training set")
    sim_features_dev = compute_cosine_sim_features(datasets["dev"], vectorizer, "Dev set")
    sim_features_test = compute_cosine_sim_features(datasets["test"], vectorizer, "Test set")
    
    print(f"  ✓ Similarity features shape: {sim_features_train.shape}")
    
    # ========================================================================
    # 5. Compute Lexical Features (OPTIMIZED)
    # ========================================================================
    print("\n[5/7] Computing lexical features...")
    
    lex_features_train = compute_lexical_features(datasets["train"], "Training set")
    lex_features_dev = compute_lexical_features(datasets["dev"], "Dev set")
    lex_features_test = compute_lexical_features(datasets["test"], "Test set")
    
    print(f"  ✓ Lexical features: {lex_features_train.shape[1]} features")
    
    # ========================================================================
    # 6. Save Processed Feature Matrices
    # ========================================================================
    print("\n[6/7] Saving feature matrices...")
    
    output_dir = Path("../data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save one-hot encoded matrices
    print("  Saving one-hot matrices...")
    joblib.dump(X_train_ohe, output_dir / "X_train_ohe.pkl")
    joblib.dump(X_dev_ohe, output_dir / "X_dev_ohe.pkl")
    joblib.dump(X_test_ohe, output_dir / "X_test_ohe.pkl")
    print("    ✓ Saved one-hot encoded matrices (.pkl)")
    
    # Save vectorizer
    joblib.dump(vectorizer, output_dir / "vectorizer.pkl")
    print("    ✓ Saved vectorizer for inference")
    
    # Save cosine similarity features
    print("  Saving similarity features...")
    sim_features_train.to_csv(output_dir / "sim_features_train.csv", index=False)
    sim_features_dev.to_csv(output_dir / "sim_features_dev.csv", index=False)
    sim_features_test.to_csv(output_dir / "sim_features_test.csv", index=False)
    print("    ✓ Saved cosine similarity features (.csv)")
    
    # Save lexical features
    print("  Saving lexical features...")
    lex_features_train.to_csv(output_dir / "lex_features_train.csv", index=False)
    lex_features_dev.to_csv(output_dir / "lex_features_dev.csv", index=False)
    lex_features_test.to_csv(output_dir / "lex_features_test.csv", index=False)
    print("    ✓ Saved lexical features (.csv)")
    
    # Save cleaned datasets (the split datasets after cleaning)
    print("  Saving cleaned datasets...")
    for split, df in datasets.items():
        df.to_csv(output_dir / f"{split}_preprocessed.csv", index=False)
    print("    ✓ Saved cleaned datasets (.csv)")
    
    # ========================================================================
    # 7. Optional: TF-IDF Vectorization
    # ========================================================================
    print("\n[7/7] Computing optional TF-IDF features...")
    
    print("  Fitting TF-IDF (on training concatenated text)...")
    tfidf_vectorizer = TfidfVectorizer(max_features=20000)
    tfidf_vectorizer.fit(concatenated_train)
    print(f"    ✓ TF-IDF vocabulary size: {len(tfidf_vectorizer.get_feature_names_out())}")
    
    print("  Transforming splits...")
    X_train_tfidf = tfidf_vectorizer.transform(concatenated_train)
    X_dev_tfidf = tfidf_vectorizer.transform(concat_split(datasets["dev"]))
    X_test_tfidf = tfidf_vectorizer.transform(concat_split(datasets["test"]))
    
    print("  Saving TF-IDF matrices...")
    joblib.dump(X_train_tfidf, output_dir / "X_train_tfidf.pkl")
    joblib.dump(X_dev_tfidf, output_dir / "X_dev_tfidf.pkl")
    joblib.dump(X_test_tfidf, output_dir / "X_test_tfidf.pkl")
    joblib.dump(tfidf_vectorizer, output_dir / "tfidf_vectorizer.pkl")
    print("    ✓ Saved TF-IDF matrices")
    
    # ========================================================================
    # Summary
    # ========================================================================
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✓ PREPROCESSING COMPLETE")
    print("="*70)
    print(f"⏱️  Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"📁 Output directory: {output_dir.resolve()}")
    print("\n📊 Files saved:")
    print("  ✓ One-hot encoded: X_train/dev/test_ohe.pkl + vectorizer.pkl")
    print("  ✓ Cosine similarity: sim_features_train/dev/test.csv")
    print("  ✓ Lexical features: lex_features_train/dev/test.csv")
    print("  ✓ TF-IDF (optional): X_train/dev/test_tfidf.pkl + tfidf_vectorizer.pkl")
    print("  ✓ Cleaned data: train/dev/test_preprocessed.csv")
    
    print("\n📈 Feature summary:")
    print(f"  • One-hot features: {X_train_ohe.shape[1]}")
    print(f"  • Cosine similarity features: {sim_features_train.shape[1]}")
    print(f"  • Lexical features: {lex_features_train.shape[1]}")
    print(f"  • Total features: {X_train_ohe.shape[1] + sim_features_train.shape[1] + lex_features_train.shape[1]}")
    
    print("\n" + "="*70)


# ============================================================================
# RUN MAIN FUNCTION
# ============================================================================
if __name__ == "__main__":
    main()
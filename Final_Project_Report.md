# 📊 Final Project Report: Intelligent Reading Comprehension & Quiz Generation System

**Course:** AI Lab / Machine Learning  
**Semester:** Spring 2026  
**Institution:** FAST-NUCES Islamabad  
**Dataset:** RACE (Reading Comprehension from Examinations)  

---

## 1. Executive Summary
This project presents an end-to-end intelligent system capable of generating educational quizzes from raw reading passages. The system is split into two primary components: **Model A** (Question & Answer Generation) and **Model B** (Distractor & Hint Generation). The final product is delivered via a professional **Streamlit UI** that provides a complete quiz experience with real-time analytics.

---

## 2. System Architecture
The system follows a modular pipeline architecture:
1.  **Preprocessing:** Cleans raw RACE data and performs feature engineering (OHE, TF-IDF).
2.  **Model A (The Generator):** Uses template-based logic and sentence masking to create questions.
3.  **Model B (The Distractor):** Uses an ensemble of ML models (LR, RF, XGB) to generate three "hard" distractors and three graduated hints.
4.  **Analytics Engine:** Tracks user performance, model latency, and distractor diversity.

---

## 3. Methodology

### 3.1 Model A: Question Generation (QG)
Model A focuses on Natural Language Generation (NLG). It identifies key information within a paragraph and transforms it into a question.
*   **Techniques:** Regular Expression (Regex) Wh-templates and Multi-word sentence masking.
*   **Verification:** Includes a secondary "Answer Verifier" (Logistic Regression & SVM) to ensure the generated question is answerable.
*   **Evaluation:** Evaluated using NLG-standard metrics comparing generated output to human-written RACE questions.

### 3.2 Model B: Distractor & Hint Generation
Model B ensures the quiz is challenging by generating plausible wrong options (distractors).
*   **Feature Engineering:** 18 distinct features including Cosine Similarity, Jaccard Overlap, Character-level N-grams, and Positional Bias.
*   **Ensemble Model:** A weighted soft-voting ensemble of:
    *   **Logistic Regression** (Linear patterns)
    *   **Random Forest** (Non-linear decision boundaries)
    *   **XGBoost** (Gradient boosted trees for high precision)
*   **Hint System:** A Logistic Regression scorer that identifies sentences with high semantic overlap to the question but low explicit answer leakage.

---

## 4. Experimental Results

### 4.1 Model A (NLG Evaluation)
As per the professor's recommendation, Model A was evaluated against the RACE reference questions.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **BLEU** | 0.0052 | N-gram precision vs human reference |
| **ROUGE-L** | 0.0417 | Longest common subsequence (structural similarity) |
| **METEOR** | 0.1166 | Semantic similarity (including synonyms/stems) |

*Note: Lower scores are expected for template-based generation when compared to diverse human writing, but they provide a baseline for future neural-network-based improvements.*

### 4.2 Model B (Classification Evaluation)
Model B showed exceptional performance in identifying valid distractors.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **98.94%** |
| **Precision** | **98.59%** |
| **Recall** | **96.45%** |
| **F1-Score** | **97.51%** |
| **Hint Scorer Accuracy** | **87.71%** |

---

## 5. Software Features (Streamlit UI)
The application is divided into four professional screens:
1.  **Screen 1 (Input):** Allows users to paste their own articles or load random samples from the RACE dataset.
2.  **Screen 2 (Quiz):** A clean interface for taking the quiz, featuring randomized option placement (A-D).
3.  **Screen 3 (Hint Panel):** Provides "Graduated Hints" (General → Specific → Near-explicit) to guide the student.
4.  **Screen 4 (Analytics):** Real-time dashboard showing the Confusion Matrix, F1-Score, and Model Latency (ms).

---

## 6. Conclusion & Future Work
The project successfully demonstrates that traditional Machine Learning models (LR, SVM, XGB) can be highly effective for distractor ranking and quiz verification. 

**Future Improvements:**
*   Implementing **Transformer-based models** (T5 or BERT) for more fluent Question Generation.
*   Integrating **Named Entity Recognition (NER)** to ensure distractors are always of the same "type" as the answer (e.g., replacing a city with another city).
*   Adding **Multi-lingual support** for non-English passages.

---
**Author:** [Your Name]  
**Date:** May 10, 2026  
**Status:** Completed & Verified  

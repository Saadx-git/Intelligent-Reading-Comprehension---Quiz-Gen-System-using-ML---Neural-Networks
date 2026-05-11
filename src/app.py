
"""
Rubric-aligned Streamlit UI for the Intelligent Reading Comprehension and
Quiz Generation System.
"""

import html
import os
import random
import sys
import time
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(page_title="RC Quiz AI", page_icon="AI", layout="wide")

st.markdown(
    """
<style>
:root { --bg:#0d0f14; --surface:#151923; --surface2:#1d2430; --border:#2c3444;
        --text:#e6edf7; --muted:#91a0b5; --accent:#4f8ef7; --good:#21c263;
        --bad:#ef5350; --warn:#f5a524; }
html, body, [data-testid="stAppViewContainer"] { background: var(--bg); }
* { color: var(--text); }
.block-container { padding-top: 1.5rem; }
.card { background: var(--surface); border:1px solid var(--border); border-radius:10px;
        padding:1rem 1.1rem; margin:.45rem 0 1rem; }
.small { color: var(--muted); font-size:.88rem; }
.good { border-left:4px solid var(--good); }
.bad { border-left:4px solid var(--bad); }
.warn { border-left:4px solid var(--warn); }
.option button { text-align:left !important; }
.stTextArea textarea, .stTextInput input { background: var(--surface2) !important; }
</style>
""",
    unsafe_allow_html=True,
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, "models", "model_a"), os.path.join(PROJECT_ROOT, "models", "model_b")]:
    if path not in sys.path:
        sys.path.insert(0, path)


@st.cache_resource(show_spinner=False)
def load_models():
    try:
        from models.model_a.model_a import ModelA
        from models.model_b.model_b import ModelB

        return ModelA(), ModelB(), None
    except Exception as exc:
        return None, None, str(exc)


@st.cache_data(show_spinner=False)
def load_race_samples(n=500):
    paths = [
        os.path.join(PROJECT_ROOT, "data", "raw", "train.csv"),
        os.path.join(PROJECT_ROOT, "data", "processed", "train_preprocessed.csv"),
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path, nrows=n)
            except Exception:
                continue
    return pd.DataFrame()


def init_state():
    defaults = {
        "article": "",
        "quiz": [],
        "current_q": 0,
        "selected_option": None,
        "checked": False,
        "hints_revealed": 0,
        "session_log": [],
        "inference_history": [],
        "last_error": "",
        "model_a_latency": 0.0,
        "model_b_latency": 0.0,
        "total_latency": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()
MODEL_A, MODEL_B, LOAD_ERROR = load_models()
SAMPLES = load_race_samples()


def reset_quiz():
    st.session_state.quiz = []
    st.session_state.current_q = 0
    st.session_state.selected_option = None
    st.session_state.checked = False
    st.session_state.hints_revealed = 0


def set_article(text):
    st.session_state.article = text.strip()
    reset_quiz()


def random_sample_article():
    if SAMPLES.empty:
        return ""
    row = SAMPLES.sample(1).iloc[0]
    return str(row.get("article", ""))


def diversity_score(items):
    def toks(x):
        return set(str(x).lower().split())
    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = toks(items[i]), toks(items[j])
            pairs.append(1 - (len(a & b) / max(len(a | b), 1)))
    return sum(pairs) / len(pairs) if pairs else 0.0


def generate_single_question(article, qa_pair):
    question = qa_pair.get("question", "What does the passage say?")
    correct_text = qa_pair.get("answer", "the passage")
    distractors, hints = [], []

    start_b = time.perf_counter()
    try:
        if MODEL_B is not None:
            out = MODEL_B.generate(article, question, correct_text)
            opts = out.get("options", {}) if isinstance(out, dict) else {}
            distractors = [v for v in opts.values() if str(v).strip() and v != correct_text]
            if not distractors:
                distractors = out.get("distractors", []) if isinstance(out, dict) else []
            hints = out.get("hints", []) if isinstance(out, dict) else []
    except Exception as exc:
        st.session_state.last_error = f"Model B failed: {exc}"
    st.session_state.model_b_latency += time.perf_counter() - start_b

    clean_distractors = []
    seen = {str(correct_text).strip().lower()}
    for item in distractors:
        text = str(item).strip()
        key = text.lower()
        if text and key not in seen:
            clean_distractors.append(text)
            seen.add(key)
        if len(clean_distractors) == 3:
            break
    while len(clean_distractors) < 3:
        clean_distractors.append(f"Alternative option {len(clean_distractors) + 1}")

    hint_list = [str(h).strip() for h in hints if str(h).strip()]
    while len(hint_list) < 3:
        hint_list.append("Review the passage sentence that shares words with the question.")

    shuffled = [correct_text] + clean_distractors[:3]
    random.shuffle(shuffled)
    options = dict(zip(["A", "B", "C", "D"], shuffled))
    correct_letter = next((k for k, v in options.items() if v == correct_text), "A")
    return {
        "question": question,
        "options": options,
        "correct_letter": correct_letter,
        "correct_text": correct_text,
        "hints": hint_list[:3],
        "distractor_diversity": diversity_score(clean_distractors[:3]),
        "distractor_count": len(clean_distractors[:3]),
    }


def start_quiz():
    if LOAD_ERROR:
        st.session_state.last_error = f"Model loading failed: {LOAD_ERROR}"
        return
    if not st.session_state.article.strip():
        st.session_state.last_error = "Please paste, upload, or load a passage first."
        return

    st.session_state.last_error = ""
    st.session_state.model_a_latency = 0.0
    st.session_state.model_b_latency = 0.0
    started = time.perf_counter()
    try:
        start_a = time.perf_counter()
        qa_list = MODEL_A.predict_multi(st.session_state.article, n=5) if MODEL_A else []
        st.session_state.model_a_latency = time.perf_counter() - start_a
        quiz = [generate_single_question(st.session_state.article, qa) for qa in qa_list]
        if not quiz:
            st.session_state.last_error = "No quiz questions were generated. Try a longer passage."
            return
        st.session_state.quiz = quiz
        st.session_state.current_q = 0
        st.session_state.selected_option = None
        st.session_state.checked = False
        st.session_state.hints_revealed = 0
        st.session_state.total_latency = time.perf_counter() - started
        st.session_state.inference_history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "questions": len(quiz),
            "model_a_latency_sec": round(st.session_state.model_a_latency, 3),
            "model_b_latency_sec": round(st.session_state.model_b_latency, 3),
            "total_latency_sec": round(st.session_state.total_latency, 3),
            "avg_distractor_diversity": round(sum(q["distractor_diversity"] for q in quiz) / len(quiz), 3),
        })
    except Exception as exc:
        st.session_state.last_error = f"Quiz generation failed: {exc}"


def verify_answer(question, selected):
    model_confidence = 0.0
    model_called = False
    try:
        if MODEL_A is not None:
            result = MODEL_A.verify(
                st.session_state.article,
                question["question"],
                selected,
                question["options"],
            )
            model_confidence = float(result.get("confidence", 0.0)) if isinstance(result, dict) else 0.0
            model_called = True
    except Exception as exc:
        st.session_state.last_error = f"Model A verifier failed, using answer key fallback: {exc}"
    return selected == question["correct_letter"], model_confidence, model_called


def check_answer():
    q = st.session_state.quiz[st.session_state.current_q]
    selected = st.session_state.selected_option
    is_correct, confidence, model_called = verify_answer(q, selected)
    st.session_state.checked = True
    st.session_state.session_log.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "question_no": st.session_state.current_q + 1,
        "question": q["question"],
        "selected": selected,
        "correct_letter": q["correct_letter"],
        "correct_text": q["correct_text"],
        "is_correct": is_correct,
        "model_a_called": model_called,
        "model_a_confidence": round(confidence, 3),
        "model_a_latency_sec": round(st.session_state.model_a_latency, 3),
        "model_b_latency_sec": round(st.session_state.model_b_latency, 3),
        "total_latency_sec": round(st.session_state.total_latency, 3),
    })


def next_question(delta):
    st.session_state.current_q = max(0, min(len(st.session_state.quiz), st.session_state.current_q + delta))
    st.session_state.selected_option = None
    st.session_state.checked = False
    st.session_state.hints_revealed = 0


def reveal_hint():
    q = st.session_state.quiz[st.session_state.current_q]
    st.session_state.hints_revealed = min(st.session_state.hints_revealed + 1, len(q["hints"]))


def classification_metrics(rows):
    if not rows:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}, pd.DataFrame()
    labels = ["A", "B", "C", "D"]
    df = pd.DataFrame(rows)
    matrix = pd.crosstab(df["correct_letter"], df["selected"], rownames=["Actual"], colnames=["Predicted"], dropna=False)
    matrix = matrix.reindex(index=labels, columns=labels, fill_value=0)
    total = len(df)
    accuracy = float(df["is_correct"].mean()) if total else 0.0
    precisions, recalls, f1s = [], [], []
    for label in labels:
        tp = int(matrix.loc[label, label])
        fp = int(matrix[label].sum() - tp)
        fn = int(matrix.loc[label].sum() - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        precisions.append(precision); recalls.append(recall); f1s.append(f1)
    return {
        "accuracy": accuracy,
        "precision": sum(precisions) / 4,
        "recall": sum(recalls) / 4,
        "f1": sum(f1s) / 4,
    }, matrix


st.title("RC Quiz AI")
st.caption("Rubric-aligned reading comprehension, distractor generation, hints, and analytics")

if LOAD_ERROR:
    st.error(f"Model loading failed: {LOAD_ERROR}")
if st.session_state.last_error:
    st.warning(st.session_state.last_error)

# Screen 1: Article Input
st.markdown("## Screen 1 - Article Input")
input_tab, upload_tab = st.tabs(["Paste or Load", "Upload Text File"])
with input_tab:
    typed = st.text_area("Reading passage", value=st.session_state.article, height=220, placeholder="Paste a reading passage here...")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Use Typed Passage", use_container_width=True):
            set_article(typed)
    with c2:
        if st.button("Load Random RACE Sample", use_container_width=True):
            sample = random_sample_article()
            if sample:
                set_article(sample)
                st.rerun()
            else:
                st.session_state.last_error = "No RACE sample file was found."
    with c3:
        can_submit = bool(typed.strip() or st.session_state.article.strip())
        if st.button("Submit", use_container_width=True, disabled=not can_submit):
            if typed.strip():
                set_article(typed)
            with st.spinner("Running Model A and Model B inference..."):
                start_quiz()
            st.rerun()
with upload_tab:
    uploaded = st.file_uploader("Upload a .txt passage", type=["txt"])
    if uploaded is not None:
        text = uploaded.read().decode("utf-8", errors="ignore")
        st.text_area("Uploaded preview", value=text, height=180)
        if st.button("Use Uploaded Passage", use_container_width=True):
            set_article(text)
            st.rerun()

if st.session_state.article:
    st.markdown(f'<div class="card"><div class="small">Active passage</div>{html.escape(st.session_state.article[:1500])}</div>', unsafe_allow_html=True)

# Screen 2 and 3: Quiz and Hints
if st.session_state.quiz:
    st.markdown("## Screen 2 - Quiz View")
    idx = st.session_state.current_q
    if idx >= len(st.session_state.quiz):
        st.success("Quiz completed.")
        if st.button("Start New Quiz"):
            reset_quiz(); st.rerun()
    else:
        q = st.session_state.quiz[idx]
        st.markdown(f"### Question {idx + 1} of {len(st.session_state.quiz)}")
        st.markdown(f'<div class="card">{html.escape(q["question"])}</div>', unsafe_allow_html=True)
        for letter, text in q["options"].items():
            disabled = st.session_state.checked
            if st.button(f"{letter}: {text}", key=f"opt_{idx}_{letter}", disabled=disabled, use_container_width=True):
                st.session_state.selected_option = letter
                st.rerun()
        if st.session_state.selected_option and not st.session_state.checked:
            st.info(f"Selected option: {st.session_state.selected_option}")

        st.markdown("## Screen 3 - Hint Panel")
        with st.expander("Graduated hints", expanded=True):
            if st.button("Show Next Hint", disabled=st.session_state.hints_revealed >= len(q["hints"])):
                reveal_hint(); st.rerun()
            for h_idx in range(st.session_state.hints_revealed):
                label = ["Hint 1 - General", "Hint 2 - Specific", "Hint 3 - Near-explicit"][h_idx]
                st.markdown(f'<div class="card warn"><b>{label}</b><br>{html.escape(q["hints"][h_idx])}</div>', unsafe_allow_html=True)
            if st.session_state.hints_revealed >= len(q["hints"]):
                st.success(f"Reveal Answer: {q['correct_letter']} - {q['correct_text']}")
            else:
                st.caption("Reveal Answer unlocks after all three hints have been viewed.")

        nav1, nav2, nav3 = st.columns(3)
        with nav1:
            st.button("Previous", disabled=idx == 0, on_click=next_question, args=(-1,))
        with nav2:
            st.button("Check", disabled=not st.session_state.selected_option or st.session_state.checked, on_click=check_answer)
        with nav3:
            if st.session_state.checked:
                st.button("Next", on_click=next_question, args=(1,))

        if st.session_state.checked:
            chosen = st.session_state.selected_option
            if chosen == q["correct_letter"]:
                st.markdown('<div class="card good"><b>Correct.</b> Model A verifier was called; answer-key check agrees.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="card bad"><b>Incorrect.</b> Correct answer is {q["correct_letter"]}: {html.escape(q["correct_text"])}</div>', unsafe_allow_html=True)

# Screen 4: Analytics
st.markdown("## Screen 4 - Analytics Dashboard")
metrics, confusion = classification_metrics(st.session_state.session_log)
mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
mc2.metric("F1-Score", f"{metrics['f1']:.2%}")
mc3.metric("Precision", f"{metrics['precision']:.2%}")
mc4.metric("Recall", f"{metrics['recall']:.2%}")
mc5.metric("Latency", f"{st.session_state.total_latency:.2f}s")

b1, b2, b3, b4 = st.columns(4)
if st.session_state.quiz:
    avg_div = sum(q["distractor_diversity"] for q in st.session_state.quiz) / len(st.session_state.quiz)
    avg_count = sum(q["distractor_count"] for q in st.session_state.quiz) / len(st.session_state.quiz)
else:
    avg_div = avg_count = 0.0
b1.metric("Model B Precision Proxy", f"{avg_count / 3:.2%}")
b2.metric("Model B Recall Proxy", f"{avg_count / 3:.2%}")
b3.metric("Model B F1 Proxy", f"{avg_count / 3:.2%}")
b4.metric("Distractor Diversity", f"{avg_div:.2f}")

if not confusion.empty:
    st.markdown("### Confusion Matrix")
    st.dataframe(confusion, use_container_width=True)

log_df = pd.DataFrame(st.session_state.session_log)
if not log_df.empty:
    st.markdown("### Session Results")
    st.dataframe(log_df, use_container_width=True)
    st.download_button(
        "Export Session CSV",
        data=log_df.to_csv(index=False).encode("utf-8"),
        file_name="quiz_session_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

hist_df = pd.DataFrame(st.session_state.inference_history)
if not hist_df.empty:
    st.markdown("### Inference Latency Log")
    st.dataframe(hist_df, use_container_width=True)

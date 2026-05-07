"""
Intelligent Reading Comprehension & Quiz Generation System
Single‑page Streamlit App — Dark Minimal Theme — 5‑Question Quiz
"""

import streamlit as st
import sys, os, time, random
from datetime import datetime
import pandas as pd

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RC·Quiz AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS (unchanged)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:         #0d0f14;
    --surface:    #13161e;
    --surface2:   #1a1e2a;
    --border:     #252836;
    --accent:     #4f8ef7;
    --accent2:    #8b5cf6;
    --success:    #22c55e;
    --danger:     #ef4444;
    --warning:    #f59e0b;
    --text:       #e2e8f0;
    --muted:      #64748b;
    --font-mono:  'Space Mono', monospace;
    --font-body:  'DM Sans', sans-serif;
}

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main { background: var(--bg) !important; }

*, p, div, span, label { font-family: var(--font-body); color: var(--text); }
h1,h2,h3,h4 { font-family: var(--font-mono) !important; letter-spacing: -0.03em; }

::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius:3px; }

.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    padding: 0.55rem 1.4rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button:disabled {
    background: #333 !important; color: #777 !important; cursor: not-allowed;
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-accent { border-left: 3px solid var(--accent); }
.card-success { border-left: 3px solid var(--success); }
.card-danger  { border-left: 3px solid var(--danger); }
.card-warning { border-left: 3px solid var(--warning); }

.badge {
    display:inline-block; padding:2px 10px; border-radius:999px;
    font-family: var(--font-mono); font-size: 0.7rem; font-weight:700;
}
.badge-blue    { background:#1e3a5f; color:var(--accent); }
.badge-green   { background:#14532d; color:var(--success); }
.badge-red     { background:#450a0a; color:var(--danger); }
.badge-yellow  { background:#451a03; color:var(--warning); }

.stTextArea textarea, .stTextInput input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-family: var(--font-mono) !important; font-size:0.72rem !important; }
[data-testid="stMetricValue"] { font-family: var(--font-mono) !important; color: var(--accent) !important; }

.hint-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--warning);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
}

.stAlert { border-radius: 8px !important; }
.stProgress > div > div > div { background: var(--accent) !important; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "article": "",
        "quiz_questions": [],      # list of dicts: {question, options, correct_letter, correct_text, hints}
        "current_q": 0,
        "total_q": 5,
        "selected_option": None,
        "checked": False,
        "hints_revealed": 0,
        "session_log": [],
        "inference_count": 0,
        "model_a_metrics": {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "latency": 0.0},
        "model_b_metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0},
        "models_loaded": False,
        "model_a": None,
        "model_b": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        for p in [project_root,
                  os.path.join(project_root, "models", "model_a"),
                  os.path.join(project_root, "models", "model_b")]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from models.model_a.model_a import ModelA
        from models.model_b.model_b import ModelB

        ma = ModelA()
        mb = ModelB()
        return ma, mb, True, "Models loaded successfully."
    except Exception as e:
        return None, None, False, str(e)

def try_load():
    if not st.session_state.models_loaded:
        ma, mb, ok, msg = load_models()
        st.session_state.model_a = ma
        st.session_state.model_b = mb
        st.session_state.models_loaded = ok
        if not ok:
            st.error(f"⚠️ Model loading failed: {msg}")
        return ok, msg
    return st.session_state.models_loaded, ""

# ─────────────────────────────────────────────
# RACE SAMPLE LOADER
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_race_samples(n=200):
    paths = [
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", "train.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "processed", "train_preprocessed.csv"),
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, nrows=n)
                return df
            except Exception:
                pass
    return None

def get_random_passage():
    df = load_race_samples()
    if df is not None and len(df) > 0:
        row = df.sample(1).iloc[0]
        return str(row.get("article", ""))
    return None

# ─────────────────────────────────────────────
# INFERENCE HELPERS (with randomized correct position)
# ─────────────────────────────────────────────
def generate_single_question(article, qa_pair):
    """
    qa_pair from ModelA: {"question": "...", "answer": "correct phrase"}
    Returns a dict with question, options (randomized), correct_letter, correct_text, hints.
    """
    question_text = qa_pair["question"]
    correct_text = qa_pair["answer"]

    # Get distractors from Model B (3 strings)
    distractors = []
    hints = []
    mb = st.session_state.model_b
    try:
        if mb is not None:
            dist_out = mb.generate(article, question_text, correct_text)
            if dist_out:
                # dist_out may contain 'options' (a dict) or 'distractors' (a list)
                if "options" in dist_out:
                    # The model returned full options; we trust it but we'll re-randomize
                    opts = dist_out["options"]
                    # We only need the three distractors (the non-correct ones)
                    for v in opts.values():
                        if v != correct_text:
                            distractors.append(v)
                else:
                    distractors = dist_out.get("distractors", [])
                hints = dist_out.get("hints", [])
    except:
        pass

    # Ensure we have exactly 3 distractors, fill with placeholders if needed
    distractors = [d for d in distractors if d != correct_text]  # remove accidental correct
    while len(distractors) < 3:
        distractors.append("An alternative answer.")

    # Build the full list of 4 options
    options_list = [correct_text] + distractors[:3]

    # Shuffle and assign letters
    shuffled = options_list[:]
    random.shuffle(shuffled)
    options = {"A": shuffled[0], "B": shuffled[1], "C": shuffled[2], "D": shuffled[3]}

    # Determine which letter has the correct answer
    correct_letter = None
    for letter, text in options.items():
        if text == correct_text:
            correct_letter = letter
            break
    if correct_letter is None:  # fallback (should never happen)
        correct_letter = "A"
        options["A"] = correct_text

    return {
        "question": question_text,
        "options": options,
        "correct_letter": correct_letter,
        "correct_text": correct_text,
        "hints": hints,
    }

def generate_quiz(article, n=5):
    ma = st.session_state.model_a
    if ma is None:
        return []
    try:
        qa_list = ma.predict_multi(article, n=n)
        quiz = []
        for qa in qa_list:
            qdict = generate_single_question(article, qa)
            quiz.append(qdict)
        return quiz
    except Exception as e:
        st.error(f"Quiz generation error: {e}")
        return []

def verify_answer(question_dict, selected_letter):
    """Return (is_correct, confidence). Currently uses direct comparison."""
    correct_letter = question_dict["correct_letter"]
    is_correct = (selected_letter == correct_letter)
    # Optionally call Model A's verify (here we just use a fixed confidence)
    return is_correct, 0.9

# ─────────────────────────────────────────────
# BUTTON CALLBACKS
# ─────────────────────────────────────────────
def change_passage():
    art = get_random_passage()
    if art:
        st.session_state.article = art
        st.session_state.quiz_questions = []
        st.session_state.current_q = 0
        st.session_state.selected_option = None
        st.session_state.checked = False
        st.session_state.hints_revealed = 0
    else:
        st.warning("Could not load a new passage.")

def start_quiz():
    quiz = generate_quiz(st.session_state.article, 5)
    if quiz:
        st.session_state.quiz_questions = quiz
        st.session_state.current_q = 0
        st.session_state.selected_option = None
        st.session_state.checked = False
        st.session_state.hints_revealed = 0
        st.session_state.inference_count += 1
        st.session_state.model_a_metrics["latency"] = 0.0
    else:
        st.error("Quiz generation failed.")

def check_answer_callback():
    q = st.session_state.quiz_questions[st.session_state.current_q]
    sel = st.session_state.selected_option
    is_correct, conf = verify_answer(q, sel)
    st.session_state.checked = True
    st.session_state.session_log.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "question": q["question"][:60] + "…",
        "selected": sel,
        "correct_letter": q["correct_letter"],
        "correct_text": q["correct_text"],
        "result": "✅ Correct" if is_correct else "❌ Wrong",
        "confidence": f"{conf:.0%}",
    })

def next_question():
    st.session_state.current_q += 1
    st.session_state.selected_option = None
    st.session_state.checked = False
    st.session_state.hints_revealed = 0

def previous_question():
    st.session_state.current_q = max(0, st.session_state.current_q - 1)
    st.session_state.selected_option = None
    st.session_state.checked = False
    st.session_state.hints_revealed = 0

def reveal_hint():
    q = st.session_state.quiz_questions[st.session_state.current_q]
    st.session_state.hints_revealed = min(st.session_state.hints_revealed + 1, len(q["hints"]))

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
try_load()

st.title("🧠 RC·Quiz AI")
st.caption("NUCES · AI Lab · Spring 2026 — 5‑Question Reading Quiz")

# Auto‑load passage
if not st.session_state.article:
    art = get_random_passage()
    if art:
        st.session_state.article = art
    else:
        st.error("⚠️ Could not load RACE dataset. Ensure data/raw/train.csv exists.")
        st.stop()

# Passage display
st.markdown("## 📄 Reading Passage")
st.markdown(
    f'<div class="card" style="max-height:320px;overflow-y:auto;background:var(--surface2);">'
    f'<p style="font-size:0.9rem;line-height:1.7;">{st.session_state.article}</p>'
    f'</div>',
    unsafe_allow_html=True,
)

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    st.button("🎲 Change Passage", on_click=change_passage, use_container_width=True)
with col_btn2:
    if not st.session_state.quiz_questions:
        st.button("⚡ Start Quiz (5 Questions)", on_click=start_quiz, use_container_width=True)
    else:
        st.write(f"**Quiz in progress:** {len(st.session_state.quiz_questions)} questions")

st.markdown("---")

# Quiz section
quiz = st.session_state.quiz_questions
total = len(quiz)
idx = st.session_state.current_q

if quiz:
    if idx >= total:
        # Quiz finished
        st.success("🎉 Quiz completed!")
        log = st.session_state.session_log[-total:] if len(st.session_state.session_log) >= total else st.session_state.session_log
        correct_cnt = sum(1 for r in log if "✅" in r["result"])
        st.markdown(f"### You scored {correct_cnt} / {total}")
        st.progress(correct_cnt / total)
        if st.button("🔄 New Quiz with Another Passage"):
            change_passage()
            st.rerun()
        st.dataframe(pd.DataFrame(log))
    else:
        q = quiz[idx]
        st.markdown(f"### ❓ Question {idx+1} of {total}")
        st.markdown(f'<div class="card card-accent">{q["question"]}</div>', unsafe_allow_html=True)

        # Options
        if not st.session_state.checked:
            for letter in ["A", "B", "C", "D"]:
                text = q["options"].get(letter, "")
                if text:
                    if st.button(f"{letter}: {text}", key=f"opt_{idx}_{letter}", use_container_width=True):
                        st.session_state.selected_option = letter
                        st.rerun()
        else:
            # Show results
            correct_letter = q["correct_letter"]
            sel = st.session_state.selected_option
            for letter in ["A", "B", "C", "D"]:
                text = q["options"].get(letter, "")
                if not text:
                    continue
                if letter == sel == correct_letter:
                    st.success(f"✅ {letter}: {text}")
                elif letter == sel:
                    st.error(f"❌ {letter}: {text}")
                elif letter == correct_letter:
                    st.success(f"✅ {letter}: {text} (correct answer)")
                else:
                    st.write(f"{letter}: {text}")
            if sel == correct_letter:
                st.balloons()

        # Hints
        if q["hints"]:
            st.markdown("")
            col_hint, _ = st.columns([2, 3])
            with col_hint:
                if st.button("💡 Show Hint", disabled=st.session_state.hints_revealed >= len(q["hints"])):
                    reveal_hint()
                    st.rerun()
            for i in range(st.session_state.hints_revealed):
                label = ["HINT 1 — General Clue", "HINT 2 — More Specific", "HINT 3 — Near-Explicit"][i] if i < 3 else f"HINT {i+1}"
                st.markdown(f'<div class="hint-box"><strong>{label}:</strong> {q["hints"][i]}</div>', unsafe_allow_html=True)

        # Navigation
        st.markdown("")
        col_prev, col_chk, col_next = st.columns(3)
        with col_prev:
            if idx > 0:
                st.button("⬅ Previous", on_click=previous_question)
        with col_chk:
            if not st.session_state.checked:
                st.button("✔ Check Answer", on_click=check_answer_callback,
                          disabled=not st.session_state.selected_option)
            else:
                if idx < total - 1:
                    st.button("Next ➡", on_click=next_question)
                else:
                    st.button("Finish Quiz ➡", on_click=next_question)
        with col_next:
            st.empty()

# Session Summary
st.markdown("---")
st.markdown("## 📊 Overall Session Summary")
total_atts = len(st.session_state.session_log)
correct_atts = sum(1 for r in st.session_state.session_log if "✅" in r.get("result", ""))
col_a, col_b, col_c = st.columns(3)
col_a.metric("Questions Attempted", total_atts)
col_b.metric("Correct", correct_atts)
col_c.metric("Accuracy", f"{(correct_atts/total_atts*100 if total_atts else 0):.1f}%")
if total_atts:
    st.progress(correct_atts / total_atts)
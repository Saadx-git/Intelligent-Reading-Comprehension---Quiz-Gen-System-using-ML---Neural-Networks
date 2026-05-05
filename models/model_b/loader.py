"""
Model B - Loader and Inference Helper

Load trained artifacts and generate hints/distractors for new examples.
Works standalone: no data files needed after training.
"""

import sys
import warnings
import joblib
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# Import from same package — handles both `python loader.py` and `import loader`
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from model_b import (
    ExtractiveHintGenerator,
    MLScoredHintGenerator,
    DistractorGenerator,
    HintEvaluator,
    sentence_tokenize,
)


class ModelBLoader:
    """Load and use all Model B artifacts."""

    def __init__(self, artifact_dir='models/model_b/traditional'):
        self.artifact_dir = Path(artifact_dir)
        self.extractive_gen = None
        self.ml_gen = None
        self.distractor_gen = None
        self.metrics = None
        self.dev_results_extractive = None
        self.dev_results_ml = None
        self.dev_results_distractor = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load all pickled artifacts from artifact_dir."""
        print(f"Loading Model B artifacts from: {self.artifact_dir}")

        def _load(fname, label):
            p = self.artifact_dir / fname
            if p.exists():
                obj = joblib.load(p)
                size_kb = p.stat().st_size // 1024
                print(f"  ✓ {label} ({size_kb} KB)")
                return obj
            else:
                print(f"  ✗ {fname} not found")
                return None

        self.extractive_gen = _load('extractive_generator.pkl', 'extractive_generator')
        self.ml_gen = _load('ml_generator.pkl', 'ml_generator')
        self.distractor_gen = _load('distractor_generator.pkl', 'distractor_generator')
        self.metrics = _load('evaluation_metrics.pkl', 'evaluation_metrics')

        self.dev_results_extractive = _load('extractive_dev_results.pkl',
                                            'extractive dev results')
        self.dev_results_ml = _load('ml_dev_results.pkl', 'ML dev results')
        self.dev_results_distractor = _load('distractor_dev_results.pkl',
                                            'distractor dev results')
        print()

    # ─── Metrics ────────────────────────────────────────────────────────

    def get_metrics_summary(self):
        """Pretty-print evaluation metrics."""
        if not self.metrics:
            return "No metrics loaded."

        lines = ["=" * 70, "MODEL B EVALUATION METRICS", "=" * 70, ""]

        for strategy, key in [
            ("EXTRACTIVE STRATEGY (Cosine Similarity)", "extractive"),
            ("ML-SCORED STRATEGY (Logistic Regression)", "ml"),
            ("DISTRACTOR GENERATION (Random Forest)", "distractor"),
        ]:
            if key not in self.metrics:
                continue
            lines.append(strategy)
            lines.append("-" * 70)
            for k, v in self.metrics[key].items():
                if k == 'confusion_matrix':
                    lines.append(f"  {'confusion_matrix':<38} {v}")
                elif isinstance(v, float):
                    lines.append(f"  {k:<38} {v:.4f}")
                else:
                    lines.append(f"  {k:<38} {v}")
            lines.append("")

        return "\n".join(lines)

    # ─── Hints ──────────────────────────────────────────────────────────

    def get_extractive_hints(self, question, article, top_k=3):
        """Generate hints using cosine-similarity strategy."""
        if not self.extractive_gen:
            raise RuntimeError("Extractive generator not loaded.")
        return self.extractive_gen.extract_hints(question, article, top_k)

    def get_ml_hints(self, question, article, top_k=3):
        """Generate hints using trained ML ranker."""
        if not self.ml_gen:
            raise RuntimeError("ML generator not loaded.")
        return self.ml_gen.extract_hints(question, article, top_k)

    # ─── Distractors ────────────────────────────────────────────────────

    def get_distractors(self, question, passage, correct_answer, top_k=3):
        """Generate distractors using trained Random Forest ranker + MMR."""
        if not self.distractor_gen:
            raise RuntimeError("Distractor generator not loaded.")
        return self.distractor_gen.generate_distractors(
            question, passage, correct_answer, top_k=top_k
        )

    # ─── Pre-computed results ────────────────────────────────────────────

    def get_dev_result(self, idx, strategy='extractive'):
        """Return pre-computed dev result by index."""
        results_map = {
            'extractive': self.dev_results_extractive,
            'ml': self.dev_results_ml,
            'distractor': self.dev_results_distractor,
        }
        results = results_map.get(strategy)
        if results is None or idx >= len(results):
            return None
        return results[idx]

    # ─── Display helpers ─────────────────────────────────────────────────

    def print_hints(self, hints_dict, label="Hints"):
        """Pretty-print a hints dictionary."""
        print(f"\n{label}")
        print("=" * 70)
        for i, hint in enumerate(hints_dict.get('hints', []), 1):
            print(f"\n  Hint {i} (score={hint['score']:.4f})")
            print(f"  {hint['sentence'][:120]}")

    def print_distractors(self, dist_dict, label="Distractors"):
        """Pretty-print a distractors dictionary."""
        print(f"\n{label}")
        print("=" * 70)
        distractors = dist_dict.get('distractors', [])
        scores = dist_dict.get('scores', [])
        if not distractors:
            print("  (no distractors generated)")
            return
        for i, (d, s) in enumerate(zip(distractors, scores), 1):
            print(f"  [{i}] (score={s:.4f}) {d}")


# ============================================================================
# QUICK START DEMO
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("MODEL B — QUICK START DEMO")
    print("=" * 70 + "\n")

    # Try to find artifact directory
    candidate_dirs = [
        'models/model_b/traditional',
        'traditional',
        str(Path(__file__).parent / 'traditional'),
    ]
    artifact_dir = None
    for d in candidate_dirs:
        if Path(d).exists():
            artifact_dir = d
            break

    if artifact_dir is None:
        print("ERROR: Cannot find artifact directory. Run model_b.py first.")
        sys.exit(1)

    loader = ModelBLoader(artifact_dir)
    print(loader.get_metrics_summary())

    # ── Example Inference ────────────────────────────────────────────────
    sample_question = "Why did students prefer the library for studying?"
    sample_article = (
        "The city library was a beloved institution in the community. "
        "Many students visited every day to complete their assignments. "
        "The library offered free Wi-Fi, comfortable chairs, and thousands "
        "of reference books. Students preferred the library because it was "
        "quiet and free from distractions. The helpful librarians could "
        "always find the right resource for any research project. "
        "The library also hosted weekly study groups and tutoring sessions."
    )
    sample_correct = "quiet and free from distractions"

    print("\nQUESTION:", sample_question)
    print("CORRECT ANSWER:", sample_correct)

    # Extractive hints
    ext = loader.get_extractive_hints(sample_question, sample_article)
    loader.print_hints(ext, "EXTRACTIVE HINTS")

    # ML hints
    ml = loader.get_ml_hints(sample_question, sample_article)
    loader.print_hints(ml, "ML-SCORED HINTS")

    # Distractors
    dist = loader.get_distractors(
        sample_question, sample_article, sample_correct, top_k=3
    )
    loader.print_distractors(dist, "GENERATED DISTRACTORS")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

# explainer.py
import numpy as np
import lime
import lime.lime_text
import shap
from transformers import pipeline

_clf = None

def get_clf():
    global _clf
    if _clf is None:
        print("[explainer] Loading classifier for attribution...")
        _clf = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=True,
        )
    return _clf

def predict_proba(texts):
    """Probability array of shape (n_samples, 2) for LIME."""
    clf = get_clf()
    results = clf(texts, truncation=True, max_length=512)
    return np.array([[r["score"] for r in res] for res in results])

# ── LIME ────────────────────────────────────────────────────────────────────
def get_lime_explanation(text: str, num_features: int = 10):
    """
    Returns list of (word, weight) tuples.
    Positive weight → pushed towards POSITIVE class.
    """
    try:
        explainer = lime.lime_text.LimeTextExplainer(
            class_names=["NEGATIVE", "POSITIVE"]
        )
        exp = explainer.explain_instance(
            text[:512],
            predict_proba,
            num_features=num_features,
            num_samples=150,
        )
        return [{"word": w, "weight": round(float(v), 4)} for w, v in exp.as_list()]
    except Exception as e:
        return [{"word": f"[LIME error: {e}]", "weight": 0.0}]

# ── SHAP ────────────────────────────────────────────────────────────────────
def get_shap_explanation(text: str):
    """
    Returns list of (token, shap_value) dicts.
    Values are for the POSITIVE class (index 1).
    """
    try:
        clf = get_clf()
        explainer = shap.Explainer(clf)
        shap_values = explainer([text[:512]])
        tokens = shap_values.data[0]
        values = shap_values.values[0][:, 1]  # POSITIVE class
        return [
            {"token": str(tok), "value": round(float(val), 4)}
            for tok, val in zip(tokens, values)
        ]
    except Exception as e:
        return [{"token": f"[SHAP error: {e}]", "value": 0.0}]

# ── Sentence-level TF-IDF attribution (lightweight attention proxy) ─────────
def get_sentence_attribution(text: str):
    """
    Ranks sentences by TF-IDF importance score.
    Used as a simple explainability proxy for summarization.
    """
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) < 2:
        return [{"sentence": text, "score": 1.0}]

    try:
        vec = TfidfVectorizer(stop_words="english")
        tfidf = vec.fit_transform(sentences)
        scores = tfidf.toarray().sum(axis=1)
        max_score = scores.max() or 1.0
        result = [
            {"sentence": sent, "score": round(float(score / max_score), 4)}
            for sent, score in zip(sentences, scores)
        ]
        result.sort(key=lambda x: x["score"], reverse=True)
        return result
    except Exception as e:
        return [{"sentence": f"[Attribution error: {e}]", "score": 0.0}]

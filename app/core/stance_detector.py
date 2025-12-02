from transformers import pipeline

# Load zero-shot classifier once
nli_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

LABELS = ["supports", "refutes", "discusses"]

def detect_stance(best_sentence: str, claim: str):
    """
    Performs zero-shot stance detection.
    Returns:
        label: supports/refutes/discusses
        confidence: score between 0–1
    """
    if not best_sentence:
        return {"label": "discusses", "confidence": 0}

    result = nli_classifier(best_sentence, LABELS)

    return {
        "label": result["labels"][0],
        "confidence": float(result["scores"][0])
    }

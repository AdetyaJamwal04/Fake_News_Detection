from transformers import pipeline


def load_nli_classifier():
    """Load NLI classifier with error handling."""
    try:
        print("Loading stance detection model...")
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",  # ✅ Back to BART-large for better accuracy
            device=-1  # Use CPU
        )
        print("✓ Stance detector loaded successfully (BART-large)")
        return classifier
    except Exception as e:
        raise RuntimeError(
            f"Failed to load NLI classifier. "
            f"Check internet connection and try again. Error: {e}"
        )

# Load zero-shot classifier once
nli_classifier = load_nli_classifier()

LABELS = ["supports", "refutes", "neutral"]


def detect_stance(evidence_sentence: str, claim: str):
    """
    Performs zero-shot stance detection using NLI.
    
    Uses BART-large for best accuracy (slower but more reliable).
    
    Args:
        evidence_sentence: The sentence from the article
        claim: The claim to fact-check
    
    Returns:
        dict: {label: str, confidence: float}
    """
    if not evidence_sentence:
        return {"label": "neutral", "confidence": 0}
    
    # Better formatting for financial/numeric claims
    premise = evidence_sentence.strip()
    
    try:
        # Use better hypothesis template for entailment
        result = nli_classifier(
            premise, 
            LABELS,
            hypothesis_template="This text says that {}.",
            multi_label=False
        )
        
        label = result["labels"][0]
        confidence = float(result["scores"][0])
        
        # Map neutral to discusses for consistency
        if label == "neutral":
            label = "discusses"
        
        return {
            "label": label,
            "confidence": confidence
        }
    except Exception as e:
        print(f"Stance detection error: {e}")
        return {"label": "discusses", "confidence": 0}

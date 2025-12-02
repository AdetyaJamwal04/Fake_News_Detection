from transformers import pipeline


def load_nli_classifier():
    """Load NLI classifier with error handling."""
    try:
        print("Loading stance detection model...")
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # Use CPU
        )
        print("✓ Stance detector loaded successfully")
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
    
    Improved approach: Format as entailment task for better accuracy.
    - "supports" = evidence entails/supports the claim
    - "refutes" = evidence contradicts the claim  
    - "neutral" = evidence discusses but doesn't clearly support/refute
    
    Args:
        evidence_sentence: The sentence from the article
        claim: The claim to fact-check
    
    Returns:
        dict: {label: str, confidence: float}
    """
    if not evidence_sentence:
        return {"label": "neutral", "confidence": 0}
    
    # Format as premise-hypothesis for NLI
    # The evidence is the premise, claim is the hypothesis
    premise = f"{evidence_sentence}"
    
    try:
        result = nli_classifier(premise, LABELS, hypothesis_template="This supports the claim that {}. {}".format(claim, "{}"))
        
        label = result["labels"][0]
        confidence = float(result["scores"][0])
        
        # Map neutral to discusses for consistency with rest of system
        if label == "neutral":
            label = "discusses"
        
        return {
            "label": label,
            "confidence": confidence
        }
    except Exception as e:
        print(f"Stance detection error: {e}")
        return {"label": "discusses", "confidence": 0}

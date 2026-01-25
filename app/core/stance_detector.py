"""
Stance Detector Module with Optimizations

Uses zero-shot classification to determine if evidence supports, refutes, 
or discusses a claim.

Default model: BART-MNLI (~1.6GB, higher accuracy)
Alternative: DeBERTa-v3 (set STANCE_MODEL=deberta, ~700MB, faster)

Optimizations applied:
- Confidence calibration (Rank 13)
- Outcome modifier detection (v2.2 - fixes false positives)
- Stricter entailment hypothesis (v2.2)
"""

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

MODELS = {
    "deberta": {
        "name": "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        "display": "DeBERTa-v3",
        "size": "~700MB"
    },
    "bart": {
        "name": "facebook/bart-large-mnli",
        "display": "BART-large",
        "size": "~1.6GB"
    }
}

# Outcome modifiers that indicate non-completion or different action
OUTCOME_MODIFIERS = {
    "negating": [
        "attempted", "attempt", "tried", "try", "failed", "almost", "nearly",
        "survived", "survive", "escaped", "avoided", "thwarted", "foiled",
        "unsuccessful", "aborted", "prevented", "stopped", "blocked"
    ],
    "action_change": [
        "repositioned", "reposition", "moved", "deployed", "positioned",
        "threatened", "warned", "considered", "planned", "proposed"
    ]
}

# High-stakes claim keywords that require stricter checking
HIGH_STAKES_KEYWORDS = [
    "killed", "dead", "died", "assassinated", "assassination",
    "attack", "attacked", "war", "nuclear", "bomb", "explosion",
    "arrested", "convicted", "resigned", "impeached"
]


def get_current_model() -> str:
    """Return the currently loaded model name."""
    from app.core.model_registry import get_nli_classifier
    _, model_key = get_nli_classifier()
    return MODELS.get(model_key, MODELS["deberta"])["display"]


# Optimization #13: Confidence calibration
def calibrate_confidence(raw_score: float, temperature: float = 1.3) -> float:
    """Apply temperature scaling for better calibrated confidence scores."""
    return raw_score ** (1 / temperature)


def is_high_stakes_claim(claim: str) -> bool:
    """Check if the claim involves high-stakes assertions requiring stricter verification."""
    claim_lower = claim.lower()
    return any(keyword in claim_lower for keyword in HIGH_STAKES_KEYWORDS)


def detect_outcome_mismatch(claim: str, evidence: str) -> Tuple[bool, str]:
    """
    Detect if evidence describes an attempted/modified action while claim asserts completion.
    
    Examples:
    - Claim: "Trump was assassinated" + Evidence: "survived assassination attempt" → mismatch
    - Claim: "ordered attack" + Evidence: "ordered repositioning" → mismatch
    
    Returns:
        Tuple of (has_mismatch, reason)
    """
    claim_lower = claim.lower()
    evidence_lower = evidence.lower()
    
    # Check for negating modifiers in evidence but not in claim
    for modifier in OUTCOME_MODIFIERS["negating"]:
        if modifier in evidence_lower and modifier not in claim_lower:
            # Evidence says "attempted/survived/failed" but claim doesn't mention it
            return True, f"outcome_negated:{modifier}"
    
    # Check for action changes
    for modifier in OUTCOME_MODIFIERS["action_change"]:
        if modifier in evidence_lower:
            # Evidence describes a different action (repositioned vs attacked)
            # Only flag if claim doesn't contain this action
            if modifier not in claim_lower:
                return True, f"action_changed:{modifier}"
    
    # Check for explicit negation patterns
    negation_patterns = [
        r"not\s+dead",
        r"is\s+alive",
        r"survived",
        r"did\s+not\s+(die|attack|kill|bomb)",
        r"no\s+(attack|assassination|death)",
        r"false\s+claim",
        r"debunked",
        r"hoax",
        r"misinformation",
        r"conspiracy\s+theor"
    ]
    
    for pattern in negation_patterns:
        if re.search(pattern, evidence_lower):
            if "hoax" not in claim_lower and "false" not in claim_lower:
                return True, f"explicit_negation:{pattern}"
    
    return False, ""


def detect_stance(evidence_sentence: str, claim: str) -> Dict:
    """
    Performs zero-shot stance detection using NLI with enhanced verification.
    
    Uses natural language inference to determine if the evidence
    supports, refutes, or is neutral towards the claim.
    
    v2.2 Improvements:
    - Stricter hypothesis formulation for high-stakes claims
    - Outcome modifier detection to catch "attempted" vs "completed" confusion
    - Explicit negation pattern matching
    
    Args:
        evidence_sentence: The sentence from the article (premise)
        claim: The claim to fact-check (used to form hypotheses)
    
    Returns:
        dict: {label: str, confidence: float, modifier_detected: bool}
    """
    from app.core.model_registry import get_nli_classifier
    
    if not evidence_sentence or not claim:
        return {"label": "neutral", "confidence": 0, "modifier_detected": False}
    
    premise = evidence_sentence.strip()
    claim = claim.strip()
    
    # Check for outcome mismatch BEFORE running NLI
    has_mismatch, mismatch_reason = detect_outcome_mismatch(claim, premise)
    
    if has_mismatch:
        logger.info(f"Outcome mismatch detected: {mismatch_reason}")
        # Override to "refutes" or "discusses" based on mismatch type
        if "negated" in mismatch_reason or "explicit_negation" in mismatch_reason:
            return {
                "label": "refutes",
                "confidence": 0.75,  # High confidence in the override
                "modifier_detected": True,
                "mismatch_reason": mismatch_reason
            }
        else:
            # Action changed - less certain, mark as discusses
            return {
                "label": "discusses",
                "confidence": 0.6,
                "modifier_detected": True,
                "mismatch_reason": mismatch_reason
            }
    
    try:
        nli_classifier, _ = get_nli_classifier()
        
        # Use stricter hypothesis for high-stakes claims
        if is_high_stakes_claim(claim):
            hypotheses = [
                f"This proves that: {claim}",
                f"This contradicts or disproves: {claim}",
                f"This does not confirm or deny: {claim}"
            ]
        else:
            hypotheses = [
                f"This supports the claim: {claim}",
                f"This contradicts the claim: {claim}",
                f"This is unrelated to the claim: {claim}"
            ]
        
        result = nli_classifier(
            premise, 
            hypotheses,
            multi_label=False
        )
        
        label_map = {
            hypotheses[0]: "supports",
            hypotheses[1]: "refutes",
            hypotheses[2]: "neutral"
        }
        
        top_hypothesis = result["labels"][0]
        label = label_map.get(top_hypothesis, "neutral")
        raw_confidence = float(result["scores"][0])
        
        # Apply confidence calibration
        confidence = calibrate_confidence(raw_confidence)
        
        # For high-stakes claims, require higher confidence for "supports"
        if is_high_stakes_claim(claim) and label == "supports":
            if confidence < 0.7:
                # Not confident enough - downgrade to "discusses"
                label = "discusses"
                logger.info(f"High-stakes claim, low confidence ({confidence:.2f}) - downgrading to discusses")
        
        if label == "neutral":
            label = "discusses"
        
        return {
            "label": label,
            "confidence": confidence,
            "modifier_detected": False
        }
    except Exception as e:
        logger.error(f"Stance detection error: {e}")
        return {"label": "discusses", "confidence": 0, "modifier_detected": False}


def batch_detect_stance(premises: List[str], claim: str) -> List[Dict]:
    """
    Batch stance detection for multiple premises.
    
    Args:
        premises: List of evidence sentences
        claim: The claim to check against
        
    Returns:
        List of stance results
    """
    return [detect_stance(premise, claim) for premise in premises]


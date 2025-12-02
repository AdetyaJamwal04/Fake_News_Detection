# app/core/verdict_engine.py

import math

def sigmoid(x):
    """Sigmoid function for confidence calculation."""
    return 1 / (1 + math.exp(-x))


def compute_weighted_score(evidence):
    """
    Calculate weighted score for a single evidence.
    """
    similarity = evidence["similarity"]
    stance_score = evidence["stance_score"]

    # stance weight
    if evidence["stance"] == "supports":
        stance_w = +1
    elif evidence["stance"] == "refutes":
        stance_w = -1
    else:
        stance_w = 0  # discusses / neutral
    
    source_weight = evidence.get("source_weight", 1.0)

    return similarity * stance_score * stance_w * source_weight


def filter_quality_evidence(evidences, min_similarity=0.5, min_stance_conf=0.6):
    """
    Filter out low-quality evidence based on similarity and stance confidence.
    
    Args:
        evidences: List of evidence dictionaries
        min_similarity: Minimum semantic similarity threshold (0-1)
        min_stance_conf: Minimum stance detection confidence threshold (0-1)
    
    Returns:
        Filtered list of high-quality evidence
    """
    quality_evidences = []
    
    for ev in evidences:
        # Skip evidence with low similarity to claim
        if ev.get("similarity", 0) < min_similarity:
            continue
        
        # Skip evidence with uncertain stance
        if ev.get("stance_score", 0) < min_stance_conf:
            continue
        
        # Skip neutral/discusses stance (not informative for fact-checking)
        if ev.get("stance") not in ["supports", "refutes"]:
            continue
        
        quality_evidences.append(ev)
    
    return quality_evidences


def compute_final_verdict(evidences):
    """
    Compute the final verdict based on weighted aggregation.
    
    Improvements:
    - Filters low-quality evidence
    - Requires minimum evidence count
    - Uses stricter thresholds
    - Better confidence calculation
    
    Returns:
        {
          verdict: str,
          confidence: float,
          net_score: float,
          evidence_count: int,
          quality_evidence_count: int
        }
    """
    if not evidences:
        return {
            "verdict": "UNVERIFIED",
            "confidence": 0.0,
            "net_score": 0,
            "evidence_count": 0,
            "quality_evidence_count": 0
        }
    
    # Filter for quality evidence
    quality_evidences = filter_quality_evidence(evidences)
    
    # Not enough quality evidence
    if len(quality_evidences) < 3:
        return {
            "verdict": "UNVERIFIED",
            "confidence": 0.0,
            "net_score": 0,
            "evidence_count": len(evidences),
            "quality_evidence_count": len(quality_evidences)
        }
    
    # Calculate scores from quality evidence only
    scores = [compute_weighted_score(e) for e in quality_evidences]
    net_score = sum(scores)
    
    # Calculate confidence based on score magnitude and evidence count
    # More evidence and stronger scores = higher confidence
    score_magnitude = abs(net_score)
    evidence_factor = min(len(quality_evidences) / 5.0, 1.0)  # Cap at 5 evidences
    confidence = sigmoid(score_magnitude) * evidence_factor
    
    # Stricter decision thresholds - require strong consensus
    # With quality filtering, typical scores range from -3 to +3
    if net_score > 1.5:
        verdict = "LIKELY TRUE"
    elif net_score < -1.5:
        verdict = "LIKELY FALSE"
    elif abs(net_score) < 0.5:
        verdict = "UNVERIFIED"
    else:
        verdict = "MIXED / MISLEADING"

    return {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "net_score": round(net_score, 3),
        "evidence_count": len(evidences),
        "quality_evidence_count": len(quality_evidences)
    }

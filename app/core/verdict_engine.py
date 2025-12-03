# app/core/verdict_engine.py

import math

def sigmoid(x):
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


def compute_final_verdict(evidences):
    """
    Compute the final verdict based on weighted aggregation.
    Returns:
        {
          verdict: str,
          confidence: float,
          net_score: float
        }
    """
    if not evidences:
        return {
            "verdict": "UNVERIFIED",
            "confidence": 0.0,
            "net_score": 0
        }

    scores = [compute_weighted_score(e) for e in evidences]
    net_score = sum(scores)

    confidence = sigmoid(abs(net_score))

    # Decision thresholds
    if net_score > 0.4:
        verdict = "LIKELY TRUE"
    elif net_score < -0.4:
        verdict = "LIKELY FALSE"
    else:
        verdict = "MIXED / MISLEADING"

    return {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "net_score": round(net_score, 3)
    }

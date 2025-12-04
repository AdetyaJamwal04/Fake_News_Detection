from sentence_transformers import SentenceTransformer, util
import torch

# Load SBERT model globally only once
# For faster performance, use: "all-MiniLM-L6-v2" (80MB, 5x faster, slight accuracy drop)
# For best accuracy, use: "sentence-transformers/all-mpnet-base-v2" (420MB, slower, best quality)
sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def get_best_matching_sentence(claim: str, sentences: list):
    """
    Returns:
        best_sentence (str)
        best_score (float)
        all_scores (list of floats)
    """

    if not sentences:
        return None, 0, []

    # Encode claim and sentences
    claim_emb = sbert_model.encode(claim, convert_to_tensor=True)
    sent_embs = sbert_model.encode(sentences, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_scores = util.pytorch_cos_sim(claim_emb, sent_embs)[0]

    # Convert to Python list
    scores_list = cosine_scores.cpu().tolist()

    # Get index of highest match
    best_idx = scores_list.index(max(scores_list))

    best_sentence = sentences[best_idx]
    best_score = scores_list[best_idx]

    return best_sentence, best_score, scores_list

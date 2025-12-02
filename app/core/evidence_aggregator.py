from app.core.scraper import scrape_article
from app.core.embedder import get_best_matching_sentence
from app.core.stance_detector import detect_stance
import spacy

# Load spacy for better sentence splitting
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None  # Fallback to simple splitting if spacy not available


def split_into_sentences(text):
    """
    Split text into sentences using spacy (better than str.split("."))
    Falls back to simple splitting if spacy is not available.
    """
    if nlp:
        doc = nlp(text[:100000])  # Limit to avoid memory issues
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
    else:
        # Fallback: simple splitting
        return [s.strip() for s in text.split(".") if len(s.strip()) > 20]


def build_evidence(claim: str, search_results: list):
    """
    Build evidence from search results with quality checks.
    
    Args:
        claim: The claim to fact-check
        search_results: List of search result dictionaries
    
    Returns:
        List of evidence dictionaries with quality metrics
    """
    evidences = []

    for item in search_results:
        url = item.get("href")
        if not url:
            continue

        article_text = scrape_article(url)
        if not article_text or len(article_text) < 100:
            continue  # Skip articles with too little content

        # Better sentence splitting
        sentences = split_into_sentences(article_text)
        
        if not sentences:
            continue

        best_sentence, sim_score, _ = get_best_matching_sentence(claim, sentences)
        
        # Skip if no good match found
        if not best_sentence or sim_score < 0.3:
            continue

        stance = detect_stance(best_sentence, claim)

        evidences.append({
            "url": url,
            "best_sentence": best_sentence,
            "similarity": sim_score,
            "stance": stance["label"],
            "stance_score": stance["confidence"],
        })

    return evidences

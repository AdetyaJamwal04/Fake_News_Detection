from app.core.scraper import scrape_article
from app.core.embedder import get_best_matching_sentence
from app.core.stance_detector import detect_stance

def build_evidence(claim: str, search_results: list):
    evidences = []

    for item in search_results:
        url = item.get("href")
        if not url:
            continue

        article_text = scrape_article(url)
        if not article_text:
            continue

        sentences = [s.strip() for s in article_text.split(".") if len(s.strip()) > 20]

        best_sentence, sim_score, _ = get_best_matching_sentence(claim, sentences)

        stance = detect_stance(best_sentence, claim)

        evidences.append({
            "url": url,
            "best_sentence": best_sentence,
            "similarity": sim_score,
            "stance": stance["label"],
            "stance_score": stance["confidence"],
        })

    return evidences

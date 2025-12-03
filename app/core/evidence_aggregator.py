from app.core.scraper import scrape_article
from app.core.embedder import get_best_matching_sentence
from app.core.stance_detector import detect_stance
from concurrent.futures import ThreadPoolExecutor, as_completed
import spacy

# Load spacy for better sentence splitting
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None


def split_into_sentences(text):
    """
    Split text into sentences using spacy or fallback to simple splitting.
    """
    if nlp:
        doc = nlp(text[:100000])
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
    else:
        return [s.strip() for s in text.split(".") if len(s.strip()) > 20]


def process_single_result(item, claim):
    """
    Process a single search result to extract evidence.
    """
    try:
        url = item.get("href")
        if not url:
            return None
            
        article_text = scrape_article(url)
        if not article_text:
            return None
            
        sentences = split_into_sentences(article_text)
        if not sentences:
            return None
            
        best_sentence, sim_score, _ = get_best_matching_sentence(claim, sentences)
        
        if not best_sentence:
            return None
            
        stance = detect_stance(best_sentence, claim)
        
        return {
            "url": url,
            "best_sentence": best_sentence,
            "similarity": sim_score,
            "stance": stance["label"],
            "stance_score": stance["confidence"],
        }
    except Exception as e:
        print(f"Error processing {item.get('href', 'unknown')}: {e}")
        return None


def build_evidence(claim: str, search_results: list, max_workers: int = 5):
    """
    Build evidence from search results with parallel processing.
    """
    evidences = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_result, item, claim): item 
            for item in search_results
        }
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                evidences.append(result)
    
    return evidences

import spacy
from keybert import KeyBERT
from trafilatura import fetch_url, extract
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()


def extract_text_from_url(url: str) -> str:
    """
    Fetch and clean article content from a URL.
    """
    try:
        html = fetch_url(url)
        if not html:
            return ""
        text = extract(html)
        return text or ""
    except Exception:
        return ""


def clean_text(text: str) -> str:
    """
    Normalize whitespace and formatting.
    """
    if not text:
        return ""
    return " ".join(text.split())


def extract_claim_from_text(text: str) -> str:
    """
    Extracts the main claim by choosing the first meaningful sentence.
    Uses KeyBERT to refine understanding but returns the sentence only.
    """
    text = clean_text(text)
    if not text:
        return ""

    doc = nlp(text[:3000])  # Limit for speed
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]

    if not sentences:
        return text

    claim = sentences[0]  # Lead sentence = main claim

    # extract keywords for metadata (optional)
    _ = kw_model.extract_keywords(claim, top_n=3)

    return claim
import spacy
from keybert import KeyBERT
from trafilatura import fetch_url, extract

nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()
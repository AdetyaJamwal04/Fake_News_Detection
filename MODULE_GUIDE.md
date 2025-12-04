# 📘 Module Guide - Fake News Detector

**Complete technical walkthrough of every module in the project**

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Module Architecture](#module-architecture)
3. [Core Modules Detailed Guide](#core-modules-detailed-guide)
4. [Streamlit Application](#streamlit-application)
5. [Data Flow Example](#data-flow-example)
6. [Key Concepts Explained](#key-concepts-explained)

---

## 🎯 Project Overview

This is an **AI-powered fact-checking system** that verifies claims by:
1. Extracting claims from text/URL
2. Searching the web for evidence
3. Analyzing evidence semantically
4. Determining if evidence supports or refutes the claim
5. Computing a final verdict with confidence

**Tech Stack:**
- **Sentence-BERT**: Semantic similarity (finding relevant evidence)
- **BART-MNLI**: Stance detection (does evidence support/refute claim?)
- **Spacy**: Natural language processing
- **DuckDuckGo**: Web search
- **Trafilatura**: Web scraping
- **Streamlit**: User interface

---

## 🏗️ Module Architecture

```
app/core/
├── claim_extractor.py      # Extract claims from text/URL
├── query_generator.py      # Generate search queries
├── web_search.py           # Search the web (DuckDuckGo)
├── scraper.py              # Scrape article content
├── embedder.py             # Semantic similarity (SBERT)
├── stance_detector.py      # Stance classification (BART)
├── evidence_aggregator.py  # Parallel evidence gathering
└── verdict_engine.py       # Final verdict computation

streamlit_app/
└── home_page.py            # User interface
```

---

## 📖 Core Modules Detailed Guide

### 1. `claim_extractor.py`

**Purpose:** Extract the main claim from text or URL

**Dependencies:**
```python
import spacy                    # NLP library
from keybert import KeyBERT     # Keyword extraction
from trafilatura import fetch_url, extract  # Web scraping
import nltk                     # Natural Language Toolkit
```

#### Function 1: `extract_text_from_url(url: str) -> str`

**What it does:** Fetches a webpage and extracts clean article text

**How it works:**
```python
def extract_text_from_url(url: str) -> str:
    try:
        html = fetch_url(url)           # Get HTML
        if not html:
            return ""
        text = extract(html)             # Extract main content only
        return text or ""
    except Exception:
        return ""
```

**Technical Concepts:**
- **HTML → Clean Text**: Trafilatura removes ads, navigation, footers
- **DOM Parsing**: Identifies main article content automatically
- **Error Handling**: Returns empty string on failure (graceful degradation)

**Example:**
```python
url = "https://example.com/news/article"
text = extract_text_from_url(url)
# Returns: "The rupee fell to 85.20 against the dollar today..."
```

---

#### Function 2: `clean_text(text: str) -> str`

**What it does:** Normalizes whitespace and formatting

**How it works:**
```python
def clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split())  # Collapse multiple spaces/newlines
```

**Why needed:**
- Articles have irregular spacing, tabs, multiple newlines
- Makes text uniform for processing

**Example:**
```python
messy = "The    rupee\n\nfell    today"
clean = clean_text(messy)
# Returns: "The rupee fell today"
```

---

#### Function 3: `extract_claim_from_text(text: str) -> str`

**What it does:** Extracts the main claim (usually first meaningful sentence)

**How it works:**
```python
def extract_claim_from_text(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""

    # Use Spacy for sentence segmentation
    doc = nlp(text[:3000])  # Limit to first 3000 chars for speed
    sentences = [sent.text.strip() for sent in doc.sents 
                 if len(sent.text.strip()) > 20]

    if not sentences:
        return text

    claim = sentences[0]  # First sentence = main claim
    
    # Extract keywords (for metadata, not returned)
    _ = kw_model.extract_keywords(claim, top_n=3)

    return claim
```

**Technical Concepts:**
- **Sentence Boundary Detection**: Spacy identifies sentence endings (not just periods!)
  - Handles "Dr. Smith said..." correctly
  - Handles "U.S.A." correctly
- **Lead Sentence Assumption**: Journalistic style puts main claim first
- **Keyword Extraction**: KeyBERT finds important terms (e.g., "rupee", "dollar")

**Example:**
```python
article = """The Indian rupee hit an all-time low today. 
             Economists predict further decline. 
             The RBI may intervene."""
             
claim = extract_claim_from_text(article)
# Returns: "The Indian rupee hit an all-time low today."
```

---

### 2. `query_generator.py`

**Purpose:** Generate multiple search queries to find diverse evidence

**Why multiple queries?**
- Single query might miss relevant information
- Different phrasings find different sources
- Entities generate targeted searches

#### Function: `generate_queries(claim: str) -> list`

**How it works:**
```python
def generate_queries(claim: str):
    doc = nlp(claim)
    entities = [ent.text for ent in doc.ents]  # Extract named entities

    base = claim.lower()

    queries = [
        base,                           # Original
        base + " fact check",          # Fact-checking sites
        base + " true or false",       # Verification queries
        base + " hoax",                # Debunking sites
        base + " authenticity check",  # Verification
    ]

    # Entity-based queries
    for e in entities:
        queries.append(f"{e} {base}")
        queries.append(f"{base} {e} false")
        queries.append(f"{e} controversy")
        queries.append(f"{e} news verification")

    return list(set(queries))  # Remove duplicates
```

**Technical Concepts:**
- **Named Entity Recognition (NER)**: Spacy identifies:
  - PERSON: "Joe Biden"
  - ORG: "RBI"
  - GPE: "India"
  - MONEY: "90 rupees"
- **Query Diversification**: Different angles increase evidence coverage
- **Deduplication**: `set()` removes duplicate queries

**Example:**
```python
claim = "The RBI announced rate cuts in India"
queries = generate_queries(claim)

# Returns something like:
# [
#   "the rbi announced rate cuts in india",
#   "the rbi announced rate cuts in india fact check",
#   "RBI the rbi announced rate cuts in india",
#   "India the rbi announced rate cuts in india",
#   "RBI controversy",
#   "India news verification",
#   ...
# ]
```

---

### 3. `web_search.py`

**Purpose:** Search the web using DuckDuckGo API

#### Function: `web_search(queries: List[str], max_results: int = 6) -> List[Dict]`

**How it works:**
```python
def web_search(queries: List[str], max_results: int = 6):
    all_results = []
    seen_urls = set()
    
    with DDGS() as ddgs:
        for query in queries:
            try:
                results = ddgs.text(
                    query,
                    max_results=max_results,
                    region='wt-wt',        # Worldwide
                    safesearch='moderate'
                )
                
                for result in results:
                    url = result.get('href') or result.get('link')
                    
                    # Deduplicate
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append({
                            'href': url,
                            'title': result.get('title', ''),
                            'body': result.get('snippet', '')
                        })
                
                time.sleep(0.5)  # Prevent rate limiting
                
            except Exception as e:
                print(f"Search failed for query '{query}': {e}")
                continue
    
    return all_results
```

**Technical Concepts:**
- **DDGS (DuckDuckGo Search)**: Privacy-focused search API
  - No API key needed
  - Unlimited queries
  - `text()` method for web search
- **Deduplication**: Same article may appear for multiple queries
  - Use `set()` to track URLs
  - Prevents processing same source twice
- **Rate Limiting**: 0.5s delay prevents API throttling
- **Error Handling**: Continue if one query fails

**Example:**
```python
queries = ["rupee hits 90", "rupee 90 dollar fact check"]
results = web_search(queries, max_results=3)

# Returns:
# [
#   {
#     'href': 'https://economictimes.com/...',
#     'title': 'Rupee hits all-time low of 90...',
#     'body': 'The Indian rupee depreciated...'
#   },
#   ...
# ]
```

---

### 4. `scraper.py`

**Purpose:** Extract clean article text from URLs

#### Function: `scrape_article(url: str) -> str`

**How it works:**
```python
def scrape_article(url: str) -> str:
    try:
        html = fetch_url(url)
        if not html:
            return ""
        return extract(html) or ""
    except Exception as e:
        return ""
```

**Technical Concepts:**
- **Trafilatura**: Smart content extraction
  - Automatically finds main article
  - Removes boilerplate (ads, menus, comments)
  - Works on most news sites
- **Graceful Failure**: Returns empty string instead of crashing

**Why Trafilatura?**
- **Better than BeautifulSoup**: Doesn't need custom selectors per site
- **Fast**: Optimized for speed
- **Accurate**: Trained on thousands of sites

---

### 5. `embedder.py`

**Purpose:** Find semantically similar sentences using AI embeddings

**Key Concept: Vector Embeddings**
- Converts sentences to 768-dimensional vectors
- Similar meanings → similar vectors
- Uses cosine similarity to compare

#### Function: `get_best_matching_sentence(claim: str, sentences: list)`

**How it works:**
```python
def get_best_matching_sentence(claim: str, sentences: list):
    if not sentences:
        return None, 0, []

    # Convert to embeddings (vectors)
    claim_emb = sbert_model.encode(claim, convert_to_tensor=True)
    sent_embs = sbert_model.encode(sentences, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_scores = util.pytorch_cos_sim(claim_emb, sent_embs)[0]

    # Find best match
    scores_list = cosine_scores.cpu().tolist()
    best_idx = scores_list.index(max(scores_list))

    best_sentence = sentences[best_idx]
    best_score = scores_list[best_idx]

    return best_sentence, best_score, scores_list
```

**Technical Concepts:**

**1. Sentence Embeddings:**
```
Input:  "The rupee fell to 90"
Output: [0.12, -0.34, 0.89, ..., 0.45]  # 768 numbers
```

**2. Cosine Similarity:**
```
similarity = (A · B) / (||A|| × ||B||)

Where:
- A · B = dot product
- ||A|| = magnitude of vector A
```

**Range:** -1 to +1 (usually 0 to 1 for similar texts)
- 1.0 = identical meaning
- 0.5 = somewhat related
- 0.0 = unrelated

**Example:**
```python
claim = "Rupee hits 90 against dollar"
sentences = [
    "The rupee depreciated to 90.",      # High similarity
    "Stock market crashed today.",       # Low similarity
    "Currency at all-time low of 90."    # High similarity
]

best, score, all_scores = get_best_matching_sentence(claim, sentences)

# best = "The rupee depreciated to 90."
# score = 0.87 (very similar)
# all_scores = [0.87, 0.23, 0.81]
```

---

### 6. `stance_detector.py`

**Purpose:** Determine if evidence supports, refutes, or just discusses a claim

**Key Concept: Natural Language Inference (NLI)**
- Given premise and hypothesis, predict relationship
- 3 classes: entailment, contradiction, neutral

#### Function: `detect_stance(evidence_sentence: str, claim: str)`

**How it works:**
```python
def detect_stance(evidence_sentence: str, claim: str):
    if not evidence_sentence:
        return {"label": "neutral", "confidence": 0}
    
    premise = evidence_sentence.strip()
    
    try:
        result = nli_classifier(
            premise, 
            LABELS,                         # ["supports", "refutes", "neutral"]
            hypothesis_template="This text says that {}.",
            multi_label=False
        )
        
        label = result["labels"][0]
        confidence = float(result["scores"][0])
        
        if label == "neutral":
            label = "discusses"  # Rename for clarity
        
        return {"label": label, "confidence": confidence}
    except Exception as e:
        return {"label": "discusses", "confidence": 0}
```

**Technical Concepts:**

**1. Zero-Shot Classification:**
- No training needed
- Model pre-trained on MNLI dataset
- Can classify any text into any categories

**2. NLI Framework:**
```
Premise: "The rupee fell to 90 today"
Hypothesis: "This text says that rupee breached 90-mark"

Model predicts:
- supports (entailment)    → 0.92
- refutes (contradiction)  → 0.03
- neutral                  → 0.05
```

**3. Hypothesis Template:**
- Formats the claim as a hypothesis
- "This text says that {claim}." works well for factual claims
- Alternative: "This supports the claim that {claim}."

**Example:**
```python
claim = "Rupee hits 90"
evidence = "The Indian rupee depreciated to an all-time low of 90.21"

stance = detect_stance(evidence, claim)

# Returns:
# {
#   "label": "supports",
#   "confidence": 0.94
# }

# If evidence was: "The rupee is expected to stabilize at 85"
# Returns:
# {
#   "label": "refutes",
#   "confidence": 0.78
# }
```

---

### 7. `evidence_aggregator.py`

**Purpose:** Coordinate parallel evidence gathering from multiple URLs

#### Function: `process_single_result(item, claim)`

**What it does:** Process one URL to extract evidence

**How it works:**
```python
def process_single_result(item, claim):
    try:
        url = item.get("href")
        if not url:
            return None
            
        # Step 1: Scrape article
        article_text = scrape_article(url)
        if not article_text:
            return None
            
        # Step 2: Split into sentences
        sentences = split_into_sentences(article_text)
        if not sentences:
            return None
            
        # Step 3: Find most relevant sentence
        best_sentence, sim_score, _ = get_best_matching_sentence(claim, sentences)
        if not best_sentence:
            return None
            
        # Step 4: Detect stance
        stance = detect_stance(best_sentence, claim)
        
        # Return evidence
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
```

---

#### Function: `build_evidence(...)`

**What it does:** Process ALL URLs in parallel

**How it works:**
```python
def build_evidence(claim: str, search_results: list, max_workers: int = 5):
    evidences = []
    
    # Parallel processing with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_result, item, claim): item 
            for item in search_results
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            if result:
                evidences.append(result)
    
    return evidences
```

**Technical Concepts:**

**1. Parallel Processing:**
```
Sequential (slow):
URL1 → scrape → embed → stance (3s)
URL2 → scrape → embed → stance (3s)
URL3 → scrape → embed → stance (3s)
Total: 9 seconds

Parallel (fast):
URL1 → scrape → embed → stance (3s) ┐
URL2 → scrape → embed → stance (3s) ├─ All at once!
URL3 → scrape → embed → stance (3s) ┘
Total: 3 seconds
```

**2. ThreadPoolExecutor:**
- Python's built-in parallel processing
- `max_workers=5` → 5 URLs processed simultaneously
- Each worker is a separate thread

**3. as_completed:**
- Returns results as soon as available
- Doesn't wait for slowest URL
- More responsive

**Example:**
```python
claim = "Rupee hits 90"
search_results = [
    {'href': 'url1.com'},
    {'href': 'url2.com'},
    {'href': 'url3.com'},
]

evidences = build_evidence(claim, search_results)

# Returns:
# [
#   {
#     'url': 'url1.com',
#     'best_sentence': 'Rupee fell to 90.21 today',
#     'similarity': 0.89,
#     'stance': 'supports',
#     'stance_score': 0.94
#   },
#   ...
# ]
```

---

### 8. `verdict_engine.py`

**Purpose:** Compute final verdict from all evidence

#### Function: `compute_weighted_score(evidence: dict) -> float`

**What it does:** Calculate a score for one piece of evidence

**Formula:**
```
score = similarity × stance_score × stance_weight × source_weight
```

**How it works:**
```python
def compute_weighted_score(evidence):
    similarity = evidence["similarity"]        # 0-1
    stance_score = evidence["stance_score"]    # 0-1

    # Stance weight
    if evidence["stance"] == "supports":
        stance_w = +1
    elif evidence["stance"] == "refutes":
        stance_w = -1
    else:
        stance_w = 0  # discusses/neutral
    
    source_weight = evidence.get("source_weight", 1.0)

    return similarity * stance_score * stance_w * source_weight
```

**Example:**
```python
evidence = {
    'similarity': 0.85,        # Highly relevant
    'stance': 'supports',      # Supports claim
    'stance_score': 0.92,      # Very confident
    'source_weight': 1.0       # Standard weight
}

score = compute_weighted_score(evidence)
# = 0.85 × 0.92 × (+1) × 1.0
# = 0.782

# If stance was "refutes":
# = 0.85 × 0.92 × (-1) × 1.0  
# = -0.782  # Negative!
```

---

#### Function: `compute_final_verdict(evidences: list) -> dict`

**What it does:** Combine all evidence scores and make final decision

**How it works:**
```python
def compute_final_verdict(evidences):
    if not evidences:
        return {
            "verdict": "UNVERIFIED",
            "confidence": 0.0,
            "net_score": 0
        }

    # Sum all weighted scores
    scores = [compute_weighted_score(e) for e in evidences]
    net_score = sum(scores)

    # Calculate confidence (sigmoid)
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
```

**Technical Concepts:**

**1. Net Score:**
```
Evidence 1: +0.78 (supports)
Evidence 2: +0.65 (supports)
Evidence 3: -0.23 (refutes)
Net Score: 0.78 + 0.65 - 0.23 = 1.20
```

**2. Sigmoid Function:**
```python
sigmoid(x) = 1 / (1 + e^(-x))
```

Converts any number to 0-1 range:
- sigmoid(0) = 0.5
- sigmoid(2) = 0.88
- sigmoid(-2) = 0.12

Used for confidence score.

**3. Thresholds:**
```
    -∞         -0.4      0      0.4         +∞
     ├──────────┼────────┼────────┼──────────┤
   FALSE      MIXED   MIXED   TRUE      TRUE
```

**Example:**
```python
evidences = [
    {'similarity': 0.85, 'stance': 'supports', 'stance_score': 0.92},
    {'similarity': 0.78, 'stance': 'supports', 'stance_score': 0.88},
    {'similarity': 0.65, 'stance': 'refutes', 'stance_score': 0.45},
]

result = compute_final_verdict(evidences)

# Calculation:
# Score 1: 0.85 × 0.92 × 1 = 0.782
# Score 2: 0.78 × 0.88 × 1 = 0.686
# Score 3: 0.65 × 0.45 × (-1) = -0.293
# Net: 0.782 + 0.686 - 0.293 = 1.175

# Returns:
# {
#   "verdict": "LIKELY TRUE",      # net_score > 0.4
#   "confidence": 0.764,            # sigmoid(1.175)
#   "net_score": 1.175
# }
```

---

## 🖥️ Streamlit Application

### `streamlit_app/home_page.py`

**Purpose:** User interface for the fact-checker

**Key Components:**

#### 1. UI Setup
```python
st.set_page_config(
    page_title="Fake News Fact Checker",
    page_icon="📰",
    layout="wide"
)
```

#### 2. Sidebar Settings
```python
st.sidebar.title("Settings")
max_results = st.sidebar.slider("Max search results per query", 1, 10, 3)
```

#### 3. Input Form
```python
with st.form("input_form"):
    url_input = st.text_input("Enter Article URL", "")
    text_input = st.text_area("Or paste article text", height=180)
    submit = st.form_submit_button("Check")
```

#### 4. Processing Pipeline
```python
if submit:
    # Extract claim
    claim = extract_claim_from_text(full_text)
    
    # Generate queries
    queries = generate_queries(claim)
    
    # Search web
    search_results = web_search(queries, max_results)
    
    # Build evidence (with progress bar)
    evidences = []
    progress = st.progress(0)
    for idx, item in enumerate(search_results):
        sub_evs = build_evidence(claim, [item])
        evidences.extend(sub_evs)
        progress.progress(int(((idx+1)/total)*100))
    
    # Compute verdict
    verdict_result = compute_final_verdict(evidences)
    
    # Display
    st.markdown(f"**Verdict:** {verdict_result['verdict']}")
```

#### 5. Evidence Display
```python
# Categorize evidence
supports = [e for e in evidences if e.get("stance") == "supports"]
refutes = [e for e in evidences if e.get("stance") == "refutes"]

# Display in columns
s_col, r_col = st.columns(2)
with s_col:
    st.markdown("#### Supporting evidence")
    for ev in supports:
        with st.expander(ev.get("url")):
            st.markdown(format_evidence_card(ev))
```

---

## 🔄 Data Flow Example

Let's trace a complete example: **"Rupee hits 90 against dollar"**

### Step 1: Claim Extraction
```
Input: "Rupee hits 90 against dollar. FIIs outflow continues..."
↓ clean_text()
↓ spacy sentence segmentation
Output: "Rupee hits 90 against dollar"
```

### Step 2: Query Generation
```
Input: "Rupee hits 90 against dollar"
↓ NER: entities = ["Rupee", "dollar"]
↓ generate variations
Output: [
  "rupee hits 90 against dollar",
  "rupee hits 90 against dollar fact check",
  "Rupee controversy",
  "dollar news verification",
  ...
]
```

### Step 3: Web Search
```
Input: ["rupee hits 90 fact check", ...]
↓ DDGS API calls
Output: [
  {href: "economictimes.com/...", title: "Rupee at 90.21..."},
  {href: "hindustantimes.com/...", title: "Currency hits low..."},
  ...
]
```

### Step 4: Evidence Gathering (Parallel)

**Thread 1:**
```
URL: economictimes.com/...
↓ scrape_article()
↓ split_into_sentences()
↓ get_best_matching_sentence()
  → "Indian rupee depreciated to 90.21 against dollar"
  → similarity: 0.89
↓ detect_stance()
  → stance: "supports"
  → confidence: 0.94
Output: {url, sentence, 0.89, "supports", 0.94}
```

**Thread 2:** (simultaneous)
```
URL: hindustantimes.com/...
[same process]
Output: {url, sentence, 0.82, "supports", 0.88}
```

### Step 5: Verdict Computation
```
Evidence 1: 0.89 × 0.94 × 1 = 0.837
Evidence 2: 0.82 × 0.88 × 1 = 0.722
Net Score: 1.559

↓ net_score > 0.4
Verdict: "LIKELY TRUE"
Confidence: sigmoid(1.559) = 0.826
```

### Step 6: Display
```
UI shows:
✅ LIKELY TRUE
Confidence: 82.6%
Net Score: 1.559

Supporting Evidence:
- economictimes.com: "Indian rupee depreciated to 90.21..."
- hindustantimes.com: "Currency hits all-time low..."
```

---

## 🧠 Key Concepts Explained

### 1. Embeddings

**What are they?**
- Mathematical representation of text as vectors
- Capture semantic meaning

**Example:**
```
"cat" → [0.2, 0.8, 0.1, ...]
"dog" → [0.3, 0.7, 0.2, ...]  # Similar to "cat"
"car" → [0.9, 0.1, 0.8, ...]  # Very different
```

**Why useful?**
- Can measure similarity mathematically
- Works across paraphrases
- Language-agnostic (with multilingual models)

### 2. Cosine Similarity

**Formula:**
```
cos(θ) = (A · B) / (||A|| × ||B||)
```

**Interpretation:**
- 1.0 = Same direction (identical meaning)
- 0.0 = Perpendicular (unrelated)
- -1.0 = Opposite direction (contradictory)

**Visual:**
```
Vector A: "Rupee falls" →
Vector B: "Currency drops" → (similar angle)
Vector C: "Economy grows" ↗ (different angle)
```

### 3. Zero-Shot Classification

**Concept:**
- Classify without training examples
- Uses pre-trained knowledge

**How BART-MNLI works:**
1. Trained on MNLI dataset (400k examples)
2. Learned general entailment patterns
3. Can apply to any new text

**Example:**
```
Pre-training:
- "It's raining" entails "Weather is wet" ✓
- "It's raining" contradicts "It's sunny" ✓

Application:
- "Rupee at 90" entails "Rupee hit 90"? → 0.95 (yes)
- "Rupee at 85" entails "Rupee hit 90"? → 0.12 (no)
```

### 4. Weighted Scoring

**Why weights?**
- Not all evidence is equal
- High similarity + high confidence = strong evidence
- Neutral stance contributes 0

**Formula breakdown:**
```
score = similarity × stance_conf × stance_dir × source

similarity:    How relevant is the evidence?
stance_conf:   How confident is stance detection?
stance_dir:    +1 (supports) / -1 (refutes) / 0 (neutral)
source:        Source credibility (future feature)
```

### 5. Sigmoid Function

**Purpose:** Normalize scores to probability range (0-1)

**Properties:**
- Always between 0 and 1
- Smooth S-curve
- Good for confidence scores

**Graph:**
```
1.0 |         ___---
    |      _--
0.5 |   _-
    | _-
0.0 |--
    └───────────────
   -5   0   5   10
```

---

## 🎓 Learning Path

**Beginner:**
1. Start with `claim_extractor.py` - Simple text processing
2. Then `query_generator.py` - String manipulation
3. Study `web_search.py` - API usage

**Intermediate:**
4. `embedder.py` - Vector mathematics
5. `scraper.py` - Web scraping
6. `verdict_engine.py` - Scoring logic

**Advanced:**
7. `stance_detector.py` - NLP models
8. `evidence_aggregator.py` - Parallel processing
9. `home_page.py` - Full integration

---

## 📚 Further Reading

**Sentence-BERT:**
- Paper: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)

**BART:**
- Paper: "BART: Denoising Sequence-to-Sequence Pre-training"
- [https://arxiv.org/abs/1910.13461](https://arxiv.org/abs/1910.13461)

**MNLI Dataset:**
- MultiNLI: Natural Language Inference
- [https://cims.nyu.edu/~sbowman/multinli/](https://cims.nyu.edu/~sbowman/multinli/)

---

**This guide covers every technical aspect of the Fake News Detector. Use it to understand, modify, or extend the system!**

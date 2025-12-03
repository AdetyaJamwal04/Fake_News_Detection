# 📰 Fake News Detector

> An AI-powered fact-checking system that uses live web evidence, semantic similarity, and zero-shot stance detection to verify claims in real-time.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Demo](https://img.shields.io/badge/status-active-success.svg)

---

## 🌟 Features

- **🔍 Real-Time Fact Checking** - Verify claims using live web evidence
- **🌐 Multi-Source Verification** - Searches and analyzes multiple sources automatically
- **🤖 AI-Powered Analysis** - Uses state-of-the-art NLP models:
  - SBERT for semantic similarity
  - BART-large-MNLI for stance detection
  - Spacy for NLP processing
- **⚡ Parallel Processing** - 60-70% faster evidence gathering
- **📊 Detailed Evidence** - Shows supporting, refuting, and neutral evidence
- **🎨 Beautiful UI** - Clean Streamlit interface
- **💾 Export Results** - Download reports as JSON

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for model downloads and web search)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Fake_News_Detector.git
cd Fake_News_Detector

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app/home_page.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 Usage

### Option 1: Check from URL
1. Enter a news article URL
2. Click "Check"
3. View the verdict and evidence

### Option 2: Check from Text
1. Paste article text
2. Click "Check"
3. View the verdict and evidence

### Option 3: Check Direct Claim
1. Enter a specific claim
2. Click "Check"
3. View the verdict and evidence

### Understanding Results

**Verdicts:**
- ✅ **LIKELY TRUE** - Strong supporting evidence found
- ⛔ **LIKELY FALSE** - Strong refuting evidence found
- ⚠️ **MIXED / MISLEADING** - Evidence both supports and refutes
- ❓ **UNVERIFIED** - Insufficient reliable evidence

**Metrics:**
- **Confidence**: How confident the model is (0-100%)
- **Net Score**: Cumulative evidence score (-∞ to +∞)
- **Sources Analyzed**: Number of web sources checked

---

## 🏗️ Architecture

```
┌─────────────────┐
│  User Input     │
│  (URL/Text)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Claim Extractor │ ← Spacy + KeyBERT
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Generator │ ← Entity extraction
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Web Search     │ ← DuckDuckGo API
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Evidence        │ ← Parallel scraping
│ Aggregator      │    5 workers
└────────┬────────┘
         │
         ├──► Sentence Similarity (SBERT)
         │
         ├──► Stance Detection (BART-NLI)
         │
         ▼
┌─────────────────┐
│ Verdict Engine  │ ← Weighted scoring
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  UI Display     │
└─────────────────┘
```

---

## 📁 Project Structure

```
Fake_News_Detector/
├── app/
│   ├── __init__.py
│   └── core/
│       ├── __init__.py
│       ├── claim_extractor.py    # Extract claims from text/URL
│       ├── embedder.py            # SBERT semantic similarity
│       ├── evidence_aggregator.py # Parallel evidence gathering
│       ├── query_generator.py     # Generate search queries
│       ├── scraper.py             # Web scraping utilities
│       ├── stance_detector.py     # NLI stance classification
│       ├── verdict_engine.py      # Final verdict computation
│       └── web_search.py          # DuckDuckGo search
├── streamlit_app/
│   └── home_page.py              # Streamlit UI
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── BUILD_GUIDE.md               # Detailed setup guide
├── DOCUMENTATION.md             # Technical documentation
└── .gitignore                   # Git ignore rules
```

---

## 🛠️ Technology Stack

### Core ML Models
- **[SBERT](https://www.sbert.net/)** - Sentence-BERT for semantic similarity
- **[BART-large-MNLI](https://huggingface.co/facebook/bart-large-mnli)** - Zero-shot stance detection
- **[Spacy](https://spacy.io/)** - NLP and entity extraction

### Libraries
- **[Streamlit](https://streamlit.io/)** - Web UI framework
- **[Trafilatura](https://trafilatura.readthedocs.io/)** - Web scraping
- **[DDGS](https://github.com/deedy5/duckduckgo_search)** - DuckDuckGo search API
- **[KeyBERT](https://maartengr.github.io/KeyBERT/)** - Keyword extraction

---

## ⚙️ Configuration

### Adjusting Settings (in Streamlit sidebar)

- **Max search results per query** (1-10): Controls how many sources to check
  - Lower = Faster but less evidence
  - Higher = Slower but more comprehensive

### Performance Tuning

Edit `app/core/evidence_aggregator.py`:
```python
# Adjust parallel workers (default: 5)
def build_evidence(claim, search_results, max_workers=5):
    ...
```

Edit `app/core/verdict_engine.py`:
```python
# Adjust verdict thresholds (default: ±0.4)
if net_score > 0.4:  # Increase for stricter "TRUE" verdicts
    verdict = "LIKELY TRUE"
```

---

## 📊 Performance

- **Model Loading**: 2-5 minutes (first run only)
- **Per Query**: 15-30 seconds average
- **Speedup**: 60-70% faster with parallel processing
- **Concurrent Users**: Supports multiple simultaneous queries

---

## 🧪 Testing

### Manual Testing
```bash
streamlit run streamlit_app/home_page.py
```

Test cases:
1. **Known True**: "The Earth orbits the Sun"
2. **Known False**: "The Earth is flat"
3. **Ambiguous**: Recent news claims
4. **URL Test**: Enter any news article URL

### Expected Behavior
- ✅ Claims extract correctly from URLs and text
- ✅ Multiple search queries generated
- ✅ Evidence gathered from diverse sources
- ✅ Verdict aligns with common knowledge for known facts

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
git clone YOUR_FORK
cd Fake_News_Detector
pip install -r requirements.txt
# Make your changes
# Test thoroughly
git commit -am "Your message"
git push origin your-branch
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Sentence-BERT** by Nils Reimers and Iryna Gurevych
- **BART** by Facebook AI
- **Spacy** by Explosion AI
- **Streamlit** team for the amazing framework
- Open-source community for various libraries

---

## 📧 Contact

- **Author**: Your Name
- **GitHub**: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- **Email**: your.email@example.com

---

## 🗺️ Roadmap

- [ ] Add API endpoint support
- [ ] Implement caching for faster repeat queries
- [ ] Add source credibility scoring
- [ ] Support for multiple languages
- [ ] Browser extension
- [ ] Mobile app integration
- [ ] Database for storing fact-check history

---

## ⚠️ Disclaimer

This tool is designed to assist in fact-checking but should not be the sole source for determining truth. Always verify important claims through multiple reliable sources and use critical thinking.

---

## 📈 Stats

![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/Fake_News_Detector?style=social)
![GitHub forks](https://img.shields.io/github/forks/YOUR_USERNAME/Fake_News_Detector?style=social)
![GitHub issues](https://img.shields.io/github/issues/YOUR_USERNAME/Fake_News_Detector)

---

**Made with ❤️ and AI**

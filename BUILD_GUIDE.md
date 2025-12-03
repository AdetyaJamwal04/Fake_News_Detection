# 🏗️ Build Guide - Fake News Detector

Complete step-by-step guide to set up and run the Fake News Detector project.

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [First-Time Setup](#first-time-setup)
4. [Running the Application](#running-the-application)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Configuration](#advanced-configuration)
7. [Deployment](#deployment)

---

## 💻 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum (16 GB recommended)
- **Storage**: 5 GB free space (for models)
- **Internet**: Required for model downloads and web search

### Recommended Specifications
- **CPU**: 4+ cores
- **RAM**: 16 GB
- **GPU**: Not required (CPU-only inference)
- **Internet**: Stable broadband connection

---

## 📥 Installation Steps

### Step 1: Install Python

**Windows:**
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run installer
3. ✅ Check "Add Python to PATH"
4. Click "Install Now"
5. Verify:
   ```bash
   python --version
   pip --version
   ```

**macOS:**
```bash
# Using Homebrew
brew install python@3.10

# Verify
python3 --version
pip3 --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.10 python3-pip
python3 --version
pip3 --version
```

### Step 2: Clone the Repository

```bash
# Via HTTPS
git clone https://github.com/YOUR_USERNAME/Fake_News_Detector.git

# Or via SSH
git clone git@github.com:YOUR_USERNAME/Fake_News_Detector.git

# Navigate to project directory
cd Fake_News_Detector
```

### Step 3: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 4: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

**This will install:**
- Streamlit (UI framework)
- Transformers + BART model (~1.6 GB)
- Sentence-Transformers + SBERT model (~500 MB)
- Spacy + language model
- Web scraping libraries
- Search API libraries

**⏱️ Installation time:** 5-15 minutes depending on internet speed

---

## 🎬 First-Time Setup

### Step 1: Verify Spacy Model

The Spacy model should install automatically, but verify:

```bash
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✓ Spacy OK')"
```

If error, install manually:
```bash
python -m spacy download en_core_web_sm
```

### Step 2: Verify NLTK Data

NLTK data downloads automatically on first run, but you can pre-download:

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Step 3: Test Model Loading

```bash
python -c "from app.core.embedder import sbert_model; from app.core.stance_detector import nli_classifier; print('✓ Models loaded successfully')"
```

**Expected output:**
```
Loading sentence transformer model...
✓ Sentence transformer loaded successfully
Loading stance detection model...
✓ Stance detector loaded successfully
✓ Models loaded successfully
```

**⏱️ First model load:** 2-5 minutes (downloads models from internet)

---

## 🚀 Running the Application

### Standard Run

```bash
streamlit run streamlit_app/home_page.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

### Custom Port

```bash
streamlit run streamlit_app/home_page.py --server.port 8080
```

### Run in Background (Linux/macOS)

```bash
nohup streamlit run streamlit_app/home_page.py &
```

### Access from Network

```bash
streamlit run streamlit_app/home_page.py --server.address 0.0.0.0
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError

**Problem:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. Spacy Model Not Found

**Problem:**
```
OSError: [E050] Can't find model 'en_core_web_sm'
```

**Solution:**
```bash
python -m spacy download en_core_web_sm
```

#### 3. CUDA/GPU Errors (Can Ignore)

**Problem:**
```
Warning: No GPU found, using CPU
```

**Solution:** This is normal! The app works fine on CPU. To silence warnings:
```python
# In stance_detector.py, device=-1 already set for CPU
```

#### 4. Port Already in Use

**Problem:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Use different port
streamlit run streamlit_app/home_page.py --server.port 8502

# Or kill existing process
lsof -ti:8501 | xargs kill  # macOS/Linux
netstat -ano | findstr :8501  # Windows (then kill PID)
```

#### 5. DuckDuckGo Search Fails

**Problem:**
```
Search failed for query 'xyz': Rate limit exceeded
```

**Solution:**
- Wait 1 minute and retry
- Reduce "Max search results per query" in sidebar
- Check internet connection

#### 6. Memory Issues

**Problem:**
```
MemoryError: Unable to allocate array
```

**Solution:**
- Close other applications
- Reduce max_results in sidebar
- Restart Python/Streamlit
- Consider upgrading RAM

---

## ⚙️ Advanced Configuration

### Environment Variables

Create `.env` file:
```bash
# Model cache directories
TRANSFORMERS_CACHE=/path/to/cache
SENTENCE_TRANSFORMERS_HOME=/path/to/cache

# API settings
MAX_SEARCH_RESULTS=3
PARALLEL_WORKERS=5

# Debug
DEBUG=False
```

### Custom Model Paths

Edit `app/core/embedder.py`:
```python
# Use local model
sbert_model = SentenceTransformer("/path/to/local/model")
```

### Adjust Performance

**Faster (less accurate):**
```python
# In evidence_aggregator.py
max_workers = 10  # More parallel workers

# In streamlit sidebar
max_results = 2  # Fewer sources
```

**More accurate (slower):**
```python
# In evidence_aggregator.py
max_workers = 3  # Fewer parallel workers

# In streamlit sidebar
max_results = 6  # More sources

# In verdict_engine.py
if net_score > 0.6:  # Stricter threshold
```

---

## 🌐 Deployment

### Deploy to Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select your repo
5. Main file: `streamlit_app/home_page.py`
6. Click "Deploy"

**Note:** First deployment may take 10-15 minutes for model downloads.

### Deploy to Heroku

```bash
# Install Heroku CLI
# Create Procfile
echo "web: sh setup.sh && streamlit run streamlit_app/home_page.py" > Procfile

# Create setup.sh
cat > setup.sh << EOF
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = \$PORT
enableCORS = false
" > ~/.streamlit/config.toml
EOF

# Deploy
heroku create your-app-name
git push heroku main
```

### Deploy to Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app/home_page.py"]
```

Build and run:
```bash
docker build -t fake-news-detector .
docker run -p 8501:8501 fake-news-detector
```

---

## 📊 Performance Optimization

### 1. Pre-load Models

Create `preload.py`:
```python
from app.core.embedder import sbert_model
from app.core.stance_detector import nli_classifier
print("Models loaded and cached!")
```

Run before starting app:
```bash
python preload.py
streamlit run streamlit_app/home_page.py
```

### 2. Use Model Caching

Models are cached after first load. To clear cache:
```bash
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/torch
```

### 3. Reduce Model Size

For production, consider smaller models:
```python
# In embedder.py - use smaller SBERT
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")  # 80MB vs 420MB

# In stance_detector.py - use DistilBART
classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")
```

---

## 🧪 Testing the Build

### Quick Test

```bash
# Run quick test
python << EOF
from app.core.verdict_engine import compute_final_verdict
test_evidence = [{"similarity": 0.8, "stance": "supports", "stance_score": 0.9, "source_weight": 1.0}]
result = compute_final_verdict(test_evidence)
print(f"Test verdict: {result['verdict']}")
assert result['verdict'] == "LIKELY TRUE"
print("✓ Test passed!")
EOF
```

### Full Integration Test

1. Start the app
2. Test URL: Enter any news article URL
3. Test Text: Paste sample text
4. Test Claim: "Python is a programming language"
5. Verify: All steps complete without errors

---

## 📝 Build Checklist

- [ ] Python 3.8+ installed
- [ ] Git installed
- [ ] Repository cloned
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Spacy model verified
- [ ] NLTK data downloaded
- [ ] Models pre-loaded successfully
- [ ] App starts without errors
- [ ] Can submit a test query successfully
- [ ] Results display correctly

**If all checked:** ✅ **Build successful!**

---

## 🆘 Getting Help

If you encounter issues:

1. Check [Troubleshooting](#troubleshooting) section
2. Review error messages carefully
3. Search [GitHub Issues](https://github.com/YOUR_USERNAME/Fake_News_Detector/issues)
4. Create new issue with:
   - Error message
   - Python version
   - OS version
   - Steps to reproduce

---

**Build guide last updated:** 2024

**Happy building! 🚀**

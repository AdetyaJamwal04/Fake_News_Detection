# Build Guide

Complete guide for building, running, and deploying VeriFact — the AI-Powered Claim Verification System.

## Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- Git

## Local Development

### 1. Clone the Repository

```bash
git clone https://github.com/AdetyaJamwal04/Fake_News_Detection.git
cd Fake_News_Detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Set Environment Variables

```bash
cp .env.example .env
# Edit .env and add your API keys (see Environment Variables section below)
```

### 5. Run the Application

```bash
python app_flask.py
```

Access at: http://localhost:5000

### 6. Warm Up Models (Optional)

ML models (DeBERTa-v3, MiniLM, spaCy) are lazy-loaded on first request. To pre-load them and avoid cold-start latency:

```bash
curl -X POST http://localhost:5000/api/warmup
```

## Docker Build

### Build Image

```bash
docker build -t verifact .
```

### Run Container

```bash
docker run -d \
  --name verifact \
  -p 5000:5000 \
  -e TAVILY_API_KEY=your_key \
  -e GROQ_API_KEY=your_key \
  --restart unless-stopped \
  verifact
```

### Using Docker Compose

```bash
docker-compose up -d
```

### Post-Deployment Model Warmup

After the container starts, warm up the ML models to avoid cold-start latency on the first request:

```bash
curl -X POST http://localhost:5000/api/warmup
```

## AWS EC2 Deployment

### 1. Launch EC2 Instance

- **AMI:** Ubuntu 22.04 LTS
- **Instance Type:** t3.medium or c7i-flex.large (2+ vCPU, 4GB+ RAM minimum)
- **Storage:** 30GB+ (Docker images + ML model downloads)
- **Security Group:** Allow ports 22 (SSH), 5000 (App)

### 2. Install Docker on EC2

```bash
sudo apt update
sudo apt install -y docker.io
sudo usermod -aG docker ubuntu
```

### 3. Pull and Run

```bash
docker pull adetyajamwal/fake-news-detector:latest
docker run -d --name verifact -p 5000:5000 \
  -e TAVILY_API_KEY=your_key \
  -e GROQ_API_KEY=your_key \
  --restart unless-stopped \
  adetyajamwal/fake-news-detector:latest

# Warm up models after container is running
curl -X POST http://localhost:5000/api/warmup
```

### 4. Set Up Elastic IP (Recommended)

By default, EC2 public IPs change when you stop/start the instance. To get a static IP:

1. Go to **AWS Console → EC2 → Elastic IPs**
2. Click **Allocate Elastic IP address**
3. Click **Allocate**
4. Select the new IP → **Actions → Associate Elastic IP address**
5. Choose your instance and click **Associate**
6. Update the `EC2_HOST` secret in GitHub with this new static IP

### Instance Restart Behavior

The container uses `--restart unless-stopped`, which means:

| Scenario | Container Status |
|----------|------------------|
| Instance reboot | ✅ Auto-starts |
| Instance stop → start | ✅ Auto-starts |
| Container crashes | ✅ Auto-restarts |
| Manual `docker stop` before shutdown | ❌ Won't auto-start |

**Note:** If you don't have an Elastic IP, the public IP will change after stop/start. You'll need to update:
- Your browser bookmarks
- The `EC2_HOST` GitHub secret

## CI/CD with GitHub Actions

The project uses automated deployment via `.github/workflows/deploy.yml`:

1. Push to `main` branch triggers the workflow
2. Tests run first (`pytest`)
3. Docker image is built and pushed to DockerHub
4. EC2 instance pulls and runs the new image

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `DOCKERHUB_USERNAME` | DockerHub username |
| `DOCKERHUB_TOKEN` | DockerHub access token |
| `EC2_HOST` | EC2 public IP (or Elastic IP) |
| `EC2_SSH_KEY` | Private key for SSH |
| `TAVILY_API_KEY` | Tavily search API key |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TAVILY_API_KEY` | Recommended | Primary search API key (best quality results) |
| `BRAVE_API_KEY` | No | Brave Search API key (2nd tier fallback) |
| `GROQ_API_KEY` | No | Enables LLM features: AI summary, query decomposition, verdict tiebreaker |
| `HF_API_TOKEN` | No | HuggingFace API token (not used in current V2 — all models run locally) |
| `PORT` | No | Server port (default: 5000) |
| `DEBUG` | No | Enable debug mode (default: false) |

> **Note:** The system works fully without `GROQ_API_KEY` — it falls back to NER-based queries and rule-based explanations. Groq enhances accuracy but is not required.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/health` | GET | Health check with model status and metrics |
| `/api/check` | POST | Fact-check a claim |
| `/api/warmup` | POST | Pre-load all ML models |

## Troubleshooting

### Disk Space Issues

The Docker image is ~4-5GB. If deployment fails with "no space left":

```bash
docker system prune -a --volumes -f
df -h
```

### Container Logs

```bash
docker logs verifact --tail 100
```

### Health Check

```bash
curl http://localhost:5000/api/health
```

### Memory Issues (OOM Kills)

If the container is killed due to OOM on a 4GB instance:
- Ensure no other heavy processes are running
- The system already optimizes memory: 3 ThreadPool workers, 50 sentence cap, 10K char spaCy limit, gradients disabled
- Consider using a larger instance type if issues persist

## Running Tests

```bash
pytest tests/ -v
```

Individual test files:
```bash
pytest tests/test_source_scorer.py -v
pytest tests/test_query_generator.py -v
pytest tests/test_verdict_engine.py -v
pytest tests/test_claim_extractor.py -v
pytest tests/test_web_search.py -v
pytest tests/test_llm_summary.py -v
pytest tests/test_integration.py -v
```

## ML Models

All core ML models run **locally** on CPU — no GPU required.

| Model | Purpose | Size | Source |
|-------|---------|------|--------|
| `nli-deberta-v3-small` | Stance detection (NLI) | ~180 MB | HuggingFace |
| `all-MiniLM-L6-v2` | Sentence embeddings (SBERT) | ~90 MB | HuggingFace |
| `en_core_web_sm` | Tokenization + NER | ~12 MB | spaCy |
| `llama-3.3-70b-versatile` | LLM reasoning (optional) | API-based | Groq |

Models are lazy-loaded on first request. Use `/api/warmup` to pre-load them.

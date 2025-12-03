# Contributing to Fake News Detector

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

---

## 🤝 How to Contribute

### Reporting Bugs

1. Check [existing issues](https://github.com/YOUR_USERNAME/Fake_News_Detector/issues)
2. Create a new issue with:
   - Clear title
   - Detailed description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version)
   - Error messages/logs

### Suggesting Features

1. Check existing feature requests
2. Open a new issue with `[FEATURE]` prefix
3. Describe the feature and its benefits
4. Provide use cases

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Test thoroughly
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

---

## 💻 Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/Fake_News_Detector.git
cd Fake_News_Detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

---

## 📝 Code Style

### Python Style Guide

Follow [PEP 8](https://pep8.org/) guidelines:

```python
# Good
def compute_weighted_score(evidence: dict) -> float:
    """Calculate weighted score for evidence."""
    similarity = evidence["similarity"]
    return similarity * stance_weight

# Use type hints
def process_claim(text: str, max_length: int = 100) -> str:
    ...

# Clear variable names
evidence_score = compute_score(evidence)  # Good
s = compute_score(e)  # Bad
```

### Formatting

Use `black` for auto-formatting:

```bash
black app/ streamlit_app/
```

### Linting

```bash
flake8 app/ streamlit_app/
mypy app/
```

---

## 🧪 Testing

### Write Tests

```python
# tests/test_verdict.py
import pytest
from app.core.verdict_engine import compute_final_verdict

def test_verdict_true():
    evidence = [{"similarity": 0.9, "stance": "supports", 
                 "stance_score": 0.95, "source_weight": 1.0}]
    result = compute_final_verdict(evidence)
    assert result["verdict"] == "LIKELY TRUE"
```

### Run Tests

```bash
pytest tests/
```

---

## 📋 Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add source credibility scoring
fix: correct stance detection for edge cases
docs: update API documentation
test: add tests for verdict engine
refactor: simplify evidence aggregation logic
perf: optimize parallel scraping
```

---

## 🎯 Areas for Contribution

### High Priority

- [ ] Add unit tests (coverage < 50%)
- [ ] Improve documentation
- [ ] Fix known bugs
- [ ] Performance optimizations

### Features

- [ ] Source credibility scoring
- [ ] Caching for repeated queries
- [ ] Multi-language support
- [ ] API endpoints
- [ ] Browser extension

### Code Quality

- [ ] Add type hints
- [ ] Improve error handling
- [ ] Refactor long functions
- [ ] Add docstrings

---

## 🚀 Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new features
3. **Ensure all tests pass**
4. **Update CHANGELOG.md**
5. **Request review** from maintainers
6. **Address feedback**
7. **Squash commits** if requested

---

## ✅ Checklist

Before submitting PR:

- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] No unnecessary dependencies
- [ ] Commits are clear
- [ ] PR description is detailed

---

## 📞 Questions?

- Open an issue
- Email: your.email@example.com
- Discord: [Link if available]

---

**Thank you for contributing! 🎉**

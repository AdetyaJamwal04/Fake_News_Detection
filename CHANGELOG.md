# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-03

### Added
- Initial release of Fake News Detector
- Real-time fact-checking using web evidence
- SBERT for semantic similarity
- BART-large-MNLI for stance detection
- Parallel evidence gathering (5 workers)
- Streamlit web interface
- Support for URL, text, and direct claim inputs
- Evidence breakdown (supporting/refuting/neutral)
- JSON export functionality
- Comprehensive documentation

### Features
- Multi-source verification
- Weighted evidence scoring
- Confidence calculation using sigmoid
- Automatic claim extraction
- Entity-based query generation
- DuckDuckGo web search integration
- Trafilatura web scraping
- Better sentence splitting with Spacy

### Performance
- 60-70% faster with parallel processing
- Model caching for faster subsequent runs
- Efficient batch processing

### Documentation
- README with installation and usage
- BUILD_GUIDE with step-by-step setup
- DOCUMENTATION with technical details
- CONTRIBUTING guidelines
- MIT License

## [Unreleased]

### Planned
- [ ] API endpoint support
- [ ] Result caching
- [ ] Source credibility scoring
- [ ] Multi-language support
- [ ] Browser extension
- [ ] Mobile app integration
- [ ] Unit tests
- [ ] CI/CD pipeline

---

**Format:**
- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` for vulnerability fixes

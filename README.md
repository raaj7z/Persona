# Persona: AI Stylometry & Pseudonymous Continuity Analysis Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Pytest](https://img.shields.io/badge/Pytest-Passed-green.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)]()

> Advanced AI-based authorship attribution, stylometry profiling, semantic embedding comparison, and candidate pseudonymous handle continuity detection microservice.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture & Processing Pipeline](#architecture--processing-pipeline)
- [Repository Structure](#repository-structure)
- [Key Features](#key-features)
- [Environment Requirements](#environment-requirements)
- [Installation & Setup](#installation--setup)
- [Running the Microservice](#running-the-microservice)
  - [FastAPI REST Server](#fastapi-rest-server)
  - [Integration Script](#integration-script)
  - [Running Unit Tests](#running-unit-tests)
- [API Documentation & Endpoints](#api-documentation--endpoints)
- [Multimodal Feature Fusion](#multimodal-feature-fusion)

---

## 🔍 Overview

**Persona** is a microservice designed to de-anonymize threat actors who operate under multiple handles, forum personas, or rebranded accounts across dark web and clearnet platforms. It extracts 50+ linguistic features (character/word n-grams, function word distributions, punctuation habits, syntax tree complexity, emoji usage), computes semantic embedding similarities, analyzes temporal activity patterns, and produces confidence-scored pseudonymous handle migration hypotheses.

---

## 🏗️ Architecture & Processing Pipeline

```
+-------------------------------------------------------------------+
|                        Post Corpus Input                          |
|         (Posts JSON: author_id, text, platform, timestamp)        |
+---------------------------------+---------------------------------+
                                  |
                                  v
+---------------------------------+---------------------------------+
|                   Preprocessing Pipeline                          |
| (cleaner, quote_remover, deduplicator, language_detector)         |
+---------------------------------+---------------------------------+
                                  |
    +-----------------------------+-----------------------------+
    |                             |                             |
    v                             v                             v
+---------------+         +---------------+             +---------------+
|  Stylometry   |         |   Semantic    |             |  Behavioral   |
|    Engine     |         |  Embeddings   |             |    Engine     |
| (ngrams,      |         |  (topics,     |             | (temporal,    |
|  function     |         |   cosine sim) |             |  burst,       |
|  words,       |         +-------+-------+             |  lifecycle)   |
|  syntax,      |                 |                     +-------+-------+
|  ai_profiler) |                 |                             |
+-------+-------+                 |                             |
        |                         |                             |
        +-------------------------+-----------------------------+
                                  |
                                  v
+---------------------------------+---------------------------------+
|                   Multimodal Fusion Engine                        |
|                     (src/fusion/engine.py)                        |
+---------------------------------+---------------------------------+
                                  |
    +-----------------------------+-----------------------------+
    |                                                           |
    v                                                           v
+---------------------------------+             +-------------------+
|        Profile Builder          |             | Handle Migration  |
| (src/persona/profile_builder.py)|             | Detector          |
+---------------------------------+             | (src/persona/     |
                                                |  migration.py)    |
                                                +-------------------+
```

---

## 📁 Repository Structure

```
Persona/
├── config.yaml                # Module threshold and weight configurations
├── scripts/
│   └── run_integration.py     # End-to-end integration test runner
├── src/
│   ├── main.py                # FastAPI REST API entry point (:8010)
│   ├── models/                # Data models and schemas
│   │   ├── post.py            # Post model
│   │   ├── profile.py         # Persona profile model
│   │   ├── corpus.py          # Post corpus collection model
│   │   ├── evidence.py        # Evidence provenance model
│   │   └── comparison.py      # Pairwise comparison model
│   ├── preprocessing/         # Text Cleaning & Normalization
│   │   ├── cleaner.py         # Text cleaner and sanitizer
│   │   ├── quote_remover.py   # Removes quoted forum replies
│   │   ├── deduplicator.py    # Removes duplicate posts
│   │   ├── language_detector.py # Filters non-target language text
│   │   └── template_detector.py # Strips bot and template noise
│   ├── stylometry/            # Linguistic & Stylometric Extraction
│   │   ├── engine.py          # Master stylometry extractor
│   │   ├── ngrams.py          # Character & word n-gram distributions
│   │   ├── function_words.py  # Function word frequency analyzer
│   │   ├── punctuation.py     # Punctuation habit & ratio profiler
│   │   ├── syntax.py          # Sentence structure & syntax complexity
│   │   ├── spelling.py        # Common typos & spelling variant detector
│   │   ├── emoji.py           # Emoji & emoticon distribution profiler
│   │   └── ai_profiler.py     # Advanced LLM-assisted stylometry profiler
│   ├── semantic/              # Topic & Embedding Similarity
│   │   ├── embeddings.py      # Dense vector embedding generator
│   │   ├── similarity.py      # Cosine & Jaccard semantic similarity
│   │   └── topics.py          # Topic modeling & keyword extraction
│   ├── behavior/              # Temporal & Platform Dynamics
│   │   ├── engine.py          # Behavioral engine orchestrator
│   │   ├── temporal.py        # Posting time & diurnal activity distribution
│   │   ├── burst.py           # High-frequency post burst analyzer
│   │   ├── lifecycle.py       # Handle creation & inactivity timeline
│   │   └── human/             # Human communication pattern models
│   │       ├── alias_behavior.py # Handle naming convention analyzer
│   │       └── communication_patterns.py # Communication style profiler
│   ├── persona/               # Core Persona Logic
│   │   ├── profile_builder.py # Builds consolidated persona profiles
│   │   ├── migration.py       # Candidate handle continuity detector
│   │   └── evolution.py       # Persona style evolution over time
│   ├── fusion/                # Feature Fusion Engine
│   │   └── engine.py          # Weighted fusion score calculator
│   ├── comparison/            # Profile Comparison & Matrix
│   │   ├── pairwise.py        # 1-to-1 profile comparator
│   │   ├── matrix.py          # All-to-all similarity matrix builder
│   │   └── ranking.py         # Top candidate match ranker
│   ├── evidence/              # Provenance & Limitations
│   │   ├── builder.py         # Evidence object builder
│   │   └── limitations.py     # Statistical confidence & limitation flags
│   └── reports/               # Output Exporters
│       ├── json_report.py     # JSON report generator
│       ├── html_report.py     # Interactive HTML report renderer
│       └── pdf_report.py      # PDF document generator
├── tests/                     # Pytest Unit Test Suite
│   ├── test_api.py            # FastAPI endpoint tests
│   ├── test_stylometry.py     # Stylometry extraction tests
│   ├── test_persona.py        # Profile builder & migration tests
│   ├── test_semantic.py       # Semantic embedding similarity tests
│   ├── test_behavior.py       # Temporal behavior tests
│   └── test_preprocessing.py  # Text cleaner unit tests
├── .gitignore                 # Git ignore rules for cache & python artifacts
└── requirements.txt           # Python dependencies
```

---

## ✨ Key Features

- **50+ Stylometric Signals**: Character 3/4/5-grams, word length distributions, type-token ratio (TTR), function word frequencies, punctuation ratios, capitalization habits.
- **Candidate Pseudonymous Handle Continuity**: Detects handle migrations across forums (e.g. `DarkViper_2024` ➔ `AetherSec_2026`) with fusion similarity scoring.
- **FastAPI Microservice Interface**: Standardized REST API endpoints (`/analyze`, `/migration`, `/health`).
- **Multimodal Feature Fusion**: Combines stylometry similarity, semantic topic alignment, and temporal activity overlap into a unified confidence score (0.0 to 1.0).
- **Synthetic Demonstration Generator**: Provides built-in synthetic rebrand corpora for testing without using real person data.

---

## 🔧 Environment Requirements

- **Python**: Python 3.10+ (Python 3.12 tested).
- **Pytest**: For running unit tests.

---

## 🚀 Installation & Setup

1. **Clone Repository**:
   ```bash
   git clone https://github.com/devpathak0029-droid/Persona.git
   cd Persona
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Running the Microservice

### FastAPI REST Server

Start the live microservice on `127.0.0.1:8010`:

```bash
python -m uvicorn src.main:app --host 127.0.0.1 --port 8010 --reload
```

- **Health Check**: `http://127.0.0.1:8010/health`
- **Interactive Swagger Docs**: `http://127.0.0.1:8010/docs`

### Integration Script

Run the end-to-end synthetic corpus integration test:

```bash
python scripts/run_integration.py
```

### Running Unit Tests

Execute the complete Pytest suite:

```bash
pytest tests/ -v
```

---

## 📡 API Documentation & Endpoints

### 1. `POST /analyze`
Runs stylometry feature extraction and profile building on a given post corpus.

**Payload Example**:
```json
{
  "persona_id": "DarkViper_2024",
  "posts": [
    {
      "post_id": "P101",
      "author_id": "DarkViper_2024",
      "platform": "forum_v1",
      "text": "Greeting colleagues. Always verify SHA256 checksums before deploying -- security first.",
      "timestamp": "2024-03-15T14:20:00Z"
    }
  ]
}
```

### 2. `POST /migration`
Compares two post corpora (`persona_a` and `persona_b`) to calculate handle continuity confidence.

**Payload Example**:
```json
{
  "investigation_id": "INV-DEMO-2026",
  "persona_a": { "persona_id": "DarkViper_2024", "posts": [...] },
  "persona_b": { "persona_id": "AetherSec_2026", "posts": [...] }
}
```

---

## 📜 License

Proprietary — AI Stylometry & Pseudonymous Continuity Analysis Engine.

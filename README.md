# 🛡️ SemanticTruth | AI FactGuard Intelligence

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask Framework](https://img.shields.io/badge/Framework-Flask%203.1-black?logo=flask)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-98.6%25-brightgreen)](#-machine-learning--nlp-architecture)
[![License](https://img.shields.io/badge/License-MIT-purple)](#-license)

> **SemanticTruth** is an intelligent, real-time machine learning and natural language processing platform engineered to detect fake news, audit manipulated media, and evaluate information credibility across published web URLs and text statements.

---

## 🌟 Key Highlights

* ⚡ **Real-Time News Credibility Auditing**: Instant asynchronous inference that audits claims and headlines without page reloads.
* 🌐 **Autonomous URL Article Scraper**: Paste any live news link (e.g., BBC, Reuters, blogs). FactGuard strips advertisements, parses the genuine article body, and inspects domain reputation.
* 🧠 **Explainable AI (XAI) Grounding**: Answers are explained on the specific basis of the news:
  * Detects formal journalistic attributions (*"confirmed"*, *"according to"*, *"officials stated"*, *"study published"*).
  * Flags clickbait & sensationalism buzzwords (*"shocking"*, *"secret"*, *"exposed"*, *"banned"*, *"elites"*).
  * Measures emotional capitalization ratio (CAPS %) and urgency punctuation (`!`, `!!`).
* 🔒 **User Authentication & Audit Trail**: User registration, login, and a personal dashboard where verified scans are automatically saved.
* 🎨 **Futuristic Cyber Glassmorphism UI**: High-end responsive dark interface with animated truth meters and interactive test samples.

---

## 📸 Screenshots & Architecture

```
[User Input: Web URL or News Text]
                │
                ▼
      ┌──────────────────┐
      │  Headless Scraper│ (If URL: extracts title, body, strips boilerplate)
      └─────────┬────────┘
                │
                ▼
      ┌─────────────────────────┐
      │  NLP Feature Extraction │ (TF-IDF bi-gram coordinates + Tokenizer)
      └─────────┬───────────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
┌──────────────┐  ┌────────────────────────┐
│ ML Posterior │  │ Stylometric Heuristics │ (Sensationalism, Caps, Quotes,
│ Probability  │  │ & Source Attributions  │  Journalistic attribution signals)
└──────┬───────┘  └───────────┬────────────┘
       │                      │
       └──────────┬───────────┘
                  ▼
      ┌─────────────────────────┐
      │ Hybrid Consensus Engine │
      └───────────┬─────────────┘
                  │
                  ▼
 [Verdict: REAL / FAKE + Confidence % + "Basis of Answer" Explanation]
```

---

## ⚙️ Tech Stack

* **Backend**: Python 3.10+, Flask 3.1, Werkzeug Security
* **Machine Learning & NLP**: Scikit-Learn, TF-IDF Vectorizer (N-grams 1–2), Logistic Regression, Joblib
* **Data Extraction**: BeautifulSoup4, Requests, urllib
* **Database**: SQLite3 (Users, Analysis Audit Logs)
* **Frontend**: HTML5, Modern CSS3 (Glassmorphism, CSS Custom Properties), JavaScript (Async Fetch API), FontAwesome 6

---

## 📊 Machine Learning & NLP Architecture

The core model was trained on a benchmark corpus of **44,890+ verified articles** (ISOT News Dataset) sanitized to eliminate publisher watermarks and wire agency artifacts:

| Metric | Score |
| :--- | :--- |
| **Validation Accuracy** | **98.63%** |
| **Inference Latency** | **< 0.25 seconds** |
| **Vocabulary Matrix** | **35,000 N-Gram Coordinates** |
| **Model Size** | **0.70 MB (Serialized via Joblib)** |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/SemanticTruth-AIFactGuard.git
cd SemanticTruth-AIFactGuard
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## 🧪 Automated Testing

Run the full end-to-end integration test suite:
```bash
python test_app.py
```
**Test Coverage Includes:**
* Landing page routing & component rendering
* Real-time API classification (`/api/analyze`) for genuine & deceptive news
* User registration, session authentication, and history dashboard logging
* Health API checks

---

## 📁 Project Directory Structure

```
SemanticTruth-AIFactGuard/
├── app.py                # Main Flask web application & API routing
├── database.py           # SQLite3 database helper & user session manager
├── ml_engine.py          # Unified hybrid ML engine & linguistic explainer
├── fake_news.py          # Entrypoint wrapper
├── train_unified.py      # Script to train & serialize the clean ML pipeline
├── test_app.py           # Automated integration test suite
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git exclusion rules
├── saved_models/         # Pre-trained serialized models (.joblib)
│   └── model_unified.joblib
├── templates/            # HTML templates (Jinja2)
│   ├── base.html         # Master layout & navigation
│   ├── landing.html      # Project showcase & overview
│   ├── detector.html     # Real-time credibility checker
│   ├── login.html        # User authentication: Sign in
│   ├── register.html     # User authentication: Sign up
│   ├── dashboard.html    # User history & audit analytics
│   └── partials/         # Modular template snippets
└── static/               # Static assets
    ├── css/style.css     # Cyber glassmorphism styling
    └── images/           # Badges & institutional assets
```

---

## 👨‍💻 Project Creator

**Vipul Yadav**  
*Master of Computer Applications (MCA) Candidate*  
**Cochin University of Science and Technology (CUSAT)**  
*Focus*: Applied Artificial Intelligence, NLP, and Full-Stack Engineering.

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use it for academic, research, and non-commercial development.

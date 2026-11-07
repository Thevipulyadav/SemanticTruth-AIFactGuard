import os
import re
import math
import joblib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
_MODEL_CACHE = None

CLICKBAIT_PATTERNS = [
    r'\b(shocking|unbelievable|mind-blowing|secret|exposed|bombshell|miracle|cure|banned)\b',
    r'\b(they don\'?t want you to know|wake up|sheeple|hoax|conspiracy|elites|deep state)\b',
    r'\b(share this|breaking alert|urgent warning|censored|hidden truth|whistleblower)\b',
    r'\b(poisoning|microchip|agenda 21|flat earth|illuminati|cabal|aliens|reptilian)\b',
    r'\b(doctors terrified|secret leaked|mainstream media won\'?t show you)\b'
]

JOURNALISTIC_PATTERNS = [
    r'\b(according to|stated|confirmed|reported|spokesperson|announced|in a statement)\b',
    r'\b(officials said|researchers found|study published|department of|ministry of)\b',
    r'\b(press conference|investigation|authorities|council voted|representatives)\b',
    r'\b(unanimously|reuters|associated press|bbc|findings|published in|spokesman)\b',
    r'\b(reuters|bloomberg|the associated press|peer-reviewed|clinical trials)\b'
]

TRUSTED_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "thehindu.com",
    "ndtv.com", "indianexpress.com", "timesofindia.indiatimes.com",
    "cnn.com", "nytimes.com", "washingtonpost.com", "theguardian.com",
    "aljazeera.com", "nature.com", "sciencemag.org", "nasa.gov",
    "who.int", "bloomberg.com", "wsj.com", "npr.org", "cnbc.com"
}

def get_unified_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    candidates = [
        os.path.join(MODELS_DIR, "model_unified.joblib"),
        os.path.join(MODELS_DIR, "model_lr.joblib"),
        os.path.join(MODELS_DIR, "model_nb.joblib")
    ]

    for p in candidates:
        if os.path.exists(p):
            try:
                _MODEL_CACHE = joblib.load(p)
                return _MODEL_CACHE
            except Exception as e:
                print(f"[ML] Error loading model from {p}: {e}")

    return None

def is_valid_url(url):
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False

def fetch_news_from_url(url):
    url = url.strip()
    if not is_valid_url(url):
        raise ValueError("Invalid URL format. Please include http:// or https://")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, timeout=12, headers=headers)
        response.raise_for_status()
    except Exception as e:
        raise ValueError(f"Could not reach the URL ({str(e)}). Please check the link or paste the text directly.")

    soup = BeautifulSoup(response.text, "html.parser")

    title = ""
    if soup.find("h1"):
        title = soup.find("h1").get_text().strip()
    elif soup.find("title"):
        title = soup.find("title").get_text().strip()

    # Clean out boilerplate
    for s in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        s.decompose()

    paragraphs = soup.find_all("p")
    text_pieces = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 25]
    content = " ".join(text_pieces)

    if len(content.strip()) < 60:
        raise ValueError("Unable to extract sufficient article text from the URL. Please copy and paste the article text directly.")

    domain = urlparse(url).netloc.lower().replace("www.", "")
    return {
        "title": title or f"Article from {domain}",
        "content": content.strip(),
        "domain": domain
    }

def analyze_news_content(text="", url=""):
    """
    Analyzes news on the basis of:
    1. Pre-trained NLP Machine Learning Model
    2. Sensationalism & Clickbait Markers
    3. Journalistic Source Attribution & Objective Phrasing
    4. Capitalization & Punctuation Emotion Signals
    5. Domain Credibility (if URL provided)
    """
    text = (text or "").strip()
    url = (url or "").strip()

    # Auto-detect if user entered a URL into the text box
    if not url and text.startswith(("http://", "https://")):
        url = text
        text = ""

    extracted_title = ""
    domain = None
    is_trusted_domain = False

    if url:
        scraped = fetch_news_from_url(url)
        content = scraped["content"]
        extracted_title = scraped["title"]
        domain = scraped["domain"]
        for td in TRUSTED_DOMAINS:
            if domain == td or domain.endswith("." + td):
                is_trusted_domain = True
                break
    elif text:
        content = text
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        if lines:
            extracted_title = lines[0][:90] + ("..." if len(lines[0]) > 90 else "")
        else:
            extracted_title = content[:60] + "..."
    else:
        raise ValueError("Please provide either a news article link or enter news text.")

    if len(content) < 25:
        raise ValueError("Input is too brief to analyze accurately. Please provide at least one complete sentence.")

    model = get_unified_model()
    text_lower = content.lower()

    # 1. Base ML model prediction
    ml_prob_real = 0.50
    if model is not None:
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([content])[0]
                ml_prob_real = float(probs[1])
            elif hasattr(model, "decision_function"):
                df_val = float(model.decision_function([content])[0])
                ml_prob_real = 1.0 / (1.0 + math.exp(-df_val))
        except Exception as e:
            print(f"[ML] Prediction fallback: {e}")
            ml_prob_real = 0.50

    # 2. Linguistic analysis
    words = re.findall(r'\b\w+\b', content)
    word_count = len(words)
    read_time = max(1, math.ceil(word_count / 200))

    # Detect clickbait / sensational words
    sensational_findings = []
    for pat in CLICKBAIT_PATTERNS:
        matches = re.findall(pat, text_lower)
        for m in matches:
            if m not in sensational_findings:
                sensational_findings.append(m)

    # Detect journalistic attribution phrases
    journalistic_findings = []
    for pat in JOURNALISTIC_PATTERNS:
        matches = re.findall(pat, text_lower)
        for m in matches:
            if m not in journalistic_findings:
                journalistic_findings.append(m)

    # Quotes detection
    quote_matches = re.findall(r'["“\'].+?["”\']', content)
    has_quotes = len(quote_matches) > 0

    # Capitalization & Exclamations
    long_words = [w for w in words if len(w) > 2]
    caps_count = sum(1 for w in long_words if w.isupper())
    caps_pct = round((caps_count / max(1, len(long_words))) * 100, 1)
    excl_count = content.count("!")

    # 3. Compute heuristic score
    heuristic_score = 0.50

    if journalistic_findings:
        heuristic_score += min(0.35, len(journalistic_findings) * 0.10)
    if has_quotes:
        heuristic_score += 0.08
    if is_trusted_domain:
        heuristic_score += 0.20

    if sensational_findings:
        heuristic_score -= min(0.40, len(sensational_findings) * 0.14)
    if caps_pct > 7.0:
        heuristic_score -= min(0.20, (caps_pct - 7.0) * 0.02)
    if excl_count >= 2:
        heuristic_score -= min(0.18, excl_count * 0.06)

    heuristic_score = max(0.04, min(0.96, heuristic_score))

    # 4. Final hybrid consensus
    final_prob_real = (ml_prob_real * 0.40) + (heuristic_score * 0.60)
    final_prob_real = max(0.02, min(0.98, final_prob_real))

    is_real = bool(final_prob_real >= 0.50)
    raw_conf = (final_prob_real if is_real else (1.0 - final_prob_real)) * 100
    confidence = round(min(99.4, max(52.0, raw_conf)), 1)

    status_code = "real" if is_real else "fake"
    status_label = "VERIFIED REAL / CREDIBLE" if is_real else "UNVERIFIED / MISLEADING"

    # Sensationalism rating
    if len(sensational_findings) >= 2 or caps_pct > 12 or excl_count >= 3:
        sensationalism_level = "High"
    elif len(sensational_findings) == 1 or caps_pct > 6 or excl_count >= 1:
        sensationalism_level = "Moderate"
    else:
        sensationalism_level = "Low"

    # 5. Formulate Human-Readable "Basis of Answer"
    basis_points = []
    if is_real:
        if journalistic_findings:
            basis_points.append(f"Objective journalistic attribution detected ({', '.join(repr(x) for x in journalistic_findings[:3])}).")
        else:
            basis_points.append("Consistent factual reporting structure without sensational hyperbole.")
        if has_quotes:
            basis_points.append("Contains direct quoted statements and verifiable source material.")
        if is_trusted_domain:
            basis_points.append(f"Originates from established reputable news organization ({domain}).")
        if caps_pct <= 5.0 and excl_count <= 1:
            basis_points.append("Tone is professional, calm, and neutral.")
    else:
        if sensational_findings:
            basis_points.append(f"Sensational clickbait keywords detected: {', '.join(repr(x) for x in sensational_findings[:4])}.")
        if caps_pct > 6.0:
            basis_points.append(f"Elevated capitalization ({caps_pct}%) indicating emotional agitation.")
        if excl_count >= 2:
            basis_points.append(f"Excessive urgent punctuation ({excl_count} exclamation marks).")
        if not journalistic_findings:
            basis_points.append("Lacks credible official sources, studies, or confirmed attributions.")

    explanation_summary = " ".join(basis_points) if basis_points else (
        "Classified based on contextual semantic patterns and structural similarity to verified reporting." if is_real else
        "Classified as potentially deceptive due to unverified statements and language patterns."
    )

    return {
        "status": status_code,
        "status_label": status_label,
        "is_real": is_real,
        "confidence": confidence,
        "title": extracted_title,
        "preview": content[:240] + ("..." if len(content) > 240 else ""),
        "full_text": content,
        "domain": domain,
        "is_trusted_domain": is_trusted_domain,
        "metrics": {
            "word_count": word_count,
            "reading_time_min": read_time,
            "uppercase_pct": caps_pct,
            "exclamation_count": excl_count,
            "sensationalism": sensationalism_level,
            "sensational_keywords": sensational_findings[:5],
            "journalistic_signals": journalistic_findings[:5],
            "has_quotes": has_quotes
        },
        "basis_explanation": explanation_summary
    }

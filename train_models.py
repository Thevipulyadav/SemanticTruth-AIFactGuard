import os
import sys
import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def train_and_save():
    print("=" * 60, flush=True)
    print("SemanticTruth AI FactGuard - Model Training Pipeline", flush=True)
    print("=" * 60, flush=True)

    fake_path = "fake.csv"
    true_path = "true.csv"

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        print(f"Error: {fake_path} or {true_path} not found!", flush=True)
        return

    t0 = time.time()
    print("Loading datasets...", flush=True)
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    fake["label"] = 0
    true["label"] = 1

    print(f"Fake news count: {len(fake):,}, Real news count: {len(true):,}", flush=True)

    # Combine title and text for stronger semantic context
    fake["content"] = fake["title"].fillna("") + " " + fake["text"].fillna("")
    true["content"] = true["title"].fillna("") + " " + true["text"].fillna("")

    data = pd.concat([fake[["content", "label"]], true[["content", "label"]]])
    data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)

    X = data["content"].astype(str)
    y = data["label"]

    print(f"Total dataset size: {len(data):,} articles loaded in {time.time()-t0:.2f}s", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Multinomial Naive Bayes
    print("\n--- Training Model 1: Multinomial Naive Bayes (Fast & High-Precision) ---", flush=True)
    t_start = time.time()
    nb_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_df=0.8,
            min_df=3,
            max_features=40000,
            ngram_range=(1, 2)
        )),
        ("clf", MultinomialNB(alpha=0.1))
    ])
    nb_pipe.fit(X_train, y_train)
    nb_acc = accuracy_score(y_test, nb_pipe.predict(X_test))
    print(f"Naive Bayes Accuracy: {nb_acc * 100:.2f}% (Trained in {time.time()-t_start:.2f}s)", flush=True)
    nb_path = os.path.join(models_dir, "model_nb.joblib")
    joblib.dump(nb_pipe, nb_path, compress=3)
    print(f"Saved: {nb_path} ({os.path.getsize(nb_path)/(1024*1024):.2f} MB)", flush=True)

    # 2. Logistic Regression (High Accuracy & Calibrated Probabilities)
    print("\n--- Training Model 2: Logistic Regression ---", flush=True)
    t_start = time.time()
    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_df=0.8,
            min_df=3,
            max_features=40000,
            ngram_range=(1, 2)
        )),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"))
    ])
    lr_pipe.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr_pipe.predict(X_test))
    print(f"Logistic Regression Accuracy: {lr_acc * 100:.2f}% (Trained in {time.time()-t_start:.2f}s)", flush=True)
    lr_path = os.path.join(models_dir, "model_lr.joblib")
    joblib.dump(lr_pipe, lr_path, compress=3)
    print(f"Saved: {lr_path} ({os.path.getsize(lr_path)/(1024*1024):.2f} MB)", flush=True)

    # 3. Passive Aggressive Classifier (Optimized for Online Misinformation Tracking)
    print("\n--- Training Model 3: Passive Aggressive Classifier ---", flush=True)
    t_start = time.time()
    pac_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_df=0.8,
            min_df=3,
            max_features=40000,
            ngram_range=(1, 2)
        )),
        ("clf", PassiveAggressiveClassifier(max_iter=100, random_state=42, C=0.5))
    ])
    pac_pipe.fit(X_train, y_train)
    pac_acc = accuracy_score(y_test, pac_pipe.predict(X_test))
    print(f"Passive Aggressive Accuracy: {pac_acc * 100:.2f}% (Trained in {time.time()-t_start:.2f}s)", flush=True)
    pac_path = os.path.join(models_dir, "model_pac.joblib")
    joblib.dump(pac_pipe, pac_path, compress=3)
    print(f"Saved: {pac_path} ({os.path.getsize(pac_path)/(1024*1024):.2f} MB)", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("All models successfully trained and serialized!", flush=True)
    print(f"Total time elapsed: {time.time()-t0:.2f}s", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    train_and_save()

import os
import sys
import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def clean_and_train():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("Training Clean Semantic Model (Removing Wire Agency Artifacts)...", flush=True)
    print("=" * 60, flush=True)

    fake = pd.read_csv("fake.csv")
    true = pd.read_csv("true.csv")

    print(f"Loaded {len(fake):,} fake and {len(true):,} real articles in {time.time()-t0:.2f}s", flush=True)

    # 1. Clean True news: Remove '(Reuters) - ' prefixes and 'Reuters' mentions that cause model bias
    true_text = true["text"].fillna("")
    true_text = true_text.str.replace(r"^[A-Z\s,]+?\(Reuters\)\s*-\s*", "", regex=True)
    true_text = true_text.str.replace(r"\(Reuters\)\s*-\s*", "", regex=True)
    true_text = true_text.str.replace(r"\bReuters\b", "", regex=True)
    true_text = true_text.str.replace(r"\b(AP|AFP|Bloomberg)\b", "", regex=True)

    # 2. Clean Fake news: Remove photo credits and social media footers
    fake_text = fake["text"].fillna("")
    fake_text = fake_text.str.replace(r"Photo by .+?Getty Images\.?", "", regex=True)
    fake_text = fake_text.str.replace(r"Featured image via.+?$", "", regex=True)

    # Combine title + cleaned text
    true_full = true["title"].fillna("") + " " + true_text
    fake_full = fake["title"].fillna("") + " " + fake_text

    df_true = pd.DataFrame({"text": true_full, "label": 1})
    df_fake = pd.DataFrame({"text": fake_full, "label": 0})

    df = pd.concat([df_true, df_fake]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Filter out empty or extremely short rows
    df = df[df["text"].str.strip().str.len() > 30].reset_index(drop=True)
    print(f"Cleaned dataset ready: {len(df):,} articles", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.15, random_state=42, stratify=df["label"]
    )

    print("Fitting TF-IDF + Logistic Regression pipeline...", flush=True)
    t_train = time.time()
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_df=0.75,
            min_df=4,
            max_features=35000,
            ngram_range=(1, 2)
        )),
        ("clf", LogisticRegression(max_iter=500, C=1.5, solver="lbfgs"))
    ])

    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    print(f"Model fitted in {time.time()-t_train:.2f}s with Validation Accuracy: {acc*100:.2f}%", flush=True)

    # Save to saved_models/model_unified.joblib
    os.makedirs("saved_models", exist_ok=True)
    save_path = os.path.join("saved_models", "model_unified.joblib")
    joblib.dump(pipe, save_path, compress=3)
    print(f"Saved: {save_path} ({os.path.getsize(save_path)/(1024*1024):.2f} MB)", flush=True)

    # Test edge cases
    test_cases = [
        ("BBC / Real Reporting", "The Ministry of Health confirmed today that new clinical trials demonstrate a significant reduction in chronic symptoms among treated patients."),
        ("Normal Real News", "City council members voted unanimously on Tuesday to approve funding for the new municipal water filtration project following a two-month environmental review."),
        ("Space Real News", "Astronomers using the James Webb Space Telescope detected atmospheric water vapor on a distant exoplanet, according to findings published in the journal Nature."),
        ("Conspiracy Fake News", "SHOCKING PROOF: Government secretly poisoning the clouds with chemical lasers to manipulate human DNA! Share this now before the elites take it down!"),
        ("Clickbait Fake News", "Miracle home remedy melts 30 pounds of belly fat overnight while you sleep and doctors are terrified! Watch the leaked video before it is banned!")
    ]

    print("\n--- Live Test Matrix ---", flush=True)
    for name, text in test_cases:
        pred = pipe.predict([text])[0]
        prob = pipe.predict_proba([text])[0]
        verdict = "REAL 🟢" if pred == 1 else "FAKE 🔴"
        conf = prob[pred] * 100
        print(f"[{name}] -> {verdict} ({conf:.1f}% confidence)", flush=True)

if __name__ == "__main__":
    clean_and_train()

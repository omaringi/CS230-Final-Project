import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
df = pd.read_csv("data/processed/documents.csv")
X_train_emb = np.load("data/processed/X_train.npy")
X_test_emb = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

with open("data/processed/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Fill NaN texts with empty string
df['text'] = df['text'].fillna('')

# Recreate labels for text split
le_temp = LabelEncoder()
df['label'] = le_temp.fit_transform(df['section'])

print("=" * 50)
print("BASELINE 1: TF-IDF + Logistic Regression")
print("=" * 50)

# Recreate train/test text split (same random state as create_splits.py)
texts = df['text'].tolist()
labels = df['label'].values
X_text_train, X_text_test, _, _ = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf_train = tfidf.fit_transform(X_text_train)
X_tfidf_test = tfidf.transform(X_text_test)

# Logistic Regression on TF-IDF
clf_tfidf = LogisticRegression(max_iter=1000, multi_class='multinomial')
clf_tfidf.fit(X_tfidf_train, y_train)
y_pred_tfidf = clf_tfidf.predict(X_tfidf_test)

acc_tfidf = accuracy_score(y_test, y_pred_tfidf)
print(f"TF-IDF + LogReg Accuracy: {acc_tfidf:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tfidf, target_names=le.classes_, zero_division=0))

print("\n" + "=" * 50)
print("BASELINE 2: BioBERT Embeddings + Logistic Regression")
print("=" * 50)

# Logistic Regression on BioBERT embeddings
clf_biobert = LogisticRegression(max_iter=1000, multi_class='multinomial')
clf_biobert.fit(X_train_emb, y_train)
y_pred_biobert = clf_biobert.predict(X_test_emb)

acc_biobert = accuracy_score(y_test, y_pred_biobert)
print(f"BioBERT + LogReg Accuracy: {acc_biobert:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_biobert, target_names=le.classes_, zero_division=0))

# Save results
results = {
    'tfidf_logreg': acc_tfidf,
    'biobert_logreg': acc_biobert,
}

with open("results/baseline_results.pkl", "wb") as f:
    pickle.dump(results, f)

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"TF-IDF + LogReg:   {acc_tfidf:.4f}")
print(f"BioBERT + LogReg:  {acc_biobert:.4f}")
print("\nBaseline results saved!")
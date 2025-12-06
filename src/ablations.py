import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

# Load data
X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

results = []

print("=" * 60)
print("ABLATION 1: Different Classifiers on BioBERT Embeddings")
print("=" * 60)

# LogReg
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
results.append({'Experiment': 'BioBERT + LogReg', 'Accuracy': acc})
print(f"LogReg: {acc:.4f}")

# SVM
clf = SVC(kernel='rbf', C=1.0)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
results.append({'Experiment': 'BioBERT + SVM', 'Accuracy': acc})
print(f"SVM: {acc:.4f}")

# MLP
clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
results.append({'Experiment': 'BioBERT + MLP', 'Accuracy': acc})
print(f"MLP: {acc:.4f}")

print("\n" + "=" * 60)
print("ABLATION 2: Different Embedding Models")
print("=" * 60)

# Load original texts for re-embedding
df = pd.read_csv("data/processed/documents.csv")
df['text'] = df['text'].fillna('')

# Create labels from section
le = LabelEncoder()
df['label'] = le.fit_transform(df['section'])

texts = df['text'].tolist()
labels = df['label'].values
X_text_train, X_text_test, y_train_txt, y_test_txt = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

embedding_models = [
    ('BERT (general)', 'bert-base-uncased'),
    ('BioBERT', 'dmis-lab/biobert-base-cased-v1.2'),
]

for name, model_name in embedding_models:
    print(f"\nTesting {name}...")
    try:
        model = SentenceTransformer(model_name)
        X_train_emb = model.encode(X_text_train, show_progress_bar=True, batch_size=8)
        X_test_emb = model.encode(X_text_test, show_progress_bar=True, batch_size=8)
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_emb, y_train_txt)
        acc = accuracy_score(y_test_txt, clf.predict(X_test_emb))
        
        results.append({'Experiment': f'{name} + LogReg', 'Accuracy': acc})
        print(f"{name}: {acc:.4f}")
    except Exception as e:
        print(f"Error with {name}: {e}")

# Save results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)
print("\n" + "=" * 60)
print("ALL ABLATION RESULTS")
print("=" * 60)
print(results_df.to_string(index=False))
results_df.to_csv('results/ablation_results.csv', index=False)
print("\nSaved to results/ablation_results.csv")
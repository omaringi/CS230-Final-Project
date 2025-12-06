import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from model import HierarchicalCTDClassifier

# Load data
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

with open("data/processed/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("data/processed/label_encoder_l1.pkl", "rb") as f:
    le_l1 = pickle.load(f)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_level1 = len(le_l1.classes_)
n_level2 = len(le.classes_)

model = HierarchicalCTDClassifier(
    embedding_dim=768,
    hidden_dim=256,
    n_level1=n_level1,
    n_level2=n_level2,
).to(device)

model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
model.eval()

# Predict
X_test_t = torch.FloatTensor(X_test).to(device)
with torch.no_grad():
    pred1, pred2 = model(X_test_t)
    y_pred = pred2.argmax(1).cpu().numpy()
    y_pred_l1 = pred1.argmax(1).cpu().numpy()

# Results
print("=" * 60)
print("HIERARCHICAL MODEL RESULTS")
print("=" * 60)
print(f"\nSection Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Hierarchical CTD Classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
print("\nSaved figures/confusion_matrix.png")
plt.show()

# Load baseline results for comparison
with open("results/baseline_results.pkl", "rb") as f:
    baseline = pickle.load(f)

# Create comparison table
results_df = pd.DataFrame({
    'Method': ['TF-IDF + LogReg', 'BioBERT + LogReg', 'Hierarchical NN (Ours)'],
    'Accuracy': [
        baseline['tfidf_logreg'],
        baseline['biobert_logreg'],
        accuracy_score(y_test, y_pred)
    ]
})

print("\n" + "=" * 60)
print("COMPARISON TABLE")
print("=" * 60)
print(results_df.to_string(index=False))

# Save
results_df.to_csv('results/comparison.csv', index=False)

# Bar chart
plt.figure(figsize=(10, 6))
colors = ['#ff9999', '#66b3ff', '#99ff99']
bars = plt.bar(results_df['Method'], results_df['Accuracy'], color=colors)
plt.title('Model Comparison: CTD Document Classification', fontsize=14)
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for bar, v in zip(bars, results_df['Accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}', 
             ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/model_comparison.png', dpi=150)
print("Saved figures/model_comparison.png")
plt.show()

print("\nEvaluation complete!")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
df = pd.read_csv("data/processed/documents.csv")
embeddings = np.load("data/embeddings/biobert_embeddings.npy")

print(f"Loaded {len(df)} documents, {embeddings.shape[0]} embeddings")

# Create label encodings
le_section = LabelEncoder()
df['label'] = le_section.fit_transform(df['section'])

# Create level 1 labels (Module level)
def get_module(section):
    if section.startswith('2.'):
        return 'Module2'
    elif section.startswith('3.'):
        return 'Module3'
    elif section.startswith('5.'):
        return 'Module5'
    return 'Other'

df['module'] = df['section'].apply(get_module)
le_module = LabelEncoder()
df['label_level1'] = le_module.fit_transform(df['module'])

# Save label mappings
print("\nLevel 1 (Module) classes:")
for i, cls in enumerate(le_module.classes_):
    count = (df['module'] == cls).sum()
    print(f"  {i}: {cls} ({count} docs)")

print("\nLevel 2 (Section) classes:")
for i, cls in enumerate(le_section.classes_):
    count = (df['section'] == cls).sum()
    print(f"  {i}: {cls} ({count} docs)")

# Train/test split (stratified by section)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    embeddings, 
    df['label'].values,
    df.index.values,
    test_size=0.2, 
    stratify=df['label'],
    random_state=42
)

# Get level1 labels for train/test
y_train_l1 = df.loc[idx_train, 'label_level1'].values
y_test_l1 = df.loc[idx_test, 'label_level1'].values

# Save everything
np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_test.npy", y_test)
np.save("data/processed/y_train_l1.npy", y_train_l1)
np.save("data/processed/y_test_l1.npy", y_test_l1)

with open("data/processed/label_encoder.pkl", "wb") as f:
    pickle.dump(le_section, f)

with open("data/processed/label_encoder_l1.pkl", "wb") as f:
    pickle.dump(le_module, f)

print(f"\nSaved splits: Train={len(X_train)}, Test={len(X_test)}")
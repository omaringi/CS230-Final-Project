import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

print(f"CUDA available: {torch.cuda.is_available()}")

# Load data
df = pd.read_csv("data/processed/documents.csv")
print(f"Loaded {len(df)} documents")

# Load BioBERT model
print("Loading BioBERT...")
model = SentenceTransformer('dmis-lab/biobert-base-cased-v1.2')

# Compute embeddings
print("Computing embeddings...")
texts = df['text'].tolist()
embeddings = model.encode(texts, show_progress_bar=True, batch_size=8)

# Save
np.save("data/embeddings/biobert_embeddings.npy", embeddings)
print(f"Saved embeddings shape: {embeddings.shape}")
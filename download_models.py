print("Downloading models (one-time)...")

from sentence_transformers import SentenceTransformer

print("1/3 BioBERT...")
m1 = SentenceTransformer('dmis-lab/biobert-base-cased-v1.2')

print("2/3 PubMedBERT...")
m2 = SentenceTransformer('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')

print("3/3 BERT base...")
m3 = SentenceTransformer('bert-base-uncased')

print("\n✓ All models cached and ready")
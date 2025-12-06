import torch

print("=" * 50)
print("GPU TEST")
print("=" * 50)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("No GPU - will use CPU (slower but works fine)")

# Test BioBERT download
print("\n" + "=" * 50)
print("BIOBERT TEST")
print("=" * 50)

from sentence_transformers import SentenceTransformer

print("Downloading BioBERT (first time takes a few minutes)...")
model = SentenceTransformer('dmis-lab/biobert-base-cased-v1.2')
print("✓ BioBERT loaded")

# Test embedding
test_text = "Stability studies were conducted at 25°C/60% RH for 12 months"
embedding = model.encode(test_text)
print(f"✓ Test embedding shape: {embedding.shape}")

print("\n" + "=" * 50)
print("ALL TESTS PASSED - READY TO GO")
print("=" * 50)
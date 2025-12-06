import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import HierarchicalCTDClassifier, HierarchicalLoss

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train.npy")  # Fine labels (section)
y_test = np.load("data/processed/y_test.npy")

# Load level1 labels (already saved by create_splits.py)
y_train_l1 = np.load("data/processed/y_train_l1.npy")
y_test_l1 = np.load("data/processed/y_test_l1.npy")

# Load label encoders
with open("data/processed/label_encoder.pkl", "rb") as f:
    le_section = pickle.load(f)

with open("data/processed/label_encoder_l1.pkl", "rb") as f:
    le_module = pickle.load(f)

print(f"Level 1 classes: {le_module.classes_}")
print(f"Level 2 classes: {le_section.classes_}")
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
X_test_t = torch.FloatTensor(X_test)
y_train_l1_t = torch.LongTensor(y_train_l1)
y_train_l2_t = torch.LongTensor(y_train)
y_test_l1_t = torch.LongTensor(y_test_l1)
y_test_l2_t = torch.LongTensor(y_test)

# DataLoaders
train_dataset = TensorDataset(X_train_t, y_train_l1_t, y_train_l2_t)
test_dataset = TensorDataset(X_test_t, y_test_l1_t, y_test_l2_t)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model
n_level1 = len(le_module.classes_)
n_level2 = len(le_section.classes_)

model = HierarchicalCTDClassifier(
    embedding_dim=768,
    hidden_dim=256,
    n_level1=n_level1,
    n_level2=n_level2,
    dropout=0.3
).to(device)

print(f"\nModel: {n_level1} level1 classes, {n_level2} level2 classes")

# Loss and optimizer
criterion = HierarchicalLoss(weight_level1=0.3, weight_level2=0.7)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Training
n_epochs = 50
best_acc = 0
train_losses = []
val_accs_l1 = []
val_accs_l2 = []

print("\nStarting training...")
print("=" * 60)

for epoch in range(n_epochs):
    # Train
    model.train()
    epoch_loss = 0
    for X_batch, y1_batch, y2_batch in train_loader:
        X_batch = X_batch.to(device)
        y1_batch = y1_batch.to(device)
        y2_batch = y2_batch.to(device)
        
        optimizer.zero_grad()
        pred1, pred2 = model(X_batch)
        loss = criterion(pred1, pred2, y1_batch, y2_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # Evaluate
    model.eval()
    correct_l1, correct_l2, total = 0, 0, 0
    with torch.no_grad():
        for X_batch, y1_batch, y2_batch in test_loader:
            X_batch = X_batch.to(device)
            y1_batch = y1_batch.to(device)
            y2_batch = y2_batch.to(device)
            
            pred1, pred2 = model(X_batch)
            
            correct_l1 += (pred1.argmax(1) == y1_batch).sum().item()
            correct_l2 += (pred2.argmax(1) == y2_batch).sum().item()
            total += y1_batch.size(0)
    
    acc_l1 = correct_l1 / total
    acc_l2 = correct_l2 / total
    val_accs_l1.append(acc_l1)
    val_accs_l2.append(acc_l2)
    
    scheduler.step(avg_loss)
    
    print(f"Epoch {epoch+1:2d}/{n_epochs} | Loss: {avg_loss:.4f} | "
          f"Module Acc: {acc_l1:.4f} | Section Acc: {acc_l2:.4f}")
    
    # Save best model
    if acc_l2 > best_acc:
        best_acc = acc_l2
        torch.save(model.state_dict(), "models/best_model.pt")
        print(f"  -> Saved best model (acc={best_acc:.4f})")

print("=" * 60)
print(f"Training complete! Best Section Accuracy: {best_acc:.4f}")

# Plot training curves
plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 3, 2)
plt.plot(val_accs_l1, label='Module (Level 1)')
plt.plot(val_accs_l2, label='Section (Level 2)')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(val_accs_l2)
plt.title('Section Accuracy (Level 2)')
plt.xlabel('Epoch')

plt.tight_layout()
plt.savefig('figures/training_curves.png', dpi=150)
print("Saved figures/training_curves.png")
plt.show()

print(f"\nBest Level 2 (Section) Accuracy: {best_acc:.4f}")
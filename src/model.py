import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalCTDClassifier(nn.Module):
    """
    Hierarchical classifier for CTD sections.
    Predicts coarse level first, then uses that to help fine prediction.
    """
    def __init__(self, embedding_dim=768, hidden_dim=256, 
                 n_level1=5, n_level2=10, dropout=0.3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Level 1 head (coarse: Module 2, 3, 5, etc.)
        self.level1_head = nn.Linear(hidden_dim, n_level1)
        
        # Level 2 head (fine: 3.2.P.2, 3.2.P.4, etc.)
        # Takes shared features + level1 prediction
        self.level2_head = nn.Linear(hidden_dim + n_level1, n_level2)
        
    def forward(self, x):
        # Shared encoding
        shared = self.encoder(x)
        
        # Level 1 prediction
        level1_logits = self.level1_head(shared)
        level1_probs = F.softmax(level1_logits, dim=-1)
        
        # Level 2 prediction (conditioned on level 1)
        level2_input = torch.cat([shared, level1_probs], dim=-1)
        level2_logits = self.level2_head(level2_input)
        
        return level1_logits, level2_logits


class HierarchicalLoss(nn.Module):
    """
    Combined loss for hierarchical classification.
    Weights coarse and fine predictions.
    """
    def __init__(self, weight_level1=0.3, weight_level2=0.7):
        super().__init__()
        self.weight_level1 = weight_level1
        self.weight_level2 = weight_level2
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, pred_level1, pred_level2, target_level1, target_level2):
        loss1 = self.ce(pred_level1, target_level1)
        loss2 = self.ce(pred_level2, target_level2)
        return self.weight_level1 * loss1 + self.weight_level2 * loss2
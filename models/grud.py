"""
GRU-D Model: Handles missing data with time-decay mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GRUD(nn.Module):
    """
    GRU-D Model: Gated Recurrent Unit with Decay mechanism for handling missing data
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.2):
        """
        Args:
            input_size: Number of input features
            hidden_size: Hidden dimension size
            num_layers: Number of GRU layers
            dropout: Dropout rate
        """
        super(GRUD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Decay mechanism for missing values
        self.decay_gamma = nn.Parameter(torch.ones(input_size))
        
        # GRU layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            features: [batch_size, seq_len, input_size] - input features (NaN filled with 0)
            masks: [batch_size, seq_len, input_size] - mask indicating observed values
            
        Returns:
            Logits: [batch_size, 1]
        """
        batch_size, seq_len, input_size = features.shape
        
        # Last observed values and time since last observation
        last_observed = torch.zeros_like(features)
        time_since_last = torch.zeros_like(features)
        
        # Compute decayed values
        for t in range(seq_len):
            if t == 0:
                last_observed[:, t] = features[:, t] * masks[:, t]
                time_since_last[:, t] = torch.ones_like(features[:, t])
            else:
                # Update last observed values
                update_mask = masks[:, t].bool()
                last_observed[:, t] = torch.where(
                    update_mask,
                    features[:, t],
                    last_observed[:, t-1]
                )
                
                # Time since last observation (simplified - using step increment)
                time_since_last[:, t] = torch.where(
                    update_mask,
                    torch.ones_like(time_since_last[:, t]),
                    time_since_last[:, t-1] + 1.0
                )
        
        # Decay mechanism: exp(-gamma * delta_t)
        decay = torch.exp(-torch.relu(self.decay_gamma) * time_since_last)
        
        # Impute missing values using decay
        mask_float = masks.float()
        imputed_features = mask_float * features + (1 - mask_float) * (last_observed * decay)
        
        # GRU processing
        gru_out, _ = self.gru(imputed_features)
        
        # Use last hidden state
        last_hidden = gru_out[:, -1, :]
        
        # Output layer
        output = self.fc(self.dropout(last_hidden))
        
        return output

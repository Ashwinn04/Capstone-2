"""
Transformer Model: Attention-based modeling for long-range dependencies
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Transformer(nn.Module):
    """
    Transformer Model: Uses multi-head attention for modeling long-range dependencies
    """
    
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.2):
        """
        Args:
            input_size: Number of input features
            d_model: Model dimension (must be divisible by nhead)
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dropout: Dropout rate
        """
        super(Transformer, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection to model dimension
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            features: [batch_size, seq_len, input_size] - input features
            masks: [batch_size, seq_len, input_size] - mask indicating observed values
            
        Returns:
            Logits: [batch_size, 1]
        """
        batch_size, seq_len, input_size = features.shape
        
        # Apply mask to features
        masked_features = features * masks.float()
        
        # Project to model dimension
        x = self.input_projection(masked_features)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask (1 for valid positions, 0 for padding)
        # For simplicity, assume all positions are valid
        padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        
        # Transformer encoder
        # Note: PyTorch's TransformerEncoder expects mask where True = ignore
        src_key_padding_mask = ~padding_mask
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Use mean pooling over sequence dimension
        # Or use CLS token if available
        pooled = encoded.mean(dim=1)  # [batch_size, d_model]
        
        # Output layer
        output = self.fc(self.dropout(pooled))
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

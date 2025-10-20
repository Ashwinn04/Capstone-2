"""
Transformer model implementation for sepsis prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input
        
        Args:
            x: Input tensor [seq_len, batch_size, d_model]
            
        Returns:
            Input with positional encoding
        """
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    """
    Transformer model for sepsis prediction
    """
    
    def __init__(self, input_size: int, d_model: int = 128, 
                 nhead: int = 8, num_layers: int = 4,
                 dim_feedforward: int = 512, dropout: float = 0.3,
                 output_size: int = 1, max_seq_len: int = 100, use_attention: bool = False,
                 multitask: bool = False):
        super(TransformerModel, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        # Augmentation projection to compress [x | mask | delta_t] -> input_size
        self.aug_projection = nn.Linear(input_size * 3, input_size)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output layers
        self.dropout_layer = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, output_size)
        self.multitask = multitask
        if self.multitask:
            self.tto_head = nn.Linear(d_model, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def create_padding_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask for transformer
        
        Args:
            mask: Missing value mask [batch_size, seq_len, input_size]
            
        Returns:
            Padding mask [batch_size, seq_len]
        """
        # If any feature is missing at a time step, mask that time step
        padding_mask = torch.all(mask, dim=2)  # [batch_size, seq_len]
        return ~padding_mask  # True for padding positions
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor, delta_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input sequences [batch_size, seq_len, input_size]
            mask: Missing value masks [batch_size, seq_len, input_size]
            delta_t: Optional time since last observation [batch_size, seq_len, input_size]
            
        Returns:
            Output logits [batch_size, output_size]
        """
        batch_size, seq_len, input_size = x.shape
        
        # Handle missing values
        x_masked = x * mask.float()
        # Fuse time and missingness information when available
        if delta_t is not None:
            x_aug = torch.cat([x_masked, mask.float(), delta_t], dim=2)
            x_base = self.aug_projection(x_aug)
        else:
            x_base = x_masked
        
        # Project input to model dimension
        x_proj = self.input_projection(x_base)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x_proj = x_proj.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x_proj = self.pos_encoding(x_proj)
        x_proj = x_proj.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Create padding mask
        padding_mask = self.create_padding_mask(mask)
        
        # Transformer encoder
        transformer_out = self.transformer_encoder(
            x_proj, 
            src_key_padding_mask=padding_mask
        )  # [batch_size, seq_len, d_model]
        
        # Use the last non-padded output
        # For simplicity, use the last time step
        final_output = transformer_out[:, -1, :]  # [batch_size, d_model]
        
        # Apply dropout
        final_output = self.dropout_layer(final_output)
        
        # Output layer
        logits = self.output_layer(final_output)
        if self.multitask:
            tto_pred = self.tto_head(final_output)
            return logits, tto_pred
        return logits
    
    def get_attention_weights(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability
        
        Args:
            x: Input sequences
            mask: Missing value masks
            
        Returns:
            Attention weights [batch_size, seq_len]
        """
        batch_size, seq_len, input_size = x.shape
        
        # Handle missing values
        x_masked = x * mask.float()
        
        # Project input
        x_proj = self.input_projection(x_masked)
        
        # Add positional encoding
        x_proj = x_proj.transpose(0, 1)
        x_proj = self.pos_encoding(x_proj)
        x_proj = x_proj.transpose(0, 1)
        
        # Create padding mask
        padding_mask = self.create_padding_mask(mask)
        
        # Get attention weights from transformer
        # This is a simplified version - in practice, you'd need to modify
        # the transformer to return attention weights
        transformer_out = self.transformer_encoder(x_proj, src_key_padding_mask=padding_mask)
        
        # Calculate attention weights using final output
        final_hidden = transformer_out[:, -1, :]
        
        # Compute attention scores
        attention_scores = torch.sum(transformer_out * final_hidden.unsqueeze(1), dim=2)
        
        # Mask out padding positions
        attention_scores = attention_scores.masked_fill(padding_mask, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=1)
        
        return attention_weights


class AttentionTransformerModel(nn.Module):
    """
    Transformer model with explicit attention mechanism
    """
    
    def __init__(self, input_size: int, d_model: int = 128, 
                 nhead: int = 8, num_layers: int = 4,
                 dim_feedforward: int = 512, dropout: float = 0.3,
                 output_size: int = 1, max_seq_len: int = 100, use_attention: bool = False,
                 multitask: bool = False):
        super(AttentionTransformerModel, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        # Augmentation projection to compress [x | mask | delta_t] -> input_size
        self.aug_projection = nn.Linear(input_size * 3, input_size)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Additional attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.dropout_layer = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, output_size)
        self.multitask = multitask
        if self.multitask:
            self.tto_head = nn.Linear(d_model, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def create_padding_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Create padding mask for transformer"""
        padding_mask = torch.all(mask, dim=2)
        return ~padding_mask
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor, delta_t: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention
        
        Args:
            x: Input sequences [batch_size, seq_len, input_size]
            mask: Missing value masks [batch_size, seq_len, input_size]
            delta_t: Optional time since last observation [batch_size, seq_len, input_size]
            
        Returns:
            Tuple of (output_logits, attention_weights)
        """
        batch_size, seq_len, input_size = x.shape
        
        # Handle missing values
        x_masked = x * mask.float()
        # Fuse time and missingness information when available
        if delta_t is not None:
            x_aug = torch.cat([x_masked, mask.float(), delta_t], dim=2)
            x_base = self.aug_projection(x_aug)
        else:
            x_base = x_masked
        
        # Project input
        x_proj = self.input_projection(x_base)
        
        # Add positional encoding
        x_proj = x_proj.transpose(0, 1)
        x_proj = self.pos_encoding(x_proj)
        x_proj = x_proj.transpose(0, 1)
        
        # Create padding mask
        padding_mask = self.create_padding_mask(mask)
        
        # Transformer encoder
        transformer_out = self.transformer_encoder(x_proj, src_key_padding_mask=padding_mask)
        
        # Additional attention layer
        attn_output, attn_weights = self.attention(
            transformer_out, transformer_out, transformer_out,
            key_padding_mask=padding_mask
        )
        
        # Use the last non-padded output
        final_output = attn_output[:, -1, :]
        
        # Apply dropout
        final_output = self.dropout_layer(final_output)
        
        # Output layer
        logits = self.output_layer(final_output)
        
        # Average attention weights across heads
        attn_weights_avg = attn_weights.mean(dim=1)  # [batch_size, seq_len, seq_len]
        
        # Get attention weights for the last time step
        final_attention = attn_weights_avg[:, -1, :]  # [batch_size, seq_len]
        if self.multitask:
            tto_pred = self.tto_head(final_output)
            return (logits, tto_pred), final_attention
        return logits, final_attention


def create_transformer_model(input_size: int, d_model: int = 128, 
                           nhead: int = 8, num_layers: int = 4,
                           dim_feedforward: int = 512, dropout: float = 0.3,
                           use_attention: bool = False, multitask: bool = False) -> nn.Module:
    """
    Create Transformer model with specified parameters
    
    Args:
        input_size: Number of input features
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout rate
        use_attention: Whether to use additional attention layer
        
    Returns:
        Transformer model instance
    """
    if use_attention:
        model = AttentionTransformerModel(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            multitask=multitask
        )
    else:
        model = TransformerModel(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            multitask=multitask
        )
    
    return model


def count_transformer_parameters(model: nn.Module) -> int:
    """Count trainable parameters in Transformer model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage and testing
if __name__ == "__main__":
    # Test basic Transformer model
    batch_size = 32
    seq_len = 24
    input_size = 20
    
    print("Testing Basic Transformer Model:")
    model = create_transformer_model(input_size=input_size, use_attention=False)
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, input_size)
    mask = torch.ones(batch_size, seq_len, input_size)
    
    # Add some missing values
    mask[:, :, :5] = torch.rand(batch_size, seq_len, 5) > 0.1
    
    # Forward pass
    output = model(x, mask)
    print(f"Model output shape: {output.shape}")
    print(f"Number of parameters: {count_transformer_parameters(model):,}")
    
    # Test attention weights
    attention_weights = model.get_attention_weights(x, mask)
    print(f"Attention weights shape: {attention_weights.shape}")
    
    print("\nTesting Attention Transformer Model:")
    # Test attention Transformer model
    att_model = create_transformer_model(input_size=input_size, use_attention=True)
    
    # Forward pass
    output, att_weights = att_model(x, mask)
    print(f"Model output shape: {output.shape}")
    print(f"Attention weights shape: {att_weights.shape}")
    print(f"Number of parameters: {count_transformer_parameters(att_model):,}")

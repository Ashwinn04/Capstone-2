"""
Bidirectional LSTM model implementation for sepsis prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class LSTMModel(nn.Module):
    """
    Bidirectional LSTM model for sepsis prediction
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3,
                 output_size: int = 1, bidirectional: bool = True):
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        # Augmentation projection: [x | mask | delta_t] -> input_size
        self.aug_projection = nn.Linear(input_size * 3, input_size)
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Linear(lstm_output_size, output_size)
        
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
        
        # Handle missing values by zeroing them out
        x_masked = x * mask.float()
        if delta_t is not None:
            x_aug = torch.cat([x_masked, mask.float(), delta_t], dim=2)
            x_base = self.aug_projection(x_aug)
        else:
            x_base = x_masked
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x_base)
        
        # Use the last output (final hidden state)
        # For bidirectional LSTM, this contains both forward and backward information
        final_output = lstm_out[:, -1, :]  # [batch_size, lstm_output_size]
        
        # Apply dropout
        final_output = self.dropout_layer(final_output)
        
        # Output layer
        output = self.output_layer(final_output)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability using LSTM outputs
        
        Args:
            x: Input sequences
            mask: Missing value masks
            
        Returns:
            Attention weights [batch_size, seq_len]
        """
        batch_size, seq_len, input_size = x.shape
        
        # Handle missing values
        x_masked = x * mask.float()
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x_masked)  # [batch_size, seq_len, lstm_output_size]
        
        # Calculate attention weights using final hidden state
        final_hidden = lstm_out[:, -1, :]  # [batch_size, lstm_output_size]
        
        # Compute attention scores
        attention_scores = torch.sum(lstm_out * final_hidden.unsqueeze(1), dim=2)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        return attention_weights
    
    def get_hidden_states(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Get all hidden states for analysis
        
        Args:
            x: Input sequences
            mask: Missing value masks
            
        Returns:
            Hidden states [batch_size, seq_len, lstm_output_size]
        """
        x_masked = x * mask.float()
        lstm_out, _ = self.lstm(x_masked)
        return lstm_out


class AttentionLSTMModel(nn.Module):
    """
    LSTM model with attention mechanism for better interpretability
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3,
                 output_size: int = 1, bidirectional: bool = True):
        super(AttentionLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        # Augmentation projection: [x | mask | delta_t] -> input_size
        self.aug_projection = nn.Linear(input_size * 3, input_size)
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism
        self.attention = nn.Linear(lstm_output_size, 1)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Linear(lstm_output_size, output_size)
        
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
        if delta_t is not None:
            x_aug = torch.cat([x_masked, mask.float(), delta_t], dim=2)
            x_base = self.aug_projection(x_aug)
        else:
            x_base = x_masked
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x_base)  # [batch_size, seq_len, lstm_output_size]
        
        # Calculate attention weights
        attention_scores = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, seq_len]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention to get context vector
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        
        # Apply dropout
        context_vector = self.dropout_layer(context_vector)
        
        # Output layer
        output = self.output_layer(context_vector)
        
        return output, attention_weights


def create_lstm_model(input_size: int, hidden_size: int = 128, 
                     num_layers: int = 2, dropout: float = 0.3,
                     use_attention: bool = False) -> nn.Module:
    """
    Create LSTM model with specified parameters
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden state size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        use_attention: Whether to use attention mechanism
        
    Returns:
        LSTM model instance
    """
    if use_attention:
        model = AttentionLSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    
    return model


def count_lstm_parameters(model: nn.Module) -> int:
    """Count trainable parameters in LSTM model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage and testing
if __name__ == "__main__":
    # Test basic LSTM model
    batch_size = 32
    seq_len = 24
    input_size = 20
    
    print("Testing Basic LSTM Model:")
    model = create_lstm_model(input_size=input_size, use_attention=False)
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, input_size)
    mask = torch.ones(batch_size, seq_len, input_size)
    
    # Add some missing values
    mask[:, :, :5] = torch.rand(batch_size, seq_len, 5) > 0.1
    
    # Forward pass
    output = model(x, mask)
    print(f"Model output shape: {output.shape}")
    print(f"Number of parameters: {count_lstm_parameters(model):,}")
    
    # Test attention weights
    attention_weights = model.get_attention_weights(x, mask)
    print(f"Attention weights shape: {attention_weights.shape}")
    
    print("\nTesting Attention LSTM Model:")
    # Test attention LSTM model
    att_model = create_lstm_model(input_size=input_size, use_attention=True)
    
    # Forward pass
    output, att_weights = att_model(x, mask)
    print(f"Model output shape: {output.shape}")
    print(f"Attention weights shape: {att_weights.shape}")
    print(f"Number of parameters: {count_lstm_parameters(att_model):,}")

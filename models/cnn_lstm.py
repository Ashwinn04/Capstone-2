"""
CNN-LSTM hybrid model implementation for sepsis prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM hybrid model for sepsis prediction
    Combines 1D CNN for local temporal patterns with LSTM for sequence modeling
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3,
                 output_size: int = 1, 
                 cnn_filters: int = 64, kernel_sizes: list = [3, 5, 7],
                 pooling_size: int = 2):
        super(CNNLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.cnn_filters = cnn_filters
        self.kernel_sizes = kernel_sizes
        self.pooling_size = pooling_size
        
        # CNN layers for local temporal feature extraction
        self.cnn_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        
        for i, kernel_size in enumerate(kernel_sizes):
            # Input channels: input_size for first layer, cnn_filters for subsequent layers
            in_channels = input_size if i == 0 else cnn_filters
            
            # CNN layer
            cnn_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=cnn_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2  # Same padding
            )
            self.cnn_layers.append(cnn_layer)
            
            # Pooling layer
            pool_layer = nn.MaxPool1d(kernel_size=pooling_size)
            self.pooling_layers.append(pool_layer)
        
        # Calculate CNN output size (simplified - use only first layer)
        cnn_output_size = cnn_filters
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Calculate LSTM output size (bidirectional)
        lstm_output_size = hidden_size * 2
        
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
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input sequences [batch_size, seq_len, input_size]
            mask: Missing value masks [batch_size, seq_len, input_size]
            
        Returns:
            Output logits [batch_size, output_size]
        """
        batch_size, seq_len, input_size = x.shape
        
        # Handle missing values
        x_masked = x * mask.float()
        
        # Reshape for CNN: [batch_size, input_size, seq_len]
        x_cnn = x_masked.transpose(1, 2)
        
        # Apply CNN layers (simplified - use only first layer)
        cnn_layer = self.cnn_layers[0]
        pool_layer = self.pooling_layers[0]
        
        # CNN forward pass
        cnn_out = F.relu(cnn_layer(x_cnn))
        
        # Apply pooling
        pooled_out = pool_layer(cnn_out)
        
        # Transpose back: [batch_size, seq_len, filters]
        cnn_concat = pooled_out.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(cnn_concat)
        
        # Use the last output
        final_output = lstm_out[:, -1, :]  # [batch_size, lstm_output_size]
        
        # Apply dropout
        final_output = self.dropout_layer(final_output)
        
        # Output layer
        output = self.output_layer(final_output)
        
        return output
    
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
        
        # CNN processing
        x_cnn = x_masked.transpose(1, 2)
        cnn_outputs = []
        
        for cnn_layer, pool_layer in zip(self.cnn_layers, self.pooling_layers):
            cnn_out = F.relu(cnn_layer(x_cnn))
            pooled_out = pool_layer(cnn_out)
            pooled_out = pooled_out.transpose(1, 2)
            cnn_outputs.append(pooled_out)
        
        cnn_concat = torch.cat(cnn_outputs, dim=2)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(cnn_concat)
        
        # Calculate attention weights using final hidden state
        final_hidden = lstm_out[:, -1, :]
        
        # Compute attention scores
        attention_scores = torch.sum(lstm_out * final_hidden.unsqueeze(1), dim=2)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        return attention_weights
    
    def get_cnn_features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Get CNN features for analysis
        
        Args:
            x: Input sequences
            mask: Missing value masks
            
        Returns:
            CNN features [batch_size, seq_len, cnn_output_size]
        """
        batch_size, seq_len, input_size = x.shape
        
        # Handle missing values
        x_masked = x * mask.float()
        
        # CNN processing
        x_cnn = x_masked.transpose(1, 2)
        cnn_outputs = []
        
        for cnn_layer, pool_layer in zip(self.cnn_layers, self.pooling_layers):
            cnn_out = F.relu(cnn_layer(x_cnn))
            pooled_out = pool_layer(cnn_out)
            pooled_out = pooled_out.transpose(1, 2)
            cnn_outputs.append(pooled_out)
        
        cnn_concat = torch.cat(cnn_outputs, dim=2)
        return cnn_concat


class AttentionCNNLSTMModel(nn.Module):
    """
    CNN-LSTM model with attention mechanism
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3,
                 output_size: int = 1, 
                 cnn_filters: int = 64, kernel_sizes: list = [3, 5, 7],
                 pooling_size: int = 2):
        super(AttentionCNNLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.cnn_filters = cnn_filters
        self.kernel_sizes = kernel_sizes
        self.pooling_size = pooling_size
        
        # CNN layers
        self.cnn_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        
        for i, kernel_size in enumerate(kernel_sizes):
            in_channels = input_size if i == 0 else cnn_filters
            
            cnn_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=cnn_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
            self.cnn_layers.append(cnn_layer)
            
            pool_layer = nn.MaxPool1d(kernel_size=pooling_size)
            self.pooling_layers.append(pool_layer)
        
        cnn_output_size = cnn_filters * len(kernel_sizes)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        lstm_output_size = hidden_size * 2
        
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
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention
        
        Args:
            x: Input sequences [batch_size, seq_len, input_size]
            mask: Missing value masks [batch_size, seq_len, input_size]
            
        Returns:
            Tuple of (output_logits, attention_weights)
        """
        batch_size, seq_len, input_size = x.shape
        
        # Handle missing values
        x_masked = x * mask.float()
        
        # CNN processing
        x_cnn = x_masked.transpose(1, 2)
        cnn_outputs = []
        
        for cnn_layer, pool_layer in zip(self.cnn_layers, self.pooling_layers):
            cnn_out = F.relu(cnn_layer(x_cnn))
            pooled_out = pool_layer(cnn_out)
            pooled_out = pooled_out.transpose(1, 2)
            cnn_outputs.append(pooled_out)
        
        cnn_concat = torch.cat(cnn_outputs, dim=2)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(cnn_concat)
        
        # Calculate attention weights
        attention_scores = self.attention(lstm_out)
        attention_scores = attention_scores.squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention to get context vector
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        
        # Apply dropout
        context_vector = self.dropout_layer(context_vector)
        
        # Output layer
        output = self.output_layer(context_vector)
        
        return output, attention_weights


def create_cnn_lstm_model(input_size: int, hidden_size: int = 128, 
                         num_layers: int = 2, dropout: float = 0.3,
                         cnn_filters: int = 64, kernel_sizes: list = [3, 5, 7],
                         use_attention: bool = False) -> nn.Module:
    """
    Create CNN-LSTM model with specified parameters
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden state size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        cnn_filters: Number of CNN filters
        kernel_sizes: List of kernel sizes for CNN layers
        use_attention: Whether to use attention mechanism
        
    Returns:
        CNN-LSTM model instance
    """
    if use_attention:
        model = AttentionCNNLSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cnn_filters=cnn_filters,
            kernel_sizes=kernel_sizes
        )
    else:
        model = CNNLSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cnn_filters=cnn_filters,
            kernel_sizes=kernel_sizes
        )
    
    return model


def count_cnn_lstm_parameters(model: nn.Module) -> int:
    """Count trainable parameters in CNN-LSTM model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage and testing
if __name__ == "__main__":
    # Test basic CNN-LSTM model
    batch_size = 32
    seq_len = 24
    input_size = 20
    
    print("Testing Basic CNN-LSTM Model:")
    model = create_cnn_lstm_model(input_size=input_size, use_attention=False)
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, input_size)
    mask = torch.ones(batch_size, seq_len, input_size)
    
    # Add some missing values
    mask[:, :, :5] = torch.rand(batch_size, seq_len, 5) > 0.1
    
    # Forward pass
    output = model(x, mask)
    print(f"Model output shape: {output.shape}")
    print(f"Number of parameters: {count_cnn_lstm_parameters(model):,}")
    
    # Test attention weights
    attention_weights = model.get_attention_weights(x, mask)
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Test CNN features
    cnn_features = model.get_cnn_features(x, mask)
    print(f"CNN features shape: {cnn_features.shape}")
    
    print("\nTesting Attention CNN-LSTM Model:")
    # Test attention CNN-LSTM model
    att_model = create_cnn_lstm_model(input_size=input_size, use_attention=True)
    
    # Forward pass
    output, att_weights = att_model(x, mask)
    print(f"Model output shape: {output.shape}")
    print(f"Attention weights shape: {att_weights.shape}")
    print(f"Number of parameters: {count_cnn_lstm_parameters(att_model):,}")

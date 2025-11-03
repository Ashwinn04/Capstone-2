"""
CNN-LSTM Hybrid Model: Combines local pattern extraction with temporal modeling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    """
    CNN-LSTM Hybrid Model: Uses CNN for local pattern extraction, then LSTM for temporal modeling
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1,
                 dropout: float = 0.2, cnn_filters: int = 64, kernel_size: int = 3):
        """
        Args:
            input_size: Number of input features
            hidden_size: Hidden dimension size for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            cnn_filters: Number of CNN filters
            kernel_size: CNN kernel size
        """
        super(CNNLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 1D CNN for local pattern extraction
        # Input: [batch, features, seq_len] -> [batch, filters, seq_len]
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=cnn_filters,
                                kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate LSTM input size after CNN
        # After pooling: seq_len // 2
        self.lstm = nn.LSTM(cnn_filters, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
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
        
        # Reshape for CNN: [batch, seq_len, features] -> [batch, features, seq_len]
        cnn_input = masked_features.transpose(1, 2)
        
        # CNN processing
        cnn_out = self.conv1d(cnn_input)
        cnn_out = self.relu(cnn_out)
        cnn_out = self.maxpool(cnn_out)
        
        # Reshape back for LSTM: [batch, filters, seq_len'] -> [batch, seq_len', filters]
        lstm_input = cnn_out.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(lstm_input)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Output layer
        output = self.fc(self.dropout(last_hidden))
        
        return output

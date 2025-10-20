"""
GRU-D (GRU with Decay) model implementation for handling missing data in ICU time-series
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class GRUDCell(nn.Module):
    """
    GRU-D cell with decay mechanism for missing values
    """
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0):
        super(GRUDCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        # Decay parameters
        self.W_decay = nn.Parameter(torch.randn(input_size))
        self.b_decay = nn.Parameter(torch.zeros(input_size))
        
        # GRU gates
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size)
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor, 
                h_prev: torch.Tensor, x_prev: torch.Tensor,
                delta_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of GRU-D cell
        
        Args:
            x: Current input [batch_size, input_size]
            mask: Missing value mask [batch_size, input_size]
            h_prev: Previous hidden state [batch_size, hidden_size]
            x_prev: Previous input [batch_size, input_size]
            delta_t: Time delta since last observation [batch_size, input_size]
            
        Returns:
            Tuple of (new_hidden_state, new_input)
        """
        # Calculate decay factors
        decay = torch.exp(-torch.max(torch.zeros_like(delta_t), 
                                   self.W_decay * delta_t + self.b_decay))
        
        # Decay previous input
        x_decayed = decay * x_prev
        
        # Use current input where available, decayed previous input otherwise
        x_imputed = mask * x + (1 - mask) * x_decayed
        
        # Update gate
        z = torch.sigmoid(self.W_z(x_imputed) + self.U_z(h_prev) + self.b_z)
        
        # Reset gate
        r = torch.sigmoid(self.W_r(x_imputed) + self.U_r(h_prev) + self.b_r)
        
        # Candidate hidden state
        h_tilde = torch.tanh(self.W_h(x_imputed) + self.U_h(r * h_prev) + self.b_h)
        
        # New hidden state
        h_new = (1 - z) * h_prev + z * h_tilde
        
        # Apply dropout
        h_new = self.dropout_layer(h_new)
        
        return h_new, x_imputed


class GRUDModel(nn.Module):
    """
    Simplified GRU-D model for sepsis prediction with missing data handling
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3,
                 output_size: int = 1, use_attention: bool = False):
        super(GRUDModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Input projection; augment with mask and delta_t when available
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.aug_projection = nn.Linear(input_size * 3, input_size)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Optional attention over time
        if self.use_attention:
            self.attention = nn.Linear(hidden_size, 1)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
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
            delta_t: Time since last observation per feature [batch_size, seq_len, input_size]
            
        Returns:
            Output logits [batch_size, output_size]
        """
        batch_size, seq_len, input_size = x.shape
        
        # Handle missing values with simple imputation
        x_masked = x * mask.float()
        # Fuse missingness and time info into representation
        if delta_t is not None:
            x_aug = torch.cat([x_masked, mask.float(), delta_t], dim=2)
            x_base = self.aug_projection(x_aug)
        else:
            x_base = x_masked
        
        # Project input
        x_proj = self.input_projection(x_base)
        
        # GRU forward pass
        gru_out, _ = self.gru(x_proj)
        
        if self.use_attention:
            # Attention weights over time steps
            attn_scores = self.attention(gru_out).squeeze(-1)  # [batch, seq_len]
            attn_weights = F.softmax(attn_scores, dim=1)
            context = torch.sum(gru_out * attn_weights.unsqueeze(-1), dim=1)  # [batch, hidden]
            output = self.output_layer(context)
        else:
            # Use the last output
            final_output = gru_out[:, -1, :]
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
        
        # Initialize hidden states
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
             for _ in range(self.num_layers)]
        
        # Store hidden states for each time step
        hidden_states = []
        
        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :]
            mask_t = mask[:, t, :]
            delta_t = torch.ones_like(x_t)
            
            # Process through layers
            for layer_idx, grud_cell in enumerate(self.grud_cells):
                h[layer_idx], _ = grud_cell(x_t, mask_t, h[layer_idx], 
                                          torch.zeros_like(x_t), delta_t)
                x_t = h[layer_idx]
            
            hidden_states.append(h[-1])
        
        # Stack hidden states
        hidden_states = torch.stack(hidden_states, dim=1)  # [batch_size, seq_len, hidden_size]
        
        # Calculate attention weights using final hidden state
        final_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        
        # Compute attention scores
        attention_scores = torch.sum(hidden_states * final_hidden.unsqueeze(1), dim=2)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        return attention_weights


def create_grud_model(input_size: int, hidden_size: int = 128, 
                     num_layers: int = 2, dropout: float = 0.3,
                     use_attention: bool = False) -> GRUDModel:
    """
    Create GRU-D model with specified parameters
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden state size
        num_layers: Number of GRU-D layers
        dropout: Dropout rate
        
    Returns:
        GRUDModel instance
    """
    model = GRUDModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        use_attention=use_attention
    )
    
    return model


def count_grud_parameters(model: GRUDModel) -> int:
    """Count trainable parameters in GRU-D model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage and testing
if __name__ == "__main__":
    # Test model
    batch_size = 32
    seq_len = 24
    input_size = 20
    
    model = create_grud_model(input_size=input_size)
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, input_size)
    mask = torch.ones(batch_size, seq_len, input_size)
    
    # Add some missing values
    mask[:, :, :5] = torch.rand(batch_size, seq_len, 5) > 0.1
    
    # Forward pass
    output = model(x, mask)
    print(f"Model output shape: {output.shape}")
    print(f"Number of parameters: {count_grud_parameters(model):,}")
    
    # Test attention weights
    attention_weights = model.get_attention_weights(x, mask)
    print(f"Attention weights shape: {attention_weights.shape}")

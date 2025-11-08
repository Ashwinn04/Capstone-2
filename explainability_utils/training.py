"""
Training utilities for deep learning models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """
        Check if training should stop
        
        Args:
            val_score: Current validation score
            model: PyTorch model
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
            
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save model weights"""
        self.best_weights = model.state_dict().copy()


class ModelTrainer:
    """Training class for deep learning models"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_scores = []
        self.val_scores = []
        
    def train_epoch(self, train_loader, criterion, optimizer) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        # For calculating AUROC during training
        all_preds = []
        all_targets = []
        
        for batch_idx, (features, masks, targets) in enumerate(train_loader):
            features = features.to(self.device)
            masks = masks.to(self.device)
            targets = targets.to(self.device)
            
            use_cuda_amp = (self.device == 'cuda')
            with torch.autocast(device_type=('cuda' if use_cuda_amp else 'cpu'), dtype=torch.float16, enabled=use_cuda_amp):
                # Forward pass
                outputs = self.model(features, masks, delta_t)
                
                # Squeeze outputs if necessary
                if outputs.dim() > 1 and targets.dim() == 1:
                    outputs = outputs.squeeze(-1)

                # Calculate loss
                loss = criterion(outputs, targets.float())
            
            # Backward pass
            optimizer.zero_grad()
            if use_cuda_amp:
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_value)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_value)
                optimizer.step()
            
            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)
            
            # Store predictions for AUROC calculation
            with torch.no_grad():
                preds = torch.sigmoid(outputs).cpu().numpy()
                targets_cpu = targets.cpu().numpy()
                all_preds.extend(preds.flatten())
                all_targets.extend(targets_cpu.flatten())
        
        avg_loss = total_loss / total_samples
        avg_score = self._calculate_score(all_targets, all_preds)
        
        return avg_loss, avg_score
    
    def validate_epoch(self, val_loader, criterion) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for features, masks, targets in val_loader:
                features = features.to(self.device)
                masks = masks.to(self.device)
                targets = targets.to(self.device)
                
                use_cuda_amp = (self.device == 'cuda')
                with torch.autocast(device_type=('cuda' if use_cuda_amp else 'cpu'), dtype=torch.float16, enabled=use_cuda_amp):
                    outputs = self.model(features, masks, delta_t)

                    if outputs.dim() > 1 and targets.dim() == 1:
                        outputs = outputs.squeeze(-1)
                        
                    loss = criterion(outputs, targets.float())
                
                total_loss += loss.item() * features.size(0)
                total_samples += features.size(0)
                
                preds = torch.sigmoid(outputs).cpu().numpy()
                targets_cpu = targets.cpu().numpy()
                all_preds.extend(preds.flatten())
                all_targets.extend(targets_cpu.flatten())
        
        avg_loss = total_loss / total_samples
        avg_score = self._calculate_score(all_targets, all_preds)
        
        return avg_loss, avg_score
    
    def _calculate_score(self, targets: List, preds: List) -> float:
        """Calculate AUROC score"""
        from sklearn.metrics import roc_auc_score
        try:
            if len(np.unique(targets)) > 1:
                return roc_auc_score(targets, preds)
            else:
                return 0.5
        except:
            return 0.5
    
    def train(self, train_loader, val_loader, 
              epochs: int = 100, learning_rate: float = 0.001,
              weight_decay: float = 1e-5, patience: int = 10,
              class_weights: Optional[List[float]] = None,
              save_path: Optional[str] = None) -> Dict:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
            class_weights: Class weights for imbalanced data
            save_path: Path to save model weights
            
        Returns:
            Training history dictionary
        """
        # Setup optimizer
        if optimizer_config['name'] == 'AdamW':
            optimizer = optim.AdamW(self.model.parameters(), lr=optimizer_config['lr'],
                                    weight_decay=optimizer_config['weight_decay'])
        else: # Default to Adam
            optimizer = optim.Adam(self.model.parameters(), lr=optimizer_config['lr'],
                                   weight_decay=optimizer_config.get('weight_decay', 1e-5))

        # Setup scheduler
        if scheduler_config['name'] == 'CosineAnnealing':
            T_max = epochs - scheduler_config['warmup_epochs']
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-5)
        else: # Default to ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        # Warmup scheduler
        warmup_epochs = scheduler_config.get('warmup_epochs', 0)
        
        # Setup loss function
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1]/class_weights[0])
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # Early stopping
        early_stopping = EarlyStopping(patience=patience)
        
        # Training loop
        best_val_score = 0.0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_score = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_score = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_score)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_scores.append(train_score)
            self.val_scores.append(val_score)
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train AUROC: {train_score:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val AUROC: {val_score:.4f}')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Early stopping
            if early_stopping(val_score, self.model):
                print(f'Early stopping at epoch {epoch+1}')
                break
            
            # Save best model
            if val_score > best_val_score:
                best_val_score = val_score
                if save_path:
                    self.save_model(save_path)
        
        # Return training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_scores': self.train_scores,
            'val_scores': self.val_scores,
            'best_val_score': best_val_score,
            'epochs_trained': len(self.train_losses)
        }
        
        return history
    
    def save_model(self, path: str):
        """Save model weights"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")


def calculate_class_weights(train_loader) -> List[float]:
    """Calculate class weights for imbalanced data"""
    total_samples = 0
    positive_samples = 0
    
    for _, _, targets in train_loader:
        total_samples += len(targets)
        positive_samples += targets.sum().item()
    
    negative_samples = total_samples - positive_samples
    
    if positive_samples == 0 or negative_samples == 0:
        return [1.0, 1.0]
    
    # Return weights as [negative_weight, positive_weight]
    return [total_samples / (2 * negative_samples), 
            total_samples / (2 * positive_samples)]


def save_training_results(results: Dict, model_name: str, 
                         output_dir: str = "outputs/results"):
    """Save training results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_training_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, (np.int64, np.int32)):
            json_results[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            json_results[key] = float(value)
        else:
            json_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Training results saved to {filepath}")
    return filepath


def load_training_results(filepath: str) -> Dict:
    """Load training results from JSON file"""
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    # Convert lists back to numpy arrays where appropriate
    for key, value in results.items():
        if isinstance(value, list) and key in ['train_losses', 'val_losses', 
                                              'train_scores', 'val_scores']:
            results[key] = np.array(value)
    
    return results


def get_device() -> str:
    """Get available device (CPU or GPU)"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        print("Using CPU")
    
    return device


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seeds set to {seed}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

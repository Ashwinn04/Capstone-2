"""
Main training script for all deep learning models
Trains: GRU-D, LSTM, CNN-LSTM, Transformer
"""
import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import GRUD, LSTM, CNNLSTM, Transformer
from explainability_utils.data_loader import (
    load_preprocessed_data, 
    get_train_val_test_split,
    create_data_loaders
)
from explainability_utils.training import (
    ModelTrainer,
    get_device,
    set_random_seeds,
    save_training_results,
    calculate_class_weights
)
from explainability_utils.visualization import plot_training_history
import warnings
warnings.filterwarnings('ignore')


# Model configurations
MODEL_CONFIGS = {
    'grud': {
        'class': GRUD,
        'kwargs': {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2},
        'save_name': 'grud_demo_model.pt'
    },
    'lstm': {
        'class': LSTM,
        'kwargs': {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2},
        'save_name': 'lstm_demo_model.pt'
    },
    'cnn_lstm': {
        'class': CNNLSTM,
        'kwargs': {'hidden_size': 64, 'num_layers': 1, 'dropout': 0.2,
                   'cnn_filters': 64, 'kernel_size': 3},
        'save_name': 'cnn_lstm_demo_model.pt'
    },
    'transformer': {
        'class': Transformer,
        'kwargs': {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.2},
        'save_name': 'transformer_demo_model.pt'
    }
}


def train_model(model_name: str, train_loader, val_loader, 
                input_size: int, device: str, config: dict,
                epochs: int = 50, learning_rate: float = 0.001,
                batch_size: int = 32, use_class_weights: bool = True):
    """
    Train a single model
    
    Args:
        model_name: Name of the model ('grud', 'lstm', 'cnn_lstm', 'transformer')
        train_loader: Training data loader
        val_loader: Validation data loader
        input_size: Number of input features
        device: Device to train on ('cpu' or 'cuda')
        config: Model configuration dictionary
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        use_class_weights: Whether to use class weights for imbalanced data
        
    Returns:
        Trained model and training history
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} Model")
    print(f"{'='*60}")
    
    # Create model
    model_class = config['class']
    model_kwargs = config['kwargs'].copy()
    model_kwargs['input_size'] = input_size
    
    model = model_class(**model_kwargs)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    # Create trainer
    trainer = ModelTrainer(model, device=device)
    
    # Calculate class weights if needed
    class_weights = None
    if use_class_weights:
        try:
            class_weights = calculate_class_weights(train_loader)
            print(f"Class weights: {class_weights}")
        except Exception as e:
            print(f"Warning: Could not calculate class weights: {e}")
            class_weights = None
    
    # Prepare save path
    save_path = os.path.join('outputs', 'models', config['save_name'])
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=1e-5,
        patience=10,
        class_weights=class_weights,
        save_path=None  # We'll save manually after training
    )
    
    # Save full model (not just state_dict) for integration compatibility
    model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)
    print(f"Full model saved to {save_path}")
    
    print(f"\nTraining completed!")
    print(f"Best validation AUROC: {history['best_val_score']:.4f}")
    print(f"Epochs trained: {history['epochs_trained']}")
    
    return trainer.model, history


def main():
    parser = argparse.ArgumentParser(description='Train deep learning models for sepsis prediction')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to preprocessed data CSV file')
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['grud', 'lstm', 'cnn_lstm', 'transformer', 'all'],
                       default=['all'],
                       help='Models to train (default: all)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--sequence_length', type=int, default=24,
                       help='Sequence length (default: 24)')
    parser.add_argument('--prediction_horizon', type=int, default=4,
                       help='Prediction horizon in hours (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--no_class_weights', action='store_true',
                       help='Disable class weight balancing')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    set_random_seeds(args.seed)
    
    # Get device
    device = get_device()
    
    # Load data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    
    if args.data_path and os.path.exists(args.data_path):
        print(f"Loading data from {args.data_path}")
        data, feature_cols = load_preprocessed_data(args.data_path)
    else:
        print("No data file provided or file not found. Using sample data.")
        from explainability_utils.data_loader import create_sample_data
        data, feature_cols = create_sample_data(n_patients=200, n_hours=48, n_features=20)
    
    input_size = len(feature_cols)
    print(f"Input size: {input_size} features")
    
    # Split data
    print("\nSplitting data...")
    train_data, val_data, test_data = get_train_val_test_split(
        data, train_ratio=0.7, val_ratio=0.15
    )
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        features=feature_cols
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Determine which models to train
    models_to_train = []
    if 'all' in args.models:
        models_to_train = ['grud', 'lstm', 'cnn_lstm', 'transformer']
    else:
        models_to_train = args.models
    
    # Create output directory
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    
    # Train models
    results = {}
    
    for model_name in models_to_train:
        try:
            config = MODEL_CONFIGS[model_name]
            
            # Train model
            model, history = train_model(
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                input_size=input_size,
                device=device,
                config=config,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                use_class_weights=not args.no_class_weights
            )
            
            # Save training results
            results[model_name] = history
            save_training_results(history, model_name, output_dir='outputs/results')
            
            # Plot training history
            plot_path = os.path.join('outputs', 'figures', f'{model_name}_training_history.png')
            plot_training_history(history, model_name=model_name.upper(), save_path=plot_path)
            
        except Exception as e:
            print(f"\n❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    for model_name, history in results.items():
        print(f"{model_name.upper()}: Best Val AUROC = {history['best_val_score']:.4f}")
    
    print("\n✅ Training completed!")
    print(f"Models saved to: outputs/models/")
    print(f"Results saved to: outputs/results/")
    print(f"Figures saved to: outputs/figures/")


if __name__ == '__main__':
    main()

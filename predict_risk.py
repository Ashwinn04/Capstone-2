"""
CLI to produce calibrated risk predictions and risk flags for a trained model.

Usage example:
    python predict_risk.py \
        --model_name lstm \
        --seed 42 \
        --data_path Dataset.csv \
        --sequence_length 48 \
        --prediction_horizon 6 \
        --threshold r85
"""
import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch

from models import GRUD, LSTM, CNNLSTM, Transformer
from explainability_utils.data_loader import (
    load_preprocessed_data,
    create_patient_splits,
    get_train_val_test_split,
    fit_and_save_imputer_scaler,
    apply_imputation_and_scaling,
    create_data_loaders,
)
from explainability_utils.inference import predict_with_model
from explainability_utils.training import get_device


MODEL_CONFIGS_FOR_INFERENCE = {
    'grud': {'class': GRUD},
    'lstm': {'class': LSTM},
    'cnn_lstm': {'class': CNNLSTM},
    'transformer': {'class': Transformer},
}


def build_model(model_name: str, input_size: int, models_dir: str, seed: int) -> torch.nn.Module:
    # Attempt to load the exact model config that was saved.
    # Fallback to reasonable defaults if config file is missing.
    
    # Try both improved and original naming conventions
    base_improved = Path(models_dir) / f"{model_name}_improved_seed{seed}"
    base_original = Path(models_dir) / f"{model_name}_seed{seed}"
    
    if os.path.exists(str(base_improved) + "_config.json"):
        base = base_improved
    else:
        base = base_original
        
    config_path = str(base) + "_config.json"
    kwargs = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
            kwargs = cfg.get('model_kwargs')
    if kwargs is None:
        # Fallback defaults (match training defaults)
        if model_name in ('grud', 'lstm'):
            kwargs = {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3}
        elif model_name == 'cnn_lstm':
            kwargs = {'hidden_size': 128, 'num_layers': 1, 'dropout': 0.3, 'cnn_filters': [64, 128], 'kernel_size': 3}
        elif model_name == 'transformer':
            kwargs = {'d_model': 128, 'nhead': 8, 'num_layers': 4, 'dropout': 0.1}
    kwargs['input_size'] = input_size

    model_class = MODEL_CONFIGS_FOR_INFERENCE[model_name]['class']
    model = model_class(**kwargs)

    # Load weights
    weights_path = str(base) + ".pt"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    device = get_device()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="Predict calibrated risk for a trained model")
    parser.add_argument('--model_name', type=str, required=True, choices=['grud', 'lstm', 'cnn_lstm', 'transformer'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, default='fully_cleaned_sepsis_dataset.csv')
    parser.add_argument('--sequence_length', type=int, default=48)
    parser.add_argument('--prediction_horizon', type=int, default=6)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--models_dir', type=str, default='outputs/models')
    parser.add_argument('--threshold', type=str, default='r85', choices=['r85', 'custom'])
    parser.add_argument('--threshold_value', type=float, default=None, help='Used when --threshold custom')

    args = parser.parse_args()

    # Load data and preprocess
    data, feature_cols = load_preprocessed_data(args.data_path)
    patient_splits = create_patient_splits(data, seed=args.seed)
    train_data, val_data, test_data = get_train_val_test_split(data, patient_splits)

    # Fit imputer/scaler on train and transform test
    imputer, scaler = fit_and_save_imputer_scaler(train_data, feature_cols)
    test_data = apply_imputation_and_scaling(test_data, feature_cols, imputer, scaler)

    # Data loaders
    _, _, test_loader = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        features=feature_cols,
        batch_size=128,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        step_size=args.step_size,
    )

    # Build model
    input_size = len(feature_cols)
    model = build_model(args.model_name, input_size, args.models_dir, args.seed)

    # Predict calibrated risk on test set and log
    all_probs = []
    all_flags = []
    all_targets = []
    used_thr = None

    # Open the file once and write line by line
    out_dir = Path('outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'prediction_log.jsonl'
    
    with out_path.open('w') as f:
        for features, masks, delta_t, targets in test_loader:
            probs, flags, used_thr = predict_with_model(
                model=model,
                features=features,
                masks=masks,
                delta_t=delta_t,
                model_name=args.model_name,
                seed=args.seed,
                threshold=(args.threshold_value if args.threshold == 'custom' else None),
                models_dir=args.models_dir,
            )
            
            # Iterate through batch results
            batch_targets = targets.numpy().astype(int).tolist()
            for i in range(len(probs)):
                p = probs[i]
                yhat = flags[i]
                y = batch_targets[i]
                if isinstance(y, list):
                    y = y[0] # Handle nested list case
                f.write(json.dumps({'prob': float(p), 'risk_flag': int(yhat), 'target': int(y)}) + '\n')

    print(f"Saved calibrated predictions to {out_path}")
    if used_thr is not None:
        print(f"Used threshold: {used_thr}")


if __name__ == '__main__':
    main()



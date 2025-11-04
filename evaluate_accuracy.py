import os
import torch
import numpy as np
import pandas as pd
import json
import joblib
import argparse
from typing import Tuple
from torch.utils.data import DataLoader

from explainability_utils.data_loader import (
    load_preprocessed_data,
    create_patient_splits,
    get_train_val_test_split,
    apply_imputation_and_scaling,
    ICUDataset,
)
from explainability_utils.metrics import calculate_all_metrics
from explainability_utils.calibration import ModelCalibrator


def evaluate_model(model, data_loader, device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for features, masks, delta_t, targets in data_loader:
            features = features.to(device)
            masks = masks.to(device)
            delta_t = delta_t.to(device)
            
            logits = model(features, masks, delta_t)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_targets.extend(targets.cpu().numpy().flatten())
            
    # Ensure float32 for MPS compatibility
    return np.array(all_probs, dtype=np.float32), np.array(all_targets, dtype=np.float32)


def get_model_class(model_name: str):
    name = model_name.lower()
    if name == 'grud':
        from models.grud import GRUD
        return GRUD
    if name == 'lstm':
        from models.lstm import LSTM
        return LSTM
    if name == 'cnn_lstm':
        from models.cnn_lstm import CNNLSTM
        return CNNLSTM
    if name == 'transformer':
        from models.transformer import Transformer
        return Transformer
    raise ValueError(f"Unknown model name: {model_name}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models on the test set.')
    parser.add_argument('--data_path', type=str, default='Dataset.csv', help='Path to preprocessed data CSV file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data splitting')
    args = parser.parse_args()

    # --- Load Data and Preprocessing ---
    data, feature_cols = load_preprocessed_data(args.data_path)
    patient_splits = create_patient_splits(data, seed=args.seed)
    train_data, val_data, test_data = get_train_val_test_split(data, patient_splits)

    imputer = joblib.load('outputs/cache/imputer.joblib')
    scaler = joblib.load('outputs/cache/scaler.joblib')

    test_data = apply_imputation_and_scaling(test_data, feature_cols, imputer, scaler)

    # --- Create Test Loader (directly) ---
    try:
        with open('outputs/models/lstm_seed42_config.json', 'r') as f:
            config = json.load(f)
        seq_len = config.get('sequence_length', 48)
        pred_h = config.get('prediction_horizon', 6)
        step = config.get('step_size', 1)
    except FileNotFoundError:
        print("Could not find a model config, using default loader params.")
        seq_len, pred_h, step = 48, 6, 1

    test_dataset = ICUDataset(test_data, features=feature_cols, sequence_length=seq_len, prediction_horizon=pred_h, step_size=step)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    # --- Evaluate Models ---
    model_names = ['grud', 'lstm', 'cnn_lstm', 'transformer']
    num_seeds = 3 
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    results_df = pd.DataFrame()

    for model_name in model_names:
        print(f"\n--- Evaluating {model_name.upper()} ---")
        
        # Load models, configs, thresholds, and calibrators for all seeds
        models, thresholds, calibrators = [], [], []
        for i in range(num_seeds):
            seed = args.seed + i
            base_path = f'outputs/models/{model_name}_seed{seed}'
            
            model_path = f'{base_path}.pt'
            cfg_path = f'{base_path}_config.json'
            thr_path = f'{base_path}_threshold.json'
            cal_path = f'{base_path}_calibrator.joblib'

            if not os.path.exists(model_path) or not os.path.exists(cfg_path):
                print(f"Warning: Artifacts for seed {seed} not found. Skipping.")
                continue

            # Load model
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
            ModelClass = get_model_class(cfg.get('model_name', model_name))
            model = ModelClass(**cfg['model_kwargs'])
            model.load_state_dict(torch.load(model_path, map_location=device))
            models.append(model)
            
            # Load threshold and calibrator (optional)
            if os.path.exists(thr_path):
                with open(thr_path, 'r') as f:
                    thresholds.append(json.load(f).get('threshold_r85', 0.5))
            if os.path.exists(cal_path):
                calibrators.append(ModelCalibrator.load(cal_path, method='platt'))

        if not models:
            print(f"No models found for {model_name}.")
            continue
            
        # Get predictions from each model in the ensemble
        all_probs_list = []
        for model in models:
            probs, targets = evaluate_model(model, test_loader, device=device)
            all_probs_list.append(probs)
        
        # --- Ensemble and Calibrate Predictions ---
        # 1. Average probabilities
        ensemble_probs = np.mean(all_probs_list, axis=0)

        # 2. Average after calibration (if available)
        if calibrators and len(calibrators) == len(all_probs_list):
            calibrated_probs_list = [cal.predict(p) for cal, p in zip(calibrators, all_probs_list)]
            ensemble_calibrated_probs = np.mean(calibrated_probs_list, axis=0)
        else:
            ensemble_calibrated_probs = ensemble_probs

        # --- Calculate Metrics ---
        metrics_uncalibrated = calculate_all_metrics(targets, ensemble_probs)
        metrics_calibrated = calculate_all_metrics(targets, ensemble_calibrated_probs)

        print("\n** Uncalibrated Ensemble Metrics **")
        for key, val in metrics_uncalibrated.items():
            print(f"  {key}: {val:.4f}")
        
        print("\n** Calibrated Ensemble Metrics **")
        for key, val in metrics_calibrated.items():
            print(f"  {key}: {val:.4f}")

        # Store results
        metrics_calibrated['model'] = model_name
        results_df = pd.concat([results_df, pd.DataFrame([metrics_calibrated])])

    # Save final results
    if not results_df.empty:
        results_df.set_index('model', inplace=True)
        test_csv_path = 'outputs/results/test_set_model_comparison.csv'
        results_df.to_csv(test_csv_path)
        print(f"\nFinal test set results saved to {test_csv_path}")
        print(results_df)


if __name__ == '__main__':
    main()



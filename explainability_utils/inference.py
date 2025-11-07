"""
Inference utilities for calibrated risk prediction.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from .calibration import ModelCalibrator


def load_calibrator_and_threshold(model_name: str, seed: int, models_dir: str = "outputs/models") -> Tuple[ModelCalibrator, float]:
    """Load saved Platt calibrator and threshold for Recall@0.85.

    Expects files saved by the training pipeline: `{model}_{suffix}`.
    """
    # Try both improved and original naming conventions
    base_improved = Path(models_dir) / f"{model_name}_improved_seed{seed}"
    base_original = Path(models_dir) / f"{model_name}_seed{seed}"

    if os.path.exists(str(base_improved) + "_calibrator.joblib"):
        base = base_improved
    else:
        base = base_original

    # Try common filenames for calibrator and threshold
    candidates_cal = [
        str(base) + "_calibrator.joblib",
        str(base) + "_calibrator_platt.joblib",
    ]
    candidates_thr = [
        str(base) + "_threshold.json",
        str(base) + "_thresholds.json",
    ]

    calibrator_obj = None
    for p in candidates_cal:
        if os.path.exists(p):
            import joblib
            calibrator_obj = joblib.load(p)
            break
    
    if calibrator_obj is None:
        raise FileNotFoundError(f"Calibrator not found for {model_name} seed {seed} under {models_dir}")

    # Handle both old and new calibrator formats
    if isinstance(calibrator_obj, dict):
        calibrator = ModelCalibrator(method=calibrator_obj['method'])
        calibrator.calibrator = calibrator_obj['model']
    else:
        # Legacy format: assume 'platt' and the object is the model itself
        calibrator = ModelCalibrator(method='platt')
        calibrator.calibrator = calibrator_obj
    calibrator.is_fitted = True


    threshold_r85 = None
    for p in candidates_thr:
        if os.path.exists(p):
            with open(p, "r") as f:
                data = json.load(f)
            threshold_r85 = data.get("threshold_r85") or data.get("threshold_at_recall_0.85")
            break
    if threshold_r85 is None:
        raise FileNotFoundError(f"Threshold file not found for {model_name} seed {seed} under {models_dir}")

    return calibrator, float(threshold_r85)


def calibrated_predict(
    logits: torch.Tensor,
    calibrator: ModelCalibrator,
    device: Optional[str] = None,
) -> np.ndarray:
    """Convert logits to calibrated probabilities using the provided calibrator."""
    if isinstance(logits, torch.Tensor):
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1).astype(np.float32)
    else:
        probs = np.asarray(logits, dtype=np.float32).reshape(-1)
    return calibrator.predict(probs)


def risk_flag_from_probability(
    prob: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return boolean risk flags given probabilities and a threshold."""
    return (prob >= threshold).astype(np.int8)


def predict_with_model(
    model: torch.nn.Module,
    features: torch.Tensor,
    masks: torch.Tensor,
    delta_t: torch.Tensor,
    model_name: str,
    seed: int,
    device: Optional[str] = None,
    threshold: Optional[float] = None,
    models_dir: str = "outputs/models",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run the model forward pass and return calibrated probabilities and risk flags.

    - Loads the saved Platt calibrator and default threshold (Recall@0.85) if not provided.
    - Returns (calibrated_probs, risk_flags, used_threshold).
    """
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    with torch.no_grad():
        logits = model(features.to(device), masks.to(device), delta_t.to(device))

    calibrator, threshold_r85 = load_calibrator_and_threshold(model_name, seed, models_dir=models_dir)
    used_threshold = threshold if threshold is not None else threshold_r85

    calibrated_probs = calibrated_predict(logits, calibrator, device=device)
    flags = risk_flag_from_probability(calibrated_probs, used_threshold)
    return calibrated_probs, flags, used_threshold



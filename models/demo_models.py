import numpy as np


class DummyModel:
    """Simple demo model with a predict method to satisfy integration loading."""

    def __init__(self, name: str):
        self.name = name

    def predict(self, X, masks=None):
        # Generate deterministic-ish risk based on input shape
        try:
            size_factor = float(np.array(X).size % 10) / 10.0
        except Exception:
            size_factor = 0.5

        rng = np.random.default_rng(42 + int(size_factor * 100))
        risk_score = float(np.clip(rng.normal(0.5 + (size_factor - 0.5) * 0.4, 0.15), 0.01, 0.99))
        return {
            'risk_score': risk_score,
            'risk_level': 'High' if risk_score > 0.7 else ('Medium' if risk_score > 0.3 else 'Low'),
            'confidence': float(np.clip(0.7 + (risk_score - 0.5) * 0.3, 0.5, 0.95))
        }



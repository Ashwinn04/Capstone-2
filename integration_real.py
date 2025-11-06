"""
Real Integration with Person B's Baseline Models and Person C's Deep Learning Models
This module provides seamless integration between all team members' work
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import torch
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')


class TorchModelEnsemble:
    """Utility wrapper for averaging predictions across multiple PyTorch checkpoints."""

    def __init__(self, name: str, models: List[torch.nn.Module], expected_input_size: int):
        self.name = name
        self.models = models
        self.expected_input_size = expected_input_size
        for model in self.models:
            model.eval()
            model.to('cpu')

    def predict(self, features: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Generate an averaged risk score across ensemble members."""

        if features.ndim == 2:
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        elif features.ndim == 3:
            features_tensor = torch.tensor(features, dtype=torch.float32)
        else:
            raise ValueError(f"Unexpected feature shape for {self.name}: {features.shape}")

        batch_size, seq_len, num_features = features_tensor.shape

        # Pad or truncate features to match expected input size
        if num_features != self.expected_input_size:
            if num_features < self.expected_input_size:
                # Pad with zeros
                pad_size = self.expected_input_size - num_features
                padding = torch.zeros((batch_size, seq_len, pad_size), dtype=torch.float32)
                features_tensor = torch.cat([features_tensor, padding], dim=2)
            else:
                # Truncate to expected size
                features_tensor = features_tensor[:, :, :self.expected_input_size]
            num_features = self.expected_input_size

        if mask is not None:
            mask_tensor = torch.tensor(mask, dtype=torch.float32)
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            mask_tensor = mask_tensor.to(features_tensor.dtype)
            
            # Pad or truncate mask to match features
            if mask_tensor.shape[2] != num_features:
                if mask_tensor.shape[2] < num_features:
                    pad_size = num_features - mask_tensor.shape[2]
                    padding = torch.zeros((mask_tensor.shape[0], mask_tensor.shape[1], pad_size), dtype=torch.float32)
                    mask_tensor = torch.cat([mask_tensor, padding], dim=2)
                else:
                    mask_tensor = mask_tensor[:, :, :num_features]
        else:
            mask_tensor = torch.ones((batch_size, seq_len, num_features), dtype=torch.float32)

        if mask_tensor.shape != features_tensor.shape:
            mask_tensor = torch.ones_like(features_tensor)

        min_seq_len = 4
        if features_tensor.shape[1] < min_seq_len:
            pad_length = min_seq_len - features_tensor.shape[1]
            pad_features = features_tensor[:, -1:, :].repeat(1, pad_length, 1)
            pad_mask = mask_tensor[:, -1:, :].repeat(1, pad_length, 1)
            features_tensor = torch.cat([features_tensor, pad_features], dim=1)
            mask_tensor = torch.cat([mask_tensor, pad_mask], dim=1)

        batch_size, seq_len, num_features = features_tensor.shape

        delta_t = torch.zeros((batch_size, seq_len, 1), dtype=torch.float32)

        member_scores: List[float] = []
        with torch.no_grad():
            for model in self.models:
                logits = model(features_tensor, mask_tensor, delta_t)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                member_scores.extend(probs.tolist())

        risk_score = float(np.mean(member_scores)) if member_scores else 0.5
        variance = float(np.var(member_scores)) if len(member_scores) > 1 else 0.0
        confidence = float(np.clip(1.0 - variance, 0.0, 1.0))

        if risk_score >= 0.7:
            risk_level = 'High'
        elif risk_score >= 0.3:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'

        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'confidence': confidence,
            'ensemble_members': len(self.models),
            'variance': variance
        }

class SepsisPredictionIntegration:
    """
    Comprehensive integration class that combines:
    - Person A's data preprocessing
    - Person B's baseline models (SIRS, qSOFA, NEWS2, SOFA, ML models)
    - Person C's deep learning models (GRU-D, LSTM, CNN-LSTM, Transformer)
    - Person D's explainability and dashboard
    """
    
    def __init__(self):
        """Initialize the integration system"""
        self.baseline_models = {}
        self.deep_learning_models = {}
        self.feature_names = []
        self.scaler = None
        self.feature_defaults: Dict[str, float] = {}
        self.is_initialized = False

        # Initialize the system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all models and preprocessing components"""
        try:
            print("🔄 Initializing Sepsis Prediction Integration System...")
            
            # Load feature names (from Person A/B data definitions)
            self._determine_feature_names()

            # Load default feature values from available datasets
            self._load_feature_defaults()

            # Load Person B's baseline models
            self._load_baseline_models()
            
            # Load Person C's deep learning models
            self._load_deep_learning_models()
            
            # Load preprocessing components
            self._load_preprocessing()
            
            self.is_initialized = True
            print("✅ Integration system initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            self.is_initialized = False

    def _determine_feature_names(self):
        """Determine feature names from available datasets."""
        dataset_candidates = [
            'Dataset.csv',
            'sample_patients_fixed.csv',
            'sample_patients_correct.csv',
            'sample_patients.csv'
        ]

        metadata_columns = {'Patient_ID', 'Time', 'Sepsis_Label', 'Unnamed: 0'}
        for path in dataset_candidates:
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path, nrows=1)
                column_mapping = {}
                if 'Hour' in df.columns and 'Time' not in df.columns:
                    column_mapping['Hour'] = 'Time'
                if 'SepsisLabel' in df.columns and 'Sepsis_Label' not in df.columns:
                    column_mapping['SepsisLabel'] = 'Sepsis_Label'
                if column_mapping:
                    df = df.rename(columns=column_mapping)

                features = [col for col in df.columns if col not in metadata_columns]
                if features:
                    self.feature_names = features
                    return
            except Exception as exc:
                print(f"⚠️ Unable to infer feature names from {path}: {exc}")

        if not self.feature_names:
            self.feature_names = [
                'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
                'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST',
                'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
                'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
                'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb',
                'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'Age', 'Gender',
                'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS',
                'MAP_rolling_mean_6hr', 'MAP_delta_1hr', 'qSOFA_Resp', 'qSOFA_MAP',
                'qSOFA_Score_simple'
            ]

    def _load_feature_defaults(self):
        """Populate default values for features using available sample datasets."""
        candidate_paths = [
            'sample_patients_fixed.csv',
            'sample_patients_correct.csv',
            'sample_patients.csv',
            'Dataset.csv'
        ]

        defaults: Dict[str, float] = {}
        for path in candidate_paths:
            if not os.path.exists(path):
                continue
            try:
                # Use a subset of rows for very large datasets to reduce memory pressure
                read_kwargs = {'nrows': 5000} if path == 'Dataset.csv' else {}
                df = pd.read_csv(path, **read_kwargs)
                for feature in self.feature_names:
                    if feature in df.columns:
                        median_value = pd.to_numeric(df[feature], errors='coerce').median()
                        if not np.isnan(median_value):
                            defaults.setdefault(feature, float(median_value))
                if len(defaults) == len(self.feature_names):
                    break
            except Exception as exc:
                print(f"⚠️ Unable to derive defaults from {path}: {exc}")

        self.feature_defaults = defaults

    def _load_baseline_models(self):
        """Load Person B's baseline models"""
        try:
            # Possible filenames for baseline models saved by collaborators
            candidate_paths: Dict[str, List[str]] = {
                'logistic_regression': [
                    'outputs/models/logistic_regression.pkl',
                    'outputs/models/model_logistic.pkl',
                    'model_logistic.pkl'
                ],
                'random_forest': [
                    'outputs/models/random_forest.pkl',
                    'outputs/models/model_rf.pkl',
                    'model_rf.pkl'
                ],
                'xgboost': [
                    'outputs/models/xgboost.pkl',
                    'outputs/models/model_xgb.pkl',
                    'model_xgb_or_hgb.pkl'
                ]
            }

            for name, paths in candidate_paths.items():
                model = self._load_first_available_model(paths)
                if model is not None:
                    self.baseline_models[name] = model
                    print(f"✅ Loaded {name}")
                else:
                    print(f"⚠️ Trained artifact for {name} not found. Checked: {paths}")

            if not self.baseline_models:
                print("⚠️ No baseline model artifacts were loaded. Baseline predictions will be unavailable until artifacts are provided.")

        except Exception as e:
            print(f"⚠️ Error loading baseline models: {e}")

    def _load_first_available_model(self, paths: List[str]):
        """Helper to load the first existing model from a list of candidate paths."""
        for path in paths:
            if not os.path.exists(path):
                continue
            try:
                if path.endswith('.joblib'):
                    return joblib.load(path)
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as exc:
                print(f"⚠️ Failed loading model at {path}: {exc}")
        return None
    
    def _load_deep_learning_models(self):
        """Load Person C's deep learning models"""
        try:
            from models import GRUD, LSTM, CNNLSTM, Transformer

            # Try to load input_size from config file (models were trained with this size)
            input_size = None
            config_paths = [
                'outputs/config.json',
                'Capstone/outputs/config.json'
            ]
            for config_path in config_paths:
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            if 'n_features' in config:
                                input_size = config['n_features']
                                print(f"📋 Loaded input_size={input_size} from {config_path}")
                                break
                    except Exception as e:
                        print(f"⚠️ Could not read config from {config_path}: {e}")
            
            # Fallback to feature_names length if config not found
            if input_size is None:
                input_size = len(self.feature_names)
                print(f"⚠️ Using input_size={input_size} from feature_names (config not found)")

            model_configs = {
                'grud': {
                    'class': GRUD,
                    'init_args': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3}
                },
                'lstm': {
                    'class': LSTM,
                    'init_args': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3}
                },
                'cnn_lstm': {
                    'class': CNNLSTM,
                    'init_args': {'hidden_size': 128, 'num_layers': 1, 'dropout': 0.3,
                                  'cnn_filters': [64, 128], 'kernel_size': 3}
                },
                'transformer': {
                    'class': Transformer,
                    'init_args': {'d_model': 128, 'nhead': 8, 'num_layers': 4, 'dropout': 0.1}
                }
            }

            seeds = [42, 43, 44]

            for name, cfg in model_configs.items():
                models: List[torch.nn.Module] = []
                for seed in seeds:
                    model_path = os.path.join('outputs', 'models', f'{name}_seed{seed}.pt')
                    if not os.path.exists(model_path):
                        continue
                    try:
                        model = cfg['class'](input_size=input_size, **cfg['init_args'])
                        state_dict = torch.load(model_path, map_location='cpu')
                        model.load_state_dict(state_dict)
                        models.append(model)
                    except Exception as e:
                        print(f"⚠️ Error loading {name}_seed{seed}: {e}")

                if not models:
                    fallback_path = os.path.join('outputs', 'models', f'{name}_real_data.pt')
                    if os.path.exists(fallback_path):
                        try:
                            model = cfg['class'](input_size=input_size, **cfg['init_args'])
                            state_dict = torch.load(fallback_path, map_location='cpu')
                            model.load_state_dict(state_dict)
                            models.append(model)
                        except Exception as e:
                            print(f"⚠️ Error loading {name}_real_data: {e}")

                if models:
                    self.deep_learning_models[name] = TorchModelEnsemble(name, models, input_size)
                    print(f"✅ Loaded {name} ({len(models)} checkpoint(s))")
                else:
                    print(f"⚠️ No checkpoints found for {name}")

        except Exception as e:
            print(f"⚠️ Error loading deep learning models: {e}")
            import traceback
            traceback.print_exc()

    def _load_preprocessing(self):
        """Load preprocessing components"""
        try:
            # Combined logic: Try several candidate paths, support both .pkl and .joblib
            scaler_candidates = [
                'outputs/models/scaler.pkl',
                'outputs/models/scaler.joblib',
                'outputs/cache/scaler.joblib',
                'scaler_lr.pkl',
                'outputs/scaler_final.pkl',
                'Capstone/outputs/models/scaler.pkl'
            ]

            self.scaler = None
            for path in scaler_candidates:
                if not os.path.exists(path):
                    continue
                try:
                    if path.endswith('.joblib'):
                        self.scaler = joblib.load(path)
                    else:
                        with open(path, 'rb') as f:
                            self.scaler = pickle.load(f)
                    # Check if scaler is fitted (has mean_)
                    if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                        print(f"✅ Loaded fitted scaler from {path}")
                    else:
                        print(f"⚠️ Loaded scaler from {path} but it's not fitted")
                    break
                except Exception as exc:
                    print(f"⚠️ Failed loading scaler at {path}: {exc}")

            if self.scaler is None:
                self.scaler = StandardScaler()
                print("⚠️ No pretrained scaler found. A new StandardScaler will be fitted on incoming data.")
        except Exception as e:
            print(f"⚠️ Error loading preprocessing: {e}")
            self.scaler = StandardScaler()
    
    def preprocess_patient_data(self, patient_data):
        """Preprocess patient data using Person A's methods"""
        try:
            # Convert to DataFrame if not already
            if isinstance(patient_data, dict):
                df = pd.DataFrame([patient_data])
            else:
                df = patient_data.copy()

            # If Series, convert to DataFrame
            if isinstance(df, pd.Series):
                df = df.to_frame().T

            df = df.reset_index(drop=True)

            # Handle categorical columns (Gender, Unit1, Unit2)
            if 'Gender' in df.columns:
                # Convert Gender to numeric: Male=1, Female=0
                df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0}).fillna(0)

            if 'Unit1' in df.columns:
                df['Unit1'] = pd.to_numeric(df['Unit1'], errors='coerce').fillna(0)

            if 'Unit2' in df.columns:
                df['Unit2'] = pd.to_numeric(df['Unit2'], errors='coerce').fillna(0)

            # Default values for common features (used when feature is missing)
            # If class/instance has self.feature_defaults, prefer it, else fallback defaults
            default_values = getattr(self, 'feature_defaults', None)
            if default_values is None:
                default_values = {
                    'HR': 80, 'O2Sat': 95, 'Temp': 37, 'SBP': 120, 'MAP': 75, 'DBP': 80,
                    'Resp': 16, 'EtCO2': 40, 'BaseExcess': 0, 'HCO3': 24, 'FiO2': 21,
                    'pH': 7.4, 'PaCO2': 40, 'SaO2': 95, 'AST': 30, 'BUN': 15,
                    'Alkalinephos': 100, 'Calcium': 9, 'Chloride': 100, 'Creatinine': 1.0,
                    'Bilirubin_direct': 0.2, 'Lactate': 1.0, 'Magnesium': 2.0, 'Phosphate': 3.5,
                    'Potassium': 4.0, 'Bilirubin_total': 1.0, 'TroponinI': 0.01, 'Hct': 40,
                    'Hgb': 12, 'PTT': 30, 'WBC': 8, 'Fibrinogen': 300, 'Platelets': 250,
                    'Age': 65, 'Gender': 0, 'Unit1': 0, 'Unit2': 0, 'HospAdmTime': 0, 'ICULOS': 0
                }

            # Build features and observation mask in the same way as the HEAD version
            feature_columns: Dict[str, pd.Series] = {}
            for feature in self.feature_names:
                if feature in df.columns:
                    series = pd.to_numeric(df[feature], errors='coerce')
                else:
                    series = pd.Series(np.nan, index=df.index, dtype=float)
                feature_columns[feature] = series

            df_features = pd.DataFrame(feature_columns)
            observation_mask = (~df_features.isna()).astype(np.float32).values

            for feature in self.feature_names:
                default_value = default_values.get(feature, 0.0)
                df_features[feature].fillna(default_value, inplace=True)

            feature_matrix = df_features.astype(np.float32).values

            # Scale features if scaler is available and fitted
            if self.scaler is not None:
                try:
                    # Check if scaler is fitted
                    is_fitted = hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None

                    if is_fitted:
                        # Check if scaler expects the same number of features
                        if hasattr(self.scaler, 'n_features_in_'):
                            expected_features = self.scaler.n_features_in_
                            if feature_matrix.shape[1] != expected_features:
                                # Try to use only the first N features if we have more
                                if feature_matrix.shape[1] > expected_features:
                                    feature_matrix = feature_matrix[:, :expected_features]
                                else:
                                    # Pad with zeros if we have fewer features
                                    padding = np.zeros((feature_matrix.shape[0], expected_features - feature_matrix.shape[1]))
                                    feature_matrix = np.hstack([feature_matrix, padding])
                        scaled_features = self.scaler.transform(feature_matrix)
                    else:
                        # Scaler not fitted - avoid fitting on single sample
                        scaled_features = feature_matrix
                except Exception as exc:
                    print(f"⚠️ Falling back to unscaled features: {exc}")
                    scaled_features = feature_matrix
            else:
                scaled_features = feature_matrix

            # For compatibility, return feature names as the third return value
            return scaled_features.astype(np.float32), observation_mask, self.feature_names

        except Exception as e:
            print(f"❌ Error preprocessing data: {e}")
            import traceback
            traceback.print_exc()
            return None, None, []

    
    def predict_baseline_models(self, patient_data):
        """Get predictions from Person B's baseline models"""
        predictions = {}
        
        try:
            # Preprocess data
            X_processed, _, _ = self.preprocess_patient_data(patient_data)

            if X_processed is None:
                return predictions

            # Get predictions from each baseline model
            for name, model in self.baseline_models.items():
                try:
                    # Check what features the model expects
                    X_for_model = X_processed.copy()
                    
                    if hasattr(model, 'n_features_in_'):
                        expected_features = model.n_features_in_
                        current_features = X_for_model.shape[1]
                        
                        if current_features != expected_features:
                            if current_features > expected_features:
                                # Use only the first N features
                                X_for_model = X_for_model[:, :expected_features]
                            else:
                                # Pad with zeros
                                padding = np.zeros((X_for_model.shape[0], expected_features - current_features))
                                X_for_model = np.hstack([X_for_model, padding])
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_for_model)[0]
                        prob = proba[1] if len(proba) > 1 else proba[0]
                        # Calculate confidence based on probability distribution
                        # Higher confidence when probability is more extreme (closer to 0 or 1)
                        # Also consider the difference between class probabilities
                        if len(proba) > 1:
                            max_prob = max(proba)
                            # Confidence increases with distance from 0.5 and with class separation
                            confidence = float(np.clip(
                                0.5 + abs(prob - 0.5) * 0.8 + (max_prob - 0.5) * 0.3,
                                0.5, 0.95
                            ))
                        else:
                            # Fallback if only one probability
                            confidence = float(np.clip(0.5 + abs(prob - 0.5) * 0.8, 0.5, 0.95))
                    else:
                        prob = model.predict(X_for_model)[0]
                        # If predict returns binary, convert to probability-like score
                        if isinstance(prob, (int, np.integer)):
                            prob = float(prob)
                            # For binary predictions, lower confidence since we don't have probability distribution
                            confidence = 0.65
                        elif prob > 1.0:
                            prob = prob / 100.0
                            confidence = float(np.clip(0.5 + abs(prob - 0.5) * 0.6, 0.5, 0.9))
                        else:
                            # Assume it's already a probability
                            confidence = float(np.clip(0.5 + abs(prob - 0.5) * 0.8, 0.5, 0.95))
                    
                    predictions[name] = {
                        'risk_score': prob,
                        'risk_level': 'High' if prob > 0.7 else 'Medium' if prob > 0.3 else 'Low',
                        'confidence': confidence
                    }
                except Exception as e:
                    print(f"⚠️ Error with {name}: {e}")
                    import traceback
                    traceback.print_exc()
                    predictions[name] = {
                        'risk_score': 0.5,
                        'risk_level': 'Medium',
                        'confidence': 0.5
                    }
            
            # Calculate clinical scores
            predictions['clinical_scores'] = self._calculate_clinical_scores(patient_data)
            
        except Exception as e:
            print(f"❌ Error in baseline predictions: {e}")
        
        return predictions
    
    def predict_deep_learning_models(self, patient_data):
        """Get predictions from Person C's deep learning models"""
        predictions = {}
        
        try:
            # Preprocess data
            X_processed, observation_mask, _ = self.preprocess_patient_data(patient_data)

            if X_processed is None or observation_mask is None:
                return predictions

            # Get predictions from each deep learning model
            for name, model in self.deep_learning_models.items():
                try:
                    if hasattr(model, 'predict'):
                        seq_features = X_processed[np.newaxis, ...]
                        seq_mask = observation_mask[np.newaxis, ...]
                        predictions[name] = model.predict(seq_features, seq_mask)
                    else:
                        print(f"⚠️ Loaded deep learning component for {name} lacks predict method")
                except Exception as e:
                    print(f"⚠️ Error with {name}: {e}")
                    predictions[name] = {
                        'risk_score': 0.5,
                        'risk_level': 'Medium',
                        'confidence': 0.5
                    }
        
        except Exception as e:
            print(f"❌ Error in deep learning predictions: {e}")
        
        return predictions
    
    def _calculate_clinical_scores(self, patient_data):
        """Calculate clinical scores (Person B's implementation)"""
        scores = {}

        try:
            if isinstance(patient_data, pd.DataFrame):
                record = patient_data.iloc[-1].to_dict()
            elif isinstance(patient_data, dict):
                record = patient_data
            else:
                try:
                    record = dict(patient_data)
                except Exception:
                    record = {}

            # SIRS Score
            sirs_score = 0
            temp = record.get('Temp', 37)
            if temp > 38 or temp < 36:
                sirs_score += 1
            if record.get('HR', 80) > 90:
                sirs_score += 1
            if record.get('Resp', 16) > 20:
                sirs_score += 1
            wbc = record.get('WBC', 8)
            if wbc > 12 or wbc < 4:
                sirs_score += 1
            scores['sirs'] = sirs_score

            # qSOFA Score
            qsofa_score = 0
            if record.get('Resp', 16) >= 22:
                qsofa_score += 1
            if record.get('SBP', 120) <= 100:
                qsofa_score += 1
            scores['qsofa'] = qsofa_score

            # SOFA Score (partial)
            sofa_score = 0
            platelets = record.get('Platelets', 250)
            if platelets < 20:
                sofa_score += 4
            elif platelets < 50:
                sofa_score += 3
            elif platelets < 100:
                sofa_score += 2
            elif platelets < 150:
                sofa_score += 1

            bilirubin = record.get('Bilirubin_total', 1.0)
            if bilirubin >= 12:
                sofa_score += 4
            elif bilirubin >= 6:
                sofa_score += 3
            elif bilirubin >= 2:
                sofa_score += 2
            elif bilirubin >= 1.2:
                sofa_score += 1
            
            scores['sofa'] = sofa_score
            
        except Exception as e:
            print(f"⚠️ Error calculating clinical scores: {e}")
            scores = {'sirs': 0, 'qsofa': 0, 'sofa': 0}
        
        return scores
    
    def get_comprehensive_prediction(self, patient_data):
        """Get comprehensive prediction combining all models"""
        if not self.is_initialized:
            return None
        
        try:
            # Get baseline predictions
            baseline_preds = self.predict_baseline_models(patient_data)
            
            # Get deep learning predictions
            dl_preds = self.predict_deep_learning_models(patient_data)
            
            # Combine all predictions
            all_predictions = {**baseline_preds, **dl_preds}
            
            # Calculate ensemble prediction
            risk_scores = []
            for pred in all_predictions.values():
                if isinstance(pred, dict) and 'risk_score' in pred:
                    risk_scores.append(pred['risk_score'])
            
            if risk_scores:
                ensemble_score = np.mean(risk_scores)
                ensemble_level = 'High' if ensemble_score > 0.7 else 'Medium' if ensemble_score > 0.3 else 'Low'
            else:
                ensemble_score = 0.5
                ensemble_level = 'Medium'
            
            return {
                'individual_predictions': all_predictions,
                'ensemble_prediction': {
                    'average_risk_score': float(ensemble_score),
                    'risk_level': ensemble_level,
                    'agreement_score': 0.85,  # Placeholder
                    'recommended_action': self._get_recommendation(ensemble_level)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error in comprehensive prediction: {e}")
            return None
    
    def _get_recommendation(self, risk_level):
        """Get clinical recommendation based on risk level"""
        if risk_level == 'High':
            return 'Immediate clinical attention required'
        elif risk_level == 'Medium':
            return 'Close monitoring recommended'
        else:
            return 'Continue routine monitoring'
    
    def get_feature_importance(self, patient_data):
        """Get feature importance for explainability"""
        try:
            # Simple feature importance based on deviation from normal values
            importance = []
            
            normal_values = {
                'HR': 80, 'Temp': 37, 'Resp': 16, 'WBC': 8, 'SBP': 120,
                'O2Sat': 95, 'Creatinine': 1.0, 'Bilirubin_total': 1.0
            }
            
            for feature, normal_val in normal_values.items():
                if feature in patient_data:
                    value = patient_data[feature]
                    deviation = abs(value - normal_val) / normal_val
                    importance.append({
                        'feature': feature.lower(),
                        'importance': min(deviation, 1.0),
                        'trend': 'increasing' if value > normal_val else 'stable'
                    })
            
            # Sort by importance
            importance.sort(key=lambda x: x['importance'], reverse=True)
            
            return {'top_features': importance[:5]}
            
        except Exception as e:
            print(f"❌ Error calculating feature importance: {e}")
            return {'top_features': []}

# Global instance
integration_system = SepsisPredictionIntegration()

def get_integration_system():
    """Get the global integration system instance"""
    return integration_system

def check_baseline_models_status():
    """Helper function to check if baseline models are loaded and available"""
    system = get_integration_system()
    if not system.is_initialized:
        return {
            'initialized': False,
            'baseline_models_loaded': 0,
            'baseline_models': []
        }
    
    baseline_models = list(system.baseline_models.keys())
    return {
        'initialized': True,
        'baseline_models_loaded': len(baseline_models),
        'baseline_models': baseline_models,
        'deep_learning_models_loaded': len(system.deep_learning_models),
        'deep_learning_models': list(system.deep_learning_models.keys())
    }

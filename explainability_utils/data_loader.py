"""
Data loading utilities for ICU time-series data
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class ICUDataset(Dataset):
    """
    PyTorch Dataset for ICU time-series data with variable-length sequences
    """
    
    def __init__(self, data: pd.DataFrame, sequence_length: int = 24, 
                 prediction_horizon: int = 4, features: List[str] = None):
        """
        Initialize ICU Dataset
        
        Args:
            data: DataFrame with columns ['Patient_ID', 'Time', 'Sepsis_Label', ...features]
            sequence_length: Maximum sequence length for padding
            prediction_horizon: Hours ahead to predict (4 or 6)
            features: List of feature column names
        """
        self.data = data.copy()
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Get feature columns (exclude Patient_ID, Time, Sepsis_Label)
        if features is None:
            self.features = [col for col in data.columns 
                           if col not in ['Patient_ID', 'Time', 'Sepsis_Label']]
        else:
            self.features = features
            
        self.n_features = len(self.features)
        
        # Group by patient and create sequences
        self.sequences = self._create_sequences()
        
    def _create_sequences(self) -> List[Dict]:
        """Create sequences grouped by patient"""
        sequences = []
        
        for patient_id, patient_data in self.data.groupby('Patient_ID'):
            # Sort by time
            patient_data = patient_data.sort_values('Time').reset_index(drop=True)
            
            # Create sequences with sliding window
            for i in range(len(patient_data) - self.sequence_length):
                # Input sequence
                seq_data = patient_data.iloc[i:i + self.sequence_length]
                
                # Target: sepsis label at prediction_horizon hours ahead
                target_idx = min(i + self.sequence_length + self.prediction_horizon - 1, 
                               len(patient_data) - 1)
                target_label = patient_data.iloc[target_idx]['Sepsis_Label']
                
                # Extract features and create masks for missing values
                features_matrix = seq_data[self.features].values
                mask_matrix = ~np.isnan(features_matrix)
                
                # Fill NaN values with 0 (will be masked)
                features_matrix = np.nan_to_num(features_matrix, nan=0.0)
                
                sequences.append({
                    'features': features_matrix,
                    'mask': mask_matrix,
                    'target': target_label,
                    'patient_id': patient_id,
                    'time_start': patient_data.iloc[i]['Time'],
                    'time_end': patient_data.iloc[i + self.sequence_length - 1]['Time']
                })
                
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sequence"""
        seq = self.sequences[idx]
        
        # Convert to tensors
        features = torch.FloatTensor(seq['features'])
        mask = torch.BoolTensor(seq['mask'])
        target = torch.LongTensor([seq['target']])
        
        return features, mask, target


def create_data_loaders(train_data: pd.DataFrame, val_data: pd.DataFrame, 
                       test_data: pd.DataFrame, batch_size: int = 32,
                       sequence_length: int = 24, prediction_horizon: int = 4,
                       features: List[str] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        train_data: Training data
        val_data: Validation data  
        test_data: Test data
        batch_size: Batch size for training
        sequence_length: Maximum sequence length
        prediction_horizon: Prediction horizon in hours
        features: Feature column names
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create datasets
    train_dataset = ICUDataset(train_data, sequence_length, prediction_horizon, features)
    val_dataset = ICUDataset(val_data, sequence_length, prediction_horizon, features)
    test_dataset = ICUDataset(test_data, sequence_length, prediction_horizon, features)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    
    return train_loader, val_loader, test_loader


def load_preprocessed_data(data_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load preprocessed data from Person A
    
    Args:
        data_path: Path to the preprocessed dataset
        
    Returns:
        Tuple of (data, feature_columns)
    """
    try:
        data = pd.read_csv(data_path)
        print(f"Loaded data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Handle different column name formats
        # Rename columns to standard format if needed
        column_mapping = {}
        if 'Hour' in data.columns and 'Time' not in data.columns:
            column_mapping['Hour'] = 'Time'
        if 'SepsisLabel' in data.columns and 'Sepsis_Label' not in data.columns:
            column_mapping['SepsisLabel'] = 'Sepsis_Label'
        
        if column_mapping:
            data = data.rename(columns=column_mapping)
            print(f"Renamed columns: {column_mapping}")
        
        # Get feature columns (exclude metadata columns)
        metadata_cols = ['Patient_ID', 'Time', 'Sepsis_Label', 'Unnamed: 0']
        feature_cols = [col for col in data.columns if col not in metadata_cols]
        
        print(f"Number of features: {len(feature_cols)}")
        print(f"Unique patients: {data['Patient_ID'].nunique()}")
        print(f"Class distribution:")
        print(data['Sepsis_Label'].value_counts())
        print(f"Percentage:")
        print(data['Sepsis_Label'].value_counts(normalize=True) * 100)
        
        return data, feature_cols
        
    except FileNotFoundError:
        print(f"Data file not found at {data_path}")
        print("Creating sample data for testing...")
        return create_sample_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        print("Creating sample data for testing...")
        return create_sample_data()


def create_sample_data(n_patients: int = 100, n_hours: int = 48, 
                     n_features: int = 20) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create sample data for testing when real data is not available
    
    Args:
        n_patients: Number of patients
        n_hours: Hours per patient
        n_features: Number of features
        
    Returns:
        Tuple of (sample_data, feature_columns)
    """
    np.random.seed(42)
    
    data = []
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    
    for patient_id in range(n_patients):
        # Random sepsis onset time (if any)
        sepsis_onset = np.random.choice([None, np.random.randint(12, n_hours-6)])
        
        for hour in range(n_hours):
            # Generate features with some missing values
            features = np.random.normal(0, 1, n_features)
            
            # Add missing values (10% missing)
            missing_mask = np.random.random(n_features) < 0.1
            features[missing_mask] = np.nan
            
            # Sepsis label
            sepsis_label = 0
            if sepsis_onset is not None and hour >= sepsis_onset:
                sepsis_label = 1
            
            row = {
                'Patient_ID': patient_id,
                'Time': hour,
                'Sepsis_Label': sepsis_label,
                **{col: val for col, val in zip(feature_cols, features)}
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    print(f"Created sample data: {df.shape}")
    print(f"Class distribution: {df['Sepsis_Label'].value_counts()}")
    
    return df, feature_cols


def get_train_val_test_split(data: pd.DataFrame, 
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by patient ID to avoid data leakage
    
    Args:
        data: Full dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    unique_patients = data['Patient_ID'].unique()
    np.random.shuffle(unique_patients)
    
    n_train = int(len(unique_patients) * train_ratio)
    n_val = int(len(unique_patients) * val_ratio)
    
    train_patients = unique_patients[:n_train]
    val_patients = unique_patients[n_train:n_train + n_val]
    test_patients = unique_patients[n_train + n_val:]
    
    train_data = data[data['Patient_ID'].isin(train_patients)].copy()
    val_data = data[data['Patient_ID'].isin(val_patients)].copy()
    test_data = data[data['Patient_ID'].isin(test_patients)].copy()
    
    print(f"Train: {len(train_patients)} patients, {len(train_data)} records")
    print(f"Val: {len(val_patients)} patients, {len(val_data)} records")
    print(f"Test: {len(test_patients)} patients, {len(test_data)} records")
    
    return train_data, val_data, test_data

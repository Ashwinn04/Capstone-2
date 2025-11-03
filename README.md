# Sepsis Digital Twin - Deep Learning Models (Person C)

This repository contains the deep learning implementation for sepsis prediction 4-6 hours before onset using ICU time-series data.

## Project Overview

This component implements and evaluates four advanced deep learning architectures:
- **GRU-D**: Handles missing data with time-decay mechanism
- **LSTM**: Bidirectional LSTM for sequential modeling
- **CNN-LSTM**: Hybrid architecture for local temporal patterns
- **Transformer**: Attention-based modeling for long-range dependencies

## Setup Instructions

### Local Environment
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run notebooks in order:
```bash
jupyter notebook notebooks/
```

### Google Colab Setup
1. Upload the project to Google Drive
2. Open notebooks in Google Colab
3. Run the setup cell in each notebook to install dependencies
4. Mount Google Drive for data persistence

## Project Structure

```
├── notebooks/           # Jupyter notebooks for each model
├── models/             # PyTorch model implementations
├── utils/              # Utility functions and classes
├── outputs/            # Model weights, figures, and results
└── requirements.txt    # Python dependencies
```

## Notebook Execution Order

1. `01_data_loading_and_exploration.ipynb` - Data pipeline setup
2. `02_model_grud.ipynb` - GRU-D implementation
3. `03_model_lstm.ipynb` - LSTM implementation
4. `04_model_cnn_lstm.ipynb` - CNN-LSTM implementation
5. `05_model_transformer.ipynb` - Transformer implementation
6. `06_model_calibration.ipynb` - Probability calibration
7. `07_comparative_evaluation.ipynb` - Final evaluation and comparison

## Key Features

- **CPU Optimized**: Designed for CPU training with Google Colab compatibility
- **Reproducible**: Fixed random seeds and documented hyperparameters
- **Modular**: Clean separation of models, utilities, and notebooks
- **Integration Ready**: JSON outputs for dashboard integration

## Success Metrics

- Target AUROC ≥ 0.85
- 4-6 hour prediction lead-time
- Outperform baseline models
- Calibrated probability outputs

## Dependencies

- **Person A**: Preprocessed ICU dataset
- **Person B**: Baseline model results for comparison
- **Person D**: Dashboard integration requirements

## Contact

Person C - Deep Learning Models Implementation
 
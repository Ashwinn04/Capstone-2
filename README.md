# Sepsis Digital Twin - AI-Powered Early Detection System

A comprehensive sepsis prediction system that combines clinical scoring, traditional machine learning, and deep learning models for early sepsis detection 4-6 hours before onset using ICU time-series data.

## Project Overview

The Sepsis Digital Twin system integrates multiple prediction approaches:

### Deep Learning Models
- **GRU-D**: Handles missing data with time-decay mechanism
- **LSTM**: Bidirectional LSTM for sequential modeling
- **CNN-LSTM**: Hybrid architecture for local temporal patterns
- **Transformer**: Attention-based modeling for long-range dependencies

### Traditional Machine Learning Models
- **Logistic Regression**: Baseline linear classifier
- **Random Forest**: Ensemble decision trees
- **XGBoost**: Gradient boosting for sepsis risk classification

### Clinical Scoring Systems
- **SIRS**: Systemic Inflammatory Response Syndrome criteria
- **qSOFA**: Quick Sequential Organ Failure Assessment
- **NEWS2**: National Early Warning Score 2
- **SOFA**: Sequential Organ Failure Assessment

### Interactive Dashboard
- **Real-Time Patient Monitoring**: Analyze ICU patients with live data
- **Manual Patient Entry**: Add and analyze custom patient profiles
- **Comprehensive Risk Analysis**: Clinical scores + ML/DL model predictions
- **Explainable AI**: Feature importance and model interpretability

## Quick Start

### Running the Interactive Dashboard
```bash
# Activate virtual environment (if using venv)
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard_real.py --server.port 8502
```

Then open your browser to: **http://localhost:8502**

### Features Available
1. **Browse Real Data**: Analyze ICU patients from the dataset
2. **Add Manual Patients**: Enter custom patient data for risk analysis
3. **Upload CSV Files**: Bulk analyze multiple patients
4. **View Risk Predictions**: See risk scores from all 7 models
5. **Clinical Scores**: SIRS, qSOFA, NEWS2, SOFA calculations
6. **Explainability**: Understand which features drive predictions

### Model Training (Optional)
If you want to train new models:
```bash
# Run training scripts
python train_grud_demo.py           # Train GRU-D on sample data
python train_with_real_data.py     # Train on real ICU data
python evaluate_models.py           # Evaluate model performance
```

## Project Structure

```
├── dashboard_real.py           # Main interactive dashboard (Streamlit)
├── models/                     # Deep learning model implementations
│   ├── grud.py                # GRU-D model
│   ├── lstm.py                 # LSTM model
│   ├── cnn_lstm.py            # CNN-LSTM hybrid
│   └── transformer.py         # Transformer architecture
├── utils/                      # Utility functions
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── training.py            # Training utilities
│   ├── metrics.py            # Evaluation metrics
│   └── visualization.py       # Plotting functions
├── explainability_utils/       # Model interpretability
│   └── explainability.py     # SHAP, Integrated Gradients
├── notebooks/                  # Jupyter notebooks for exploration
├── outputs/                    # Model artifacts and results
│   ├── models/               # Trained model weights
│   ├── figures/              # Generated visualizations
│   └── results/              # Evaluation results
├── Dataset.csv                # ICU patient data
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Enhanced Features

### Add Patient with Comprehensive Risk Analysis
The dashboard now supports **manual patient entry** with complete clinical risk assessment:

1. **Input All Required Data**: 28 medical parameters including:
   - Demographics (Age, Gender)
   - Vital Signs (HR, BP, Temperature, Respiratory Rate, O2 Saturation)
   - Arterial Blood Gas (pH, PaCO2, SaO2, etc.)
   - Laboratory Values (Lactate, WBC, Platelets, Creatinine, Bilirubin)

2. **Automated Clinical Scoring**:
   - SIRS, qSOFA, NEWS2, and SOFA scores
   - Risk factor identification (8+ clinical indicators)
   - Risk level classification (High/Medium/Low)

3. **Model-Based Predictions**:
   - Attempts to use trained XGBoost model
   - Ensemble predictions from 7 models (GRU-D, LSTM, CNN-LSTM, Transformer, Logistic Regression, Random Forest, XGBoost)
   - Robust fallback to clinical rules if models unavailable

4. **Intelligent Risk Assessment**:
   - Combines clinical and model predictions
   - Takes conservative approach (maximum of both)
   - Provides actionable clinical recommendations

### Dashboard Capabilities
- **Real-Time Analysis**: Process ICU patient data in real-time
- **Multiple Data Sources**: Real data, uploaded CSV, or manual entry
- **Visual Patient Indicators**: Icons show data source (➕ Manual, 🏥 Real, 📁 Uploaded)
- **Comprehensive Tabs**: Risk Overview, Model Predictions, Risk Trajectory, Explainability, Patient Data
- **Export Functionality**: Export reports, risk data, and performance metrics

## Technical Architecture

- **Frontend**: Streamlit web application
- **Backend**: Python with NumPy, Pandas, Scikit-learn
- **Deep Learning**: PyTorch for neural networks
- **Data Preprocessing**: Custom loaders with imputation and scaling
- **Model Inference**: XGBoost with fallback to clinical scoring
- **Explainability**: Captum for feature attribution (SHAP, Integrated Gradients)

## Performance Metrics

- Target AUROC ≥ 0.85
- 4-6 hour prediction lead-time
- Outperform baseline models
- Calibrated probability outputs
- High sensitivity for early detection

## Dependencies

- Python 3.7+
- PyTorch 2.0+
- XGBoost
- Streamlit
- NumPy, Pandas, Scikit-learn
- Captum (for explainability)
- See `requirements.txt` for complete list

## Contributing

This project represents a collaborative effort with multiple team members contributing different components:
- Data preprocessing and feature engineering
- Traditional ML model development
- Deep learning architecture design and training
- Dashboard development and integration

## License

Academic/Educational Project
 
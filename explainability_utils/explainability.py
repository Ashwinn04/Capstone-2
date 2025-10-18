"""
Explainability tools for sepsis prediction models
Implements SHAP, LIME, and Integrated Gradients
"""
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, GradientShap, Saliency
# import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class SepsisExplainer:
    """
    Comprehensive explainability class for sepsis prediction models
    """
    
    def __init__(self, model, feature_names: List[str], background_data: np.ndarray):
        """
        Initialize explainer with model and background data
        
        Args:
            model: Trained sepsis prediction model
            feature_names: List of feature names
            background_data: Background data for SHAP/LIME (shape: [n_samples, n_features])
        """
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.n_features = len(feature_names)
        
        # Initialize explainers
        self._init_explainers()
    
    def _init_explainers(self):
        """Initialize various explainability tools"""
        try:
            # SHAP explainer
            self.shap_explainer = shap.Explainer(self.model, self.background_data)
            print("✅ SHAP explainer initialized")
        except Exception as e:
            print(f"⚠️ SHAP explainer failed: {e}")
            self.shap_explainer = None
        
        try:
            # LIME explainer
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.background_data,
                feature_names=self.feature_names,
                class_names=['No Sepsis', 'Sepsis'],
                mode='classification',
                random_state=42
            )
            print("✅ LIME explainer initialized")
        except Exception as e:
            print(f"⚠️ LIME explainer failed: {e}")
            self.lime_explainer = None
        
        try:
            # Captum explainers
            self.integrated_gradients = IntegratedGradients(self.model)
            self.gradient_shap = GradientShap(self.model)
            self.saliency = Saliency(self.model)
            print("✅ Captum explainers initialized")
        except Exception as e:
            print(f"⚠️ Captum explainers failed: {e}")
            self.integrated_gradients = None
            self.gradient_shap = None
            self.saliency = None
    
    def get_shap_explanations(self, X: np.ndarray, max_evals: int = 100) -> Dict[str, Any]:
        """
        Generate SHAP explanations for input data
        
        Args:
            X: Input data (shape: [n_samples, n_features])
            max_evals: Maximum evaluations for SHAP
            
        Returns:
            Dictionary with SHAP values and plots
        """
        if self.shap_explainer is None:
            return {"error": "SHAP explainer not available"}
        
        try:
            # Calculate SHAP values
            shap_values = self.shap_explainer(X, max_evals=max_evals)
            
            # Get feature importance
            feature_importance = np.abs(shap_values.values).mean(axis=0)
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            return {
                'shap_values': shap_values.values,
                'feature_importance': feature_importance_df,
                'base_value': shap_values.base_values,
                'data': shap_values.data
            }
        except Exception as e:
            return {"error": f"SHAP calculation failed: {e}"}
    
    def get_lime_explanations(self, X: np.ndarray, num_features: int = 10) -> List[Dict[str, Any]]:
        """
        Generate LIME explanations for input data
        
        Args:
            X: Input data (shape: [n_samples, n_features])
            num_features: Number of top features to explain
            
        Returns:
            List of LIME explanations for each sample
        """
        if self.lime_explainer is None:
            return [{"error": "LIME explainer not available"}]
        
        explanations = []
        for i in range(len(X)):
            try:
                exp = self.lime_explainer.explain_instance(
                    X[i], 
                    self.model.predict_proba, 
                    num_features=num_features
                )
                
                # Extract explanation data
                explanation_data = {
                    'sample_idx': i,
                    'prediction': exp.predicted_value,
                    'local_prediction': exp.local_pred,
                    'intercept': exp.intercept,
                    'explanation': exp.as_list(),
                    'score': exp.score
                }
                explanations.append(explanation_data)
            except Exception as e:
                explanations.append({"sample_idx": i, "error": f"LIME failed: {e}"})
        
        return explanations
    
    def get_integrated_gradients(self, X: torch.Tensor, target: int = 1) -> np.ndarray:
        """
        Generate Integrated Gradients explanations
        
        Args:
            X: Input tensor (shape: [batch_size, n_features])
            target: Target class for attribution
            
        Returns:
            Attribution values
        """
        if self.integrated_gradients is None:
            return np.zeros(X.shape)
        
        try:
            attributions = self.integrated_gradients.attribute(
                X, 
                target=target,
                n_steps=50
            )
            return attributions.detach().numpy()
        except Exception as e:
            print(f"Integrated Gradients failed: {e}")
            return np.zeros(X.shape)
    
    def get_gradient_shap(self, X: torch.Tensor, baselines: torch.Tensor, target: int = 1) -> np.ndarray:
        """
        Generate Gradient SHAP explanations
        
        Args:
            X: Input tensor
            baselines: Baseline tensor
            target: Target class
            
        Returns:
            Attribution values
        """
        if self.gradient_shap is None:
            return np.zeros(X.shape)
        
        try:
            attributions = self.gradient_shap.attribute(
                X,
                baselines=baselines,
                target=target
            )
            return attributions.detach().numpy()
        except Exception as e:
            print(f"Gradient SHAP failed: {e}")
            return np.zeros(X.shape)
    
    def get_saliency_maps(self, X: torch.Tensor, target: int = 1) -> np.ndarray:
        """
        Generate Saliency maps
        
        Args:
            X: Input tensor
            target: Target class
            
        Returns:
            Saliency values
        """
        if self.saliency is None:
            return np.zeros(X.shape)
        
        try:
            saliency_attr = self.saliency.attribute(X, target=target)
            return saliency_attr.detach().numpy()
        except Exception as e:
            print(f"Saliency failed: {e}")
            return np.zeros(X.shape)
    
    def get_comprehensive_explanations(self, X: np.ndarray, 
                                     X_tensor: Optional[torch.Tensor] = None,
                                     baselines: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Get comprehensive explanations using all available methods
        
        Args:
            X: Input data (numpy array)
            X_tensor: Input data (PyTorch tensor)
            baselines: Baseline data for Gradient SHAP
            
        Returns:
            Dictionary with all explanations
        """
        explanations = {}
        
        # SHAP explanations
        explanations['shap'] = self.get_shap_explanations(X)
        
        # LIME explanations
        explanations['lime'] = self.get_lime_explanations(X)
        
        # Captum explanations (if tensor provided)
        if X_tensor is not None:
            explanations['integrated_gradients'] = self.get_integrated_gradients(X_tensor)
            explanations['saliency'] = self.get_saliency_maps(X_tensor)
            
            if baselines is not None:
                explanations['gradient_shap'] = self.get_gradient_shap(X_tensor, baselines)
        
        return explanations
    
    def create_feature_importance_plot(self, explanations: Dict[str, Any], 
                                     method: str = 'shap', 
                                     top_k: int = 10):
        """
        Create feature importance visualization using Plotly
        
        Args:
            explanations: Dictionary with explanations
            method: Method to use ('shap', 'lime', 'integrated_gradients')
            top_k: Number of top features to show
            
        Returns:
            Plotly figure
        """
        if method == 'shap' and 'shap' in explanations:
            if 'error' not in explanations['shap']:
                df = explanations['shap']['feature_importance'].head(top_k)
                fig = px.bar(
                    df, 
                    x='importance', 
                    y='feature',
                    orientation='h',
                    title='Top Features by SHAP Importance',
                    labels={'importance': 'SHAP Value', 'feature': 'Feature'}
                )
                return fig
        
        elif method == 'lime' and 'lime' in explanations:
            # Aggregate LIME explanations across samples
            feature_importance = {}
            for exp in explanations['lime']:
                if 'explanation' in exp:
                    for feature, importance in exp['explanation']:
                        if feature not in feature_importance:
                            feature_importance[feature] = []
                        feature_importance[feature].append(abs(importance))
            
            # Calculate average importance
            avg_importance = {k: np.mean(v) for k, v in feature_importance.items()}
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            features, importances = zip(*sorted_features)
            fig = px.bar(
                x=importances,
                y=features,
                orientation='h',
                title='Top Features by LIME Importance',
                labels={'x': 'LIME Weight', 'y': 'Feature'}
            )
            return fig
        
        return None
    
    def create_waterfall_plot(self, explanations: Dict[str, Any], 
                            sample_idx: int = 0):
        """
        Create SHAP waterfall plot for a single sample using Plotly
        
        Args:
            explanations: Dictionary with explanations
            sample_idx: Index of sample to explain
            
        Returns:
            Plotly figure
        """
        if 'shap' not in explanations or 'error' in explanations['shap']:
            fig = go.Figure()
            fig.add_annotation(
                text="SHAP waterfall plot not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="SHAP Waterfall Plot")
            return fig
        
        shap_values = explanations['shap']['shap_values']
        base_value = explanations['shap']['base_value']
        
        if len(shap_values) <= sample_idx:
            sample_idx = 0
        
        sample_shap = shap_values[sample_idx]
        sample_data = explanations['shap']['data'][sample_idx]
        
        # Sort features by absolute SHAP value
        feature_order = np.argsort(np.abs(sample_shap))[::-1]
        
        # Create waterfall plot using Plotly
        features = [self.feature_names[i] for i in feature_order]
        values = sample_shap[feature_order]
        
        fig = go.Figure(go.Waterfall(
            name="SHAP Values",
            orientation="v",
            measure=["relative"] * len(features),
            x=features,
            y=values,
            connector={"line":{"color":"rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=f"SHAP Waterfall Plot (Sample {sample_idx})",
            showlegend=False,
            height=600
        )
        
        return fig


def create_clinical_interpretation(explanations: Dict[str, Any], 
                                 feature_names: List[str]) -> Dict[str, Any]:
    """
    Convert technical explanations to clinical interpretations
    
    Args:
        explanations: Dictionary with explanations
        feature_names: List of feature names
        
    Returns:
        Dictionary with clinical interpretations
    """
    clinical_terms = {
        'HR': 'Heart Rate',
        'O2Sat': 'Oxygen Saturation',
        'Temp': 'Temperature',
        'SBP': 'Systolic Blood Pressure',
        'MAP': 'Mean Arterial Pressure',
        'DBP': 'Diastolic Blood Pressure',
        'Resp': 'Respiratory Rate',
        'WBC': 'White Blood Cell Count',
        'Platelets': 'Platelet Count',
        'Creatinine': 'Creatinine Level',
        'Bilirubin_total': 'Total Bilirubin',
        'Lactate': 'Lactate Level',
        'Age': 'Patient Age'
    }
    
    interpretation = {
        'clinical_summary': [],
        'risk_factors': [],
        'protective_factors': [],
        'recommendations': []
    }
    
    # Process SHAP explanations if available
    if 'shap' in explanations and 'error' not in explanations['shap']:
        feature_importance = explanations['shap']['feature_importance']
        
        for _, row in feature_importance.head(5).iterrows():
            feature = row['feature']
            importance = row['importance']
            clinical_name = clinical_terms.get(feature, feature)
            
            if importance > 0.1:  # High importance threshold
                interpretation['risk_factors'].append({
                    'feature': clinical_name,
                    'technical_name': feature,
                    'importance': importance,
                    'description': f'{clinical_name} is a significant risk factor for sepsis'
                })
    
    # Process LIME explanations if available
    if 'lime' in explanations:
        for exp in explanations['lime']:
            if 'explanation' in exp:
                for feature, importance in exp['explanation']:
                    clinical_name = clinical_terms.get(feature, feature)
                    
                    if importance > 0.05:  # Positive contribution
                        interpretation['risk_factors'].append({
                            'feature': clinical_name,
                            'technical_name': feature,
                            'importance': abs(importance),
                            'description': f'{clinical_name} contributes to sepsis risk'
                        })
                    elif importance < -0.05:  # Negative contribution
                        interpretation['protective_factors'].append({
                            'feature': clinical_name,
                            'technical_name': feature,
                            'importance': abs(importance),
                            'description': f'{clinical_name} is protective against sepsis'
                        })
    
    # Generate clinical recommendations
    if interpretation['risk_factors']:
        interpretation['recommendations'].append("Monitor high-risk features closely")
        interpretation['recommendations'].append("Consider early intervention protocols")
    
    if interpretation['protective_factors']:
        interpretation['recommendations'].append("Maintain current treatment protocols")
    
    interpretation['recommendations'].append("Continue regular monitoring")
    
    return interpretation


def get_dynamic_explainability(patient_data: Dict[str, Any], 
                             feature_names: List[str],
                             hours: int = 24) -> Dict[str, Any]:
    """
    Generate dynamic explainability over time (Explainability-Over-Time)
    
    Args:
        patient_data: Patient data dictionary
        feature_names: List of feature names
        hours: Number of hours to analyze
        
    Returns:
        Dictionary with temporal explainability data
    """
    try:
        # Generate hourly feature importance
        feature_importance_over_time = {}
        top_features_over_time = {}
        
        for hour in range(hours):
            # Simulate temporal changes in patient data
            temporal_data = _simulate_temporal_changes(patient_data, hour, feature_names)
            
            # Calculate simulated SHAP values for this hour
            feature_importance = np.abs(temporal_data)
            
            feature_importance_over_time[hour] = feature_importance
            top_features_over_time[hour] = _get_top_features(feature_importance, feature_names, 5)
        
        return {
            'feature_importance_over_time': feature_importance_over_time,
            'top_features_over_time': top_features_over_time,
            'hours': list(range(hours)),
            'feature_names': feature_names
        }
    except Exception as e:
        return {"error": f"Dynamic explainability failed: {e}"}


def _simulate_temporal_changes(patient_data: Dict[str, Any], hour: int, feature_names: List[str]) -> np.ndarray:
    """Simulate how patient data changes over time"""
    # Convert patient data to feature vector
    feature_vector = np.zeros(len(feature_names))
    
    # Map patient data to features (simplified mapping)
    feature_mapping = {
        'heart_rate': 0, 'map': 1, 'temperature': 2, 'respiratory_rate': 3,
        'lactate': 4, 'wbc': 5, 'creatinine': 6, 'age': 7
    }
    
    for key, value in patient_data.items():
        if key in feature_mapping and feature_mapping[key] < len(feature_names):
            idx = feature_mapping[key]
            # Add temporal variation based on hour
            temporal_factor = 1 + 0.1 * np.sin(hour * np.pi / 12)  # 12-hour cycle
            feature_vector[idx] = value * temporal_factor
    
    return feature_vector


def _get_top_features(importance_scores: np.ndarray, feature_names: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
    """Get top K features by importance"""
    if len(importance_scores) != len(feature_names):
        return []
    
    # Create feature-importance pairs
    feature_importance = [(feature_names[i], importance_scores[i]) 
                        for i in range(len(feature_names))]
    
    # Sort by importance and return top K
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    return feature_importance[:top_k]


def generate_case_narrative(patient_data: Dict[str, Any], 
                          predictions: Dict[str, Any], 
                          feature_importance: Dict[str, Any]) -> str:
    """
    Generate human-readable case narrative
    
    Args:
        patient_data: Patient data dictionary
        predictions: Model predictions
        feature_importance: Feature importance data
        
    Returns:
        Human-readable narrative string
    """
    try:
        # Extract key information
        risk_score = predictions.get('risk_score', 0.0)
        risk_level = predictions.get('risk_level', 'low')
        
        # Get top features
        top_features = []
        if 'shap' in feature_importance and 'feature_importance' in feature_importance['shap']:
            top_features = feature_importance['shap']['feature_importance'].head(3)
        
        # Clinical terms mapping
        clinical_terms = {
            'heart_rate': 'Heart Rate',
            'map': 'Mean Arterial Pressure',
            'temperature': 'Temperature',
            'respiratory_rate': 'Respiratory Rate',
            'lactate': 'Lactate',
            'wbc': 'White Blood Cell Count',
            'creatinine': 'Creatinine'
        }
        
        # Generate narrative based on risk level
        if risk_level == 'high':
            narrative = f"🚨 **High Sepsis Risk Detected**: "
            if not top_features.empty:
                top_feature = top_features.iloc[0]
                feature_name = clinical_terms.get(top_feature['feature'], top_feature['feature'])
                narrative += f"Rising {feature_name} ({top_feature['importance']:.3f}) "
                
                if len(top_features) > 1:
                    second_feature = top_features.iloc[1]
                    second_name = clinical_terms.get(second_feature['feature'], second_feature['feature'])
                    narrative += f"combined with elevated {second_name} "
            
            narrative += f"suggests sepsis onset within 4-6 hours. Risk score: {risk_score:.3f}. Immediate intervention recommended."
            
        elif risk_level == 'medium':
            narrative = f"⚠️ **Moderate Sepsis Risk**: "
            if not top_features.empty:
                top_feature = top_features.iloc[0]
                feature_name = clinical_terms.get(top_feature['feature'], top_feature['feature'])
                narrative += f"Elevated {feature_name} ({top_feature['importance']:.3f}) "
                
                if len(top_features) > 1:
                    second_feature = top_features.iloc[1]
                    second_name = clinical_terms.get(second_feature['feature'], second_feature['feature'])
                    narrative += f"with concerning trends in {second_name}. "
            
            narrative += f"Risk score: {risk_score:.3f}. Monitor closely for 2-4 hours."
            
        else:  # low risk
            narrative = f"✅ **Low Sepsis Risk**: "
            if not top_features.empty:
                top_feature = top_features.iloc[0]
                feature_name = clinical_terms.get(top_feature['feature'], top_feature['feature'])
                narrative += f"Stable {feature_name} ({top_feature['importance']:.3f}) "
            
            narrative += f"with normal ranges. Risk score: {risk_score:.3f}. Continue routine monitoring."
        
        return narrative
        
    except Exception as e:
        return f"❌ Error generating narrative: {e}"

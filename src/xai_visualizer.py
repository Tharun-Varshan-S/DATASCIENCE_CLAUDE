# src/xai_visualizer.py
"""
Explainable AI (XAI) Visualization Module
Integrates SHAP and LIME for model interpretability and visual explanations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

try:
    from lime import lime_tabular, lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not available. Install with: pip install lime")

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime

class XAIVisualizer:
    """Advanced XAI visualization with SHAP and LIME integration."""
    
    def __init__(self, model, X_train: np.ndarray, y_train: np.ndarray, 
                 feature_names: Optional[List[str]] = None, 
                 target_names: Optional[List[str]] = None,
                 data_type: str = 'tabular'):
        """
        Initialize XAI Visualizer.
        
        Args:
            model: Trained ML model
            X_train: Training features
            y_train: Training labels
            feature_names: Names of features
            target_names: Names of target classes
            data_type: 'tabular' or 'image'
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        self.target_names = target_names or [str(i) for i in range(len(np.unique(y_train)))]
        self.data_type = data_type
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self._initialize_explainers()
        
    def _initialize_explainers(self):
        """Initialize SHAP and LIME explainers."""
        print("Initializing XAI explainers...")
        
        # Initialize SHAP explainer
        if SHAP_AVAILABLE:
            try:
                if hasattr(self.model, 'predict_proba'):
                    # Use TreeExplainer for tree-based models
                    if hasattr(self.model, 'estimators_') or hasattr(self.model, 'tree_'):
                        self.shap_explainer = shap.TreeExplainer(self.model)
                    else:
                        # Use KernelExplainer for other models
                        self.shap_explainer = shap.KernelExplainer(
                            self.model.predict_proba, 
                            self.X_train[:100]  # Use subset for speed
                        )
                print("✅ SHAP explainer initialized")
            except Exception as e:
                print(f"⚠️ SHAP initialization failed: {e}")
                self.shap_explainer = None
        
        # Initialize LIME explainer
        if LIME_AVAILABLE:
            try:
                if self.data_type == 'tabular':
                    self.lime_explainer = lime_tabular.LimeTabularExplainer(
                        self.X_train,
                        feature_names=self.feature_names,
                        class_names=self.target_names,
                        mode='classification',
                        discretize_continuous=True
                    )
                elif self.data_type == 'image':
                    self.lime_explainer = lime_image.LimeImageExplainer()
                print("✅ LIME explainer initialized")
            except Exception as e:
                print(f"⚠️ LIME initialization failed: {e}")
                self.lime_explainer = None
    
    def explain_prediction_shap(self, X_sample: np.ndarray, sample_idx: int = 0) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single prediction.
        
        Args:
            X_sample: Input sample (1D array for single sample)
            sample_idx: Index of the sample for reference
            
        Returns:
            SHAP explanation results
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return {"error": "SHAP not available"}
        
        try:
            # Ensure X_sample is 2D
            if X_sample.ndim == 1:
                X_sample = X_sample.reshape(1, -1)

            # Compute prediction and probabilities once
            prediction_class = int(self.model.predict(X_sample)[0])
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_sample)[0]
                confidence = float(probabilities[prediction_class])
            else:
                probabilities = None
                confidence = 0.5

            # Get SHAP values (handle different explainer types and multiclass outputs)
            shap_values_raw = self.shap_explainer.shap_values(X_sample)

            # Normalize to a 1D vector for the predicted class
            if isinstance(shap_values_raw, list):
                # Multiclass: list of arrays per class
                shap_values_vec = np.array(shap_values_raw[prediction_class][0])
            else:
                # Binary/classic: array of shape (n_samples, n_features)
                shap_values_vec = np.array(shap_values_raw[0])

            # Determine base value robustly
            base_value = 0.0
            if hasattr(self.shap_explainer, 'expected_value'):
                ev = self.shap_explainer.expected_value
                if isinstance(ev, (list, tuple, np.ndarray)):
                    try:
                        base_value = float(np.array(ev).reshape(-1)[prediction_class])
                    except Exception:
                        # fallback to first element
                        base_value = float(np.array(ev).reshape(-1)[0])
                else:
                    base_value = float(ev)

            # Create feature importance data
            feature_importance = []
            for i, (feature_name, shap_val) in enumerate(zip(self.feature_names, shap_values_vec)):
                shap_val_f = float(shap_val)
                feature_importance.append({
                    'feature': feature_name,
                    'shap_value': shap_val_f,
                    'feature_value': float(X_sample[0, i]),
                    'abs_shap_value': abs(shap_val_f)
                })

            # Sort by absolute SHAP value
            feature_importance.sort(key=lambda x: x['abs_shap_value'], reverse=True)

            explanation = {
                'method': 'SHAP',
                'sample_idx': sample_idx,
                'prediction': prediction_class,
                'predicted_class': self.target_names[prediction_class],
                'confidence': float(confidence),
                'feature_importance': feature_importance[:10],
                'shap_values': shap_values_vec.tolist(),
                'base_value': base_value
            }

            return explanation

        except Exception as e:
            return {"error": f"SHAP explanation failed: {str(e)}"}
    
    def explain_prediction_lime(self, X_sample: np.ndarray, sample_idx: int = 0, 
                               num_features: int = 10) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single prediction.
        
        Args:
            X_sample: Input sample
            sample_idx: Index of the sample for reference
            num_features: Number of top features to return
            
        Returns:
            LIME explanation results
        """
        if not LIME_AVAILABLE or self.lime_explainer is None:
            return {"error": "LIME not available"}
        
        try:
            # Ensure X_sample is 2D
            if X_sample.ndim == 1:
                X_sample = X_sample.reshape(1, -1)
            
            # Get LIME explanation
            explanation = self.lime_explainer.explain_instance(
                X_sample[0], 
                self.model.predict_proba,
                num_features=num_features
            )
            
            # Extract explanation data
            feature_importance = []
            for feature_idx, weight in explanation.as_list():
                # Parse feature name and value
                if '=' in feature_idx:
                    feature_name, feature_value = feature_idx.split('=')
                    feature_value = float(feature_value)
                else:
                    feature_name = feature_idx
                    feature_value = X_sample[0, int(feature_name.split('_')[-1])] if '_' in feature_name else 0.0
                
                feature_importance.append({
                    'feature': feature_name,
                    'lime_weight': float(weight),
                    'feature_value': float(feature_value),
                    'abs_weight': abs(float(weight))
                })
            
            # Get prediction
            prediction = self.model.predict(X_sample)[0]
            if hasattr(self.model, 'predict_proba'):
                probability = self.model.predict_proba(X_sample)[0]
                confidence = probability[prediction]
            else:
                confidence = 0.5
            
            explanation_result = {
                'method': 'LIME',
                'sample_idx': sample_idx,
                'prediction': int(prediction),
                'predicted_class': self.target_names[prediction],
                'confidence': float(confidence),
                'feature_importance': feature_importance,
                'explanation_text': str(explanation)
            }
            
            return explanation_result
            
        except Exception as e:
            return {"error": f"LIME explanation failed: {str(e)}"}
    
    def explain_error_case(self, X_sample: np.ndarray, true_label: int, 
                          predicted_label: int, sample_idx: int = 0) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for an error case.
        
        Args:
            X_sample: Input sample
            true_label: True class label
            predicted_label: Predicted class label
            sample_idx: Index of the sample
            
        Returns:
            Comprehensive error explanation
        """
        print(f"Explaining error case: True={true_label}, Predicted={predicted_label}")
        
        # Get both SHAP and LIME explanations
        shap_explanation = self.explain_prediction_shap(X_sample, sample_idx)
        lime_explanation = self.explain_prediction_lime(X_sample, sample_idx)
        
        # Get prediction confidence
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_sample)[0]
            confidence = probabilities[predicted_label]
            all_probabilities = probabilities.tolist()
        else:
            confidence = 0.5
            all_probabilities = [0.5] * len(self.target_names)
        
        # Create error analysis
        error_analysis = {
            'sample_idx': sample_idx,
            'true_label': int(true_label),
            'true_class': self.target_names[true_label],
            'predicted_label': int(predicted_label),
            'predicted_class': self.target_names[predicted_label],
            'confidence': float(confidence),
            'all_probabilities': all_probabilities,
            'error_type': self._classify_error_type(confidence),
            'shap_explanation': shap_explanation,
            'lime_explanation': lime_explanation,
            'human_readable_explanation': self._generate_human_explanation(
                X_sample, true_label, predicted_label, shap_explanation, lime_explanation
            )
        }
        
        return error_analysis
    
    def _classify_error_type(self, confidence: float) -> str:
        """Classify the type of error based on confidence."""
        if confidence < 0.4:
            return "Low Confidence Error"
        elif confidence < 0.7:
            return "Medium Confidence Error"
        else:
            return "High Confidence Error"
    
    def _generate_human_explanation(self, X_sample: np.ndarray, true_label: int, 
                                   predicted_label: int, shap_explanation: Dict, 
                                   lime_explanation: Dict) -> str:
        """Generate human-readable explanation for the error."""
        
        # Get top contributing features from SHAP
        if 'feature_importance' in shap_explanation and not shap_explanation.get('error'):
            top_features = shap_explanation['feature_importance'][:3]
            feature_text = ", ".join([f"{f['feature']} (value: {f['feature_value']:.2f})" 
                                    for f in top_features])
        else:
            feature_text = "key features"
        
        # Generate explanation based on dataset type
        if self.data_type == 'image':
            explanation = f"""
            The model predicted '{self.target_names[predicted_label]}' instead of '{self.target_names[true_label]}' 
            because it focused on {feature_text}. This suggests the model may be confused by similar 
            visual patterns between these classes.
            """
        else:
            explanation = f"""
            The model predicted '{self.target_names[predicted_label]}' instead of '{self.target_names[true_label]}' 
            based on the values of {feature_text}. The model's decision boundary may not be optimal 
            for distinguishing between these classes.
            """
        
        return explanation.strip()
    
    def visualize_feature_importance(self, explanation: Dict[str, Any], 
                                   save_path: Optional[str] = None) -> str:
        """
        Create visualization of feature importance.
        
        Args:
            explanation: SHAP or LIME explanation result
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        if 'feature_importance' not in explanation or explanation.get('error'):
            return None
        
        # Extract data
        features = [f['feature'] for f in explanation['feature_importance'][:10]]
        values = [f['shap_value'] if 'shap_value' in f else f['lime_weight'] 
                 for f in explanation['feature_importance'][:10]]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color bars based on positive/negative values
        colors = ['red' if v < 0 else 'green' for v in values]
        
        bars = ax.barh(features, values, color=colors, alpha=0.7)
        ax.set_xlabel(f"{explanation['method']} Values")
        ax.set_title(f"Feature Importance - {explanation['method']} Explanation\n"
                    f"Prediction: {explanation['predicted_class']} "
                    f"(Confidence: {explanation['confidence']:.3f})")
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left' if width >= 0 else 'right', va='center')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"results/xai_feature_importance_{explanation['method'].lower()}_{timestamp}.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_error_heatmap(self, error_cases: List[Dict], save_path: Optional[str] = None) -> str:
        """
        Create heatmap showing which classes the model confuses most.
        
        Args:
            error_cases: List of error case explanations
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        # Create confusion matrix for errors
        n_classes = len(self.target_names)
        confusion_matrix = np.zeros((n_classes, n_classes))
        
        for error_case in error_cases:
            true_label = error_case['true_label']
            predicted_label = error_case['predicted_label']
            confusion_matrix[true_label, predicted_label] += 1
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(confusion_matrix, 
                   annot=True, 
                   fmt='.0f',
                   cmap='Reds',
                   xticklabels=self.target_names,
                   yticklabels=self.target_names,
                   ax=ax)
        
        ax.set_title('Error Confusion Matrix\n(True vs Predicted Classes)')
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"results/error_confusion_heatmap_{timestamp}.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_xai_report(self, error_cases: List[Dict], save_path: str = "results/xai_report.txt"):
        """
        Generate comprehensive XAI report for error cases.
        
        Args:
            error_cases: List of error case explanations
            save_path: Path to save the report
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EXPLAINABLE AI (XAI) ERROR ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total error cases analyzed: {len(error_cases)}\n\n")
            
            # Summary statistics
            error_types = {}
            class_confusions = {}
            
            for case in error_cases:
                error_type = case['error_type']
                error_types[error_type] = error_types.get(error_type, 0) + 1
                
                confusion = f"{case['true_class']} → {case['predicted_class']}"
                class_confusions[confusion] = class_confusions.get(confusion, 0) + 1
            
            f.write("ERROR TYPE DISTRIBUTION:\n")
            f.write("-" * 30 + "\n")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{error_type}: {count} cases ({count/len(error_cases)*100:.1f}%)\n")
            
            f.write("\nMOST COMMON CLASS CONFUSIONS:\n")
            f.write("-" * 35 + "\n")
            for confusion, count in sorted(class_confusions.items(), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"{confusion}: {count} cases\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("DETAILED ERROR CASE ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            # Detailed analysis for each error case
            for i, case in enumerate(error_cases[:10], 1):  # Limit to first 10 cases
                f.write(f"--- Error Case {i} ---\n")
                f.write(f"Sample Index: {case['sample_idx']}\n")
                f.write(f"True Class: {case['true_class']}\n")
                f.write(f"Predicted Class: {case['predicted_class']}\n")
                f.write(f"Confidence: {case['confidence']:.3f}\n")
                f.write(f"Error Type: {case['error_type']}\n")
                f.write(f"Human Explanation: {case['human_readable_explanation']}\n")
                
                # SHAP explanation
                if 'shap_explanation' in case and not case['shap_explanation'].get('error'):
                    f.write("\nTop SHAP Features:\n")
                    for j, feature in enumerate(case['shap_explanation']['feature_importance'][:5], 1):
                        f.write(f"  {j}. {feature['feature']}: {feature['shap_value']:.4f}\n")
                
                # LIME explanation
                if 'lime_explanation' in case and not case['lime_explanation'].get('error'):
                    f.write("\nTop LIME Features:\n")
                    for j, feature in enumerate(case['lime_explanation']['feature_importance'][:5], 1):
                        f.write(f"  {j}. {feature['feature']}: {feature['lime_weight']:.4f}\n")
                
                f.write("\n" + "-" * 50 + "\n\n")
        
        print(f"XAI report saved to: {save_path}")

# Example usage
if __name__ == "__main__":
    # This would typically be called from main.py
    from data_loader import DataLoader
    from baseline_model import BaselineModel
    
    # Load data and train model
    loader = DataLoader('digits')
    data = loader.load_dataset()
    
    model = BaselineModel('random_forest')
    model.train(data['X_train'], data['y_train'])
    
    # Initialize XAI visualizer
    xai_visualizer = XAIVisualizer(
        model.model,
        data['X_train'],
        data['y_train'],
        data.get('feature_names'),
        data['target_names'],
        data['data_type']
    )
    
    # Test with a sample
    test_sample = data['X_test'][0:1]
    explanation = xai_visualizer.explain_prediction_shap(test_sample)
    print("SHAP explanation:", explanation)

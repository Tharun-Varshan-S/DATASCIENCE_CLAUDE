# src/explainer.py
"""
Model Explainability Module
Provides explanations for model predictions and errors using various interpretation techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    def __init__(self, model, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str] = None):
        """
        Initialize model explainer.
        
        Args:
            model: Trained ML model
            X_train: Training features
            y_train: Training labels
            feature_names: Names of features
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        self.explanations = {}
        
        print(f"Model explainer initialized for {type(model).__name__}")
        
    def explain_prediction(self, 
                          X_sample: np.ndarray, 
                          sample_idx: int = 0,
                          explanation_methods: List[str] = None) -> Dict[str, Any]:
        """
        Explain a single prediction using multiple methods.
        
        Args:
            X_sample: Input sample(s)
            sample_idx: Index of sample to explain (if multiple samples)
            explanation_methods: List of methods to use
            
        Returns:
            Explanation results
        """
        if explanation_methods is None:
            explanation_methods = ['feature_importance', 'local_surrogate', 'counterfactual']
        
        # Ensure X_sample is 2D
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
        
        sample = X_sample[sample_idx:sample_idx+1]
        
        # Get prediction
        prediction = self.model.predict(sample)[0]
        prediction_proba = None
        if hasattr(self.model, 'predict_proba'):
            prediction_proba = self.model.predict_proba(sample)[0]
        
        explanation = {
            'sample_index': sample_idx,
            'sample_features': sample.flatten().tolist(),
            'prediction': int(prediction),
            'prediction_probability': prediction_proba.tolist() if prediction_proba is not None else None,
            'explanations': {}
        }
        
        explanation = {
            'sample_index': sample_idx,
            'sample_features': sample.flatten().tolist(),
            'prediction': int(prediction),
            'prediction_probability': prediction_proba.tolist() if prediction_proba is not None else None,
            'explanations': {}
        }
        
        # Apply explanation methods
        for method in explanation_methods:
            try:
                if method == 'feature_importance':
                    explanation['explanations']['feature_importance'] = self._explain_feature_importance(sample)
                elif method == 'local_surrogate':
                    explanation['explanations']['local_surrogate'] = self._explain_local_surrogate(sample)
                elif method == 'counterfactual':
                    explanation['explanations']['counterfactual'] = self._explain_counterfactual(sample)
                elif method == 'lime':
                    explanation['explanations']['lime'] = self._explain_lime(sample)
                elif method == 'shap':
                    explanation['explanations']['shap'] = self._explain_shap(sample)
            except Exception as e:
                explanation['explanations'][method] = {'error': str(e)}
        
        return explanation
    
    def explain_errors(self, 
                      X_errors: np.ndarray, 
                      y_true_errors: np.ndarray, 
                      y_pred_errors: np.ndarray,
                      max_explanations: int = 10) -> Dict[str, Any]:
        """
        Explain why specific predictions were errors.
        
        Args:
            X_errors: Features of error samples
            y_true_errors: True labels of error samples
            y_pred_errors: Predicted labels of error samples
            max_explanations: Maximum number of errors to explain
            
        Returns:
            Error explanations
        """
        print(f"Explaining {min(len(X_errors), max_explanations)} prediction errors...")
        
        error_explanations = {
            'n_errors': len(X_errors),
            'n_explained': min(len(X_errors), max_explanations),
            'explanations': [],
            'common_patterns': {}
        }
        
        # Explain individual errors
        for i in range(min(len(X_errors), max_explanations)):
            sample = X_errors[i:i+1]
            true_label = y_true_errors[i]
            pred_label = y_pred_errors[i]
            
            explanation = self.explain_prediction(sample, 0, ['feature_importance', 'local_surrogate'])
            
            error_explanation = {
                'error_index': i,
                'true_label': int(true_label),
                'predicted_label': int(pred_label),
                'confidence': explanation['prediction_probability'][pred_label] if explanation['prediction_probability'] else None,
                'feature_values': explanation['sample_features'],
                'explanations': explanation['explanations'],
                'error_type': self._classify_error_type(explanation, true_label, pred_label)
            }
            
            error_explanations['explanations'].append(error_explanation)
        
        # Find common patterns in errors
        error_explanations['common_patterns'] = self._find_error_patterns(
            X_errors[:max_explanations], 
            y_true_errors[:max_explanations], 
            y_pred_errors[:max_explanations]
        )
        
        return error_explanations
    
    def _explain_feature_importance(self, sample: np.ndarray) -> Dict[str, Any]:
        """Explain prediction using global feature importance."""
        
        # Get model feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use coefficient magnitudes
            if self.model.coef_.ndim > 1:
                importances = np.mean(np.abs(self.model.coef_), axis=0)
            else:
                importances = np.abs(self.model.coef_)
        else:
            # Use permutation importance as fallback
            perm_importance = permutation_importance(
                self.model, self.X_train, self.y_train, 
                n_repeats=5, random_state=42, n_jobs=-1
            )
            importances = perm_importance.importances_mean
        
        # Get feature contributions for this sample
        feature_contributions = importances * sample.flatten()
        
        # Sort by absolute contribution
        feature_indices = np.argsort(np.abs(feature_contributions))[::-1]
        
        top_features = []
        for idx in feature_indices[:10]:  # Top 10 features
            top_features.append({
                'feature_name': self.feature_names[idx],
                'feature_index': int(idx),
                'feature_value': float(sample.flatten()[idx]),
                'importance': float(importances[idx]),
                'contribution': float(feature_contributions[idx])
            })
        
        return {
            'method': 'feature_importance',
            'top_features': top_features,
            'total_positive_contribution': float(np.sum(feature_contributions[feature_contributions > 0])),
            'total_negative_contribution': float(np.sum(feature_contributions[feature_contributions < 0]))
        }
    
    def _explain_local_surrogate(self, sample: np.ndarray) -> Dict[str, Any]:
        """Explain prediction using a local surrogate model (decision tree)."""
        
        # Generate local neighborhood around the sample
        neighborhood_size = min(1000, len(self.X_train))
        
        # Create perturbations around the sample
        noise_std = np.std(self.X_train, axis=0) * 0.1  # 10% of feature std
        perturbations = np.random.normal(0, noise_std, size=(neighborhood_size, sample.shape[1]))
        neighborhood = sample + perturbations
        
        # Get predictions for the neighborhood
        neighborhood_predictions = self.model.predict(neighborhood)
        
        # Train a simple decision tree on the neighborhood
        surrogate = DecisionTreeClassifier(max_depth=5, random_state=42)
        surrogate.fit(neighborhood, neighborhood_predictions)
        
        # Get the decision path for our sample
        sample_prediction = surrogate.predict(sample)[0]
        
        # Extract decision rules
        tree_rules = export_text(
            surrogate, 
            feature_names=self.feature_names,
            max_depth=3,
            decimals=3
        )
        
        # Get feature importance from surrogate
        surrogate_importance = surrogate.feature_importances_
        important_features = []
        
        for idx, importance in enumerate(surrogate_importance):
            if importance > 0.01:  # Only include features with >1% importance
                important_features.append({
                    'feature_name': self.feature_names[idx],
                    'feature_index': int(idx),
                    'feature_value': float(sample.flatten()[idx]),
                    'surrogate_importance': float(importance)
                })
        
        # Sort by importance
        important_features.sort(key=lambda x: x['surrogate_importance'], reverse=True)
        
        return {
            'method': 'local_surrogate',
            'surrogate_prediction': int(sample_prediction),
            'neighborhood_size': neighborhood_size,
            'important_features': important_features[:5],  # Top 5
            'decision_rules': tree_rules.split('\n')[:10],  # First 10 lines
            'surrogate_accuracy': float(np.mean(
                surrogate.predict(neighborhood) == neighborhood_predictions
            ))
        }
    
    def _explain_counterfactual(self, sample: np.ndarray) -> Dict[str, Any]:
        """Generate counterfactual explanations."""
        
        original_prediction = self.model.predict(sample)[0]
        sample_flat = sample.flatten()
        
        # Find nearby samples with different predictions in training data
        distances = np.sum((self.X_train - sample_flat) ** 2, axis=1)
        nearest_indices = np.argsort(distances)
        
        counterfactuals = []
        
        # Look for nearest neighbors with different predictions
        for idx in nearest_indices[:100]:  # Check 100 nearest neighbors
            neighbor = self.X_train[idx]
            neighbor_prediction = self.model.predict(neighbor.reshape(1, -1))[0]
            
            if neighbor_prediction != original_prediction:
                # Calculate what features changed
                feature_changes = []
                for i, (orig_val, new_val) in enumerate(zip(sample_flat, neighbor)):
                    if abs(orig_val - new_val) > 1e-6:  # Feature changed
                        feature_changes.append({
                            'feature_name': self.feature_names[i],
                            'feature_index': i,
                            'original_value': float(orig_val),
                            'counterfactual_value': float(new_val),
                            'change': float(new_val - orig_val)
                        })
                
                counterfactuals.append({
                    'counterfactual_prediction': int(neighbor_prediction),
                    'distance': float(distances[idx]),
                    'n_features_changed': len(feature_changes),
                    'feature_changes': sorted(feature_changes, 
                                            key=lambda x: abs(x['change']), 
                                            reverse=True)[:5]  # Top 5 changes
                })
                
                if len(counterfactuals) >= 3:  # Limit to 3 counterfactuals
                    break
        
        return {
            'method': 'counterfactual',
            'original_prediction': int(original_prediction),
            'counterfactuals': counterfactuals
        }
    
    def _explain_lime(self, sample: np.ndarray) -> Dict[str, Any]:
        """Explain using LIME (if available)."""
        try:
            import lime
            import lime.lime_tabular
            
            explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train,
                feature_names=self.feature_names,
                mode='classification',
                discretize_continuous=True
            )
            
            explanation = explainer.explain_instance(
                sample.flatten(),
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                num_features=10
            )
            
            # Extract LIME results
            lime_features = []
            for feature, weight in explanation.as_list():
                lime_features.append({
                    'feature_description': feature,
                    'weight': float(weight)
                })
            
            return {
                'method': 'lime',
                'lime_features': lime_features,
                'lime_score': float(explanation.score),
                'intercept': float(explanation.intercept[explanation.predicted_label]) if hasattr(explanation, 'intercept') else None
            }
            
        except ImportError:
            return {'method': 'lime', 'error': 'LIME not available. Install with: pip install lime'}
        except Exception as e:
            return {'method': 'lime', 'error': f'LIME explanation failed: {str(e)}'}
    
    def _explain_shap(self, sample: np.ndarray) -> Dict[str, Any]:
        """Explain using SHAP (if available)."""
        try:
            import shap
            
            # Choose appropriate SHAP explainer
            if hasattr(self.model, 'predict_proba'):
                # For tree-based models
                if hasattr(self.model, 'estimators_') or 'forest' in str(type(self.model)).lower():
                    explainer = shap.TreeExplainer(self.model)
                else:
                    # Use KernelExplainer as fallback
                    explainer = shap.KernelExplainer(
                        self.model.predict_proba, 
                        shap.sample(self.X_train, 100)  # Use sample for speed
                    )
            else:
                explainer = shap.KernelExplainer(
                    self.model.predict,
                    shap.sample(self.X_train, 100)
                )
            
            # Get SHAP values
            shap_values = explainer.shap_values(sample)
            
            if isinstance(shap_values, list):
                # Multi-class case - use values for predicted class
                predicted_class = self.model.predict(sample)[0]
                shap_values_class = shap_values[predicted_class][0]
            else:
                shap_values_class = shap_values[0]
            
            # Create feature importance from SHAP values
            shap_features = []
            for i, shap_value in enumerate(shap_values_class):
                shap_features.append({
                    'feature_name': self.feature_names[i],
                    'feature_index': i,
                    'feature_value': float(sample.flatten()[i]),
                    'shap_value': float(shap_value)
                })
            
            # Sort by absolute SHAP value
            shap_features.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            
            return {
                'method': 'shap',
                'shap_features': shap_features[:10],  # Top 10
                'expected_value': float(explainer.expected_value) if hasattr(explainer, 'expected_value') else None
            }
            
        except ImportError:
            return {'method': 'shap', 'error': 'SHAP not available. Install with: pip install shap'}
        except Exception as e:
            return {'method': 'shap', 'error': f'SHAP explanation failed: {str(e)}'}
    
    def _classify_error_type(self, explanation: Dict, true_label: int, pred_label: int) -> str:
        """Classify the type of error based on explanation."""
        
        prediction_proba = explanation.get('prediction_probability')
        
        if prediction_proba is not None and len(prediction_proba) > 0:
            confidence = prediction_proba[pred_label]
            
            if confidence < 0.6:
                return 'low_confidence_error'
            elif confidence > 0.8:
                return 'high_confidence_error'
            else:
                return 'medium_confidence_error'
        
        return 'unknown_error'
    
    def _find_error_patterns(self, 
                            X_errors: np.ndarray, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray) -> Dict[str, Any]:
        """Find common patterns in prediction errors."""
        
        patterns = {
            'feature_ranges': {},
            'common_confusions': {},
            'feature_correlations': {}
        }
        
        # Analyze feature ranges where errors occur
        for i, feature_name in enumerate(self.feature_names):
            if i < X_errors.shape[1]:
                feature_values = X_errors[:, i]
                patterns['feature_ranges'][feature_name] = {
                    'mean': float(np.mean(feature_values)),
                    'std': float(np.std(feature_values)),
                    'min': float(np.min(feature_values)),
                    'max': float(np.max(feature_values)),
                    'percentile_25': float(np.percentile(feature_values, 25)),
                    'percentile_75': float(np.percentile(feature_values, 75))
                }
        
        # Common confusion patterns
        confusion_pairs = {}
        for true, pred in zip(y_true, y_pred):
            pair = (int(true), int(pred))
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
        # Sort by frequency
        sorted_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        patterns['common_confusions'] = [
            {'true_label': pair[0], 'predicted_label': pair[1], 'frequency': count}
            for pair, count in sorted_confusions[:5]
        ]
        
        return patterns
    
    def visualize_explanation(self, explanation: Dict, save_path: str = None) -> None:
        """Visualize explanation results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Feature importance
        if 'feature_importance' in explanation['explanations']:
            fi_exp = explanation['explanations']['feature_importance']
            if 'error' not in fi_exp:
                ax = axes[0, 0]
                features = fi_exp['top_features'][:8]  # Top 8 features
                names = [f['feature_name'] for f in features]
                contributions = [f['contribution'] for f in features]
                colors = ['green' if c > 0 else 'red' for c in contributions]
                
                bars = ax.barh(names, contributions, color=colors, alpha=0.7)
                ax.set_title('Feature Contributions')
                ax.set_xlabel('Contribution')
                ax.grid(True, alpha=0.3)
        
        # 2. Local surrogate importance
        if 'local_surrogate' in explanation['explanations']:
            ls_exp = explanation['explanations']['local_surrogate']
            if 'error' not in ls_exp:
                ax = axes[0, 1]
                features = ls_exp['important_features'][:6]
                if features:
                    names = [f['feature_name'] for f in features]
                    importances = [f['surrogate_importance'] for f in features]
                    
                    bars = ax.bar(names, importances, alpha=0.7, color='blue')
                    ax.set_title('Local Surrogate Importance')
                    ax.set_ylabel('Importance')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3)
        
        # 3. Feature values
        ax = axes[1, 0]
        sample_features = explanation['sample_features']
        if len(sample_features) <= 20:  # Only for reasonable number of features
            feature_indices = range(len(sample_features))
            ax.bar(feature_indices, sample_features, alpha=0.7, color='orange')
            ax.set_title('Feature Values')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Feature Value')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'Too many features to display\n({len(sample_features)} features)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Values')
        
        # 4. Prediction confidence
        ax = axes[1, 1]
        if explanation['prediction_probability'] is not None and len(explanation['prediction_probability']) > 0:
            proba = explanation['prediction_probability']
            classes = range(len(proba))
            predicted_class = explanation['prediction']
            
            colors = ['gold' if i == predicted_class else 'lightblue' for i in classes]
            bars = ax.bar(classes, proba, color=colors, alpha=0.7)
            ax.set_title('Prediction Probabilities')
            ax.set_xlabel('Class')
            ax.set_ylabel('Probability')
            ax.grid(True, alpha=0.3)
            
            # Highlight predicted class
            if predicted_class < len(bars):
                bars[predicted_class].set_edgecolor('red')
                bars[predicted_class].set_linewidth(2)
        else:
            ax.text(0.5, 0.5, 'Prediction probabilities\nnot available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Prediction Probabilities')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Explanation visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_explanation_report(self, 
                                   explanations: List[Dict], 
                                   save_path: str = 'explanation_report.txt') -> str:
        """Generate a human-readable explanation report."""
        
        report = []
        report.append("=" * 60)
        report.append("MODEL PREDICTION EXPLANATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated for {len(explanations)} predictions\n")
        
        for i, explanation in enumerate(explanations[:5]):  # Limit to first 5
            report.append(f"--- Explanation {i+1} ---")
            report.append(f"Prediction: Class {explanation['prediction']}")
            
            if explanation['prediction_probability'] is not None and len(explanation['prediction_probability']) > 0:
                confidence = max(explanation['prediction_probability'])
                report.append(f"Confidence: {confidence:.3f}")
            
            # Feature importance explanation
            if 'feature_importance' in explanation['explanations']:
                fi_exp = explanation['explanations']['feature_importance']
                if 'error' not in fi_exp:
                    report.append("\nMost Important Features:")
                    for feature in fi_exp['top_features'][:5]:
                        report.append(f"  • {feature['feature_name']}: {feature['feature_value']:.3f} "
                                    f"(contribution: {feature['contribution']:+.3f})")
            
            # Local surrogate explanation
            if 'local_surrogate' in explanation['explanations']:
                ls_exp = explanation['explanations']['local_surrogate']
                if 'error' not in ls_exp and ls_exp['important_features']:
                    report.append(f"\nLocal Decision Factors:")
                    for feature in ls_exp['important_features'][:3]:
                        report.append(f"  • {feature['feature_name']}: {feature['feature_value']:.3f}")
            
            # Counterfactual explanation
            if 'counterfactual' in explanation['explanations']:
                cf_exp = explanation['explanations']['counterfactual']
                if 'error' not in cf_exp and cf_exp['counterfactuals']:
                    report.append(f"\nTo change prediction:")
                    cf = cf_exp['counterfactuals'][0]  # First counterfactual
                    for change in cf['feature_changes'][:3]:
                        direction = "increase" if change['change'] > 0 else "decrease"
                        report.append(f"  • {direction} {change['feature_name']} "
                                    f"by {abs(change['change']):.3f}")
            
            report.append("")
        
        report_text = "\n".join(report)
        
        # Save to file
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"Explanation report saved to {save_path}")
        return report_text

# Example usage
if __name__ == "__main__":
    print("Model explainer initialized")
    print("This module provides comprehensive model explanation capabilities")
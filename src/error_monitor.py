# src/error_monitor.py
"""
Error Monitoring Module
Monitors model predictions and logs errors for analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import json
import os
from datetime import datetime
from loguru import logger
from sklearn.metrics import confusion_matrix
import pickle

class ErrorMonitor:
    def __init__(self, log_dir='logs'):
        """
        Initialize error monitor.
        
        Args:
            log_dir (str): Directory to store error logs
        """
        self.log_dir = log_dir
        self.errors = []
        self.predictions_log = []
        self.error_stats = {}
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logger
        log_file = os.path.join(log_dir, 'error_monitor.log')
        logger.add(log_file, rotation="10 MB", retention="10 days")
        
        print(f"Error monitor initialized. Logs will be saved to {log_dir}")
    
    def monitor_predictions(self, 
                          X_test: np.ndarray,
                          y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          y_proba: np.ndarray,
                          model_name: str = "baseline") -> Dict[str, Any]:
        """
        Monitor predictions and identify errors.
        
        Args:
            X_test: Test features
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            model_name: Name of the model being monitored
            
        Returns:
            Error monitoring results
        """
        print(f"Monitoring predictions for {model_name}...")
        
        # Identify misclassifications
        error_mask = y_true != y_pred
        error_indices = np.where(error_mask)[0]
        
        # Get confidence scores
        max_probabilities = np.max(y_proba, axis=1)
        predicted_class_probs = y_proba[np.arange(len(y_pred)), y_pred]
        
        # Log all predictions
        timestamp = datetime.now().isoformat()
        
        for i in range(len(y_true)):
            prediction_entry = {
                'timestamp': timestamp,
                'model': model_name,
                'sample_index': i,
                'true_label': int(y_true[i]),
                'predicted_label': int(y_pred[i]),
                'confidence': float(max_probabilities[i]),
                'predicted_class_prob': float(predicted_class_probs[i]),
                'is_error': bool(error_mask[i]),
                'features': X_test[i].tolist() if X_test[i].size < 50 else None  # Store features for small datasets
            }
            self.predictions_log.append(prediction_entry)
        
        # Process errors in detail
        error_details = []
        for idx in error_indices:
            error_info = {
                'error_id': f"{model_name}_{timestamp}_{idx}",
                'timestamp': timestamp,
                'model': model_name,
                'sample_index': idx,
                'true_label': int(y_true[idx]),
                'predicted_label': int(y_pred[idx]),
                'confidence': float(max_probabilities[idx]),
                'predicted_class_prob': float(predicted_class_probs[idx]),
                'probability_distribution': y_proba[idx].tolist(),
                'features': X_test[idx].tolist() if X_test[idx].size < 50 else X_test[idx][:10].tolist(),
                'error_magnitude': abs(int(y_true[idx]) - int(y_pred[idx])),
                'is_low_confidence': max_probabilities[idx] < 0.6,
                'is_high_confidence_error': max_probabilities[idx] > 0.8,
            }
            
            error_details.append(error_info)
            self.errors.append(error_info)
        
        # Calculate error statistics
        n_errors = len(error_indices)
        error_rate = n_errors / len(y_true)
        
        # Confidence-based statistics
        low_confidence_errors = sum(1 for e in error_details if e['is_low_confidence'])
        high_confidence_errors = sum(1 for e in error_details if e['is_high_confidence_error'])
        
        # Per-class error analysis
        unique_classes = np.unique(y_true)
        class_error_rates = {}
        
        for class_label in unique_classes:
            class_mask = y_true == class_label
            class_errors = np.sum(error_mask & class_mask)
            class_total = np.sum(class_mask)
            class_error_rates[int(class_label)] = {
                'error_count': int(class_errors),
                'total_count': int(class_total),
                'error_rate': float(class_errors / class_total) if class_total > 0 else 0.0
            }
        
        # Confusion matrix for detailed error analysis
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        monitoring_results = {
            'model_name': model_name,
            'timestamp': timestamp,
            'total_samples': len(y_true),
            'total_errors': n_errors,
            'error_rate': error_rate,
            'low_confidence_errors': low_confidence_errors,
            'high_confidence_errors': high_confidence_errors,
            'class_error_rates': class_error_rates,
            'confusion_matrix': conf_matrix.tolist(),
            'error_details': error_details,
            'average_confidence': float(np.mean(max_probabilities)),
            'confidence_std': float(np.std(max_probabilities)),
            'min_confidence': float(np.min(max_probabilities)),
            'max_confidence': float(np.max(max_probabilities))
        }
        
        # Log summary
        logger.info(f"Model: {model_name}")
        logger.info(f"Total samples: {len(y_true)}")
        logger.info(f"Total errors: {n_errors} ({error_rate:.4f})")
        logger.info(f"Low confidence errors: {low_confidence_errors}")
        logger.info(f"High confidence errors: {high_confidence_errors}")
        
        # Save monitoring results
        self._save_monitoring_results(monitoring_results)
        
        print(f"Error monitoring completed:")
        print(f"  - Total errors: {n_errors}/{len(y_true)} ({error_rate:.4f})")
        print(f"  - Low confidence errors: {low_confidence_errors}")
        print(f"  - High confidence errors: {high_confidence_errors}")
        print(f"  - Average confidence: {np.mean(max_probabilities):.4f}")
        
        return monitoring_results
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all monitored errors."""
        if not self.errors:
            return {"message": "No errors recorded yet"}
        
        total_errors = len(self.errors)
        
        # Error by model
        model_errors = {}
        for error in self.errors:
            model = error['model']
            if model not in model_errors:
                model_errors[model] = 0
            model_errors[model] += 1
        
        # Confidence distribution
        confidences = [error['confidence'] for error in self.errors]
        low_confidence_count = sum(1 for c in confidences if c < 0.6)
        high_confidence_count = sum(1 for c in confidences if c > 0.8)
        
        # Most common error types
        error_patterns = {}
        for error in self.errors:
            pattern = f"{error['true_label']} -> {error['predicted_label']}"
            if pattern not in error_patterns:
                error_patterns[pattern] = 0
            error_patterns[pattern] += 1
        
        # Sort error patterns by frequency
        sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
        
        summary = {
            'total_errors': total_errors,
            'errors_by_model': model_errors,
            'confidence_distribution': {
                'low_confidence': low_confidence_count,
                'medium_confidence': total_errors - low_confidence_count - high_confidence_count,
                'high_confidence': high_confidence_count
            },
            'most_common_error_patterns': sorted_patterns[:10],
            'average_confidence': np.mean(confidences) if confidences else 0,
            'confidence_std': np.std(confidences) if confidences else 0
        }
        
        return summary
    
    def get_errors_by_criteria(self, 
                              confidence_threshold: float = 0.6,
                              error_type: str = None,
                              model_name: str = None) -> List[Dict]:
        """
        Get errors that match specific criteria.
        
        Args:
            confidence_threshold: Confidence threshold for filtering
            error_type: 'low_confidence', 'high_confidence', or None
            model_name: Filter by specific model name
            
        Returns:
            List of filtered errors
        """
        filtered_errors = []
        
        for error in self.errors:
            # Filter by model
            if model_name and error['model'] != model_name:
                continue
            
            # Filter by confidence
            if error_type == 'low_confidence' and error['confidence'] >= confidence_threshold:
                continue
            if error_type == 'high_confidence' and error['confidence'] <= 0.8:
                continue
            
            filtered_errors.append(error)
        
        return filtered_errors
    
    def identify_outliers(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        """
        Identify outlier samples that might cause errors.
        
        Args:
            X: Feature matrix
            contamination: Expected proportion of outliers
            
        Returns:
            Boolean array indicating outliers
        """
        from sklearn.ensemble import IsolationForest
        
        # Use Isolation Forest for outlier detection
        outlier_detector = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        
        outlier_predictions = outlier_detector.fit_predict(X)
        outliers = outlier_predictions == -1
        
        n_outliers = np.sum(outliers)
        print(f"Identified {n_outliers} outliers ({n_outliers/len(X):.4f})")
        
        return outliers
    
    def detect_data_drift(self, X_reference: np.ndarray, X_current: np.ndarray) -> Dict[str, Any]:
        """
        Detect data drift between reference and current data.
        
        Args:
            X_reference: Reference dataset (e.g., training data)
            X_current: Current dataset (e.g., test data)
            
        Returns:
            Data drift analysis results
        """
        from scipy import stats
        
        drift_results = {
            'n_features': X_reference.shape[1],
            'feature_drifts': [],
            'overall_drift_score': 0,
            'drifted_features': []
        }
        
        # Analyze each feature for drift
        drift_scores = []
        
        for i in range(X_reference.shape[1]):
            ref_feature = X_reference[:, i]
            cur_feature = X_current[:, i]
            
            # Use KS test for drift detection
            ks_stat, p_value = stats.ks_2samp(ref_feature, cur_feature)
            
            # Calculate mean and std differences
            ref_mean, ref_std = np.mean(ref_feature), np.std(ref_feature)
            cur_mean, cur_std = np.mean(cur_feature), np.std(cur_feature)
            
            mean_diff = abs(ref_mean - cur_mean) / (ref_std + 1e-8)
            std_diff = abs(ref_std - cur_std) / (ref_std + 1e-8)
            
            feature_drift = {
                'feature_index': i,
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'mean_difference': mean_diff,
                'std_difference': std_diff,
                'is_drifted': p_value < 0.05  # Significant drift
            }
            
            drift_results['feature_drifts'].append(feature_drift)
            drift_scores.append(ks_stat)
            
            if p_value < 0.05:
                drift_results['drifted_features'].append(i)
        
        drift_results['overall_drift_score'] = np.mean(drift_scores)
        drift_results['n_drifted_features'] = len(drift_results['drifted_features'])
        
        print(f"Data drift analysis:")
        print(f"  - Overall drift score: {drift_results['overall_drift_score']:.4f}")
        print(f"  - Drifted features: {drift_results['n_drifted_features']}/{drift_results['n_features']}")
        
        return drift_results
    
    def _save_monitoring_results(self, monitoring_results):
        """Save monitoring results to JSON file with proper serialization."""
        import numpy as np
        import json
        import os
        from datetime import datetime
        
        def make_json_serializable(obj):
            """Convert numpy types to native Python types recursively."""
            # np.bool was removed; accept Python bool and numpy bool_
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):  # Handle any remaining numpy scalars
                return obj.item()
            else:
                return obj
        
        try:
            # Convert the entire results dictionary
            clean_results = make_json_serializable(monitoring_results)
            
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Ensure log directory exists
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Save to file
            filename = os.path.join(self.log_dir, f"monitoring_results_{timestamp}.json")
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(clean_results, f, indent=2, ensure_ascii=False)
                
            print(f"Monitoring results saved to: {filename}")
            
        except Exception as e:
            print(f"Warning: Could not save monitoring results: {e}")
            print("Continuing without saving...")
            # Don't raise the exception - just continue
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def save_error_log(self, filename: str = None):
        """Save complete error log to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"error_log_{timestamp}.pkl"
        
        filepath = os.path.join(self.log_dir, filename)
        
        error_data = {
            'errors': self.errors,
            'predictions_log': self.predictions_log,
            'error_stats': self.error_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(error_data, f)
        
        print(f"Error log saved to {filepath}")
    
    def load_error_log(self, filepath: str):
        """Load error log from file."""
        with open(filepath, 'rb') as f:
            error_data = pickle.load(f)
        
        self.errors = error_data.get('errors', [])
        self.predictions_log = error_data.get('predictions_log', [])
        self.error_stats = error_data.get('error_stats', {})
        
        print(f"Error log loaded from {filepath}")
        print(f"Loaded {len(self.errors)} errors and {len(self.predictions_log)} prediction logs")

# Example usage
if __name__ == "__main__":
    # This would typically be called from main.py
    monitor = ErrorMonitor()
    
    # Example with dummy data
    y_true = np.array([0, 1, 2, 1, 0, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 2, 0, 0])  # Some errors
    y_proba = np.random.rand(8, 3)
    X_test = np.random.rand(8, 10)
    
    # Monitor predictions
    results = monitor.monitor_predictions(X_test, y_true, y_pred, y_proba, "test_model")
    
    # Get error summary
    summary = monitor.get_error_summary()
    print(f"Error summary: {summary}")
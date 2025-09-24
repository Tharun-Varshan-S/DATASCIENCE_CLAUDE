# src/error_analyzer.py
"""
Error Analysis Module
Categorizes and analyzes different types of errors for targeted mitigation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class ErrorAnalyzer:
    def __init__(self):
        """Initialize error analyzer."""
        self.error_categories = {
            'low_confidence': [],
            'high_confidence': [],
            'class_imbalance': [],
            'data_drift': [],
            'outliers': [],
            'boundary_cases': [],
            'systematic_bias': []
        }
        
        self.analysis_results = {}
        
    def analyze_errors(self, 
                      error_data: List[Dict],
                      X_test: np.ndarray,
                      y_true: np.ndarray,
                      class_distribution: Dict = None) -> Dict[str, Any]:
        """
        Comprehensive error analysis and categorization.
        
        Args:
            error_data: List of error dictionaries from ErrorMonitor
            X_test: Test features
            y_true: True labels
            class_distribution: Training class distribution
            
        Returns:
            Detailed analysis results
        """
        print("Performing comprehensive error analysis...")
        
        if not error_data:
            return {"message": "No errors to analyze"}
        
        # Extract error information
        error_indices = [e['sample_index'] for e in error_data]
        error_features = X_test[error_indices]
        error_true_labels = [e['true_label'] for e in error_data]
        error_pred_labels = [e['predicted_label'] for e in error_data]
        error_confidences = [e['confidence'] for e in error_data]
        
        # 1. Confidence-based categorization
        confidence_analysis = self._analyze_confidence_errors(error_data)
        
        # 2. Class imbalance analysis
        imbalance_analysis = self._analyze_class_imbalance_errors(
            error_data, y_true, class_distribution
        )
        
        # 3. Feature-based error clustering
        cluster_analysis = self._analyze_error_clusters(error_features, error_data)
        
        # 4. Boundary case analysis
        boundary_analysis = self._analyze_boundary_cases(error_data)
        
        # 5. Systematic bias detection
        bias_analysis = self._analyze_systematic_bias(error_data)
        
        # 6. Error pattern analysis
        pattern_analysis = self._analyze_error_patterns(error_data)
        
        # Compile comprehensive analysis
        analysis_results = {
            'total_errors': len(error_data),
            'confidence_analysis': confidence_analysis,
            'class_imbalance_analysis': imbalance_analysis,
            'cluster_analysis': cluster_analysis,
            'boundary_analysis': boundary_analysis,
            'bias_analysis': bias_analysis,
            'pattern_analysis': pattern_analysis,
            'error_categories': self._categorize_errors(error_data),
            'mitigation_recommendations': self._generate_mitigation_recommendations()
        }
        
        self.analysis_results = analysis_results
        
        # Print summary
        self._print_analysis_summary(analysis_results)
        
        return analysis_results
    
    def _analyze_confidence_errors(self, error_data: List[Dict]) -> Dict[str, Any]:
        """Analyze errors based on prediction confidence."""
        confidences = [e['confidence'] for e in error_data]
        
        # Categorize by confidence levels
        low_confidence = [e for e in error_data if e['confidence'] < 0.6]
        medium_confidence = [e for e in error_data if 0.6 <= e['confidence'] <= 0.8]
        high_confidence = [e for e in error_data if e['confidence'] > 0.8]
        
        # Update categories
        self.error_categories['low_confidence'] = low_confidence
        self.error_categories['high_confidence'] = high_confidence
        
        return {
            'total_errors': len(error_data),
            'low_confidence_errors': len(low_confidence),
            'medium_confidence_errors': len(medium_confidence),
            'high_confidence_errors': len(high_confidence),
            'confidence_distribution': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences),
                'percentiles': {
                    '25': np.percentile(confidences, 25),
                    '50': np.percentile(confidences, 50),
                    '75': np.percentile(confidences, 75)
                }
            }
        }
    
    def _analyze_class_imbalance_errors(self, 
                                      error_data: List[Dict],
                                      y_true: np.ndarray,
                                      class_distribution: Dict = None) -> Dict[str, Any]:
        """Analyze errors related to class imbalance."""
        # Count errors per class
        error_by_class = defaultdict(int)
        for error in error_data:
            error_by_class[error['true_label']] += 1
        
        # Calculate class frequencies in true labels
        unique_classes, class_counts = np.unique(y_true, return_counts=True)
        total_samples = len(y_true)
        
        class_analysis = {}
        imbalanced_classes = []
        
        for class_label, count in zip(unique_classes, class_counts):
            class_freq = count / total_samples
            class_errors = error_by_class.get(class_label, 0)
            class_error_rate = class_errors / count if count > 0 else 0
            
            class_analysis[int(class_label)] = {
                'total_samples': int(count),
                'frequency': class_freq,
                'error_count': class_errors,
                'error_rate': class_error_rate,
                'is_minority': class_freq < 0.1,  # Consider <10% as minority
                'is_high_error': class_error_rate > np.mean(list(error_by_class.values())) / np.mean(class_counts)
            }
            
            # Identify imbalanced classes with high error rates
            if class_freq < 0.15 and class_error_rate > 0.2:
                imbalanced_classes.append(class_label)
        
        # Find errors from imbalanced classes
        imbalance_errors = [e for e in error_data if e['true_label'] in imbalanced_classes]
        self.error_categories['class_imbalance'] = imbalance_errors
        
        return {
            'class_analysis': class_analysis,
            'imbalanced_classes': [int(c) for c in imbalanced_classes],
            'imbalance_errors': len(imbalance_errors),
            'most_problematic_class': int(max(error_by_class.keys(), key=error_by_class.get)) if error_by_class else None
        }
    
    def _analyze_error_clusters(self, error_features: np.ndarray, error_data: List[Dict]) -> Dict[str, Any]:
        """Cluster errors to find common patterns in feature space."""
        if len(error_features) < 3:
            return {"message": "Too few errors for clustering analysis"}
        
        # Determine optimal number of clusters (max 5)
        max_clusters = min(5, len(error_features) - 1)
        
        if max_clusters < 2:
            return {"message": "Insufficient errors for clustering"}
        
        # Reduce dimensionality if needed
        if error_features.shape[1] > 50:
            pca = PCA(n_components=min(50, len(error_features) - 1))
            error_features_reduced = pca.fit_transform(error_features)
        else:
            error_features_reduced = error_features
        
        # Perform clustering
        kmeans = KMeans(n_clusters=max_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(error_features_reduced)
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(max_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_errors = [error_data[i] for i, mask in enumerate(cluster_mask) if mask]
            
            if cluster_errors:
                # Analyze cluster characteristics
                cluster_confidences = [e['confidence'] for e in cluster_errors]
                cluster_true_labels = [e['true_label'] for e in cluster_errors]
                cluster_pred_labels = [e['predicted_label'] for e in cluster_errors]
                
                cluster_analysis[cluster_id] = {
                    'size': len(cluster_errors),
                    'avg_confidence': np.mean(cluster_confidences),
                    'most_common_true_class': max(set(cluster_true_labels), key=cluster_true_labels.count),
                    'most_common_pred_class': max(set(cluster_pred_labels), key=cluster_pred_labels.count),
                    'error_indices': [e['sample_index'] for e in cluster_errors]
                }
        
        return {
            'n_clusters': max_clusters,
            'cluster_analysis': cluster_analysis,
            'cluster_labels': cluster_labels.tolist()
        }
    
    def _analyze_boundary_cases(self, error_data: List[Dict]) -> Dict[str, Any]:
        """Identify errors that occur near decision boundaries."""
        # Boundary cases are typically low-medium confidence with similar probabilities for top classes
        boundary_errors = []
        
        for error in error_data:
            prob_dist = error.get('probability_distribution', [])
            if prob_dist:
                # Sort probabilities in descending order
                sorted_probs = sorted(prob_dist, reverse=True)
                
                # Check if top two probabilities are close (boundary case)
                if len(sorted_probs) >= 2:
                    prob_diff = sorted_probs[0] - sorted_probs[1]
                    
                    # Boundary case: small difference between top predictions
                    if prob_diff < 0.3 and error['confidence'] < 0.8:
                        boundary_errors.append(error)
        
        self.error_categories['boundary_cases'] = boundary_errors
        
        return {
            'boundary_errors': len(boundary_errors),
            'boundary_error_rate': len(boundary_errors) / len(error_data) if error_data else 0,
            'avg_confidence': np.mean([e['confidence'] for e in boundary_errors]) if boundary_errors else 0
        }
    
    def _analyze_systematic_bias(self, error_data: List[Dict]) -> Dict[str, Any]:
        """Detect systematic biases in errors."""
        # Analyze confusion patterns
        true_labels = [e['true_label'] for e in error_data]
        pred_labels = [e['predicted_label'] for e in error_data]
        
        # Count confusion pairs
        confusion_pairs = defaultdict(int)
        for true_label, pred_label in zip(true_labels, pred_labels):
            pair = (true_label, pred_label)
            confusion_pairs[pair] += 1
        
        # Find most common confusion patterns
        sorted_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        
        # Detect systematic bias: same misclassification happening frequently
        bias_threshold = max(2, len(error_data) * 0.2)  # At least 20% or 2 errors
        systematic_biases = []
        biased_errors = []
        
        for (true_class, pred_class), count in sorted_confusions:
            if count >= bias_threshold:
                systematic_biases.append({
                    'true_class': true_class,
                    'predicted_class': pred_class,
                    'frequency': count,
                    'percentage': count / len(error_data)
                })
                
                # Add these errors to bias category
                bias_errors = [e for e in error_data 
                             if e['true_label'] == true_class and e['predicted_label'] == pred_class]
                biased_errors.extend(bias_errors)
        
        self.error_categories['systematic_bias'] = biased_errors
        
        return {
            'systematic_biases': systematic_biases,
            'n_biased_errors': len(biased_errors),
            'most_common_confusions': sorted_confusions[:5]
        }
    
    def _analyze_error_patterns(self, error_data: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal and other patterns in errors."""
        # Group errors by time if timestamps are available
        temporal_patterns = {}
        
        # Analyze error magnitude distribution
        error_magnitudes = [e.get('error_magnitude', 0) for e in error_data]
        
        # Analyze confidence vs error relationship
        confidences = [e['confidence'] for e in error_data]
        
        return {
            'error_magnitude_stats': {
                'mean': np.mean(error_magnitudes) if error_magnitudes else 0,
                'std': np.std(error_magnitudes) if error_magnitudes else 0,
                'max': np.max(error_magnitudes) if error_magnitudes else 0
            },
            'confidence_error_correlation': np.corrcoef(confidences, error_magnitudes)[0,1] if len(confidences) > 1 else 0
        }
    
    def _categorize_errors(self, error_data: List[Dict]) -> Dict[str, List]:
        """Categorize all errors into different types."""
        categorized = {}
        
        for category, errors in self.error_categories.items():
            categorized[category] = {
                'count': len(errors),
                'percentage': len(errors) / len(error_data) * 100 if error_data else 0,
                'sample_errors': errors[:3]  # First 3 examples
            }
        
        return categorized
    
    def _generate_mitigation_recommendations(self) -> List[Dict[str, str]]:
        """Generate mitigation strategy recommendations based on error analysis."""
        recommendations = []
        
        # Check each error category and suggest mitigations
        if self.error_categories['low_confidence']:
            recommendations.append({
                'category': 'Low Confidence Errors',
                'strategy': 'Confidence Thresholding',
                'description': 'Reject predictions with confidence below threshold and use human review or ensemble methods.',
                'priority': 'High'
            })
        
        if self.error_categories['high_confidence']:
            recommendations.append({
                'category': 'High Confidence Errors',
                'strategy': 'Model Retraining',
                'description': 'These represent systematic model failures. Retrain with additional data or feature engineering.',
                'priority': 'Critical'
            })
        
        if self.error_categories['class_imbalance']:
            recommendations.append({
                'category': 'Class Imbalance Errors',
                'strategy': 'Data Augmentation & Resampling',
                'description': 'Use SMOTE, oversampling, or synthetic data generation for minority classes.',
                'priority': 'High'
            })
        
        if self.error_categories['boundary_cases']:
            recommendations.append({
                'category': 'Boundary Case Errors',
                'strategy': 'Ensemble Methods',
                'description': 'Use multiple models to improve decision boundaries and reduce ambiguous predictions.',
                'priority': 'Medium'
            })
        
        if self.error_categories['systematic_bias']:
            recommendations.append({
                'category': 'Systematic Bias',
                'strategy': 'Feature Engineering & Active Learning',
                'description': 'Add discriminative features and collect targeted training data for biased patterns.',
                'priority': 'Critical'
            })
        
        return recommendations
    
    def _print_analysis_summary(self, analysis: Dict[str, Any]):
        """Print a summary of the error analysis."""
        print(f"\n=== ERROR ANALYSIS SUMMARY ===")
        print(f"Total Errors Analyzed: {analysis['total_errors']}")
        
        print(f"\n--- Confidence Analysis ---")
        conf_analysis = analysis['confidence_analysis']
        print(f"Low Confidence (<0.6): {conf_analysis['low_confidence_errors']}")
        print(f"Medium Confidence (0.6-0.8): {conf_analysis['medium_confidence_errors']}")
        print(f"High Confidence (>0.8): {conf_analysis['high_confidence_errors']}")
        
        print(f"\n--- Class Imbalance Analysis ---")
        imb_analysis = analysis['class_imbalance_analysis']
        if imb_analysis.get('imbalanced_classes'):
            print(f"Imbalanced Classes: {imb_analysis['imbalanced_classes']}")
            print(f"Imbalance-related Errors: {imb_analysis['imbalance_errors']}")
        else:
            print("No significant class imbalance detected")
        
        print(f"\n--- Error Categories ---")
        categories = analysis['error_categories']
        for category, info in categories.items():
            if info['count'] > 0:
                print(f"{category.replace('_', ' ').title()}: {info['count']} ({info['percentage']:.1f}%)")
        
        print(f"\n--- Mitigation Recommendations ---")
        for i, rec in enumerate(analysis['mitigation_recommendations'], 1):
            print(f"{i}. {rec['strategy']} ({rec['priority']} priority)")
            print(f"   Target: {rec['category']}")
            print(f"   Action: {rec['description']}")
    
    def visualize_error_analysis(self, save_path: str = 'results'):
        """Create visualizations for error analysis."""
        if not self.analysis_results:
            print("No analysis results to visualize. Run analyze_errors() first.")
            return
        
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Error category distribution
        categories = self.analysis_results['error_categories']
        category_names = []
        category_counts = []
        
        for category, info in categories.items():
            if info['count'] > 0:
                category_names.append(category.replace('_', ' ').title())
                category_counts.append(info['count'])
        
        if category_names:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(category_names, category_counts)
            plt.title('Error Distribution by Category')
            plt.xlabel('Error Category')
            plt.ylabel('Number of Errors')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/error_categories.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Confidence distribution
        conf_analysis = self.analysis_results['confidence_analysis']
        conf_counts = [
            conf_analysis['low_confidence_errors'],
            conf_analysis['medium_confidence_errors'],
            conf_analysis['high_confidence_errors']
        ]
        conf_labels = ['Low\n(<0.6)', 'Medium\n(0.6-0.8)', 'High\n(>0.8)']
        
        plt.figure(figsize=(8, 6))
        colors = ['red', 'orange', 'darkred']
        bars = plt.bar(conf_labels, conf_counts, color=colors, alpha=0.7)
        plt.title('Error Distribution by Confidence Level')
        plt.xlabel('Confidence Level')
        plt.ylabel('Number of Errors')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Error analysis visualizations saved to {save_path}/")

# Example usage
if __name__ == "__main__":
    # This would typically be called from main.py
    analyzer = ErrorAnalyzer()
    
    # Example with dummy error data
    dummy_errors = [
        {
            'sample_index': 0,
            'true_label': 0,
            'predicted_label': 1,
            'confidence': 0.4,
            'probability_distribution': [0.4, 0.5, 0.1],
            'error_magnitude': 1
        },
        {
            'sample_index': 1,
            'true_label': 2,
            'predicted_label': 0,
            'confidence': 0.9,
            'probability_distribution': [0.9, 0.05, 0.05],
            'error_magnitude': 2
        }
    ]
    
    X_dummy = np.random.rand(10, 5)
    y_dummy = np.array([0, 1, 2, 1, 0, 2, 1, 0, 1, 2])
    
    # Analyze errors
    results = analyzer.analyze_errors(dummy_errors, X_dummy, y_dummy)
    print("Error analysis completed!")
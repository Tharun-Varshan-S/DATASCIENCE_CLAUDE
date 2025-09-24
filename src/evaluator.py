# src/evaluator.py
"""
Model Evaluation Module
Comprehensive evaluation and comparison of model performance before and after mitigation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import os
from datetime import datetime

class ModelEvaluator:
    def __init__(self, save_dir='results'):
        """
        Initialize model evaluator.
        
        Args:
            save_dir: Directory to save evaluation results
        """
        self.save_dir = save_dir
        self.evaluation_history = []
        
        # Create results directory
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Model evaluator initialized. Results will be saved to {save_dir}")
    
    def comprehensive_evaluation(self,
                                model_before,
                                model_after,
                                X_test: np.ndarray,
                                y_test: np.ndarray,
                                X_train: np.ndarray = None,
                                y_train: np.ndarray = None,
                                target_names: List[str] = None,
                                evaluation_name: str = "evaluation") -> Dict[str, Any]:
        """
        Comprehensive evaluation comparing before and after models.
        
        Args:
            model_before: Model before mitigation
            model_after: Model after mitigation
            X_test, y_test: Test data
            X_train, y_train: Training data (optional, for cross-validation)
            target_names: Class names
            evaluation_name: Name for this evaluation
            
        Returns:
            Comprehensive evaluation results
        """
        print(f"Performing comprehensive evaluation: {evaluation_name}")
        
        timestamp = datetime.now()
        
        # Get predictions
        y_pred_before = model_before.predict(X_test)
        y_pred_after = model_after.predict(X_test)
        
        # Get probabilities if available
        y_proba_before = self._get_probabilities(model_before, X_test)
        y_proba_after = self._get_probabilities(model_after, X_test)
        
        # Basic metrics comparison
        metrics_comparison = self._compare_basic_metrics(
            y_test, y_pred_before, y_pred_after
        )
        
        # Detailed classification reports
        before_report = classification_report(
            y_test, y_pred_before, 
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        
        after_report = classification_report(
            y_test, y_pred_after,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrices
        cm_before = confusion_matrix(y_test, y_pred_before)
        cm_after = confusion_matrix(y_test, y_pred_after)
        
        # Error analysis
        error_analysis = self._analyze_error_changes(
            y_test, y_pred_before, y_pred_after
        )
        
        # Class-wise improvements
        class_improvements = self._analyze_class_improvements(
            y_test, y_pred_before, y_pred_after, target_names
        )
        
        # Cross-validation if training data available
        cv_results = None
        if X_train is not None and y_train is not None:
            cv_results = self._cross_validation_comparison(
                model_before, model_after, X_train, y_train
            )
        
        # Confidence analysis
        confidence_analysis = self._analyze_confidence_changes(
            y_proba_before, y_proba_after, y_test, y_pred_before, y_pred_after
        )
        
        # ROC analysis for binary/multiclass
        roc_analysis = None
        if len(np.unique(y_test)) <= 10:  # Only for reasonable number of classes
            roc_analysis = self._analyze_roc_changes(
                y_test, y_proba_before, y_proba_after
            )
        
        # Compile results
        evaluation_results = {
            'evaluation_name': evaluation_name,
            'timestamp': timestamp.isoformat(),
            'dataset_info': {
                'n_samples': len(X_test),
                'n_features': X_test.shape[1],
                'n_classes': len(np.unique(y_test)),
                'class_distribution': dict(zip(*np.unique(y_test, return_counts=True)))
            },
            'metrics_comparison': metrics_comparison,
            'before_report': before_report,
            'after_report': after_report,
            'confusion_matrices': {
                'before': cm_before.tolist(),
                'after': cm_after.tolist()
            },
            'error_analysis': error_analysis,
            'class_improvements': class_improvements,
            'confidence_analysis': confidence_analysis,
            'cross_validation': cv_results,
            'roc_analysis': roc_analysis
        }
        
        # Save results
        self._save_evaluation_results(evaluation_results, evaluation_name)
        
        # Create visualizations
        self._create_evaluation_plots(evaluation_results, evaluation_name)
        
        # Print summary
        self._print_evaluation_summary(evaluation_results)
        
        # Store in history
        self.evaluation_history.append(evaluation_results)
        
        return evaluation_results
    
    def _get_probabilities(self, model, X: np.ndarray) -> Optional[np.ndarray]:
        """Get prediction probabilities if available."""
        try:
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)
            elif hasattr(model, 'decision_function'):
                # Convert decision function to probabilities
                decision_scores = model.decision_function(X)
                if decision_scores.ndim == 1:
                    # Binary case
                    exp_scores = np.exp(decision_scores)
                    return np.column_stack([1/(1+exp_scores), exp_scores/(1+exp_scores)])
                else:
                    # Multiclass case
                    exp_scores = np.exp(decision_scores)
                    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        except:
            pass
        return None
    
    def _compare_basic_metrics(self, 
                              y_true: np.ndarray,
                              y_pred_before: np.ndarray,
                              y_pred_after: np.ndarray) -> Dict[str, Any]:
        """Compare basic classification metrics."""
        
        # Calculate metrics for both models
        metrics_before = {
            'accuracy': accuracy_score(y_true, y_pred_before),
            'precision': precision_score(y_true, y_pred_before, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred_before, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred_before, average='weighted', zero_division=0)
        }
        
        metrics_after = {
            'accuracy': accuracy_score(y_true, y_pred_after),
            'precision': precision_score(y_true, y_pred_after, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred_after, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred_after, average='weighted', zero_division=0)
        }
        
        # Calculate improvements
        improvements = {}
        for metric in metrics_before.keys():
            improvements[metric] = metrics_after[metric] - metrics_before[metric]
        
        return {
            'before': metrics_before,
            'after': metrics_after,
            'improvements': improvements,
            'relative_improvements': {
                metric: (improvements[metric] / metrics_before[metric] * 100) 
                if metrics_before[metric] > 0 else 0
                for metric in improvements.keys()
            }
        }
    
    def _analyze_error_changes(self,
                              y_true: np.ndarray,
                              y_pred_before: np.ndarray,
                              y_pred_after: np.ndarray) -> Dict[str, Any]:
        """Analyze how errors changed between models."""
        
        # Identify errors
        errors_before = y_true != y_pred_before
        errors_after = y_true != y_pred_after
        
        # Error change analysis
        errors_fixed = errors_before & ~errors_after  # Was error, now correct
        new_errors = ~errors_before & errors_after    # Was correct, now error
        persistent_errors = errors_before & errors_after  # Still an error
        still_correct = ~errors_before & ~errors_after    # Still correct
        
        # Count changes
        n_errors_fixed = np.sum(errors_fixed)
        n_new_errors = np.sum(new_errors)
        n_persistent_errors = np.sum(persistent_errors)
        n_still_correct = np.sum(still_correct)
        
        # Error reduction metrics
        total_errors_before = np.sum(errors_before)
        total_errors_after = np.sum(errors_after)
        error_reduction_rate = (total_errors_before - total_errors_after) / total_errors_before if total_errors_before > 0 else 0
        
        return {
            'errors_fixed': int(n_errors_fixed),
            'new_errors': int(n_new_errors),
            'persistent_errors': int(n_persistent_errors),
            'still_correct': int(n_still_correct),
            'net_error_reduction': int(n_errors_fixed - n_new_errors),
            'error_reduction_rate': float(error_reduction_rate),
            'total_errors_before': int(total_errors_before),
            'total_errors_after': int(total_errors_after),
            'error_indices': {
                'fixed': np.where(errors_fixed)[0].tolist(),
                'new': np.where(new_errors)[0].tolist(),
                'persistent': np.where(persistent_errors)[0].tolist()
            }
        }
    
    def _analyze_class_improvements(self,
                                   y_true: np.ndarray,
                                   y_pred_before: np.ndarray,
                                   y_pred_after: np.ndarray,
                                   target_names: List[str] = None) -> Dict[str, Any]:
        """Analyze improvements per class."""
        
        # Normalize target names to a Python list if provided (handle numpy arrays)
        target_names_list = None
        if target_names is not None:
            try:
                target_names_list = target_names.tolist() if hasattr(target_names, 'tolist') else list(target_names)
            except Exception:
                target_names_list = None
        
        unique_classes = np.unique(y_true)
        class_analysis = {}
        
        for class_id in unique_classes:
            class_mask = y_true == class_id
            class_name = (
                str(target_names_list[class_id])
                if (target_names_list is not None and len(target_names_list) > int(class_id))
                else str(int(class_id))
            )
            
            if np.sum(class_mask) == 0:
                continue
            
            # Class-specific metrics
            class_true = y_true[class_mask]
            class_pred_before = y_pred_before[class_mask]
            class_pred_after = y_pred_after[class_mask]
            
            accuracy_before = accuracy_score(class_true, class_pred_before)
            accuracy_after = accuracy_score(class_true, class_pred_after)
            
            # Precision and recall for this class vs all others
            precision_before = precision_score(y_true == class_id, y_pred_before == class_id, zero_division=0)
            precision_after = precision_score(y_true == class_id, y_pred_after == class_id, zero_division=0)
            
            recall_before = recall_score(y_true == class_id, y_pred_before == class_id, zero_division=0)
            recall_after = recall_score(y_true == class_id, y_pred_after == class_id, zero_division=0)
            
            class_analysis[class_name] = {
                'sample_count': int(np.sum(class_mask)),
                'accuracy_before': float(accuracy_before),
                'accuracy_after': float(accuracy_after),
                'accuracy_improvement': float(accuracy_after - accuracy_before),
                'precision_before': float(precision_before),
                'precision_after': float(precision_after),
                'precision_improvement': float(precision_after - precision_before),
                'recall_before': float(recall_before),
                'recall_after': float(recall_after),
                'recall_improvement': float(recall_after - recall_before)
            }
        
        # Find classes with most improvement
        improvements = [(name, info['accuracy_improvement']) 
                       for name, info in class_analysis.items()]
        improvements.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'per_class_analysis': class_analysis,
            'most_improved_classes': improvements[:3],
            'least_improved_classes': improvements[-3:] if len(improvements) > 3 else []
        }
    
    def _cross_validation_comparison(self,
                                   model_before,
                                   model_after,
                                   X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   cv_folds: int = 5) -> Dict[str, Any]:
        """Compare models using cross-validation."""
        
        try:
            # Cross-validation scores
            cv_scores_before = cross_val_score(model_before, X_train, y_train, cv=cv_folds, scoring='accuracy')
            cv_scores_after = cross_val_score(model_after, X_train, y_train, cv=cv_folds, scoring='accuracy')
            
            return {
                'cv_folds': cv_folds,
                'cv_scores_before': cv_scores_before.tolist(),
                'cv_scores_after': cv_scores_after.tolist(),
                'cv_mean_before': float(cv_scores_before.mean()),
                'cv_mean_after': float(cv_scores_after.mean()),
                'cv_std_before': float(cv_scores_before.std()),
                'cv_std_after': float(cv_scores_after.std()),
                'cv_improvement': float(cv_scores_after.mean() - cv_scores_before.mean()),
                'statistical_significance': self._test_statistical_significance(cv_scores_before, cv_scores_after)
            }
        except Exception as e:
            return {'error': f"Cross-validation failed: {str(e)}"}
    
    def _test_statistical_significance(self, scores_before: np.ndarray, scores_after: np.ndarray) -> Dict[str, Any]:
        """Test statistical significance of improvement."""
        from scipy import stats
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores_after, scores_before)
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'is_significant': bool(p_value < 0.05),
            'confidence_level': 0.95
        }
    
    def _analyze_confidence_changes(self,
                                   y_proba_before: Optional[np.ndarray],
                                   y_proba_after: Optional[np.ndarray],
                                   y_true: np.ndarray,
                                   y_pred_before: np.ndarray,
                                   y_pred_after: np.ndarray) -> Optional[Dict[str, Any]]:
        """Analyze how prediction confidence changed."""
        
        if y_proba_before is None or y_proba_after is None:
            return None
        
        # Get max probabilities (confidence scores)
        confidence_before = np.max(y_proba_before, axis=1)
        confidence_after = np.max(y_proba_after, axis=1)
        
        # Analyze confidence changes
        confidence_improvement = confidence_after - confidence_before
        
        # Separate analysis for correct and incorrect predictions
        correct_before = y_true == y_pred_before
        correct_after = y_true == y_pred_after
        
        analysis = {
            'average_confidence_before': float(np.mean(confidence_before)),
            'average_confidence_after': float(np.mean(confidence_after)),
            'confidence_improvement': float(np.mean(confidence_improvement)),
            'confidence_std_before': float(np.std(confidence_before)),
            'confidence_std_after': float(np.std(confidence_after))
        }
        
        # Confidence for correct predictions
        if np.sum(correct_before) > 0:
            analysis['correct_confidence_before'] = float(np.mean(confidence_before[correct_before]))
        if np.sum(correct_after) > 0:
            analysis['correct_confidence_after'] = float(np.mean(confidence_after[correct_after]))
        
        # Confidence for incorrect predictions
        if np.sum(~correct_before) > 0:
            analysis['incorrect_confidence_before'] = float(np.mean(confidence_before[~correct_before]))
        if np.sum(~correct_after) > 0:
            analysis['incorrect_confidence_after'] = float(np.mean(confidence_after[~correct_after]))
        
        return analysis
    
    def _analyze_roc_changes(self,
                            y_true: np.ndarray,
                            y_proba_before: Optional[np.ndarray],
                            y_proba_after: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Analyze ROC curve changes."""
        
        if y_proba_before is None or y_proba_after is None:
            return None
        
        n_classes = len(np.unique(y_true))
        
        try:
            if n_classes == 2:
                # Binary classification
                auc_before = roc_auc_score(y_true, y_proba_before[:, 1])
                auc_after = roc_auc_score(y_true, y_proba_after[:, 1])
                
                return {
                    'auc_before': float(auc_before),
                    'auc_after': float(auc_after),
                    'auc_improvement': float(auc_after - auc_before)
                }
            else:
                # Multiclass - use ovr (one-vs-rest)
                auc_before = roc_auc_score(y_true, y_proba_before, multi_class='ovr', average='weighted')
                auc_after = roc_auc_score(y_true, y_proba_after, multi_class='ovr', average='weighted')
                
                return {
                    'auc_before': float(auc_before),
                    'auc_after': float(auc_after),
                    'auc_improvement': float(auc_after - auc_before),
                    'multiclass': True
                }
        except Exception as e:
            return {'error': f"ROC analysis failed: {str(e)}"}
    
    def _create_evaluation_plots(self, evaluation_results: Dict, evaluation_name: str):
        """Create visualization plots for the evaluation."""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Metrics comparison bar plot
        plt.subplot(3, 3, 1)
        metrics = evaluation_results['metrics_comparison']
        metric_names = list(metrics['before'].keys())
        before_values = list(metrics['before'].values())
        after_values = list(metrics['after'].values())
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        plt.bar(x - width/2, before_values, width, label='Before', alpha=0.7, color='red')
        plt.bar(x + width/2, after_values, width, label='After', alpha=0.7, color='green')
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Performance Metrics Comparison')
        plt.xticks(x, metric_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Confusion Matrix - Before
        plt.subplot(3, 3, 2)
        cm_before = np.array(evaluation_results['confusion_matrices']['before'])
        sns.heatmap(cm_before, annot=True, fmt='d', cmap='Reds', alpha=0.7)
        plt.title('Confusion Matrix - Before')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # 3. Confusion Matrix - After
        plt.subplot(3, 3, 3)
        cm_after = np.array(evaluation_results['confusion_matrices']['after'])
        sns.heatmap(cm_after, annot=True, fmt='d', cmap='Greens', alpha=0.7)
        plt.title('Confusion Matrix - After')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # 4. Error Analysis
        plt.subplot(3, 3, 4)
        error_analysis = evaluation_results['error_analysis']
        error_categories = ['Errors Fixed', 'New Errors', 'Persistent Errors']
        error_counts = [
            error_analysis['errors_fixed'],
            error_analysis['new_errors'],
            error_analysis['persistent_errors']
        ]
        colors = ['green', 'red', 'orange']
        
        bars = plt.bar(error_categories, error_counts, color=colors, alpha=0.7)
        plt.title('Error Changes Analysis')
        plt.xlabel('Error Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 5. Class-wise Improvements
        plt.subplot(3, 3, 5)
        class_improvements = evaluation_results['class_improvements']
        class_names = list(class_improvements['per_class_analysis'].keys())
        accuracy_improvements = [
            info['accuracy_improvement'] 
            for info in class_improvements['per_class_analysis'].values()
        ]
        
        colors = ['green' if imp > 0 else 'red' for imp in accuracy_improvements]
        bars = plt.bar(range(len(class_names)), accuracy_improvements, color=colors, alpha=0.7)
        plt.title('Per-Class Accuracy Improvements')
        plt.xlabel('Classes')
        plt.ylabel('Accuracy Improvement')
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 6. Confidence Analysis (if available)
        confidence_analysis = evaluation_results.get('confidence_analysis')
        if confidence_analysis:
            plt.subplot(3, 3, 6)
            conf_metrics = ['Average Confidence', 'Correct Predictions', 'Incorrect Predictions']
            before_conf = [
                confidence_analysis.get('average_confidence_before', 0),
                confidence_analysis.get('correct_confidence_before', 0),
                confidence_analysis.get('incorrect_confidence_before', 0)
            ]
            after_conf = [
                confidence_analysis.get('average_confidence_after', 0),
                confidence_analysis.get('correct_confidence_after', 0),
                confidence_analysis.get('incorrect_confidence_after', 0)
            ]
            
            x = np.arange(len(conf_metrics))
            plt.bar(x - width/2, before_conf, width, label='Before', alpha=0.7, color='red')
            plt.bar(x + width/2, after_conf, width, label='After', alpha=0.7, color='green')
            
            plt.title('Confidence Analysis')
            plt.xlabel('Confidence Type')
            plt.ylabel('Confidence Score')
            plt.xticks(x, conf_metrics, rotation=45)
            plt.legend()
        
        # 7. Cross-validation results (if available)
        cv_results = evaluation_results.get('cross_validation')
        if cv_results and 'error' not in cv_results:
            plt.subplot(3, 3, 7)
            cv_before = cv_results['cv_scores_before']
            cv_after = cv_results['cv_scores_after']
            
            folds = range(1, len(cv_before) + 1)
            plt.plot(folds, cv_before, 'o-', label='Before', color='red', alpha=0.7)
            plt.plot(folds, cv_after, 'o-', label='After', color='green', alpha=0.7)
            
            plt.title('Cross-Validation Scores')
            plt.xlabel('Fold')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 8. Improvement summary
        plt.subplot(3, 3, 8)
        improvements = evaluation_results['metrics_comparison']['improvements']
        relative_improvements = evaluation_results['metrics_comparison']['relative_improvements']
        
        # Show relative improvements as percentages
        metric_names = list(relative_improvements.keys())
        rel_improvements = list(relative_improvements.values())
        colors = ['green' if imp > 0 else 'red' for imp in rel_improvements]
        
        bars = plt.bar(metric_names, rel_improvements, color=colors, alpha=0.7)
        plt.title('Relative Improvements (%)')
        plt.xlabel('Metrics')
        plt.ylabel('Improvement (%)')
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, rel_improvements):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + (0.5 if height > 0 else -0.5),
                    f'{value:.1f}%', ha='center', 
                    va='bottom' if height > 0 else 'top')
        
        # 9. Error Reduction Rate
        plt.subplot(3, 3, 9)
        error_analysis = evaluation_results['error_analysis']
        error_reduction_rate = error_analysis['error_reduction_rate'] * 100
        
        # Create a gauge-like visualization
        categories = ['Error Reduction\nRate']
        values = [error_reduction_rate]
        color = 'green' if error_reduction_rate > 0 else 'red'
        
        bars = plt.bar(categories, values, color=color, alpha=0.7)
        plt.title('Error Reduction Rate')
        plt.ylabel('Percentage (%)')
        plt.ylim(-10, max(20, error_reduction_rate + 5))
        
        # Add value label
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + (0.5 if height > 0 else -0.5),
                    f'{value:.1f}%', ha='center', 
                    va='bottom' if height > 0 else 'top',
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.save_dir, f'{evaluation_name}_evaluation.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation plots saved to {plot_path}")
    
    def _print_evaluation_summary(self, evaluation_results: Dict):
        """Print a comprehensive evaluation summary."""
        
        print(f"\n" + "="*60)
        print(f"EVALUATION SUMMARY: {evaluation_results['evaluation_name']}")
        print("="*60)
        
        # Dataset info
        dataset_info = evaluation_results['dataset_info']
        print(f"Dataset: {dataset_info['n_samples']} samples, {dataset_info['n_features']} features, {dataset_info['n_classes']} classes")
        
        # Metrics comparison
        metrics = evaluation_results['metrics_comparison']
        print(f"\nMETRICS COMPARISON:")
        print(f"{'Metric':<12} {'Before':<8} {'After':<8} {'Change':<8} {'Rel. Change'}")
        print("-" * 50)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            before = metrics['before'][metric]
            after = metrics['after'][metric]
            change = metrics['improvements'][metric]
            rel_change = metrics['relative_improvements'][metric]
            
            print(f"{metric.capitalize():<12} {before:<8.4f} {after:<8.4f} {change:+8.4f} {rel_change:+7.1f}%")
        
        # Error analysis
        error_analysis = evaluation_results['error_analysis']
        print(f"\nERROR ANALYSIS:")
        print(f"  Errors Fixed: {error_analysis['errors_fixed']}")
        print(f"  New Errors: {error_analysis['new_errors']}")
        print(f"  Net Reduction: {error_analysis['net_error_reduction']}")
        print(f"  Error Reduction Rate: {error_analysis['error_reduction_rate']:.2%}")
        
        # Best and worst performing classes
        class_improvements = evaluation_results['class_improvements']
        most_improved = class_improvements['most_improved_classes']
        least_improved = class_improvements['least_improved_classes']
        
        if most_improved:
            print(f"\nMOST IMPROVED CLASSES:")
            for class_name, improvement in most_improved:
                print(f"  {class_name}: {improvement:+.4f}")
        
        if least_improved:
            print(f"\nLEAST IMPROVED CLASSES:")
            for class_name, improvement in least_improved[-3:]:
                print(f"  {class_name}: {improvement:+.4f}")
        
        # Cross-validation results
        cv_results = evaluation_results.get('cross_validation')
        if cv_results and 'error' not in cv_results:
            print(f"\nCROSS-VALIDATION:")
            print(f"  Before: {cv_results['cv_mean_before']:.4f} (±{cv_results['cv_std_before']:.4f})")
            print(f"  After:  {cv_results['cv_mean_after']:.4f} (±{cv_results['cv_std_after']:.4f})")
            print(f"  Improvement: {cv_results['cv_improvement']:+.4f}")
            
            if 'statistical_significance' in cv_results:
                sig = cv_results['statistical_significance']
                significance = "significant" if sig['is_significant'] else "not significant"
                print(f"  Statistical significance: {significance} (p={sig['p_value']:.4f})")
        
        # ROC analysis
        roc_analysis = evaluation_results.get('roc_analysis')
        if roc_analysis and 'error' not in roc_analysis:
            print(f"\nROC ANALYSIS:")
            print(f"  AUC Before: {roc_analysis['auc_before']:.4f}")
            print(f"  AUC After: {roc_analysis['auc_after']:.4f}")
            print(f"  AUC Improvement: {roc_analysis['auc_improvement']:+.4f}")
        
        print("="*60)
    
    def _save_evaluation_results(self, evaluation_results: Dict, evaluation_name: str):
        """Save evaluation results to JSON file."""
        
        # Make results JSON serializable
        serializable_results = self._make_json_serializable(evaluation_results)
        
        # Save to JSON
        json_path = os.path.join(self.save_dir, f'{evaluation_name}_results.json')
        import json
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Evaluation results saved to {json_path}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            # Ensure keys are JSON-serializable (e.g., convert numpy types to strings)
            return {str(key): self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return obj
    
    def compare_multiple_evaluations(self, evaluation_names: List[str] = None) -> Dict[str, Any]:
        """Compare multiple evaluations from history."""
        
        if not self.evaluation_history:
            return {"message": "No evaluations in history"}
        
        evaluations_to_compare = self.evaluation_history
        if evaluation_names:
            evaluations_to_compare = [
                eval_result for eval_result in self.evaluation_history 
                if eval_result['evaluation_name'] in evaluation_names
            ]
        
        if len(evaluations_to_compare) < 2:
            return {"message": "Need at least 2 evaluations to compare"}
        
        comparison = {
            'n_evaluations': len(evaluations_to_compare),
            'evaluation_names': [e['evaluation_name'] for e in evaluations_to_compare],
            'metric_trends': {},
            'best_evaluation': None,
            'worst_evaluation': None
        }
        
        # Extract metrics for comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in metrics:
            metric_values = []
            improvements = []
            
            for evaluation in evaluations_to_compare:
                after_value = evaluation['metrics_comparison']['after'][metric]
                improvement = evaluation['metrics_comparison']['improvements'][metric]
                
                metric_values.append(after_value)
                improvements.append(improvement)
            
            comparison['metric_trends'][metric] = {
                'values': metric_values,
                'improvements': improvements,
                'best_value': max(metric_values),
                'worst_value': min(metric_values),
                'average_improvement': np.mean(improvements)
            }
        
        # Find best and worst evaluations based on accuracy
        accuracy_values = comparison['metric_trends']['accuracy']['values']
        best_idx = np.argmax(accuracy_values)
        worst_idx = np.argmin(accuracy_values)
        
        comparison['best_evaluation'] = {
            'name': evaluations_to_compare[best_idx]['evaluation_name'],
            'accuracy': accuracy_values[best_idx]
        }
        comparison['worst_evaluation'] = {
            'name': evaluations_to_compare[worst_idx]['evaluation_name'],
            'accuracy': accuracy_values[worst_idx]
        }
        
        return comparison

# Example usage
if __name__ == "__main__":
    print("Model evaluator initialized")
    print("This module provides comprehensive model evaluation and comparison capabilities")
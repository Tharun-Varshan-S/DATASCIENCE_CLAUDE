# src/self_correction.py
"""
Self-Correction Module
Implements automated error detection and correction with incremental retraining.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.base import clone
from sklearn.metrics import accuracy_score
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SelfCorrectionSystem:
    def __init__(self, 
                 correction_threshold: float = 0.05,
                 min_errors_for_correction: int = 10,
                 max_iterations: int = 5,
                 improvement_threshold: float = 0.01,
                 log_dir: str = 'logs'):
        """
        Initialize self-correction system.
        
        Args:
            correction_threshold: Error rate threshold to trigger correction
            min_errors_for_correction: Minimum errors needed to trigger correction
            max_iterations: Maximum correction iterations
            improvement_threshold: Minimum improvement required to continue
            log_dir: Directory for correction logs
        """
        self.correction_threshold = correction_threshold
        self.min_errors_for_correction = min_errors_for_correction
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.log_dir = log_dir
        
        # Correction history
        self.correction_history = []
        self.performance_history = []
        self.current_iteration = 0
        
        # Models and data
        self.current_model = None
        self.best_model = None
        self.best_accuracy = 0.0
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        print(f"Self-correction system initialized")
        print(f"  - Correction threshold: {correction_threshold}")
        print(f"  - Min errors for correction: {min_errors_for_correction}")
        print(f"  - Max iterations: {max_iterations}")
        
    def run_self_correction(self,
                           initial_model,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_test: np.ndarray,
                           y_test: np.ndarray,
                           error_monitor,
                           error_analyzer,
                           mitigation_strategies) -> Dict[str, Any]:
        """
        Run the complete self-correction process.
        
        Args:
            initial_model: Initial trained model
            X_train, y_train: Training data
            X_test, y_test: Test data
            error_monitor: ErrorMonitor instance
            error_analyzer: ErrorAnalyzer instance
            mitigation_strategies: MitigationStrategies instance
            
        Returns:
            Self-correction results
        """
        print("\n" + "="*50)
        print("STARTING SELF-CORRECTION PROCESS")
        print("="*50)
        
        # Initialize
        self.current_model = initial_model
        self.current_iteration = 0
        
        # Store original data for potential reuse
        X_train_original = X_train.copy()
        y_train_original = y_train.copy()
        X_train_current = X_train.copy()
        y_train_current = y_train.copy()
        
        # Initial evaluation
        initial_accuracy = self._evaluate_model(self.current_model, X_test, y_test)
        self.best_accuracy = initial_accuracy
        # Keep the fitted model instance; do not clone (clone creates an unfitted estimator)
        self.best_model = self.current_model
        
        print(f"Initial model accuracy: {initial_accuracy:.4f}")
        
        correction_results = {
            'initial_accuracy': initial_accuracy,
            'iterations': [],
            'final_accuracy': initial_accuracy,
            'total_improvement': 0.0,
            'best_iteration': 0,
            'correction_successful': False,
            'early_stopping': False
        }
        
        # Main correction loop
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration + 1
            print(f"\n--- Correction Iteration {self.current_iteration} ---")
            
            # Step 1: Monitor errors
            print("Step 1: Monitoring errors...")
            y_pred = self.current_model.predict(X_test)
            y_proba = self.current_model.predict_proba(X_test)
            
            monitoring_results = error_monitor.monitor_predictions(
                X_test, y_test, y_pred, y_proba, 
                f"iteration_{self.current_iteration}"
            )
            
            error_rate = monitoring_results['error_rate']
            n_errors = monitoring_results['total_errors']
            
            print(f"Current error rate: {error_rate:.4f} ({n_errors} errors)")
            
            # Check if correction is needed
            if error_rate < self.correction_threshold or n_errors < self.min_errors_for_correction:
                print(f"Error rate below threshold ({self.correction_threshold}) or insufficient errors. Stopping.")
                correction_results['early_stopping'] = True
                break
            
            # Step 2: Analyze errors
            print("Step 2: Analyzing errors...")
            error_analysis = error_analyzer.analyze_errors(
                monitoring_results['error_details'],
                X_test,
                y_test
            )
            
            # Step 3: Select and apply best mitigation strategy
            print("Step 3: Applying mitigation strategies...")
            iteration_result = self._apply_best_mitigation(
                mitigation_strategies,
                self.current_model,
                X_train_current,
                y_train_current,
                X_test,
                y_test,
                error_analysis,
                monitoring_results
            )
            
            # Step 4: Evaluate improvement
            new_accuracy = iteration_result['best_accuracy']
            improvement = new_accuracy - (correction_results['final_accuracy'] if iteration > 0 else initial_accuracy)
            
            print(f"Iteration {self.current_iteration} results:")
            print(f"  - Previous accuracy: {correction_results['final_accuracy']:.4f}")
            print(f"  - New accuracy: {new_accuracy:.4f}")
            print(f"  - Improvement: {improvement:+.4f}")
            print(f"  - Best strategy: {iteration_result['best_strategy']}")
            
            # Update current model and data if improvement found
            if improvement > self.improvement_threshold:
                self.current_model = iteration_result['best_model']
                correction_results['final_accuracy'] = new_accuracy
                correction_results['correction_successful'] = True
                
                # Update training data if strategy modified it
                if 'X_train_modified' in iteration_result:
                    X_train_current = iteration_result['X_train_modified']
                    y_train_current = iteration_result['y_train_modified']
                
                # Update best model if this is the best so far
                if new_accuracy > self.best_accuracy:
                    self.best_accuracy = new_accuracy
                    # Preserve the fitted model as-is
                    self.best_model = self.current_model
                    correction_results['best_iteration'] = self.current_iteration
                
            else:
                print(f"Improvement ({improvement:.4f}) below threshold ({self.improvement_threshold}). Stopping.")
                break
            
            # Store iteration results
            iteration_info = {
                'iteration': self.current_iteration,
                'error_rate': error_rate,
                'n_errors': n_errors,
                'accuracy_before': correction_results['final_accuracy'] - improvement,
                'accuracy_after': new_accuracy,
                'improvement': improvement,
                'best_strategy': iteration_result['best_strategy'],
                'strategies_tried': iteration_result['strategies_tried'],
                'error_analysis_summary': self._summarize_error_analysis(error_analysis)
            }
            
            correction_results['iterations'].append(iteration_info)
            self.correction_history.append(iteration_info)
        
        # Finalize results
        correction_results['total_improvement'] = correction_results['final_accuracy'] - initial_accuracy
        correction_results['iterations_completed'] = self.current_iteration
        
        # Use best model found
        if self.best_model is not None:
            self.current_model = self.best_model
            correction_results['final_accuracy'] = self.best_accuracy
        
        # Log results
        self._log_correction_results(correction_results)
        
        print(f"\n" + "="*50)
        print("SELF-CORRECTION COMPLETED")
        print("="*50)
        print(f"Initial accuracy: {initial_accuracy:.4f}")
        print(f"Final accuracy: {correction_results['final_accuracy']:.4f}")
        print(f"Total improvement: {correction_results['total_improvement']:+.4f}")
        print(f"Iterations completed: {correction_results['iterations_completed']}")
        print(f"Best iteration: {correction_results['best_iteration']}")
        
        return correction_results
    
    def _evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate model accuracy."""
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    def _apply_best_mitigation(self,
                              mitigation_strategies,
                              current_model,
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              error_analysis: Dict,
                              monitoring_results: Dict) -> Dict[str, Any]:
        """
        Try multiple mitigation strategies and select the best one.
        """
        strategies_to_try = self._select_strategies(error_analysis)
        
        print(f"Trying {len(strategies_to_try)} mitigation strategies...")
        
        best_accuracy = self._evaluate_model(current_model, X_test, y_test)
        best_strategy = 'none'
        best_model = current_model
        best_result = None
        
        strategy_results = {}
        
        for strategy_name, strategy_params in strategies_to_try:
            try:
                print(f"  Trying {strategy_name}...")
                
                result = mitigation_strategies.apply_mitigation(
                    strategy_name,
                    current_model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    error_analysis,
                    **strategy_params
                )
                
                # Get accuracy from result
                if 'improved_accuracy' in result:
                    accuracy = result['improved_accuracy']
                elif 'ensemble_accuracy' in result:
                    accuracy = result['ensemble_accuracy']
                elif 'augmented_accuracy' in result:
                    accuracy = result['augmented_accuracy']
                elif 'final_accuracy' in result:
                    accuracy = result['final_accuracy']
                elif 'balanced_accuracy' in result:
                    accuracy = result['balanced_accuracy']
                elif 'selected_accuracy' in result:
                    accuracy = result['selected_accuracy']
                elif 'clean_accuracy' in result:
                    accuracy = result['clean_accuracy']
                else:
                    accuracy = best_accuracy
                
                strategy_results[strategy_name] = {
                    'accuracy': accuracy,
                    'improvement': accuracy - best_accuracy,
                    'result': result
                }
                
                print(f"    {strategy_name}: {accuracy:.4f} ({accuracy - best_accuracy:+.4f})")
                
                # Update best if this is better
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_strategy = strategy_name
                    best_result = result
                    
                    # Get the improved model
                    if 'improved_predictions' in result:
                        best_model = current_model  # Predictions modified, not model
                    elif 'ensemble_model' in result:
                        best_model = result['ensemble_model']
                    elif 'augmented_model' in result:
                        best_model = result['augmented_model']
                    elif 'active_model' in result:
                        best_model = result['active_model']
                    elif 'balanced_model' in result:
                        best_model = result['balanced_model']
                    elif 'selected_model' in result:
                        best_model = result['selected_model']
                    elif 'clean_model' in result:
                        best_model = result['clean_model']
                
            except Exception as e:
                print(f"    {strategy_name}: FAILED - {str(e)}")
                strategy_results[strategy_name] = {
                    'accuracy': best_accuracy,
                    'improvement': 0,
                    'error': str(e)
                }
        
        result_summary = {
            'best_accuracy': best_accuracy,
            'best_strategy': best_strategy,
            'best_model': best_model,
            'best_result': best_result,
            'strategies_tried': list(strategy_results.keys()),
            'strategy_results': strategy_results
        }
        
        # Add modified training data if applicable
        if best_result:
            if 'X_augmented' in best_result and 'y_augmented' in best_result:
                result_summary['X_train_modified'] = best_result['X_augmented']
                result_summary['y_train_modified'] = best_result['y_augmented']
            elif 'X_active' in best_result and 'y_active' in best_result:
                result_summary['X_train_modified'] = best_result['X_active']
                result_summary['y_train_modified'] = best_result['y_active']
            elif 'X_train_clean' in best_result and 'y_train_clean' in best_result:
                result_summary['X_train_modified'] = best_result['X_train_clean']
                result_summary['y_train_modified'] = best_result['y_train_clean']
        
        return result_summary
    
    def _select_strategies(self, error_analysis: Dict) -> List[Tuple[str, Dict]]:
        """
        Select appropriate mitigation strategies based on error analysis.
        """
        strategies = []
        
        # Check error categories and recommend strategies
        error_categories = error_analysis.get('error_categories', {})
        
        # Low confidence errors -> confidence thresholding
        if error_categories.get('low_confidence', {}).get('count', 0) > 0:
            strategies.append(('confidence_thresholding', {'confidence_threshold': 0.7}))
        
        # High confidence errors -> ensemble learning
        if error_categories.get('high_confidence', {}).get('count', 0) > 0:
            strategies.append(('ensemble_learning', {'ensemble_type': 'bagging', 'n_estimators': 5}))
        
        # Class imbalance -> data augmentation
        if error_categories.get('class_imbalance', {}).get('count', 0) > 0:
            strategies.append(('data_augmentation', {'augmentation_method': 'smote'}))
        
        # Boundary cases -> ensemble learning
        if error_categories.get('boundary_cases', {}).get('count', 0) > 0:
            strategies.append(('ensemble_learning', {'ensemble_type': 'voting', 'n_estimators': 3}))
        
        # Systematic bias -> active learning
        if error_categories.get('systematic_bias', {}).get('count', 0) > 0:
            strategies.append(('active_learning', {'n_queries': 20, 'query_strategy': 'uncertainty'}))
        
        # Always try class balancing and outlier removal as they're generally applicable
        strategies.append(('class_balancing', {'balancing_method': 'class_weight'}))
        strategies.append(('outlier_removal', {'contamination': 0.1}))
        
        # If no specific strategies identified, try common ones
        if not strategies:
            strategies = [
                ('ensemble_learning', {'ensemble_type': 'bagging', 'n_estimators': 5}),
                ('data_augmentation', {'augmentation_method': 'smote'}),
                ('confidence_thresholding', {'confidence_threshold': 0.6})
            ]
        
        return strategies[:4]  # Limit to 4 strategies per iteration
    
    def _summarize_error_analysis(self, error_analysis: Dict) -> Dict:
        """Create a summary of error analysis for logging."""
        return {
            'total_errors': error_analysis.get('total_errors', 0),
            'low_confidence_errors': error_analysis.get('confidence_analysis', {}).get('low_confidence_errors', 0),
            'high_confidence_errors': error_analysis.get('confidence_analysis', {}).get('high_confidence_errors', 0),
            'imbalance_errors': error_analysis.get('class_imbalance_analysis', {}).get('imbalance_errors', 0),
            'boundary_errors': error_analysis.get('boundary_analysis', {}).get('boundary_errors', 0),
            'systematic_biases': len(error_analysis.get('bias_analysis', {}).get('systematic_biases', []))
        }
    
    def _log_correction_results(self, results: Dict):
        """Log correction results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"self_correction_{timestamp}.json")
        
        # Make results JSON serializable
        serializable_results = self._make_json_serializable(results)
        
        with open(log_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Correction results logged to {log_file}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
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
        elif hasattr(obj, 'predict'):  # Skip model objects
            return str(type(obj).__name__)
        else:
            return obj
    
    def get_correction_summary(self) -> Dict[str, Any]:
        """Get summary of all correction attempts."""
        if not self.correction_history:
            return {"message": "No corrections performed yet"}
        
        total_improvements = [h['improvement'] for h in self.correction_history]
        successful_corrections = [h for h in self.correction_history if h['improvement'] > 0]
        
        summary = {
            'total_corrections': len(self.correction_history),
            'successful_corrections': len(successful_corrections),
            'success_rate': len(successful_corrections) / len(self.correction_history) if self.correction_history else 0,
            'total_improvement': sum(total_improvements),
            'average_improvement': np.mean(total_improvements) if total_improvements else 0,
            'best_improvement': max(total_improvements) if total_improvements else 0,
            'most_effective_strategies': self._get_most_effective_strategies()
        }
        
        return summary
    
    def _get_most_effective_strategies(self) -> List[Dict]:
        """Get the most effective strategies across all corrections."""
        strategy_effectiveness = {}
        
        for correction in self.correction_history:
            strategy = correction['best_strategy']
            improvement = correction['improvement']
            
            if strategy not in strategy_effectiveness:
                strategy_effectiveness[strategy] = {
                    'total_improvement': 0,
                    'times_used': 0,
                    'times_best': 0
                }
            
            strategy_effectiveness[strategy]['total_improvement'] += improvement
            strategy_effectiveness[strategy]['times_used'] += 1
            if improvement > 0:
                strategy_effectiveness[strategy]['times_best'] += 1
        
        # Calculate average effectiveness
        for strategy_info in strategy_effectiveness.values():
            strategy_info['average_improvement'] = (
                strategy_info['total_improvement'] / strategy_info['times_used']
            )
        
        # Sort by average improvement
        sorted_strategies = sorted(
            strategy_effectiveness.items(),
            key=lambda x: x[1]['average_improvement'],
            reverse=True
        )
        
        return [
            {
                'strategy': strategy,
                'average_improvement': info['average_improvement'],
                'times_used': info['times_used'],
                'times_best': info['times_best']
            }
            for strategy, info in sorted_strategies
        ]

# Example usage
if __name__ == "__main__":
    print("Self-correction system initialized")
    print("This module is designed to work with the complete error mitigation pipeline")
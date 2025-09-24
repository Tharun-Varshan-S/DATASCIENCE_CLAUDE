# src/mitigation_strategies.py
"""
Error Mitigation Strategies Module
Implements various strategies to mitigate identified errors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

class MitigationStrategies:
    def __init__(self, random_state=42):
        """Initialize mitigation strategies."""
        self.random_state = random_state
        self.strategies = {
            'confidence_thresholding': self._confidence_thresholding,
            'ensemble_learning': self._ensemble_learning,
            'data_augmentation': self._data_augmentation,
            'active_learning': self._active_learning,
            'class_balancing': self._class_balancing,
            'feature_selection': self._feature_selection,
            'outlier_removal': self._outlier_removal
        }
        
    def apply_mitigation(self, 
                        strategy_name: str,
                        model,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        error_analysis: Dict,
                        **kwargs) -> Dict[str, Any]:
        """
        Apply a specific mitigation strategy.
        
        Args:
            strategy_name: Name of mitigation strategy
            model: Base ML model
            X_train, y_train: Training data
            X_test, y_test: Test data
            error_analysis: Results from error analysis
            **kwargs: Strategy-specific parameters
            
        Returns:
            Mitigation results
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy {strategy_name} not supported")
        
        print(f"Applying {strategy_name} mitigation strategy...")
        
        strategy_func = self.strategies[strategy_name]
        return strategy_func(model, X_train, y_train, X_test, y_test, error_analysis, **kwargs)
    
    def _confidence_thresholding(self, 
                               model,
                               X_train: np.ndarray,
                               y_train: np.ndarray,
                               X_test: np.ndarray,
                               y_test: np.ndarray,
                               error_analysis: Dict,
                               confidence_threshold: float = 0.6,
                               **kwargs) -> Dict[str, Any]:
        """
        Apply confidence thresholding to reject low-confidence predictions.
        """
        print(f"Applying confidence thresholding with threshold = {confidence_threshold}")
        
        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        max_proba = np.max(y_proba, axis=1)
        
        # Apply threshold
        confident_mask = max_proba >= confidence_threshold
        rejected_mask = ~confident_mask
        
        n_confident = np.sum(confident_mask)
        n_rejected = np.sum(rejected_mask)
        
        # Calculate metrics for confident predictions only
        if n_confident > 0:
            confident_accuracy = accuracy_score(y_test[confident_mask], y_pred[confident_mask])
        else:
            confident_accuracy = 0.0
        
        # Overall accuracy including rejected as wrong
        overall_accuracy = accuracy_score(y_test, y_pred)
        
        # For rejected samples, we could use a fallback strategy
        # Here we'll use the most common class as fallback
        fallback_class = np.bincount(y_train).argmax()
        
        # Create improved predictions
        improved_predictions = y_pred.copy()
        improved_predictions[rejected_mask] = fallback_class
        
        improved_accuracy = accuracy_score(y_test, improved_predictions)
        
        results = {
            'strategy': 'confidence_thresholding',
            'confidence_threshold': confidence_threshold,
            'n_confident_predictions': n_confident,
            'n_rejected_predictions': n_rejected,
            'rejection_rate': n_rejected / len(y_test),
            'confident_accuracy': confident_accuracy,
            'original_accuracy': overall_accuracy,
            'improved_accuracy': improved_accuracy,
            'improvement': improved_accuracy - overall_accuracy,
            'confident_mask': confident_mask,
            'improved_predictions': improved_predictions
        }
        
        print(f"Confidence thresholding results:")
        print(f"  - Rejected {n_rejected}/{len(y_test)} predictions ({n_rejected/len(y_test):.2%})")
        print(f"  - Confident predictions accuracy: {confident_accuracy:.4f}")
        print(f"  - Overall improvement: {improved_accuracy - overall_accuracy:+.4f}")
        
        return results
    
    def _ensemble_learning(self, 
                          base_model,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          error_analysis: Dict,
                          n_estimators: int = 5,
                          ensemble_type: str = 'bagging',
                          **kwargs) -> Dict[str, Any]:
        """
        Create ensemble model to improve predictions.
        """
        print(f"Creating {ensemble_type} ensemble with {n_estimators} estimators")

        original_accuracy = accuracy_score(y_test, base_model.predict(X_test))

        if ensemble_type == 'bagging':
            # Bagging ensemble
            ensemble = BaggingClassifier(
                estimator=clone(base_model),
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )

        elif ensemble_type == 'voting':
            # Create diverse base models
            from sklearn.svm import SVC
            from sklearn.neural_network import MLPClassifier
            from sklearn.naive_bayes import GaussianNB

            estimators = [
                ('rf', clone(base_model)),
                ('svm', SVC(probability=True, random_state=self.random_state)),
                ('mlp', MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=self.random_state)),
                ('nb', GaussianNB())
            ]

            ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',
                n_jobs=-1
            )

        else:
            raise ValueError(f"Ensemble type {ensemble_type} not supported")

        # Train ensemble
        ensemble.fit(X_train, y_train)

        # Make predictions
        ensemble_predictions = ensemble.predict(X_test)
        ensemble_proba = ensemble.predict_proba(X_test)

        # Calculate metrics
        ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
        improvement = ensemble_accuracy - original_accuracy

        # Calculate prediction confidence
        ensemble_confidence = np.max(ensemble_proba, axis=1)
        avg_confidence = np.mean(ensemble_confidence)

        results = {
            'strategy': 'ensemble_learning',
            'ensemble_type': ensemble_type,
            'n_estimators': n_estimators,
            'original_accuracy': original_accuracy,
            'ensemble_accuracy': ensemble_accuracy,
            'improvement': improvement,
            'average_confidence': avg_confidence,
            'ensemble_model': ensemble,
            'ensemble_predictions': ensemble_predictions,
            'ensemble_probabilities': ensemble_proba
        }

        print(f"Ensemble learning results:")
        print(f"  - Original accuracy: {original_accuracy:.4f}")
        print(f"  - Ensemble accuracy: {ensemble_accuracy:.4f}")
        print(f"  - Improvement: {improvement:+.4f}")
        print(f"  - Average confidence: {avg_confidence:.4f}")

        return results
    
    def _data_augmentation(self, 
                          model,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          error_analysis: Dict,
                          augmentation_method: str = 'smote',
                          **kwargs) -> Dict[str, Any]:
        """
        Apply data augmentation to address class imbalance.
        """
        print(f"Applying data augmentation using {augmentation_method}")
        
        original_accuracy = accuracy_score(y_test, model.predict(X_test))
        
        # Identify classes that need augmentation
        class_counts = np.bincount(y_train)
        max_count = np.max(class_counts)
        minority_classes = np.where(class_counts < max_count * 0.5)[0]
        
        print(f"Original class distribution: {dict(zip(range(len(class_counts)), class_counts))}")
        print(f"Minority classes identified: {minority_classes}")
        
        # Apply augmentation
        if augmentation_method == 'smote':
            try:
                smote = SMOTE(random_state=self.random_state, k_neighbors=min(3, len(X_train)//10))
                X_augmented, y_augmented = smote.fit_resample(X_train, y_train)
            except ValueError as e:
                print(f"SMOTE failed: {e}. Using ADASYN instead.")
                adasyn = ADASYN(random_state=self.random_state, n_neighbors=min(3, len(X_train)//10))
                X_augmented, y_augmented = adasyn.fit_resample(X_train, y_train)
        
        elif augmentation_method == 'adasyn':
            adasyn = ADASYN(random_state=self.random_state, n_neighbors=min(3, len(X_train)//10))
            X_augmented, y_augmented = adasyn.fit_resample(X_train, y_train)
        
        elif augmentation_method == 'random_oversampling':
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=self.random_state)
            X_augmented, y_augmented = ros.fit_resample(X_train, y_train)
        
        else:
            raise ValueError(f"Augmentation method {augmentation_method} not supported")
        
        print(f"Augmented class distribution: {dict(zip(*np.unique(y_augmented, return_counts=True)))}")
        
        # Train model on augmented data
        augmented_model = clone(model)
        augmented_model.fit(X_augmented, y_augmented)
        
        # Evaluate
        augmented_predictions = augmented_model.predict(X_test)
        augmented_accuracy = accuracy_score(y_test, augmented_predictions)
        improvement = augmented_accuracy - original_accuracy
        
        # Check improvement for minority classes specifically
        minority_improvement = {}
        for class_id in minority_classes:
            class_mask = y_test == class_id
            if np.sum(class_mask) > 0:
                original_class_acc = accuracy_score(y_test[class_mask], model.predict(X_test)[class_mask])
                augmented_class_acc = accuracy_score(y_test[class_mask], augmented_predictions[class_mask])
                minority_improvement[int(class_id)] = augmented_class_acc - original_class_acc
        
        results = {
            'strategy': 'data_augmentation',
            'augmentation_method': augmentation_method,
            'original_samples': len(X_train),
            'augmented_samples': len(X_augmented),
            'augmentation_ratio': len(X_augmented) / len(X_train),
            'original_accuracy': original_accuracy,
            'augmented_accuracy': augmented_accuracy,
            'improvement': improvement,
            'minority_classes': minority_classes.tolist(),
            'minority_improvement': minority_improvement,
            'augmented_model': augmented_model,
            'augmented_predictions': augmented_predictions,
            'X_augmented': X_augmented,
            'y_augmented': y_augmented
        }
        
        print(f"Data augmentation results:")
        print(f"  - Original samples: {len(X_train)}")
        print(f"  - Augmented samples: {len(X_augmented)}")
        print(f"  - Original accuracy: {original_accuracy:.4f}")
        print(f"  - Augmented accuracy: {augmented_accuracy:.4f}")
        print(f"  - Improvement: {improvement:+.4f}")
        
        return results
    
    def _active_learning(self, 
                        model,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        error_analysis: Dict,
                        n_queries: int = 50,
                        query_strategy: str = 'uncertainty',
                        **kwargs) -> Dict[str, Any]:
        """
        Apply active learning to improve model with targeted samples.
        """
        print(f"Applying active learning with {n_queries} queries using {query_strategy} strategy")
        
        original_accuracy = accuracy_score(y_test, model.predict(X_test))
        
        # For simulation, we'll use a portion of test set as "unlabeled" pool
        # In practice, you would have actual unlabeled data
        pool_size = min(len(X_test) // 2, n_queries * 3)
        pool_indices = np.random.choice(len(X_test), pool_size, replace=False)
        
        X_pool = X_test[pool_indices]
        y_pool = y_test[pool_indices]  # In practice, these would be unknown
        
        # Initialize active learning
        X_active = X_train.copy()
        y_active = y_train.copy()
        
        queried_indices = []
        accuracies = [original_accuracy]
        
        for query_round in range(min(n_queries, len(X_pool))):
            # Get predictions and uncertainties on pool
            pool_proba = model.predict_proba(X_pool)
            
            if query_strategy == 'uncertainty':
                # Select sample with highest uncertainty (lowest max probability)
                uncertainties = 1 - np.max(pool_proba, axis=1)
                query_idx = np.argmax(uncertainties)
            
            elif query_strategy == 'margin':
                # Select sample with smallest margin between top two predictions
                sorted_proba = np.sort(pool_proba, axis=1)
                margins = sorted_proba[:, -1] - sorted_proba[:, -2]
                query_idx = np.argmin(margins)
            
            elif query_strategy == 'entropy':
                # Select sample with highest entropy
                entropies = -np.sum(pool_proba * np.log(pool_proba + 1e-8), axis=1)
                query_idx = np.argmax(entropies)
            
            else:
                raise ValueError(f"Query strategy {query_strategy} not supported")
            
            # Add queried sample to training set
            X_active = np.vstack([X_active, X_pool[query_idx:query_idx+1]])
            y_active = np.append(y_active, y_pool[query_idx])
            queried_indices.append(pool_indices[query_idx])
            
            # Remove from pool
            pool_mask = np.ones(len(X_pool), dtype=bool)
            pool_mask[query_idx] = False
            X_pool = X_pool[pool_mask]
            y_pool = y_pool[pool_mask]
            pool_indices = pool_indices[pool_mask]
            
            # Retrain model
            active_model = clone(model)
            active_model.fit(X_active, y_active)
            
            # Evaluate
            test_mask = np.ones(len(X_test), dtype=bool)
            test_mask[queried_indices] = False  # Don't test on queried samples
            
            if np.sum(test_mask) > 0:
                active_accuracy = accuracy_score(y_test[test_mask], 
                                               active_model.predict(X_test[test_mask]))
                accuracies.append(active_accuracy)
            
            model = active_model  # Update for next iteration
        
        final_accuracy = accuracies[-1]
        improvement = final_accuracy - original_accuracy
        
        results = {
            'strategy': 'active_learning',
            'query_strategy': query_strategy,
            'n_queries': len(queried_indices),
            'original_accuracy': original_accuracy,
            'final_accuracy': final_accuracy,
            'improvement': improvement,
            'accuracy_progression': accuracies,
            'queried_indices': queried_indices,
            'active_model': model,
            'X_active': X_active,
            'y_active': y_active
        }
        
        print(f"Active learning results:")
        print(f"  - Queries made: {len(queried_indices)}")
        print(f"  - Original accuracy: {original_accuracy:.4f}")
        print(f"  - Final accuracy: {final_accuracy:.4f}")
        print(f"  - Improvement: {improvement:+.4f}")
        
        return results
    
    def _class_balancing(self, 
                        model,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        error_analysis: Dict,
                        balancing_method: str = 'class_weight',
                        **kwargs) -> Dict[str, Any]:
        """
        Apply class balancing techniques.
        """
        print(f"Applying class balancing using {balancing_method}")
        
        original_accuracy = accuracy_score(y_test, model.predict(X_test))
        
        if balancing_method == 'class_weight':
            # Compute class weights
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            weight_dict = dict(zip(classes, class_weights))
            
            # Create model with class weights
            balanced_model = clone(model)
            if hasattr(balanced_model, 'class_weight'):
                balanced_model.set_params(class_weight=weight_dict)
            else:
                print(f"Model {type(model).__name__} doesn't support class_weight parameter")
                return {'strategy': 'class_balancing', 'error': 'Model does not support class weights'}
        
        elif balancing_method == 'undersampling':
            # Random undersampling
            undersampler = RandomUnderSampler(random_state=self.random_state)
            X_balanced, y_balanced = undersampler.fit_resample(X_train, y_train)
            
            balanced_model = clone(model)
            balanced_model.fit(X_balanced, y_balanced)
        
        else:
            raise ValueError(f"Balancing method {balancing_method} not supported")
        
        # Train balanced model
        if balancing_method == 'class_weight':
            balanced_model.fit(X_train, y_train)
        
        # Evaluate
        balanced_predictions = balanced_model.predict(X_test)
        balanced_accuracy = accuracy_score(y_test, balanced_predictions)
        improvement = balanced_accuracy - original_accuracy
        
        # Per-class performance
        unique_classes = np.unique(y_test)
        class_performance = {}
        
        for class_id in unique_classes:
            class_mask = y_test == class_id
            if np.sum(class_mask) > 0:
                original_class_acc = accuracy_score(y_test[class_mask], 
                                                  model.predict(X_test)[class_mask])
                balanced_class_acc = accuracy_score(y_test[class_mask], 
                                                  balanced_predictions[class_mask])
                class_performance[int(class_id)] = {
                    'original_accuracy': original_class_acc,
                    'balanced_accuracy': balanced_class_acc,
                    'improvement': balanced_class_acc - original_class_acc
                }
        
        results = {
            'strategy': 'class_balancing',
            'balancing_method': balancing_method,
            'original_accuracy': original_accuracy,
            'balanced_accuracy': balanced_accuracy,
            'improvement': improvement,
            'class_performance': class_performance,
            'balanced_model': balanced_model,
            'balanced_predictions': balanced_predictions
        }
        
        if balancing_method == 'class_weight':
            results['class_weights'] = weight_dict
        elif balancing_method == 'undersampling':
            results['balanced_samples'] = len(X_balanced)
            results['original_samples'] = len(X_train)
        
        print(f"Class balancing results:")
        print(f"  - Original accuracy: {original_accuracy:.4f}")
        print(f"  - Balanced accuracy: {balanced_accuracy:.4f}")
        print(f"  - Improvement: {improvement:+.4f}")
        
        return results
    
    def _feature_selection(self, 
                          model,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          error_analysis: Dict,
                          selection_method: str = 'importance',
                          n_features: int = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Apply feature selection to improve model performance.
        """
        print(f"Applying feature selection using {selection_method}")
        
        original_accuracy = accuracy_score(y_test, model.predict(X_test))
        
        if n_features is None:
            n_features = min(X_train.shape[1] // 2, 20)  # Select half or max 20 features
        
        if selection_method == 'importance':
            # Use model feature importance if available
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).mean(axis=0)
            else:
                # Use Random Forest for feature importance
                rf_selector = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
                rf_selector.fit(X_train, y_train)
                importances = rf_selector.feature_importances_
            
            # Select top features
            top_features = np.argsort(importances)[-n_features:]
        
        elif selection_method == 'variance':
            # Select features with highest variance
            from sklearn.feature_selection import VarianceThreshold
            variances = np.var(X_train, axis=0)
            top_features = np.argsort(variances)[-n_features:]
        
        elif selection_method == 'univariate':
            # Univariate feature selection
            from sklearn.feature_selection import SelectKBest, f_classif
            selector = SelectKBest(score_func=f_classif, k=n_features)
            selector.fit(X_train, y_train)
            top_features = selector.get_support(indices=True)
        
        else:
            raise ValueError(f"Selection method {selection_method} not supported")
        
        # Create reduced datasets
        X_train_selected = X_train[:, top_features]
        X_test_selected = X_test[:, top_features]
        
        # Train model on selected features
        selected_model = clone(model)
        selected_model.fit(X_train_selected, y_train)
        
        # Evaluate
        selected_predictions = selected_model.predict(X_test_selected)
        selected_accuracy = accuracy_score(y_test, selected_predictions)
        improvement = selected_accuracy - original_accuracy
        
        results = {
            'strategy': 'feature_selection',
            'selection_method': selection_method,
            'original_features': X_train.shape[1],
            'selected_features': len(top_features),
            'selected_feature_indices': top_features.tolist(),
            'original_accuracy': original_accuracy,
            'selected_accuracy': selected_accuracy,
            'improvement': improvement,
            'selected_model': selected_model,
            'selected_predictions': selected_predictions,
            'X_train_selected': X_train_selected,
            'X_test_selected': X_test_selected
        }
        
        print(f"Feature selection results:")
        print(f"  - Original features: {X_train.shape[1]}")
        print(f"  - Selected features: {len(top_features)}")
        print(f"  - Original accuracy: {original_accuracy:.4f}")
        print(f"  - Selected accuracy: {selected_accuracy:.4f}")
        print(f"  - Improvement: {improvement:+.4f}")
        
        return results
    
    def _outlier_removal(self, 
                        model,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        error_analysis: Dict,
                        contamination: float = 0.1,
                        **kwargs) -> Dict[str, Any]:
        """
        Remove outliers from training data.
        """
        print(f"Applying outlier removal with contamination = {contamination}")
        
        from sklearn.ensemble import IsolationForest
        
        original_accuracy = accuracy_score(y_test, model.predict(X_test))
        
        # Detect outliers in training data
        outlier_detector = IsolationForest(
            contamination=contamination,
            random_state=self.random_state
        )
        outlier_labels = outlier_detector.fit_predict(X_train)
        
        # Remove outliers
        inlier_mask = outlier_labels == 1
        X_train_clean = X_train[inlier_mask]
        y_train_clean = y_train[inlier_mask]
        
        n_outliers = np.sum(~inlier_mask)
        
        print(f"Removed {n_outliers}/{len(X_train)} outliers ({n_outliers/len(X_train):.2%})")
        
        # Train model on clean data
        clean_model = clone(model)
        clean_model.fit(X_train_clean, y_train_clean)
        
        # Evaluate
        clean_predictions = clean_model.predict(X_test)
        clean_accuracy = accuracy_score(y_test, clean_predictions)
        improvement = clean_accuracy - original_accuracy
        
        results = {
            'strategy': 'outlier_removal',
            'contamination': contamination,
            'original_samples': len(X_train),
            'clean_samples': len(X_train_clean),
            'outliers_removed': n_outliers,
            'outlier_rate': n_outliers / len(X_train),
            'original_accuracy': original_accuracy,
            'clean_accuracy': clean_accuracy,
            'improvement': improvement,
            'clean_model': clean_model,
            'clean_predictions': clean_predictions,
            'X_train_clean': X_train_clean,
            'y_train_clean': y_train_clean,
            'outlier_mask': ~inlier_mask
        }
        
        print(f"Outlier removal results:")
        print(f"  - Outliers removed: {n_outliers}/{len(X_train)} ({n_outliers/len(X_train):.2%})")
        print(f"  - Original accuracy: {original_accuracy:.4f}")
        print(f"  - Clean accuracy: {clean_accuracy:.4f}")
        print(f"  - Improvement: {improvement:+.4f}")
        
        return results

# Example usage
if __name__ == "__main__":
    # This would typically be called from main.py
    from data_loader import DataLoader
    from baseline_model import BaselineModel
    
    # Load data
    loader = DataLoader('digits')
    data = loader.load_dataset()
    
    # Train baseline model
    model = BaselineModel('random_forest')
    model.train(data['X_train'], data['y_train'])
    
    # Initialize mitigation strategies
    mitigator = MitigationStrategies()
    
    # Mock error analysis (in practice, this comes from ErrorAnalyzer)
    dummy_error_analysis = {
        'confidence_analysis': {'low_confidence_errors': 10},
        'class_imbalance_analysis': {'imbalanced_classes': [0, 1]}
    }
    
    # Apply confidence thresholding
    results = mitigator.apply_mitigation(
        'confidence_thresholding',
        model.model,
        data['X_train'],
        data['y_train'],
        data['X_test'],
        data['y_test'],
        dummy_error_analysis,
        confidence_threshold=0.7
    )
    
    print("Mitigation strategy testing completed!")
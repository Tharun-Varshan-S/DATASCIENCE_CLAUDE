# src/baseline_model.py
"""
Baseline ML Model Training Module
Implements various ML algorithms and provides a unified interface.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import joblib
import os
from typing import Dict, Any, Tuple
import time

class BaselineModel:
    def __init__(self, model_type='random_forest', random_state=42):
        """
        Initialize baseline model.
        
        Args:
            model_type (str): Type of ML model to use
            random_state (int): Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self.training_time = 0
        self.feature_importance = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specified ML model."""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'svm': SVC(
                probability=True,  # Enable probability prediction
                random_state=self.random_state
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'naive_bayes': GaussianNB()
        }
        
        if self.model_type not in models:
            raise ValueError(f"Model type {self.model_type} not supported")
        
        self.model = models[self.model_type]
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train the baseline model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Training metrics dictionary
        """
        print(f"Training {self.model_type} model...")
        start_time = time.time()
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = np.abs(self.model.coef_[0])
        
        # Evaluate on training data
        train_pred = self.model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        training_info = {
            'model_type': self.model_type,
            'training_time': self.training_time,
            'train_accuracy': train_metrics['accuracy'],
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'n_samples': len(X_train),
            'n_features': X_train.shape[1]
        }
        
        print(f"Training completed in {self.training_time:.2f} seconds")
        print(f"Training accuracy: {train_metrics['accuracy']:.4f}")
        print(f"CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return training_info
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, return confidence scores
            predictions = self.model.predict(X)
            n_classes = len(np.unique(predictions))
            probabilities = np.zeros((len(predictions), n_classes))
            for i, pred in enumerate(predictions):
                probabilities[i, pred] = 0.8  # Arbitrary confidence
            return probabilities
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics dictionary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Add prediction probabilities for error analysis
        max_probabilities = np.max(y_proba, axis=1)
        
        evaluation_results = {
            'predictions': y_pred,
            'probabilities': y_proba,
            'max_probabilities': max_probabilities,
            'metrics': metrics,
            'n_test_samples': len(X_test)
        }
        
        print(f"Test Evaluation - {self.model_type}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1']:.4f}")
        
        return evaluation_results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate standard classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.feature_importance
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_info = {
            'model': self.model,
            'model_type': self.model_type,
            'training_time': self.training_time,
            'feature_importance': self.feature_importance,
            'random_state': self.random_state
        }
        
        joblib.dump(model_info, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        model_info = joblib.load(filepath)
        
        self.model = model_info['model']
        self.model_type = model_info['model_type']
        self.training_time = model_info.get('training_time', 0)
        self.feature_importance = model_info.get('feature_importance', None)
        self.random_state = model_info.get('random_state', 42)
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")

class ModelEnsemble:
    """Ensemble of multiple baseline models for better performance."""
    
    def __init__(self, model_types=['random_forest', 'gradient_boosting', 'svm'], random_state=42):
        """Initialize ensemble with multiple models."""
        self.model_types = model_types
        self.random_state = random_state
        self.models = {}
        self.is_trained = False
        
        for model_type in model_types:
            self.models[model_type] = BaselineModel(model_type, random_state)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train all models in the ensemble."""
        print("Training ensemble models...")
        training_results = {}
        
        for model_type, model in self.models.items():
            print(f"\nTraining {model_type}...")
            result = model.train(X_train, y_train)
            training_results[model_type] = result
        
        self.is_trained = True
        return training_results
    
    def predict(self, X: np.ndarray, method='voting') -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        if method == 'voting':
            # Hard voting
            predictions = []
            for model in self.models.values():
                predictions.append(model.predict(X))
            
            # Majority vote
            predictions = np.array(predictions)
            ensemble_pred = []
            for i in range(len(X)):
                votes = predictions[:, i]
                ensemble_pred.append(np.bincount(votes).argmax())
            
            return np.array(ensemble_pred)
        
        elif method == 'averaging':
            # Soft voting using probabilities
            probabilities = []
            for model in self.models.values():
                probabilities.append(model.predict_proba(X))
            
            avg_probabilities = np.mean(probabilities, axis=0)
            return np.argmax(avg_probabilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        probabilities = []
        for model in self.models.values():
            probabilities.append(model.predict_proba(X))
        
        return np.mean(probabilities, axis=0)

# Example usage
if __name__ == "__main__":
    # This would typically be called from main.py
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader('digits')
    data = loader.load_dataset()
    
    # Train baseline model
    model = BaselineModel('random_forest')
    training_info = model.train(data['X_train'], data['y_train'])
    
    # Evaluate model
    evaluation = model.evaluate(data['X_test'], data['y_test'])
    
    print("\nBaseline model training and evaluation completed!")
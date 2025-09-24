# src/data_loader.py
"""
Data Loading and Preprocessing Module
Handles loading of public datasets and preprocessing for ML models.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
"""
TensorFlow-dependent datasets (MNIST, CIFAR-10) should not import TensorFlow
at module import time to avoid DLL/runtime issues on systems without TF.
We import them lazily inside the respective loader functions.
"""
import os
import pickle

class DataLoader:
    def __init__(self, dataset_name='digits', test_size=0.2, random_state=42):
        """
        Initialize DataLoader with dataset configuration.
        
        Args:
            dataset_name (str): Name of dataset ('digits', 'breast_cancer', 'wine', 'mnist', 'cifar10')
            test_size (float): Test set proportion
            random_state (int): Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self):
        """Load and preprocess the specified dataset."""
        if self.dataset_name == 'digits':
            return self._load_digits()
        elif self.dataset_name == 'breast_cancer':
            return self._load_breast_cancer()
        elif self.dataset_name == 'wine':
            return self._load_wine()
        elif self.dataset_name == 'mnist':
            return self._load_mnist()
        elif self.dataset_name == 'cifar10':
            return self._load_cifar10()
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
    
    def _load_digits(self):
        """Load and preprocess digits dataset (8x8 handwritten digits)."""
        data = load_digits()
        X, y = data.data, data.target
        
        # Normalize features
        X = X / 16.0  # Digits are 0-16 scale
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': data.feature_names if hasattr(data, 'feature_names') else None,
            'target_names': data.target_names,
            'n_classes': len(np.unique(y)),
            'data_type': 'tabular'
        }
    
    def _load_breast_cancer(self):
        """Load and preprocess breast cancer dataset."""
        data = load_breast_cancer()
        X, y = data.data, data.target
        
        # Standardize features
        X = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': data.feature_names,
            'target_names': data.target_names,
            'n_classes': len(np.unique(y)),
            'data_type': 'tabular'
        }
    
    def _load_wine(self):
        """Load and preprocess wine dataset."""
        data = load_wine()
        X, y = data.data, data.target
        
        # Standardize features
        X = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': data.feature_names,
            'target_names': data.target_names,
            'n_classes': len(np.unique(y)),
            'data_type': 'tabular'
        }
    
    def _load_mnist(self):
        """Load and preprocess MNIST dataset."""
        try:
            from tensorflow.keras.datasets import mnist
        except Exception as exc:
            raise ImportError(
                "TensorFlow is required for 'mnist' dataset loading. "
                "Install a compatible TensorFlow build or choose a non-TF dataset (e.g., 'digits').\n"
                f"Original error: {exc}"
            )

        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        # Normalize pixel values
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Flatten for traditional ML models
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        return {
            'X_train': X_train_flat,
            'X_test': X_test_flat,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_2d': X_train,  # Keep 2D for visualization
            'X_test_2d': X_test,
            'feature_names': None,
            'target_names': [str(i) for i in range(10)],
            'n_classes': 10,
            'data_type': 'image'
        }
    
    def _load_cifar10(self):
        """Load and preprocess CIFAR-10 dataset."""
        try:
            from tensorflow.keras.datasets import cifar10
        except Exception as exc:
            raise ImportError(
                "TensorFlow is required for 'cifar10' dataset loading. "
                "Install a compatible TensorFlow build or choose a non-TF dataset (e.g., 'digits').\n"
                f"Original error: {exc}"
            )

        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        
        # Normalize pixel values
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Flatten labels
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        
        # Flatten for traditional ML models (use smaller subset for speed)
        n_samples = 5000  # Reduce for faster processing
        indices = np.random.choice(len(X_train), n_samples, replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train[indices]
        
        X_train_flat = X_train_subset.reshape(X_train_subset.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        target_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        return {
            'X_train': X_train_flat,
            'X_test': X_test_flat,
            'y_train': y_train_subset,
            'y_test': y_test,
            'X_train_2d': X_train_subset,
            'X_test_2d': X_test,
            'feature_names': None,
            'target_names': target_names,
            'n_classes': 10,
            'data_type': 'image'
        }
    
    def save_processed_data(self, data_dict, filepath):
        """Save processed data to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f)
    
    def load_processed_data(self, filepath):
        """Load processed data from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# Example usage
if __name__ == "__main__":
    # Test data loading
    loader = DataLoader('digits')
    data = loader.load_dataset()
    
    print(f"Dataset: {loader.dataset_name}")
    print(f"Training samples: {data['X_train'].shape[0]}")
    print(f"Test samples: {data['X_test'].shape[0]}")
    print(f"Features: {data['X_train'].shape[1]}")
    print(f"Classes: {data['n_classes']}")
    print(f"Target names: {data['target_names']}")
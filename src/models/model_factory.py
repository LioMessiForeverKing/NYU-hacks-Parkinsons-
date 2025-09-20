"""
Model Factory for Parkinson's Disease Detection
Creates model instances based on configuration using factory pattern
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Type, List
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from ..core.base import BaseModel

class ModelFactory:
    """
    Factory class for creating model instances
    Supports multiple algorithms with consistent interface
    """
    
    # Registry of available models
    _models = {
        'RandomForest': RandomForestClassifier,
        'GradientBoosting': GradientBoostingClassifier,
        'ExtraTrees': ExtraTreesClassifier,
        'SVM': SVC,
        'LogisticRegression': LogisticRegression,
        'KNeighbors': KNeighborsClassifier,
        'GaussianNB': GaussianNB,
        'DecisionTree': DecisionTreeClassifier,
        'MLPClassifier': MLPClassifier
    }
    
    @classmethod
    def create_model(cls, config: Dict[str, Any]) -> BaseModel:
        """
        Create model instance based on configuration
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Model instance
        """
        algorithm = config.get('algorithm', 'RandomForest')
        
        if algorithm not in cls._models:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(cls._models.keys())}")
        
        # Create appropriate model wrapper
        if algorithm == 'RandomForest':
            return RandomForestModel(config)
        elif algorithm == 'GradientBoosting':
            return GradientBoostingModel(config)
        elif algorithm == 'ExtraTrees':
            return ExtraTreesModel(config)
        elif algorithm == 'SVM':
            return SVMModel(config)
        elif algorithm == 'LogisticRegression':
            return LogisticRegressionModel(config)
        elif algorithm == 'KNeighbors':
            return KNeighborsModel(config)
        elif algorithm == 'GaussianNB':
            return GaussianNBModel(config)
        elif algorithm == 'DecisionTree':
            return DecisionTreeModel(config)
        elif algorithm == 'MLPClassifier':
            return MLPClassifierModel(config)
        else:
            raise ValueError(f"No model wrapper available for: {algorithm}")
    
    @classmethod
    def get_available_models(cls) -> list:
        """
        Get list of available model algorithms
        
        Returns:
            List of available model names
        """
        return list(cls._models.keys())
    
    @classmethod
    def register_model(cls, name: str, model_class: Type) -> None:
        """
        Register a new model type
        
        Args:
            name: Model name
            model_class: Model class
        """
        cls._models[name] = model_class
        logging.info(f"Registered new model: {name}")


class BaseSKLearnModel(BaseModel):
    """
    Base class for scikit-learn models
    Provides common functionality for all sklearn-based models
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model with configuration
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.algorithm = config['algorithm']
        self.parameters = config.get('parameters', {})
        self.model = None
        self.is_fitted = False
        self.feature_importance = None
        self.n_features = None
    
    def build_model(self) -> Any:
        """
        Build model instance based on configuration
        
        Returns:
            Model instance
        """
        model_class = ModelFactory._models[self.algorithm]
        return model_class(**self.parameters)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseSKLearnModel':
        """
        Train model on data
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Training {self.algorithm} model")
        
        try:
            if self.model is None:
                self.model = self.build_model()
            
            self.n_features = X.shape[1]
            self.model.fit(X, y)
            self.is_fitted = True
            
            # Extract feature importance if available
            self._extract_feature_importance()
            
            self.logger.info(f"{self.algorithm} model trained successfully")
            return self
            
        except Exception as e:
            self.logger.error(f"Error training {self.algorithm} model: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting probabilities")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, use decision_function
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(X)
                # Convert to probabilities using sigmoid
                import numpy as np
                proba = 1 / (1 + np.exp(-scores))
                return np.column_stack([1 - proba, proba])
            else:
                raise ValueError(f"{self.algorithm} does not support probability prediction")
    
    def _extract_feature_importance(self) -> None:
        """
        Extract feature importance if available
        """
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients
            self.feature_importance = np.abs(self.model.coef_[0])
        else:
            self.feature_importance = None
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance scores
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if self.feature_importance is None:
            self.logger.warning(f"{self.algorithm} does not support feature importance")
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        self.logger.info("Feature importance computed")
        return df


# Specific model implementations
class RandomForestModel(BaseSKLearnModel):
    """Random Forest model implementation"""
    pass

class GradientBoostingModel(BaseSKLearnModel):
    """Gradient Boosting model implementation"""
    pass

class ExtraTreesModel(BaseSKLearnModel):
    """Extra Trees model implementation"""
    pass

class SVMModel(BaseSKLearnModel):
    """Support Vector Machine model implementation"""
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for SVM
        Requires probability=True in parameters
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting probabilities")
        
        if not self.parameters.get('probability', False):
            self.logger.warning("SVM probability prediction requires probability=True")
            # Fallback to decision function
            scores = self.model.decision_function(X)
            proba = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - proba, proba])
        
        return self.model.predict_proba(X)

class LogisticRegressionModel(BaseSKLearnModel):
    """Logistic Regression model implementation"""
    pass

class KNeighborsModel(BaseSKLearnModel):
    """K-Nearest Neighbors model implementation"""
    pass

class GaussianNBModel(BaseSKLearnModel):
    """Gaussian Naive Bayes model implementation"""
    pass

class DecisionTreeModel(BaseSKLearnModel):
    """Decision Tree model implementation"""
    pass

class MLPClassifierModel(BaseSKLearnModel):
    """Multi-layer Perceptron model implementation"""
    pass

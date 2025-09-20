"""
Abstract Base Classes for Parkinson's Disease Detection Pipeline
Defines interfaces and common functionality for all pipeline components
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging

class BaseDataLoader(ABC):
    """
    Abstract base class for data loading operations
    Defines interface for loading and preparing datasets
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data loader with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data = None
        self.features = None
        self.target = None
        self.subjects = None
        self.feature_names = None
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Load data from source
        
        Returns:
            Loaded DataFrame
        """
        pass
    
    @abstractmethod
    def extract_subjects(self, data: pd.DataFrame) -> pd.Series:
        """
        Extract subject identifiers from data
        
        Args:
            data: Input DataFrame
            
        Returns:
            Series of subject identifiers
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate data integrity and structure
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if data is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix, target vector, and subject identifiers
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (X, y, subjects, feature_names)
        """
        pass
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about loaded data
        
        Returns:
            Dictionary with data information
        """
        if self.data is None:
            return {"status": "No data loaded"}
        
        return {
            "n_samples": len(self.data),
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "n_subjects": len(np.unique(self.subjects)) if self.subjects is not None else 0,
            "target_distribution": dict(pd.Series(self.target).value_counts()) if self.target is not None else {},
            "missing_values": self.data.isnull().sum().sum()
        }


class BasePreprocessor(ABC):
    """
    Abstract base class for data preprocessing operations
    Defines interface for feature scaling, selection, and transformation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.scaler = None
        self.feature_selector = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BasePreprocessor':
        """
        Fit preprocessor to data
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted preprocessor
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        pass
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit preprocessor and transform data
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            
        Returns:
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)
    
    @abstractmethod
    def get_feature_names(self, feature_names: List[str]) -> List[str]:
        """
        Get names of selected features after preprocessing
        
        Args:
            feature_names: Original feature names
            
        Returns:
            Selected feature names
        """
        pass
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about preprocessing operations
        
        Returns:
            Dictionary with preprocessing information
        """
        return {
            "scaler_type": type(self.scaler).__name__ if self.scaler else None,
            "feature_selector_type": type(self.feature_selector).__name__ if self.feature_selector else None,
            "is_fitted": self.is_fitted,
            "n_features_before": getattr(self, 'n_features_before', None),
            "n_features_after": getattr(self, 'n_features_after', None)
        }


class BaseModel(ABC):
    """
    Abstract base class for machine learning models
    Defines interface for model training, prediction, and evaluation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.is_fitted = False
        self.feature_importance = None
    
    @abstractmethod
    def build_model(self) -> BaseEstimator:
        """
        Build model instance based on configuration
        
        Returns:
            Model instance
        """
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """
        Train model on data
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance scores
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_type": type(self.model).__name__ if self.model else None,
            "is_fitted": self.is_fitted,
            "parameters": self.config.get('parameters', {}),
            "n_features": getattr(self, 'n_features', None)
        }


class BaseEvaluator(ABC):
    """
    Abstract base class for model evaluation
    Defines interface for performance metrics, visualization, and reporting
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}
        self.plots = {}
    
    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        pass
    
    @abstractmethod
    def cross_validate(self, model: BaseModel, X: np.ndarray, y: np.ndarray, 
                      cv_strategy: Any) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target vector
            cv_strategy: Cross-validation strategy
            
        Returns:
            Dictionary with CV results
        """
        pass
    
    @abstractmethod
    def plot_results(self, save_path: Optional[str] = None) -> Dict[str, str]:
        """
        Generate evaluation plots
        
        Args:
            save_path: Directory to save plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        pass
    
    @abstractmethod
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate evaluation report
        
        Args:
            save_path: Path to save report
            
        Returns:
            Report content or file path
        """
        pass
    
    def get_evaluation_info(self) -> Dict[str, Any]:
        """
        Get information about evaluation results
        
        Returns:
            Dictionary with evaluation information
        """
        return {
            "metrics_computed": list(self.results.keys()),
            "plots_generated": list(self.plots.keys()),
            "evaluation_complete": len(self.results) > 0
        }


class BasePipeline(ABC):
    """
    Abstract base class for complete ML pipeline
    Orchestrates data loading, preprocessing, modeling, and evaluation
    """
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline with configuration
        
        Args:
            config_path: Path to configuration file
        """
        from ..config.config_manager import ConfigManager
        
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.data_loader = None
        self.preprocessor = None
        self.model = None
        self.evaluator = None
        
        # Results storage
        self.results = {}
        self.predictions = {}
        self.feature_importance = None
    
    @abstractmethod
    def setup_components(self) -> None:
        """
        Setup pipeline components based on configuration
        """
        pass
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Run complete pipeline
        
        Returns:
            Dictionary with pipeline results
        """
        pass
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about pipeline components and results
        
        Returns:
            Dictionary with pipeline information
        """
        return {
            "config_path": self.config_manager.config_path,
            "data_loader": type(self.data_loader).__name__ if self.data_loader else None,
            "preprocessor": type(self.preprocessor).__name__ if self.preprocessor else None,
            "model": type(self.model).__name__ if self.model else None,
            "evaluator": type(self.evaluator).__name__ if self.evaluator else None,
            "results_available": len(self.results) > 0
        }

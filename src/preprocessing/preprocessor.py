"""
Preprocessing Module for Parkinson's Disease Detection
Handles feature scaling, selection, and transformation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif, chi2, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ..core.base import BasePreprocessor

class ParkinsonPreprocessor(BasePreprocessor):
    """
    Preprocessor specifically designed for Parkinson's disease detection
    Handles scaling, feature selection, and missing value imputation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.scaler_type = config.get('scaler', 'StandardScaler')
        self.handle_missing = config.get('handle_missing', 'drop')
        self.feature_selection = config.get('feature_selection')
        self.n_features = config.get('n_features')
        
        # Initialize components
        self.scaler = None
        self.feature_selector = None
        self.imputer = None
        self.n_features_before = None
        self.n_features_after = None
        
    def _create_scaler(self) -> Any:
        """
        Create scaler based on configuration
        
        Returns:
            Scaler instance
        """
        scaler_map = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'QuantileTransformer': QuantileTransformer(output_distribution='normal')
        }
        
        if self.scaler_type not in scaler_map:
            self.logger.warning(f"Unknown scaler type: {self.scaler_type}. Using StandardScaler.")
            return StandardScaler()
        
        return scaler_map[self.scaler_type]
    
    def _create_feature_selector(self, n_features: int, method: str) -> Any:
        """
        Create feature selector based on configuration
        
        Args:
            n_features: Number of features to select
            method: Selection method
            
        Returns:
            Feature selector instance
        """
        method_map = {
            'mutual_info': mutual_info_classif,
            'chi2': chi2,
            'f_classif': f_classif
        }
        
        if method not in method_map:
            self.logger.warning(f"Unknown selection method: {method}. Using mutual_info.")
            method = 'mutual_info'
        
        return SelectKBest(score_func=method_map[method], k=n_features)
    
    def _create_imputer(self) -> Any:
        """
        Create imputer based on configuration
        
        Returns:
            Imputer instance
        """
        strategy_map = {
            'impute_mean': 'mean',
            'impute_median': 'median',
            'impute_mode': 'most_frequent'
        }
        
        strategy = strategy_map.get(self.handle_missing, 'mean')
        return SimpleImputer(strategy=strategy)
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ParkinsonPreprocessor':
        """
        Fit preprocessor to data
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Fitting preprocessor to data")
        
        try:
            self.n_features_before = X.shape[1]
            
            # Handle missing values
            if self.handle_missing == 'drop':
                # Remove rows with missing values
                mask = ~np.isnan(X).any(axis=1)
                X = X[mask]
                if y is not None:
                    y = y[mask]
                self.logger.info(f"Removed {np.sum(~mask)} rows with missing values")
            elif self.handle_missing.startswith('impute'):
                # Impute missing values
                self.imputer = self._create_imputer()
                X = self.imputer.fit_transform(X)
                self.logger.info(f"Imputed missing values using {self.handle_missing}")
            
            # Create and fit scaler
            self.scaler = self._create_scaler()
            X_scaled = self.scaler.fit_transform(X)
            self.logger.info(f"Fitted scaler: {type(self.scaler).__name__}")
            
            # Feature selection
            if self.feature_selection is not None and self.n_features is not None:
                if self.n_features < X_scaled.shape[1]:
                    self.feature_selector = self._create_feature_selector(
                        self.n_features, self.feature_selection
                    )
                    self.feature_selector.fit(X_scaled, y)
                    self.logger.info(f"Fitted feature selector: {type(self.feature_selector).__name__}")
                    self.logger.info(f"Selected {self.n_features} features from {X_scaled.shape[1]}")
                else:
                    self.logger.info("Number of features to select >= total features. Skipping feature selection.")
            
            self.n_features_after = X_scaled.shape[1] if self.feature_selector is None else self.n_features
            self.is_fitted = True
            
            self.logger.info("Preprocessor fitting completed successfully")
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting preprocessor: {str(e)}")
            raise
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted preprocessor
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        self.logger.info("Transforming data using fitted preprocessor")
        
        try:
            # Handle missing values
            if self.handle_missing.startswith('impute') and self.imputer is not None:
                X = self.imputer.transform(X)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Feature selection
            if self.feature_selector is not None:
                X_scaled = self.feature_selector.transform(X_scaled)
            
            self.logger.info(f"Data transformed: {X.shape} -> {X_scaled.shape}")
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"Error transforming data: {str(e)}")
            raise
    
    def get_feature_names(self, feature_names: List[str]) -> List[str]:
        """
        Get names of selected features after preprocessing
        
        Args:
            feature_names: Original feature names
            
        Returns:
            Selected feature names
        """
        if not self.is_fitted:
            return feature_names
        
        if self.feature_selector is not None:
            # Get selected feature indices
            selected_indices = self.feature_selector.get_support(indices=True)
            selected_names = [feature_names[i] for i in selected_indices]
            self.logger.info(f"Selected features: {selected_names}")
            return selected_names
        
        return feature_names
    
    def get_feature_scores(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature selection scores
        
        Args:
            feature_names: Original feature names
            
        Returns:
            DataFrame with feature scores
        """
        if not self.is_fitted or self.feature_selector is None:
            return pd.DataFrame()
        
        try:
            scores = self.feature_selector.scores_
            selected = self.feature_selector.get_support()
            
            df = pd.DataFrame({
                'feature': feature_names,
                'score': scores,
                'selected': selected
            }).sort_values('score', ascending=False)
            
            self.logger.info("Feature selection scores computed")
            return df
            
        except Exception as e:
            self.logger.error(f"Error computing feature scores: {str(e)}")
            return pd.DataFrame()
    
    def get_scaling_info(self) -> Dict[str, Any]:
        """
        Get information about scaling operations
        
        Returns:
            Dictionary with scaling information
        """
        if not self.is_fitted or self.scaler is None:
            return {}
        
        info = {
            'scaler_type': type(self.scaler).__name__,
            'n_features': self.n_features_before
        }
        
        # Add scaler-specific information
        if hasattr(self.scaler, 'mean_'):
            info['mean'] = self.scaler.mean_.tolist()
        if hasattr(self.scaler, 'scale_'):
            info['scale'] = self.scaler.scale_.tolist()
        if hasattr(self.scaler, 'min_'):
            info['min'] = self.scaler.min_.tolist()
        if hasattr(self.scaler, 'max_'):
            info['max'] = self.scaler.max_.tolist()
        
        return info
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform data (useful for interpretation)
        
        Args:
            X: Transformed feature matrix
            
        Returns:
            Original scale feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse transforming data")
        
        try:
            # Inverse feature selection (not possible with SelectKBest)
            if self.feature_selector is not None:
                self.logger.warning("Cannot inverse transform feature selection. Returning scaled data.")
                return X
            
            # Inverse scaling
            X_original = self.scaler.inverse_transform(X)
            self.logger.info("Data inverse transformed successfully")
            return X_original
            
        except Exception as e:
            self.logger.error(f"Error inverse transforming data: {str(e)}")
            raise

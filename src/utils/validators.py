"""
Comprehensive Input Validation System for Parkinson's Disease Detection
Validates data format, missing values, feature ranges, and more
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import re

from .error_handlers import ValidationError, DataLoadError, ConfigurationError

class DataValidator:
    """
    Comprehensive data validation for Parkinson's disease detection
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.validation_rules = {}
        self.setup_default_rules()
    
    def setup_default_rules(self) -> None:
        """Setup default validation rules for Parkinson's data"""
        self.validation_rules = {
            'required_columns': ['name', 'status'],
            'target_values': [0, 1],
            'feature_ranges': {
                'MDVP:Fo(Hz)': (50, 300),  # Fundamental frequency range
                'MDVP:Fhi(Hz)': (100, 500),  # Highest frequency range
                'MDVP:Flo(Hz)': (50, 200),   # Lowest frequency range
                'MDVP:Jitter(%)': (0, 10),   # Jitter percentage range
                'MDVP:Jitter(Abs)': (0, 0.01),  # Absolute jitter range
                'NHR': (0, 1),              # Noise-to-harmonics ratio
                'HNR': (0, 50),             # Harmonics-to-noise ratio
            },
            'min_samples_per_subject': 1,
            'max_missing_percentage': 50.0,
            'subject_id_pattern': r'^S\d{2}$'  # Pattern for subject IDs like S01, S02
        }
    
    def validate_data_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate data file before loading
        
        Args:
            file_path: Path to data file
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }
        
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                results['valid'] = False
                results['errors'].append(f"File does not exist: {file_path}")
                return results
            
            # Check file size
            file_size = file_path.stat().st_size
            results['file_info']['size_bytes'] = file_size
            
            if file_size == 0:
                results['valid'] = False
                results['errors'].append("File is empty")
                return results
            
            # Check file extension
            if file_path.suffix.lower() not in ['.csv', '.data']:
                results['warnings'].append(f"Unexpected file extension: {file_path.suffix}")
            
            # Try to read first few lines to check format
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        results['valid'] = False
                        results['errors'].append("File appears to be empty or corrupted")
                        return results
                    
                    # Check if it looks like CSV
                    if ',' not in first_line and '\t' not in first_line:
                        results['warnings'].append("File may not be in CSV format")
                    
                    results['file_info']['first_line'] = first_line[:100] + "..." if len(first_line) > 100 else first_line
                    
            except Exception as e:
                results['valid'] = False
                results['errors'].append(f"Cannot read file: {str(e)}")
                return results
            
            self.logger.info(f"Data file validation completed: {file_path}")
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"File validation error: {str(e)}")
        
        return results
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate loaded DataFrame
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'data_info': {}
        }
        
        try:
            # Basic structure validation
            if df.empty:
                results['valid'] = False
                results['errors'].append("DataFrame is empty")
                return results
            
            results['data_info']['shape'] = df.shape
            results['data_info']['columns'] = list(df.columns)
            
            # Check required columns
            missing_columns = []
            for col in self.validation_rules['required_columns']:
                if col not in df.columns:
                    missing_columns.append(col)
            
            if missing_columns:
                results['valid'] = False
                results['errors'].append(f"Missing required columns: {missing_columns}")
            
            # Check target column values
            if 'status' in df.columns:
                target_values = df['status'].unique()
                invalid_targets = [v for v in target_values if v not in self.validation_rules['target_values']]
                
                if invalid_targets:
                    results['valid'] = False
                    results['errors'].append(f"Invalid target values: {invalid_targets}")
                
                # Check class distribution
                class_counts = df['status'].value_counts()
                results['data_info']['class_distribution'] = class_counts.to_dict()
                
                # Check for class imbalance
                min_class_count = class_counts.min()
                max_class_count = class_counts.max()
                imbalance_ratio = max_class_count / min_class_count
                
                if imbalance_ratio > 5:
                    results['warnings'].append(f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f})")
                elif imbalance_ratio > 2:
                    results['warnings'].append(f"Class imbalance detected (ratio: {imbalance_ratio:.1f})")
            
            # Check for missing values
            missing_counts = df.isnull().sum()
            total_missing = missing_counts.sum()
            missing_percentage = (total_missing / (df.shape[0] * df.shape[1])) * 100
            
            results['data_info']['missing_values'] = {
                'total': int(total_missing),
                'percentage': missing_percentage,
                'by_column': missing_counts[missing_counts > 0].to_dict()
            }
            
            if missing_percentage > self.validation_rules['max_missing_percentage']:
                results['valid'] = False
                results['errors'].append(f"Too many missing values: {missing_percentage:.1f}%")
            
            # Validate feature ranges
            feature_range_errors = self._validate_feature_ranges(df)
            if feature_range_errors:
                results['warnings'].extend(feature_range_errors)
            
            # Check for duplicate rows
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                results['warnings'].append(f"Found {duplicate_count} duplicate rows")
            
            # Check for constant columns
            constant_columns = []
            for col in df.columns:
                if df[col].nunique() <= 1:
                    constant_columns.append(col)
            
            if constant_columns:
                results['warnings'].append(f"Found constant columns: {constant_columns}")
            
            self.logger.info(f"DataFrame validation completed: {df.shape[0]} rows, {df.shape[1]} columns")
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"DataFrame validation error: {str(e)}")
        
        return results
    
    def _validate_feature_ranges(self, df: pd.DataFrame) -> List[str]:
        """Validate feature value ranges"""
        warnings = []
        
        for feature, (min_val, max_val) in self.validation_rules['feature_ranges'].items():
            if feature in df.columns:
                feature_data = df[feature].dropna()
                
                if len(feature_data) > 0:
                    actual_min = feature_data.min()
                    actual_max = feature_data.max()
                    
                    if actual_min < min_val or actual_max > max_val:
                        warnings.append(
                            f"Feature {feature} has values outside expected range "
                            f"[{min_val}, {max_val}]: actual range [{actual_min:.3f}, {actual_max:.3f}]"
                        )
        
        return warnings
    
    def validate_subject_ids(self, subject_ids: Union[List[str], np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Validate subject ID format and distribution
        
        Args:
            subject_ids: List or array of subject IDs
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'subject_info': {}
        }
        
        try:
            if isinstance(subject_ids, (np.ndarray, pd.Series)):
                subject_ids = subject_ids.tolist()
            
            unique_subjects = list(set(subject_ids))
            results['subject_info']['total_subjects'] = len(unique_subjects)
            results['subject_info']['unique_subjects'] = unique_subjects
            
            # Check subject ID format
            pattern = re.compile(self.validation_rules['subject_id_pattern'])
            invalid_format = [sid for sid in unique_subjects if not pattern.match(str(sid))]
            
            if invalid_format:
                results['warnings'].append(f"Subject IDs with unexpected format: {invalid_format}")
            
            # Check subject distribution
            from collections import Counter
            subject_counts = Counter(subject_ids)
            results['subject_info']['subject_counts'] = dict(subject_counts)
            
            min_samples = min(subject_counts.values())
            max_samples = max(subject_counts.values())
            
            if min_samples < self.validation_rules['min_samples_per_subject']:
                results['warnings'].append(f"Some subjects have fewer than {self.validation_rules['min_samples_per_subject']} samples")
            
            # Check for extreme imbalance in subject samples
            if max_samples / min_samples > 3:
                results['warnings'].append("Significant imbalance in samples per subject")
            
            self.logger.info(f"Subject validation completed: {len(unique_subjects)} unique subjects")
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Subject validation error: {str(e)}")
        
        return results
    
    def validate_feature_matrix(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Validate feature matrix
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'feature_info': {}
        }
        
        try:
            results['feature_info']['shape'] = X.shape
            results['feature_info']['feature_names'] = feature_names
            
            # Check for NaN values
            nan_count = np.isnan(X).sum()
            if nan_count > 0:
                results['warnings'].append(f"Found {nan_count} NaN values in feature matrix")
            
            # Check for infinite values
            inf_count = np.isinf(X).sum()
            if inf_count > 0:
                results['warnings'].append(f"Found {inf_count} infinite values in feature matrix")
            
            # Check for constant features
            constant_features = []
            for i, feature_name in enumerate(feature_names):
                if np.std(X[:, i]) == 0:
                    constant_features.append(feature_name)
            
            if constant_features:
                results['warnings'].append(f"Found constant features: {constant_features}")
            
            # Check feature ranges
            for i, feature_name in enumerate(feature_names):
                if feature_name in self.validation_rules['feature_ranges']:
                    min_val, max_val = self.validation_rules['feature_ranges'][feature_name]
                    feature_data = X[:, i]
                    feature_data = feature_data[~np.isnan(feature_data)]  # Remove NaN values
                    
                    if len(feature_data) > 0:
                        actual_min = np.min(feature_data)
                        actual_max = np.max(feature_data)
                        
                        if actual_min < min_val or actual_max > max_val:
                            results['warnings'].append(
                                f"Feature {feature_name} has values outside expected range "
                                f"[{min_val}, {max_val}]: actual range [{actual_min:.3f}, {actual_max:.3f}]"
                            )
            
            self.logger.info(f"Feature matrix validation completed: {X.shape}")
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Feature matrix validation error: {str(e)}")
        
        return results

class ModelValidator:
    """
    Validation for model parameters and performance
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_model_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate model configuration
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check required fields
            required_fields = ['algorithm']
            missing_fields = [field for field in required_fields if field not in config]
            
            if missing_fields:
                results['valid'] = False
                results['errors'].append(f"Missing required fields: {missing_fields}")
            
            # Validate algorithm
            if 'algorithm' in config:
                valid_algorithms = [
                    'RandomForest', 'GradientBoosting', 'ExtraTrees', 'SVM',
                    'LogisticRegression', 'KNeighbors', 'GaussianNB', 'DecisionTree', 'MLPClassifier'
                ]
                
                if config['algorithm'] not in valid_algorithms:
                    results['valid'] = False
                    results['errors'].append(f"Invalid algorithm: {config['algorithm']}")
            
            # Validate parameters
            if 'parameters' in config:
                param_validation = self._validate_parameters(config['algorithm'], config['parameters'])
                results['errors'].extend(param_validation['errors'])
                results['warnings'].extend(param_validation['warnings'])
            
            self.logger.info(f"Model configuration validation completed")
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Model configuration validation error: {str(e)}")
        
        return results
    
    def _validate_parameters(self, algorithm: str, parameters: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate model parameters"""
        errors = []
        warnings = []
        
        if algorithm == 'RandomForest':
            if 'n_estimators' in parameters:
                n_estimators = parameters['n_estimators']
                if not isinstance(n_estimators, int) or n_estimators <= 0:
                    errors.append("n_estimators must be a positive integer")
                elif n_estimators > 1000:
                    warnings.append("Large number of estimators may cause slow training")
            
            if 'max_depth' in parameters:
                max_depth = parameters['max_depth']
                if max_depth is not None and (not isinstance(max_depth, int) or max_depth <= 0):
                    errors.append("max_depth must be a positive integer or None")
        
        elif algorithm == 'SVM':
            if 'C' in parameters:
                C = parameters['C']
                if not isinstance(C, (int, float)) or C <= 0:
                    errors.append("C must be a positive number")
                elif C > 100:
                    warnings.append("Large C value may cause overfitting")
        
        return {'errors': errors, 'warnings': warnings}

class ConfigValidator:
    """
    Validation for configuration files
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration dictionary
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check required sections
            required_sections = ['data', 'cross_validation', 'model', 'preprocessing', 'evaluation']
            missing_sections = [section for section in required_sections if section not in config]
            
            if missing_sections:
                results['valid'] = False
                results['errors'].append(f"Missing required sections: {missing_sections}")
            
            # Validate data section
            if 'data' in config:
                data_validation = self._validate_data_config(config['data'])
                results['errors'].extend(data_validation['errors'])
                results['warnings'].extend(data_validation['warnings'])
            
            # Validate cross-validation section
            if 'cross_validation' in config:
                cv_validation = self._validate_cv_config(config['cross_validation'])
                results['errors'].extend(cv_validation['errors'])
                results['warnings'].extend(cv_validation['warnings'])
            
            # Validate model section
            if 'model' in config:
                model_validator = ModelValidator(self.logger)
                model_validation = model_validator.validate_model_config(config['model'])
                results['errors'].extend(model_validation['errors'])
                results['warnings'].extend(model_validation['warnings'])
            
            self.logger.info("Configuration validation completed")
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Configuration validation error: {str(e)}")
        
        return results
    
    def _validate_data_config(self, data_config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate data configuration section"""
        errors = []
        warnings = []
        
        if 'path' not in data_config:
            errors.append("Data path not specified")
        
        if 'target_column' not in data_config:
            errors.append("Target column not specified")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_cv_config(self, cv_config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate cross-validation configuration section"""
        errors = []
        warnings = []
        
        if 'n_splits' not in cv_config:
            errors.append("Number of CV splits not specified")
        elif not isinstance(cv_config['n_splits'], int) or cv_config['n_splits'] <= 0:
            errors.append("Number of CV splits must be a positive integer")
        elif cv_config['n_splits'] > 20:
            warnings.append("Large number of CV splits may cause slow execution")
        
        if 'strategy' in cv_config:
            valid_strategies = ['StratifiedGroupKFold', 'GroupKFold', 'StratifiedKFold', 'KFold']
            if cv_config['strategy'] not in valid_strategies:
                errors.append(f"Invalid CV strategy: {cv_config['strategy']}")
        
        return {'errors': errors, 'warnings': warnings}

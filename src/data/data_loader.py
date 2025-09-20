"""
Data Loading Module for Parkinson's Disease Detection
Handles CSV loading, subject extraction, and data validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import re

from ..core.base import BaseDataLoader

class ParkinsonDataLoader(BaseDataLoader):
    """
    Data loader specifically designed for Parkinson's disease detection dataset
    Handles subject extraction from recording names and data validation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data loader with configuration
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.data_path = config['path']
        self.target_column = config.get('target_column', 'status')
        self.subject_id_column = config.get('subject_id_column', 'subject_id')
        self.feature_columns = config.get('feature_columns')
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Returns:
            Loaded DataFrame
        """
        try:
            self.logger.info(f"Loading data from: {self.data_path}")
            
            # Check if file exists
            if not Path(self.data_path).exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            # Load CSV
            self.data = pd.read_csv(self.data_path)
            self.logger.info(f"Data loaded successfully: {self.data.shape[0]} samples, {self.data.shape[1]} columns")
            
            # Extract subject IDs
            self.data[self.subject_id_column] = self.extract_subjects(self.data)
            
            # Validate data
            if not self.validate_data(self.data):
                raise ValueError("Data validation failed")
            
            # Prepare features
            self.features, self.target, self.subjects, self.feature_names = self.prepare_features(self.data)
            
            self.logger.info("Data preparation completed successfully")
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def extract_subjects(self, data: pd.DataFrame) -> pd.Series:
        """
        Extract subject identifiers from recording names
        Converts 'phon_R01_S01_1' to 'S01' to group recordings by person
        
        Args:
            data: Input DataFrame
            
        Returns:
            Series of subject identifiers
        """
        self.logger.info("Extracting subject identifiers from recording names")
        
        def extract_subject_id(name: str) -> str:
            """
            Extract subject ID from recording name
            Handles various naming patterns: phon_R01_S01_1 -> S01
            """
            try:
                # Pattern 1: phon_R01_S01_1 -> S01
                pattern1 = r'phon_R\d+_S(\d+)_\d+'
                match = re.search(pattern1, name)
                if match:
                    return f"S{match.group(1).zfill(2)}"
                
                # Pattern 2: S01_1 -> S01
                pattern2 = r'S(\d+)_\d+'
                match = re.search(pattern2, name)
                if match:
                    return f"S{match.group(1).zfill(2)}"
                
                # Pattern 3: S01 -> S01
                pattern3 = r'S(\d+)'
                match = re.search(pattern3, name)
                if match:
                    return f"S{match.group(1).zfill(2)}"
                
                # Fallback: return original name
                self.logger.warning(f"Could not extract subject ID from: {name}")
                return name
                
            except Exception as e:
                self.logger.warning(f"Error extracting subject ID from {name}: {str(e)}")
                return name
        
        # Apply extraction to name column
        if 'name' in data.columns:
            subject_ids = data['name'].apply(extract_subject_id)
        else:
            raise ValueError("'name' column not found in data")
        
        # Log subject distribution
        subject_counts = subject_ids.value_counts()
        self.logger.info(f"Extracted {len(subject_counts)} unique subjects")
        self.logger.info(f"Subject distribution: {subject_counts.to_dict()}")
        
        return subject_ids
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate data integrity and structure
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if data is valid, False otherwise
        """
        self.logger.info("Validating data integrity and structure")
        
        try:
            # Check required columns
            required_columns = ['name', self.target_column]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check for empty dataset
            if len(data) == 0:
                self.logger.error("Dataset is empty")
                return False
            
            # Check target column values
            target_values = data[self.target_column].unique()
            if not all(val in [0, 1] for val in target_values):
                self.logger.error(f"Target column contains invalid values: {target_values}")
                return False
            
            # Check for missing values in target
            if data[self.target_column].isnull().any():
                self.logger.error("Target column contains missing values")
                return False
            
            # Check subject ID extraction
            if self.subject_id_column in data.columns:
                unique_subjects = data[self.subject_id_column].nunique()
                if unique_subjects < 2:
                    self.logger.error("Less than 2 unique subjects found")
                    return False
                
                # Check subject-target relationship
                subject_target_counts = data.groupby(self.subject_id_column)[self.target_column].nunique()
                mixed_subjects = subject_target_counts[subject_target_counts > 1]
                if len(mixed_subjects) > 0:
                    self.logger.warning(f"Subjects with mixed labels: {mixed_subjects.index.tolist()}")
            
            # Check feature columns
            if self.feature_columns is not None:
                missing_features = [col for col in self.feature_columns if col not in data.columns]
                if missing_features:
                    self.logger.error(f"Missing feature columns: {missing_features}")
                    return False
            
            self.logger.info("Data validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            return False
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix, target vector, and subject identifiers
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (X, y, subjects, feature_names)
        """
        self.logger.info("Preparing features for machine learning")
        
        try:
            # Determine feature columns
            if self.feature_columns is not None:
                feature_cols = self.feature_columns
            else:
                # Auto-detect feature columns (exclude name, target, and subject_id)
                exclude_cols = ['name', self.target_column, self.subject_id_column]
                feature_cols = [col for col in data.columns if col not in exclude_cols]
            
            self.logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
            
            # Extract features
            X = data[feature_cols].values
            y = data[self.target_column].values
            subjects = data[self.subject_id_column].values
            
            # Check for missing values in features
            missing_count = np.isnan(X).sum()
            if missing_count > 0:
                self.logger.warning(f"Found {missing_count} missing values in features")
            
            # Log data shapes
            self.logger.info(f"Feature matrix shape: {X.shape}")
            self.logger.info(f"Target vector shape: {y.shape}")
            self.logger.info(f"Subjects shape: {subjects.shape}")
            
            # Log class distribution
            unique, counts = np.unique(y, return_counts=True)
            class_dist = dict(zip(unique, counts))
            self.logger.info(f"Class distribution: {class_dist}")
            
            return X, y, subjects, feature_cols
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def get_subject_info(self) -> Dict[str, Any]:
        """
        Get detailed information about subjects
        
        Returns:
            Dictionary with subject information
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        subject_info = {}
        
        # Subject-level statistics
        subject_stats = self.data.groupby(self.subject_id_column).agg({
            self.target_column: ['count', 'mean', 'std'],
            'name': 'count'
        }).round(3)
        
        subject_info['subject_stats'] = subject_stats.to_dict()
        
        # Subject distribution
        subject_counts = self.data[self.subject_id_column].value_counts()
        subject_info['subject_counts'] = subject_counts.to_dict()
        
        # Target distribution by subject
        target_by_subject = self.data.groupby(self.subject_id_column)[self.target_column].apply(
            lambda x: x.value_counts().to_dict()
        ).to_dict()
        subject_info['target_by_subject'] = target_by_subject
        
        return subject_info
    
    def save_processed_data(self, output_path: str) -> None:
        """
        Save processed data to file
        
        Args:
            output_path: Path to save processed data
        """
        if self.data is None:
            self.logger.error("No data loaded to save")
            return
        
        try:
            # Create output directory
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save processed data
            self.data.to_csv(output_path, index=False)
            self.logger.info(f"Processed data saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving processed data: {str(e)}")
            raise

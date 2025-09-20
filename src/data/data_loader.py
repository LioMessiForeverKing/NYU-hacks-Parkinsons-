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
from ..utils.logger import get_logger
from ..utils.validators import DataValidator
from ..utils.error_handlers import DataLoadError, ValidationError, create_error_handler
from ..utils.progress import TrainingProgressTracker

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
        
        # Enhanced logging and validation
        self.logger = get_logger(self.__class__.__name__)
        self.validator = DataValidator(self.logger)
        self.error_handler = create_error_handler("data", self.logger)
        self.progress_tracker = TrainingProgressTracker(self.logger)
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file with enhanced validation and error handling
        
        Returns:
            Loaded DataFrame
        """
        self.progress_tracker.start_phase("Data Loading", 5)
        
        try:
            # Step 1: Validate file before loading
            self.progress_tracker.update_phase(1, "Validating data file")
            file_validation = self.validator.validate_data_file(self.data_path)
            
            if not file_validation['valid']:
                error_msg = f"Data file validation failed: {'; '.join(file_validation['errors'])}"
                raise DataLoadError(error_msg, error_code="FILE_VALIDATION_FAILED", 
                                  details=file_validation)
            
            # Log warnings if any
            for warning in file_validation['warnings']:
                self.logger.warning(f"File validation warning: {warning}")
            
            # Step 2: Load CSV with error handling
            self.progress_tracker.update_phase(1, "Loading CSV file")
            self.logger.start_operation("CSV Loading")
            
            try:
                self.data = pd.read_csv(self.data_path)
                self.logger.end_operation("CSV Loading", 
                                        samples=self.data.shape[0], 
                                        columns=self.data.shape[1])
            except Exception as e:
                raise DataLoadError(f"Failed to load CSV file: {str(e)}", 
                                  error_code="CSV_LOAD_FAILED")
            
            # Step 3: Validate DataFrame structure
            self.progress_tracker.update_phase(1, "Validating DataFrame structure")
            df_validation = self.validator.validate_dataframe(self.data)
            
            if not df_validation['valid']:
                error_msg = f"DataFrame validation failed: {'; '.join(df_validation['errors'])}"
                raise ValidationError(error_msg, error_code="DATAFRAME_VALIDATION_FAILED",
                                    details=df_validation)
            
            # Log warnings and data info
            for warning in df_validation['warnings']:
                self.logger.warning(f"DataFrame validation warning: {warning}")
            
            self.logger.log_data_info(df_validation['data_info'])
            
            # Step 4: Extract subject IDs
            self.progress_tracker.update_phase(1, "Extracting subject IDs")
            self.data[self.subject_id_column] = self.extract_subjects(self.data)
            
            # Step 5: Final validation and feature preparation
            self.progress_tracker.update_phase(1, "Preparing features")
            self.features, self.target, self.subjects, self.feature_names = self.prepare_features(self.data)
            
            self.progress_tracker.finish_phase("Data loading completed successfully")
            return self.data
            
        except (DataLoadError, ValidationError) as e:
            self.error_handler.handle_error(e, "Data loading")
            self.progress_tracker.finish_phase("Data loading failed")
            raise
        except Exception as e:
            self.error_handler.handle_error(e, "Data loading")
            self.progress_tracker.finish_phase("Data loading failed")
            raise DataLoadError(f"Unexpected error during data loading: {str(e)}", 
                              error_code="UNEXPECTED_ERROR")
    
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
        Validate data integrity and structure with enhanced error reporting
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if data is valid, False otherwise
        """
        self.logger.info("Validating data integrity and structure")
        
        try:
            # Use the comprehensive validator
            validation_results = self.validator.validate_dataframe(data)
            
            # Log all validation results
            if validation_results['errors']:
                for error in validation_results['errors']:
                    self.logger.error(f"Validation error: {error}")
                return False
            
            if validation_results['warnings']:
                for warning in validation_results['warnings']:
                    self.logger.warning(f"Validation warning: {warning}")
            
            # Additional subject-specific validation
            if self.subject_id_column in data.columns:
                subject_validation = self.validator.validate_subject_ids(data[self.subject_id_column])
                
                if not subject_validation['valid']:
                    for error in subject_validation['errors']:
                        self.logger.error(f"Subject validation error: {error}")
                    return False
                
                if subject_validation['warnings']:
                    for warning in subject_validation['warnings']:
                        self.logger.warning(f"Subject validation warning: {warning}")
                
                # Check for subject leakage (subjects with mixed labels)
                subject_target_counts = data.groupby(self.subject_id_column)[self.target_column].nunique()
                mixed_subjects = subject_target_counts[subject_target_counts > 1]
                if len(mixed_subjects) > 0:
                    self.logger.warning(f"Subjects with mixed labels detected: {mixed_subjects.index.tolist()}")
                    self.logger.warning("This may indicate data quality issues or legitimate mixed cases")
            
            # Check feature columns if specified
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
        Prepare feature matrix, target vector, and subject identifiers with enhanced validation
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (X, y, subjects, feature_names)
        """
        self.logger.info("Preparing features for machine learning")
        
        try:
            # Step 1: Determine feature columns
            if self.feature_columns is not None:
                feature_cols = self.feature_columns
            else:
                # Auto-detect feature columns (exclude name, target, and subject_id)
                exclude_cols = ['name', self.target_column, self.subject_id_column]
                feature_cols = [col for col in data.columns if col not in exclude_cols]
            
            self.logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
            
            # Step 2: Extract features with validation
            X = data[feature_cols].values
            y = data[self.target_column].values
            subjects = data[self.subject_id_column].values
            
            # Step 3: Validate feature matrix
            feature_validation = self.validator.validate_feature_matrix(X, feature_cols)
            
            if not feature_validation['valid']:
                error_msg = f"Feature matrix validation failed: {'; '.join(feature_validation['errors'])}"
                raise ValidationError(error_msg, error_code="FEATURE_VALIDATION_FAILED",
                                    details=feature_validation)
            
            # Log warnings
            for warning in feature_validation['warnings']:
                self.logger.warning(f"Feature validation warning: {warning}")
            
            # Step 4: Check for missing values in features
            missing_count = np.isnan(X).sum()
            if missing_count > 0:
                self.logger.warning(f"Found {missing_count} missing values in features")
                self.logger.warning("Consider using imputation strategies in preprocessing")
            
            # Step 5: Log comprehensive data information
            self.logger.info(f"Feature matrix shape: {X.shape}")
            self.logger.info(f"Target vector shape: {y.shape}")
            self.logger.info(f"Subjects shape: {subjects.shape}")
            
            # Log class distribution with detailed analysis
            unique, counts = np.unique(y, return_counts=True)
            class_dist = dict(zip(unique, counts))
            self.logger.info(f"Class distribution: {class_dist}")
            
            # Calculate and log class imbalance ratio
            if len(counts) == 2:
                imbalance_ratio = max(counts) / min(counts)
                self.logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")
                if imbalance_ratio > 2:
                    self.logger.warning(f"Significant class imbalance detected (ratio: {imbalance_ratio:.2f})")
                    self.logger.warning("Consider using class balancing strategies")
            
            # Log feature statistics
            self.logger.info("Feature statistics:")
            for i, feature_name in enumerate(feature_cols):
                feature_data = X[:, i]
                feature_data = feature_data[~np.isnan(feature_data)]  # Remove NaN values
                if len(feature_data) > 0:
                    self.logger.info(f"  {feature_name}: mean={np.mean(feature_data):.3f}, "
                                   f"std={np.std(feature_data):.3f}, "
                                   f"range=[{np.min(feature_data):.3f}, {np.max(feature_data):.3f}]")
            
            return X, y, subjects, feature_cols
            
        except ValidationError as e:
            self.error_handler.handle_error(e, "Feature preparation")
            raise
        except Exception as e:
            self.error_handler.handle_error(e, "Feature preparation")
            raise ProcessingError(f"Error preparing features: {str(e)}", 
                                error_code="FEATURE_PREPARATION_FAILED")
    
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

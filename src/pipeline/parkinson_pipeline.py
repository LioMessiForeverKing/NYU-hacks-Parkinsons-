"""
Main Pipeline for Parkinson's Disease Detection
Orchestrates data loading, preprocessing, modeling, and evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.pipeline import Pipeline as SKPipeline

from ..core.base import BasePipeline
from ..config.config_manager import ConfigManager
from ..data.data_loader import ParkinsonDataLoader
from ..preprocessing.preprocessor import ParkinsonPreprocessor
from ..models.model_factory import ModelFactory
from ..evaluation.evaluator import ParkinsonEvaluator

class ParkinsonPipeline(BasePipeline):
    """
    Complete pipeline for Parkinson's disease detection
    Orchestrates all components with subject-independent validation
    """
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline with configuration
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__(config_path)
        self.setup_components()
        self.create_output_directories()
    
    def setup_components(self) -> None:
        """
        Setup pipeline components based on configuration
        """
        self.logger.info("Setting up pipeline components")
        
        try:
            # Initialize data loader
            data_config = self.config_manager.get_data_config()
            self.data_loader = ParkinsonDataLoader(data_config)
            
            # Initialize preprocessor
            preprocessing_config = self.config_manager.get_preprocessing_config()
            self.preprocessor = ParkinsonPreprocessor(preprocessing_config)
            
            # Initialize model
            model_config = self.config_manager.get_model_config()
            self.model = ModelFactory.create_model(model_config)
            
            # Initialize evaluator
            evaluation_config = self.config_manager.get_evaluation_config()
            self.evaluator = ParkinsonEvaluator(evaluation_config)
            
            self.logger.info("Pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up components: {str(e)}")
            raise
    
    def create_output_directories(self) -> None:
        """Create output directories based on configuration"""
        self.config_manager.create_output_directories()
    
    def run(self) -> Dict[str, Any]:
        """
        Run complete pipeline
        
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info("Starting Parkinson's Disease Detection Pipeline")
        
        try:
            # Step 1: Load and prepare data
            self.logger.info("Step 1: Loading and preparing data")
            data = self.data_loader.load_data()
            X, y, subjects, feature_names = self.data_loader.prepare_features(data)
            
            # Step 2: Setup cross-validation strategy
            self.logger.info("Step 2: Setting up cross-validation strategy")
            cv_strategy = self._setup_cross_validation()
            
            # Step 3: Preprocess data
            self.logger.info("Step 3: Preprocessing data")
            X_processed = self.preprocessor.fit_transform(X, y)
            processed_feature_names = self.preprocessor.get_feature_names(feature_names)
            
            # Step 4: Train model
            self.logger.info("Step 4: Training model")
            self.model.fit(X_processed, y)
            
            # Step 5: Cross-validation evaluation
            self.logger.info("Step 5: Cross-validation evaluation")
            cv_results = self.evaluator.cross_validate(self.model, X_processed, y, cv_strategy, subjects)
            
            # Step 6: Final evaluation
            self.logger.info("Step 6: Final evaluation")
            y_pred = self.model.predict(X_processed)
            y_prob = self.model.predict_proba(X_processed)
            # Use only the probability for the positive class (Parkinson's)
            y_prob_positive = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]
            evaluation_results = self.evaluator.evaluate(y, y_pred, y_prob_positive)
            
            # Step 7: Feature importance analysis
            self.logger.info("Step 7: Feature importance analysis")
            feature_importance = self.model.get_feature_importance(processed_feature_names)
            if not feature_importance.empty:
                self.evaluator.set_feature_importance_data(feature_importance)
            
            # Step 8: Generate visualizations and reports
            self.logger.info("Step 8: Generating visualizations and reports")
            output_config = self.config_manager.get_output_config()
            plots_dir = output_config.get('plots_dir', 'plots')
            reports_dir = output_config.get('reports_dir', 'reports')
            
            plots = self.evaluator.plot_results(plots_dir)
            report_path = self.evaluator.generate_report(f"{reports_dir}/evaluation_report.md")
            
            # Step 9: Save model and predictions
            if output_config.get('save_model', True):
                self._save_model()
            
            if output_config.get('save_predictions', True):
                self._save_predictions(y, y_pred, y_prob)
            
            # Compile results
            self.results = {
                'data_info': self.data_loader.get_data_info(),
                'preprocessing_info': self.preprocessor.get_preprocessing_info(),
                'model_info': self.model.get_model_info(),
                'evaluation_info': self.evaluator.get_evaluation_info(),
                'cross_validation': cv_results,
                'evaluation': evaluation_results,
                'feature_importance': feature_importance.to_dict() if not feature_importance.empty else {},
                'plots': plots,
                'report_path': report_path,
                'pipeline_info': self.get_pipeline_info()
            }
            
            # Log final results
            self._log_final_results()
            
            self.logger.info("Pipeline completed successfully")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _setup_cross_validation(self) -> Any:
        """
        Setup cross-validation strategy based on configuration
        
        Returns:
            Cross-validation strategy
        """
        cv_config = self.config_manager.get_cv_config()
        strategy = cv_config.get('strategy', 'StratifiedGroupKFold')
        n_splits = cv_config.get('n_splits', 10)
        shuffle = cv_config.get('shuffle', True)
        random_state = cv_config.get('random_state', 42)
        
        if strategy == 'StratifiedGroupKFold':
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        elif strategy == 'GroupKFold':
            return GroupKFold(n_splits=n_splits)
        else:
            raise ValueError(f"Unknown CV strategy: {strategy}")
    
    def _save_model(self) -> None:
        """Save trained model"""
        try:
            output_config = self.config_manager.get_output_config()
            models_dir = output_config.get('models_dir', 'models')
            model_path = Path(models_dir) / 'trained_model.pkl'
            
            import joblib
            joblib.dump(self.model, model_path)
            self.logger.info(f"Model saved to: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
    
    def _save_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
        """Save predictions"""
        try:
            output_config = self.config_manager.get_output_config()
            results_dir = output_config.get('results_dir', 'results')
            
            # Handle both 1D and 2D probability arrays
            if y_prob.ndim == 1:
                predictions_df = pd.DataFrame({
                    'true_label': y_true,
                    'predicted_label': y_pred,
                    'probability_positive': y_prob
                })
            else:
                predictions_df = pd.DataFrame({
                    'true_label': y_true,
                    'predicted_label': y_pred,
                    'probability_healthy': y_prob[:, 0],
                    'probability_parkinsons': y_prob[:, 1]
                })
            
            predictions_path = Path(results_dir) / 'predictions.csv'
            predictions_df.to_csv(predictions_path, index=False)
            self.logger.info(f"Predictions saved to: {predictions_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving predictions: {str(e)}")
    
    def _log_final_results(self) -> None:
        """Log final pipeline results"""
        if not self.results:
            return
        
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE RESULTS SUMMARY")
        self.logger.info("=" * 60)
        
        # Data info
        data_info = self.results.get('data_info', {})
        self.logger.info(f"Samples: {data_info.get('n_samples', 'N/A')}")
        self.logger.info(f"Features: {data_info.get('n_features', 'N/A')}")
        self.logger.info(f"Subjects: {data_info.get('n_subjects', 'N/A')}")
        
        # Model info
        model_info = self.results.get('model_info', {})
        self.logger.info(f"Model: {model_info.get('model_type', 'N/A')}")
        
        # Evaluation results
        evaluation = self.results.get('evaluation', {})
        if evaluation:
            self.logger.info("Performance Metrics:")
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                if metric in evaluation:
                    self.logger.info(f"  {metric.upper()}: {evaluation[metric]:.3f}")
        
        # Clinical metrics
        if 'sensitivity' in evaluation and 'specificity' in evaluation:
            self.logger.info("Clinical Metrics:")
            self.logger.info(f"  Sensitivity: {evaluation['sensitivity']:.3f}")
            self.logger.info(f"  Specificity: {evaluation['specificity']:.3f}")
        
        # Cross-validation results
        cv_results = self.results.get('cross_validation', {})
        if cv_results:
            self.logger.info("Cross-Validation Results:")
            for metric, stats in cv_results.items():
                self.logger.info(f"  {metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        
        self.logger.info("=" * 60)
    
    def run_subject_validation(self) -> Dict[str, Any]:
        """
        Run subject-independent validation to ensure no data leakage
        
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Running subject-independent validation")
        
        try:
            # Load data
            data = self.data_loader.load_data()
            X, y, subjects, feature_names = self.data_loader.prepare_features(data)
            
            # Setup CV
            cv_strategy = self._setup_cross_validation()
            
            # Preprocess data
            X_processed = self.preprocessor.fit_transform(X, y)
            
            # Validate no subject leakage
            validation_results = {'subject_leakage_detected': False, 'cv_folds': []}
            
            for i, (train_idx, test_idx) in enumerate(cv_strategy.split(X_processed, y, groups=subjects)):
                train_subjects = set(subjects[train_idx])
                test_subjects = set(subjects[test_idx])
                
                # Check for overlap
                overlap = train_subjects.intersection(test_subjects)
                if overlap:
                    validation_results['subject_leakage_detected'] = True
                    validation_results['leakage_fold'] = i
                    validation_results['leaking_subjects'] = list(overlap)
                    self.logger.error(f"Subject leakage detected in fold {i}: {overlap}")
                    break
                
                validation_results['cv_folds'].append({
                    'fold': i,
                    'train_subjects': len(train_subjects),
                    'test_subjects': len(test_subjects),
                    'train_samples': len(train_idx),
                    'test_samples': len(test_idx)
                })
            
            if not validation_results['subject_leakage_detected']:
                self.logger.info("✓ Subject-independent validation passed - no data leakage detected")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error in subject validation: {str(e)}")
            raise
    
    def get_feature_analysis(self) -> Dict[str, Any]:
        """
        Get detailed feature analysis
        
        Returns:
            Dictionary with feature analysis
        """
        if not self.results or 'feature_importance' not in self.results:
            return {"error": "No feature analysis available"}
        
        feature_importance = self.results['feature_importance']
        if not feature_importance:
            return {"error": "No feature importance data available"}
        
        # Convert to DataFrame if it's a dict
        if isinstance(feature_importance, dict):
            df = pd.DataFrame(feature_importance)
        else:
            df = feature_importance
        
        analysis = {
            'top_features': df.head(10).to_dict('records'),
            'feature_count': len(df),
            'importance_stats': {
                'mean': df['importance'].mean(),
                'std': df['importance'].std(),
                'min': df['importance'].min(),
                'max': df['importance'].max()
            }
        }
        
        return analysis

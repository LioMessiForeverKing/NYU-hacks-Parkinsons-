"""
Example script for custom analysis using the new architecture
Demonstrates how to use individual components and create custom workflows
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config.config_manager import ConfigManager
from src.data.data_loader import ParkinsonDataLoader
from src.preprocessing.preprocessor import ParkinsonPreprocessor
from src.models.model_factory import ModelFactory
from src.evaluation.evaluator import ParkinsonEvaluator

def custom_analysis_example():
    """Example of custom analysis using individual components"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Starting custom analysis example")
    
    try:
        # Step 1: Load configuration
        config_manager = ConfigManager("configs/random_forest_100_10cv.yaml")
        
        # Step 2: Load and explore data
        logger.info("Step 1: Loading and exploring data")
        data_loader = ParkinsonDataLoader(config_manager.get_data_config())
        data = data_loader.load_data()
        
        # Get detailed data information
        data_info = data_loader.get_data_info()
        subject_info = data_loader.get_subject_info()
        
        print("Data Information:")
        print(f"  Samples: {data_info['n_samples']}")
        print(f"  Features: {data_info['n_features']}")
        print(f"  Subjects: {data_info['n_subjects']}")
        print(f"  Target Distribution: {data_info['target_distribution']}")
        
        # Step 3: Prepare features
        logger.info("Step 2: Preparing features")
        X, y, subjects, feature_names = data_loader.prepare_features(data)
        
        # Step 4: Custom preprocessing
        logger.info("Step 3: Custom preprocessing")
        preprocessor = ParkinsonPreprocessor(config_manager.get_preprocessing_config())
        X_processed = preprocessor.fit_transform(X, y)
        
        # Get preprocessing information
        preprocessing_info = preprocessor.get_preprocessing_info()
        print(f"\nPreprocessing Information:")
        print(f"  Scaler: {preprocessing_info['scaler_type']}")
        print(f"  Features before: {preprocessing_info['n_features_before']}")
        print(f"  Features after: {preprocessing_info['n_features_after']}")
        
        # Step 5: Train multiple models
        logger.info("Step 4: Training multiple models")
        
        # Define different model configurations
        model_configs = [
            {
                'algorithm': 'RandomForest',
                'parameters': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'class_weight': 'balanced'}
            },
            {
                'algorithm': 'GradientBoosting',
                'parameters': {'n_estimators': 100, 'max_depth': 6, 'random_state': 42}
            },
            {
                'algorithm': 'SVM',
                'parameters': {'kernel': 'rbf', 'C': 1.0, 'probability': True, 'random_state': 42}
            },
            {
                'algorithm': 'LogisticRegression',
                'parameters': {'random_state': 42, 'max_iter': 1000}
            }
        ]
        
        model_results = []
        
        for i, model_config in enumerate(model_configs, 1):
            logger.info(f"Training model {i}/{len(model_configs)}: {model_config['algorithm']}")
            
            # Create and train model
            model = ModelFactory.create_model(model_config)
            model.fit(X_processed, y)
            
            # Make predictions
            y_pred = model.predict(X_processed)
            y_prob = model.predict_proba(X_processed)
            
            # Evaluate model
            evaluator = ParkinsonEvaluator(config_manager.get_evaluation_config())
            evaluation = evaluator.evaluate(y, y_pred, y_prob)
            
            # Get feature importance
            feature_importance = model.get_feature_importance(preprocessor.get_feature_names(feature_names))
            
            model_results.append({
                'algorithm': model_config['algorithm'],
                'accuracy': evaluation['accuracy'],
                'f1': evaluation['f1'],
                'roc_auc': evaluation['roc_auc'],
                'sensitivity': evaluation.get('sensitivity', 0),
                'specificity': evaluation.get('specificity', 0),
                'feature_importance': feature_importance
            })
            
            print(f"  {model_config['algorithm']}: Accuracy={evaluation['accuracy']:.3f}, F1={evaluation['f1']:.3f}")
        
        # Step 6: Compare models
        logger.info("Step 5: Comparing models")
        
        print(f"\nModel Comparison Results:")
        print(f"{'Algorithm':<20} {'Accuracy':<10} {'F1-Score':<10} {'ROC-AUC':<10} {'Sensitivity':<12} {'Specificity':<12}")
        print("-" * 80)
        
        for result in model_results:
            print(f"{result['algorithm']:<20} "
                  f"{result['accuracy']:<10.3f} "
                  f"{result['f1']:<10.3f} "
                  f"{result['roc_auc']:<10.3f} "
                  f"{result['sensitivity']:<12.3f} "
                  f"{result['specificity']:<12.3f}")
        
        # Step 7: Feature importance analysis
        logger.info("Step 6: Feature importance analysis")
        
        print(f"\nFeature Importance Analysis (Random Forest):")
        rf_result = next(r for r in model_results if r['algorithm'] == 'RandomForest')
        top_features = rf_result['feature_importance'].head(10)
        
        for _, row in top_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        # Step 8: Custom analysis - subject-level performance
        logger.info("Step 7: Subject-level performance analysis")
        
        # Group predictions by subject
        subject_performance = []
        for subject in np.unique(subjects):
            subject_mask = subjects == subject
            subject_y_true = y[subject_mask]
            subject_y_pred = y_pred[subject_mask]  # Using Random Forest predictions
            
            if len(subject_y_true) > 0:
                accuracy = np.mean(subject_y_true == subject_y_pred)
                subject_performance.append({
                    'subject': subject,
                    'samples': len(subject_y_true),
                    'accuracy': accuracy,
                    'true_label': subject_y_true[0],  # All samples from same subject have same label
                    'predicted_label': subject_y_pred[0]
                })
        
        subject_df = pd.DataFrame(subject_performance)
        print(f"\nSubject-level Performance:")
        print(f"  Average subject accuracy: {subject_df['accuracy'].mean():.3f}")
        print(f"  Subjects with perfect accuracy: {(subject_df['accuracy'] == 1.0).sum()}")
        print(f"  Subjects with mixed predictions: {(subject_df['accuracy'] < 1.0).sum()}")
        
        # Step 9: Save custom results
        logger.info("Step 8: Saving custom results")
        
        # Save model comparison
        comparison_df = pd.DataFrame(model_results)
        comparison_df.to_csv('custom_model_comparison.csv', index=False)
        
        # Save subject performance
        subject_df.to_csv('custom_subject_performance.csv', index=False)
        
        print(f"\nCustom analysis completed successfully!")
        print(f"Results saved to:")
        print(f"  - custom_model_comparison.csv")
        print(f"  - custom_subject_performance.csv")
        
    except Exception as e:
        logger.error(f"Custom analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    custom_analysis_example()

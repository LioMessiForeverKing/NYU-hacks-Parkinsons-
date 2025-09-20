"""
Main Script for Parkinson's Disease Detection
Uses the new modular architecture with configuration management
"""

import sys
import os
from pathlib import Path
import argparse
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline.parkinson_pipeline import ParkinsonPipeline
from src.config.config_manager import ConfigManager

def main():
    """Main function to run the Parkinson's detection pipeline"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Parkinson's Disease Detection Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/random_forest_100_10cv.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--validate-only", 
        action="store_true",
        help="Only run subject-independent validation"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize pipeline
        logger.info("Initializing Parkinson's Disease Detection Pipeline")
        logger.info(f"Using configuration: {args.config}")
        
        pipeline = ParkinsonPipeline(args.config)
        
        # Run subject-independent validation
        logger.info("Running subject-independent validation...")
        validation_results = pipeline.run_subject_validation()
        
        if validation_results['subject_leakage_detected']:
            logger.error("Subject leakage detected! Pipeline cannot proceed.")
            return 1
        
        logger.info("✓ Subject-independent validation passed")
        
        if args.validate_only:
            logger.info("Validation-only mode completed successfully")
            return 0
        
        # Run complete pipeline
        logger.info("Running complete pipeline...")
        results = pipeline.run()
        
        # Print summary
        print("\n" + "="*60)
        print("PARKINSON'S DISEASE DETECTION - RESULTS SUMMARY")
        print("="*60)
        
        # Data info
        data_info = results.get('data_info', {})
        print(f"Dataset: {data_info.get('n_samples', 'N/A')} samples, {data_info.get('n_features', 'N/A')} features")
        print(f"Subjects: {data_info.get('n_subjects', 'N/A')} people")
        
        # Model info
        model_info = results.get('model_info', {})
        print(f"Model: {model_info.get('model_type', 'N/A')}")
        
        # Performance metrics
        evaluation = results.get('evaluation', {})
        if evaluation:
            print("\nPerformance Metrics:")
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                if metric in evaluation:
                    print(f"  {metric.upper()}: {evaluation[metric]:.3f}")
        
        # Clinical metrics
        if 'sensitivity' in evaluation and 'specificity' in evaluation:
            print("\nClinical Metrics:")
            print(f"  Sensitivity (PD Detection): {evaluation['sensitivity']:.3f}")
            print(f"  Specificity (Healthy Detection): {evaluation['specificity']:.3f}")
        
        # Cross-validation results
        cv_results = results.get('cross_validation', {})
        if cv_results:
            print("\nCross-Validation Results:")
            for metric, stats in cv_results.items():
                print(f"  {metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        
        # Output locations
        print("\nOutput Files:")
        plots = results.get('plots', {})
        if plots:
            print("  Plots:")
            for plot_name, plot_path in plots.items():
                print(f"    {plot_name}: {plot_path}")
        
        report_path = results.get('report_path')
        if report_path:
            print(f"  Report: {report_path}")
        
        print("="*60)
        print("Pipeline completed successfully!")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\nError: {e}")
        print("Make sure the data file and configuration exist.")
        return 1
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\nError: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

"""
Example script to run all model configurations
Demonstrates the power of the new modular architecture
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.pipeline.parkinson_pipeline import ParkinsonPipeline

def run_all_configurations():
    """Run all available model configurations and compare results"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # List of all configurations
    configurations = [
        "configs/est100_d10_3cv_s2_l1_rs42.yaml",
        "configs/est100_d10_4cv_s2_l1_rs42.yaml", 
        "configs/est100_d10_10cv_s2_l1_rs42.yaml",
        "configs/est100_d10_20cv_s2_l1_rs42.yaml",
        "configs/est200_d10_3cv_s2_l1_rs42.yaml",
        "configs/est200_d10_4cv_s2_l1_rs42.yaml",
        "configs/random_forest_100_10cv.yaml"
    ]
    
    results_summary = []
    
    logger.info("Starting comprehensive model comparison")
    logger.info(f"Running {len(configurations)} different configurations")
    
    for i, config_path in enumerate(configurations, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Configuration {i}/{len(configurations)}: {config_path}")
        logger.info(f"{'='*60}")
        
        try:
            # Run pipeline
            pipeline = ParkinsonPipeline(config_path)
            results = pipeline.run()
            
            # Extract key metrics
            evaluation = results.get('evaluation', {})
            cv_results = results.get('cross_validation', {})
            
            config_name = Path(config_path).stem
            summary = {
                'configuration': config_name,
                'accuracy': evaluation.get('accuracy', 0),
                'precision': evaluation.get('precision', 0),
                'recall': evaluation.get('recall', 0),
                'f1': evaluation.get('f1', 0),
                'roc_auc': evaluation.get('roc_auc', 0),
                'sensitivity': evaluation.get('sensitivity', 0),
                'specificity': evaluation.get('specificity', 0),
                'cv_accuracy_mean': cv_results.get('accuracy', {}).get('mean', 0),
                'cv_accuracy_std': cv_results.get('accuracy', {}).get('std', 0)
            }
            
            results_summary.append(summary)
            
            # Log results
            logger.info(f"Results for {config_name}:")
            logger.info(f"  Accuracy: {summary['accuracy']:.3f}")
            logger.info(f"  F1-Score: {summary['f1']:.3f}")
            logger.info(f"  ROC-AUC: {summary['roc_auc']:.3f}")
            logger.info(f"  Sensitivity: {summary['sensitivity']:.3f}")
            logger.info(f"  Specificity: {summary['specificity']:.3f}")
            logger.info(f"  CV Accuracy: {summary['cv_accuracy_mean']:.3f} ± {summary['cv_accuracy_std']:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to run {config_path}: {str(e)}")
            results_summary.append({
                'configuration': Path(config_path).stem,
                'error': str(e)
            })
    
    # Print final comparison
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Sort by accuracy
    successful_results = [r for r in results_summary if 'error' not in r]
    successful_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"{'Configuration':<30} {'Accuracy':<10} {'F1-Score':<10} {'ROC-AUC':<10} {'Sensitivity':<12} {'Specificity':<12}")
    print("-" * 80)
    
    for result in successful_results:
        print(f"{result['configuration']:<30} "
              f"{result['accuracy']:<10.3f} "
              f"{result['f1']:<10.3f} "
              f"{result['roc_auc']:<10.3f} "
              f"{result['sensitivity']:<12.3f} "
              f"{result['specificity']:<12.3f}")
    
    # Find best configuration
    if successful_results:
        best = successful_results[0]
        print(f"\n🏆 Best Configuration: {best['configuration']}")
        print(f"   Accuracy: {best['accuracy']:.3f}")
        print(f"   F1-Score: {best['f1']:.3f}")
        print(f"   ROC-AUC: {best['roc_auc']:.3f}")
    
    # Error summary
    failed_results = [r for r in results_summary if 'error' in r]
    if failed_results:
        print(f"\n❌ Failed Configurations ({len(failed_results)}):")
        for result in failed_results:
            print(f"   {result['configuration']}: {result['error']}")
    
    print("="*80)
    logger.info("Comprehensive model comparison completed")

if __name__ == "__main__":
    run_all_configurations()

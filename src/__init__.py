"""
Parkinson's Disease Detection Package
A comprehensive machine learning pipeline for detecting Parkinson's disease using voice biomarkers
"""

from .config.config_manager import ConfigManager
from .data.data_loader import ParkinsonDataLoader
from .preprocessing.preprocessor import ParkinsonPreprocessor
from .models.model_factory import ModelFactory
from .evaluation.evaluator import ParkinsonEvaluator
from .pipeline.parkinson_pipeline import ParkinsonPipeline

__version__ = "1.0.0"
__author__ = "Parkinson's Detection Team"

__all__ = [
    'ConfigManager',
    'ParkinsonDataLoader', 
    'ParkinsonPreprocessor',
    'ModelFactory',
    'ParkinsonEvaluator',
    'ParkinsonPipeline'
]

"""
Enhanced Logging System for Parkinson's Disease Detection
Provides comprehensive logging with progress tracking and performance metrics
"""

import logging
import sys
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
from datetime import datetime
import threading
from contextlib import contextmanager

class PerformanceLogger:
    """
    Performance logging utility for tracking execution times and metrics
    """
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.lock = threading.Lock()
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation"""
        with self.lock:
            self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration"""
        with self.lock:
            if operation in self.start_times:
                duration = time.time() - self.start_times[operation]
                self.metrics[operation] = duration
                del self.start_times[operation]
                return duration
            return 0.0
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all performance metrics"""
        with self.lock:
            return self.metrics.copy()
    
    def reset(self) -> None:
        """Reset all metrics"""
        with self.lock:
            self.metrics.clear()
            self.start_times.clear()

class ProgressTracker:
    """
    Progress tracking utility for long-running operations
    """
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.last_update = 0
        self.update_interval = 0.5  # Update every 0.5 seconds
    
    def update(self, increment: int = 1, description: Optional[str] = None) -> None:
        """Update progress"""
        self.current += increment
        if description:
            self.description = description
        
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self._print_progress()
            self.last_update = current_time
    
    def _print_progress(self) -> None:
        """Print progress bar"""
        if self.total <= 0:
            return
        
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        # Calculate ETA
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --"
        
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * self.current // self.total)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        # Print progress
        print(f"\r{self.description}: |{bar}| {percentage:.1f}% ({self.current}/{self.total}) {eta_str}", end="", flush=True)
    
    def finish(self, description: Optional[str] = None) -> None:
        """Finish progress tracking"""
        self.current = self.total
        if description:
            self.description = description
        self._print_progress()
        print()  # New line after completion

class EnhancedLogger:
    """
    Enhanced logger with performance tracking and structured logging
    """
    
    def __init__(self, name: str, log_file: Optional[str] = None, level: str = "INFO"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.performance_logger = PerformanceLogger()
        self.setup_logger(log_file, level)
    
    def setup_logger(self, log_file: Optional[str], level: str) -> None:
        """Setup logger with file and console handlers"""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set level
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message"""
        self.logger.critical(message, **kwargs)
    
    def performance(self, operation: str, duration: float, **metrics) -> None:
        """Log performance metrics"""
        metrics_str = ", ".join([f"{k}={v:.3f}" for k, v in metrics.items()])
        self.info(f"Performance - {operation}: {duration:.3f}s ({metrics_str})")
    
    def start_operation(self, operation: str) -> None:
        """Start timing an operation"""
        self.performance_logger.start_timer(operation)
        self.info(f"Starting: {operation}")
    
    def end_operation(self, operation: str, **metrics) -> None:
        """End timing an operation and log performance"""
        duration = self.performance_logger.end_timer(operation)
        self.performance(operation, duration, **metrics)
    
    @contextmanager
    def operation(self, operation: str, **metrics):
        """Context manager for timing operations"""
        self.start_operation(operation)
        try:
            yield
        finally:
            self.end_operation(operation, **metrics)
    
    def log_data_info(self, data_info: Dict[str, Any]) -> None:
        """Log data information in a structured way"""
        self.info("Data Information:")
        for key, value in data_info.items():
            self.info(f"  {key}: {value}")
    
    def log_model_info(self, model_info: Dict[str, Any]) -> None:
        """Log model information in a structured way"""
        self.info("Model Information:")
        for key, value in model_info.items():
            self.info(f"  {key}: {value}")
    
    def log_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Log evaluation results in a structured way"""
        self.info("Evaluation Results:")
        
        # Basic metrics
        basic_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in basic_metrics:
            if metric in results:
                self.info(f"  {metric.upper()}: {results[metric]:.3f}")
        
        # Clinical metrics
        clinical_metrics = ['sensitivity', 'specificity', 'positive_predictive_value', 'negative_predictive_value']
        if any(metric in results for metric in clinical_metrics):
            self.info("  Clinical Metrics:")
            for metric in clinical_metrics:
                if metric in results:
                    self.info(f"    {metric.replace('_', ' ').title()}: {results[metric]:.3f}")
    
    def log_cross_validation_results(self, cv_results: Dict[str, Any]) -> None:
        """Log cross-validation results"""
        self.info("Cross-Validation Results:")
        for metric, stats in cv_results.items():
            if isinstance(stats, dict) and 'mean' in stats and 'std' in stats:
                self.info(f"  {metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    def log_feature_importance(self, feature_importance: Dict[str, float], top_n: int = 10) -> None:
        """Log feature importance results"""
        self.info(f"Top {top_n} Most Important Features:")
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(sorted_features[:top_n], 1):
            self.info(f"  {i:2d}. {feature}: {importance:.3f}")
    
    def save_metrics(self, filepath: str) -> None:
        """Save performance metrics to JSON file"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': self.performance_logger.get_metrics(),
            'logger_name': self.name
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.info(f"Performance metrics saved to: {filepath}")

def get_logger(name: str, log_file: Optional[str] = None, level: str = "INFO") -> EnhancedLogger:
    """
    Get an enhanced logger instance
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        EnhancedLogger instance
    """
    return EnhancedLogger(name, log_file, level)

def setup_root_logging(log_file: Optional[str] = None, level: str = "INFO") -> None:
    """
    Setup root logging configuration
    
    Args:
        log_file: Optional log file path
        level: Logging level
    """
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[]
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

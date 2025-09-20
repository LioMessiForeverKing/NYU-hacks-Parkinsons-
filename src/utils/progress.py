"""
Progress Tracking and Visual Feedback for Parkinson's Disease Detection
Provides progress bars and visual feedback for long training processes
"""

import time
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Union
from contextlib import contextmanager
import logging

class ProgressBar:
    """
    Enhanced progress bar with ETA, speed, and custom styling
    """
    
    def __init__(self, total: int, description: str = "Processing", 
                 width: int = 50, show_eta: bool = True, show_speed: bool = True):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.show_eta = show_eta
        self.show_speed = show_speed
        self.start_time = time.time()
        self.last_update = 0
        self.update_interval = 0.1  # Update every 100ms
        self.lock = threading.Lock()
        self.finished = False
    
    def update(self, increment: int = 1, description: Optional[str] = None) -> None:
        """Update progress"""
        with self.lock:
            if self.finished:
                return
            
            self.current += increment
            if description:
                self.description = description
            
            current_time = time.time()
            if current_time - self.last_update >= self.update_interval:
                self._print_progress()
                self.last_update = current_time
    
    def _print_progress(self) -> None:
        """Print progress bar"""
        if self.total <= 0 or self.finished:
            return
        
        # Calculate progress percentage
        percentage = min(100.0, (self.current / self.total) * 100)
        
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        
        # Calculate ETA
        eta_str = ""
        if self.show_eta and self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            if eta < 60:
                eta_str = f"ETA: {eta:.1f}s"
            elif eta < 3600:
                eta_str = f"ETA: {eta/60:.1f}m"
            else:
                eta_str = f"ETA: {eta/3600:.1f}h"
        
        # Calculate speed
        speed_str = ""
        if self.show_speed and elapsed > 0:
            speed = self.current / elapsed
            if speed < 1:
                speed_str = f"Speed: {speed:.2f}/s"
            else:
                speed_str = f"Speed: {speed:.1f}/s"
        
        # Create progress bar
        filled_length = int(self.width * self.current // self.total)
        bar = "█" * filled_length + "░" * (self.width - filled_length)
        
        # Create status line
        status_parts = [f"{self.description}: |{bar}| {percentage:.1f}% ({self.current}/{self.total})"]
        
        if eta_str:
            status_parts.append(eta_str)
        if speed_str:
            status_parts.append(speed_str)
        
        status_line = " ".join(status_parts)
        
        # Print progress (overwrite previous line)
        print(f"\r{status_line}", end="", flush=True)
    
    def finish(self, description: Optional[str] = None) -> None:
        """Finish progress tracking"""
        with self.lock:
            if self.finished:
                return
            
            self.current = self.total
            if description:
                self.description = description
            
            self._print_progress()
            self.finished = True
            
            # Print completion message
            elapsed = time.time() - self.start_time
            print(f"\n✓ {self.description} completed in {elapsed:.2f}s")
    
    def set_description(self, description: str) -> None:
        """Update description"""
        with self.lock:
            self.description = description

class MultiProgressTracker:
    """
    Track multiple progress bars simultaneously
    """
    
    def __init__(self):
        self.trackers = {}
        self.lock = threading.Lock()
    
    def create_tracker(self, name: str, total: int, description: str = "Processing") -> ProgressBar:
        """Create a new progress tracker"""
        with self.lock:
            tracker = ProgressBar(total, description)
            self.trackers[name] = tracker
            return tracker
    
    def update_tracker(self, name: str, increment: int = 1, description: Optional[str] = None) -> None:
        """Update a specific tracker"""
        with self.lock:
            if name in self.trackers:
                self.trackers[name].update(increment, description)
    
    def finish_tracker(self, name: str, description: Optional[str] = None) -> None:
        """Finish a specific tracker"""
        with self.lock:
            if name in self.trackers:
                self.trackers[name].finish(description)
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all trackers"""
        with self.lock:
            status = {}
            for name, tracker in self.trackers.items():
                status[name] = {
                    'current': tracker.current,
                    'total': tracker.total,
                    'percentage': (tracker.current / tracker.total * 100) if tracker.total > 0 else 0,
                    'description': tracker.description,
                    'finished': tracker.finished
                }
            return status

class TrainingProgressTracker:
    """
    Specialized progress tracker for machine learning training
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.multi_tracker = MultiProgressTracker()
        self.current_phase = None
        self.phase_start_time = None
    
    def start_phase(self, phase_name: str, total_steps: Optional[int] = None) -> None:
        """Start a new training phase"""
        if self.current_phase:
            self.finish_phase()
        
        self.current_phase = phase_name
        self.phase_start_time = time.time()
        
        if total_steps:
            self.multi_tracker.create_tracker(phase_name, total_steps, f"Phase: {phase_name}")
        
        self.logger.info(f"Starting phase: {phase_name}")
    
    def update_phase(self, increment: int = 1, description: Optional[str] = None) -> None:
        """Update current phase progress"""
        if self.current_phase:
            self.multi_tracker.update_tracker(self.current_phase, increment, description)
    
    def finish_phase(self, description: Optional[str] = None) -> None:
        """Finish current phase"""
        if self.current_phase:
            if description is None:
                elapsed = time.time() - self.phase_start_time if self.phase_start_time else 0
                description = f"Phase: {self.current_phase} completed in {elapsed:.2f}s"
            
            self.multi_tracker.finish_tracker(self.current_phase, description)
            self.logger.info(f"Completed phase: {self.current_phase}")
            
            self.current_phase = None
            self.phase_start_time = None
    
    def track_cross_validation(self, n_folds: int) -> ProgressBar:
        """Create a progress tracker for cross-validation"""
        return self.multi_tracker.create_tracker(
            "cross_validation", 
            n_folds, 
            "Cross-Validation"
        )
    
    def track_feature_processing(self, n_features: int) -> ProgressBar:
        """Create a progress tracker for feature processing"""
        return self.multi_tracker.create_tracker(
            "feature_processing", 
            n_features, 
            "Feature Processing"
        )
    
    def track_model_training(self, n_estimators: int) -> ProgressBar:
        """Create a progress tracker for model training"""
        return self.multi_tracker.create_tracker(
            "model_training", 
            n_estimators, 
            "Model Training"
        )

@contextmanager
def progress_context(description: str, total: int, show_eta: bool = True):
    """
    Context manager for progress tracking
    
    Args:
        description: Description of the operation
        total: Total number of steps
        show_eta: Whether to show ETA
    """
    progress = ProgressBar(total, description, show_eta=show_eta)
    try:
        yield progress
    finally:
        progress.finish()

def track_cross_validation(func: Callable, *args, **kwargs) -> Any:
    """
    Decorator to track cross-validation progress
    
    Args:
        func: Function to wrap
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
    """
    def wrapper(*args, **kwargs):
        # Extract CV parameters if available
        cv = kwargs.get('cv')
        n_splits = getattr(cv, 'n_splits', 10) if cv else 10
        
        with progress_context(f"Cross-Validation ({n_splits} folds)", n_splits) as progress:
            # Monkey patch the function to update progress
            original_func = func
            
            def tracked_func(*args, **kwargs):
                result = original_func(*args, **kwargs)
                progress.update(1)
                return result
            
            # Replace the function temporarily
            func.__code__ = tracked_func.__code__
            
            try:
                return func(*args, **kwargs)
            finally:
                # Restore original function
                func.__code__ = original_func.__code__
    
    return wrapper

def track_iterations(description: str, total: int):
    """
    Decorator to track iterations in a function
    
    Args:
        description: Description of the operation
        total: Total number of iterations
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with progress_context(description, total) as progress:
                # Create a wrapper that updates progress
                def tracked_func(*args, **kwargs):
                    result = func(*args, **kwargs)
                    progress.update(1)
                    return result
                
                return tracked_func(*args, **kwargs)
        
        return wrapper
    return decorator

class ProgressLogger:
    """
    Logger that integrates with progress tracking
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.progress_tracker = TrainingProgressTracker(logger)
    
    def log_progress(self, message: str, level: str = "INFO") -> None:
        """Log message with progress context"""
        if self.progress_tracker.current_phase:
            message = f"[{self.progress_tracker.current_phase}] {message}"
        
        getattr(self.logger, level.lower())(message)
    
    def start_operation(self, operation: str, total_steps: Optional[int] = None) -> None:
        """Start tracking an operation"""
        self.progress_tracker.start_phase(operation, total_steps)
    
    def update_operation(self, increment: int = 1, description: Optional[str] = None) -> None:
        """Update operation progress"""
        self.progress_tracker.update_phase(increment, description)
    
    def finish_operation(self, description: Optional[str] = None) -> None:
        """Finish operation tracking"""
        self.progress_tracker.finish_phase(description)

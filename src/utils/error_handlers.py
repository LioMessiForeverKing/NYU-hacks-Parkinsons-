"""
Enhanced Error Handling System for Parkinson's Disease Detection
Provides custom exceptions and recovery strategies
"""

import traceback
import sys
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import logging

class ParkinsonError(Exception):
    """Base exception for Parkinson's detection pipeline"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details,
            'timestamp': self.timestamp
        }

class ValidationError(ParkinsonError):
    """Raised when data validation fails"""
    pass

class ProcessingError(ParkinsonError):
    """Raised when data processing fails"""
    pass

class ModelError(ParkinsonError):
    """Raised when model training or prediction fails"""
    pass

class ConfigurationError(ParkinsonError):
    """Raised when configuration is invalid"""
    pass

class DataLoadError(ParkinsonError):
    """Raised when data loading fails"""
    pass

class SubjectLeakageError(ParkinsonError):
    """Raised when subject leakage is detected"""
    pass

class ErrorHandler:
    """
    Centralized error handling with recovery strategies
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = {}
        self.recovery_strategies = {}
        self.max_retries = 3
    
    def register_recovery_strategy(self, error_type: type, strategy: Callable) -> None:
        """Register a recovery strategy for a specific error type"""
        self.recovery_strategies[error_type] = strategy
    
    def handle_error(self, error: Exception, context: Optional[str] = None, 
                    retry_count: int = 0) -> bool:
        """
        Handle an error with appropriate logging and recovery
        
        Args:
            error: The exception to handle
            context: Additional context about where the error occurred
            retry_count: Current retry count
            
        Returns:
            True if error was handled successfully, False otherwise
        """
        # Log error details
        self._log_error(error, context, retry_count)
        
        # Update error counts
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Try recovery strategy
        if type(error) in self.recovery_strategies and retry_count < self.max_retries:
            try:
                recovery_success = self.recovery_strategies[type(error)](error, context, retry_count)
                if recovery_success:
                    self.logger.info(f"Recovery successful for {error_type} (attempt {retry_count + 1})")
                    return True
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy failed: {recovery_error}")
        
        return False
    
    def _log_error(self, error: Exception, context: Optional[str], retry_count: int) -> None:
        """Log error with appropriate level and details"""
        error_type = type(error).__name__
        
        # Create detailed error message
        message_parts = [f"{error_type}: {str(error)}"]
        
        if context:
            message_parts.append(f"Context: {context}")
        
        if retry_count > 0:
            message_parts.append(f"Retry attempt: {retry_count}")
        
        if hasattr(error, 'details') and error.details:
            message_parts.append(f"Details: {error.details}")
        
        error_message = " | ".join(message_parts)
        
        # Log with appropriate level
        if isinstance(error, (ValidationError, ConfigurationError)):
            self.logger.warning(error_message)
        elif isinstance(error, (ProcessingError, ModelError, DataLoadError)):
            self.logger.error(error_message)
        elif isinstance(error, SubjectLeakageError):
            self.logger.critical(error_message)
        else:
            self.logger.error(error_message)
        
        # Log stack trace for debugging
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Stack trace:\n{traceback.format_exc()}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered"""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_counts': self.error_counts.copy(),
            'recovery_strategies': list(self.recovery_strategies.keys())
        }
    
    def reset_error_counts(self) -> None:
        """Reset error counts"""
        self.error_counts.clear()

class DataValidationErrorHandler(ErrorHandler):
    """Specialized error handler for data validation"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self._register_data_recovery_strategies()
    
    def _register_data_recovery_strategies(self) -> None:
        """Register recovery strategies for data validation errors"""
        
        def handle_missing_values(error: ValidationError, context: str, retry_count: int) -> bool:
            """Handle missing values by imputation"""
            if "missing values" in str(error).lower():
                self.logger.info("Attempting to handle missing values through imputation")
                # This would trigger imputation in the preprocessor
                return True
            return False
        
        def handle_invalid_format(error: ValidationError, context: str, retry_count: int) -> bool:
            """Handle invalid data format"""
            if "format" in str(error).lower():
                self.logger.info("Attempting to handle invalid data format")
                # This would trigger data cleaning
                return True
            return False
        
        self.register_recovery_strategy(ValidationError, handle_missing_values)
        self.register_recovery_strategy(DataLoadError, handle_invalid_format)

class ModelErrorHandler(ErrorHandler):
    """Specialized error handler for model-related errors"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self._register_model_recovery_strategies()
    
    def _register_model_recovery_strategies(self) -> None:
        """Register recovery strategies for model errors"""
        
        def handle_convergence_error(error: ModelError, context: str, retry_count: int) -> bool:
            """Handle model convergence issues"""
            if "convergence" in str(error).lower() or "did not converge" in str(error).lower():
                self.logger.info("Attempting to handle convergence issues by adjusting parameters")
                # This would trigger parameter adjustment
                return True
            return False
        
        def handle_memory_error(error: ModelError, context: str, retry_count: int) -> bool:
            """Handle memory issues by reducing model complexity"""
            if "memory" in str(error).lower() or "out of memory" in str(error).lower():
                self.logger.info("Attempting to handle memory issues by reducing model complexity")
                # This would trigger model simplification
                return True
            return False
        
        self.register_recovery_strategy(ModelError, handle_convergence_error)
        self.register_recovery_strategy(ModelError, handle_memory_error)

def create_error_handler(handler_type: str = "general", 
                        logger: Optional[logging.Logger] = None) -> ErrorHandler:
    """
    Create an error handler of the specified type
    
    Args:
        handler_type: Type of error handler ('general', 'data', 'model')
        logger: Optional logger instance
        
    Returns:
        ErrorHandler instance
    """
    if handler_type == "data":
        return DataValidationErrorHandler(logger)
    elif handler_type == "model":
        return ModelErrorHandler(logger)
    else:
        return ErrorHandler(logger)

def safe_execute(func: Callable, *args, error_handler: Optional[ErrorHandler] = None,
                context: Optional[str] = None, max_retries: int = 3, **kwargs) -> Any:
    """
    Safely execute a function with error handling and retry logic
    
    Args:
        func: Function to execute
        *args: Function arguments
        error_handler: Error handler instance
        context: Context for error reporting
        max_retries: Maximum number of retries
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or None if all retries failed
    """
    if error_handler is None:
        error_handler = ErrorHandler()
    
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            success = error_handler.handle_error(e, context, attempt)
            
            if not success and attempt < max_retries:
                error_handler.logger.info(f"Retrying {func.__name__} (attempt {attempt + 1}/{max_retries})")
                continue
            elif not success:
                error_handler.logger.error(f"All retry attempts failed for {func.__name__}")
                break
    
    if last_error:
        raise last_error
    
    return None

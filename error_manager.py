"""
Unified Error Management System for ARC Project
Provides consistent error handling, logging, and recovery mechanisms
"""
from __future__ import annotations
import logging
import traceback
import functools
import time
from collections.abc import Callable
from typing import Any, Callable, Dict, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class ErrorContext:
    """Context information for errors"""
    component: str
    operation: str
    timestamp: float
    severity: ErrorSeverity
    error_type: str
    message: str
    traceback_info: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False

class ErrorManager:
    """Centralized error management system"""
    
    def __init__(self, log_level: int = logging.INFO):
        self.logger = logging.getLogger("ARC_ErrorManager")
        self.logger.setLevel(log_level)
        
        # Error statistics
        self.error_counts: Dict[str, int] = {}
        self.error_history: list[ErrorContext] = []
        self.max_history_size = 1000
        
        # Recovery strategies
        self.recovery_strategies: Dict[str, Callable] = {}
        
        # Setup default handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default error handlers"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """Register a recovery strategy for a specific error type"""
        self.recovery_strategies[error_type] = strategy
        self.logger.info(f"Registered recovery strategy for {error_type}")
    
    def handle_error(self, 
                    error: Exception, 
                    component: str, 
                    operation: str,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    attempt_recovery: bool = True) -> ErrorContext:
        """Handle an error with optional recovery"""
        
        error_type = type(error).__name__
        error_context = ErrorContext(
            component=component,
            operation=operation,
            timestamp=time.time(),
            severity=severity,
            error_type=error_type,
            message=str(error),
            traceback_info=traceback.format_exc()
        )
        
        # Update statistics
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log the error
        log_message = f"[{component}::{operation}] {error_type}: {error}"
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Attempt recovery if enabled
        if attempt_recovery and error_type in self.recovery_strategies:
            try:
                error_context.recovery_attempted = True
                recovery_result = self.recovery_strategies[error_type](error, error_context)
                error_context.recovery_successful = bool(recovery_result)
                if error_context.recovery_successful:
                    self.logger.info(f"Recovery successful for {error_type}")
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed for {error_type}: {recovery_error}")
        
        # Store in history
        self.error_history.append(error_context)
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
        
        return error_context
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts.copy(),
            'recent_errors': self.error_history[-10:] if self.error_history else [],
            'recovery_success_rate': self._calculate_recovery_rate()
        }
    
    def _calculate_recovery_rate(self) -> float:
        """Calculate recovery success rate"""
        recovery_attempts = [ctx for ctx in self.error_history if ctx.recovery_attempted]
        if not recovery_attempts:
            return 0.0
        
        successful_recoveries = [ctx for ctx in recovery_attempts if ctx.recovery_successful]
        return len(successful_recoveries) / len(recovery_attempts)

# Global error manager instance
error_manager = ErrorManager()

def safe_execute(component: str, 
                operation: str, 
                severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                default_return: Any = None,
                attempt_recovery: bool = True):
    """Decorator for safe function execution with error handling"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = error_manager.handle_error(
                    e, component, operation, severity, attempt_recovery
                )
                
                if error_context.recovery_successful:
                    # Try to execute again after recovery
                    try:
                        return func(*args, **kwargs)
                    except Exception:
                        pass
                
                return default_return
        return wrapper
    return decorator

def log_performance(component: str, operation: str):
    """Decorator to log performance metrics"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                error_manager.logger.debug(
                    f"[{component}::{operation}] Completed in {duration:.3f}s"
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                error_manager.logger.error(
                    f"[{component}::{operation}] Failed after {duration:.3f}s: {e}"
                )
                raise
        return wrapper
    return decorator

# Common recovery strategies
def import_recovery_strategy(error: Exception, context: ErrorContext) -> bool:
    """Recovery strategy for import errors"""
    if isinstance(error, ImportError):
        error_manager.logger.info(f"Attempting to use fallback for missing import")
        return True
    return False

def memory_recovery_strategy(error: Exception, context: ErrorContext) -> bool:
    """Recovery strategy for memory errors"""
    if isinstance(error, MemoryError):
        error_manager.logger.info("Attempting memory cleanup")
        import gc
        gc.collect()
        return True
    return False

# Register default recovery strategies
error_manager.register_recovery_strategy("ImportError", import_recovery_strategy)
error_manager.register_recovery_strategy("MemoryError", memory_recovery_strategy)

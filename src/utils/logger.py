"""
Advanced logging configuration for the Elite Fantasy Football Predictor
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from loguru import logger
from datetime import datetime
import json


class LoggerSetup:
    """Configure and manage application logging"""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_dir: Optional[Path] = None,
                 app_name: str = "fantasy_football"):
        """
        Initialize logger configuration
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            app_name: Application name for log files
        """
        self.log_level = log_level
        self.log_dir = log_dir or Path("logs")
        self.app_name = app_name
        
        # Remove default logger
        logger.remove()
        
        # Configure logger
        self._setup_console_logging()
        self._setup_file_logging()
        self._setup_error_logging()
        self._setup_performance_logging()
        
    def _setup_console_logging(self):
        """Configure console logging with rich formatting"""
        logger.add(
            sys.stdout,
            level=self.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
    def _setup_file_logging(self):
        """Configure file logging with rotation"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # General application log
        logger.add(
            self.log_dir / f"{self.app_name}_{{time:YYYY-MM-DD}}.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="00:00",  # Daily rotation
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
    def _setup_error_logging(self):
        """Configure error-specific logging"""
        logger.add(
            self.log_dir / f"{self.app_name}_errors_{{time:YYYY-MM-DD}}.log",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="00:00",
            retention="90 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
    def _setup_performance_logging(self):
        """Configure performance metrics logging"""
        logger.add(
            self.log_dir / f"{self.app_name}_performance_{{time:YYYY-MM-DD}}.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | PERF | {message}",
            filter=lambda record: "performance" in record["extra"],
            rotation="00:00",
            retention="7 days",
            compression="zip"
        )
        
    @staticmethod
    def log_performance(operation: str, duration: float, **kwargs):
        """
        Log performance metrics
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            **kwargs: Additional metrics
        """
        metrics = {
            "operation": operation,
            "duration_seconds": duration,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        logger.bind(performance=True).info(json.dumps(metrics))
        
    @staticmethod
    def log_model_metrics(model_name: str, metrics: dict):
        """
        Log model performance metrics
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
        """
        log_data = {
            "model": model_name,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics
        }
        logger.bind(performance=True).info(f"Model Metrics: {json.dumps(log_data)}")
        
    @staticmethod
    def log_data_processing(operation: str, records_processed: int, duration: float):
        """
        Log data processing metrics
        
        Args:
            operation: Data operation name
            records_processed: Number of records processed
            duration: Processing duration in seconds
        """
        throughput = records_processed / duration if duration > 0 else 0
        log_data = {
            "operation": operation,
            "records": records_processed,
            "duration_seconds": duration,
            "throughput_per_second": throughput,
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.bind(performance=True).info(f"Data Processing: {json.dumps(log_data)}")


# Singleton logger instance
_logger_instance: Optional[LoggerSetup] = None


def setup_logger(log_level: str = "INFO",
                 log_dir: Optional[Path] = None,
                 app_name: str = "fantasy_football") -> LoggerSetup:
    """
    Setup or get the logger singleton
    
    Args:
        log_level: Logging level
        log_dir: Directory for log files
        app_name: Application name
        
    Returns:
        LoggerSetup instance
    """
    global _logger_instance
    
    if _logger_instance is None:
        _logger_instance = LoggerSetup(log_level, log_dir, app_name)
    
    return _logger_instance


# Export logger for direct use
def get_logger():
    """Get the configured logger instance"""
    if _logger_instance is None:
        setup_logger()
    return logger


# Convenience decorators for logging
def log_execution_time(func):
    """Decorator to log function execution time"""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {str(e)}")
            raise
    
    return wrapper


def log_model_training(model_name: str):
    """Decorator to log model training"""
    def decorator(func):
        import time
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Starting training for model: {model_name}")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log training completion
                logger.success(f"Model {model_name} trained successfully in {duration:.2f}s")
                
                # If result contains metrics, log them
                if isinstance(result, dict) and 'metrics' in result:
                    LoggerSetup.log_model_metrics(model_name, result['metrics'])
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Model {model_name} training failed after {duration:.2f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test logging setup
    setup_logger("DEBUG", Path("logs"), "test")
    log = get_logger()
    
    log.debug("Debug message")
    log.info("Info message")
    log.warning("Warning message")
    log.error("Error message")
    
    # Test performance logging
    LoggerSetup.log_performance("test_operation", 1.23, records=1000)
    LoggerSetup.log_model_metrics("test_model", {"accuracy": 0.95, "mae": 2.3})
    LoggerSetup.log_data_processing("data_load", 10000, 5.67)

# src/logs/logger.py

import logging
import sys
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from src.config.config import config

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    green = "\x1b[32;20m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logger(name: str = __name__) -> logging.Logger:
    """Set up logger with both file and console handlers"""
    
    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    file_handler = RotatingFileHandler(
        filename=log_dir / f"app_{current_date}.log",
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(config.LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt='%Y-%m-%d %H:%M:%S'))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(config.LOG_LEVEL)
    console_handler.setFormatter(CustomFormatter())
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger('unify_data_engine')  # Define logger here for internal module use

def log_exception(exc: Exception, message: str = "An error occurred"):
    logger.exception(f"{message}: {str(exc)}")

class LoggerContext:
    def __init__(self, operation: str):
        self.operation = operation
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"Starting {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            duration = datetime.now() - self.start_time
            logger.info(f"Completed {self.operation} in {duration}")
        else:
            logger.error(f"Failed {self.operation}: {str(exc_val)}")
            return False
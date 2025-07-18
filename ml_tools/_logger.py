import logging
import sys


def _get_logger(name: str = "ml_tools", level: int = logging.INFO):
    """
    Initializes and returns a configured logger instance.
    
    - `logger.info()`
    - `logger.warning()`
    - `logger.error()` the program can potentially recover.
    - `logger.critical()` the program is going to crash.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevents adding handlers multiple times if the function is called again
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # Define the format string and the date format separately
        log_format = '\n🐉%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M' # Format: Year-Month-Day Hour:Minute
        
        # Pass both the format and the date format to the Formatter
        formatter = logging.Formatter(log_format, datefmt=date_format)
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.propagate = False
    
    return logger

# Create a single logger instance to be imported by other modules
_LOGGER = _get_logger()

import logging
import sys

# Step 1: Conditionally import colorlog
try:
    import colorlog # type: ignore
except ImportError:
    colorlog = None




# --- Centralized Configuration ---
LEVEL_EMOJIS = {
    logging.INFO: "✅",
    logging.WARNING: "⚠️ ",
    logging.ERROR: "🚨",
    logging.CRITICAL: "❌"
}

# Define base format strings.
BASE_INFO_FORMAT = '\n🐉 %(asctime)s [%(emoji)s %(levelname)s] - %(message)s'
BASE_WARN_FORMAT = '\n🐉 %(asctime)s [%(emoji)s %(levelname)s] [%(filename)s:%(lineno)d] - %(message)s'


# --- Unified Formatter ---
# Determine the base class and format strings based on colorlog availability
if colorlog:
    # If colorlog is available, use it as the base and use colorized formats.
    _BaseFormatter = colorlog.ColoredFormatter
    _INFO_FORMAT = BASE_INFO_FORMAT.replace('%(levelname)s', '%(log_color)s%(levelname)s%(reset)s')
    _WARN_FORMAT = BASE_WARN_FORMAT.replace('%(levelname)s', '%(log_color)s%(levelname)s%(reset)s')
else:
    # Otherwise, fall back to the standard logging.Formatter.
    _BaseFormatter = logging.Formatter
    _INFO_FORMAT = BASE_INFO_FORMAT
    _WARN_FORMAT = BASE_WARN_FORMAT


class _UnifiedFormatter(_BaseFormatter): # type: ignore
    """
    A unified log formatter that adds emojis, uses level-specific formats,
    and applies colors if colorlog is available.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the formatter, creating sub-formatters for each level."""
        # The base class __init__ is called implicitly. We prepare our custom formatters here.
        self.datefmt = kwargs.get('datefmt')
        
        # We need to pass the correct arguments to the correct formatter type
        if colorlog:
            log_colors = kwargs.get('log_colors', {})
            self.info_formatter = colorlog.ColoredFormatter(_INFO_FORMAT, datefmt=self.datefmt, log_colors=log_colors)
            self.warn_formatter = colorlog.ColoredFormatter(_WARN_FORMAT, datefmt=self.datefmt, log_colors=log_colors)
        else:
            self.info_formatter = logging.Formatter(_INFO_FORMAT, datefmt=self.datefmt)
            self.warn_formatter = logging.Formatter(_WARN_FORMAT, datefmt=self.datefmt)

    def format(self, record):
        """Adds a custom emoji attribute to the record before formatting."""
        # Add the new attribute to the record. Use .get() for a safe default.
        record.emoji = LEVEL_EMOJIS.get(record.levelno, "")
        
        # Select the appropriate formatter and let it handle the rest.
        if record.levelno >= logging.WARNING:
            return self.warn_formatter.format(record)
        else:
            return self.info_formatter.format(record)


def _get_logger(name: str = "ml_tools", level: int = logging.INFO):
    """
    Initializes and returns a configured logger instance.
    
    - `logger.info()`
    - `logger.warning()`
    - `logger.error()` the program can potentially recover.
    - `logger.exception()` inside an except block.
    - `logger.critical()` the program is going to crash.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevents adding handlers multiple times if the function is called again
    if not logger.handlers:
        # Prepare arguments for the unified formatter
        formatter_kwargs = {
            'datefmt': '%Y-%m-%d %H:%M'
        }
        
        # Use colorlog's handler if available, and add color arguments
        if colorlog:
            handler = colorlog.StreamHandler()
            formatter_kwargs["log_colors"] = { # type: ignore
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            }
        else:
            handler = logging.StreamHandler(sys.stdout)
        
        # Create and set the single, unified formatter
        formatter = _UnifiedFormatter(**formatter_kwargs)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.propagate = False
    
    return logger


# Create a single logger instance to be imported by other modules
_LOGGER = _get_logger()


def _log_and_exit(message: str, exit_code: int = 1):
    """Logs a critical message inside an exception block and terminates the program."""
    _LOGGER.exception(message)
    sys.exit(exit_code)


if __name__ == "__main__":
    _LOGGER.info("Data loading process started.")
    _LOGGER.warning("A non-critical configuration value is missing.")
    
    try:
        x = 1 / 0
    except ZeroDivisionError:
        _LOGGER.exception("Critical error during calculation.")
    
    _LOGGER.critical("Total failure.")

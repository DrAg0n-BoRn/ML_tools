import logging
import sys
from typing import Optional, Union, Any

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


class _UnifiedFormatter(logging.Formatter):
    """
    A unified log formatter that adds emojis, uses level-specific formats,
    and applies colors if colorlog is available.
    """
    def __init__(self, datefmt: Optional[str] = None, log_colors: Optional[dict[str, str]] = None):
        """Initializes the formatter, creating sub-formatters for each level."""
        # Initialize the base logging.Formatter correctly
        super().__init__(datefmt=datefmt)

        # Prepare formats based on availability of colorlog
        if colorlog and log_colors:
            # Add color codes to the base formats
            info_fmt = BASE_INFO_FORMAT.replace('%(levelname)s', '%(log_color)s%(levelname)s%(reset)s')
            warn_fmt = BASE_WARN_FORMAT.replace('%(levelname)s', '%(log_color)s%(levelname)s%(reset)s')
            
            self.info_formatter = colorlog.ColoredFormatter(info_fmt, datefmt=datefmt, log_colors=log_colors)
            self.warn_formatter = colorlog.ColoredFormatter(warn_fmt, datefmt=datefmt, log_colors=log_colors)
        else:
            # Fallback to standard logging
            self.info_formatter = logging.Formatter(BASE_INFO_FORMAT, datefmt=datefmt)
            self.warn_formatter = logging.Formatter(BASE_WARN_FORMAT, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """Adds a custom emoji attribute to the record before formatting."""
        # Add the new attribute to the record. Use .get() for a safe default.
        record.emoji = LEVEL_EMOJIS.get(record.levelno, "")

        # Select the appropriate formatter and let it handle the rest.
        if record.levelno >= logging.WARNING:
            return self.warn_formatter.format(record)
        return self.info_formatter.format(record)


class _ContextAdapter(logging.LoggerAdapter):
    """
    Wraps the logger to automatically prepend the context name to the message.
    """
    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        # Retrieve the context name from the extra dict passed during init
        context = self.extra.get('context_name', 'Unknown') # type: ignore
        return f"[{context}] {msg}", kwargs


def _setup_main_logger(name: str = "ml_tools", level: int = logging.INFO) -> logging.Logger:
    """
    Internal function to configure the singleton logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevents adding handlers multiple times if imported multiple times
    if not logger.handlers:
        formatter_kwargs: dict[str, Any] = {
            'datefmt': '%Y-%m-%d %H:%M'
        }

        # Setup Handler
        if colorlog:
            handler = colorlog.StreamHandler()
            formatter_kwargs["log_colors"] = {
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            }
        else:
            handler = logging.StreamHandler(sys.stdout)

        # Initialize the Unified Formatter with specific kwargs
        formatter = _UnifiedFormatter(**formatter_kwargs)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


# Initialize the main configured logger instance once
_ROOT_LOGGER = _setup_main_logger()


def get_logger(name: Optional[str] = None) -> Union[logging.Logger, logging.LoggerAdapter]:
    """
    Returns a logger. If a name is provided, returns an Adapter that prefixes 
    log messages with '[name]'.
    
    Usage:
        from ._logger import get_logger
        _LOGGER = get_logger("InferenceHandler")
        
        #### Output: 🐉 ... [✅ INFO] - [InferenceHandler] Message
    """
    if name:
        return _ContextAdapter(_ROOT_LOGGER, {'context_name': name})
    return _ROOT_LOGGER


if __name__ == "__main__":
    _ROOT_LOGGER.info("Data loading process started.")
    _ROOT_LOGGER.warning("A non-critical configuration value is missing.")
    
    try:
        x = 1 / 0
    except ZeroDivisionError:
        _ROOT_LOGGER.exception("Critical error during calculation.")
    
    _ROOT_LOGGER.critical("Total failure.")
    
    test_logger = get_logger("SUPER CONTEXT")
    
    test_logger.info("hello")
    test_logger.warning("world")
    test_logger.error("for coders")

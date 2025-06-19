from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
import os 

def logging_setup(log_dir: str, general_log: str, error_log: str, loggerName):
    """
    Args:
        log_dir (str): the log directory where log files will be stored. log/nameOfSubDirectory
        general_log (str): Name of the general log file, must end with '.log'.
        error_log (str): Name of the error log file, must end with '.log'.
    """
    if not '.log' in general_log:
        raise ValueError("The general_log parameter must end with '.log'")
    if not '.log' in error_log:
        raise ValueError("The error_log parameter must end with '.log'")
    LOG_DIR = log_dir
    GENERAL_LOG = os.path.join(LOG_DIR, general_log)
    ERROR_LOG = os.path.join(LOG_DIR, error_log)

    # Create log directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)

    # === Utility to clear logs 7day ===
    def clear_log_daily(log_file_path):
        """Clear the log file if it's older than one month days."""
        if os.path.exists(log_file_path):
            last_modified = datetime.fromtimestamp(os.path.getmtime(log_file_path))
            now = datetime.now()
            # Check if the log file is older than three days
            if last_modified < now - timedelta(days=30):
                with open(log_file_path, 'w'):
                    pass  # This truncates the 

    # Clear logs if they're from an older day
    clear_log_daily(ERROR_LOG)
    clear_log_daily(GENERAL_LOG)

    # === Logger Configuration ===
    logger = logging.getLogger(loggerName)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)

    # Formatter for logs
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(module)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler for INFO and above
    info_handler = RotatingFileHandler(GENERAL_LOG, maxBytes=5*1024*1024, backupCount=1)
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)

    # Handler for ERROR and above
    error_handler = RotatingFileHandler(ERROR_LOG, maxBytes=5*1024*1024, backupCount=1)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    # Attach handlers to logger
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)

    return logger
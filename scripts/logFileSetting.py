import logging
import os

def setup_logger(task_name, log_dir="results"):
    """
    Setting the loger file for the intermediate steos of the chemReservoir tool.

    Args:
        task_name (string): title of the log file
        log_dir (string): where the file is stored

    Returns:
        logging.Logger: logger file for intermediate steps.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        log_path = os.path.join(log_dir, f"{task_name}.log")
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

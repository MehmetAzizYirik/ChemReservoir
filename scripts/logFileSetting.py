# logger_check.py

import logging
import os

def setup_logger(task_name, log_dir="results"):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # Avoid adding multiple handlers
        log_path = os.path.join(log_dir, f"{task_name}.log")
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

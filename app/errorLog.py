import os
import logging
from datetime import datetime

def setup_logger():
    logger = logging.getLogger("credit_risk_app")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(base_dir, ".."))
        logs_dir = os.path.join(root_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        log_filename = f"error_log_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
        log_path = os.path.join(logs_dir, log_filename)

        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
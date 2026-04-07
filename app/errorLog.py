import os
import logging

def setup_logger():
    logger = logging.getLogger("credit_risk_app")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        log_path = os.path.join(logs_dir, "app.log")

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)

        logger.info(f"Logger started. Log file path: {log_path}")

    return logger
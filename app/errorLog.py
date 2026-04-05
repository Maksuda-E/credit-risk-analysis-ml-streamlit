import os  # import os to work with file and folder paths
import logging  # import logging to create application logs
from datetime import datetime  # import datetime to create timestamped log filenames

def setup_logger():
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")  # build the logs folder path from the app folder
    os.makedirs(log_dir, exist_ok=True)  # create the logs folder if it does not already exist

    logger = logging.getLogger("credit_risk_app")  # create or get a named logger for the app
    logger.setLevel(logging.INFO)  # set the logger level so info, warning, error, and critical messages are captured

    if logger.handlers:  # check whether handlers were already added to avoid duplicate log lines
        return logger  # return the existing logger if already configured

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")  # create a timestamp string for the log filename
    log_file = os.path.join(log_dir, f"error_log_{timestamp}.log")  # create the full log file path

    file_handler = logging.FileHandler(log_file, encoding="utf-8")  # create a file handler that writes logs to the file
    file_handler.setLevel(logging.INFO)  # allow info and above messages to be written to file

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")  # define how each log line should look
    file_handler.setFormatter(formatter)  # apply the formatter to the file handler

    logger.addHandler(file_handler)  # attach the file handler to the logger
    logger.propagate = False  # stop logs from being duplicated by parent loggers

    return logger  # return the configured logger
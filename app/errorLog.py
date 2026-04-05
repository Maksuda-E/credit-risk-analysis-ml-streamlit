# import os so the code can create folders and file paths
import os

# import logging so the application can write logs to files
import logging

# import datetime so each log file can have a timestamp in its name
from datetime import datetime


# define a function that creates and returns a configured logger
def setup_logger():
    # create the logs folder if it does not already exist
    os.makedirs("logs", exist_ok=True)

    # create a unique timestamp string for the log file name
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # build the error log file path inside the logs folder
    log_file_path = os.path.join("logs", f"error_log_{timestamp}.log")

    # create a logger object with a fixed logger name
    logger = logging.getLogger("credit_risk_app_logger")

    # set the logger level so info, warning, error, and critical logs are captured
    logger.setLevel(logging.INFO)

    # stop duplicate handlers from being added if Streamlit reruns the script
    if logger.handlers:
        return logger

    # create a file handler so logs are written to a file
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")

    # create a formatter so each log line has time, level, and message
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # attach the formatter to the file handler
    file_handler.setFormatter(formatter)

    # add the file handler to the logger
    logger.addHandler(file_handler)

    # return the ready-to-use logger
    return logger
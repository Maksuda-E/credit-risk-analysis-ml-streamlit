# import os module for working with folders and file paths
import os

# import logging module for creating application logs
import logging

# import datetime module for generating unique timestamped file names
from datetime import datetime


# define a function that creates and returns the logger
def setup_logger():
    # create or get a logger with a fixed application name
    logger = logging.getLogger("credit_risk_app")

    # set the logger level to DEBUG so all messages are captured
    logger.setLevel(logging.DEBUG)

    # stop logs from being passed to the root logger
    logger.propagate = False

    # only add handlers once so duplicate logs are not written
    if not logger.handlers:
        # get the folder where this errorLog.py file is located
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # create the logs folder directly inside the same project folder
        logs_dir = os.path.join(base_dir, "logs")

        # create the logs folder if it does not already exist
        os.makedirs(logs_dir, exist_ok=True)

        # create a unique log file name using the current date and time
        log_filename = "app.log"

        # create the full path of the log file
        log_path = os.path.join(logs_dir, log_filename)

        # define the format of each log message
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # create a handler that writes logs to the file
        file_handler = logging.FileHandler(log_path, encoding="utf-8")

        # apply the formatter to the file handler
        file_handler.setFormatter(formatter)

        # set the file handler level to DEBUG
        file_handler.setLevel(logging.DEBUG)

        # attach the file handler to the logger
        logger.addHandler(file_handler)

        # create a handler that also prints logs in the terminal
        console_handler = logging.StreamHandler()

        # apply the formatter to the console handler
        console_handler.setFormatter(formatter)

        # set the console handler level to DEBUG
        console_handler.setLevel(logging.DEBUG)

        # attach the console handler to the logger
        logger.addHandler(console_handler)

        # write one startup message so you can confirm the exact path
        logger.info(f"Logger started. Log file path: {log_path}")

    # return the configured logger
    return logger
# import os module to work with file paths and directories
import os

# import logging module to create logs for debugging and monitoring
import logging

# import datetime to generate timestamp-based log file names
from datetime import datetime


# define a function to configure and return a logger object
def setup_logger():
    # create or get a logger with a specific name for your application
    logger = logging.getLogger("credit_risk_app")

    # set the logging level to INFO so it captures info, warnings, and errors
    logger.setLevel(logging.DEBUG)

    # check if handlers already exist to avoid duplicate logs
    if not logger.handlers:

        # get the current file's directory path
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # move one level up to reach the project root directory
        root_dir = os.path.abspath(os.path.join(base_dir, ".."))

        # define the logs directory path inside the project
        logs_dir = os.path.join(root_dir, "logs")

        # create the logs directory if it does not exist
        os.makedirs(logs_dir, exist_ok=True)

        # generate a unique log file name using the current timestamp
        log_filename = f"error_log_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

        # create the full path for the log file
        log_path = os.path.join(logs_dir, log_filename)

        # define the log message format including time, level, and message
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # create a file handler to write logs into the file
        file_handler = logging.FileHandler(log_path)

        # attach the formatter to the file handler
        file_handler.setFormatter(formatter)

        # add the file handler to the logger
        logger.addHandler(file_handler)

        # create a console handler to also display logs in terminal
        console_handler = logging.StreamHandler()

        # attach the same formatter to the console handler
        console_handler.setFormatter(formatter)

        # add the console handler to the logger
        logger.addHandler(console_handler)

    # return the configured logger
    return logger
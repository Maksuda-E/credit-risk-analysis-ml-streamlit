# import logging module to record errors and system messages
import logging

# import os to create a folder for log files
import os

# import datetime to include date and time in log file names
from datetime import datetime

# define a function to configure the logger
def setup_logger():

    # create a folder named logs if it does not exist
    os.makedirs("../logs", exist_ok=True)

    # generate a log file name using current date and time
    log_filename = datetime.now().strftime("error_log_%Y_%m_%d_%H_%M_%S.log")

    # combine the folder path and file name
    log_path = os.path.join("../logs", log_filename)

    # configure logging settings such as file location and message format
    logging.basicConfig(
        filename=log_path,
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # return the configured logger
    return logging
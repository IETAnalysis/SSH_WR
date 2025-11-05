import torch

import logging


# Check if CUDA is available and set device accordingly (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import gc  # Importing gc (garbage collection) for managing memory usage on the GPU


# Function to clear GPU memory, helps prevent memory overflow during training
def clear_gpu_memory():
    """
    This function clears unused GPU memory to prevent memory overflow during model training.
    It forces garbage collection, empties the CUDA cache, and performs IPC collection.
    """
    gc.collect()  # Python's garbage collection to clear memory
    torch.cuda.empty_cache()  # Clears the cache of unused GPU memory
    torch.cuda.ipc_collect()  # Cleans up the inter-process communication on the GPU


# Function to get a logger instance for logging events
def get_logger(logger_name, log_file, level=logging.INFO):
    """
    This function sets up a logger to log messages with timestamps to a specified log file.

    Args:
        logger_name (str): The name of the logger instance.
        log_file (str): The file where logs will be saved.
        level (logging level): The log level (default is INFO).

    Returns:
        logger (logging.Logger): A logger instance configured with the given parameters.
    """
    logger = logging.getLogger(logger_name)  # Get the logger by name
    formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")  # Define log format with timestamp
    fileHandler = logging.FileHandler(log_file, mode='a')  # Handler to write log messages to file
    fileHandler.setFormatter(formatter)  # Set the formatter for the file handler

    logger.setLevel(level)  # Set the logging level (e.g., INFO, DEBUG)
    logger.addHandler(fileHandler)  # Add the file handler to the logger

    return logger  # Return the configured logger

import logging
from pathlib import Path


def setup_logger(path_save: Path):
    """
    Setup logging both to console and the file.

    :param path_save: Path to save the log file
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    handler_stream = logging.StreamHandler()
    handler_stream.setLevel(logging.INFO)
    handler_stream.setFormatter(formatter)
    logger.addHandler(handler_stream)

    if path_save.exists():
        path_save.unlink()
    handler_file = logging.FileHandler(path_save, encoding="utf-8")
    handler_file.setLevel(logging.INFO)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)



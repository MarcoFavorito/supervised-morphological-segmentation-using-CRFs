import logging
import sys
import settings

log_fileHandler = None

def setup_logger(name):
    logger = logging.getLogger(name)
    formatter = logging.Formatter(fmt=settings.log_fmt)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)

    fileHandler = log_fileHandler
    if not fileHandler == None:
        fileHandler.setFormatter(formatter)
        fileHandler.setLevel(logging.DEBUG)
        logger.addHandler(fileHandler)

    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger

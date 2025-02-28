import logging

LOG_FORMAT = "%(levelname) -10s %(asctime) " "-30s: %(message)s"
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

def get_logger(name):
    logger = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    return logger

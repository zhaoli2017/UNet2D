import logging
import sys

loggers = {}

def get_logger(name):
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        raise NotImplementedError('Please setup a logger before using it.')

def set_logger(name, level=logging.INFO, log_path=None):
    global loggers

    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        # Logging to file
        if log_path is not None:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        loggers[name] = logger

        return logger

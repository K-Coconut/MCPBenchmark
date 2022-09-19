import logging


def getLogger(name):
    logging.basicConfig(level=logging.INFO, format='%(process)d - %(asctime)s - %(message)s', filemode='w')
    logger = logging.getLogger(name)
    handler = logging.FileHandler(name, 'a')
    handler.setFormatter(fmt=logging.Formatter('%(process)d - %(asctime)s - %(message)s'))
    logger.addHandler(handler)
    return logger

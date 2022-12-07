import logging
from datetime import datetime
import os

import yaml


formatter = '%(levelname)s : %(asctime)s : %(message)s'
logging.basicConfig(format=formatter)

log_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..',
    'log'
)
os.makedirs(log_dir, exist_ok=True)
filename = os.path.join(
    log_dir,
    f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
)
file_handler = logging.FileHandler(
    filename=filename
)
file_handler.setFormatter(logging.Formatter(formatter))

LOG_NAME = 'ai_speech_caht'


class Logger:

    @staticmethod
    def d(tag: str, message: str):
        """debug log"""
        logger = logging.getLogger(LOG_NAME)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.debug('[%s] %s', tag, message)

    @staticmethod
    def i(tag: str, message: str):
        """infomation log"""
        logger = logging.getLogger(LOG_NAME)
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.info('[%s] %s', tag, message)

    @staticmethod
    def e(tag: str, message: str):
        """error log"""
        message = '\033[31m'+str(message)+'\033[0m'
        logger = logging.getLogger(LOG_NAME)
        logger.setLevel(logging.ERROR)
        logger.addHandler(file_handler)
        logger.error('[%s] %s', tag, message)

    @staticmethod
    def w(tag: str, message: str):
        """warning log"""
        message = '\033[33m'+str(message)+'\033[0m'
        logger = logging.getLogger(LOG_NAME)
        logger.setLevel(logging.WARNING)
        logger.addHandler(file_handler)
        logger.warn('[%s] %s', tag, message)

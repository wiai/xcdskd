"""
logging configuration for aloe
"""

import logging

logging.basicConfig(filename='aloe.log',
    level=logging.DEBUG,
    style='{',
    format="{filename} {asctime} [{levelname:8}] {message}",
    datefmt="%d.%m.%Y %H:%M:%S")
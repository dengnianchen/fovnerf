import sys
from logging import *
from pathlib import Path


enable_logging = False

def _log_exception(exc_type, exc_value, exc_traceback):
    if not issubclass(exc_type, KeyboardInterrupt):
        exception(exc_value, exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


def initialize(path: Path):
    global enable_logging
    basicConfig(format='%(asctime)s[%(levelname)s] %(message)s', level=INFO,
                filename=path, filemode='a' if path.exists() else 'w')
    sys.excepthook = _log_exception
    enable_logging = True


def print_and_log(msg: str):
    print(msg)
    if enable_logging:
        info(msg)

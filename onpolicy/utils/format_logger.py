import logging
from pathlib import Path
import os

cur = Path(__file__).parent.parent
if not os.path.exists(cur / "logs"):
    os.makedirs(cur / "logs")


def singleton(cls):
    _instance = {}

    def wrapper(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return wrapper


@singleton
class AppLogger:
    def __init__(self, name="app_logger", log_file="app.log"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        if self.logger.handlers:
            return

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

        file_handler = logging.FileHandler(cur / "logs" / log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

    # -------------- 安全增强版，绝对不传递 mute 给原生 logger --------------
    def info(self, msg, *args, **kwargs):
        # 自动提取 mute，不传给底层
        mute = kwargs.pop("mute", False)
        if not mute:
            self.logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        mute = kwargs.pop("mute", False)
        if not mute:
            self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        mute = kwargs.pop("mute", False)
        if not mute:
            self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        mute = kwargs.pop("mute", False)
        if not mute:
            self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        mute = kwargs.pop("mute", False)
        if not mute:
            self.logger.critical(msg, *args, **kwargs)

    # --------------------------------------------------------------------
    def get_logger(self):
        return self

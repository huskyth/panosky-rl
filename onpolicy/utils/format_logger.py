import logging
from pathlib import Path
import os
cur = Path(__file__).parent.parent
if not os.path.exists(cur / "logs"):
    os.makedirs(cur / "logs")

def singleton(cls):
    """单例装饰器"""
    _instance = {}

    def wrapper(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return wrapper


@singleton
class AppLogger:
    """
    应用单例日志类
    保证全局只有一个日志实例，多文件共享同一配置
    """

    def __init__(self, name: str = "app_logger", log_file: str = "app.log"):
        """
        初始化日志器
        :param name: 日志器名称
        :param log_file: 输出的日志文件名
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # 全局最高级别

        # 避免重复添加Handler
        if self.logger.handlers:
            return

        # 1. 格式器
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 2. 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)  # 控制台打印级别
        self.logger.addHandler(console_handler)

        # 3. 文件处理器 (写入文件)
        file_handler = logging.FileHandler(cur / "logs" / log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)  # 文件记录级别
        self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """获取日志实例"""
        return self.logger

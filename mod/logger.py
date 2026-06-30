"""
日志工具模块
提供统一的日志记录功能，将运行内容写入 log 文件
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = "MoD",
    log_dir: str = "logs",
    level: int = logging.INFO,
    format_str: str = None,
    console_output: bool = True
) -> logging.Logger:
    """
    设置并返回一个 logger 实例

    参数:
        name: logger 名称
        log_dir: 日志文件存储目录
        level: 日志级别 (logging.DEBUG/INFO/WARNING/ERROR/CRITICAL)
        format_str: 日志格式字符串
        console_output: 是否同时输出到控制台

    返回:
        logging.Logger 实例
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{name}_{timestamp}.log"

    # 创建 formatter
    formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")

    # 文件 handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台 handler (可选)
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info(f"={'='*58}")
    logger.info(f"日志初始化完成 - Log file: {log_file}")
    logger.info(f"{'='*59}")

    return logger


def get_latest_log(log_dir: str = "logs", name: str = "MoD") -> str:
    """
    获取最新的日志文件路径

    参数:
        log_dir: 日志目录
        name: logger 名称前缀

    返回:
        最新日志文件的路径
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return None

    log_files = sorted(log_path.glob(f"{name}_*.log"), key=os.path.getmtime)
    return str(log_files[-1]) if log_files else None


# 便捷函数
def log_info(message: str, name: str = "MoD"):
    """记录 INFO 级别日志"""
    logger = logging.getLogger(name)
    logger.info(message)


def log_debug(message: str, name: str = "MoD"):
    """记录 DEBUG 级别日志"""
    logger = logging.getLogger(name)
    logger.debug(message)


def log_warning(message: str, name: str = "MoD"):
    """记录 WARNING 级别日志"""
    logger = logging.getLogger(name)
    logger.warning(message)


def log_error(message: str, name: str = "MoD"):
    """记录 ERROR 级别日志"""
    logger = logging.getLogger(name)
    logger.error(message)

"""Logging configuration for the trading bot"""

import sys
from pathlib import Path
from loguru import logger

# Remove default logger
logger.remove()

# Create logs directory
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Console logging
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

# File logging - General
logger.add(
    log_dir / "trading_bot_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)

# File logging - Trades only
logger.add(
    log_dir / "trades_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="90 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
    level="INFO",
    filter=lambda record: "TRADE" in record["extra"]
)

# File logging - Errors
logger.add(
    log_dir / "errors_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="90 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR"
)

def get_logger(name: str):
    """Get a logger instance with the given name"""
    return logger.bind(name=name)

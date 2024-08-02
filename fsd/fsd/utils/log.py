import logging
import sys
from pathlib import Path
from typing import Optional, Union


def get_logger(out_path: Optional[Union[str, Path]] = None, name: Optional[str] = None) -> logging.Logger:
    """Get a logger with optional file and console handlers.

    Args:
        out_path (Optional[Union[str, Path]]): The path to the log file. If None, only console logging is used.
        name (Optional[str]): The name of the logger. Defaults to None.

    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        formatter = logging.Formatter(
            fmt="%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        if out_path is not None:
            if isinstance(out_path, str):
                out_path = Path(out_path)
            out_path.parent.mkdir(exist_ok=True, parents=True)
            fh = logging.FileHandler(out_path)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.propagate = False
        logger.addHandler(sh)
    return logger

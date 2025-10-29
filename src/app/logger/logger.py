import logging
import sys

from config.params import Params


def get_logger(name: str):
    """
    Configura e retorna um logger padrão do Python.
    """
    logger = logging.getLogger(name)

    # Evita adicionar handlers duplicados
    if not logger.handlers:
        logger.setLevel(getattr(logging, Params.LOG_LEVEL, logging.INFO))

        formatter = logging.Formatter(
            fmt=Params.LOG_FORMAT,
            datefmt=Params.LOG_DATE_FORMAT
        )

        # Handler para console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.propagate = False

    return logger


# Logger padrão para ser importado
logger = get_logger("LSTM_API_Logger")
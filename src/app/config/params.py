import os
from typing import List


class Params:
    """
    Configurações globais e hiperparâmetros do sistema.
    """
    # --- Configurações de Dados ---
    PERIODO_DADOS: str = "3y"
    INTERVALO_DADOS: str = "1d"

    # Lista de tickers a serem treinados (deve corresponder aos modelos salvos)
    TICKERS: List[str] = [
        "ITSA4.SA", "VALE3.SA", "TAEE11.SA", "PETR4.SA", "MGLU3.SA"
    ]

    # --- Configurações do Modelo ---
    LOOK_BACK: int = 60  # Parâmetro de look_back do LSTM

    # --- Configurações de Paths ---
    # Define o caminho raiz do projeto
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    PATH_DB_MERCADO: str = os.path.join(PROJECT_ROOT, "dados",
                                        "dados_mercado.db")

    # Caminho onde os modelos LSTM e artefatos (scalers, metrics) são salvos
    PATH_MODELOS_LSTM: str = os.path.join(PROJECT_ROOT, "modelos_treinados_lstm")

    # --- Configurações de Logging ---
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'
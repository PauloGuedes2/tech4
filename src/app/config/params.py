import os
from typing import List
from pathlib import Path

class Params:
    """
    Configurações globais e hiperparâmetros do sistema.
    """
    # --- Configurações de Dados ---
    PERIODO_DADOS: str = "3y"
    INTERVALO_DADOS: str = "1d"

    # Lista de tickers a serem treinados
    TICKERS: List[str] = [
        "ITSA4.SA", "VALE3.SA", "TAEE11.SA", "PETR4.SA", "MGLU3.SA"
    ]

    # --- Configurações do Modelo ---
    LOOK_BACK: int = 60

    # --- Configurações de Paths ---
    # Define a raiz do projeto de forma robusta subindo 3 níveis:
    # (src/app/config/params.py -> src/app/config -> src/app -> src)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    # Caminhos absolutos corrigidos para evitar "src/src" no Render
    # No Render, a estrutura de montagem deve alinhar-se com estes caminhos
    PATH_DB_MERCADO: str = os.path.join(BASE_DIR, "app", "dados", "dados_mercado.db")
    PATH_MODELOS_LSTM: str = os.path.join(BASE_DIR, "app", "modelos_treinados_lstm")

    # Garante a criação dos diretórios necessários ao carregar os parâmetros
    os.makedirs(os.path.dirname(PATH_DB_MERCADO), exist_ok=True)
    os.makedirs(PATH_MODELOS_LSTM, exist_ok=True)

    # --- Configurações de Logging ---
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'
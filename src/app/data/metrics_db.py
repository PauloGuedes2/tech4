import os
import sqlite3
from contextlib import contextmanager
from src.app.config.params import Params
from src.app.logger.logger import logger

class MetricsDB:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or Params.PATH_DB_MERCADO
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._criar_tabela()

    @contextmanager
    def _conexao(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _criar_tabela(self):
        with self._conexao() as conn:
            cursor = conn.cursor()
            # O 'id' como PK permite múltiplas entradas para o mesmo ticker (v1, v2, v3...)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    versao TEXT,
                    mae REAL,
                    rmse REAL,
                    mape REAL,
                    created_at TEXT
                )
            """)
            conn.commit()
        logger.info("Tabela 'metrics' preparada para histórico.")

    def salvar_metricas(self, ticker: str, versao: str, mae: float, rmse: float, mape: float):
        with self._conexao() as conn:
            # INSERT sem REPLACE para não apagar as versões anteriores
            conn.execute("""
                INSERT INTO metrics (ticker, versao, mae, rmse, mape, created_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            """, (ticker, versao, mae, rmse, mape))
            conn.commit()
        logger.info(f"Métricas no BD - {ticker} ({versao}): MAE={mae:.4f}")
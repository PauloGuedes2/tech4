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
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    ticker TEXT PRIMARY KEY,
                    mae REAL,
                    rmse REAL,
                    mape REAL,
                    created_at TEXT
                )
            """)
            conn.commit()
        logger.info("Tabela 'metrics' verificada/criada com sucesso.")

    def salvar_metricas(self, ticker: str, mae: float, rmse: float, mape: float):
        with self._conexao() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO metrics (ticker, mae, rmse, mape, created_at)
                VALUES (?, ?, ?, ?, datetime('now'))
            """, (ticker, mae, rmse, mape))
            conn.commit()

        logger.info(f"Métricas armazenadas no banco - {ticker}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")

    def carregar_metricas(self, ticker: str):
        with self._conexao() as conn:
            cursor = conn.execute("""
                SELECT mae, rmse, mape, created_at
                FROM metrics
                WHERE ticker = ?
            """, (ticker,))
            row = cursor.fetchone()

        if row:
            mae, rmse, mape, created_at = row
            return {
                "ticker": ticker,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "created_at": created_at,
            }
        return None

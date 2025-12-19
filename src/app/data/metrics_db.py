import sqlite3
import os
from src.app.config.params import Params
from src.app.logger.logger import logger

class MetricsDB:
    def __init__(self):
        self.db_path = Params.PATH_DB_MERCADO
        # Garante que a pasta existe antes de criar o BD
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._criar_tabela()

    def _criar_tabela(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Tabela atualizada com a coluna 'versao'
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metricas_lstm (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    versao TEXT,
                    mae REAL,
                    rmse REAL,
                    mape REAL,
                    data_treino TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def salvar_metricas(self, ticker, versao, mae, rmse, mape):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO metricas_lstm (ticker, versao, mae, rmse, mape)
                    VALUES (?, ?, ?, ?, ?)
                """, (ticker, versao, mae, rmse, mape))
                conn.commit()
            logger.info(f"📊 Métricas armazenadas no banco - {ticker} ({versao}): MAE={mae:.4f}")
        except Exception as e:
            logger.error(f"❌ Erro ao salvar métricas no BD: {e}")
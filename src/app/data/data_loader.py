import os
import re
import sqlite3
import requests  # Adicionado para gerenciar a sessão HTTP
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Tuple

import pandas as pd
import yfinance as yf
from fastapi import HTTPException

from src.app.config.params import Params
from src.app.logger.logger import logger

os.environ["YF_DISABLE_IMPERSONATION"] = "1"

class DataLoader:
    """
    Carrega e gerencia dados de mercado do Yahoo Finance,
    com cache em um banco de dados local.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or Params.PATH_DB_MERCADO
        # Garante que o diretório 'dados/' exista
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._criar_tabelas()

    @contextmanager
    def _conexao(self):
        """Context manager para conexões SQLite."""
        conexao = sqlite3.connect(self.db_path)
        try:
            yield conexao
        finally:
            conexao.close()

    def _criar_tabelas(self):
        """Cria a tabela `ohlcv` para armazenar os dados de mercado."""
        with self._conexao() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    ticker TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (ticker, date)
                )
            """)
            conn.commit()

    @staticmethod
    def _processar_dados_yfinance(dados_completos: pd.DataFrame, ticker: str) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """Processa o DataFrame bruto do yfinance."""
        df_ticker = pd.DataFrame({
            'Open': dados_completos['Open'][ticker],
            'High': dados_completos['High'][ticker],
            'Low': dados_completos['Low'][ticker],
            'Close': dados_completos['Close'][ticker],
            'Volume': dados_completos['Volume'][ticker]
        }).dropna()

        df_ibov = dados_completos['Close']['^BVSP'].to_frame('Close_IBOV')
        return df_ticker, df_ibov

    def baixar_dados_yf(self, ticker: str, periodo: str = None,
                        intervalo: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Baixa dados do yfinance de forma robusta, especificando datas e usando sessão HTTP.
        """
        intervalo = intervalo or Params.INTERVALO_DADOS
        periodo_config = periodo or Params.PERIODO_DADOS

        end_date = datetime.now() + timedelta(days=1)
        match = re.match(r"(\d+)(\w+)", periodo_config)
        if not match:
            raise ValueError(f"Formato de período inválido: '{periodo_config}'.")

        valor, unidade = int(match.group(1)), match.group(2).lower()
        if unidade == 'y':
            start_date = end_date - timedelta(days=valor * 365)
        elif unidade in ['mo', 'm']:
            start_date = end_date - timedelta(days=valor * 30)
        else:
            start_date = end_date - timedelta(days=valor)

        start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        logger.info(f"Baixando dados para {ticker} - De: {start_str} até {end_str}")

        try:
            # 1. Tenta baixar do yfinance PRIMEIRO
            yf.set_tz_cache_location("/tmp/yf_cache")
            logger.info(f"Tentando download atualizado para {ticker}...")
            dados_completos = yf.download(
                tickers=f"{ticker} ^BVSP",
                start=start_str,
                end=end_str,
                interval=intervalo,
                progress=False,
                auto_adjust=True,
                timeout=30,
                session=session  # Força o uso da sessão configurada
            )

            if dados_completos.empty or ticker not in dados_completos['Close']:
                raise ValueError(f"Nenhum dado retornado do yfinance para o ticker {ticker}.")

            df_ticker, df_ibov = self._processar_dados_yfinance(dados_completos, ticker)

            # 2. Se baixou, ATUALIZA o cache
            self.salvar_ohlcv(ticker, df_ticker)

            logger.info(f"Dados atualizados e salvos no cache - {ticker}: {len(df_ticker)} registros")
            return df_ticker, df_ibov

        except Exception as e:
            # 3. Se o download falhar, usa o cache como FALLBACK
            logger.warning(f"Falha ao baixar dados do yfinance para {ticker}: {e}. Tentando carregar do cache local...")

            df_cache = self.carregar_do_bd(ticker)
            if not df_cache.empty:
                logger.info(f"Dados carregados do cache (fallback) para {ticker}.")
                return df_cache, pd.DataFrame()
            else:
                logger.error(f"Erro crítico: Falha no download e cache vazio para {ticker}.")
                raise HTTPException(status_code=503, detail=f"Serviço de dados indisponível e sem cache para {ticker}.")

    def salvar_ohlcv(self, ticker: str, df: pd.DataFrame):
        """Salva dados OHLCV no banco de dados."""
        with self._conexao() as conn:
            df_para_salvar = df.reset_index()

            for _, linha in df_para_salvar.iterrows():
                valores = (
                    ticker,
                    linha["Date"].strftime("%Y-%m-%d"),
                    float(linha["Open"]),
                    float(linha["High"]),
                    float(linha["Low"]),
                    float(linha["Close"]),
                    float(linha["Volume"])
                )
                conn.execute("""
                    INSERT OR REPLACE INTO ohlcv 
                    (ticker, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, valores)
            conn.commit()
        logger.info(f"Dados salvos no BD - {ticker}: {len(df)} registros")

    def carregar_do_bd(self, ticker: str) -> pd.DataFrame:
        """Carrega dados OHLCV do banco de dados."""
        with self._conexao() as conn:
            query = "SELECT * FROM ohlcv WHERE ticker = ? ORDER BY date ASC"
            df = pd.read_sql(query, conn, params=(ticker,))

        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        return df[["Open", "High", "Low", "Close", "Volume"]]
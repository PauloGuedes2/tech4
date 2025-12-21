import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Tuple

import pandas as pd
import yfinance as yf
from fastapi import HTTPException

from src.app.config.params import Params
from src.app.logger.logger import logger

# Configurações globais para estabilidade no servidor
yf.set_tz_cache_location(None)

class DataLoader:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or Params.PATH_DB_MERCADO
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._criar_tabelas()

    @contextmanager
    def _conexao(self):
        conexao = sqlite3.connect(self.db_path)
        try:
            yield conexao
        finally:
            conexao.close()

    def _criar_tabelas(self):
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
    def _processar_dados_yfinance(dados_completos: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Suporte a MultiIndex (quando baixa múltiplos tickers)
        if isinstance(dados_completos.columns, pd.MultiIndex):
            df_ticker = pd.DataFrame({
                'Open': dados_completos['Open'][ticker],
                'High': dados_completos['High'][ticker],
                'Low': dados_completos['Low'][ticker],
                'Close': dados_completos['Close'][ticker],
                'Volume': dados_completos['Volume'][ticker]
            }).dropna()
            
            try:
                df_ibov = dados_completos['Close']['^BVSP'].to_frame('Close_IBOV')
            except:
                df_ibov = pd.DataFrame(index=df_ticker.index)
                df_ibov['Close_IBOV'] = df_ticker['Close'] # Fallback
        else:
            # Suporte a Single Index (quando baixa apenas um ticker)
            df_ticker = dados_completos[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            df_ibov = pd.DataFrame(index=df_ticker.index)
            df_ibov['Close_IBOV'] = df_ticker['Close']

        return df_ticker, df_ibov

    def baixar_dados_yf(self, ticker: str, periodo: str = None, intervalo: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
            logger.info(f"Tentando download atualizado para {ticker}...")
            
            # Deixamos o yfinance gerenciar a sessão agora que curl_cffi estará disponível
            # O yfinance usará automaticamente a melhor estratégia de conexão
            dados_completos = yf.download(
                tickers=f"{ticker} ^BVSP",
                start=start_str,
                end=end_str,
                interval=intervalo,
                progress=False,
                auto_adjust=True,
                timeout=30
            )

            if dados_completos.empty:
                # Fallback: tentar apenas o ticker principal sem o índice
                logger.warning("Download conjunto falhou. Tentando download individual...")
                dados_completos = yf.download(ticker, start=start_str, end=end_str, interval=intervalo, progress=False)

            if dados_completos.empty:
                raise ValueError(f"Nenhum dado retornado do yfinance.")

            df_ticker, df_ibov = self._processar_dados_yfinance(dados_completos, ticker)
            self.salvar_ohlcv(ticker, df_ticker)

            return df_ticker, df_ibov

        except Exception as e:
            logger.warning(f"Falha no download para {ticker}: {e}. Usando cache...")
            df_cache = self.carregar_do_bd(ticker)
            if not df_cache.empty:
                # Cria um df_ibov falso para manter a compatibilidade de retorno
                df_ibov_mock = pd.DataFrame(index=df_cache.index)
                df_ibov_mock['Close_IBOV'] = df_cache['Close']
                return df_cache, df_ibov_mock
            
            raise HTTPException(status_code=503, detail=f"Sem dados e sem cache para {ticker}")

    def salvar_ohlcv(self, ticker: str, df: pd.DataFrame):
        with self._conexao() as conn:
            df_para_salvar = df.reset_index()
            for _, linha in df_para_salvar.iterrows():
                valores = (ticker, linha["Date"].strftime("%Y-%m-%d"), float(linha["Open"]), 
                           float(linha["High"]), float(linha["Low"]), float(linha["Close"]), float(linha["Volume"]))
                conn.execute("INSERT OR REPLACE INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?)", valores)
            conn.commit()

    def carregar_do_bd(self, ticker: str) -> pd.DataFrame:
        with self._conexao() as conn:
            df = pd.read_sql("SELECT * FROM ohlcv WHERE ticker = ? ORDER BY date ASC", conn, params=(ticker,))
        if df.empty: return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
        return df[["Open", "High", "Low", "Close", "Volume"]]
import yfinance as yf
from datetime import datetime, timedelta

class YFinanceClient:
    def __init__(self):
        # O yfinance é importado na inicialização
        self.yf = yf

    def get_stock_data(self, ticker: str, date: str | None = None):
        
        # Lógica para tratar tickers brasileiros (B3)
        # O Yahoo Finance exige o sufixo '.SA' para ações brasileiras.
        adjusted_ticker = ticker
        
        # Verifica se o ticker termina com um número que indica ação brasileira (ex: 3, 4, 11)
        # e não possui o sufixo '.SA' ainda.
        if (
            adjusted_ticker.upper().endswith(('3', '4', '11')) and 
            not adjusted_ticker.upper().endswith('.SA')
        ):
            adjusted_ticker += '.SA'

        stock = self.yf.Ticker(adjusted_ticker)
        
        if date:
            # Se uma data for fornecida, busca dados para esse dia específico.
            # O parâmetro 'end' do history() é exclusivo, então buscamos até o dia seguinte.
            try:
                start_dt = datetime.strptime(date, '%Y-%m-%d').date()
                end_dt = start_dt + timedelta(days=1)
                data = stock.history(start=date, end=end_dt.strftime('%Y-%m-%d'))
            except ValueError:
                # Caso o formato da data esteja incorreto, retorna a última cotação (fallback).
                data = stock.history(period="1d")
        else:
            # Padrão: Busca o último dia de cotação.
            data = stock.history(period="1d")
            
        if data.empty:
            return None
        
        # Pega a cotação (última linha do DataFrame)
        latest_data = data.iloc[-1]
        
        # Mapeia para um dicionário que corresponde ao schema 'Stock'
        # O campo 'symbol' de retorno usa o ticker original (ex: VALE3), não o ajustado (VALE3.SA)
        return {
            "symbol": ticker,
            "name": f"{ticker} ",
            "price": float(latest_data['Close']),
            "market_cap": 0.0,
            "volume": int(latest_data['Volume']),
            "change_percent": 0.0
        }
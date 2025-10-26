from fastapi import APIRouter, HTTPException, Query
from src.app.services.yfinance_client import YFinanceClient
from src.app.schemas.stock import Stock

router = APIRouter()
yfinance_client = YFinanceClient()

@router.get("/{symbol}", response_model=Stock)
# Removido 'async' e adicionado o parâmetro de consulta 'date'
def get_stock(
    symbol: str, 
    date: str | None = Query(
        None, 
        description="Data da cotação no formato YYYY-MM-DD. Se omitida, retorna a última cotação disponível."
    )
):
    # Removido 'await' e corrigido o nome do método para 'get_stock_data'
    stock_data = yfinance_client.get_stock_data(symbol, date=date)

    if stock_data is None:
        raise HTTPException(status_code=404, detail=f"Stock data not found for {symbol} on {date if date else 'latest date'}")

    return stock_data
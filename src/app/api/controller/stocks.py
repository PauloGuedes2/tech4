from fastapi import APIRouter, HTTPException, Query, Path
from src.app.services.yfinance_client import YFinanceClient
from src.app.schemas.stock import Stock

router = APIRouter()
yfinance_client = YFinanceClient()

@router.get("/{acao}", response_model=Stock)
# Removido 'async' e adicionado o parâmetro de consulta 'date'
def get_stock(
    acao: str = Path(
        ..., 
        title="Código da ação",
        description="Código/ticker da ação.",
        example="VALE3"
    ),
    data: str | None = Query(
        None, 
        title = "Data",
        description="Data da cotação no formato YYYY-MM-DD. Se omitida, retorna a  previsão disponível."
    )
):
    # Removido 'await' e corrigido o nome do método para 'get_stock_data'
    stock_data = yfinance_client.get_stock_data(acao, date=data)

    if stock_data is None:
        raise HTTPException(status_code=404, detail=f"Stock data not found for {acao} on {data if data else 'latest date'}")

    return stock_data
from fastapi import APIRouter, Path
from typing import List 
# [NOVO] Importar Enum
from enum import Enum 

from src.app.schemas.stock import Prediction
from src.app.services.prediction_service import PredictionService

# [NOVO] 1. Definir o Enum com os tickers permitidos
class StockTickerOptions(str, Enum):
    """
    Define a lista de tickers disponíveis para seleção.
    A herança de 'str' garante que o valor passado para a rota será uma string.
    """
    vale3 = "VALE3"
    petr4 = "PETR4"
    itsa4 = "ITSA4"
    mglu3 = "MGLU3"
    taee11 = "TAEE11"


router = APIRouter(tags=["Previsões de Ações"]) 

prediction_service = PredictionService()


@router.get("/{acao}", response_model=Prediction)
# [MODIFICADO] 2. Mudar o tipo do parâmetro 'acao' para o Enum
def get_stock_prediction(
        acao: StockTickerOptions = Path(
            ...,
            title="Código da ação",
            description="Código/ticker da ação (Selecione uma opção).",
            # O campo 'example' continua útil para a sugestão
            example=StockTickerOptions.vale3.value 
        )
):
    """
    Retorna a PREVISÃO de preço de fechamento para o próximo dia útil
    para uma determinada ação, junto com as métricas de avaliação do modelo.
    """
    # 3. Usar acao.value para obter a string 'VALE3', 'PETR4', etc.
    return prediction_service.get_prediction_for_ticker(acao.value)

@router.get(
    "/historico/{acao}", 
    response_model=List[Prediction] 
)
# [MODIFICADO] 2. Mudar o tipo do parâmetro 'acao' para o Enum
def get_stock_historical_prediction(
        acao: StockTickerOptions = Path(
            ...,
            title="Código da ação",
            description="Código/ticker da ação (Selecione uma opção).",
            example=StockTickerOptions.vale3.value 
        )
):
    """
    Retorna uma sequência de PREVISÕES para os últimos 7 dias úteis, 
    rodando o modelo novamente com os dados disponíveis antes de cada dia.
    Atenção: O campo 'name' no retorno inclui o Preço Real para comparação.
    """
    # 3. Usar acao.value para obter a string
    return prediction_service.get_historical_prediction_for_ticker(acao.value, days=7)
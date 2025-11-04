from fastapi import APIRouter, Path
from typing import List 

from src.app.schemas.stock import Prediction
from src.app.services.prediction_service import PredictionService

router = APIRouter()

prediction_service = PredictionService()


@router.get("/previsao/{acao}", response_model=Prediction)
def get_stock_prediction(
        acao: str = Path(
            ...,
            title="Código da ação",
            description="Código/ticker da ação (ex: VALE3).",
            example="VALE3"
        )
):
    """
    Retorna a PREVISÃO de preço de fechamento para o próximo dia útil
    para uma determinada ação, junto com as métricas de avaliação do modelo.
    """
    return prediction_service.get_prediction_for_ticker(acao)

# [NOVO ENDPOINT]
@router.get(
    "/historico/{acao}", 
    # O endpoint retorna uma lista de previsões (os 7 dias)
    response_model=List[Prediction] 
)
def get_stock_historical_prediction(
        acao: str = Path(
            ...,
            title="Código da ação",
            description="Código/ticker da ação (ex: VALE3).",
            example="VALE3"
        )
):
    """
    Retorna o histórico das PREVISÕES dos últimos 7 dias úteis, 
    rodando o modelo novamente com os dados disponíveis antes de cada dia.
    Atenção: O campo 'name' no retorno inclui o Preço Real para comparação.
    """
    # Chama o novo método do service com days=7 fixo
    return prediction_service.get_historical_prediction_for_ticker(acao, days=7)
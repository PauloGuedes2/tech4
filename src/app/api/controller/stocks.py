from fastapi import APIRouter, Path
from typing import List 

from src.app.schemas.stock import Prediction
from src.app.services.prediction_service import PredictionService

# [CORREÇÃO]: O parâmetro 'tags' define o título da seção no Swagger UI
router = APIRouter(tags=[""]) 

prediction_service = PredictionService()


@router.get("/{acao}", response_model=Prediction)
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

@router.get(
    "/historico/{acao}", 
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
    Retorna uma sequência de PREVISÕES para os últimos 7 dias úteis, 
    rodando o modelo novamente com os dados disponíveis antes de cada dia.
    Atenção: O campo 'name' no retorno inclui o Preço Real para comparação.
    """
    return prediction_service.get_historical_prediction_for_ticker(acao, days=7)
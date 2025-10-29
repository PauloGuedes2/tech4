from fastapi import APIRouter, Path

from src.app.schemas.stock import Prediction
from src.app.services.prediction_service import PredictionService

router = APIRouter()

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

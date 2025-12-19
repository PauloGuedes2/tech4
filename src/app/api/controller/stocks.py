import os
from enum import Enum
from fastapi import APIRouter, Path, Query, BackgroundTasks
from typing import List
from src.app.schemas.stock import Prediction
from src.app.services.prediction_service import PredictionService
from src.app.train_lstm import treinar_modelos_lstm
from src.app.config.params import Params

class StockTickerOptions(str, Enum):
    vale3 = "VALE3"
    petr4 = "PETR4"
    itsa4 = "ITSA4"
    mglu3 = "MGLU3"
    taee11 = "TAEE11"

def listar_versoes():
    base = Params.PATH_MODELOS_LSTM
    if not os.path.exists(base): return ["v1"]
    v = [d for d in os.listdir(base) if d.startswith('v')]
    v.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    return v if v else ["v1"]

router = APIRouter(tags=["Stocks"])
service = PredictionService()

@router.get("/previsao/{acao}", response_model=Prediction)
def get_prediction(acao: StockTickerOptions = Path(...), versao: str = Query("v1")):
    """Atualize a página (F5) para ver novas versões detetadas: {listar_versoes()}"""
    return service.get_prediction_for_ticker(acao.value, versao=versao)

@router.get("/historico/{acao}", response_model=List[Prediction])
def get_history(acao: StockTickerOptions = Path(...), versao: str = Query("v1")):
    return service.get_historical_prediction_for_ticker(acao.value, days=7, versao=versao)

@router.post("/retreinar")
async def retrain(bt: BackgroundTasks, epochs: int = 10, batch: int = 32):
    bt.add_task(treinar_modelos_lstm, epochs=epochs, batch_size=batch)
    return {"status": "Treinamento iniciado em segundo plano"}
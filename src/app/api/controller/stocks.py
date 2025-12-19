import os
from enum import Enum
from fastapi import APIRouter, Path, Query, BackgroundTasks
from typing import List

from src.app.schemas.stock import Prediction
from src.app.services.prediction_service import PredictionService
from src.app.train_lstm import treinar_modelos_lstm
from src.app.config.params import Params

# Opções de Ações (Estático)
class StockTickerOptions(str, Enum):
    vale3 = "VALE3"
    petr4 = "PETR4"
    itsa4 = "ITSA4"
    mglu3 = "MGLU3"
    taee11 = "TAEE11"

def obter_enum_de_versoes():
    """
    Varre a pasta de modelos e cria um Enum dinâmico com as pastas v1, v2, v3... encontradas no disco.
    """
    base_path = Params.PATH_MODELOS_LSTM
    
    # Lista pastas que começam com 'v' e são diretórios
    if os.path.exists(base_path):
        pastas = [d for d in os.listdir(base_path) 
                  if os.path.isdir(os.path.join(base_path, d)) and d.startswith('v')]
    else:
        pastas = []

    # Garante que sempre exista pelo menos a v1 na lista para não dar erro na interface
    if not pastas:
        pastas = ["v1"]
    
    # Ordena as versões numericamente (v1, v2, v3, v4...) para que a ordem faça sentido
    pastas.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    
    # Cria o Enum dinâmico que gera o menu dropdown no Swagger
    return Enum("ModelVersions", {p: p for p in pastas})

router = APIRouter(tags=["Previsões e Treinamento"]) 
prediction_service = PredictionService()

@router.get("/previsao/{acao}", response_model=Prediction)
def get_stock_prediction(
    acao: StockTickerOptions = Path(...),
    # O tipo do parâmetro 'versao' é definido dinamicamente pela função acima
    versao: obter_enum_de_versoes() = Query(..., description="Selecione a versão do modelo disponível no disco")
):
    """Retorna a previsão de fechamento usando a versão selecionada no menu suspenso."""
    return prediction_service.get_prediction_for_ticker(acao.value, versao=versao.value)

@router.get("/historico/{acao}", response_model=List[Prediction])
def get_stock_historical_prediction(
    acao: StockTickerOptions = Path(...),
    versao: obter_enum_de_versoes() = Query(..., description="Selecione a versão para o histórico")
):
    """Retorna o histórico de previsões baseado na versão selecionada no menu suspenso."""
    return prediction_service.get_historical_prediction_for_ticker(acao.value, days=7, versao=versao.value)

@router.post("/retreinar")
async def trigger_training(
    background_tasks: BackgroundTasks,
    epochs: int = Query(100, ge=1, description="Número de épocas para o treinamento"),
    batch_size: int = Query(32, ge=1, description="Tamanho do lote (batch size)")
):
    """
    Inicia o treinamento dos modelos em segundo plano. 
    Uma nova pasta de versão será criada (ex: v5). 
    Após o término nos logs, RECARREGUE a página do Swagger (F5) para a nova versão aparecer na lista.
    """
    background_tasks.add_task(treinar_modelos_lstm, epochs=epochs, batch_size=batch_size)
    return {
        "status": "Treinamento iniciado",
        "config": {"epochs": epochs, "batch_size": batch_size},
        "info": "A nova versão será detectada automaticamente na lista após o término do processo e refresh (F5) da página."
    }
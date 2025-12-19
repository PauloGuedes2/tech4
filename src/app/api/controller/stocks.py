import os
from enum import Enum
from fastapi import APIRouter, Path, Query, BackgroundTasks
from typing import List

from src.app.schemas.stock import Prediction
from src.app.services.prediction_service import PredictionService
from src.app.train_lstm import treinar_modelos_lstm
from src.app.config.params import Params

# As opções de ações podem continuar como Enum pois são fixas no código
class StockTickerOptions(str, Enum):
    vale3 = "VALE3"
    petr4 = "PETR4"
    itsa4 = "ITSA4"
    mglu3 = "MGLU3"
    taee11 = "TAEE11"

# FUNÇÃO QUE LÊ AS PASTAS EM TEMPO REAL
def listar_versoes_disponiveis() -> List[str]:
    base_path = Params.PATH_MODELOS_LSTM
    if not os.path.exists(base_path):
        return ["v1"]
    
    # Lista pastas v1, v2, v3... no momento da chamada
    pastas = [d for d in os.listdir(base_path) 
              if os.path.isdir(os.path.join(base_path, d)) and d.startswith('v')]
    
    if not pastas:
        return ["v1"]
    
    # Ordena numericamente
    pastas.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    return pastas

router = APIRouter(tags=["Previsões e Treinamento"]) 
prediction_service = PredictionService()

@router.get("/previsao/{acao}", response_model=Prediction)
def get_stock_prediction(
    acao: StockTickerOptions = Path(...),
    # Mudamos de Enum para str. O Query padrão é 'v1'.
    # A descrição avisará quais versões existem no momento.
    versao: str = Query("v1", description="Digite a versão (ex: v1, v2, v3...)")
):
    """
    Retorna a previsão. 
    DICA: Se você acabou de treinar a v4, basta digitar 'v4' aqui e executar.
    Versões detectadas agora: {listar_versoes_disponiveis()}
    """
    versoes_no_disco = listar_versoes_disponiveis()
    
    if versao not in versoes_no_disco:
        raise Exception(f"Versão '{versao}' ainda não existe ou não foi concluída. Disponíveis: {versoes_no_disco}")
        
    return prediction_service.get_prediction_for_ticker(acao.value, versao=versao)

@router.get("/historico/{acao}", response_model=List[Prediction])
def get_stock_historical_prediction(
    acao: StockTickerOptions = Path(...),
    versao: str = Query("v1", description="Versão para o histórico")
):
    """Retorna o histórico baseado na versão digitada."""
    versoes_no_disco = listar_versoes_disponiveis()
    if versao not in versoes_no_disco:
        raise Exception(f"Versão '{versao}' não encontrada.")
        
    return prediction_service.get_historical_prediction_for_ticker(acao.value, days=7, versao=versao)

@router.post("/retreinar")
async def trigger_training(
    background_tasks: BackgroundTasks,
    epochs: int = Query(10, ge=1),
    batch_size: int = Query(32, ge=1)
):
    """Inicia o treino. A nova versão será aceita no GET assim que os arquivos forem salvos."""
    background_tasks.add_task(treinar_modelos_lstm, epochs=epochs, batch_size=batch_size)
    return {
        "status": "Treinamento iniciado",
        "instrucao": "Aguarde o log de finalização. Depois, basta usar a nova versão no GET (ex: v2) sem reiniciar."
    }
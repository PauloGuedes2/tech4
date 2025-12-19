import os
import json
from datetime import datetime, timedelta
from src.app.config.params import Params
from src.app.models.regression.regression_lstm import RegressaoLSTM
from src.app.data.data_loader import DataLoader

class PredictionService:
    def __init__(self):
        self.loader = DataLoader()

    def _obter_metricas(self, ticker_full: str, path_versao: str):
        metrics_path = os.path.join(path_versao, f"metrics_lstm_{ticker_full}.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                return json.load(f)
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

    def get_prediction_for_ticker(self, ticker: str, versao: str = "v1"):
        ticker_full = f"{ticker}.SA" if not ticker.endswith(".SA") else ticker
        path_versao = os.path.join(Params.PATH_MODELOS_LSTM, versao)
        
        if not os.path.exists(path_versao):
            raise Exception(f"Pasta '{versao}' não encontrada no servidor.")

        metrics = self._obter_metricas(ticker_full, path_versao)
        df_ticker, _ = self.loader.baixar_dados_yf(ticker_full, periodo=Params.PERIODO_DADOS)
        
        # Carrega os pesos de dentro da pasta v1, v2, etc
        modelo_lstm = RegressaoLSTM.carregar_artefatos(ticker_full, path_versao)
        preco_previsto = modelo_lstm.prever(df_ticker)
        
        return {
            "symbol": ticker_full,
            "name": ticker.upper(),
            "predicted_price": float(preco_previsto),
            "prediction_date": datetime.now().strftime("%Y-%m-%d"),
            "MAE": metrics.get("mae", 0.0),
            "RMSE": metrics.get("rmse", 0.0),
            "MAPE": metrics.get("mape", 0.0)
        }

    def get_historical_prediction_for_ticker(self, ticker: str, days: int, versao: str = "v1"):
        base_pred = self.get_prediction_for_ticker(ticker, versao)
        historico = []
        for i in range(days):
            data = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            # Simulação apenas para preencher a lista
            preco_simulado = base_pred["predicted_price"] * (1 - (i * 0.002))
            
            historico.append({
                **base_pred,
                "predicted_price": float(preco_simulado),
                "prediction_date": data
            })
        return historico
import json
import os
from typing import Tuple, Dict, Any

from fastapi import HTTPException
from pandas.tseries.offsets import BDay

from src.app.config.params import Params
from src.app.data.data_loader import DataLoader
from src.app.logger.logger import logger
from src.app.models.regression.regression_lstm import RegressaoLSTM
from src.app.schemas.stock import Prediction


class PredictionService:
    """
    Encapsula a lógica de negócio para carregar modelos e gerar previsões.
   """

    def __init__(self):
        self.loader = DataLoader()
        self.base_path = Params.PATH_MODELOS_LSTM
        logger.info(f"PredictionService inicializado. Modelos em: {self.base_path}")

    @staticmethod
    def _formatar_ticker(acao: str) -> str:
        """Ajusta o ticker para o formato do yfinance (.SA)."""
        acao_upper = acao.upper()
        if acao_upper.endswith(('3', '4', '11')) and not acao_upper.endswith('.SA'):
            return acao_upper + '.SA'
        return acao_upper

    def _carregar_artefatos_modelo(self, ticker_model: str) -> Tuple[RegressaoLSTM, Dict[str, Any]]:
        """
        Carrega o modelo, scaler e métricas do disco.Lança HTTPException em caso de falha. (SRP: Responsável apenas por carregar artefatos)
        """
        logger.debug(f"Carregando artefatos para {ticker_model} de {self.base_path}")

        try:
            modelo_lstm = RegressaoLSTM.carregar_artefatos(ticker=ticker_model, base_path=self.base_path)

            metrics_path = os.path.join(self.base_path, f"metrics_lstm_{ticker_model}.json")
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return modelo_lstm, metrics

        except FileNotFoundError:
            logger.error(f"Modelo não encontrado para {ticker_model}")
            raise HTTPException(
                status_code=404,
                detail=f"Modelo de previsão para '{ticker_model}' não encontrado. Verifique se o modelo foi treinado."
            )
        except Exception as e:
            logger.error(f"Erro interno ao carregar {ticker_model}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Erro interno ao carregar o modelo: {e}")

    def _gerar_previsao(self, modelo_lstm: RegressaoLSTM, ticker_model: str) -> Tuple[float, str]:
        """ Busca dados recentes e executa a previsão do modelo.(SRP: Responsável apenas por obter dados e prever)"""
        try:
            # Baixa os dados mais recentes
            df_ticker, _ = self.loader.baixar_dados_yf(ticker_model, periodo=Params.PERIODO_DADOS)

            if df_ticker.empty or len(df_ticker) < modelo_lstm.look_back:
                logger.warning(f"Dados insuficientes para prever {ticker_model}: {len(df_ticker)} registros")
                raise ValueError(f"Dados insuficientes para prever {ticker_model}")

            # Gera a previsão
            predicted_price = modelo_lstm.prever(df_ticker)

            # Calcula a data da previsão (próximo dia útil)
            ultimo_dia = df_ticker.index[-1]
            data_previsao = (ultimo_dia + BDay(1)).strftime('%Y-%m-%d')

            return predicted_price, data_previsao

        except Exception as e:
            logger.error(f"Erro durante a predição para {ticker_model}: {e} ", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Erro ao processar dados ou gerar previsão: {e}")

    def get_prediction_for_ticker(self, acao: str) -> Prediction:
        """
        Orquestra o processo completo de previsão para um ticker.
        """
        # 1. Formatar o Ticker
        ticker_model = self._formatar_ticker(acao)
        logger.info(f"Requisição recebida para: {acao.upper()} (Modelo: {ticker_model})")

        # 2. Carregar Modelo e Métricas
        modelo_lstm, metrics = self._carregar_artefatos_modelo(ticker_model)

        # 3. Gerar Previsão
        predicted_price, data_previsao = self._gerar_previsao(modelo_lstm, ticker_model)

        # 4. Formatar e retornar a resposta

        return Prediction(
            symbol=acao.upper(),  # Retorna o símbolo original
            name=f"{acao.upper()} Previsão (LSTM)",
            predicted_price=predicted_price,
            prediction_date=data_previsao,
            MAE=metrics.get('mae'),
            RMSE=metrics.get('rmse'),
            MAPE=metrics.get('mape')
        )

import json
import os
# [MODIFICADO] Adicionar 'List'
from typing import Tuple, Dict, Any, List 

from fastapi import HTTPException
from pandas.tseries.offsets import BDay
# [NOVO] Importar pandas
import pandas as pd 

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

    # [CORREÇÃO APLICADA]: Removido o @staticmethod para evitar AttributeError
    def _formatar_ticker(self, acao: str) -> str:
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
        
    # [NOVO MÉTODO PRIVADO] Para gerar as previsões históricas (comparando)
    def _gerar_previsoes_historicas(self, modelo_lstm: RegressaoLSTM, ticker_model: str, steps: int) -> List[Dict[str, Any]]:
        """
        Busca dados recentes e executa N previsões históricas, comparando com o valor real.
        steps = 7 para os últimos 7 dias úteis.
        """
        try:
            # Baixa os dados recentes.
            df_ticker, _ = self.loader.baixar_dados_yf(ticker_model, periodo=Params.PERIODO_DADOS)

            # Ajusta para ter certeza de que há dados suficientes para os últimos 'steps' dias + o look_back do modelo
            if df_ticker.empty or len(df_ticker) < modelo_lstm.look_back + steps:
                logger.warning(f"Dados insuficientes para histórico de {steps} dias em {ticker_model}")
                raise ValueError(f"Dados insuficientes para prever {steps} dias do histórico")

            results = []
            
            # Loop reverso: i=1 é a previsão para o dia mais recente, i=steps é para o dia mais antigo.
            for i in range(1, steps + 1):
                # 1. Dados para a Previsão: Pegamos todos os dados EXCETO os últimos 'i' dias.
                # Isso simula o conhecimento que o modelo teria *antes* do dia atual.
                df_historico_truncado = df_ticker.iloc[:-i] 

                # 2. Dia Real da Previsão: O dia que estamos tentando prever (fechamento real)
                dia_a_prever = df_ticker.index[-i] 
                preco_real = df_ticker['Close'].iloc[-i]

                # 3. Gera a previsão usando apenas o histórico truncado
                predicted_price = modelo_lstm.prever(df_historico_truncado) 

                # 4. Adiciona o resultado
                results.append({
                    'prediction_date': dia_a_prever.strftime('%Y-%m-%d'),
                    'predicted_price': predicted_price,
                    'actual_price': preco_real
                })

            # Retorna em ordem cronológica (do mais antigo ao mais recente)
            return results[::-1]

        except Exception as e:
            logger.error(f"Erro durante a predição histórica para {ticker_model}: {e} ", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Erro ao processar dados ou gerar previsão histórica: {e}")

    # [NOVO MÉTODO PÚBLICO] Para a rota /historico/{acao}
    def get_historical_prediction_for_ticker(self, acao: str, days: int) -> List[Prediction]:
        """
        Orquestra o processo de previsão histórica (N dias úteis) para um ticker.
        Compara a previsão (se fosse feita) com o valor real.
        """
        # 1. Formatar o Ticker
        ticker_model = self._formatar_ticker(acao)
        logger.info(f"Requisição histórica para: {acao.upper()} (Modelo: {ticker_model}), {days} dias")

        # 2. Carregar Modelo e Métricas
        modelo_lstm, metrics = self._carregar_artefatos_modelo(ticker_model)

        # 3. Gerar Previsões Históricas
        predictions_raw = self._gerar_previsoes_historicas(modelo_lstm, ticker_model, days)

        # 4. Formatar e retornar a resposta
        results: List[Prediction] = []
        
        # Obtém as métricas do modelo uma vez
        mae = metrics.get('mae')
        rmse = metrics.get('rmse')
        mape = metrics.get('mape')

        for data in predictions_raw:
            # Incluindo o preço real no campo 'name' para comparação:
            comparison_name = f"Prev. Histórica - Preço Real: R$ {data['actual_price']:.2f}"
            
            results.append(Prediction(
                symbol=acao.upper(),
                name=comparison_name,
                predicted_price=data['predicted_price'],
                prediction_date=data['prediction_date'],
                MAE=mae, 
                RMSE=rmse,
                MAPE=mape
            ))
            
        return results
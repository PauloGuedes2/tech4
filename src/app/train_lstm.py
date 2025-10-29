import os

from config.params import Params
from data.data_loader import DataLoader
from logger.logger import logger
from models.regression.regression_lstm import RegressaoLSTM


def _processar_ticker(loader: DataLoader, ticker: str, path_modelos: str):
    """
    Executa o pipeline de treinamento completo para um único ticker.
    (SRP: Isola a lógica de processamento de um ticker)
    """
    logger.info(f"\n--- Processando ticker: {ticker} ---")

    try:
        # Baixa ou carrega dados do cache
        df_ticker, _ = loader.baixar_dados_yf(ticker, periodo=Params.PERIODO_DADOS)

        if len(df_ticker) < 200:
            logger.warning(f"Dados insuficientes para {ticker} (registros: {len(df_ticker)}).Pulando.")
            return

        # Instancia o modelo - usa o LOOK_BACK dos Params
        modelo_lstm = RegressaoLSTM(look_back=Params.LOOK_BACK)

        # Treina o modelo
        modelo_lstm.treinar(df_ticker, ticker=ticker, path_modelos=path_modelos, epochs=100)

        # Salva os artefatos (modelo, scaler, métricas)
        modelo_lstm.salvar_artefatos(ticker=ticker, base_path=path_modelos)

        logger.info(f"✅ Pipeline completo para {ticker} executado com sucesso!")

    except Exception as e:
        logger.error(f"❌ Erro no pipeline para {ticker}: {e}", exc_info=True)


def treinar_modelos_lstm():
    """
    Orquestrador principal: Executa o processo de treinamento LSTM
    para todos os tickers definidos em Params.
    """
    logger.info("🤖 Iniciando processo de treinamento de modelos LSTM...")
    loader = DataLoader()

    path_modelos_lstm = Params.PATH_MODELOS_LSTM
    os.makedirs(path_modelos_lstm, exist_ok=True)
    logger.info(f"Salvando modelos em: {path_modelos_lstm}")

    for ticker in Params.TICKERS:
        _processar_ticker(loader, ticker, path_modelos_lstm)


if __name__ == "__main__":
    treinar_modelos_lstm()

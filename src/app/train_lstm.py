import os
from src.app.config.params import Params
from src.app.data.data_loader import DataLoader
from src.app.logger.logger import logger
from src.app.models.regression.regression_lstm import RegressaoLSTM

def _processar_ticker(loader: DataLoader, ticker: str, path_modelos: str, epochs: int, batch_size: int):
    """
    Executa o pipeline de treinamento completo para um único ticker com parâmetros dinâmicos.
    """
    logger.info(f"\n--- Processando ticker: {ticker} (Epochs: {epochs}, Batch: {batch_size}) ---")

    try:
        # Baixa ou carrega dados
        df_ticker, _ = loader.baixar_dados_yf(ticker, periodo=Params.PERIODO_DADOS)

        if len(df_ticker) < 200:
            logger.warning(f"Dados insuficientes para {ticker} (registros: {len(df_ticker)}). Pulando.")
            return

        # Instancia o modelo
        modelo_lstm = RegressaoLSTM(look_back=Params.LOOK_BACK)

        # Treina o modelo com parâmetros passados pelo endpoint
        # Nota: Certifique-se que o método treinar na sua classe RegressaoLSTM aceite batch_size
        modelo_lstm.treinar(
            df_ticker, 
            ticker=ticker, 
            path_modelos=path_modelos, 
            epochs=epochs, 
            batch_size=batch_size
        )

        # Salva os artefatos (modelo, scaler, métricas) na pasta da nova versão
        modelo_lstm.salvar_artefatos(ticker=ticker, base_path=path_modelos)

        logger.info(f"✅ Pipeline completo para {ticker} em {path_modelos}")

    except Exception as e:
        logger.error(f"❌ Erro no pipeline para {ticker}: {e}", exc_info=True)


def treinar_modelos_lstm(epochs: int = 100, batch_size: int = 32):
    """
    Orquestrador que localiza a próxima versão disponível (v1, v2...) 
    e executa o treinamento para todos os tickers.
    """
    logger.info("🤖 Iniciando processo de treinamento de modelos LSTM...")
    loader = DataLoader()

    base_path = Params.PATH_MODELOS_LSTM
    os.makedirs(base_path, exist_ok=True)

    # Lógica de versionamento: identifica pastas v1, v2... e define a próxima
    pastas_existentes = [d for d in os.listdir(base_path) 
                         if os.path.isdir(os.path.join(base_path, d)) and d.startswith('v')]
    
    numeros = []
    for p in pastas_existentes:
        try:
            numeros.append(int(p.replace('v', '')))
        except ValueError:
            continue
            
    proxima_v = max(numeros) + 1 if numeros else 1
    nome_versao = f"v{proxima_v}"
    
    path_nova_versao = os.path.join(base_path, nome_versao)
    os.makedirs(path_nova_versao, exist_ok=True)
    
    logger.info(f"📁 Nova versão detectada: {nome_versao}. Salvando em: {path_nova_versao}")

    for ticker in Params.TICKERS:
        _processar_ticker(loader, ticker, path_nova_versao, epochs, batch_size)
    
    return nome_versao


if __name__ == "__main__":
    # Execução padrão caso chamado via CLI
    treinar_modelos_lstm()
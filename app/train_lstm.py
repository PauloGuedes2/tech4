import os
from src.config.params import Params
from src.data.data_loader import DataLoader
from src.models.regression.regression_lstm import RegressaoLSTM


def treinar_modelos_lstm():
    """Executa o processo de treinamento LSTM completo para todos os tickers."""
    print("🤖 Iniciando processo de treinamento de modelos LSTM...")
    loader = DataLoader()

    path_modelos_lstm = os.path.join(Params.SRC_ROOT, "modelos_treinados_lstm")
    os.makedirs(path_modelos_lstm, exist_ok=True)

    for ticker in Params.TICKERS:
        print(f"\n--- Processando ticker: {ticker} ---")

        try:
            df_ticker, _ = loader.baixar_dados_yf(ticker, periodo=Params.PERIODO_DADOS)

            if len(df_ticker) < 200:  # Aumentar requisito mínimo para split robusto
                print(f"Dados insuficientes para {ticker} (registros: {len(df_ticker)}). Pulando.")
                continue

            modelo_lstm = RegressaoLSTM(look_back=60)

            # O método treinar agora gerencia o treino, callbacks e avaliação
            modelo_lstm.treinar(df_ticker, ticker=ticker, path_modelos=path_modelos_lstm, epochs=100)

            # O método salvar_artefatos agora salva modelo, scaler e métricas
            modelo_lstm.salvar_artefatos(ticker=ticker, base_path=path_modelos_lstm)

            print(f"✅ Pipeline completo para {ticker} executado com sucesso!")

        except Exception as e:
            print(f"❌ Erro no pipeline para {ticker}: {e}")


if __name__ == "__main__":
    treinar_modelos_lstm()
import json
import os

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model

from config.params import Params  # Importa Params
from logger.logger import logger


class RegressaoLSTM:
    """
    Encapsula o pipeline completo do modelo LSTM:
    1. Preparação dos dados
    2. Construção da arquitetura
    3. Treinamento
    4. Avaliação (MAE, RMSE, MAPE)
    5. Salvamento e Carregamento
    """

    def __init__(self, look_back=Params.LOOK_BACK):
        self.look_back = look_back
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        self.evaluation_metrics = None

    @staticmethod
    def _mean_absolute_percentage_error(y_true, y_pred):
        """Calcula o MAPE, tratando divisões por zero. (Movido para aqui)"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


    def _preparar_dados(self, df_precos: pd.DataFrame):
        """Prepara os dados com divisão em treino, validação e teste."""
        dataset = self.scaler.fit_transform(df_precos[['Close']].values)

        X, y = [], []

        for i in range(len(dataset) - self.look_back):
            X.append(dataset[i:(i + self.look_back), 0])
            y.append(dataset[i + self.look_back, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Divisão 70% treino, 15% validação, 15% teste
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.85)

        X_train, y_train = X[:train_size], y[:train_size]

        X_val, y_val = X[train_size:val_size], y[train_size:val_size]
        X_test, y_test = X[val_size:], y[val_size:]

        return X_train, y_train, X_val, y_val, X_test, y_test


    def construir_modelo(self, input_shape):
        """Constrói a arquitetura da rede neural LSTM.
    """
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=25))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        logger.info("Modelo LSTM construído com sucesso.")
        self.model.summary(print_fn=logger.info)


    def treinar(self, df_precos: pd.DataFrame, ticker: str,
                path_modelos: str, epochs=100, batch_size=32):
        """Orquestra o treinamento robusto com callbacks."""
        X_train, y_train, X_val, y_val, X_test, y_test = self._preparar_dados(df_precos)

        if self.model is None:
            self.construir_modelo(input_shape=(X_train.shape[1], 1))

        caminho_melhor_modelo = os.path.join(path_modelos, f"best_model_lstm_{ticker}.keras")

        # Salva apenas o melhor modelo baseado na perda de validação
        checkpoint = ModelCheckpoint(caminho_melhor_modelo, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

        logger.info(f"Iniciando treinamento para {ticker}...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint, early_stopping],

            verbose=1
        )

        logger.info(f"Carregando o melhor modelo salvo de: {caminho_melhor_modelo}")
        self.model = load_model(caminho_melhor_modelo)

        # Avalia o modelo final com o conjunto de teste
        self.avaliar(X_test, y_test)


    def avaliar(self, X_test, y_test):
        """Avalia o modelo e calcula as métricas MAE, RMSE e MAPE."""
        preds_scaled = self.model.predict(X_test)

        y_test_real = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        preds_real = self.scaler.inverse_transform(preds_scaled)

        mae = mean_absolute_error(y_test_real, preds_real)
        rmse = np.sqrt(mean_squared_error(y_test_real, preds_real))
        mape = self._mean_absolute_percentage_error(y_test_real, preds_real)

        self.evaluation_metrics = {
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        }
        logger.info("--- Métricas de Avaliação Final (dados de teste) ---")
        logger.info(f"MAE (Erro Absoluto Médio): R$ {mae:.2f}")
        logger.info(f"RMSE (Raiz do Erro Quadrático Médio): R$ {rmse:.2f}")
        logger.info(f"MAPE (Erro Percentual Absoluto Médio): {mape:.2f}%")


    def prever(self, df_precos_full: pd.DataFrame):
        """Faz a previsão para o próximo dia útil."""
        # Pega os últimos `look_back` dias da série de fechamento
        inputs = df_precos_full['Close'].values[-self.look_back:].reshape(-1, 1)
        inputs_scaled = self.scaler.transform(inputs)

        X_pred = np.array([inputs_scaled.flatten()])
        X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))

        predicted_price_scaled = self.model.predict(X_pred)
        predicted_price = self.scaler.inverse_transform(predicted_price_scaled)

        return float(predicted_price[0][0])


    def salvar_artefatos(self, ticker: str, base_path: str):
        """Salva o modelo, o scaler e as métricas.
    """
        # 1. Salva o modelo
        caminho_modelo = os.path.join(base_path, f"modelo_lstm_{ticker}.keras")
        self.model.save(caminho_modelo)

        # 2. Salva o scaler
        caminho_scaler = os.path.join(base_path, f"scaler_lstm_{ticker}.joblib")
        dump(self.scaler, caminho_scaler)

        # 3. Salva as métricas
        caminho_metricas = os.path.join(base_path, f"metrics_lstm_{ticker}.json")
        with open(caminho_metricas, 'w') as f:
            json.dump(self.evaluation_metrics, f, indent=4)

        logger.info(f"Artefatos para {ticker} salvos em: {base_path}")

    @classmethod
    def carregar_artefatos(cls, ticker: str, base_path: str):
        """Carrega artefatos e retorna uma instância pronta para prever."""
        caminho_modelo = os.path.join(base_path, f"modelo_lstm_{ticker}.keras")
        caminho_scaler = os.path.join(base_path, f"scaler_lstm_{ticker}.joblib")

        if not os.path.exists(caminho_modelo) or not os.path.exists(caminho_scaler):
            logger.error(f"Arquivos não encontrados em {base_path} para o ticker {ticker}")
            raise FileNotFoundError(f"Artefatos do modelo para {ticker} não encontrados.")

        instance = cls()
        instance.model = load_model(caminho_modelo)
        instance.scaler = load(caminho_scaler)

        logger.debug(f"Artefatos para {ticker} carregados com sucesso.")
        return instance

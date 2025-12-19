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

from src.app.config.params import Params
from src.app.data.metrics_db import MetricsDB
from src.app.logger.logger import logger

class RegressaoLSTM:
    def __init__(self, look_back=Params.LOOK_BACK):
        self.look_back = look_back
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        self.evaluation_metrics = None

    @staticmethod
    def _mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    def _preparar_dados(self, df_precos: pd.DataFrame):
        dataset = self.scaler.fit_transform(df_precos[['Close']].values)
        X, y = [], []
        for i in range(len(dataset) - self.look_back):
            X.append(dataset[i:(i + self.look_back), 0])
            y.append(dataset[i + self.look_back, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.85)
        return X[:train_size], y[:train_size], X[train_size:val_size], y[train_size:val_size], X[val_size:], y[val_size:]

    def construir_modelo(self, input_shape):
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        logger.info("Modelo LSTM construído com sucesso.")

    def treinar(self, df_precos: pd.DataFrame, ticker: str, path_modelos: str, epochs=100, batch_size=32):
        X_train, y_train, X_val, y_val, X_test, y_test = self._preparar_dados(df_precos)
        if self.model is None:
            self.construir_modelo(input_shape=(X_train.shape[1], 1))
        
        caminho_melhor = os.path.join(path_modelos, f"best_model_lstm_{ticker}.keras")
        checkpoint = ModelCheckpoint(caminho_melhor, monitor='val_loss', save_best_only=True, verbose=1)
        stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                       validation_data=(X_val, y_val), callbacks=[checkpoint, stop])
        
        self.model = load_model(caminho_melhor)
        # Correção: Passando o ticker para o método avaliar
        self.avaliar(X_test, y_test, ticker)

    def avaliar(self, X_test, y_test, ticker: str):
        preds = self.model.predict(X_test)
        y_real = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        p_real = self.scaler.inverse_transform(preds)
        
        self.evaluation_metrics = {
            "mae": float(mean_absolute_error(y_real, p_real)),
            "rmse": float(np.sqrt(mean_squared_error(y_real, p_real))),
            "mape": float(self._mean_absolute_percentage_error(y_real, p_real))
        }
        logger.info(f"--- Métricas de Avaliação Final: {ticker} ---")
        logger.info(f"MAE: R$ {self.evaluation_metrics['mae']:.2f}")
        logger.info(f"RMSE: R$ {self.evaluation_metrics['rmse']:.2f}")
        logger.info(f"MAPE: {self.evaluation_metrics['mape']:.2f}%")

    def prever(self, df_precos_full: pd.DataFrame):
        inputs = self.scaler.transform(df_precos_full['Close'].values[-self.look_back:].reshape(-1, 1))
        X_pred = np.reshape(np.array([inputs.flatten()]), (1, self.look_back, 1))
        return float(self.scaler.inverse_transform(self.model.predict(X_pred))[0][0])

    def salvar_artefatos(self, ticker: str, base_path: str):
        # 1. Salva Ficheiros no Disco
        self.model.save(os.path.join(base_path, f"modelo_lstm_{ticker}.keras"))
        dump(self.scaler, os.path.join(base_path, f"scaler_lstm_{ticker}.joblib"))
        with open(os.path.join(base_path, f"metrics_lstm_{ticker}.json"), 'w') as f:
            json.dump(self.evaluation_metrics, f, indent=4)

        # 2. Salva no Banco de Dados com Versionamento
        # Extrai o nome da versão do caminho (ex: de '.../v4' extrai 'v4')
        nome_versao = os.path.basename(os.path.normpath(base_path))
        
        db = MetricsDB()
        # Envia a versão para o método salvar_metricas do MetricsDB
        db.salvar_metricas(
            ticker=ticker, 
            versao=nome_versao,
            mae=self.evaluation_metrics.get('mae'), 
            rmse=self.evaluation_metrics.get('rmse'), 
            mape=self.evaluation_metrics.get('mape')
        )
        logger.info(f"✅ Artefatos e métricas da {nome_versao} salvos para {ticker}")

    @classmethod
    def carregar_artefatos(cls, ticker: str, base_path: str):
        instance = cls()
        instance.model = load_model(os.path.join(base_path, f"modelo_lstm_{ticker}.keras"))
        instance.scaler = load(os.path.join(base_path, f"scaler_lstm_{ticker}.joblib"))
        return instance
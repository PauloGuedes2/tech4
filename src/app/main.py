import os
import uvicorn
from fastapi import FastAPI
from logger.logger import logger

from src.app.api.controller.stocks import router as stocks_router

app = FastAPI(
    title="Previsões das Cotações",
    description="API para consultar previsões das ações. Use /cotacao/{codigo} para obter dados.",
    version="0.1.0"
)

app.include_router(stocks_router, prefix="/cotacao")

class App:
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = os.getenv("HOST", host)
        self.port = int(os.getenv("PORT", port))

    def run(self):
        logger.info(f"Servidor iniciando em {self.host}:{self.port}")
        uvicorn.run(app, host=self.host, port=self.port)

if __name__ == "__main__":
    application = App()
    application.run()
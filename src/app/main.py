from fastapi import FastAPI
from src.app.api.controller.stocks import router as stocks_router

app = FastAPI(
    title="Previsões das Cotações",
    description="API para consultar previsões das ações. Use /cotacao/{codigo} para obter dados.",
    version="0.1.0"
)

app.include_router(stocks_router, prefix="/cotacao")
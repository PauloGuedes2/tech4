from fastapi import FastAPI
from src.app.api.controller.stocks import router as stocks_router

app = FastAPI()

app.include_router(stocks_router, prefix="/api/v1/stocks")
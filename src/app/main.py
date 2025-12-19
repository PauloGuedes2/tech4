import os
import time
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response

from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

from src.app.logger.logger import logger
from src.app.api.controller.stocks import router as stocks_router

# ==========================
# Prometheus Metrics
# ==========================
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total de requisições HTTP",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Latência das requisições HTTP",
    ["endpoint"]
)

# ==========================
# FastAPI App
# ==========================
app = FastAPI(
    title="Previsões das Cotações",
    description="API para consultar previsões das ações. Use /cotacao/{codigo} para obter dados.",
    version="0.1.0"
)

# ==========================
# Middleware Prometheus
# ==========================
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint=request.url.path
    ).observe(time.time() - start_time)

    return response

# ==========================
# Endpoint /metrics
# ==========================
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# ==========================
# Routers
# ==========================
app.include_router(stocks_router, prefix="/cotacao")

# ==========================
# App Runner
# ==========================
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

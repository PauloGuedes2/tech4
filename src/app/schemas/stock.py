from pydantic import BaseModel

class Stock(BaseModel):
    symbol: str
    name: str
    price: float
    market_cap: float
    volume: int
    change_percent: float

# PARA A PREVISÃO
class Prediction(BaseModel):
    symbol: str
    name: str
    predicted_price: float
    prediction_date: str
    MAE: float 
    RMSE: float 
    MAPE: float
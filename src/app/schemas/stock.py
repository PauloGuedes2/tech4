from pydantic import BaseModel

class Stock(BaseModel):
    symbol: str
    name: str
    price: float
    market_cap: float
    volume: int
    change_percent: float
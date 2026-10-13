from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent

ct = joblib.load(BASE_DIR / "ct.joblib")
scaler = joblib.load(BASE_DIR / "scaler.joblib")
model = joblib.load(BASE_DIR / "svr_model.joblib")

known_cities = set(ct.named_transformers_["encoder"].categories_[0])
known_zips = set(ct.named_transformers_["encoder"].categories_[1])

app = FastAPI(title="House Price Prediction API")


class House(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    city: str
    statezip: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "bedrooms": 3,
                    "bathrooms": 1.5,
                    "sqft_living": 1340,
                    "sqft_lot": 7912,
                    "floors": 1.5,
                    "waterfront": 0,
                    "view": 0,
                    "condition": 3,
                    "sqft_above": 1340,
                    "sqft_basement": 0,
                    "yr_built": 1955,
                    "yr_renovated": 2005,
                    "city": "Shoreline",
                    "statezip": "WA 98133",
                }
            ]
        }
    }


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}


@app.post("/predict")
def predict(house: House):
    if house.city not in known_cities:
        raise HTTPException(status_code=422, detail=f"Unknown city: {house.city}")
    if house.statezip not in known_zips:
        raise HTTPException(status_code=422, detail=f"Unknown statezip: {house.statezip}")

    df = pd.DataFrame([house.model_dump()])[list(ct.feature_names_in_)]

    X = ct.transform(df)                    # 1) encode
    X = scaler.transform(X)                 # 2) scale
    log_price = model.predict(X)            # 3) predict (log scale)
    price = float(np.expm1(log_price)[0])   # 4) back to dollars

    return {"predicted_price_usd": round(price, 2)}
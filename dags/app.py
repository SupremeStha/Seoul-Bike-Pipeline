import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from config import (
    MODEL_PATH, PREPROCESSOR_PATH, ALL_FEATURES,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES,
)

app = FastAPI(
    title       = 'Seoul Bike Sharing Demand API',
    description = 'Predicts hourly bike rental demand based on weather and time features.',
    version     = '1.0.0',
)

try:
    model        = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except FileNotFoundError:
    model = preprocessor = None


class PredictRequest(BaseModel):
    seasons            : Literal['Spring', 'Summer', 'Autumn', 'Winter'] = Field(..., example='Summer')
    holiday            : Literal['Holiday', 'No Holiday']                = Field(..., example='No Holiday')
    hour               : int   = Field(..., ge=0,   le=23,  example=9)
    temperature_c      : float = Field(..., ge=-30, le=50,  example=15.2)
    humidity_pct       : float = Field(..., ge=0,   le=100, example=60.0)
    wind_speed_ms      : float = Field(..., ge=0,           example=2.1)
    visibility_10m     : float = Field(..., ge=0,           example=2000.0)
    dew_point_temp_c   : float = Field(...,                 example=5.0)
    solar_radiation_mj : float = Field(..., ge=0,           example=0.8)
    rainfall_mm        : float = Field(..., ge=0,           example=0.0)
    snowfall_cm        : float = Field(..., ge=0,           example=0.0)
    month              : int   = Field(..., ge=1,   le=12,  example=6)
    day_of_week        : int   = Field(..., ge=0,   le=6,   example=2,
                                       description='0 = Monday, 6 = Sunday')


class PredictResponse(BaseModel):
    predicted_bike_count : float
    model                : str


@app.get('/health')
def health():
    return {
        'status'             : 'ok',
        'model_loaded'       : model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'model'              : type(model).__name__ if model is not None else None,
    }

@app.post('/reload')
def reload_model():
    global model, preprocessor
    try:
        model        = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return {'status': 'reloaded', 'model': type(model).__name__}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f'File not found: {e}')


@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503,
            detail='Model or preprocessor not loaded. Call /reload first.')

    df = pd.DataFrame([{
        'seasons'           : request.seasons,
        'holiday'           : request.holiday,
        'hour'              : request.hour,
        'temperature_c'     : request.temperature_c,
        'humidity_pct'      : request.humidity_pct,
        'wind_speed_ms'     : request.wind_speed_ms,
        'visibility_10m'    : request.visibility_10m,
        'dew_point_temp_c'  : request.dew_point_temp_c,
        'solar_radiation_mj': request.solar_radiation_mj,
        'rainfall_mm'       : request.rainfall_mm,
        'snowfall_cm'       : request.snowfall_cm,
        'month'             : request.month,
        'day_of_week'       : request.day_of_week,
    }])

    X = preprocessor.transform(df[ALL_FEATURES])
    prediction = max(0.0, float(model.predict(X)[0]))

    return PredictResponse(
        predicted_bike_count=round(prediction, 1),
        model=type(model).__name__,
    )

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from app.services.bayesian_engine import run_bayesian_return_inference
from app.services.ml_classifier import train_and_evaluate_ml_classifier

router = APIRouter(prefix="/api/v1", tags=["analytics"])

class AnalysisRequest(BaseModel):
    ticker: str = "BTC-USD"
    lookback_periods: int = 150
    initial_capital: float = 100000.0
    run_bayesian: bool = True
    run_ml: bool = True

@router.post("/analyze")
def run_quantitative_analysis(payload: AnalysisRequest):
    try:
        # Simulate time-series price data
        np.random.seed(42)
        random_returns = np.random.normal(loc=0.0005, scale=0.02, size=payload.lookback_periods)
        price_series = payload.initial_capital * np.cumprod(1 + random_returns)
        
        df = pd.DataFrame({"price": price_series})
        df["returns"] = df["price"].pct_change().fillna(0)
        
        rolling_mean = float(df["returns"].mean())
        rolling_std = float(df["returns"].std())
        var_95 = float(np.percentile(df["returns"], 5))
        
        response_data = {
            "ticker": payload.ticker,
            "data_points": len(df),
            "metrics": {
                "mean_return": rolling_mean,
                "volatility": rolling_std,
                "var_95": var_95,
                "sharpe_ratio_approx": (rolling_mean / rolling_std) * np.sqrt(252) if rolling_std > 0 else 0.0
            }
        }
        
        # Run Bayesian MCMC Inference
        if payload.run_bayesian:
            response_data["bayesian_inference"] = run_bayesian_return_inference(df["returns"].values)
            
        # Run Machine Learning Classifier
        if payload.run_ml:
            response_data["ml_prediction_model"] = train_and_evaluate_ml_classifier(df["price"].values)
            
        response_data["status"] = "success"
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
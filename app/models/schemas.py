from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class AnalysisRequest(BaseModel):
    ticker: str = Field(default="BTC-USD", description="Financial asset ticker symbol")
    lookback_periods: int = Field(default=150, ge=10, le=1000, description="Number of historical time-series periods")
    initial_capital: float = Field(default=100000.0, gt=0, description="Initial portfolio capital base")
    run_bayesian: bool = Field(default=True, description="Flag to execute Bayesian MCMC inference")
    run_ml: bool = Field(default=True, description="Flag to execute XGBoost classification model")

class QuantitativeMetrics(BaseModel):
    mean_return: float
    volatility: float
    var_95: float
    sharpe_ratio_approx: float

class BayesianPosterior(BaseModel):
    inference_method: str
    samples_drawn: int
    tuning_steps: int
    posterior_estimates: Dict[str, Any]
    status: str

class MLModelResults(BaseModel):
    model_type: str
    training_samples: int
    test_samples: int
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    status: str

class AnalysisResponse(BaseModel):
    ticker: str
    data_points: int
    metrics: QuantitativeMetrics
    bayesian_inference: Optional[Dict] = None
    ml_prediction_model: Optional[Dict] = None
    status: str
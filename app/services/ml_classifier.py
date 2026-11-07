import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_evaluate_ml_classifier(prices: np.ndarray):
    df = pd.DataFrame({"price": prices})
    
    df["return"] = df["price"].pct_change()
    df["lag_1"] = df["return"].shift(1)
    df["lag_2"] = df["return"].shift(2)
    df["rolling_mean_5"] = df["return"].rolling(window=5).mean()
    df["rolling_std_5"] = df["return"].rolling(window=5).std()
    df["target"] = (df["return"].shift(-1) > 0).astype(int)
    
    df = df.dropna()
    
    # Fallback if lookback is small during testing
    if len(df) < 5:
        return {
            "model_type": "XGBoost Classifier",
            "training_samples": 0,
            "test_samples": 0,
            "metrics": {"accuracy": 0.5},
            "feature_importance": {},
            "status": "insufficient_data"
        }
        
    features = ["lag_1", "lag_2", "rolling_mean_5", "rolling_std_5"]
    X = df[features]
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    model = XGBClassifier(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test) if len(X_test) > 0 else model.predict(X_train)
    target_eval = y_test if len(y_test) > 0 else y_train
    acc = float(accuracy_score(target_eval, y_pred)) if len(target_eval) > 0 else 0.5
    
    importance = {feat: float(score) for feat, score in zip(features, model.feature_importances_)}
    
    return {
        "model_type": "XGBoost Classifier",
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": {"accuracy": acc},
        "feature_importance": importance,
        "status": "trained_successfully"
    }
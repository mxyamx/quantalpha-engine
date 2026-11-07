# QuantAlpha Engine

An automated quantitative research, Bayesian risk analytics, and machine learning prediction platform engineered with **FastAPI**, **PyMC**, **XGBoost**, and **Prometheus**. 

Designed for high-performance financial data ingestion, probabilistic parameter estimation, and real-time operational telemetry.

---

## 🚀 Key Features

* **RESTful Financial API (`FastAPI`):** High-performance asynchronous endpoints with automated interactive OpenAPI documentation (`/docs`).
* **Bayesian MCMC Inference (`PyMC`):** Computes posterior distributions and 95% Highest Density Intervals (HDI) for asset returns and volatility via the NUTS sampler.
* **Predictive Machine Learning Classifier (`XGBoost / scikit-learn`):** Extracts rolling technical indicators and trains gradient-boosted models to predict asset price movement direction.
* **Production Telemetry (`Prometheus`):** Tracks live request counts, endpoint latencies, and system metrics via a dedicated `/metrics` endpoint.
* **Robust Test Coverage (`Pytest`):** Automated unit and integration testing suite verifying API responses, Bayesian inference convergence, and ML classification outputs.

---

## 🛠️ Tech Stack

* **Language:** Python 3.13+
* **Framework:** FastAPI, Uvicorn, Pydantic
* **Quantitative & Statistics:** PyMC, ArviZ, Pandas, NumPy, SciPy
* **Machine Learning:** XGBoost, Scikit-learn
* **Observability:** Prometheus Client
* **Testing & Containerization:** Pytest, Docker / Podman (Containerfile)

---

## 📁 Project Architecture

```text
quantalpha-engine/
├── app/
│   ├── api/
│   │   └── analytics.py      # Core financial analytics and ML/Bayesian routers
│   ├── core/
│   │   └── telemetry.py      # Prometheus metrics definitions and middleware
│   ├── services/
│   │   ├── bayesian_engine.py # PyMC MCMC posterior sampling logic
│   │   └── ml_classifier.py  # XGBoost feature engineering and training pipeline
│   └── main.py               # FastAPI application entrypoint and middleware config
├── tests/
│   └── test_analytics.py     # Automated pytest suite
├── Containerfile             # Root container configuration
├── requirements.txt          # Python dependencies
└── README.md
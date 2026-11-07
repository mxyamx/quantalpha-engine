from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.analytics import router as analytics_router
from app.core.telemetry import metrics_endpoint, REQUEST_COUNT, REQUEST_LATENCY
import time
import os

app = FastAPI(
    title="QuantAlpha Engine | Bayesian Risk & ML Pipeline",
    description="Automated quantitative research platform for time-series extraction, Bayesian MCMC inference, and risk analytics.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to track request telemetry
@app.middleware("http")
async def add_prometheus_metrics(request: Request, call_next):
    start_time = time.time()
    path = request.url.path
    method = request.method
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_COUNT.labels(method=method, endpoint=path).inc()
    REQUEST_LATENCY.labels(endpoint=path).observe(duration)
    
    return response

# Include Routers
app.include_router(analytics_router)

# Mount Prometheus Metrics Endpoint
app.get("/metrics", tags=["telemetry"])(metrics_endpoint)

# Serve the BMO-themed Landing Page at the Root URL
@app.get("/", response_class=HTMLResponse, tags=["landing"])
def read_root():
    index_path = os.path.join(os.path.dirname(__file__), "../index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return {"status": "online", "service": "QuantAlpha Engine", "docs": "/docs"}

@app.get("/health", tags=["health"])
def health_check():
    return {"health": "healthy"}
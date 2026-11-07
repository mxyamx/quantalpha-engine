from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time

# Define Prometheus Metrics
REQUEST_COUNT = Counter(
    "quantalpha_requests_total",
    "Total number of requests received by endpoint",
    ["method", "endpoint"]
)

REQUEST_LATENCY = Histogram(
    "quantalpha_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)

MODEL_INFERENCE_LATENCY = Histogram(
    "quantalpha_inference_latency_seconds",
    "Time spent running Bayesian MCMC and ML inference",
    ["model_type"]
)

def metrics_endpoint():
    """Exposes Prometheus-formatted metrics for scraping."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
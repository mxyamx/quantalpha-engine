from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["service"] == "QuantAlpha Engine"

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"health": "healthy"}

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    # Verify Prometheus metrics format is returned
    assert "quantalpha_requests_total" in response.text


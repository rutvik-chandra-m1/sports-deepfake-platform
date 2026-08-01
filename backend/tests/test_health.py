from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check_returns_ok():
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert "app_name" in body
    assert "app_version" in body


def test_root_endpoint_returns_metadata():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

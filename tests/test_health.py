import pytest
from fastapi.testclient import TestClient

from memcraft.web import app as app_module


@pytest.mark.parametrize(
    ("reachable", "code", "status"), [(True, 200, "ok"), (False, 503, "degraded")]
)
def test_health_reports_the_database(monkeypatch, reachable, code, status):
    monkeypatch.setattr(app_module, "database_ok", lambda: reachable)
    response = TestClient(app_module.app).get("/health")
    assert response.status_code == code
    assert response.json()["status"] == status


def test_front_end_is_served():
    response = TestClient(app_module.app).get("/")
    assert response.status_code == 200
    assert "MemcraftAI" in response.text

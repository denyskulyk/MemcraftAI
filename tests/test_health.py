import pytest
from fastapi.testclient import TestClient

from memcraft.config import Settings
from memcraft.db import database_ok
from memcraft.web.app import create_app


@pytest.fixture
def app():
    return create_app(Settings(_env_file=None))


@pytest.mark.parametrize(
    ("reachable", "code", "status"), [(True, 200, "ok"), (False, 503, "degraded")]
)
def test_health_reports_the_database(app, reachable, code, status):
    app.dependency_overrides[database_ok] = lambda: reachable
    response = TestClient(app, base_url="http://localhost").get("/health")
    assert response.status_code == code
    assert response.json()["status"] == status


def test_front_end_is_served(app):
    response = TestClient(app, base_url="http://localhost").get("/")
    assert response.status_code == 200
    assert "MemcraftAI" in response.text
    assert "default-src 'self'" in response.headers["content-security-policy"]

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_health_live(migrated_db: Settings) -> None:
    # The app's startup check connects to the database, so it needs one.
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

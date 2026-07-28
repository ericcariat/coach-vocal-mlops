"""L'API doit démarrer et se documenter sans modèle entraîné."""

from __future__ import annotations

from fastapi.testclient import TestClient

from coachvocal.serving.api import api

client = TestClient(api)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "eloquence" in r.json()["wakewords"]


def test_config_expose_la_logique_live():
    r = client.get("/config", params={"wakeword": "eloquence"})
    assert r.status_code == 200
    live = r.json()["live"]
    assert live["n_consecutive"] == 3 and live["threshold"] == 0.8


def test_openapi_disponible():
    """Le Swagger est un livrable : s'il casse, la démo casse."""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert {"/predict", "/detect", "/models"} <= set(r.json()["paths"])


def test_modele_absent_renvoie_404_explicite():
    r = client.get("/models", params={"wakeword": "mot_inexistant"})
    assert r.status_code == 200
    assert r.json()["runs"] == []

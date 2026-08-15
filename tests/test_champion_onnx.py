"""Un champion peut être une tête ONNX (ADR-008) — toute la chaîne doit suivre.

Depuis la promotion de `oww_frab30np_s46`, `registry.model_path` peut renvoyer
un `model.onnx` et `load_detector` doit alors servir l'adaptateur openWakeWord
avec la même interface que le CNN (`window_probas`, `triggers_from`, `push`).
"""

from __future__ import annotations

import numpy as np
import pytest

from coachvocal import registry
from coachvocal.config import load_wakeword
from coachvocal.evaluation.oww_adapter import frontends_available
from coachvocal.inference.detector import load_detector

WAKEWORD = "eloquence"


def test_model_path_resout_keras_ou_onnx():
    # v17_stack (CNN) → model.keras ; le champion oww → model.onnx
    assert registry.model_path(WAKEWORD, "v17_stack").name == "model.keras"
    p = registry.model_path(WAKEWORD, "oww_frab30np_s46")
    assert p.name == "model.onnx" and p.exists()


@pytest.mark.skipif(not frontends_available(),
                    reason="front-ends openWakeWord absents (data/external/oww_models)")
def test_load_detector_champion_onnx():
    word = load_wakeword(WAKEWORD)
    det = load_detector(registry.model_path(WAKEWORD, "oww_frab30np_s46"), word, 0.8)
    # Interface commune : le banc, le live et l'API n'appellent que ça.
    audio = np.zeros(word.sample_rate * 3, np.float32)
    probas, peaks, _starts = det.window_probas(audio)
    assert len(probas) > 0 and float(np.max(probas)) < 0.8   # silence → pas de mot
    assert det.triggers_from(probas, peaks, 0.8) == []
    assert det.run_offline(audio, 0.8) == []                  # contrat de l'API
    assert det.hop > 0                                        # requis par push()/Demo


@pytest.mark.skipif(not frontends_available(),
                    reason="front-ends openWakeWord absents (data/external/oww_models)")
def test_api_sert_le_champion_onnx():
    # /predict et /detect doivent servir le champion quel que soit son format —
    # c'est ce chemin qui avait cassé à la promotion (run_offline manquant).
    import io

    import soundfile as sf
    from fastapi.testclient import TestClient

    from coachvocal.serving.api import api

    client = TestClient(api)
    # 1 s seulement : plus court que la fenêtre d'une tête (~2 s) — c'est le
    # cas qui plantait /predict (argmax sur zéro fenêtre) avant fit_to_window.
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, np.float32), 16000, format="WAV")
    r = client.post("/predict", files={"file": ("t.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 200 and r.json()["detected"] is False
    buf.seek(0)
    r = client.post("/detect", files={"file": ("t.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 200 and r.json()["triggers"] == []

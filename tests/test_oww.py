"""Adaptateur openWakeWord : cohérence hors-ligne / temps réel.

Le chemin temps réel (`push`) a produit une proba 0 permanente parce que son
tampon donnait 194 trames mel là où la tête en exige 196 — ce test rejoue le
scénario. Sauté si les modèles ONNX ne sont pas présents (dépôt allégé)."""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HEADS = sorted((ROOT / "open_wake_word_compare").glob("*.onnx"))
FRONTENDS = ROOT / "data/external/oww_models/melspectrogram.onnx"

pytestmark = pytest.mark.skipif(not HEADS or not FRONTENDS.exists(),
                                reason="modèles openWakeWord absents")


def test_push_et_offline_concordent():
    from coachvocal.config import load_wakeword
    from coachvocal.evaluation.oww_adapter import OwwDetector

    d = OwwDetector(HEADS[0], load_wakeword("eloquence"))
    rng = np.random.default_rng(0)
    audio = (0.1 * rng.normal(0, 1, 16000 * 4)).astype(np.float32)
    audio[16000:24000] += 0.4 * np.sin(2 * np.pi * 600 *
                                       np.arange(8000) / 16000).astype(np.float32)

    p_off, _, _ = d.window_probas(audio)
    assert len(p_off) > 0

    p_push = []
    for i in range(0, len(audio) - d.hop, d.hop):
        ev = d.push(audio[i:i + d.hop])
        p_push.append(ev["proba"])
    # Le bug historique : push renvoyait 0.0 partout (tête jamais exécutée).
    # Les deux chemins doivent produire des dynamiques comparables.
    assert max(p_push) > 0 or float(p_off.max()) < 1e-4
    assert abs(float(np.median(p_push)) - float(np.median(p_off))) < 0.2

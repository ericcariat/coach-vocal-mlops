"""La machine à états est la logique métier du produit : elle se teste sans modèle.

On injecte des probabilités choisies à la main, ce qui permet de vérifier
exactement les trois règles (portail d'énergie, fenêtres consécutives, cooldown)
sans dépendre d'un entraînement.
"""

from __future__ import annotations

import numpy as np

from coachvocal.config import load_wakeword
from coachvocal.inference.detector import WakeWordDetector


class _ModeleFactice:
    """Renvoie les probabilités qu'on lui donne — pas de TensorFlow ici."""

    def __init__(self, probas):
        self.probas = list(probas)
        self.i = 0

    def predict(self, specs, **kwargs):  # noqa: ARG002
        return np.array(self.probas).reshape(-1, 1)


def _detecteur(probas=()):
    return WakeWordDetector(_ModeleFactice(probas), load_wakeword("eloquence"))


def test_trois_fenetres_consecutives_requises():
    d = _detecteur()
    peaks = np.ones(10)
    assert d.triggers_from([0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0, 0, 0, 0], peaks, 0.8) == []
    assert len(d.triggers_from([0.9] * 3 + [0.0] * 7, peaks, 0.8)) == 1


def test_portail_energie_bloque_le_silence():
    """Même à proba 99 %, une fenêtre sous le seuil d'énergie ne déclenche pas :
    c'est ce qui empêche le détecteur de halluciner dans une pièce vide."""
    d = _detecteur()
    silence = np.full(10, 0.001)          # < min_peak (0.02)
    assert d.triggers_from([0.99] * 10, silence, 0.8) == []


def test_cooldown_evite_les_declenchements_en_rafale():
    d = _detecteur()
    peaks = np.ones(60)
    trigs = d.triggers_from([0.99] * 60, peaks, 0.8)
    # 60 fenêtres × 125 ms = 7,5 s ; avec 1,5 s de cooldown on attend ~5 réveils,
    # certainement pas 58.
    assert 3 <= len(trigs) <= 6
    assert all(b - a >= d.live.cooldown_s for a, b in zip(trigs, trigs[1:]))


def test_seuil_plus_haut_declenche_moins():
    d = _detecteur()
    peaks = np.ones(20)
    probas = [0.6] * 10 + [0.95] * 10
    assert len(d.triggers_from(probas, peaks, 0.5)) >= len(d.triggers_from(probas, peaks, 0.9))


def test_instant_de_trigger_correspond_a_la_fin_de_fenetre():
    d = _detecteur()
    trigs = d.triggers_from([0.9] * 3 + [0.0] * 5, np.ones(8), 0.8)
    # 3e fenêtre : 2 × hop + durée de clip
    assert abs(trigs[0] - (2 * d.live.hop_s + 1.0)) < 1e-6

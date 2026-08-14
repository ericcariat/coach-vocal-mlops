"""La mesure du mot des fragments doit résister au souffle d'un vrai micro.

Cas réel qui a invalidé la première version (2026-08-14, écoute de l'auteur) :
un clip micro d'1 s avec un souffle à RMS ~0.001 et le mot de 0,335 s à 1,0 s.
Le seuil absolu 1e-5 voyait « du mot » partout → un f45 portait 68 % du mot.
"""

from __future__ import annotations

import numpy as np

from coachvocal.data.sources.fragments import word_span

SR = 16000


def _clip_micro(debut_s: float, fin_s: float, souffle: float = 0.001) -> np.ndarray:
    """Une seconde de souffle + un « mot » (bruit modulé) entre deux bornes."""
    rng = np.random.default_rng(7)
    audio = souffle * rng.standard_normal(SR).astype(np.float32)
    i0, i1 = int(debut_s * SR), int(fin_s * SR)
    audio[i0:i1] += 0.2 * rng.standard_normal(i1 - i0).astype(np.float32)
    return audio


def test_word_span_ignore_le_souffle():
    w0, w1 = word_span(_clip_micro(0.335, 1.0), SR)
    assert abs(w0 / SR - 0.335) < 0.04          # début du mot, pas du souffle
    assert w1 / SR > 0.95                        # fin du mot au bout du clip


def test_word_span_mot_au_debut_puis_zeros():
    # Cas historique des clips YouTube : mot puis padding de zéros stricts.
    audio = np.zeros(SR, np.float32)
    audio[: int(0.7 * SR)] = 0.2 * np.random.default_rng(3).standard_normal(
        int(0.7 * SR)).astype(np.float32)
    w0, w1 = word_span(audio, SR)
    assert w0 / SR < 0.04 and abs(w1 / SR - 0.7) < 0.04


def test_fragment_reste_une_fraction_du_mot():
    # La coupe se fait depuis les bornes du MOT : un f45 d'un mot de 0,665 s
    # doit embarquer ~0,30 s de voix, jamais 0,45 s de fin de clip.
    audio = _clip_micro(0.335, 1.0)
    w0, w1 = word_span(audio, SR)
    k = int(0.45 * (w1 - w0))
    assert k / SR < 0.32                         # 45 % de 0,665 s ≈ 0,30 s
    tail = audio[w1 - k:w1]
    # le fragment sortant est de la voix (pas du souffle) et de la bonne durée
    assert len(tail) == k
    assert float(np.sqrt((tail ** 2).mean())) > 0.05

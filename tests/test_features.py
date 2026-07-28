"""Le front-end acoustique doit être identique à l'entraînement et à l'inférence.

Une divergence ici produit un modèle qui « marche en test et rate en vrai », et
aucune métrique ne la détecte. D'où ces tests.
"""

from __future__ import annotations

import numpy as np

from coachvocal.audio.features import FeatureExtractor, sliding_windows
from coachvocal.config import load_wakeword


def _word():
    return load_wakeword("eloquence")


def test_forme_annoncee_egale_forme_reelle():
    """`input_shape` est calculée analytiquement (pour construire le modèle sans
    faire tourner TF) : elle doit coïncider avec la sortie réelle."""
    fx = FeatureExtractor(_word())
    reelle = fx(np.zeros(fx.wakeword.clip_samples, np.float32)).shape
    assert tuple(reelle) == fx.input_shape


def test_normalisation_invariante_au_gain():
    """Le z-score par exemple rend la représentation insensible au niveau
    d'enregistrement : un micro plus fort ne doit pas changer la décision."""
    fx = FeatureExtractor(_word())
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 0.1, fx.wakeword.clip_samples).astype(np.float32)
    a = fx(signal).numpy()
    b = fx(signal * 4.0).numpy()
    assert np.allclose(a, b, atol=1e-3)


def test_moyenne_nulle_ecart_type_unitaire():
    fx = FeatureExtractor(_word())
    rng = np.random.default_rng(1)
    spec = fx(rng.normal(0, 0.05, fx.wakeword.clip_samples).astype(np.float32)).numpy()
    assert abs(spec.mean()) < 1e-3
    assert abs(spec.std() - 1.0) < 1e-2


def test_lot_et_unite_donnent_le_meme_resultat():
    fx = FeatureExtractor(_word())
    rng = np.random.default_rng(2)
    clips = rng.normal(0, 0.1, (3, fx.wakeword.clip_samples)).astype(np.float32)
    lot = fx.batch(clips).numpy()
    for i in range(3):
        assert np.allclose(lot[i], fx(clips[i]).numpy(), atol=1e-5)


def test_fenetres_glissantes():
    word = _word()
    n = word.clip_samples
    hop = int(word.live.hop_s * word.sample_rate)
    audio = np.arange(3 * n, dtype=np.float32) / (3 * n)
    windows, peaks, starts = sliding_windows(audio, n, hop)
    assert windows.shape == (len(starts), n)
    assert starts[1] - starts[0] == hop
    assert np.allclose(peaks, np.abs(windows).max(axis=1))


def test_signal_trop_court_est_complete():
    word = _word()
    windows, _, _ = sliding_windows(np.zeros(100, np.float32), word.clip_samples, 2000)
    assert windows.shape[1] == word.clip_samples

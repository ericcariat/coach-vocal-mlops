"""Augmentations RIR et bruit multi-SNR (ROADMAP P1)."""

import numpy as np
import pytest
import soundfile as sf
import tensorflow as tf

from coachvocal.training.datasets import _load_bank, _noise_mix, _reverb

SR = 16000


@pytest.fixture(autouse=True)
def _cpu():
    tf.config.set_soft_device_placement(True)


def _tone(n=SR, freq=440.0, level=0.3):
    t = np.arange(n) / SR
    return (level * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_noise_mix_respecte_le_snr():
    audio = tf.constant(_tone()[np.newaxis, :])
    noise = tf.constant(np.random.default_rng(0).normal(0, 0.3, (4, SR)).astype(np.float32))
    tf.random.set_seed(1)
    mixed, _ = _noise_mix(audio, tf.constant([1.0]), noise, prob=1.0,
                          snr_lo=10.0, snr_hi=10.0)
    added = mixed.numpy()[0] - audio.numpy()[0]
    a_rms = np.sqrt(np.mean(audio.numpy() ** 2))
    n_rms = np.sqrt(np.mean(added ** 2))
    snr_db = 20 * np.log10(a_rms / n_rms)
    assert abs(snr_db - 10.0) < 1.0            # clip [-1,1] peut rogner un peu


def test_noise_mix_prob_zero_inchange():
    audio = tf.constant(_tone()[np.newaxis, :])
    noise = tf.constant(np.ones((2, SR), np.float32))
    mixed, _ = _noise_mix(audio, tf.constant([1.0]), noise, prob=0.0,
                          snr_lo=5.0, snr_hi=5.0)
    np.testing.assert_array_equal(mixed.numpy(), audio.numpy())


def test_reverb_conserve_forme_et_niveau():
    audio = tf.constant(_tone()[np.newaxis, :])
    # RIR = écho simple normalisé en énergie
    rir = np.zeros(SR // 2, np.float32)
    rir[0], rir[2000] = 1.0, 0.5
    rir /= np.sqrt((rir ** 2).sum())
    tf.random.set_seed(2)
    out, _ = _reverb(audio, tf.constant([1.0]), tf.constant(rir[np.newaxis, :]), prob=1.0)
    assert out.shape == audio.shape
    peak_in = np.abs(audio.numpy()).max()
    peak_out = np.abs(out.numpy()).max()
    assert abs(peak_out - peak_in) < 1e-3      # niveau crête préservé
    assert not np.allclose(out.numpy(), audio.numpy())   # mais signal modifié


def test_load_bank_rir_et_bruit(tmp_path):
    for i in range(3):
        sf.write(tmp_path / f"n{i}.wav",
                 np.random.default_rng(i).normal(0, 0.2, SR * 3).astype(np.float32), SR)
    bank = _load_bank(str(tmp_path), SR, SR, bank_size=8, seed=42)
    assert bank.shape == (3, SR)               # bruits : crop à la longueur du clip
    rirs = _load_bank(str(tmp_path), SR, None, bank_size=8, seed=42)
    assert rirs.shape == (3, SR // 2)          # RIR : 0,5 s, énergie unitaire
    np.testing.assert_allclose(np.sqrt((rirs.numpy() ** 2).sum(axis=1)), 1.0, atol=1e-5)


def test_load_bank_dossier_vide(tmp_path):
    assert _load_bank(str(tmp_path / "vide"), SR, None, 8, 42) is None


# ── Placement du mot dans la fenêtre (TTS / studio) ───────────────────────────

def test_place_word_end_aligne_la_fin():
    from coachvocal.data.tts import place_word
    word = np.ones(8000, np.float32)             # « mot » de 0,5 s
    clip = place_word(word, SR, "end", margin=1600)   # marge 100 ms
    assert len(clip) == SR
    assert clip[-1600:].max() == 0.0             # marge de queue silencieuse
    assert clip[-1601] == 1.0                    # fin du mot juste avant
    assert clip[: SR - 8000 - 1600].max() == 0.0  # tête = padding


def test_place_word_center_historique():
    from coachvocal.data.tts import place_word
    word = np.ones(8000, np.float32)
    clip = place_word(word, SR, "center")
    assert len(clip) == SR
    lead = int((clip != 0).argmax())
    assert abs(lead - (SR - 8000) // 2) <= 1     # centré


def test_place_word_mot_plus_long_que_la_fenetre():
    from coachvocal.data.tts import place_word
    word = np.arange(SR + 4000, dtype=np.float32)
    clip = place_word(word, SR, "end")
    assert len(clip) == SR
    assert clip[-1] == word[-1]                  # end : on garde la FIN du mot

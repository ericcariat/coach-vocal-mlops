"""Porte qualité (ADR-007) : mesures, jugement à trois sorties, exclusions."""

import json

import numpy as np
import pytest
import soundfile as sf

from coachvocal.config import QualityGateConfig
from coachvocal.data import gate

SR = 16000


def _write(tmp_path, name, audio, sr=SR):
    p = tmp_path / name
    sf.write(p, audio.astype(np.float32), sr)
    return p


def _speech_like(dur_s=1.0, level=0.3, seed=0):
    """Signal type parole : syllabes énergiques séparées de quasi-pauses,
    pour un contraste fort/faible réaliste (l'estimateur SNR compare les
    trames de 100 ms fortes aux faibles)."""
    rng = np.random.default_rng(seed)
    n = int(SR * dur_s)
    frame = SR // 10                             # tranches de 100 ms, alignées
    envelope = np.ones(n)
    for i in range(0, n // frame, 3):            # 1 tranche calme sur 3
        envelope[i * frame:(i + 1) * frame] = 0.02
    return level * envelope * rng.normal(0, 0.5, n).clip(-1, 1)


@pytest.fixture
def cfg():
    return QualityGateConfig(enabled=True)


def test_clip_sain_accepte(tmp_path, cfg):
    f = _write(tmp_path, "ok.wav", _speech_like())
    m = gate.measure_clip(f, SR)
    verdict, raisons = gate.judge_clip(m, cfg, SR)
    assert verdict == "accepte", raisons


def test_clip_muet_rejete(tmp_path, cfg):
    f = _write(tmp_path, "muet.wav", np.zeros(SR))
    verdict, raisons = gate.judge_clip(gate.measure_clip(f, SR), cfg, SR)
    assert verdict == "rejete"
    assert any("muet" in r for r in raisons)


def test_clip_sature_rejete(tmp_path, cfg):
    audio = _speech_like()
    audio[: SR // 2] = 1.0                       # 50 % d'échantillons saturés
    f = _write(tmp_path, "sat.wav", audio)
    verdict, raisons = gate.judge_clip(gate.measure_clip(f, SR), cfg, SR)
    assert verdict == "rejete"
    assert any("saturation" in r for r in raisons)


def test_duree_anormale_rejetee(tmp_path, cfg):
    f = _write(tmp_path, "long.wav", _speech_like(dur_s=3.0))
    verdict, _ = gate.judge_clip(gate.measure_clip(f, SR), cfg, SR)
    assert verdict == "rejete"


def test_mauvais_sample_rate_rejete(tmp_path, cfg):
    f = _write(tmp_path, "sr.wav", _speech_like(), sr=44100)
    verdict, raisons = gate.judge_clip(gate.measure_clip(f, SR), cfg, SR)
    assert verdict == "rejete"
    assert any("sr=" in r for r in raisons)


def test_fin_chargee_douteuse(tmp_path, cfg):
    """Un mot suivi d'un autre mot qui déborde : dernière tranche énergique."""
    audio = _speech_like(level=0.05)             # clip globalement calme
    audio[-SR // 10:] = 0.4                      # 100 ms finales très fortes
    f = _write(tmp_path, "fin.wav", audio)
    m = gate.measure_clip(f, SR)
    verdict, raisons = gate.judge_clip(m, cfg, SR, pool="yt_positif")
    assert verdict == "douteux"
    assert any("fin chargée" in r for r in raisons)
    # le même clip dans un pool de parole continue n'est PAS douteux
    verdict, _ = gate.judge_clip(m, cfg, SR, pool="cv_fr")
    assert verdict == "accepte"


def test_padding_zeros_mesure(tmp_path, cfg):
    audio = _speech_like()
    audio[-SR // 4:] = 0.0                       # 250 ms de zéros stricts
    f = _write(tmp_path, "pad.wav", audio)
    m = gate.measure_clip(f, SR)
    assert 200 <= m["tail_zeros_ms"] <= 300


def test_pool_bruit_indulgent(tmp_path, cfg):
    """Un bruit de fond faible et sans contraste est NORMAL pour musan_noise —
    mais un fichier muet y reste rejeté."""
    faint = _write(tmp_path, "fond.wav", 0.01 * np.random.default_rng(1).normal(0, 0.5, SR))
    m = gate.measure_clip(faint, SR)
    verdict, _ = gate.judge_clip(m, cfg, SR, pool="musan_noise")
    assert verdict == "accepte"
    verdict, _ = gate.judge_clip(m, cfg, SR, pool="cv_fr")
    assert verdict == "douteux"                  # même clip, pool de parole
    muet = _write(tmp_path, "muet2.wav", np.zeros(SR))
    verdict, _ = gate.judge_clip(gate.measure_clip(muet, SR), cfg, SR, pool="musan_noise")
    assert verdict == "rejete"


def test_illisible_rejete(tmp_path, cfg):
    f = tmp_path / "corrompu.wav"
    f.write_bytes(b"pas du wav")
    verdict, raisons = gate.judge_clip(gate.measure_clip(f, SR), cfg, SR)
    assert verdict == "rejete"
    assert any("illisible" in r for r in raisons)


def test_run_gate_et_exclusions(tmp_path, cfg, monkeypatch):
    """Bout en bout : rapport écrit, exclusions = rejetés + douteux non tranchés,
    un « oui » humain réintègre, un « non » exclut un accepté."""
    monkeypatch.setattr(gate, "gate_dir", lambda w: tmp_path / "gate")
    ok = _write(tmp_path, "ok.wav", _speech_like())
    muet = _write(tmp_path, "muet.wav", np.zeros(SR))
    doute = _write(tmp_path, "doute.wav",
                   np.concatenate([_speech_like(0.9, 0.05), 0.4 * np.ones(SR // 10)]))

    report = gate.run_gate({"yt_positif": [ok, muet, doute]}, cfg, SR, "testword",
                           verbose=False)
    assert report["counts"] == {"accepte": 1, "rejete": 1, "douteux": 1}

    excluded = gate.load_exclusions("testword", "exclude")
    assert excluded == {"muet.wav", "doute.wav"}
    # politique include : le douteux non tranché reste
    assert gate.load_exclusions("testword", "include") == {"muet.wav"}

    # verdicts humains : oui sur le douteux, non sur l'accepté
    (tmp_path / "gate" / gate.HUMAN_NAME).write_text(json.dumps({
        str(doute): {"verdict": "oui", "saved_at": "2026-08-13"},
        str(ok): {"verdict": "non", "saved_at": "2026-08-13"},
    }))
    excluded = gate.load_exclusions("testword", "exclude")
    assert excluded == {"muet.wav", "ok.wav"}


def test_exclusions_sans_rapport():
    assert gate.load_exclusions("mot_inexistant", "exclude") is None


# ── Source studio : consommation d'une session (sans micro) ───────────────────

def test_source_studio_consomme_une_session(tmp_path, monkeypatch):
    """Une session simulée : seules les prises keep=true, recadrées 1 s,
    partent au train — le contrat de la page Studio."""
    import json

    import soundfile as _sf

    from coachvocal.config import SourceConfig, load_wakeword
    from coachvocal.data.sources import SourceContext
    from coachvocal.data.sources.studio import studio as studio_source

    word = load_wakeword("eloquence")
    session = tmp_path / "word" / "studio" / "2026-08-13"
    session.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name in ("normal_01.wav", "normal_02.wav", "fort_01.wav"):
        _sf.write(session / name,
                  (0.3 * rng.normal(0, 0.5, int(1.5 * 16000))).astype(np.float32), 16000)
    (session / "metadata.json").write_text(json.dumps({"takes": {
        "normal_01.wav": {"keep": True, "condition": "normal"},
        "normal_02.wav": {"keep": False, "condition": "normal"},   # refusée
        "fort_01.wav": {"keep": True, "condition": "fort"},
    }}))

    class Ctx(SourceContext):
        @property
        def word_dir(self):
            return tmp_path / "word"

        def cache(self, name):
            return tmp_path / "cache" / name

    ctx = Ctx(wakeword=word, dataset=None)
    src = SourceConfig(name="studio_positif", type="studio", label=1)
    pools = studio_source(src, ctx)
    assert len(pools["train"]) == 2                    # keep=false écartée
    assert pools["val"] == [] and pools["test"] == []  # train uniquement
    for f in pools["train"]:
        audio, sr = _sf.read(f)
        assert sr == 16000 and len(audio) == word.clip_samples   # recadrée 1 s

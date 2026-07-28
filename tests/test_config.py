"""La config est le contrat du projet : si elle se charge mal, tout le reste ment."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from coachvocal.config import (
    deep_merge,
    list_experiments,
    load_experiment,
    load_wakeword,
    set_by_path,
)


def test_toutes_les_experiences_sont_valides():
    """Une config invalide doit échouer ici, pas après 20 min d'entraînement."""
    names = list_experiments()
    assert names, "aucune expérience dans configs/experiment/"
    for name in names:
        cfg = load_experiment(name)
        assert cfg.dataset.sources, f"{name} : dataset sans source"
        assert cfg.model.arch


def test_extends_herite_et_surcharge_par_nom_de_source():
    base = load_experiment("v03_replica")
    noms = {s.name for s in base.dataset.sources}
    # tts500 hérite de base : toutes les sources du parent + la source TTS
    assert {"gsc", "cv_fr", "musan_noise", "proches"} <= noms
    assert "tts_positif" in noms
    tts = next(s for s in base.dataset.sources if s.name == "tts_positif")
    assert tts.params["dose"] == 500
    assert tts.splits == ["train"], "le synthétique ne doit jamais entrer en val/test"


def test_surcharge_cli():
    cfg = load_experiment("v03_replica", ["training.epochs=3", "training.seeds=[7]"])
    assert cfg.training.epochs == 3
    assert cfg.training.seeds == [7]


def test_cle_inconnue_rejetee():
    with pytest.raises(ValidationError):
        load_experiment("v03_replica", ["training.epochsss=3"])


def test_deep_merge_fusionne_les_sources_par_nom():
    base = {"sources": [{"name": "a", "copies": 1}, {"name": "b"}]}
    out = deep_merge(base, {"sources": [{"name": "a", "copies": 5}, {"name": "c"}]})
    par_nom = {s["name"]: s for s in out["sources"]}
    assert par_nom["a"]["copies"] == 5
    assert set(par_nom) == {"a", "b", "c"}


def test_set_by_path_parse_en_yaml():
    cfg = set_by_path({}, "a.b.c", "[1, 2]")
    assert cfg["a"]["b"]["c"] == [1, 2]


def test_seuil_live_plus_strict_que_le_seuil_devaluation():
    """Choix assumé : en always-on, une fausse alarme coûte plus qu'un mot raté."""
    word = load_wakeword("eloquence")
    cfg = load_experiment("v03_replica")
    assert word.live.threshold > cfg.training.threshold

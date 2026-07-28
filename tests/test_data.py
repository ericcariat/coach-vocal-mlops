"""Anti-fuite et traçabilité : les deux propriétés qui rendent les métriques crédibles."""

from __future__ import annotations

import pytest

from coachvocal.data.builder import check_leakage
from coachvocal.data.manifest import Manifest
from coachvocal.data.splits import assign, group_of


def test_groupe_deduit_du_nom_de_fichier():
    assert group_of("yt_dQw4w9WgXcQ_012.wav") == ("youtube", "yt_dQw4w9WgXcQ")
    assert group_of("moi_20260719_003.wav")[0] == "moi"


def test_affectation_deterministe_et_stable():
    """Même groupe → même split, quel que soit le moment ou la machine."""
    for g in ("yt_aaaaaaaaaaa", "yt_bbbbbbbbbbb", "moi_2026"):
        assert assign(g) == assign(g)
    assert assign("yt_aaaaaaaaaaa", seed=1) in ("train", "val", "test")


def test_repartition_globalement_conforme_au_ratio():
    groupes = [f"yt_{i:011d}" for i in range(500)]
    parts = [assign(g) for g in groupes]
    assert 0.70 < parts.count("train") / len(parts) < 0.90


def test_boost_duplique_les_chemins():
    m = Manifest()
    m.add("moi_positif", ["a.wav", "b.wav"], label=1, split="train", copies=10)
    paths, labels = m.paths_labels("train")
    assert len(paths) == 20 and sum(labels) == 20
    assert m.balance()["train"]["pos"] == 20


def test_empreinte_independante_de_lordre():
    a, b = Manifest(), Manifest()
    a.add("p", ["x.wav", "y.wav"], 1, "train")
    b.add("p", ["y.wav", "x.wav"], 1, "train")
    assert a.fingerprint() == b.fingerprint()


def test_empreinte_change_si_le_boost_change():
    a, b = Manifest(), Manifest()
    a.add("p", ["x.wav"], 1, "train", copies=1)
    b.add("p", ["x.wav"], 1, "train", copies=10)
    assert a.fingerprint() != b.fingerprint()


def test_fuite_train_test_detectee():
    m = Manifest()
    m.add("p", ["/data/clip.wav"], 1, "train")
    m.add("p", ["/autre/clip.wav"], 1, "test")     # même nom de fichier
    with pytest.raises(RuntimeError, match="FUITE"):
        check_leakage(m)


def test_pas_de_fuite_cas_nominal():
    m = Manifest()
    m.add("p", ["/data/a.wav"], 1, "train")
    m.add("p", ["/data/b.wav"], 1, "test")
    check_leakage(m)

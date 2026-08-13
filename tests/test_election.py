"""Élection des candidats : val_loss historique et fa_ambient sous contrainte."""

import pytest

from coachvocal.config import TrainingConfig
from coachvocal.training.trainer import _elect


def _c(seed, val_loss=0.1, val_recall=0.95, fa=None, val_accuracy=0.9):
    return {"seed": seed, "val_loss": val_loss, "val_recall": val_recall,
            "fa_ambient": fa, "val_accuracy": val_accuracy}


def test_val_loss_historique():
    cands = [_c(1, val_loss=0.2), _c(2, val_loss=0.1), _c(3, val_loss=0.3)]
    best, motif = _elect(cands, TrainingConfig())
    assert best["seed"] == 2 and "val_loss" in motif


def test_fa_ambient_sous_contrainte():
    cfg = TrainingConfig(selection_metric="fa_ambient", selection_min_val_recall=0.9)
    cands = [_c(1, val_recall=0.95, fa=50.0),
             _c(2, val_recall=0.92, fa=20.0),
             _c(3, val_recall=0.80, fa=5.0)]   # FA superbes mais sourd → écarté
    best, motif = _elect(cands, cfg)
    assert best["seed"] == 2
    assert "FA/h" in motif


def test_fa_ambient_repli_si_contrainte_introuvable():
    cfg = TrainingConfig(selection_metric="fa_ambient", selection_min_val_recall=0.99)
    cands = [_c(1, val_recall=0.90, fa=10.0), _c(2, val_recall=0.95, fa=40.0)]
    best, motif = _elect(cands, cfg)
    assert best["seed"] == 2                   # meilleur rappel, pas meilleures FA
    assert "AUCUN" in motif


def test_fa_ambient_sans_flux_echoue():
    cfg = TrainingConfig(selection_metric="fa_ambient")
    with pytest.raises(RuntimeError):
        _elect([_c(1, fa=None)], cfg)

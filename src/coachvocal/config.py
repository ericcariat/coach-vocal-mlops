"""Schémas de configuration (pydantic) + composition des YAML.

Principe du projet : **aucun hyperparamètre en dur dans le code**. Une
expérience = un fichier `configs/experiment/<nom>.yaml` qui référence un
mot-clé, une recette de dataset et une architecture. Le schéma pydantic sert à
la fois de validation (une faute de frappe échoue tout de suite, pas après
20 minutes d'entraînement) et de documentation auto des paramètres — c'est ce
que l'UI Streamlit affiche dans l'onglet « Entraînement ».

Composition :
    experiment.yaml  →  wakeword/<ref>.yaml
                     →  dataset/<ref>.yaml
                     →  model/<ref>.yaml
                     +  bloc `training` inline (surchargeable en CLI)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import paths

SPLITS = ("train", "val", "test")


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid")   # une clé inconnue = erreur immédiate


# ── Mot-clé ───────────────────────────────────────────────────────────────────
class AudioConfig(Base):
    """Front-end acoustique. Ces valeurs DOIVENT être identiques à
    l'entraînement et à l'inférence — c'est pourquoi elles vivent dans la config
    du mot-clé et pas dans le script d'entraînement."""

    frame_length: int = 255
    frame_step: int = 128
    num_mel_bins: int = 40
    lower_edge_hertz: float = 20.0
    upper_edge_hertz: float | None = None      # None → sample_rate / 2
    clip_seconds: float = 1.0


class LiveConfig(Base):
    """Machine à états du détecteur always-on (identique en live et au banc)."""

    threshold: float = 0.80
    hop_s: float = 0.125
    n_consecutive: int = 3
    cooldown_s: float = 1.5
    min_peak: float = 0.02                     # portail d'énergie anti-silence


class TTSVoice(Base):
    name: str
    model: str                                 # chemin du .onnx Piper
    speaker: int = 0


class TTSConfig(Base):
    """Recette de génération du pool de positifs synthétiques."""

    text: str                                   # le point final stabilise la prosodie
    voices: list[TTSVoice] = []
    length_scales: list[float] = [0.85, 0.95, 1.0, 1.05, 1.15]
    noise_scales: list[float] = [0.5, 0.667, 0.9]
    per_combo: int = 50
    pool_name: str = "tts_positives"


class WakewordConfig(Base):
    name: str
    classes: list[str] = ["pas_eloquence", "eloquence"]
    sample_rate: int = 16000
    audio: AudioConfig = AudioConfig()
    live: LiveConfig = LiveConfig()
    tts: TTSConfig | None = None

    @property
    def clip_samples(self) -> int:
        return int(self.audio.clip_seconds * self.sample_rate)


# ── Dataset ───────────────────────────────────────────────────────────────────
class SourceConfig(Base):
    """Une source de données dans la recette.

    `type` désigne un producteur enregistré dans `coachvocal.data.sources`.
    `copies` est le « boost » (duplication des chemins) qui pondère une source
    rare sans toucher à la loss — recette conservée depuis le run v01.
    """

    name: str                                   # nom du pool dans le manifest
    type: str                                   # producteur (cf. sources/registry)
    label: Literal[0, 1]
    copies: int = 1
    splits: list[str] = ["train", "val", "test"]
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class AugmentationConfig(Base):
    time_shift_ms: int = 100                   # décalage non circulaire ±N ms
    speed_min: float = 0.85
    speed_max: float = 1.15
    enabled: bool = True
    # ── Réverbération par réponses impulsionnelles (RIR) — ROADMAP P1 ────────
    # Absente des recettes historiques (prob 0 = comportement inchangé).
    # Recommandée par openWakeWord/microWakeWord/LiveKit : le direct sans
    # réverbération n'existe pas en conditions réelles.
    rir_prob: float = 0.0                      # proba par clip d'appliquer une RIR
    rir_dir: str = "data/external/rir_mit/16k" # RIR 16 kHz (MIT IR Survey)
    # ── Bruit additif à SNR tiré dans une plage (multi-SNR explicite) ────────
    noise_prob: float = 0.0                    # proba par clip de mélanger un bruit
    noise_snr_db: list[float] = [5.0, 20.0]    # plage de SNR (uniforme)
    noise_dir: str = "data/external/musan/noise"
    bank_size: int = 256                       # fichiers max chargés par banque


class QualityGateConfig(Base):
    """Porte qualité à trois sorties (accepté / rejeté / douteux) — ADR-007.

    Deux niveaux de seuils : le seuil FRANC rejette automatiquement, la zone
    entre les deux envoie le clip dans la file d'audit humain (Streamlit).
    Opt-in par recette : `enabled: false` laisse les recettes historiques
    strictement comparables."""

    enabled: bool = False
    min_duration_s: float = 0.3
    max_duration_s: float = 2.0
    reject_peak_below: float = 0.001           # muet
    doubt_peak_below: float = 0.02
    reject_saturation_above: float = 0.05      # >5 % d'échantillons saturés
    doubt_saturation_above: float = 0.01
    reject_snr_db_below: float = 0.0
    doubt_snr_db_below: float = 6.0            # seuil d'avertissement ViolaWake
    doubt_tail_energy_above: float = 0.5       # fin « chargée » (mot suivant ?)
    frame_ms: int = 100
    # Le contrôle de fin chargée n'a de sens que pour des clips de MOT ISOLÉ :
    # une tranche de parole continue ou de bruit finit énergique par nature.
    tail_check_pools: list[str] = ["positif", "proche", "guided", "moi"]
    # Pools de bruit/musique : faibles et sans contraste par nature — les
    # contrôles de DOUTE (pic faible, SNR) n'y signifient rien ; les rejets
    # francs (muet, saturé, durée, sr) s'appliquent toujours.
    lenient_pools: list[str] = ["musan", "noise", "music"]
    # Un douteux non tranché par l'humain est exclu (« exclude ») ou gardé
    # (« include ») — la règle du projet : le doute n'entre pas sans audit.
    doubt_policy: Literal["exclude", "include"] = "exclude"
    # Pools jamais filtrés (silencieux par conception — cf. QUIET_BY_DESIGN)
    skip_pools: list[str] = ["silence", "fragments"]


class DatasetConfig(Base):
    name: str
    data_seed: int = 42                        # seed DONNÉES — ne jamais changer
    splits_csv: str = "splits.csv"
    sources: list[SourceConfig]
    augmentation: AugmentationConfig = AugmentationConfig()
    quality_gate: QualityGateConfig = QualityGateConfig()

    def enabled_sources(self) -> list[SourceConfig]:
        return [s for s in self.sources if s.enabled]


# ── Modèle ────────────────────────────────────────────────────────────────────
class ModelConfig(Base):
    name: str
    arch: str                                   # clé du registre de modèles
    params: dict[str, Any] = Field(default_factory=dict)


# ── Entraînement ──────────────────────────────────────────────────────────────
class TrainingConfig(Base):
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    early_stopping_patience: int = 5
    class_weight: bool = True
    threshold: float = 0.5                      # seuil d'évaluation par clip
    # Protocole anti-variance CPU : N candidats, élu par la VAL, jamais le test.
    seeds: list[int] = [42]
    # "val_loss" | "val_accuracy" | "fa_ambient" — ce dernier élit le candidat
    # aux FA/h minimales sur le flux ambiant de validation
    # (data/wakewords/<mot>/val_ambient/, jamais les mêmes enregistrements que
    # le banc), sous contrainte de rappel val (formulation produit,
    # cf. microWakeWord/LiveKit — ROADMAP P1).
    selection_metric: str = "val_loss"
    selection_min_val_recall: float = 0.90     # contrainte pour fa_ambient
    use_gpu: bool = False                       # cf. docs/decisions/ADR-002


class ExperimentConfig(Base):
    name: str
    description: str = ""
    wakeword: WakewordConfig
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig = TrainingConfig()

    @model_validator(mode="after")
    def _check_splits(self):
        for s in self.dataset.sources:
            bad = set(s.splits) - set(SPLITS)
            if bad:
                raise ValueError(f"source '{s.name}' : splits inconnus {bad}")
        return self


# ── Chargement / composition ──────────────────────────────────────────────────
def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config introuvable : {path}")
    return yaml.safe_load(path.read_text()) or {}


def _load_file(kind: str, ref: str) -> dict:
    """Charge un fichier de config en résolvant `extends` (héritage de recette).

    `extends` évite de recopier les 12 sources d'un dataset pour n'en changer
    qu'une : une variante déclare son parent et seulement ses différences."""
    raw = _read_yaml(paths.CONFIGS / kind / f"{ref}.yaml")
    parent = raw.pop("extends", None)
    return deep_merge(_load_file(kind, parent), raw) if parent else raw


def _resolve(kind: str, ref: str | dict) -> dict:
    """Un champ peut être soit un nom de fichier (`"eloquence"`), soit un bloc
    inline. Le cas fichier + surcharge inline est supporté via `{ref: nom, ...}`."""
    if isinstance(ref, str):
        return _load_file(kind, ref)
    if "ref" in ref:
        base = _load_file(kind, ref["ref"])
        return deep_merge(base, {k: v for k, v in ref.items() if k != "ref"})
    return ref


def deep_merge(base: dict, override: dict) -> dict:
    """Fusion récursive. Cas particulier des listes de **sources** : elles sont
    fusionnées par `name` (on remplace la source homonyme, on ajoute les autres),
    sinon hériter d'une recette obligerait à recopier toute la liste."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        elif k == "sources" and isinstance(v, list) and isinstance(out.get(k), list):
            merged = {s["name"]: s for s in out[k]}
            for s in v:
                merged[s["name"]] = deep_merge(merged.get(s["name"], {}), s)
            out[k] = list(merged.values())
        else:
            out[k] = v
    return out


def set_by_path(cfg: dict, dotted: str, value: str) -> dict:
    """Surcharge CLI : `--set training.epochs=5`. La valeur est parsée en YAML
    (donc `5` → int, `[42,43]` → liste, `true` → bool)."""
    node = cfg
    *parents, leaf = dotted.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = yaml.safe_load(value)
    return cfg


def load_experiment(name: str, overrides: list[str] | None = None) -> ExperimentConfig:
    """Charge et valide `configs/experiment/<name>.yaml` avec ses références."""
    raw = _read_yaml(paths.CONFIGS / "experiment" / f"{name}.yaml")
    raw.setdefault("name", name)
    for kind in ("wakeword", "dataset", "model"):
        if kind not in raw:
            raise ValueError(f"experiment '{name}' : champ '{kind}' manquant")
        raw[kind] = _resolve(kind, raw[kind])
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"surcharge invalide '{ov}' (attendu cle.sous_cle=valeur)")
        key, value = ov.split("=", 1)
        set_by_path(raw, key, value)
    return ExperimentConfig(**raw)


def load_wakeword(name: str) -> WakewordConfig:
    return WakewordConfig(**_read_yaml(paths.CONFIGS / "wakeword" / f"{name}.yaml"))


def list_experiments() -> list[str]:
    return sorted(p.stem for p in (paths.CONFIGS / "experiment").glob("*.yaml"))

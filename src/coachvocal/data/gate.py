"""Porte qualité — le filtre à trois sorties de la construction des pools.

Philosophie (cf. `docs/ROADMAP.md` P0 et ADR-007) : la machine mesure et trie,
l'humain n'intervient que sur le doute.

- **accepte** : dans les seuils → entre dans le jeu de données ;
- **rejete**  : hors seuils francs → écarté automatiquement, mais listé (jamais
  supprimé silencieusement — les fichiers restent sur disque) ;
- **douteux** : zone grise → file d'audit humain (page Streamlit « Qualité »),
  verdict oui/non persisté dans `human.json`.

La porte est OPT-IN par recette (`dataset.quality_gate.enabled`) : les recettes
historiques restent comparables. L'activer change l'empreinte du dataset, donc
c'est une expérience comme une autre, jugée au banc.

Les mesures sont volontairement identiques à celles de l'audit du 2026-08-13
(énergie par tranches de 100 ms) et réutilisables par le futur studio
d'enregistrement — un seul code pour la collecte et le nettoyage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import paths
from ..config import QualityGateConfig

REPORT_NAME = "gate_report.json"
HUMAN_NAME = "gate_human.json"


def gate_dir(wakeword: str) -> Path:
    return paths.word_dir(wakeword) / "gate"


# ── Mesures par clip ──────────────────────────────────────────────────────────

def measure_clip(path: Path, sample_rate: int, frame_ms: int = 100) -> dict:
    """Mesures objectives d'un clip. Ne juge pas — voir `judge_clip`."""
    try:
        audio, sr = sf.read(path, dtype="float32")
    except Exception as exc:
        return {"error": f"illisible : {exc.__class__.__name__}"}
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        return {"error": "vide"}

    frame = max(1, int(sr * frame_ms / 1000))
    n_frames = max(1, len(audio) // frame)
    rms_frames = np.array([
        float(np.sqrt(np.mean(audio[i * frame:(i + 1) * frame] ** 2)))
        for i in range(n_frames)])
    rms_median = float(np.median(rms_frames))
    floor = float(np.percentile(rms_frames, 10)) + 1e-9
    voice = float(np.percentile(rms_frames, 90)) + 1e-9

    # Padding de zéros stricts en queue (signature de la découpe actuelle)
    tail_zeros = 0
    for x in audio[::-1]:
        if x != 0.0:
            break
        tail_zeros += 1

    return {
        "duration_s": round(len(audio) / sr, 3),
        "sample_rate": sr,
        "rms": round(float(np.sqrt(np.mean(audio ** 2))), 5),
        "peak": round(float(np.abs(audio).max()), 5),
        "saturation_ratio": round(float(np.mean(np.abs(audio) >= 0.999)), 5),
        # SNR grossier : contraste entre trames fortes et faibles (cf. ViolaWake)
        "snr_db": round(20 * float(np.log10(voice / floor)), 1),
        # Énergie des bords vs médiane du clip (audit du 2026-08-13)
        "head_energy_ratio": round(float(rms_frames[0] / (rms_median + 1e-9)), 3),
        "tail_energy_ratio": round(float(rms_frames[-1] / (rms_median + 1e-9)), 3),
        "tail_zeros_ms": int(tail_zeros / sr * 1000),
    }


# ── Jugement à trois sorties ──────────────────────────────────────────────────

def judge_clip(m: dict, cfg: QualityGateConfig, sample_rate: int,
               pool: str = "") -> tuple[str, list[str]]:
    """(verdict, raisons). Rejet = hors seuil franc ; douteux = zone grise.

    Le contexte `pool` module les contrôles de DOUTE : la fin chargée n'a de
    sens que pour un mot isolé (`tail_check_pools`), et pic faible/SNR n'en ont
    aucun pour du bruit de fond (`lenient_pools`). Les rejets francs (muet,
    saturé, durée, sample rate) s'appliquent toujours, à tous les pools."""
    if "error" in m:
        return "rejete", [m["error"]]
    reject, doubt = [], []
    lenient = any(tag in pool for tag in cfg.lenient_pools)
    tail_check = any(tag in pool for tag in cfg.tail_check_pools)

    if m["sample_rate"] != sample_rate:
        reject.append(f"sr={m['sample_rate']} (attendu {sample_rate})")
    if not (cfg.min_duration_s <= m["duration_s"] <= cfg.max_duration_s):
        reject.append(f"durée {m['duration_s']} s")
    if m["peak"] < cfg.reject_peak_below:
        reject.append("muet")
    elif m["peak"] < cfg.doubt_peak_below and not lenient:
        doubt.append(f"très faible (pic {m['peak']})")
    if m["saturation_ratio"] > cfg.reject_saturation_above:
        reject.append(f"saturation {m['saturation_ratio']:.1%}")
    elif m["saturation_ratio"] > cfg.doubt_saturation_above:
        doubt.append(f"saturation {m['saturation_ratio']:.1%}")
    if not lenient:
        if m["snr_db"] < cfg.reject_snr_db_below:
            reject.append(f"SNR {m['snr_db']} dB")
        elif m["snr_db"] < cfg.doubt_snr_db_below:
            doubt.append(f"SNR {m['snr_db']} dB")
    if tail_check and m["tail_energy_ratio"] > cfg.doubt_tail_energy_above:
        doubt.append(f"fin chargée ({m['tail_energy_ratio']:.2f}× la médiane)")

    if reject:
        return "rejete", reject
    if doubt:
        return "douteux", doubt
    return "accepte", []


# ── Passage d'un lot et persistance ──────────────────────────────────────────

def run_gate(files: dict[str, list[Path]], cfg: QualityGateConfig, sample_rate: int,
             wakeword: str, verbose: bool = True) -> dict:
    """Mesure et juge chaque fichier de chaque pool ; écrit `gate_report.json`.

    `files` : {nom_de_pool: [chemins]}. Le rapport est cumulatif par chemin
    absolu — relancer la porte remplace les entrées mesurées."""
    out = gate_dir(wakeword)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / REPORT_NAME
    report = json.loads(report_path.read_text()) if report_path.exists() else {"clips": {}}

    counts = {"accepte": 0, "rejete": 0, "douteux": 0}
    for pool, paths_ in sorted(files.items()):
        for f in paths_:
            m = measure_clip(f, sample_rate, cfg.frame_ms)
            verdict, reasons = judge_clip(m, cfg, sample_rate, pool)
            report["clips"][str(f)] = {"pool": pool, "verdict": verdict,
                                       "raisons": reasons, **m}
            counts[verdict] += 1
        if verbose:
            print(f"  {pool:<18} {len(paths_)} clip(s)")

    report["config"] = cfg.model_dump()
    report["counts"] = {k: sum(1 for c in report["clips"].values() if c["verdict"] == k)
                        for k in counts}
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    if verbose:
        c = report["counts"]
        print(f"\n  Porte qualité : {c['accepte']} acceptés · {c['rejete']} rejetés"
              f" · {c['douteux']} douteux → {report_path}")
    return report


def load_exclusions(wakeword: str, doubt_policy: str) -> set[str] | None:
    """Noms de fichiers à écarter du build, ou None si aucune porte n'a tourné.

    Exclus : les « rejete » automatiques, les « douteux » selon la politique
    (`exclude` tant que l'humain n'a pas tranché), et les « non » humains.
    Un « oui » humain réintègre un douteux."""
    report_path = gate_dir(wakeword) / REPORT_NAME
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    human_path = gate_dir(wakeword) / HUMAN_NAME
    human = json.loads(human_path.read_text()) if human_path.exists() else {}

    excluded: set[str] = set()
    for path, c in report["clips"].items():
        name = Path(path).name
        verdict = c["verdict"]
        h = human.get(path, {}).get("verdict")
        if verdict == "rejete" or h == "non":
            excluded.add(name)
        elif verdict == "douteux" and h != "oui" and doubt_policy == "exclude":
            excluded.add(name)
    return excluded

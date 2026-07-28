"""Détection always-on au micro + mode « test guidé ».

Le live consomme le micro par blocs de `hop` échantillons (125 ms) et applique
la machine à états partagée (`detector.py`). Les buffers des déclenchements sont
sauvegardés : ce sont les hard negatives de demain — une fausse alarme réelle
vaut dix négatifs téléchargés.
"""

from __future__ import annotations

import datetime as dt
import queue
from pathlib import Path

import numpy as np

from ..config import WakewordConfig
from .detector import WakeWordDetector, load_detector


def list_devices() -> list[dict]:
    import sounddevice as sd

    return [{"index": i, "name": d["name"], "channels": d["max_input_channels"]}
            for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


def stream(model_path: Path, wakeword: WakewordConfig, device: int | None = None,
           threshold: float | None = None, save_dir: Path | None = None,
           on_event=None) -> None:
    """Boucle temps réel. Ctrl-C pour sortir."""
    import sounddevice as sd
    import soundfile as sf

    detector = load_detector(model_path, wakeword, threshold)
    q: queue.Queue = queue.Queue()
    hop = detector.hop
    history: list[np.ndarray] = []

    def callback(indata, frames, time_info, status):   # noqa: ARG001
        if status:
            print(f"⚠️  {status}")
        q.put(indata[:, 0].copy())

    print(f"🎙  Écoute — mot « {wakeword.name} » · seuil {detector.threshold:.0%} · "
          f"{detector.live.n_consecutive} fenêtres consécutives · Ctrl-C pour arrêter")
    n_trig = 0
    with sd.InputStream(samplerate=wakeword.sample_rate, blocksize=hop, channels=1,
                        dtype="float32", device=device, callback=callback):
        try:
            while True:
                chunk = q.get()
                history.append(chunk)
                history[:] = history[-int(3 * wakeword.sample_rate / hop):]
                event = detector.push(chunk)
                if on_event:
                    on_event(event)
                if event and event["triggered"]:
                    n_trig += 1
                    stamp = dt.datetime.now().strftime("%H:%M:%S")
                    print(f"🚨  [{stamp}] « {wakeword.name} » détecté "
                          f"(proba {event['proba']:.0%}, pic {event['peak']:.2f})")
                    if save_dir:
                        save_dir.mkdir(parents=True, exist_ok=True)
                        out = save_dir / f"trigger_{dt.datetime.now():%Y%m%d_%H%M%S_%f}.wav"
                        sf.write(out, np.concatenate(history), wakeword.sample_rate,
                                 subtype="PCM_16")
        except KeyboardInterrupt:
            print(f"\n🏁  Session terminée — {n_trig} déclenchement(s).")


def guided(model_path: Path, wakeword: WakewordConfig, save_dir: Path,
           device: int | None = None, threshold: float | None = None,
           record_s: float = 1.5) -> list[dict]:
    """Test guidé essai par essai : décompte, enregistrement, verdict du modèle,
    puis **vérité terrain donnée par l'humain**. Chaque clip est sauvegardé
    nommé TP/FN/FP/TN — c'est la source la plus précieuse du projet, parce que
    l'étiquette est posée juste après la prononciation, sans ambiguïté."""
    import sounddevice as sd
    import soundfile as sf

    detector = load_detector(model_path, wakeword, threshold)
    save_dir.mkdir(parents=True, exist_ok=True)
    sr = wakeword.sample_rate
    trials: list[dict] = []

    print(f"Test guidé — seuil {detector.threshold:.0%}. Entrée = essai, q = quitter.")
    while input("\n▶️  Entrée = essai suivant, q = quitter : ").strip().lower() != "q":
        for n in ("3…", "2…", "1…", "🎙  PARLE !"):
            print(f"   {n}", flush=True)
            sd.sleep(500)
        audio = sd.rec(int(record_s * sr), samplerate=sr, channels=1,
                       dtype="float32", device=device)
        sd.wait()
        audio = audio[:, 0]

        probas, peaks, _ = detector.window_probas(audio)
        p_max = float(probas.max())
        detected = p_max > detector.threshold
        print(f"   {'🚨 DÉTECTÉ' if detected else '— pas détecté'} "
              f"(proba max {p_max:.1%}, pic {np.abs(audio).max():.2f})")

        truth = input(f"   ❓ C'était bien « {wakeword.name} » ? [o/n] ").strip().lower() in ("o", "oui", "y")
        outcome = {(True, True): "TP", (True, False): "FN",
                   (False, True): "FP", (False, False): "TN"}[(truth, detected)]
        path = save_dir / f"guided_{dt.datetime.now():%H%M%S_%f}_{outcome}.wav"
        sf.write(path, audio, sr, subtype="PCM_16")
        print(f"   [{outcome}] 💾 {path.name}")
        trials.append({"outcome": outcome, "p_max": p_max, "clip": str(path)})

    if trials:
        counts = {k: sum(t["outcome"] == k for t in trials) for k in ("TP", "FN", "FP", "TN")}
        n_pos = counts["TP"] + counts["FN"]
        print(f"\nBilan : {len(trials)} essais — détection {counts['TP']}/{n_pos} · "
              f"fausses alarmes {counts['FP']}")
    return trials


__all__ = ["WakeWordDetector", "guided", "list_devices", "stream"]

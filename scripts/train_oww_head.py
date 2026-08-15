"""Piste B, exécutée en local : une tête openWakeWord entraînée sur NOS données.

Chaîne : nos clips (1 s, 16 kHz) → padding de tête à ~2 s (mot à la FIN, la
convention oWW) → LEUR front-end gelé (mel 32 → embeddings Google, ONNX) →
features [16, 96] → tête 64x3 entraînée ici (Keras, CPU, ADR-002) → export
ONNX dans open_wake_word_compare/ → verdict au banc via l'adaptateur.

Différence assumée avec le pipeline officiel : nos négatifs (~2,6 h de clips
de recette) au lieu de leurs ~31 000 h ACAV — le silence légendaire ne
viendra peut-être pas ; ce qu'on teste, c'est « leur représentation + nos
voix réelles » contre leur tout-synthétique (16-24 % de rappel au banc).

Usage : uv run python scripts/train_oww_head.py [--seeds 42 43 44]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coachvocal import runtime  # noqa: E402
from coachvocal.evaluation.oww_adapter import (  # noqa: E402
    EMB_STEP_FRAMES,
    EMB_WIN_FRAMES,
    HEAD_EMBEDDINGS,
    OWW_DIR,
)

SR = 16000
PAD_S = 2.02                     # assez d'audio pour 196 trames mel pleines
EXPORT_DIR = ROOT / "open_wake_word_compare"
CACHE = ROOT / "artifacts" / "cache" / "eloquence" / "oww_features"


def pad_souffle(audio: np.ndarray, pad_n: int, seed_key: str,
                amp: float = 0.001) -> np.ndarray:
    """Complète un clip à `pad_n` échantillons avec un SOUFFLE léger de micro,
    jamais du silence numérique parfait (règle du 2026-08-14 : un vrai micro
    n'en produit jamais, et l'extracteur gelé traite le zéro absolu comme un
    signal anormal — mesuré : les mêmes clips passent de 9 % à 34-38 % de
    déclenchement @0.8 selon le fond). Le souffle couvre TOUTE la fenêtre
    (les clips eux-mêmes peuvent contenir des zéros internes). Déterministe."""
    import zlib

    rng = np.random.default_rng(zlib.crc32(str(seed_key).encode()))
    a = audio[-pad_n:] if len(audio) >= pad_n else np.concatenate(
        [np.zeros(pad_n - len(audio), np.float32), audio.astype(np.float32)])
    return (a + amp * rng.standard_normal(pad_n).astype(np.float32)).astype(np.float32)


def vitesse(a: np.ndarray, f: float) -> np.ndarray:
    """Changement de vitesse par rééchantillonnage linéaire (f>1 = plus
    rapide). Le mot reste ENTIER : on change sa durée, pas son contenu."""
    idx = np.linspace(0, len(a) - 1, int(len(a) / f))
    return np.interp(idx, np.arange(len(a)), a).astype(np.float32)


def collect_files() -> tuple[list[Path], list[Path], list[Path]]:
    """Positifs réels, négatifs ADVERSARIAUX (cousins moi_, hard negatives du
    banc, essais guidés — le point faible mesuré du round 5, noyés à poids 1
    parmi ~6 900 négatifs), et le reste des négatifs de recette."""
    import csv

    pos = sorted((ROOT / "exports/oww_training_b/positifs_reels").glob("*.wav"))
    adv = sorted((ROOT / "exports/oww_training_b/negatifs_adversariaux").glob("*.wav"))
    neg = sorted((ROOT / "exports/oww_training_b/negatifs_parole_continue_fr").glob("*.wav"))
    # Sous-ensemble COUSINS du pool adversarial : les mots-voisins enregistrés
    # (moi_*) et la session « éloquente/éloquen » du 2026-08-14 (guided_204*).
    # Le scalpel de la nuit 2 (S2) : les durcir SANS toucher aux 54 hard
    # negatives du banc — la leçon de v31 (le marteau global casse les
    # vitesses).
    global COUSINS_SET
    COUSINS_SET = [f for f in adv
                   if f.name.startswith("moi_") or f.name.startswith("guided_204")]
    # + les négatifs génériques du TRAIN de la recette champion (bruit, GSC,
    # CV, silence, fragments) — notre océan à nous, même s'il est petit
    manifest = ROOT / "artifacts/runs/eloquence/v17_stack/manifest.csv"
    seen = {f.name for f in neg} | {f.name for f in adv}
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            p = Path(row["file"])
            if (row["split"] == "train" and row["label"] == "0"
                    and p.name not in seen and p.exists()):
                neg.append(p)
                seen.add(p.name)
    return pos, adv, neg


def embed_files(files: list[Path], mel_sess, emb_sess, tag: str) -> np.ndarray:
    """Chaque clip → features [16, 96] (mot padé en FIN de fenêtre ~2 s).
    Cache .npy : l'extraction ne se refait pas d'un run à l'autre."""
    import soundfile as sf

    CACHE.mkdir(parents=True, exist_ok=True)
    # suffixe `s1` : depuis le 2026-08-14, complétion au SOUFFLE léger (plus
    # jamais de silence numérique) — nouveau cache, l'ancien reste intact.
    cache_file = CACHE / f"{tag}_s1_{len(files)}.npy"
    if cache_file.exists():
        return np.load(cache_file)

    pad_n = int(PAD_S * SR)
    out = np.zeros((len(files), HEAD_EMBEDDINGS, 96), np.float32)
    batch_audio, batch_idx = [], []

    def flush():
        if not batch_audio:
            return
        for k, audio in enumerate(batch_audio):
            mel = mel_sess.run(None, {"input": audio[np.newaxis, :]})[0]
            frames = np.squeeze(mel) / 10.0 + 2.0
            wins = np.stack([frames[j:j + EMB_WIN_FRAMES]
                             for j in range(0, len(frames) - EMB_WIN_FRAMES + 1,
                                            EMB_STEP_FRAMES)])
            embs = emb_sess.run(None, {"input_1": wins[:, :, :, np.newaxis]})[0]
            out[batch_idx[k]] = embs.reshape(len(embs), 96)[-HEAD_EMBEDDINGS:]
        batch_audio.clear()
        batch_idx.clear()

    for i, f in enumerate(files):
        try:
            audio, sr = sf.read(f, dtype="float32")
        except Exception:
            continue
        if sr != SR:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = pad_souffle(audio, pad_n, f.name)    # mot à la FIN, fond souffle
        batch_audio.append(audio.astype(np.float32))
        batch_idx.append(i)
        if len(batch_audio) >= 64:
            flush()
        if i % 1000 == 0:
            print(f"    {tag} : {i}/{len(files)}", flush=True)
    flush()
    np.save(cache_file, out)
    return out


def acav_windows(path: str, stride: int = 8) -> np.ndarray:
    """Le fichier oWW est un FLUX d'embeddings (N, 96), un par 80 ms : on le
    découpe en fenêtres [16, 96] — les négatifs « océan » qui sculptent la
    frontière (l'ingrédient n°2 de leur silence)."""
    stream = np.load(path, mmap_mode="r")
    if stream.ndim == 3:                         # déjà fenêtré (N, 16, 96)
        return np.asarray(stream[::stride], dtype=np.float32)
    idx = np.arange(0, len(stream) - HEAD_EMBEDDINGS, stride)
    out = np.empty((len(idx), HEAD_EMBEDDINGS, 96), np.float32)
    for k, i in enumerate(idx):
        out[k] = stream[i:i + HEAD_EMBEDDINGS]
    return out



def _noise_bank(limit: int = 200) -> list[np.ndarray]:
    """Bruits réels (MUSAN) pour remplacer le silence de padding : au micro il
    y a toujours un fond de pièce, jamais des zéros numériques."""
    import soundfile as sf

    bank = []
    for wav in sorted((ROOT / "data/external/musan").rglob("*.wav"))[:limit]:
        try:
            a, sr = sf.read(wav, dtype="float32")
        except Exception:
            continue
        if sr != SR or len(a) < SR:
            continue
        if a.ndim > 1:
            a = a.mean(axis=1)
        bank.append(a.astype(np.float32))
    return bank


def collect_context_positives(noise_pad: bool = False) -> list[np.ndarray]:
    """Positifs à CONTEXTE RÉEL (remède au round 2) : fenêtres de PAD_S s
    découpées dans les segments du corpus, fin du mot au bord droit, mot
    localisé par corrélation croisée avec le clip propre (la méthode
    auto-vérifiante de word_clips_recut). Les clips sans segment source
    (moi/guided) restent padés de silence — minoritaires."""
    import re

    import soundfile as sf

    from coachvocal.data import corpus as corpus_mod

    pad_n = int(PAD_S * SR)
    out: list[np.ndarray] = []
    bank = _noise_bank() if noise_pad else []
    seg_index = {}
    for wav in (corpus_mod.CORPUS / "audio").glob("*.wav"):
        m = re.match(r"(.+)_(\d+)-(\d+)$", wav.stem)
        if m:
            seg_index.setdefault(m.group(1), []).append(
                (int(m.group(2)), int(m.group(3)), wav))

    files = sorted((ROOT / "exports/oww_training_b/positifs_reels").glob("*.wav"))
    n_ctx = n_pad = 0
    for f in files:
        clip, csr = sf.read(f, dtype="float32")
        if csr != SR:
            continue
        if clip.ndim > 1:
            clip = clip.mean(axis=1)
        m = re.match(rf"yt_({corpus_mod.VIDEO_ID})_.*_(\d+(?:\.\d+)?)s", f.stem)
        placed = False
        if m:
            vid, t0 = m.group(1), float(m.group(2))
            for s0, s1, seg_wav in seg_index.get(vid, []):
                if not (s0 <= t0 <= s1):
                    continue
                audio, sr2 = sf.read(seg_wav, dtype="float32")
                if sr2 != SR:
                    break
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                expected = int((t0 - s0) * SR)
                lo = max(0, expected - 2 * SR)
                hi = min(len(audio), expected + 2 * SR + SR)
                template = clip[: SR // 2]
                zone = audio[lo:hi]
                if len(zone) < len(template) + 1:
                    break
                corr = np.correlate(zone, template, mode="valid")
                energy = np.convolve(zone ** 2, np.ones(len(template)), "valid")
                ncc = corr / (np.sqrt(energy * np.sum(template ** 2)) + 1e-9)
                if float(ncc.max()) < 0.6:
                    break
                w_start = lo + int(np.argmax(ncc))
                nz = (np.abs(clip) > 1e-5).nonzero()[0]
                w_len = int(nz[-1] + 1) if len(nz) else len(clip)
                end = min(len(audio), w_start + w_len + int(0.1 * SR))
                start = end - pad_n
                if start < 0:
                    break
                out.append(audio[start:end].astype(np.float32))
                n_ctx += 1
                placed = True
                break
        if not placed:                        # repli : silence ou bruit en tête
            a = clip[-pad_n:] if len(clip) >= pad_n else np.pad(
                clip, (pad_n - len(clip), 0))
            a = a.astype(np.float32)
            if noise_pad and bank:
                # Mix d'un fond réel sur TOUTE la fenêtre (le bruit d'une pièce
                # ne s'arrête pas quand on parle), SNR 12-25 dB, déterministe.
                import zlib
                rng2 = np.random.default_rng(zlib.crc32(f.name.encode()))
                noise = bank[rng2.integers(len(bank))]
                i0 = rng2.integers(max(1, len(noise) - pad_n))
                noise = np.pad(noise[i0:i0 + pad_n], (0, max(0, pad_n - (len(noise) - i0))))
                rms_c = float(np.sqrt((clip ** 2).mean()) + 1e-9)
                rms_n = float(np.sqrt((noise ** 2).mean()) + 1e-9)
                snr = rng2.uniform(12.0, 25.0)
                a = a + noise * (rms_c / rms_n) * 10 ** (-snr / 20)
            out.append(a)
            n_pad += 1
    print(f"    positifs contexte réel : {n_ctx} · repli padding : {n_pad}"
          + (" (fond MUSAN)" if noise_pad else " (silence)"))
    return out


def collect_prefix_negatives(fracs=(0.60, 0.75, 0.85),
                             speeds=(1.0,)) -> list[np.ndarray]:
    """Préfixes du mot (fin ABSENTE), collés au bord droit de la fenêtre —
    le rôle des fragments du CNN, transposé à la convention de la tête.

    Motif mesuré (2026-08-14, clips moi_) : la tête déclenche dès 80-90 % du
    mot (8/47 à 80 %, 18/47 à 90 %) — au micro elle tire sur « éloquen » et
    « éloquente » passe. Nos pools de fragments s'arrêtent à 70 %, or à 70 %
    la fuite est nulle : la zone à enseigner est 75-90 %. Plafond 85 % — plus
    haut contredirait les positifs (le mot entier doit rester positif).
    Sources : les positifs d'ENTRAÎNEMENT uniquement (pas de fuite val/test)."""
    import soundfile as sf

    from coachvocal.data.sources.fragments import word_span

    pad_n = int(PAD_S * SR)
    out: list[np.ndarray] = []
    for f in sorted((ROOT / "exports/oww_training_b/positifs_reels").glob("*.wav")):
        audio, sr = sf.read(f, dtype="float32")
        if sr != SR:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        for sp in speeds:
            aa = vitesse(audio, sp) if sp != 1.0 else audio
            w0, w1 = word_span(aa, SR)
            for frac in fracs:
                k = int(frac * (w1 - w0))
                out.append(pad_souffle(aa[w0:w0 + k], pad_n, f"{f.name}|{frac}|{sp}"))
    print(f"    négatifs-préfixes : {len(out)} fenêtres (fracs {list(fracs)}, "
          f"vitesses {list(speeds)}, fond souffle)")
    return out


def collect_suffix_negatives(fracs=(0.65, 0.80, 0.90),
                             speeds=(1.0, 1.05, 1.15)) -> list[np.ndarray]:
    """Suffixes du mot (DÉBUT absent) — le miroir des préfixes.

    Fuite entendue au micro (2026-08-15) : « loquence » déclenchait — rien
    n'exigeait le début du mot. Fractions CONSERVÉES depuis la fin : 0.90 =
    il ne manque que le « é » (le cas entendu), 0.65 ≈ « oquence ».
    Multi-vitesses pour ne pas réintroduire l'indice de durée (leçon v26/v27).
    Positifs d'entraînement uniquement."""
    import soundfile as sf

    from coachvocal.data.sources.fragments import word_span

    pad_n = int(PAD_S * SR)
    out: list[np.ndarray] = []
    for f in sorted((ROOT / "exports/oww_training_b/positifs_reels").glob("*.wav")):
        audio, sr = sf.read(f, dtype="float32")
        if sr != SR:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        for sp in speeds:
            aa = vitesse(audio, sp) if sp != 1.0 else audio
            w0, w1 = word_span(aa, SR)
            for frac in fracs:
                k = int(frac * (w1 - w0))
                out.append(pad_souffle(aa[w1 - k:w1], pad_n, f"suf|{f.name}|{frac}|{sp}"))
    print(f"    négatifs-suffixes : {len(out)} fenêtres (fracs {list(fracs)}, "
          f"vitesses {list(speeds)}, fond souffle)")
    return out


def embed_arrays(arrays, mel_sess, emb_sess, tag: str) -> np.ndarray:
    """Comme embed_files mais depuis des tampons audio déjà chargés."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"{tag}_{len(arrays)}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    out = np.zeros((len(arrays), HEAD_EMBEDDINGS, 96), np.float32)
    for i, audio in enumerate(arrays):
        mel = mel_sess.run(None, {"input": audio[np.newaxis, :]})[0]
        frames = np.squeeze(mel) / 10.0 + 2.0
        wins = np.stack([frames[j:j + EMB_WIN_FRAMES]
                         for j in range(0, len(frames) - EMB_WIN_FRAMES + 1,
                                        EMB_STEP_FRAMES)])
        embs = emb_sess.run(None, {"input_1": wins[:, :, :, np.newaxis]})[0]
        out[i] = embs.reshape(len(embs), 96)[-HEAD_EMBEDDINGS:]
        if i % 500 == 0:
            print(f"    {tag} : {i}/{len(arrays)}", flush=True)
    np.save(cache_file, out)
    return out



def collect_french_negatives(max_windows: int = 12000) -> list[np.ndarray]:
    """Négatifs FRANÇAIS à l'échelle (round 5) : fenêtres de PAD_S s de parole
    continue, hors zones du mot (±2 s, spans dédupliqués), depuis les segments
    du corpus + les réunions SUMM-RE d'entraînement. C'est le contrepoids que
    l'océan ACAV (multilingue) n'a pas."""
    import re

    import soundfile as sf

    from coachvocal.data import corpus as corpus_mod

    pad_n = int(PAD_S * SR)
    stride = SR                                  # une fenêtre par seconde
    spans = corpus_mod.db_word_spans("eloquence")
    out: list[np.ndarray] = []

    sources: list[tuple] = []
    for wav in sorted((corpus_mod.CORPUS / "audio").glob("*.wav")):
        m = re.match(r"(.+)_(\d+)-(\d+)$", wav.stem)
        if m:
            sources.append((wav, m.group(1), int(m.group(2))))
    for wav in sorted((ROOT / "data/external/summre_train").glob("*.wav")):
        sources.append((wav, None, 0))           # certifié sans le mot

    rng = np.random.default_rng(123)
    rng.shuffle(sources)
    for wav, vid, s0 in sources:
        if len(out) >= max_windows:
            break
        try:
            audio, sr = sf.read(wav, dtype="float32")
        except Exception:
            continue
        if sr != SR:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        forbidden = [(t0 - s0, t1 - s0) for t0, t1, _ in spans.get(vid, [])] if vid else []
        for start in range(0, max(len(audio) - pad_n, 0), stride):
            if len(out) >= max_windows:
                break
            t0, t1 = start / SR, (start + pad_n) / SR
            if any(a - 2.0 <= t1 and t0 <= b + 2.0 for a, b in forbidden):
                continue
            win = audio[start:start + pad_n]
            if np.abs(win).max() < 0.02:
                continue
            out.append(win.astype(np.float32))
    print(f"    négatifs français continus : {len(out)} fenêtres "
          f"({len(out) * 1 / 3600:.1f} h)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--acav", type=str, default=None,
                    help="features négatives oWW (.npy, flux N×96)")
    ap.add_argument("--acav-stride", type=int, default=8)
    ap.add_argument("--tag", type=str, default="nosdonnees")
    ap.add_argument("--context", action="store_true",
                    help="positifs à contexte réel (fenêtres 2 s du corpus)")
    ap.add_argument("--augment-pos", action="store_true",
                    help="variantes de vitesse des positifs — les MÊMES facteurs "
                         "que l'augmentation du CNN (0.85-1.15, la leçon v01 : "
                         "le modèle avait appris un débit, pas un mot)")
    ap.add_argument("--noise-pad", action="store_true",
                    help="les positifs sans segment source (moi/guided) sont "
                         "padés d'un fond MUSAN réel au lieu de silence")
    ap.add_argument("--french-neg", type=int, default=0,
                    help="nb de fenêtres négatives françaises (poids ×20)")
    ap.add_argument("--prefix-neg-weight", type=float, default=0.0,
                    help="poids des négatifs-préfixes (60/75/85 % du mot, fin "
                         "absente) — enseigne « attends la fin du mot » ; 0 = off")
    ap.add_argument("--effective-class-weight", action="store_true",
                    help="équilibre de classes par masses effectives — testé et "
                         "REJETÉ (v30 : 146.8 FA/h, sur-pondération des positifs "
                         "x21) ; conservé pour reproduire l'expérience")
    ap.add_argument("--suffix-neg-weight", type=float, default=0.0,
                    help="poids des négatifs-suffixes (65/80/90 % du mot, DÉBUT "
                         "absent, multi-vitesses) — enseigne « le é compte » ; 0 = off")
    ap.add_argument("--prefix-fast", action="store_true",
                    help="préfixes découpés AUSSI dans des variantes accélérées "
                         "(×1.05/×1.15) : la durée cesse d'être un indice, seule "
                         "la fin manquante sépare préfixe et mot rapide (v27)")
    ap.add_argument("--cousin-speeds", action="store_true",
                    help="les cousins (moi_* + session éloquente) déclinés en "
                         "vitesses ×0.85/0.95/1.05/1.15 — PLUS de données au "
                         "lieu de plus de poids (leçon v32 : au-delà de ~50, "
                         "les poids géants déstabilisent)")
    ap.add_argument("--cousin-weight", type=float, default=0.0,
                    help="poids DÉDIÉ au sous-ensemble cousins du pool "
                         "adversarial (moi_* + session éloquente) ; 0 = ils "
                         "gardent le poids adversarial commun")
    ap.add_argument("--adv-weight", type=float, default=1.0,
                    help="poids des 122 négatifs adversariaux (cousins moi_, "
                         "hard negatives du banc, guidés) — le point faible du round 5")
    args = ap.parse_args()

    import onnxruntime as ort
    mel = ort.InferenceSession(str(OWW_DIR / "melspectrogram.onnx"),
                               providers=["CPUExecutionProvider"])
    emb = ort.InferenceSession(str(OWW_DIR / "embedding_model.onnx"),
                               providers=["CPUExecutionProvider"])

    pos_files, adv_files, neg_files = collect_files()
    print(f"📦  {len(pos_files)} positifs réels · {len(adv_files)} adversariaux "
          f"(poids ×{args.adv_weight:g}) · {len(neg_files)} négatifs (recette)")
    if args.context:
        arrs = collect_context_positives(args.noise_pad)
        tag_pos = "pos_ctx_noise" if args.noise_pad else "pos_ctx"
        if args.augment_pos:
            # Vitesse : rééchantillonnage linéaire de TOUTE la fenêtre (le mot
            # reste au bord droit), même principe que l'augmentation du CNN.
            pad_n = int(PAD_S * SR)
            aug = [pad_souffle(vitesse(a, f), pad_n, f"aug|{i}|{f}")
                   for i, a in enumerate(arrs) for f in (0.85, 0.95, 1.05, 1.15)]
            print(f"    augmentation vitesse : +{len(aug)} variantes "
                  "(×0.85/0.95/1.05/1.15 — mêmes facteurs que le CNN)")
            arrs = arrs + aug
            tag_pos += "_augcnn"
        X_pos = embed_arrays(arrs, mel, emb, tag_pos)
    else:
        X_pos = embed_files(pos_files, mel, emb, "pos")
    X_adv = embed_files(adv_files, mel, emb, "adv")
    X_neg = embed_files(neg_files, mel, emb, "negrest")
    # Poids adversarial : commun, sauf le sous-ensemble cousins si un poids
    # dédié est demandé (S2, nuit 2).
    w_adv = np.full(len(adv_files), args.adv_weight)
    if args.cousin_weight:
        noms_cousins = {f.name for f in COUSINS_SET}
        for i, f in enumerate(adv_files):
            if f.name in noms_cousins:
                w_adv[i] = args.cousin_weight
        print(f"    cousins dédiés : {int((w_adv == args.cousin_weight).sum())} "
              f"clips à ×{args.cousin_weight:g} (le reste du pool à ×{args.adv_weight:g})")
    parts_X = [X_pos, X_adv, X_neg]
    parts_y = [np.ones(len(X_pos)), np.zeros(len(X_adv)), np.zeros(len(X_neg))]
    parts_w = [np.ones(len(X_pos)), w_adv, np.ones(len(X_neg))]
    if args.french_neg:
        X_fr = embed_arrays(collect_french_negatives(args.french_neg), mel, emb,
                            "frneg")
        parts_X.append(X_fr)
        parts_y.append(np.zeros(len(X_fr)))
        parts_w.append(np.full(len(X_fr), 20.0))   # contrepoids face à l'océan
    if args.prefix_neg_weight:
        speeds = (1.0, 1.05, 1.15) if args.prefix_fast else (1.0,)
        X_pre = embed_arrays(collect_prefix_negatives(speeds=speeds), mel, emb,
                             "prefixneg_s2" if args.prefix_fast else "prefixneg_s1")
        parts_X.append(X_pre)
        parts_y.append(np.zeros(len(X_pre)))
        parts_w.append(np.full(len(X_pre), args.prefix_neg_weight))
    if args.cousin_speeds:
        import soundfile as _sf
        pad_n = int(PAD_S * SR)
        arrs = []
        for f in COUSINS_SET:
            a, sr_c = _sf.read(f, dtype="float32")
            if sr_c != SR:
                continue
            if a.ndim > 1:
                a = a.mean(axis=1)
            for sp in (0.85, 0.95, 1.05, 1.15):
                arrs.append(pad_souffle(vitesse(a.astype(np.float32), sp),
                                        pad_n, f"cousp|{f.name}|{sp}"))
        X_cs = embed_arrays(arrs, mel, emb, "cousinspeeds")
        parts_X.append(X_cs)
        parts_y.append(np.zeros(len(X_cs)))
        parts_w.append(np.full(len(X_cs), args.adv_weight))
        print(f"    cousins multi-vitesses : {len(X_cs)} fenêtres à ×{args.adv_weight:g}")
    if args.suffix_neg_weight:
        X_suf = embed_arrays(collect_suffix_negatives(), mel, emb, "suffixneg_s1")
        parts_X.append(X_suf)
        parts_y.append(np.zeros(len(X_suf)))
        parts_w.append(np.full(len(X_suf), args.suffix_neg_weight))
    if args.acav:
        X_acav = acav_windows(args.acav, args.acav_stride)
        print(f"    + océan ACAV : {X_acav.shape} "
              f"({len(X_acav) * 0.08 * args.acav_stride / 3600:.1f} h de flux)")
        parts_X.append(X_acav)
        parts_y.append(np.zeros(len(X_acav)))
        parts_w.append(np.ones(len(X_acav)))
    X = np.concatenate(parts_X)
    y = np.concatenate(parts_y).astype(np.float32)
    w = np.concatenate(parts_w).astype(np.float32)
    print(f"    features : {X.shape}")

    runtime.configure(use_gpu=False)
    import tensorflow as tf
    from tensorflow import keras

    EXPORT_DIR.mkdir(exist_ok=True)
    for seed in args.seeds:
        keras.utils.set_random_seed(seed)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        n_val = int(0.1 * len(X))
        va, tr = idx[:n_val], idx[n_val:]

        head = keras.Sequential([
            keras.layers.Input(shape=(HEAD_EMBEDDINGS, 96)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        head.compile(optimizer=keras.optimizers.Adam(1e-3),
                     loss="binary_crossentropy",
                     metrics=[keras.metrics.AUC(name="auc")])
        # Équilibre de classes par comptes bruts (le calcul historique).
        # L'alternative par masses effectives a été testée et REJETÉE (v30 :
        # elle sur-pondérait les positifs x21 -> 146.8 FA/h, JOURNAL
        # 2026-08-15) — conservée en opt-in pour reproduire l'expérience.
        if args.effective_class_weight:
            masse_pos = float(w[tr][y[tr] > 0].sum())
            masse_neg = float(w[tr][y[tr] == 0].sum())
            total = masse_pos + masse_neg
            cw = {0: total / (2 * masse_neg), 1: total / (2 * masse_pos)}
        else:
            n_pos, n_neg = y[tr].sum(), len(tr) - y[tr].sum()
            cw = {0: len(tr) / (2 * n_neg), 1: len(tr) / (2 * n_pos)}
        head.fit(X[tr], y[tr] , validation_data=(X[va], y[va]),
                 sample_weight=w[tr] * np.where(y[tr] > 0, cw[1], cw[0]),
                 epochs=args.epochs, batch_size=128, verbose=2,
                 callbacks=[keras.callbacks.EarlyStopping(
                     monitor="val_auc", mode="max", patience=6,
                     restore_best_weights=True)])

        out = EXPORT_DIR / f"eloquence_{args.tag}_64x3_seed{seed}.onnx"
        import tf2onnx

        # tf2onnx ne connaît pas les modèles Keras 3 : on exporte via une
        # tf.function qui enveloppe l'appel du modèle.
        spec = (tf.TensorSpec((1, HEAD_EMBEDDINGS, 96), tf.float32, name="input"),)

        @tf.function(input_signature=spec)
        def infer(x, _head=head):                 # lie la tête de CETTE itération
            return {"proba": _head(x, training=False)}

        tf2onnx.convert.from_function(infer, input_signature=spec, opset=17,
                                      output_path=str(out))
        # contrôle : l'ONNX doit rendre les mêmes probas que Keras
        import onnxruntime as _ort
        sess = _ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        probe = X[va[:8]].astype(np.float32)
        p_onnx = np.concatenate([sess.run(None, {"input": probe[k:k+1]})[0].ravel()
                                 for k in range(len(probe))])
        p_keras = head.predict(probe, verbose=0).ravel()
        assert np.allclose(p_onnx, p_keras, atol=1e-4), "export ONNX ≠ Keras !"
        print(f"💾  {out.name}  (export vérifié, écart max "
              f"{np.abs(p_onnx - p_keras).max():.2e})")

    print("\nBanc : uv run coachvocal bench --run open_wake_word_compare/"
          "eloquence_nosdonnees_64x3_seed42.onnx --minutes 60 --thresholds 0.05,0.2,0.5,0.8")


if __name__ == "__main__":
    main()

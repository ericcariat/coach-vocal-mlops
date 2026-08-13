# Données — provenance, transformations, licences

Règle : **aucun dossier de données n'entre dans le projet sans une ligne ici.**
D'où il vient, comment il a été transformé, sous quelle licence. Un dataset dont
on ne sait plus l'origine est un dataset qu'on ne peut pas défendre.

## Emplacement et versionnement

```
data/
  external/            corpus téléchargés, immuables, partagés entre mots-clés
    gsc/               Google Speech Commands v2
    musan/             MUSAN
    common_voice_fr/   Common Voice FR
    common_voice_en/   Common Voice EN
    piper_voices/      voix TTS Piper (.onnx)
    youtube_corpus/    corpus YouTube COPIÉ depuis le scraper (audio + subs +
                       discovery.db + curation.db) — cf. ADR-006
  wakewords/<mot>/
    raw/               brut, jamais modifié
    clean/             dérivé par script (positives/, negatives_proches/)
    generated/         régénérable (pools TTS…)
    guided_clips/      sessions de test guidé, étiquetées à l'oreille
    hard_negatives/    fausses alarmes réelles confirmées
    selections/        listes CSV versionnées (on versionne les listes, pas l'audio)
    splits.csv         split figé, par groupe
```

Les octets sont hors git, suivis par **DVC** (`make dvc-init`) : git ne reçoit
qu'un fichier pointeur de quelques lignes. Ce n'est pas une sauvegarde — c'est ce
qui permet, en revenant sur le commit d'un run, de retrouver **exactement** le
jeu de données qui l'a produit.

---

## Sources

| Source | Provenance | Transformation | Licence |
|---|---|---|---|
| `raw/youtube` | Extraits de vidéos YouTube contenant le mot, récupérés par le scraper (yt-dlp + WhisperX) | Découpe autour du mot aligné, 16 kHz mono | Usage recherche/formation |
| `raw/moi_positif` | Mes enregistrements au micro | 16 kHz mono, 1 s | — |
| `clean/negatives_proches` | Mots phonétiquement voisins (« élégance », « éloquent »…) | Idem | — |
| `guided_clips/` | Sessions `coachvocal live guided` | Aucune (brut), recadrage 1 s à l'usage | — |
| **GSC v2** | <https://www.tensorflow.org/datasets/catalog/speech_commands> | Splits officiels réutilisés, chiffres exclus | CC BY 4.0 |
| **MUSAN** | <https://www.openslr.org/17/> | Crops d'1 s, 3 gains (1.0 / 0.1 / 0.02), split par position | CC BY 4.0 / domaine public |
| **Common Voice FR/EN** | <https://commonvoice.mozilla.org/datasets> | mp3 → wav 16 kHz mono, 1 s à partir de 0,5 s ; phrases contenant le mot-clé **exclues** ; split par locuteur | CC0 |
| **Voix Piper** | <https://huggingface.co/rhasspy/piper-voices> — `fr_FR-siwis-medium` (CC BY 4.0), `fr_FR-upmc-medium` (Apache 2.0) | Synthèse, 22 050 → 16 kHz, rognage des silences, recadrage 1 s, normalisation du pic | Voir par voix |
| `youtube_corpus` | Copié depuis `scraper-audio` le 2026-07-28 (1,7 Go : 1278 segments, 613 VTT, 2006 occurrences alignées) | Aucune — lecture seule | Usage recherche |
| `bench_extra` | **SUMM-RE** (<https://huggingface.co/datasets/linagora/SUMM-RE>), extrait le 2026-08-13 : 3 pistes micro de 3 réunions distinctes (`004c_PAPH_013`, `006b_EADH_017`, `015b_EBDD_051`), 64 min | 48 kHz → 16 kHz mono PCM16 (ffmpeg) ; vérité terrain = alignements mot à mot du dataset (aucune occurrence du mot-clé) → `ground_truth.json` | CC BY-SA 4.0 |

### Générés (reproductibles depuis la seed)

| Pool | Producteur | Note |
|---|---|---|
| `silence` | `sources/silence.py` | Bruit blanc 1e-5 → 3e-4. Indispensable : sans lui, le z-score amplifie le bruit de plancher et le détecteur part en vrille dans une pièce vide |
| `fragments` | `sources/fragments.py` | Début/fin de mot en négatif — évite le déclenchement sur « élo… » |
| `tts_positives` | `data/tts.py` | 3 voix × 5 vitesses × 3 variabilités × 50 |
| `speech_negatives` | `sources/speech_negatives.py` | Fenêtres de parole continue, hors zones du mot-clé, split hérité de la vidéo |

---

## Précautions systématiques

1. **Exclusion du mot-clé des corpus de négatifs.** Une phrase Common Voice
   contenant « éloquence » apprendrait littéralement au modèle que le mot cible
   est un négatif.
2. **Split par groupe.** Vidéo ou locuteur, jamais par clip : deux extraits de la
   même source partagent la voix, le micro et le bruit de fond.
3. **Le synthétique ne va qu'en train.** Évaluer sur du Piper mesurerait la
   capacité à reconnaître Piper.
4. **Corpus du banc disjoint du train.** Les vidéos des splits train/val sont
   interdites au banc streaming, sinon on mesurerait de la mémorisation.

---

## Récupérer les corpus externes

```bash
# Google Speech Commands v2
curl -O http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir -p data/external/gsc/raw && tar -xzf speech_commands_v0.02.tar.gz -C data/external/gsc/raw

# MUSAN
curl -O https://www.openslr.org/resources/17/musan.tar.gz
tar -xzf musan.tar.gz -C data/external/

# Voix Piper
cd data/external/piper_voices
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json
```

Common Voice demande une acceptation de licence : télécharger l'archive depuis le
site, la placer dans `data/external/common_voice_fr/` avec son `train.tsv`.

---

## Pistes non retenues pour l'instant

| Dataset | Intérêt | Pourquoi pas encore |
|---|---|---|
| DEMAND | Bruit de fond stationnaire réaliste | MUSAN couvre déjà le besoin |
| ESC-50 / UrbanSound8K | Événements ponctuels, bruit urbain | Utile si usage extérieur |
| FSD50K / AudioSet | Très riches | Lourds, qualité inégale |
| Hey-Snips | Référence du domaine | Licence sur demande, mot différent |

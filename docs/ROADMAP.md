# Feuille de route — détection « éloquence »

Rédigée le 2026-08-13 à partir de : l'audit chiffré des positifs YouTube,
les quatre études (`ETUDE_OPENWAKEWORD`, `ETUDE_MICRO_WAKE_WORD`,
`ETUDE_VIOLAWAKE`, `ETUDE_LIVEKIT_WAKEWORD_DICTA_VEILLE_2026`), et l'état des
runs (v03 champion ; v04 parole continue recalé ; série Metal v05-v08 close).

Règle de lecture : une priorité = un problème mesuré, pas une envie. Chaque
item devient une expérience avec critère écrit avant le run.

---

## Ce que l'audit des positifs a mesuré (2026-08-13)

Sur les 1 882 clips `yt_positif` (échantillon 200 pour l'énergie) :

| Constat | Chiffre |
|---|---|
| Clips normalisés à exactement 1.00 s | 100 % |
| Fins « chargées » (énergie dans la dernière tranche de 100 ms) | ~17-25 % |
| Fins en padding de zéros ≥ 100 ms (propres) | 62 % |
| **Débuts « chargés » (aucune marge avant le mot)** | **93,5 %** |
| **Formes élidées « l'éloquence » / « d'éloquence »** | **85 %** (48,7 + 36,3) |
| « éloquence » nu | 14,6 % |
| Nom propre « Dauphine-Éloquence » (faux positifs d'étiquetage) | 5 clips |
| Lot curé à la main dans `curation.db` (crop_start/end), **non utilisé** | 551 clips |

Le doute initial (« les fins débordent ») est partiellement fondé mais
minoritaire. Les deux vrais sujets sont ailleurs : le **placement du mot dans
la fenêtre** (attaque à l'échantillon 0, padding en queue — l'inverse de ce que
recommandent les trois pipelines de référence), et la **prépondérance des
formes élidées**.

Sur les élisions, décision fonctionnelle à assumer : en français réel,
« l'éloquence »/« d'éloquence » SONT le mot (85 % des occurrences sauvages).
On ne les retire pas — le coach doit se réveiller dessus, et le banc les
compte. En revanche on **mesure séparément** le rappel par forme au banc,
pour savoir si le modèle est plus faible sur une forme.

## Les convergences des quatre études

1. **Placement du mot : fin du mot près de la fin de la fenêtre, jitter
   ~200 ms** (openWakeWord, microWakeWord, LiveKit — unanimes). Un modèle
   streaming doit scorer au moment où le mot vient de se terminer. Nos clips
   font l'inverse.
2. **Augmentations absentes de notre pipeline** : réverbération par RIR,
   bruit à plusieurs SNR explicites, EQ paramétrique, pitch ±2-3 demi-tons,
   distorsion, bruit coloré. Nous n'avons que jitter/vitesse/bruit simple.
3. **Sélection des poids sur l'objectif produit** : atteindre une cible de
   FA/h sur un flux ambiant de validation, puis maximiser le rappel — pas la
   val_loss minimale (microWakeWord, LiveKit). Suppose une `val_ambient`
   (heures de flux négatif) distincte du banc de test.
4. **Dosage = pondération, pas duplication** : `sampling_weight` /
   `penalty_weight` par source (microWakeWord), pondération progressive des
   négatifs pilotée par l'objectif FA/h (LiveKit), et la dose traitée comme
   variable expérimentale (openWakeWord — exactement notre sweep tts500).
5. **Hard negatives français à la main** (tous) : les générateurs automatiques
   sont anglophones. Liste ViolaWake : élégance, éloquent(e), éloignement,
   conséquence, séquence, « quelle éloquence », « manque d'éloquence ».
6. **Console de collecte guidée** (ViolaWake) : 40-60 prises par campagne,
   structurées distance × voix × débit, contrôles immédiats (RMS, pic,
   saturation, SNR, réécoute), une session = un groupe indivisible du split.
7. **Comparer des architectures sur NOTRE banc uniquement**, scores bruts dans
   NOTRE machine à états (tous les docs le martèlent).

---

## Priorités

### P0 — Qualité des données et du banc (avant tout nouveau run « recette »)

- [x] **Étendre le corpus du banc à ~60 min maximum** — fait le 2026-08-13 :
  54,9 min (banc YouTube + 42,4 min SUMM-RE), cf. CHANGELOG « Banc étendu ». (décision : on ne va pas
  au-delà d'une heure). Prérequis statistique de tout le reste : à
  23 occurrences / ~15 FA, un écart de 2 occurrences ou 5 FA est du bruit (vu
  sur la série Metal). Débloque aussi les hard negatives (on peut sacrifier
  des vidéos au train sans tuer le banc).

  **Sources candidates, examinées le 2026-08-13** (voir `ETUDE_OPENWAKEWORD.md`
  §8, `ETUDE_MICRO_WAKE_WORD.md` §7.2) :
  - **YouTube non thématique via le scraper existant** — la voie principale.
    Parole française ordinaire : dialogues, bureau, café, rue, magasins,
    interviews. C'est le domaine réel du produit (le coach écoute du français),
    et le pipeline vérité-terrain WhisperX existe déjà. Traçabilité DATA.md +
    DVC comme le reste.
  - **Amazon Dinner Party Corpus (DiPCo, ~5,5 h)** — utilisé par openWakeWord
    pour évaluer les FA de ses modèles officiels ; parole lointaine + musique
    + bruit, très adversarial. Anglais : utile comme **sous-ensemble de
    stress** (10-15 min), pas comme cœur du banc — les FA sur de l'anglais ne
    prédisent pas les FA sur du français. Vérifier la licence avant usage.
  - **Dataset HF `kahrendt/microwakeword` (~9,7 Go)** — ÉCARTÉ : ce sont des
    features précalculées du front-end micro_speech, pas de l'audio brut
    (incompatible avec la règle « un seul front-end acoustique »), et
    licence CC BY-NC 4.0.
  - Rappel anti-fuite : MUSAN, Common Voice, GSC sont interdits au banc (déjà
    dans l'entraînement).

  **Corpus français de conversation continue** (sélection du 2026-08-13) :

  | Source | Description et intérêt | Accès |
  |---|---|---|
  | **SUMM-RE — choix recommandé** | ~95 h de réunions spontanées en français, 3-4 participants. Une session ≈ trois discussions de 20 min → une heure continue de type « bureau » facilement. Licence CC BY-SA 4.0. | [Hugging Face](https://huggingface.co/datasets/linagora/SUMM-RE) · [Article](https://aclanthology.org/2024.jeptalnrecital-taln.35/) |
  | **CID — Corpus of Interactional Data** | Huit conversations spontanées d'1 h entre deux francophones. Excellente parole continue, mais salle anéchoïque : à compléter avec du bruit de café/bureau. | [ATALA](https://www.atala.org/node/810) · [Accès](https://sppas.org/bigi/Doc/2015-SPPAS-Tutorial-HongKong/SPPAS-tutorial_02_introduction.html) |
  | **ESLO** | Grand corpus de français parlé : repas, réunions, commerces, quotidien. Le plus proche du contexte « repas/café », sélection moins directe. | [Présentation](https://segcor.cnrs.fr/lll-eslo/) · [ESLO-FLEU HF](https://huggingface.co/datasets/FrancophonIA/ESLO-FLEU) |
  | **TCOF** | Interactions entre adultes téléchargeables : conversations, entretiens, réunions, de quelques minutes à 45 min+. Diversifie conditions et locuteurs. | [Corpus](https://ct3.ortolang.fr/data/tcof/) · [ORTOLANG](https://hdl.handle.net/11403/tcof/v2) |
  | **Libre à vous !** | Émission de débat FR en MP3/OGG (~1 h 30, plusieurs intervenants, transcriptions). Parole continue légalement réutilisable, mais son de podcast traité, pas un vrai café. | [Archives](https://www.april.org/les-podcasts-libre-a-vous-pour-accompagner-votre-ete-4) · [Licences](https://www.april.org/libre-a-vous-diffusee-mardi-28-septembre-2021) |

  **Protocole retenu** : commencer par une session présentielle complète de
  **SUMM-RE** (~1 h de réunion française, licence claire), compléter si besoin
  par CID/ESLO/TCOF pour la conversation familière et les environnements
  naturels, ou par du YouTube non thématique via le scraper (dialogues, café,
  bureau, rue) ; garder DiPCo comme sous-ensemble de stress optionnel.
  **Chaque fichier passe par WhisperX** pour certifier l'absence de
  « éloquence » / « l'éloquence » / « d'éloquence » (et si le mot y est,
  l'occurrence entre dans la vérité terrain au lieu d'être ignorée).
  Conserver pour chaque enregistrement : source, licence, date de
  téléchargement, identifiant précis (→ ligne DATA.md, DVC si non
  régénérable). Plafond total : 1 h.
- [x] **Re-découper les positifs : fin du mot près de la fin de fenêtre,
  jitter ~200 ms** (recette de découpe, sources brutes inchangées). Expérience
  dédiée vs v03_replica, critère avant le run. C'est le levier n°1 suggéré
  par la littérature ET par l'audit. Explication visuelle :
  `docs/FENETRE_GLISSANTE_ET_JITTER.html` (à ouvrir dans un navigateur).
- [x] **Porte qualité automatique dans la construction des pools** — le
  chantier pivot, AVANT toute exploitation manuelle. Étendre l'audit existant
  (`data/quality.py`, aujourd'hui informatif) en **filtre à trois sorties**
  mesurant par clip : durée, RMS, pic/saturation, SNR estimé, énergie des
  tranches de tête et de queue (débordement du mot suivant), padding de zéros.
  1. **accepté** → entre dans le jeu de données, sans intervention humaine ;
  2. **rejeté** (hors seuils francs) → écarté automatiquement, listé dans un
     rapport écoutable — jamais supprimé silencieusement ;
  3. **douteux** (zone grise entre les deux seuils) → file d'audit humain dans
     Streamlit, verdict oui/non persisté (même mécanique que les verdicts du
     banc).
  Seuils déclarés en config (rien en dur). Les pools étant régénérables, la
  porte s'applique au prochain build ; nouvelle empreinte de dataset =
  nouvelle expérience, jugée au banc. Mêmes mesures branchées sur le futur
  studio d'enregistrement (P2) — un seul code.
- [x] **Tout repasse par la porte — après elle, pas avant** (fait ; reste
  l'audit humain des douteux dans la page Qualité) :
  - les **551 clips de `curation.db`** ne sont pas repris sur la foi de la
    curation manuelle : ils repassent par la porte comme n'importe quel clip,
    et c'est elle qui décide accepté / rejeté / à revoir dans Streamlit ;
  - le **nettoyage ciblé** n'est plus une passe manuelle : les ~17 % de fins
    chargées et les clips anormaux (encodage cassé des `moi_*.wav` compris)
    doivent être attrapés par les mesures de la porte, les 5
    « Dauphine-Éloquence » par la vérité `surface` de `discovery.db` ;
    l'humain n'intervient que sur la file « douteux ».
- [x] **Rappel par forme au banc** — fait ; première mesure : d' 3/7 (43 %)
  contre l' 13/15 (87 %) — : reporter séparément éloquence nu /
  l'éloquence / d'éloquence (la vérité `surface` est dans `discovery.db`).

### P1 — Recette : les expériences dosées

- [x] **Sweep parole continue 0/100/300/500** — FAIT : dose 300 promue champion
  (76 % · 33.9 FA/h, −49 % à rappel égal) (train seulement), sur banc
  étendu. La question v04 posée proprement : la dose, pas le tout-ou-rien.
  Arrêt : quand le gain marginal passe sous la dispersion des candidats
  (ADR-003).
- [x] **Augmentation RIR + multi-SNR** — fait ; v12 : rappel record 88 % mais
  FA/h 72 ; cumul v14 : 84 % · 51.4 (critère raté de peu) → piste v14+fa_ambient (une expérience à part, pas mélangée au
  sweep). Réponses impulsionnelles MIT/BIRD, SNR tirés dans une plage déclarée.
- [ ] **Hard negatives** : les 15 FA confirmées à l'oreille (verdicts du
  banc) + confusables TTS français de la liste ViolaWake. Après extension du
  banc uniquement (anti-fuite).
- [x] **Sélection par FA/h sous contrainte de rappel** — fait ; v13 : l'élection
  seule égale le champion (seeds de 8.7 à 31.4 FA/h ambiantes, ×3,6) : construire une
  `val_ambient` (1-2 h de flux négatif hors banc), élire les candidats dessus
  au lieu de la seule val_loss. Change `selection_metric` — gros gain
  méthodologique possible.

### P2 — Outillage de collecte (l'idée « studio guidé »)

- [~] **Page Streamlit « Studio d'enregistrement »** — codée (page 7 + source
  studio) ; test micro par l'auteur en attente sur le modèle de la
  console ViolaWake : campagne scriptée (10× normal, fort, joyeux, rapide,
  lent, 50 cm / 1 m / 2,5 m), contrôles qualité immédiats (RMS, saturation,
  SNR, réécoute, re-prise), découpe automatique, une session = un groupe de
  split, métadonnées conservées. Alimente `moi_positif`/`guided` et les
  campagnes multi-locuteurs.
- [ ] **Bootstrap d'un nouveau mot-clé** (vision « usine à mots ») : mot tapé
  → génération TTS (voix Piper FR multiples, VoxCPM2 à évaluer pour la
  variété) → campagne studio → scraping YouTube ciblé (scraper en
  sous-processus, ADR-006) → recette standard. Après synthèse, sauf
  besoin jury.

### P3 — Architectures et benchmarks externes

- [ ] Comparer `dscnn` à `cnn_baseline` au banc (déjà au backlog).
- [ ] **Brancher openWakeWord et/ou LiveKit Conv-Attention** comme concurrents
  sur notre banc (leur benchmark : FA/h ÷ 40 vs tête DNN — hypothèse à
  reproduire chez nous, pas à croire sur parole). Environnements séparés
  (motif ADR-006). Intérêt synthèse : comparaison à l'état de l'art.
- [ ] Focal loss sur notre CNN (ablation BCE vs focal) — idée LiveKit peu
  coûteuse à tester dans notre pipeline.
- [ ] Si cible embarquée confirmée un jour : branche microWakeWord (ESP32-S3),
  int8, sélection sur FA/h — voir `ETUDE_MICRO_WAKE_WORD.md` §14.

---

## La méthode de dosage (réponse à « quand s'arrête-t-on ? »)

1. **Une variable à la fois**, doses géométriques (0/100/300/1000…), N seeds,
   élection par la validation, verdict au banc.
2. **Arrêt** : quand le gain marginal entre deux doses est inférieur à la
   dispersion des candidats d'un même run (ADR-003) — mesurable uniquement si
   le banc est assez grand (d'où P0).
3. **Le plafond de vérité** : viser d'abord une cible produit (ex. « ≤ 10 FA/h
   à rappel ≥ 80 % ») et sélectionner les modèles dessus, plutôt que
   d'empiler des données à l'aveugle. Les données s'ajoutent pour combler un
   déficit *mesuré* (comme les hard negatives), pas par principe.
4. Le sweep tts500 d'origine reste le modèle du genre : c'est lui qu'on
   généralise.

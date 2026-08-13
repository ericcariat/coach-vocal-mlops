# Piste B — entraîner une tête openWakeWord sur NOS données réelles

Objectif (cf. `docs/CNN_VS_OPENWAKEWORD.html`) : vérifier que la surdité des
têtes oWW (rappel 16-24 % au banc) vient du tout-synthétique, pas de
l'architecture. Hypothèse : leur extracteur pré-entraîné + NOS voix réelles =
~1-5 FA/h AVEC du rappel.

## Contenu

- `positifs_reels/` — **1 968 clips 16 kHz mono 1 s** : les positifs YouTube
  réels (1 882, toutes formes : nu/l'/d'), les enregistrements « moi » (47) et
  les prises guidées TP/FN de l'auteur. **AUCUN clip TTS** — c'est la variable
  qu'on teste.
- `negatifs_parole_continue_fr/` — **375 fenêtres d'1 s de parole continue
  FRANÇAISE** (YouTube dose 300 + réunions SUMM-RE 75 — les mêmes que le
  champion). Leur corpus générique de négatifs est massif mais très
  anglophone ; c'est le complément de domaine qui nous a divisé les FA par
  deux chez nous.
- `negatifs_adversariaux/` — **122 clips** : cousins phonétiques
  (« élégance », « éloquent »…), FP/TN guidés de la voix de l'auteur, hard
  negatives (FA réelles du banc confirmées à l'oreille). À donner comme
  négatifs custom si l'entraîneur le permet (en PLUS de ses négatifs
  génériques, pas à la place).

## Pourquoi PAS nos bruits/musiques/silences

L'entraîneur oWW apporte ses propres négatifs génériques (~31 000 h ACAV100M :
parole, musique, bruit) — nos ~7 h de MUSAN/GSC/CV seraient redondantes. On ne
fournit que ce que leur océan n'a pas : le français continu et l'adversarial
spécifique au mot.

## Protocole

1. Dans l'entraîneur oWW (web ou notebook), fournir `positifs_reels/` comme
   clips positifs custom — idéalement en DÉSACTIVANT ou réduisant la
   génération TTS, pour isoler la variable. Si l'outil impose un mix, noter la
   proportion.
2. Ajouter `negatifs_adversariaux/` aux négatifs si possible.
3. Mêmes réglages que tes 5 têtes du 12 août (64x3, ~115 k steps) pour la
   comparabilité ; 2-3 seeds si l'outil le permet.
4. Récupérer les .onnx dans `open_wake_word_compare/` puis :
   `uv run coachvocal bench --run open_wake_word_compare/<tête>.onnx --minutes 60 --thresholds 0.05,0.2,0.5`
5. Verdict à consigner dans `artifacts/reports/comparisons/` + CHANGELOG.

Référence à battre (leurs têtes synthétiques, banc du 13/08) : 16-24 % · 1.1
FA/h. Référence champion : v17_stack 48.1 % · 6.8 FA/h @0.8.

⚠️ Fuite à surveiller : ces clips contiennent les vidéos d'ENTRAÎNEMENT de nos
splits — c'est voulu (même train que nous), le banc reste disjoint. Ne PAS y
ajouter de clips issus des vidéos du banc ni des réunions SUMM-RE du banc.

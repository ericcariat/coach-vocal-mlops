# ADR-008 — Deux front-ends acoustiques assumés

Date : 2026-08-14 · Statut : accepté

## Contexte

La règle du projet impose un front-end unique (`audio/features.py`) pour
empêcher toute divergence train/inférence — une divergence produit un modèle
qui « marche en test et rate en vrai », et aucune métrique ne la détecte.

Le transfert d'apprentissage sur l'extracteur speech_embedding de Google
(gelé, front-end mel 32 + embeddings, voir l'étude openWakeWord et le papier
arXiv:2002.01322) a atteint au banc **92.6 % · 3.4 FA/h** au seuil 0.8, là où
le CNN maison plafonnait à 48.1 % · 6.8 — avec **0 %** de déclenchement sur
les mots proches (« éloquente », « élégance »…). Test au micro validé.

## Décision

Le projet assume **deux chaînes acoustiques complètes et étanches** :

- la chaîne maison : log-mel (`audio/features.py`) → CNN → `model.keras` ;
- la chaîne openWakeWord : mel 32 → embeddings Google gelés → tête 64x3 →
  `model.onnx`, servie par `evaluation/oww_adapter.py`.

Chaque chaîne est utilisée de bout en bout — entraînement, banc, live, API —
sans jamais mélanger les étages. La règle du front-end unique devient : **un
front-end unique par chaîne**. La machine à états de décision
(`inference/detector.py` : portail d'énergie, fenêtres consécutives, cooldown)
reste commune aux deux.

## Conséquences

- Le registre résout `model.keras` OU `model.onnx` (`registry.model_path`) ;
  `load_detector` sert le bon détecteur, même interface (`window_probas`,
  `triggers_from`, `push`).
- Le banc streaming compare les deux chaînes aux mêmes règles de comptage —
  c'est ce qui rend le comparatif honnête.
- La dérive train/inférence reste impossible à l'intérieur de chaque chaîne.
- Coût : deux dépendances d'inférence (TensorFlow + onnxruntime), deux
  modèles à connaître, et un protocole d'élection à formaliser pour les
  futures têtes (les runs actuels sont assumés exploratoires, cf. JOURNAL).

## Alternatives écartées

- **Porter le front-end openWakeWord dans `features.py`** : réécrire du code
  gelé de Google, c'est créer le risque de divergence que la règle combat.
- **Abandonner la chaîne maison** : elle reste la référence comprise de bout
  en bout, et le comparatif des deux approches fait partie du dossier.
- **Rester au CNN seul** : l'écart de performance est trop grand, et mesuré
  dans les mêmes conditions.

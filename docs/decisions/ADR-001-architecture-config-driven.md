# ADR-001 — Pipeline piloté par configuration plutôt que scripts numérotés

**Date** : 2026-07-28 · **Statut** : accepté

## Contexte

L'exploration (`coach-vocal_etape1`) suivait une convention de fichiers numérotés
figés : chaque itération créait un nouveau script (`14_eloquence_v09_train.py`,
23 ko) qu'on ne modifiait plus jamais. La règle a bien joué son rôle — la trace
est complète et honnête — mais elle a atteint sa limite :

- douze scripts partageant 80 % de code copié-collé ;
- `get_spectrogram()` dupliqué dans six fichiers, avec le risque permanent qu'une
  version diverge des autres sans que rien ne le signale ;
- changer une dose de TTS = éditer un fichier et en créer un treizième ;
- impossible de comparer deux runs autrement qu'en lisant deux scripts côte à côte.

## Décision

Un **package** unique, et une **configuration YAML** par expérience. Le code
décrit *comment* faire ; la configuration décrit *quoi* faire.

- `configs/experiment/<nom>.yaml` compose mot-clé + dataset + modèle + entraînement ;
- validation pydantic stricte : une clé inconnue échoue au chargement ;
- `extends` pour hériter d'une recette et n'écrire que ses différences ;
- `--set a.b=c` pour un essai ponctuel, sans jamais éditer un fichier.

## Conséquences

**Positives.** Un run devient reproductible depuis sa seule config, archivée avec
lui. Deux expériences se comparent par `diff`. Le front-end acoustique n'existe
qu'en un exemplaire, donc train et inférence ne peuvent plus diverger. Ajouter une
source ou une architecture ne modifie pas le pipeline.

**Négatives.** Le code n'est plus une trace figée : un refactor peut casser la
reproductibilité d'un ancien run. Contreparties : les artefacts (config.json,
manifest.csv, métriques) sont archivés dans chaque run, le dépôt d'exploration
reste intact, et git assure la trace du code.

## Alternatives écartées

- **Continuer les scripts numérotés** : ne résout pas la duplication, et le risque
  de divergence train/inférence reste entier.
- **Hydra** : puissant mais introduit une couche de magie (`_target_`, groupes,
  surcharges) à expliquer en plus du reste. pydantic + YAML se lit sans documentation.
- **Notebooks paramétrés (papermill)** : mauvaise base pour tester et servir.

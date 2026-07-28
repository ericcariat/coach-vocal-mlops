# ADR-005 — Versionner les données avec DVC (remote local)

**Date** : 2026-07-28 · **Statut** : accepté

## Contexte

Le projet manipule ~20 Go de données : corpus externes immuables (MUSAN, Common
Voice, GSC), enregistrements personnels, pools régénérés. Git ne peut pas les
porter, et ce n'est pas souhaitable.

Mais l'absence de versionnement a un coût concret et déjà payé : les 2249 clips
TTS du run v03 ont été régénérés depuis (Piper échantillonne de façon
stochastique). Revenir sur le commit de v03 restitue le code et la config, pas
les données. Le run n'est donc **pas rejouable à l'identique**.

## Décision

- **DVC**, avec un *remote* local sur le même disque (aucun service externe).
- Git ne reçoit que les fichiers pointeurs `*.dvc` (quelques lignes, un hash).
- On garde en parallèle les **manifests CSV** et les **listes de sélection** :
  ils portent une information que DVC n'a pas — la provenance, la licence, le
  rôle de chaque fichier dans un run.

Ce que ce n'est pas : une sauvegarde. Le disque reste un point unique de
défaillance ; un vrai remote (S3, disque externe) pourra être ajouté plus tard
sans changer le reste.

## Conséquences

**Positives.** `git checkout <commit-du-run> && dvc checkout` restitue le dataset
exact d'un run. Le pipeline devient rejouable, ce qui est le sens même de la
reproductibilité annoncée.

**Négatives.** Un outil de plus, une étape de plus (`dvc add` / `dvc push` après
chaque enrichissement du dataset), et un cache qui double l'espace occupé par les
données suivies — d'où le choix de ne suivre que ce qui n'est pas régénérable
(`raw/`, `clean/`, `guided_clips/`, `hard_negatives/`) et de laisser les pools
dérivés hors DVC, puisqu'un script les reconstruit depuis la seed.

## Alternatives écartées

- **Manifests + sha256 seuls** : documentent, mais ne restituent rien. On saurait
  que le dataset a changé, sans pouvoir revenir en arrière.
- **git-lfs** : conçu pour de gros fichiers versionnés, pas pour des dossiers de
  dizaines de milliers de petits fichiers audio.
- **Ne rien faire** : c'est l'état actuel, et il empêche de rejouer un run.

# ADR-006 — Rendre le projet autonome : corpus copié, scraper en sous-projet

**Date** : 2026-07-28 · **Statut** : partiellement appliqué (corpus fait, scraper à décider)

## Contexte

Le banc streaming — la mesure sur laquelle **toutes les promotions se décident** —
lisait son audio, ses sous-titres et sa vérité terrain (`discovery.db`) à travers
un lien symbolique vers un autre projet, situé sur un autre disque :

```
data/external/youtube_corpus → /Users/eric/projects/alyra/…/scraper-audio/data
```

Trois façons dont cela pouvait casser sans prévenir : le projet source déplacé ou
nettoyé, le disque interne indisponible, ou le scraper régénérant ses fichiers.
Dans chacun de ces cas, le chiffre qui décide des promotions devient
irreproductible — et on ne s'en aperçoit qu'au moment de le refaire.

Un jury peut légitimement demander « et si je clone votre dépôt sur une machine
vierge, qu'est-ce qui tourne ? ». La réponse était « tout sauf l'évaluation ».

## Décision 1 — Copier le corpus (fait)

Le corpus est **copié** dans `data/external/youtube_corpus` : 1,7 Go, 1278
segments audio, 613 fichiers de sous-titres, `discovery.db` (2006 occurrences
alignées par WhisperX) et `curation.db`.

Une copie et non un lien physique : les deux projets sont sur des volumes
différents, `cp -al` ne peut pas les relier.

Vérification faite : le banc donne des résultats identiques avant et après
(80 % de rappel, 54 FA/h au seuil 0.8).

Le corpus est suivi par DVC comme le reste des données non régénérables : les
vidéos YouTube disparaissent, ces fichiers ne sont pas re-téléchargeables à
l'identique.

## Décision 2 — Le code du scraper, s'il est repris, ira en sous-projet

**Contrainte technique dure** : les deux environnements Python sont incompatibles.

| | coach-vocal-mlops | scraper-audio |
|---|---|---|
| Python | **3.11** (imposé par `tensorflow-macos`, pas de wheel 3.12) | **≥ 3.12** |
| numpy | **1.26.4** (TF 2.16 ne fonctionne pas avec numpy 2) | **≥ 2.0** |
| Cœur | tensorflow, keras, scikit-learn | whisperx, mlx-whisper, yt-dlp |

Fusionner les dépendances est impossible sans renoncer à l'un des deux cœurs.

**Forme retenue** : un sous-projet `tools/scraper/` avec son **propre**
`pyproject.toml` en 3.12, jamais importé par le code principal, appelé en
sous-processus :

```bash
uv run --project tools/scraper python -m scraper.pipeline --query "éloquence"
```

Une page Streamlit « Corpus » déclenche la collecte et affiche l'avancement —
exactement le motif déjà utilisé par la page « Entraînement », qui lance un
entraînement par sous-processus et affiche son journal en direct.

## Conséquences

**Positives.** Le dépôt couvre alors toute la chaîne, de la collecte YouTube au
service. Il tourne sur une machine vierge. L'isolement des environnements est
explicite et défendable, au lieu d'être subi.

**Négatives.** Deux environnements à installer et à maintenir. `whisperx` évolue
vite et son installation peut casser ; c'est une dépendance à surveiller. Environ
une heure de travail.

**Si on ne le fait pas** : les données étant copiées, l'entraînement et
l'évaluation sont déjà autonomes. Seule la *collecte de nouvelles données*
resterait dans l'autre projet. C'est un compromis défendable devant un jury à
condition de l'assumer explicitement plutôt que de le découvrir à la question.

## Alternatives écartées

- **Tout fusionner dans un seul environnement** : techniquement impossible.
- **Conteneuriser le scraper** : plus lourd qu'un second environnement `uv`, et
  `mlx-whisper` exploite le Neural Engine d'Apple, mal servi dans Docker.
- **Refaire un scraper en 3.11** : réécrire ce qui existe et fonctionne, pour
  un bénéfice nul.
- **Garder le lien symbolique** : c'est précisément ce qui rendait la mesure
  décisive du projet dépendante d'un élément extérieur.

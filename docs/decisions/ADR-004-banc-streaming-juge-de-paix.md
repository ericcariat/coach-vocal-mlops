# ADR-004 — Le banc streaming décide, le test par clips contrôle

**Date** : 2026-07-21 · **Statut** : accepté

## Contexte

Le test par clips (F1, FRR, FAR sur des extraits d'1 s pré-découpés) a **mal
classé** les modèles candidats :

| Modèle | F1 par clip | Rappel streaming | FA/heure |
|---|---:|---:|---:|
| candidat « meilleur clip » | 0.9615 | 60.5 % | 64.0 |
| v02 | 0.9516 | 65.8 % | 56.5 |
| v03 | 0.9262 | **68.4 %** | **52.7** |

Le meilleur en clip était le pire en réel. L'explication est structurelle : un
clip d'1 s parfaitement centré sur le mot n'existe pas en production. En
production il y a des demi-mots, du recouvrement, du bruit continu, et une
décision à prendre huit fois par seconde. Un modèle qui excelle sur des clips
propres peut n'avoir jamais appris à rester silencieux entre les mots.

## Décision

1. **La promotion se décide au banc streaming** : rappel + fausses alarmes/heure.
2. **Le test par clips reste** comme signal de contrôle (détection de régression
   grossière, breakdown par pool pour savoir *quelle* famille de négatifs pose
   problème) — jamais comme critère de décision.
3. **Vérité terrain = alignements WhisperX** (`discovery.db`), pas les sous-titres
   automatiques : ceux-ci dérivent de ±1 s et ignorent le padding de découpe de
   `yt-dlp` (jusqu'à 10 s). Cette confusion avait produit un premier banc
   entièrement faux, avec un rappel proche de zéro pour tous les modèles.
4. **Les cas douteux sont exclus**, pas arbitrés : un déclenchement non apparié
   mais proche d'un temps de sous-titre est marqué « incertain » et ne compte ni
   comme détection ni comme fausse alarme.
5. **Anti-fuite** : les vidéos ayant servi à l'entraînement sont interdites au banc.

## Conséquences

**Positives.** Les décisions reposent sur la métrique que vit l'utilisateur.
Le banc a immédiatement révélé le déficit principal — l'absence de négatifs de
parole continue — qu'aucune métrique par clip ne montrait.

**Négatives.** Le banc est plus lent (quelques minutes d'audio par modèle) et
dépend d'un corpus externe. Les FA/heure sont mesurées sur un corpus thématique,
donc pessimistes : à présenter comme un pire cas, jamais comme une moyenne.

## Règle pratique

Aucune promotion sans banc. Un run non passé au banc reste un run, pas un candidat.

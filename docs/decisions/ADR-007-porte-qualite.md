# ADR-007 — Porte qualité à trois sorties dans la construction des pools

Date : 2026-08-13 · Statut : acceptée

## Contexte

L'audit du 2026-08-13 a montré que les pools contiennent des clips défectueux
invisibles à l'entraînement : fichiers muets (134), fins « chargées » où le mot
suivant déborde (~17 % des positifs YouTube), saturations, et 5 occurrences
d'un nom propre (« Dauphine-Éloquence ») étiquetées positives. L'audit
existant (`data/quality.py`) signalait certains problèmes mais ne filtrait
rien ; le nettoyage se faisait à la main, sans critères mesurés — la curation
de `curation.db` (551 clips) en est le témoin : un travail humain sans trace
de ses règles, finalement jamais réutilisé.

## Décision

Une **porte qualité automatique** (`data/gate.py`) mesure chaque clip (durée,
RMS, pic, saturation, SNR estimé, énergie des tranches de tête/queue, padding
de zéros) et rend un verdict à **trois sorties** :

1. **accepté** → entre dans le jeu, sans intervention humaine ;
2. **rejeté** (hors seuils francs) → écarté automatiquement, listé dans la
   page « Qualité », jamais supprimé du disque ;
3. **douteux** (zone grise entre les deux seuils) → file d'audit humain dans
   Streamlit, verdict oui/non persisté ; tant qu'il n'est pas tranché, le clip
   est exclu (`doubt_policy: exclude`).

Principes structurants :

- **Seuils en config** (`dataset.quality_gate`, pydantic) — rien en dur ; un
  rejet se conteste en changeant la config et en relançant la porte.
- **Opt-in par recette** : `enabled: false` par défaut, les recettes
  historiques restent bit-à-bit comparables. Activer la porte change
  l'empreinte du dataset : c'est une expérience, jugée au banc.
- **Contrôles contextuels par pool** : la « fin chargée » n'a de sens que pour
  des clips de mot isolé (`tail_check_pools`) — une tranche de parole continue
  finit énergique par nature ; les pools de bruit (`lenient_pools`) échappent
  aux doutes de pic/SNR (le bruit de fond est faible par nature) mais pas aux
  rejets francs. Sans cela, la première passe produisait 5 610 douteux
  (inutilisable) ; avec, 622 dont 336 positifs à fin chargée — une file
  auditable.
- **Le build échoue fort** si la porte est activée sans rapport existant :
  jamais de filtrage silencieux ni implicite.
- **Un seul code de mesure** : les mêmes fonctions serviront au studio
  d'enregistrement guidé (P2) pour le contrôle qualité à la prise.

## Conséquences

**Positives.** Le nettoyage devient reproductible et contestable ; l'humain ne
juge que la zone grise (622 clips au lieu de 9 420) ; les 551 clips de
`curation.db` peuvent repasser par la porte comme n'importe quel lot.

**Négatives.** Un dataset filtré n'est plus comparable aux runs historiques
(assumé : nouvelle empreinte = nouvelle expérience). Les seuils sont un
jugement encodé — ils peuvent se tromper dans les deux sens, d'où la sortie
« douteux » et le rapport consultable plutôt qu'une suppression.

## Alternatives écartées

- **Continuer le nettoyage manuel** : non traçable, non rejouable sur un
  nouveau lot, et la preuve est faite qu'il se perd (curation.db).
- **Filtrage binaire accepté/rejeté** : oblige à des seuils either laxistes
  (les fins chargées passent) ou brutaux (des clips sains sautent) ; la zone
  grise auditée à l'oreille est exactement la leçon des verdicts du banc.
- **Filtrer dans les sources une à une** : dispersé et incohérent ; la porte
  est un étage unique du build, comme le contrôle anti-fuite.

## Chiffres de référence (première passe, recette tts500, 2026-08-13)

9 420 clips → 8 664 acceptés · 134 rejetés (muets surtout) · 622 douteux
(336 `yt_positif` fin chargée, 143 parole très faible, 90 `gsc`, 23 `proches`,
20 `moi_positif`). Rapport : `data/wakewords/eloquence/gate/gate_report.json`.

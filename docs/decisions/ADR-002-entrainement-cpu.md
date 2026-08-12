# ADR-002 — Tout sur CPU : entraînement **et** inférence

**Date** : 2026-07-12 · **Révisé le 2026-07-28** (l'inférence Metal était tolérée,
elle ne l'est plus) · **Statut** : accepté

## Contexte

Sur Apple Silicon, `tensorflow-metal` expose le GPU intégré à TensorFlow. Sur
cette machine, les entraînements lancés avec le plugin actif divergent : la loss
explose au bout de quelques centaines de batches, avec des hyperparamètres
pourtant sains (Adam, lr 1e-3, batch 64). Le même code, mêmes données, mêmes
seeds, converge normalement une fois le GPU masqué.

Plusieurs heures ont été perdues à chercher un problème d'apprentissage — taux
d'apprentissage, normalisation, déséquilibre des classes — alors que le défaut
était dans le backend de calcul.

## Révision du 2026-07-28 : l'inférence non plus

On supposait le calcul en avant sain sur Metal. Mesure faite en migrant le banc
streaming, **même modèle, même audio, même code** :

| | Rappel streaming | FA/heure (seuil 0.8) |
|---|---:|---:|
| CPU | **80 %** (4/5) | 54 |
| Metal | **0 %** (0/5) | 72 |

Sur une occurrence, la probabilité maximale passe de **1.00 (CPU) à 0.37 (Metal)**.
Ce n'est pas un écart d'arrondi : c'est la différence entre détecter et ne pas
détecter. L'ancien banc masquait le GPU explicitement, ce qui rendait le problème
invisible — et le live, lui, tournait bel et bien sur Metal.

## Décision

- **Entraînement : CPU obligatoire.** `training.use_gpu: false` par défaut, et
  `runtime.configure()` masque les GPU avant toute opération TensorFlow.
- **Inférence : CPU également.** Banc, micro, API et interface configurent
  explicitement le CPU. Un modèle doit se comporter en production exactement
  comme au banc — sinon le banc ne mesure rien.
- Le paramètre `use_gpu` reste disponible partout, pour pouvoir re-tester le jour
  où le plugin sera corrigé.

## Conséquences

**Positives.** Entraînements stables et comparables entre eux. Le piège est
documenté au lieu d'être redécouvert.

**Négatives.** Un entraînement complet prend ~40 minutes au lieu de quelques
minutes. Le nombre de candidats par run (5) est donc un vrai coût — c'est
assumé, la variance qu'il neutralise est plus coûteuse encore.

**Effet de bord.** Sur CPU, l'entraînement n'est de toute façon pas déterministe
(ordonnancement des threads, non-associativité des flottants). Voir ADR-003.

## À réévaluer si

Une version ultérieure de `tensorflow-metal` corrige le problème. Deux tests de
non-régression, à passer tous les deux :

1. `coachvocal train smoke --set training.use_gpu=true` → la loss reste finie et
   décroissante ;
2. le banc streaming sur GPU retrouve le rappel mesuré sur CPU (à ±2 points).

## Re-contrôles

**2026-08-12** — `tensorflow-metal` 1.2.0, run `v05_metal_check` (recette
v03_replica à l'identique, `use_gpu: true`). Reconfirmé, dans les deux sens :

- **Entraînement corrompu** : les 5 seeds s'effondrent (val_loss 0.36–2.5
  contre ~0.06 sur CPU, early stopping à 6 epochs partout, F1 clips 0.60–0.84).
- **Au banc (inférence CPU sur le modèle élu)** : 764 FA/h contre 47 pour
  v03_replica — le modèle tire sur tout, son rappel apparent de 95,7 % ne
  mesure rien.
- **Le gain de temps existe mais est théorique** : ~3 s/epoch sur Metal contre
  ~14 s sur CPU (×4,7). Sans apprentissage correct, il n'achète rien.

La décision tient. Prochain re-contrôle : à la prochaine montée de version de
`tensorflow-metal`, via les deux tests ci-dessus.

**2026-08-12 (soir) — cause racine identifiée.** Ce n'était ni les gradients ni
les probabilités en tant que tels : **le noyau fusionné MatMul+BiasAdd+ReLU du
plugin Metal n'applique pas le ReLU** — la sortie brute, négatifs compris,
traverse le réseau. Reproduction minimale sur cette machine (M4 Max, macOS,
tensorflow-macos 2.16.2, tensorflow-metal 1.2.0) :

| Chemin d'exécution | min de sortie (attendu 0.0) |
|---|---:|
| CPU, `Dense(activation="relu")` | 0.0 ✅ |
| **Metal, `Dense(activation="relu")`** | **−11.0 ❌** |
| **Metal, `Dense` + `Activation("relu")` séparée** | **−11.0 ❌** (le remapper fusionne quand même) |
| Metal, `tf.nn.relu` hors graphe | 0.0 ✅ |

Un réseau dont les non-linéarités disparaissent explique *tout* le tableau
constaté : loss qui diverge à l'entraînement, probabilités absurdes à
l'inférence, et des métriques parfois plausibles (le sous-réseau linéaire
apprend un peu quand même — c'est ce qui rend le bug si trompeur).

**2026-08-13 — le contournement lui-même invalidé (`v08_metal_maxrelu`).**
Même en remplaçant le ReLU par `tf.maximum(x, 0)` — mathématiquement identique,
non fusionné, vérifié sain — l'entraînement Metal produit un modèle aux
métriques amont indiscernables du CPU (val_loss 0.056, F1 clips 0.9322) mais à
**120.3 FA/h au banc contre 47.1** pour le même entraînement sur CPU. Deux
autres activations saines (leaky_relu, elu — runs v06/v07) avaient déjà échoué.
Conclusion : au-delà du bug ReLU documenté ci-dessous, le backend produit des
modèles subtilement différents d'une façon qu'aucune métrique d'entraînement ne
détecte. **La décision ne repose donc plus seulement sur un bug identifiable et
corrigeable : Metal est écarté même contourné.**

Signalé publiquement sans réponse d'Apple à ce jour :
[thread 818015](https://developer.apple.com/forums/thread/818015) (même config
que la nôtre) et [tensorflow#62137](https://github.com/tensorflow/tensorflow/issues/62137)
(même bug dès TF 2.14). Re-contrôle en une commande :
`uv run python scripts/check_metal_relu.py` — tant qu'il affiche un min
négatif, cet ADR tient.

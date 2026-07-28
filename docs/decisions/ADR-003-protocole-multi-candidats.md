# ADR-003 — N candidats, élection par la validation

**Date** : 2026-07-21 · **Statut** : accepté

## Contexte

Deux entraînements lancés avec **la même seed et les mêmes données** ne donnent
pas le même modèle sur CPU : ordonnancement des threads, non-associativité de
l'addition flottante, et un `EarlyStopping` qui coupe entre l'epoch 8 et l'epoch 20
selon le tirage. Variance mesurée : **±0.03 à 0.06 de F1**.

C'est plus que la plupart des améliorations qu'on cherche à mesurer. Une
conclusion a effectivement dû être rétractée : huit clips avaient été jugés
nuisibles sur la base de trois comparaisons appariées concordantes ; les retirer
a donné un résultat *encore pire*. Le facteur dominant n'était pas les clips,
c'était le bruit d'entraînement.

## Décision

1. **N candidats par run** (5 seeds), chacun archivé dans
   `runs/<id>/candidates/seed<N>/` avec ses métriques.
2. **Élection par la validation** (`val_loss`). Jamais par le test.
3. **Le test n'est regardé qu'une fois**, sur le modèle élu.
4. **Empreinte du dataset** enregistrée à chaque run : à empreinte identique, un
   écart de métriques est de la variance, pas un progrès.

Élire par le test reviendrait à choisir le modèle le plus chanceux sur le jeu qui
sert précisément à prouver qu'il est bon — biais de sélection classique, et le
chiffre annoncé serait optimiste.

## Conséquences

**Positives.** Les comparaisons entre recettes redeviennent interprétables. Le
tableau des candidats figure dans chaque rapport : le jury voit la dispersion, pas
seulement le meilleur.

**Négatives.** Un run coûte 5× plus cher (~40 min). `--set training.seeds=[42]`
permet de débrayer pour un essai rapide, en sachant que le résultat n'est alors
pas comparable.

## Règle pratique

Avant de conclure qu'une modification apporte un gain, vérifier qu'il dépasse la
dispersion des candidats du même run. Sinon, c'est du bruit.

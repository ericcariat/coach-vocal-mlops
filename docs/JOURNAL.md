# Journal des explorations

Le `CHANGELOG` raconte les runs officiels. Ce journal raconte les
essais, les diagnostics, les impasses.

Les preuves sont dans `artifacts/reports/` et le code dans git
(`git log --oneline`, commits du 14 août).

---
## 2026-08-15 — Nuit 2 : v28 promu, puis la campagne des réglages fins (critères AVANT)

**v28_resserre est PROMU champion** (décision d'usage après test micro,
seuil live global 0.80 → 0.90). Programme validé pour la suite de la nuit :
12 entraînements max, fine-tuning de l'extracteur autorisé en fin de nuit,
promotion autonome si la grille COMPLÈTE est verte (les six clauses, cousins
≤ 15 % @0.9 inclus). Référence à battre : v28 (FRR 8.8, FAR 1.7, banc
88.9 · 5.7 @0.9, vitesses 64/91/72/60, suffixes 4 %, cousins 20 %).

**S1 (gratuit) — balayage de la machine à états** : n_consecutive {3,4,5} ×
cooldown {1.0,1.5} × seuils {0.85,0.9,0.95} sur le champion, banc + voix +
préfixes. Attendu déclaré : un point qui gagne ≥ 1.5 FA/h sans perdre plus
de 2 pts de rappel banc ni 5 pts de voix — sinon on garde 3/1.5/0.9.

**Verdict S1 : pas de point gagnant selon le critère** (le meilleur candidat,
4 consécutives @0.9, gagne 2.3 FA/h mais perd 3.7 pts de rappel banc — au-delà
des 2 autorisés). Le réglage champion reste 3 consécutives / cooldown 1.5 /
seuil 0.9. Deux découvertes consignées : le cooldown est neutre (1.0 ≡ 1.5
partout), et la persistance est un tueur de préfixes — à 5 consécutives, les
préfixes 90 % tombent à 9 % SANS entraînement (mais rappel 81.5 % et voix
62-70 %). Un « profil silencieux » (5/0.85 : 81.5 % · 2.3 FA/h · préfixes
9 %) existe désormais sur étagère pour un usage qui privilégierait le calme.

**S2 — le scalpel cousins (v32)** : poids dédié aux 68 cousins du pool
adversarial (45 moi_ + 23 « éloquente » de la session guidée), ×120, les 54
hard negatives restant à ×50. Une variable vs v28. Critère : cousins ≤ 15 %
@0.9 ET toutes les bornes v28 conservées.

**Verdict v32 (run 4/12) : échec paradoxal — et la LOI DES POIDS se
confirme.** Cousins PIRES (36 % @0.9), FA/h ×8 (46.6), préfixes 79 % — alors
que voix et vitesses culminent (98-100 %, 96/98/94/81). À travers H3 (préfixes
×20), v31 (adv ×80) et v32 (cousins ×120), le motif est systématique :
**au-delà d'un poids ~50, les gradients géants déstabilisent l'entraînement
et produisent l'INVERSE du but**. Règle de dosage adoptée : jamais plus de
~50 ; pour insister, multiplier les DONNÉES, pas les poids.

**S3 (v33, run 5/12) — cousins multi-vitesses** : les 68 cousins déclinés en
×0.85/0.95/1.05/1.15 (272 fenêtres de plus, variété réelle) au poids
adversarial normal ×50 — plus de données au lieu de plus de poids. Une
variable vs v28. Critère inchangé : cousins ≤ 15 % @0.9, bornes v28
conservées.

## 2026-08-15 — La nuit du 15 août, racontée simplement

**1. Trois essais de modèle, trois échecs — et trois vraies leçons.**
On voulait corriger la dernière fuite entendue au micro (« loquence », le mot
sans son début). v29 a ajouté des contre-exemples de ce type : tout s'est
desserré. v30 a testé une autre façon d'équilibrer positifs et négatifs :
147 fausses alarmes par heure, la pire mesure du projet — hypothèse enterrée,
et c'est tant mieux : on sait maintenant POURQUOI les réglages faisaient de
la balançoire. v31 a durci tous les pièges d'un coup : les mots rapides sont
repassés à la trappe. Trois essais sans battre la référence = arrêt, comme
convenu avant la nuit.

**2. La vraie découverte : on chassait un fantôme.** En mesurant enfin la
référence v28 sur les « suffixes », surprise : au seuil d'usage (0.9), elle
ne fuit presque pas (4 %). La fuite entendue était marginale dans la mesure.
v28 tient TOUTES les cases de la grille de promotion sauf une (les
mots-cousins : 20 % au lieu de ≤ 15 %) : pas de promotion automatique, le
test au micro départagera v28 et le champion actuel au réveil.

**3. Le notebook est né — et il tourne.** `notebooks/
detection_mot_cle_eloquence.ipynb` : 33 cellules toutes exécutées — on y
écoute les sons, on voit l'audio devenir une image, un petit CNN s'entraîne
en direct avec ses courbes, matrice de confusion et courbe ROC, puis le
transfert d'apprentissage et l'appel de l'API. Simple et lisible, commité.

**4. Le dossier de présentation est prêt à être nourri**
(`docs/certification/`, jamais dans git) : le fil conducteur interactif de
l'oral en 9 étapes minutées, le déroulé de démo de l'API vérifié en vrai, la
liste des captures à faire, et 7 figures déjà copiées.

## 2026-08-15 — Nuit autonome : v29 et la campagne des suffixes (critères AVANT les runs)

Programme validé avant lancement (10 entraînements max, seed 46, promotion
autonome si critères verts, juge final : le test micro au réveil).

Constat au micro (2026-08-15, v28) : « loquence » déclenche — le mot amputé
de son début passe. Miroir exact des préfixes : rien n'exige le « é ». La
mesure des suffixes entre au banc des cousins (85 % conservés ≈ « loquence »,
65 % ≈ « oquence »), et **v29** ajoute les négatifs-suffixes (65/80/90 % du
mot conservés depuis la fin, multi-vitesses, souffle — même machinerie que
les préfixes).

**Critère v29, écrit avant le run** (référence v28 : FRR 8.8 %, FAR 1.7 %,
banc 88.9 % · 5.7 @0.9, vitesses 64/91/72/60, suffixe 85 % non mesuré avant) :
  - conserver v28 : FRR ≤ 10 %, FAR ≤ 5 %, banc ≥ 85 % · ≤ 6 FA/h @0.9,
    vitesses ≥ v28 − 5 pts ;
  - gagner : suffixes 85 % ≤ 15 % @0.9 ET cousins moi_ ≤ 15 % @0.9.
Les itérations suivantes (dosages) hériteront du même critère, une variable
à la fois, verdicts consignés ici.

**Verdict v29 (run 1/10, archive `eloquence_20260815_*`) : échec — et la
cause structurelle des balanciers enfin identifiée.** Les suffixes
progressent (28 % @0.8, 0 % à 65 %) et la voix monte à 98 %, mais tout le
reste se desserre : 39.8-66 FA/h, cousins 42 %, préfixes 85 %. Le motif se
répète à CHAQUE ajout de négatifs depuis H3. Cause identifiée : le
`class_weight` par comptes bruts — ajouter 17 700 négatifs augmente
mécaniquement le poids de chaque positif (la balance compense), et le modèle
devient gâchette. **v30 (run 2/10)** : même recette que v29, une seule
variable — l'équilibre de classes calculé sur les masses EFFECTIVES
(pondérations comprises). Critère identique à v29 ; baseline suffixes de v28
mesurée au passage.

**Verdict v30 : hypothèse FALSIFIÉE, violemment** — 146.8 FA/h @0.8, le pire
banc du projet. Les masses effectives sur-pondèrent les positifs (×21 au lieu
de ×2.7 par comptes) : l'inverse d'une stabilisation. Le calcul historique
redevient le défaut (`--effective-class-weight` conservé en opt-in pour
reproduire l'expérience).

**Et la mesure qui recadre la nuit** : le baseline suffixes de v28, mesuré
enfin, est déjà BON — 13 % @0.8, **4 % @0.9**, 2 % @0.95. Au seuil d'usage
0.9, la fuite « loquence » est marginale dans la mesure : v29/v30 ont dégradé
un système qui tenait déjà cette clause. Sur les critères de la nuit, v28 ne
manque QU'UNE case : **cousins moi_ 20 % @0.9 (objectif ≤ 15 %)**.

**v31 (run 3/10)** : recette v28 exacte, une variable — adversariaux
×50 → ×80 (les 45 cousins et 23 « éloquente » vivent dans ce pool). Critère :
cousins ≤ 15 % @0.9, suffixes restent ≤ 15 % @0.9, et toutes les bornes v28
conservées (FRR ≤ 10, FAR ≤ 5, banc ≥ 85 % · ≤ 6 @0.9, vitesses − 5 pts max).

**Verdict v31 : échec** — cousins 16 % @0.9 (mieux que 20) mais vitesses
effondrées (×1.15 : 26 %, ×1.3 : 9 % contre 72/60 chez v28), FRR 17.6 %,
banc 88.9 % · 10.2 FA/h. Le durcissement adversarial global re-sacrifie les
mots rapides.

**Arrêt des dosages (règle pré-déclarée : 3 essais consécutifs sans battre la
référence).** v29, v30, v31 : trois verdicts nets, trois leçons (les
suffixes étaient déjà tenus à 0.9 ; les masses effectives sur-pondèrent ;
le marteau adversarial casse les vitesses). **v28 reste le meilleur
candidat de la nuit** — il tient TOUTES les clauses de promotion sauf une
(cousins 20 % @0.9 pour ≤ 15 %) : pas de promotion autonome, le champion
reste `oww_frab30np4b_s46`, le test micro du matin tranchera entre les deux.
Piste future pour la clause cousins : un poids DÉDIÉ au sous-ensemble
cousins/« éloquente » (sans durcir les 54 hard negatives en même temps),
et/ou de nouveaux enregistrements de cousins à vitesses variées.

## 2026-08-14 — Le champion tire avant la fin du mot : diagnostic et H3 (critère AVANT le run)

Constat au micro : la tête déclenche sur « éloquen » (sans la fin), et
« éloquente » passe. Diagnostic chiffré (préfixes du mot collés au bord droit,
47 clips de la voix de référence) : **0 % de déclenchement jusqu'à 70 % du
mot, 17 % à 80 %, 38 % à 90 %** — la tête tire dès que ~85 % du mot est là,
et avec la persistance live (3 × 80 ms) le déclenchement précède la fin.
Cause : les positifs contiennent toujours le mot ENTIER, et rien n'enseigne
la frontière 80-95 % — le rôle des fragments du CNN (cf. v23), noyés à poids 1
dans l'entraînement de la tête. Nos pools s'arrêtent d'ailleurs à 70 % —
précisément là où la fuite est nulle.

**H3** (`frab30np3`) : recette du champion + négatifs-préfixes 60/75/85 % du
mot (mesure par énergie, positifs d'entraînement seulement, plafond 85 % pour
ne pas contredire le mot entier), poids ×20. Le banc des cousins mesure
désormais les préfixes 80 % et 90 % en permanence. Critère écrit avant :
  - préfixes 80 % ET 90 % : ≤ 5 % de déclenchement @0.8 (contre 17 %/38 %) ;
  - mot entier (clips) : ≥ 65 % @0.8 (niveau actuel ~68 %) ;
  - banc : rappel ≥ 85 % @0.8 avec FA/h ≤ 4.5 (champion co-mesuré 92.6 · 3.4) ;
  - verdict final : le test micro (« éloquence » détecté, « éloquen » et
    « éloquente » muets).

**Verdict H3 (archive `eloquence_20260814_205246`) : la fuite est bouchée,
mais la dose ×20 écrase le mot.** Préfixes 80/90 % : **0 %** sur les deux
seeds (contre 17/38 % chez le champion) — le mécanisme fonctionne. Mais le
mot entier s'effondre sur clips (6-9 % @0.8 contre 70 % : clause « ≥ 65 % »
échouée) : ~5 900 préfixes ×20 pèsent trois fois plus que tous les positifs.
Le banc reste haut (89 % @0.8 — ses positifs ont du vrai contexte), FA/h en
légère hausse (4.6-6.8). Sur-correction classique → **H4** : balayage de dose
(préfixes ×3 et ×8), plus les **145 adversariaux** (23 enregistrements
« éloquente/éloquen » du soir au micro, 10 FP à proba 0.98-0.9999 confirmant
la fuite en réel, 13 TN). Même critère que H3.

**Verdict H4 et PROMOTION (archive `eloquence_20260814_211029`).** Dose ×3 :
préfixes corrigés (0-2 %) mais FA/h ×5 au banc (18.2, surtout sur les
réunions SUMM-RE). Dose **×8** : préfixes 0 %, cousins 0 %, banc 81.5 % ·
2.3 FA/h — et surtout **le test micro tranche** : « éloquence » détecté,
« éloquen » et « éloquente » muets, là où le proxy par clips (9 % @0.8)
annonçait un effondrement. Leçon : le proxy « clips padés » diverge du micro
dans les deux sens, seul le test en conditions réelles départage.
**`oww_frab30np4b_s46` promu champion** — recul YouTube (22/27 vs 25/27)
accepté au titre de la priorité « voix cible d'abord ».

## 2026-08-14 — Contrôle complet après la publication et la promotion

Après le grand ménage (réécriture git, textes, promotion), revue de TOUTE la
chaîne avec le champion ONNX. Quatre trous trouvés et bouchés : le banc par
défaut cherchait `model.keras` en dur ; l'API `/detect` appelait `run_offline`,
absent de l'adaptateur ; `runtime.configure` plantait si TensorFlow était déjà
initialisé ; et les tests qui importent TensorFlow tournaient depuis toujours
avec le GPU Metal visible (contraire à l'ADR-002) — `tests/conftest.py` force
désormais le CPU pour toute la suite.

Validation finale : 72 tests verts, `make smoke` complet, chemin micro et API
vérifiés sur le champion, et une version ré-entraînée à l'identique
(`frab30np2`, archive `eloquence_20260814_201059`) qui reproduit le champion
**au chiffre près** (93 % · 3.4 @0.8 ; 89 % · 2.3 @0.95 ; cousins 0 %) — la
chaîne d'entraînement des têtes est déterministe sur features en cache.

## 2026-08-14 — 14 août - part 2

**1. On adapte le réseau dans le même style que OpenWakeWord**

L'objectif étant d'atteindre +- 1 FA/h (recommendé 0.5 FA/h (porcupine doc))
Donc l'idée c'est de garder les premières couches d'un réseau qui provient de Google/OpenWakeWord. 
voir étude:  
[Training Keyword Spotters with Limited and Synthesized Speech Data](https://arxiv.org/pdf/2002.01322)

C'est un réseau qui est (déjà bien entraîné!) entraîné par 200 millions de clips Youtube. (Finalement l'idée 
de "scrapping" est peut être pas si mauvaise ;-)).
On va juste réentraîner la tête pour qu'elles reconnaissent le mot éloquence.

**2. H1 (eloquence_frab30_...) Plus de poids pour les mots proches**
Les mots ressemblant à « éloquence », comme « éloquente », provoquaient trop de fausses détections. 
Fixé via plus de poids pendant l’entraînement.
Entraînement de la tête sur nos données réelles uniquement (pas d'océan ACAV) : positifs à contexte réel + 12 000 fenêtres de parole française en négatif (poids ×20) + les 122 pièges connus en poids ×30 (exports/oww_training_b/negatifs_adversariaux/ : 45 mots proches, 54 fausses alertes du banc, 23 essais micro).
Commande : `scripts/train_oww_head.py --context --french-neg 12000 --adv-weight 30`

Résultat du modèle `frab30`, seed 42 :

- **85 %** de détection ;
- **1,1 fausse alerte par heure** ;
- **0 %** de déclenchement sur les mots proches, contre **16 % avant**.

La mesure est réalisée avec :
- `scripts/eval_oww_cousins.py`
- `artifacts/reports/oww_cousins.png`

**3. H2 (eloquence_frab30np_...)  Ajout de bruit de fond**
Les enregistrements ont ensuite été mélangés avec du vrai bruit de fond (à la place d'un silence parfait!)
Résultat du modèle `frab30np`, seed 46 :

- **89 %** de détection ;
- **2,3 fausses alertes par heure** ;
- **0 %** de déclenchement sur les mots proches.

### 4. Comparaison avec le modèle actuel

Pour rappel, le modèle officiel actuel, `CNN v17`, atteint :

- **48 %** de détection ;
- **6,8 fausses alertes par heure**.

**Les nouveaux modèles openWakeWord font donc nettement mieux pendant les tests.**

confirmé aussi aux tests micro !

Note : Dans les noms : 
- `fr` = négatifs français ×20, 
- `ab30` = adversariaux ×30,
- `np` = noise pad (bruit de fond), 
- `seed4x` = seed du random, 
- `64x3` = la taille de la tête.

## 2026-08-14 — 14 août

**1. Les « fragments » réparés deux fois.**
Les fragments sont des bouts du mot (« élo… », « …quence ») qu'on montre au
modèle. Problème découvert : certains contenaient presque tout le mot. 
L'intuition m dit que prendre un mot quasi complet dans les négatifs va 
perturber le modèle. 

Première réparation ratée (la mesure confondait le souffle du micro avec le mot),
Première réparation ratée : pour trouver le mot, on cherchait du silence parfait, qui n'exite pas car il y 
le souffle du micro. Fixé : le mot est repéré par son énergie, comparée au clip lui-même 
(la voix est ~100 fois plus forte que le souffle)

deuxième réussie : le mot est maintenant repéré par son énergie, et un
contrôle automatique vérifie chaque fragment (`scripts/check_fragments.py`,
image de preuve `artifacts/reports/fragments_word_controle.png`).

**2. Plusieures recettes d'entraînement (v19 à v24). Aucune n'a battu le champion.**
Chaque « v » est une recette : un fichier dans `configs/experiment/` qui dit ce
qui change, et un dossier dans `artifacts/runs/eloquence/` avec les résultats.
- **v19** : fragments réparés (première version) → pareil que le champion.
- **v20** : le mot n'est plus jamais coupé au bord + du vrai son de la vidéo
  avant le mot → beaucoup plus de mots attrapés, mais beaucoup trop de fausses
  alertes. Leçon : si les exemples positifs contiennent de la parole qui coule,
  la parole qui coule finit par ressembler au mot.
- **v21** : fragments bien réparés, mais surprise, c'est PIRE. (il n'y a plus 
que des "élo" et des "quences")
- **v22** : on refait le champion à l'identique, pour vérifier que rien n'est
  cassé → tout va bien.
- **v23** : on enlève carrément les fragments → pire aussi. Donc ils sont utiles ;-)
- **v24** : fragments réparés mais plus longs (jusqu'à 70 % du mot) → mieux,
  mais pas assez.
En résumé : les vieux fragments (presque le mot entier) apprenaient probablement au modèle
à éciuter en entier. 
On garde le champion v17. et on arrête cette exploration.

## 2026-08-14 — v24 : les fragments longs confirment la tendance, sans rejoindre le champion

Fragments propres à plafond **70 %** du mot (fracs 30/45/70, pool
`fragments_word70`, f70 mesurés : 460 ms de voix en médiane, max 620 ms —
jamais un mot entier). Banc (52,7 min / 27 occ, champion co-mesuré constant) :
**59.3 % · 28.4 FA/h** @0.8 — critère échoué (FA ≫ 8.2), champion inchangé.

La lecture secondaire pré-déclarée est, elle, validée : FA/h **décroît avec la
longueur des fragments** — 38.7 (plafond 45 %) → 28.4 (70 %) → 6.8 (pool bogué
historique, fragments jusqu'à ~100 %). Et le rappel monte (+11 pts vs champion).
Le « garde-fou » se comporte comme un curseur : plus les fragments frôlent le
mot complet, plus le modèle exige un mot entier et net. Le prolongement naturel
(~85-100 %) EST le masquage temporel, écarté par décision (v19b) — la série
s'arrête donc ici : le pool historique reste dans la recette du champion, ses
propriétés sont maintenant comprises et documentées. Série complète :
v19 → v24, archives du 2026-08-14, courbe FA/h = f(plafond) reproductible.

## 2026-08-14 — Contrôle + ablation : le pipeline est sain, les fragments sont un garde-fou

Après deux runs dégradés d'affilée (v20, v21), doute légitime : « on n'a
pas un autre problème ? ». Deux diagnostics, critères écrits avant, banc commun
(52,7 min / 27 occ, seuil 0.8, champion co-mesuré à 48.1 % · 6.8 constant) :

- **v22_replica17** (v17 strictement à l'identique, ré-entraîné) :
  **55.6 % · 10.2 FA/h** — dans la zone attendue [43-55 %, ≤ 12]. Le pipeline
  reproduit un modèle de la classe du champion : les verdicts v20/v21 sont des
  effets réels, pas une panne d'instrument. Au passage : la réplique fait +7 pts
  de rappel et +3.4 FA/h que le champion — la dispersion du protocole reste
  large (ADR-003), le champion v17 est aussi un tirage heureux.
- **v23_sans_fragments** (v17 sans aucune source fragments) :
  **40.7 % · 17.1 FA/h** — dégradé sur les deux axes, seuil d'alerte pré-déclaré
  (> 15 FA/h) franchi. **Les fragments comptent** : même bogués, ils tenaient
  les FA et le rappel.

Lecture d'ensemble avec v21 (fragments nettoyés à 45 % max : 40.7 % · 38.7) :
retirer les quasi-mots fait pire que retirer tous les fragments — le signal
utile n'était pas « voici des bouts de mot » mais « **tant que le mot n'est pas
complet, tais-toi** », porté par les fragments longs. Prochaine marche logique :
fragments **longs mais jamais complets** (mesure propre par énergie conservée,
plafond remonté vers ~70 %, ex-v24). Preuves : archive
`eloquence_20260814_024226.json`, runs `v22_replica17` et `v23_sans_fragments`.

## 2026-08-14 — v20 : la découpe propre aide le rappel, le contexte réel ruine les FA

Constat mesuré d'abord (une intuition confirmée) : 1830/1882 clips `yt_` ont le mot
collé au **début** du clip et 32/47 clips `moi_` l'ont collé à la **fin** — le
`time_shift` ±100 ms de l'augmentation tronquait donc régulièrement le « é » ou
la fin du mot pendant l'entraînement du champion. Correctif v20 (une variable
vs v19) : positifs `yt_` re-découpés par `word_clips_recut` (mot entier garanti,
jitter 0-200 ms dans la découpe, **contexte réel** de la vidéo avant le mot),
positifs `moi_` ré-ancrés sans troncature (`re_anchor`), `time_shift_ms: 0`.

Verdict au banc (52,7 min / 27 occ), critère pré-déclaré **échoué** :

| Modèle | @0.5 | @0.8 |
|---|---|---|
| v17_stack (champion) | 63.0 % · 21.6 FA/h | 48.1 % · 6.8 FA/h |
| v20_recut_anchor | **85.2 %** · 248 FA/h | 44.4 % · **48.9 FA/h** |

Lecture : le rappel brut explose (+22 pts à 0.5 — l'hypothèse « la troncature
coûtait du rappel » est confirmée), mais les fausses alarmes aussi. Cause
probable : v20 changeait DEUX choses à la fois pour `yt_` — (a) mot jamais
tronqué + jitter, (b) du **vrai flux de parole** avant le mot au lieu de zéros.
Avec (b), « de la parole continue » ressemble désormais aux positifs : le modèle
déclenche sur le flux. L'élection `fa_ambient` l'avait déjà signalé (13.94/h
élu, contre ~7 chez v17). Piste suivante identifiée : un v21 qui isole (a) —
ré-ancrage sur zéros pour `yt_` aussi, sans contexte réel. Preuves : archive
`eloquence_20260814_010424.json`, run `v20_recut_anchor`.

## 2026-08-14 — Fragments : la mesure du mot doit être relative au pic

La première correction des fragments (fractions du mot, seuil absolu `1e-5`)
a été invalidée à l'écoute : sur les enregistrements micro, le **souffle du
micro** (RMS ~0.001) dépasse ce seuil dès l'échantillon 0 — la « durée du mot »
mesurée valait toute la seconde, et un `f45` embarquait 68 % du mot (« loquence »).
Correctif : RMS par trames de 20 ms, mot = zone au-dessus de 10 % du pic, découpe
ancrée sur les bornes du mot. Contrôle indépendant (`scripts/check_fragments.py`,
choix retenu à la place d'un repassage WhisperX) : 1 538 fragments, médiane 31 %
du mot, max 48 %, zéro dépassement — preuve :
`artifacts/reports/fragments_word_controle.png`. Leçon : un seuil d'énergie
absolu est une constante en dur déguisée ; seul un seuil **relatif au signal**
survit au passage du TTS au vrai micro.

## 2026-08-12 — Re-contrôle Metal : ADR-002 reconfirmé, avec chiffres

`tensorflow-metal` 1.2.0 aurait pu avoir corrigé la corruption des gradients
constatée en juillet. Plutôt que de croire l'ADR sur parole, un run de
diagnostic (`v05_metal_check`, recette v03_replica à l'identique,
`use_gpu: true`, critère écrit avant le lancement) a mesuré — et l'occasion a
servi à instrumenter le **temps d'entraînement par candidat** (`fit_s` dans
`metrics.json`, MLflow et `report.md`), désormais tracé pour tous les runs.

Résultat : effondrement identique à juillet. Les 5 seeds divergent (val_loss
0.36–2.5 contre ~0.06 sur CPU, early stopping à 6 epochs partout) et, au banc
en inférence CPU, le modèle élu fait **764 FA/h contre 47** pour v03_replica.
Metal est bien ~4,7× plus rapide par epoch (3 s contre 14 s) — vitesse pour un
modèle inutilisable. Détail daté dans `ADR-002`, section « Re-contrôles ».

**Suite le soir même : cause racine trouvée.** Un fil du forum Apple
([thread 818015](https://developer.apple.com/forums/thread/818015), même config
que la nôtre) pointait le ReLU. Reproduit ici en cinq lignes : **le noyau
fusionné MatMul+BiasAdd+ReLU du plugin Metal n'applique pas le ReLU** (min de
sortie −11.0 au lieu de 0.0, y compris avec la couche `Activation` séparée ;
`tf.nn.relu` seul est correct — c'est bien la fusion). Un réseau sans
non-linéarités explique tout le tableau de juillet d'un coup. Le mystère
d'ADR-002 n'en est plus un : re-contrôle permanent en une commande,
`uv run python scripts/check_metal_relu.py`.

**Un contournement existe — mesuré, pas adopté.** Le bug ne touche que le
motif de fusion : un ReLU « fait main » (`tf.maximum(x, 0)`, mathématiquement
identique) y échappe, tout comme `leaky_relu` et `elu` — mais `relu6` est
cassé aussi, preuve que le problème est systémique aux fusions du plugin. Sur
un problème-jouet non linéaire (cercles), l'entraînement Metal avec `relu`
standard reste au hasard (acc 0.50 : le réseau est devenu linéaire) quand
`tf.maximum(x, 0)` sur Metal converge à l'identique du CPU au chiffre près
(loss 0.0282, acc 0.9937). Réhabiliter Metal pour l'exploration serait donc
*possible* — mais exigerait le protocole complet (recette v03 sur Metal avec
activation contournée, banc CPU, qualité dans la dispersion) et une mise à
jour d'ADR-002.

**Protocole complet exécuté dans la foulée (`v06_metal_leaky`)** : activation
paramétrée dans l'architecture (`cnn_leaky`), recette v03, entraînement Metal.
L'entraînement est redevenu sain (val_loss 0.12–0.16, early stopping normal)
et la vitesse est là (**2 min 39 s les 5 candidats, contre ~17 min sur CPU**) —
mais le banc a tranché : rappel 60.9 % et 83.7 FA/h, contre 73.9 % et 47.1
pour v03_replica. Critère raté sur les deux axes. **Metal reste écarté, la
vitesse ne rachète pas la qualité.** On ne peut pas attribuer la dégradation
(backend ? leaky_relu ? variance ?) — deux facteurs changeaient à la fois,
c'était assumé dans la config ; un jumeau CPU en leaky_relu trancherait si
besoin. Un second essai avec `elu`
(`v07_metal_elu`) a confirmé : entraînement sain et rappel banc même meilleur
(78.3 %), mais 73.3 FA/h (contre 47.1) et un gain de temps réduit à ×2 (elu est
lent sur Metal). Clause d'arrêt pré-déclarée appliquée : deux activations
saines, deux échecs au banc — fin de la série Metal.

Restait l'ambiguïté d'attribution (backend ou activation ?). Le test décisif
(`v08_metal_maxrelu`, le 13 au matin) l'a levée : avec `tf.maximum(x, 0)` —
mathématiquement identique au ReLU — l'entraînement Metal donne des métriques
amont **indiscernables du CPU** (val_loss 0.056, F1 clips 0.9322)… et
**120.3 FA/h au banc, 2,5× la référence**. À fonction identique, seul le
backend restait en cause : condamné. C'est peut-être le résultat le plus
précieux de la série — la démonstration qu'un backend peut produire un modèle
*plausible sur toutes les métriques amont* et pourtant inutilisable, ce
qu'aucun test par clips n'aurait vu (ADR-002 et ADR-004 confirmés d'un même
geste).

La série Metal aura donc donné : la cause racine du bug ReLU, un contournement
démontré, et la preuve au banc — trois fois, dont une à fonction
mathématiquement identique — que ce backend ne produit pas les mêmes modèles.
Dossier refermé définitivement ; retour aux données.

Au passage, la même passe d'écoute a produit les premiers **verdicts humains
sur le banc** (page « Banc streaming » : lecture audio + jugement persisté) :
sur v03_replica au seuil 0.5, 15 FA confirmées (futurs hard negatives), 1 FA
qui était en réalité une bonne détection (vérité WhisperX à corriger), 1 extrait
inexploitable, 6 FN tous confirmés.

---

## 2026-08-13 (nuit) — Exécution de la ROADMAP P0-P2 : porte, banc étendu, découpe

Session autonome d'exécution de la feuille de route. Livré et committé :

- **Porte qualité ADR-007** (`data/gate.py`, config, CLI `data gate` /
  `gate-dir`, page Streamlit « Qualité », 11 tests). Première passe : 5 610
  douteux → recalibrage par POOL (fin chargée : mots isolés seulement ; pools
  de bruit indulgents sur pic/SNR) → 8 664 acceptés · 134 rejetés · 622
  douteux dont 336 positifs à fin chargée. Leçon : des seuils uniques sont
  aveugles au contexte — le critère dépend de la nature du pool.
- **Lot curation.db repassé par la porte** (2 007 raw) : la machine confirme
  96 % des « checked » humains, rejette 14 (durées > 3 s — clips bruts longs),
  et accepterait 4 des 10 rejets humains → les 10 rejets humains sont importés
  comme verdicts « non » (l'oreille voit ce que l'énergie ne voit pas).
- **Banc étendu à 54,9 min** avec 42,4 min de SUMM-RE (3 réunions, vérité
  terrain = alignements mot à mot du dataset, licence CC BY-SA). **Surprise :
  les réunions ordinaires font PLUS de FA que le corpus thématique** (~72/h
  contre ~48/h pour v03) — « thématique = pire cas » est réfuté.
- **Rappel par forme au banc** : « d'éloquence » 3/7 (43 %) contre
  « l'éloquence » 13/15 (87 %) — le point faible est une forme précise, pas
  « le mot » en général.
- **v09_gate** (recette filtrée) : FA/h −39 % à 0.8 mais rappel −8 pts —
  la sévérité de la porte (622 douteux exclus) coûte du rappel ; l'audit
  humain des douteux doit précéder un nouveau run.
- **Re-découpe fin-de-fenêtre** (`word_clips_recut`, dataset `tts500_recut`,
  run v10 en cours) ; **RIR MIT + bruit multi-SNR** implémentés et testés
  (v12 prêt) ; **sweep parole continue** configuré (v11 ×3) ; **studio
  d'enregistrement** écrit (page 7, source `studio`) — test micro en attente.

**Bilan des runs de la nuit (v09-v14, détail au CHANGELOG)** : le sweep a
trouvé la dose — **v11_speech_300 promu champion** (76 % · 33.9 FA/h, FA/h
divisées par deux à rappel intact ; 100 ne fait rien, 500 entame le rappel,
1500 = v04 était l'overdose). v12 (RIR) décroche le rappel record (88 %,
« d'éloquence » 5/7) mais paie en FA/h ; v13 démontre que l'élection
`fa_ambient` seule égale le champion (dispersion ×3,6 entre seeds à recette
identique) ; v14 (cumul dose+RIR) rate son critère de peu (84 % · 51.4).
La re-découpe (v10) a d'abord révélé un bug de vérité terrain (temps dérivés
de −0.1 à −0.9 s → « rappel 100 % » absurde, corrigé par corrélation croisée)
puis, propre, s'est montrée inférieure — hypothèse du couplage avec la machine
à états consignée. Prochain levier déclaré : v14 élu par `fa_ambient`.

---

## 2026-08-13 — Audit des positifs + synthèse des études : la feuille de route

Cinq lectures en parallèle (les quatre études `ETUDE_*.md` + un audit chiffré
des 1 882 positifs YouTube) pour décider de la suite. Deux surprises de
l'audit : le doute sur les fins de clips était minoritaire (~17-25 % de fins
chargées, 62 % de fins proprement zéro-paddées) — mais **93,5 % des clips
attaquent le mot à l'échantillon 0**, l'inverse exact du placement recommandé
unanimement par openWakeWord/microWakeWord/LiveKit (fin du mot près de la fin
de fenêtre, jitter ~200 ms). Et **85 % des positifs sont des formes élidées**
(« l'éloquence » 48,7 %, « d'éloquence » 36,3 %) — on les garde (c'est le
français réel) mais on mesurera le rappel par forme. Bonus : 551 clips curés à
la main dorment dans `curation.db` sans avoir jamais été utilisés, et 5
« Dauphine-Éloquence » (nom propre) polluent les positifs.

Tout est priorisé dans `docs/ROADMAP.md` : P0 = banc étendu + re-découpe des
positifs (placement fin de fenêtre) + curation dormante ; P1 = sweeps dosés
(parole continue 0/100/300/500, RIR/multi-SNR, hard negatives, sélection par
FA/h) ; P2 = studio d'enregistrement guidé ; P3 = benchmarks externes.

---

## 2026-08-12 — Reprise après deux semaines · cap fixé

Décision d'orientation : **poursuivre l'amélioration de la détection dans ce
dépôt**, et non dans `coach-vocal_etape1` (figé, et où `v04_speech_neg` — la
réponse au déficit n°1 — n'existe pas).

Objectif de sortie acté : un **notebook Jupyter de synthèse
(Deep Learning)**, rédigé une fois qu'il y aura une progression à raconter.

---

## 2026-07-28 — Reconstruction MLOps

Migration de `coach-vocal_etape1` vers une architecture pilotée par configuration.
Motivations : douze scripts partageant 80 % de code, `get_spectrogram()` dupliqué
six fois, aucune traçabilité rejouable des données. Voir ADR-001 et ADR-005.

Ce qui est repris tel quel : recettes de données, protocole multi-candidats,
logique du détecteur, banc streaming et sa vérité terrain. Ce qui est ajouté :
validation des configs, registre de sources et d'architectures, suivi MLflow,
audit qualité, API, interface, tests, CI, Docker.

Point de contrôle prévu : `v03_replica` doit retrouver les ordres de grandeur du
v03 historique. Sans lui, on ne saurait pas distinguer un effet de recette d'un
effet de migration.

---

## 2026-07-21 — Le banc streaming, et ce qu'il a démoli

**Premier banc : faux.** Rappel proche de zéro pour tous les modèles, 200-360
FA/h. Diagnostic : une sonde à ±5 s montrait que les modèles *voyaient* bien le
mot (probabilité 58-100 %). Le problème était la vérité terrain — les sous-titres
automatiques dérivent de ±1 s, et surtout `yt-dlp --download-sections` découpe aux
images-clés, ce qui ajoute jusqu'à 10 s au début des fichiers. Corrigé en passant
aux alignements WhisperX de `discovery.db`, les sous-titres ne servant plus qu'à
définir des zones d'incertitude.

**Deuxième banc : instructif.** Le test par clips classe mal les modèles (voir
ADR-004). Le champion réel rate encore une occurrence sur trois et déclenche ~50
fois par heure sur de la parole continue. Cause identifiée : le modèle n'a jamais
vu de parole continue à l'entraînement — uniquement des mots isolés et des
extraits d'1 s tronqués.

## 2026-07-21 — Rétractation : les 8 clips guidés

Un run officiel a donné une F1 en baisse. Hypothèse : huit clips guidés
enregistrés très bas (pic 0.06-0.11), boostés ×10, tiraient le modèle vers un
régime acoustique atypique. Trois comparaisons appariées semblaient confirmer.

**Puis on les a retirés : le résultat était encore pire.** La cause dominante
n'était pas les clips, c'était la variance d'entraînement sur CPU (±0.03-0.06 de
F1 à seed identique). Conclusion rétractée, et protocole multi-candidats adopté
(ADR-003). C'est l'épisode qui a le plus changé la méthode de travail du projet.

## 2026-07-21 — Dose de positifs synthétiques

Sweep 0 / 100 / 500 / 2000 clips TTS. Le gain **n'est pas monotone** : optimum
vers 500 (environ deux fois les positifs réels), effondrement à 2000 où le
synthétique noie le réel. Gain confirmé sur trois seeds appariés, puis au banc
streaming (rappel 65.8 → 68.4 %, FA/h 56.5 → 52.7).

## 2026-07-21 — Piper : le babil des voix de livres audio

`fr_FR-mls` (multi-locuteur, entraînée sur des livres audio) produit 1,5 à 3,5 s
de charabia sur un mot isolé — elle attend une phrase. `siwis` et `upmc` donnent
0,6-0,8 s propres. Deux parades retenues : un point final dans le texte
(« éloquence. »), qui stabilise la prosodie, et un contrôle de durée après
rognage avec régénération. `mls` a finalement été écartée.

## 2026-07-19/20 — Le modèle avait appris un débit, pas un mot

Vingt « éloquence » prononcés au tempo réel : quinze ratés. Les mêmes, ralentis
de 8-15 % : vingt sur vingt reconnus. Le modèle avait appris la vitesse
d'élocution des clips d'entraînement. Corrigé par l'augmentation de vitesse
×0.85–1.15, qui a fait passer le contrôle guidé de 6/20 à 20/20.

## 2026-07-12 — La loss qui explose

Des heures perdues à soupçonner le taux d'apprentissage, la normalisation, le
déséquilibre des classes. C'était `tensorflow-metal` qui corrompait les
gradients. Voir ADR-002. Leçon retenue : avant de mettre en cause les
hyperparamètres, mettre en cause le backend.

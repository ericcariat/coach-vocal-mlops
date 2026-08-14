# Journal des explorations

Le `CHANGELOG` raconte les runs officiels. Ce journal raconte le reste : les
essais, les diagnostics, et surtout **les impasses**. Une conclusion rétractée
vaut plus qu'un résultat lisse — elle dit comment on a appris à ne plus se tromper.

---

## 2026-08-14 — Goal openWakeWord, H1 : l'arme contre les cousins (critère AVANT le run)

Objectif fixé par l'auteur : tête sur extracteur Google gelé, d'abord NOS
enregistrements réels, l'océan seulement si nécessaire, viser mieux que le
champion et ~1 FA/h. Constat préalable : les fichiers océan ACAV (11/125/2000 h)
ne sont plus sur le disque — et le round 5 avait montré que 3,3 h de français
pesaient plus que 125 h d'océan : H1 se joue donc SANS océan.

Baseline mesurée avant (nouveau `scripts/eval_oww_cousins.py`, tête round 5
seed 42) : les **cousins moi_ déclenchent à 16 % @0.95** (7 % @0.99), contre
2-5 % pour les autres négatifs — la faiblesse vue au micro est maintenant un
chiffre. Levier ajouté : `--adv-weight` (les 122 adversariaux — 45 cousins
moi_, 54 hard negatives du banc, 23 guidés — pesaient 1 parmi ~6 900).

**H1** : contexte réel + 12 000 fenêtres françaises ×20 + adversariaux ×30,
3 seeds. Critère écrit avant le banc :
  - succès : à UN seuil, rappel ≥ 60 % ET FA/h ≤ 6.8 (battre le champion sur
    les deux axes) ; idéal : rappel ≥ 70 % à FA/h ≤ 1.5 ;
  - cousins : taux de déclenchement des cousins moi_ ≤ 5 % au seuil retenu
    (baseline 16 %), sans perdre plus de 5 pts sur les positifs moi_ ;
  - la promotion resterait suspendue au test guidé au micro de l'auteur (ADR-008 à
    écrire — hors périmètre docs autorisé, à valider avec lui).

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
(~85-100 %) EST le masquage temporel qu'l'auteur a écarté (v19b) — la série
s'arrête donc ici : le pool historique reste dans la recette du champion, ses
propriétés sont maintenant comprises et documentées. Série complète :
v19 → v24, archives du 2026-08-14, courbe FA/h = f(plafond) reproductible.

## 2026-08-14 — Contrôle + ablation : le pipeline est sain, les fragments sont un garde-fou

Après deux runs dégradés d'affilée (v20, v21), doute légitime de l'auteur : « on n'a
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

Constat mesuré d'abord (intuition de l'auteur) : 1830/1882 clips `yt_` ont le mot
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
a été invalidée à l'oreille par l'auteur : sur ses enregistrements, le **souffle du
micro** (RMS ~0.001) dépasse ce seuil dès l'échantillon 0 — la « durée du mot »
mesurée valait toute la seconde, et un `f45` embarquait 68 % du mot (« loquence »).
Correctif : RMS par trames de 20 ms, mot = zone au-dessus de 10 % du pic, découpe
ancrée sur les bornes du mot. Contrôle indépendant (`scripts/check_fragments.py`,
idée de l'auteur à la place d'un repassage WhisperX) : 1 538 fragments, médiane 31 %
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

Objectif de sortie acté : un **notebook Jupyter pour la synthèse
bloc 5 (Deep Learning)**, rédigé une fois qu'il y aura une progression à
raconter. Voir `docs/NOTEBOOK.md` pour ce que ça implique dès maintenant.

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

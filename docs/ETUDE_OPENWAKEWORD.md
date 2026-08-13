# Étude comparative — OpenWakeWord et le wake word « éloquence »

**Projet :** Eloquence — détection du mot d'activation « éloquence »  
**Contexte :** synthèse, bloc 05 Deep Learning  
**Date de l'étude :** 13 août 2026  
**Dépôt étudié :** [dscripka/openWakeWord](https://github.com/dscripka/openWakeWord)  
**Révision observée :** branche `main`, consultation du code et de la documentation disponibles à la date de l'étude

## 1. Objet de l'étude

Cette étude examine la manière dont OpenWakeWord :

- construit les données positives d'un wake word personnalisé ;
- exploite les enregistrements réels ;
- produit ses caractéristiques acoustiques ;
- entraîne un classifieur compact ;
- prend une décision dans un flux audio ;
- limite les faux positifs ;
- peut être comparé au détecteur « éloquence » développé pour le projet Eloquence.

L'objectif n'est pas de remplacer automatiquement le modèle Eloquence par OpenWakeWord. Il s'agit d'en faire un benchmark de transfert d'apprentissage, d'identifier les idées réutilisables et de construire une comparaison défendable devant le jury synthèse.

## 2. Résumé exécutif

OpenWakeWord adopte une stratégie très différente de celle d'un petit modèle entraîné uniquement sur des enregistrements personnels.

Son modèle de base ne demande pas à l'utilisateur d'enregistrer dix ou cinquante répétitions du wake word. Le pipeline officiel génère plutôt plusieurs milliers, voire plusieurs centaines de milliers, d'exemples positifs par synthèse vocale. Il les mélange à des bruits, de la musique et des réponses impulsionnelles de pièces, puis les oppose à une quantité très importante de parole négative.

Le signal n'est pas classé directement à partir du spectrogramme par le petit modèle final. OpenWakeWord utilise :

1. un frontend log-mel préentraîné et figé ;
2. un modèle d'embeddings vocaux préentraîné et figé ;
3. une petite tête DNN ou RNN propre au wake word.

Cette architecture apporte un avantage majeur lorsque les données spécifiques sont limitées : la représentation acoustique a déjà été apprise sur un corpus considérable. En contrepartie, les composants et recettes officiels sont principalement conçus et évalués pour l'anglais. Pour « éloquence », la prononciation française, les voix TTS, les confusions phonétiques et le banc de test réel doivent donc être contrôlés localement.

Pour Eloquence, la meilleure utilisation d'OpenWakeWord est celle d'un modèle concurrent mesuré sur exactement le même banc streaming que le CNN actuel. Cela permettrait de comparer une approche end-to-end sur log-mel à une approche par transfert d'apprentissage, sans abandonner les acquis du projet.

## 3. Comment OpenWakeWord « enregistre » le wake word

### 3.1 Le modèle de base ne repose pas sur une collecte manuelle

Contrairement à ViolaWake, OpenWakeWord ne fournit pas, dans son pipeline de base, une interface demandant à l'utilisateur d'enregistrer une série de répétitions. L'entraînement automatisé part d'une expression écrite et génère les fichiers positifs avec Piper.

Pour un wake word tel que « éloquence », le processus visé est donc approximativement :

```text
Texte « éloquence »
        ↓
voix TTS et variations de synthèse
        ↓
milliers de WAV positifs
        ↓
bruit + musique + réverbération + transformations
        ↓
embeddings OpenWakeWord
        ↓
classifieur du wake word
```

La documentation recommande au minimum plusieurs milliers de positifs synthétiques. Les modèles officiels de production ont été entraînés à une tout autre échelle : environ 100 000 positifs synthétiques sont indiqués pour « Alexa » et environ 200 000 pour « Hey Jarvis ».

Sources : [documentation sur la génération synthétique](https://github.com/dscripka/openWakeWord/blob/main/docs/synthetic_data_generation.md), [modèle Alexa](https://github.com/dscripka/openWakeWord/blob/main/docs/models/alexa.md) et [modèle Hey Jarvis](https://github.com/dscripka/openWakeWord/blob/main/docs/models/hey_jarvis.md)

### 3.2 Les enregistrements humains restent indispensables à l'évaluation

Le fait d'utiliser du TTS pour l'entraînement ne dispense pas de tester de vraies voix. OpenWakeWord recommande d'évaluer les faux rejets à partir de prises réalistes, enregistrées à plusieurs distances et dans plusieurs conditions acoustiques.

La documentation du modèle « Hey Mycroft » décrit par exemple 51 prises manuelles réalisées à des distances d'environ 0,9 à 9 mètres, avec différents bruits domestiques tels qu'un ventilateur, la climatisation ou un lave-vaisselle.

Pour Eloquence, les répétitions réelles déjà collectées doivent d'abord rester dans un ensemble d'évaluation disjoint. Les injecter immédiatement dans l'entraînement ferait perdre la possibilité de mesurer honnêtement la généralisation du modèle synthétique.

Source : [documentation des modèles OpenWakeWord](https://github.com/dscripka/openWakeWord/tree/main/docs/models)

### 3.3 Enregistrements pour un vérificateur personnalisé

OpenWakeWord propose également un vérificateur optionnel, spécifique à une personne. Ce second modèle intervient après le détecteur général afin de confirmer que l'activation ressemble aux exemples de l'utilisateur cible.

La documentation demande au minimum :

- trois exemples positifs par personne cible ;
- environ dix secondes de parole sans wake word par personne ;
- éventuellement environ cinq secondes de bruit typique ou de faux déclenchements.

Ce vérificateur est une régression logistique entraînée sur les caractéristiques OpenWakeWord. Il peut réduire certains faux positifs et personnaliser l'expérience, mais ce n'est pas un système complet et certifié de reconnaissance du locuteur. Il ne doit pas être présenté comme une authentification biométrique.

Source : [Custom verifier models](https://github.com/dscripka/openWakeWord/blob/main/docs/custom_verifier_models.md)

## 4. Construction du dataset

### 4.1 Positifs synthétiques

Le générateur officiel s'appuie sur [Piper Sample Generator](https://github.com/rhasspy/piper-sample-generator). Le pipeline fait varier les voix et certains paramètres de synthèse, puis produit des variantes de durée et de bruit. La recette de code observée emploie notamment plusieurs facteurs de vitesse ou de longueur.

Il faut distinguer deux possibilités de Piper :

- utiliser des voix Piper ordinaires, parmi lesquelles il existe des voix françaises ;
- utiliser le générateur spécial qui mélange des embeddings de locuteurs pour créer un grand nombre de voix artificielles.

La seconde possibilité est annoncée comme centrée sur l'anglais. Pour « éloquence », il est donc nécessaire d'écouter et de contrôler les fichiers produits. Un grand nombre de fichiers avec une mauvaise réalisation du son initial /e/, des syllabes ou de l'accentuation ne constituerait pas un bon corpus, même si le volume paraît impressionnant.

### 4.2 Négatifs adversariaux

Le pipeline peut produire des textes proches du wake word afin de créer des négatifs difficiles. Toutefois, les outils phonétiques observés s'appuient sur des ressources anglaises, notamment CMU Pronouncing Dictionary et des composants de phonémisation associés.

Cette génération ne peut pas être transposée telle quelle à « éloquence ». Il vaut mieux fournir explicitement des négatifs francophones tels que :

- mots ou expressions acoustiquement proches ;
- fragments contenant « élo- », « -quence » ou une cadence similaire ;
- phrases contenant le mot dans un contexte où l'application ne doit pas s'activer, si ce cas d'usage est retenu ;
- noms propres, conversations et voix lointaines susceptibles de provoquer des confusions ;
- faux positifs réellement récoltés par le modèle en fonctionnement.

Le choix exact dépend de la définition fonctionnelle : l'application doit-elle s'activer à chaque occurrence isolée de « éloquence », ou seulement lorsque le mot est prononcé comme une commande ? Cette distinction doit être fixée dans le protocole d'évaluation.

### 4.3 Négatifs génériques à grande échelle

Les modèles préentraînés officiels utilisent environ 31 000 heures de négatifs, réparties approximativement entre :

- ACAV100M ;
- Common Voice multilingue ;
- podcasts ;
- FMA pour la musique ;
- bruits et variantes réverbérées ;
- négatifs adversariaux.

La recette automatisée du notebook est beaucoup plus légère. Elle peut télécharger des caractéristiques négatives précalculées et quelques corpus de bruit, musique et réponses impulsionnelles. Elle constitue un point de départ de démonstration, pas l'équivalent exact de la recette ayant produit les modèles officiels.

### 4.4 Augmentations audio

Le pipeline d'augmentation contient notamment :

- égalisation paramétrique ;
- distorsion `tanh` ;
- variation de hauteur jusqu'à environ trois demi-tons ;
- filtre coupe-bande ;
- bruit coloré ;
- mélange de bruits de fond à plusieurs rapports signal/bruit ;
- variation de gain ;
- convolution avec des réponses impulsionnelles de pièces.

Le mot est aussi positionné dans une fenêtre fixe de façon à ce que sa fin se trouve près de la fin de la fenêtre. Ce détail est important : un modèle streaming doit apprendre à produire son score au moment où le mot vient d'être terminé, pas longtemps avant ou après.

Sources : [data.py](https://github.com/dscripka/openWakeWord/blob/main/openwakeword/data.py) et [train.py](https://github.com/dscripka/openWakeWord/blob/main/openwakeword/train.py)

## 5. Architecture d'OpenWakeWord

### 5.1 Frontend acoustique gelé

Le flux de calcul est le suivant :

```text
PCM 16 kHz mono
      ↓
melspectrogramme ONNX ou LiteRT/TFLite
      ↓
modèle d'embeddings vocaux préentraîné et gelé
      ↓
vecteurs de 96 valeurs toutes les 80 ms
      ↓
DNN ou RNN du wake word
      ↓
score de détection
```

Le modèle d'embeddings reprend un réseau publié par Google et initialement distribué via TensorFlow Hub. Le projet le réimplémente pour une inférence plus facilement déployable avec ONNX ou LiteRT/TFLite.

L'intérêt de ce montage est que le petit classifieur n'a pas à réapprendre, à partir de zéro, toutes les structures de la parole. Il travaille sur une représentation de 96 dimensions déjà porteuse d'informations phonétiques et acoustiques.

Sources : [README](https://github.com/dscripka/openWakeWord/blob/main/README.md), [utils.py](https://github.com/dscripka/openWakeWord/blob/main/openwakeword/utils.py) et [article sur le modèle d'embeddings](https://arxiv.org/abs/2002.01322)

### 5.2 Classifieur DNN

Le classifieur le plus simple aplatit plusieurs embeddings consécutifs puis applique des couches entièrement connectées avec normalisation, activation ReLU et sortie sigmoïde.

Un exemple documenté pour « Alexa » reçoit 16 embeddings de 96 valeurs, soit 1 536 valeurs en entrée, et possède environ 103 000 paramètres. Cette tête est beaucoup plus petite que le CNN Eloquence actuel, dont une grande couche dense concentre plusieurs millions de paramètres.

### 5.3 Variante RNN

OpenWakeWord propose aussi une tête récurrente basée sur un LSTM bidirectionnel à deux couches. Cette variante conserve explicitement la structure temporelle de la séquence d'embeddings.

Le meilleur choix entre DNN et RNN ne doit pas être décidé uniquement sur l'accuracy de validation. Le nombre de faux positifs par heure, le taux de faux rejets, la latence et la stabilité en streaming sont plus importants pour le cas d'usage.

## 6. Entraînement

### 6.1 Étapes du notebook automatisé

Le [notebook d'entraînement automatique](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb) enchaîne en substance :

1. installation des dépendances ;
2. génération des positifs TTS ;
3. génération de négatifs adversariaux ;
4. téléchargement de bruits, musiques, réponses impulsionnelles et caractéristiques négatives ;
5. augmentation des fichiers ;
6. extraction des embeddings ;
7. entraînement PyTorch du classifieur ;
8. sélection ou moyenne de checkpoints ;
9. export ONNX ;
10. conversion optionnelle vers LiteRT/TFLite.

Le notebook présente une expérience Colab réalisable en moins d'une heure comme un modèle de départ. Il prévient cependant que les performances peuvent être insuffisantes pour certains déploiements.

### 6.2 Optimisation du classifieur

Le code utilise notamment :

- une perte binaire ;
- Adam ;
- un planning du taux d'apprentissage avec warm-up et cycles ;
- une pondération évolutive des négatifs ;
- un rééchantillonnage des exemples difficiles ;
- une sélection fondée sur une validation positive et sur des heures de négatifs.

Cette dernière idée est importante pour Eloquence : un checkpoint ayant la meilleure perte moyenne n'est pas nécessairement celui qui produit le moins de déclenchements intempestifs dans plusieurs heures d'audio réel.

### 6.3 Écart entre notebook de démonstration et recette de production

Le notebook contient une incohérence mineure à garder en tête : son texte évoque 5 000 positifs et négatifs dans une section, alors que certaines cellules de la recette observée configurent 1 000 échantillons d'entraînement et 1 000 de validation. Cela confirme qu'il faut noter les paramètres réellement exécutés, et pas seulement recopier les commentaires du notebook.

Pour la synthèse, chaque run devrait donc conserver :

- la révision Git d'OpenWakeWord ;
- les versions des dépendances ;
- le nombre réel de fichiers générés et retenus ;
- les corpus négatifs utilisés ;
- la graine aléatoire ;
- le temps de génération TTS ;
- le temps d'augmentation ;
- le temps d'extraction des embeddings ;
- le temps d'entraînement du classifieur ;
- le temps total et la machine utilisée.

## 7. Inférence et décision streaming

### 7.1 Format et cadence

L'entrée attendue est du PCM signé 16 bits, mono, à 16 kHz. Le flux peut être fourni par blocs de longueur variable, mais des multiples de 80 ms, soit 1 280 échantillons, sont recommandés. Un score est produit toutes les 80 ms.

Des blocs plus longs réduisent le coût des appels, mais augmentent potentiellement la latence. Pour une comparaison équitable, il faut documenter la taille des blocs de chaque modèle.

Source : [model.py](https://github.com/dscripka/openWakeWord/blob/main/openwakeword/model.py)

### 7.2 Seuil, patience et cooldown

Le seuil fréquemment présenté pour les modèles officiels est autour de 0,5, mais il n'est pas universel. Il doit être recalibré sur l'environnement cible.

OpenWakeWord permet :

- une `patience`, c'est-à-dire plusieurs scores consécutifs au-dessus du seuil ;
- un temps de `debounce`, assimilable à un cooldown après activation.

Dans l'API observée, ces deux mécanismes ne sont pas combinables au sein du même appel. Le pipeline Eloquence applique déjà plusieurs fenêtres positives consécutives et un cooldown. Pour comparer les modèles, il est préférable de récupérer les scores OpenWakeWord et d'appliquer extérieurement la même machine à états aux deux modèles.

### 7.3 VAD et réduction de bruit

Une VAD Silero optionnelle peut invalider un score si aucune parole suffisante n'a été détectée dans la période précédente. Une réduction de bruit Speex est également proposée, mais son intégration officielle est limitée à Linux.

La VAD peut réduire les déclenchements sur des sons non vocaux. Elle ne résout pas les confusions avec de la parole réelle : un mot voisin, une télévision ou une conversation passent généralement le filtre de voix.

### 7.4 Vérificateur en second étage

Le vérificateur personnalisé est appelé uniquement lorsque le modèle de base dépasse un seuil. Son score remplace alors celui du modèle général. Cette architecture à deux étages peut être utile si Eloquence doit surtout répondre à son utilisateur principal.

Elle introduit cependant une autre question produit : l'application doit-elle fonctionner pour n'importe quel orateur, ou être personnalisée après une courte phase d'enrôlement ? Les deux scénarios doivent être évalués séparément.

### 7.5 Test de fichiers comme un vrai flux

La méthode `predict_clip` ajoute du silence au début et à la fin du fichier puis le traite par blocs successifs. Cette simulation est plus proche d'un usage réel qu'une seule prédiction globale sur un WAV pré-découpé.

Pour Eloquence, le test décisif reste néanmoins un long flux continu comprenant parole, silences, bruit, musique, mots proches et vraies occurrences du wake word.

## 8. Évaluation annoncée par OpenWakeWord

OpenWakeWord cite comme objectifs généraux moins de 5 % de faux rejets et moins de 0,5 fausse acceptation par heure après réglage du seuil. Ces valeurs ne constituent pas une garantie pour un nouveau modèle français.

Les faux positifs des modèles officiels sont notamment évalués sur environ 5,5 heures de l'Amazon Dinner Party Corpus, qui contient parole lointaine, musique et bruit. Les positifs sont évalués avec des prises réalistes ou des fichiers propres mélangés à du bruit et à des réponses impulsionnelles.

Le projet publie aussi des comparaisons avec Porcupine, tout en avertissant que le faible nombre d'échantillons et les différences de préparation imposent une lecture prudente. Cette prudence est à conserver dans le rapport synthèse : une comparaison n'a de valeur que si les modèles reçoivent le même flux et si les événements sont comptés avec les mêmes règles.

## 9. Comparaison avec le projet Eloquence

| Dimension | Eloquence actuel | OpenWakeWord natif |
|---|---|---|
| Signal | WAV 16 kHz mono | PCM 16 kHz mono |
| Représentation | log-mel appris avec le classifieur | log-mel et embeddings préentraînés gelés |
| Entrée du classifieur | fenêtre d'environ 1 s, 124 × 40 | séquence d'embeddings de 96 valeurs, cadence 80 ms |
| Tête | CNN Keras avec grande couche dense | petit DNN ou RNN PyTorch |
| Taille typique | plusieurs millions de paramètres | ordre de 100 000 paramètres pour certains modèles |
| Positifs | voix réelles francophones + Piper | surtout TTS, plusieurs milliers à centaines de milliers |
| Négatifs | corpus français et hard negatives ciblés | très grands corpus génériques + adversariaux |
| Décision | seuil élevé, fenêtres consécutives, pic, cooldown | seuil, patience ou debounce, VAD et vérificateur optionnels |
| Export | modèle Keras | ONNX, LiteRT/TFLite |
| Point fort | données et protocole ciblés « éloquence » | transfert d'apprentissage et forte compacité |
| Risque principal | surapprentissage et grande tête dense | décalage anglais-français et dépendance au TTS |

OpenWakeWord et ViolaWake ne sont pas deux solutions totalement indépendantes. ViolaWake réutilise précisément le frontend et les embeddings gelés d'OpenWakeWord, puis ajoute sa propre tête temporelle et sa politique de décision. Une comparaison utile comporte donc trois branches :

```text
Eloquence : log-mel → CNN end-to-end
OpenWakeWord : frontend gelé → DNN/RNN natif
ViolaWake : frontend OpenWakeWord gelé → TemporalCNN
```

Cette présentation est particulièrement intéressante pour la synthèse, car elle montre trois choix d'architecture, de volume de données et de transfert d'apprentissage.

## 10. Limites particulières pour le français « éloquence »

### 10.1 Modèles officiels principalement anglophones

Le README précise que les modèles préentraînés fournis sont en anglais. Le frontend, la VAD et l'inférence ne sont pas intrinsèquement limités à cette langue, mais l'efficacité des embeddings sur d'autres langues n'est pas garantie par les mêmes évaluations.

Dans une discussion officielle, le mainteneur identifie deux limites importantes pour les autres langues : la disponibilité d'un TTS multi-locuteurs de qualité et l'incertitude sur la représentation produite par le modèle d'embeddings hors anglais.

Source : [discussion « Support for wakewords in other languages »](https://github.com/dscripka/openWakeWord/discussions/52)

### 10.2 Mot accentué et normalisation des noms

Le texte TTS doit conserver la graphie française correcte « éloquence ». En revanche, les noms de dossiers, de modèles et les identifiants techniques pourront exiger une version ASCII telle que `eloquence`. Il faut éviter de confondre l'identifiant du modèle avec le texte réellement synthétisé.

### 10.3 Validation par des locuteurs natifs

Avant l'entraînement, un échantillon aléatoire de positifs TTS doit être écouté et annoté : prononciation correcte, mot complet, absence de clic, durée, niveau, voix et variété. Cette vérification est plus importante que l'obtention rapide d'un grand nombre de fichiers.

Après l'entraînement, l'évaluation doit contenir des locuteurs non vus, plusieurs accents francophones, des voix masculines et féminines, plusieurs microphones, différentes distances et des bruits compatibles avec l'usage d'Eloquence.

## 11. Apple Silicon, Metal et temps d'entraînement

### 11.1 OpenWakeWord ne réactive pas TensorFlow Metal pour le classifieur

La tête spécifique est entraînée avec PyTorch. Le code officiel choisit CUDA lorsqu'il est disponible, sinon le CPU. Il ne sélectionne pas le backend Apple MPS dans la version observée.

Par conséquent, essayer OpenWakeWord permet d'éviter le chemin TensorFlow Metal qui a posé problème au modèle Eloquence, mais ne fournit pas automatiquement une accélération Metal officielle sur Mac. Sur Apple Silicon, le pipeline officiel retombera principalement sur le CPU, sauf adaptation.

### 11.2 Dépendances anciennes dans la recette de conversion

La recette complète conserve plusieurs dépendances historiques, notamment un ancien TensorFlow CPU et une chaîne de conversion ONNX/TensorFlow. Le runtime moderne s'appuie sur ONNX Runtime ou LiteRT, mais reproduire le notebook complet localement peut provoquer des conflits de versions.

Il est préférable d'isoler l'expérience dans un environnement dédié et de noter les écarts éventuels avec le notebook. L'objectif expérimental n'exige pas de modifier l'environnement fiable du modèle Eloquence.

### 11.3 Possibilité communautaire MPS

Le projet communautaire [TaterTotterson/openWakeWord-Trainer](https://github.com/TaterTotterson/openWakeWord-Trainer) annonce une prise en charge Apple Silicon/MPS et une interface d'entraînement. Il ne s'agit pas du pipeline officiel : ses correctifs, dépendances, reproductibilité et licences doivent être vérifiés séparément avant de l'utiliser comme résultat de référence.

Pour une première mesure défendable, Colab/CUDA ou le CPU local officiel est préférable. Une seconde expérience MPS peut ensuite mesurer l'accélération, à condition que les sorties soient évaluées avec le même banc CPU indépendant.

### 11.4 Mesurer autre chose que la seule boucle PyTorch

La petite tête peut s'entraîner rapidement alors que la génération TTS, l'augmentation et l'extraction des embeddings dominent le temps total. Le rapport doit donc séparer :

| Phase | Mesure recommandée |
|---|---|
| Génération TTS | durée, nombre de fichiers réussis/rejetés |
| Augmentation | durée, nombre de variantes |
| Extraction frontend/embeddings | durée et débit audio traité |
| Entraînement de la tête | durée, nombre d'étapes, device |
| Export | durée, formats produits |
| Total | temps mur complet et configuration matérielle |

## 12. Protocole recommandé pour « éloquence »

### Étape 1 — Figer le banc de comparaison

Conserver un banc totalement disjoint comprenant :

- occurrences positives réelles ;
- parole française sans wake word ;
- mots proches ;
- musique, télévision et podcasts ;
- bruits non vocaux ;
- longues séquences continues ;
- conditions proches et lointaines.

Les métriques principales sont le rappel événementiel, les faux positifs par heure, la précision événementielle et la latence de déclenchement.

### Étape 2 — Construire un modèle OpenWakeWord de référence

Générer plusieurs milliers de « éloquence » avec plusieurs voix françaises Piper. Écouter un échantillon avant de poursuivre. Ajouter les négatifs français déjà établis par le projet et des négatifs adversariaux écrits manuellement.

La quantité exacte constitue une variable expérimentale. Un premier modèle modeste permet de vérifier le pipeline ; un second, plus volumineux, mesure l'effet d'échelle.

### Étape 3 — Ne pas contaminer le test

Ne pas réutiliser les prises du banc final pour l'entraînement ni pour le choix du seuil. Créer un ensemble de calibration séparé pour régler le seuil, la patience et le cooldown.

### Étape 4 — Uniformiser la décision

Exporter les scores bruts des modèles et leur appliquer la même logique événementielle : même nombre de confirmations, même cooldown et mêmes règles de fusion des fenêtres. Comparer aussi la configuration native de chaque solution, mais dans une expérience distincte.

### Étape 5 — Balayer les seuils

Évaluer plusieurs seuils afin de produire une courbe rappel contre faux positifs par heure. Comparer seulement les modèles à niveau de faux positifs équivalent, ou à niveau de rappel équivalent.

### Étape 6 — Tester les garde-fous

Mesurer séparément l'effet :

- de la VAD ;
- des confirmations consécutives ;
- du cooldown ;
- du vérificateur personnalisé ;
- des hard negatives collectés après un premier déploiement.

Cette ablation indique au jury quelle amélioration vient du réseau et laquelle vient de la politique de décision.

### Étape 7 — Tracer le coût

Noter pour chaque run le temps par phase, le matériel, la mémoire, la taille du modèle, le temps d'inférence et la latence. La comparaison « meilleure qualité » doit être accompagnée de son coût de calcul et de préparation des données.

## 13. Intérêt pour le dossier synthèse

OpenWakeWord permet de présenter plusieurs compétences attendues dans un projet Deep Learning :

- formulation d'un problème de classification audio streaming ;
- préparation de données positives et négatives ;
- augmentation et simulation acoustique ;
- transfert d'apprentissage avec backbone gelé ;
- comparaison DNN, RNN et CNN ;
- gestion du déséquilibre et des exemples difficiles ;
- choix de métriques métier plutôt que de la seule accuracy ;
- expérimentation reproductible ;
- analyse des limites linguistiques et des licences ;
- déploiement ONNX ou LiteRT/TFLite.

Une formulation claire devant le jury pourrait être :

> Le CNN Eloquence apprend directement sur des log-mels et bénéficie de données françaises ciblées. OpenWakeWord réutilise une représentation acoustique préentraînée et réduit fortement la taille de la tête, mais impose de vérifier le transfert vers le français. J'ai donc comparé les deux approches sur un même banc streaming, en contrôlant le seuil, la logique événementielle, les faux positifs par heure, les faux rejets, la latence et le coût d'entraînement.

Ce discours montre une démarche expérimentale et ne suppose pas qu'une bibliothèque externe soit supérieure par principe.

## 14. Licences et réutilisation

Le code OpenWakeWord est distribué sous licence Apache 2.0. Les modèles préentraînés inclus sont toutefois annoncés sous licence CC BY-NC-SA 4.0 en raison des conditions ou incertitudes liées à certains jeux de données d'entraînement.

Cette distinction est importante : utiliser le code pour entraîner un modèle personnalisé n'accorde pas automatiquement les mêmes droits que redistribuer un modèle officiel. Pour une future commercialisation d'Eloquence, il faudra documenter séparément :

- la licence de chaque voix TTS ;
- les licences des positifs et négatifs ;
- les conditions des corpus de bruit, musique et parole ;
- la licence du modèle produit ;
- les obligations d'attribution ou de partage à l'identique.

L'utilisation dans un prototype de synthèse ne supprime pas cette analyse pour le produit futur.

## 15. Conclusion

OpenWakeWord apporte une réponse robuste au manque de données spécifiques grâce à un frontend vocal préentraîné, une tête légère et une génération synthétique massive. Sa philosophie n'est pas d'enregistrer quelques répétitions puis d'entraîner directement un modèle : elle consiste à synthétiser une grande diversité, exploiter d'énormes négatifs et réserver les vraies voix à la validation ou à une personnalisation secondaire.

Pour « éloquence », ses principaux risques sont la qualité du TTS français, les négatifs adversariaux anglais du pipeline officiel et l'incertitude du transfert des embeddings vers le français. Ses principaux avantages sont la compacité, le streaming toutes les 80 ms, l'export portable et le transfert d'apprentissage.

La recommandation est donc de conserver le modèle Eloquence actuel comme référence, d'entraîner OpenWakeWord comme benchmark indépendant et d'évaluer les deux sur le même banc. Le résultat le plus intéressant pour la synthèse ne sera pas seulement le nom du gagnant, mais l'explication mesurée des compromis entre données réelles ciblées, synthèse vocale, préentraînement, faux positifs, latence et coût d'entraînement.

## 16. Références principales

- [Dépôt OpenWakeWord](https://github.com/dscripka/openWakeWord)
- [README et architecture](https://github.com/dscripka/openWakeWord/blob/main/README.md)
- [Notebook d'entraînement automatique](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb)
- [Tutoriel d'entraînement détaillé](https://github.com/dscripka/openWakeWord/blob/main/notebooks/training_models.ipynb)
- [Code d'inférence streaming](https://github.com/dscripka/openWakeWord/blob/main/openwakeword/model.py)
- [Frontend et embeddings](https://github.com/dscripka/openWakeWord/blob/main/openwakeword/utils.py)
- [Code d'entraînement](https://github.com/dscripka/openWakeWord/blob/main/openwakeword/train.py)
- [Pipeline de données et augmentations](https://github.com/dscripka/openWakeWord/blob/main/openwakeword/data.py)
- [Génération synthétique](https://github.com/dscripka/openWakeWord/blob/main/docs/synthetic_data_generation.md)
- [Vérificateurs personnalisés](https://github.com/dscripka/openWakeWord/blob/main/docs/custom_verifier_models.md)
- [Discussion officielle sur les autres langues](https://github.com/dscripka/openWakeWord/discussions/52)
- [Piper Sample Generator](https://github.com/rhasspy/piper-sample-generator)
- [Fork communautaire Apple Silicon/MPS](https://github.com/TaterTotterson/openWakeWord-Trainer)


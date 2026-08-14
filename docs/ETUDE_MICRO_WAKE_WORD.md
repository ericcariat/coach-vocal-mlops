# Étude comparative — microWakeWord et le wake word « éloquence » sur système embarqué

**Projet :** Eloquence — détection embarquée du mot d'activation « éloquence »  
**Contexte :** étude préparatoire au portage embarqué  
**Date de l'étude :** 13 août 2026  
**Dépôt étudié :** [OHF-Voice/micro-wake-word](https://github.com/OHF-Voice/micro-wake-word)  
**Révision observée :** `4665173` du 6 juillet 2026  
**Cible de déploiement principale étudiée :** TensorFlow Lite for Microcontrollers, notamment ESP32-S3 avec ESPHome

## 1. Objet de l'étude

Cette étude examine la manière dont microWakeWord :

- génère les données d'un wake word personnalisé ;
- transforme l'audio avec un frontend compatible microcontrôleur ;
- entraîne un réseau neuronal réellement streaming ;
- quantifie et exporte le modèle vers TensorFlow Lite Micro ;
- contrôle les faux positifs sur des heures de bruit ambiant ;
- s'intègre dans un appareil ESPHome ;
- pourrait servir de cible embarquée au projet Eloquence.

L'analyse accorde une importance particulière à la mémoire, à la taille du modèle, à la cadence d'inférence, à la latence et aux contraintes matérielles. Le but final n'est pas seulement de détecter correctement « éloquence » sur un ordinateur, mais de déterminer quelle partie du système peut fonctionner en permanence sur un petit appareil à faible consommation.

## 2. Résumé exécutif

Parmi OpenWakeWord, ViolaWake et microWakeWord, ce dernier est le plus directement conçu pour un microcontrôleur. Le résultat de l'entraînement est un modèle TensorFlow Lite quantifié qui conserve ses états entre les appels et ne recalcule que les nouvelles trames audio.

Les modèles officiels version 2 observés occupent environ 52 à 60 Ko sur disque. Leur manifeste annonce une arène TensorFlow Lite Micro d'environ 22 à 26 Ko. Le modèle VAD optionnel occupe environ 34 Ko et demande une arène d'environ 17 Ko. Ces nombres démontrent la compacité du réseau, mais ils ne représentent pas toute la RAM ou toute la flash du produit : il faut encore compter le frontend acoustique, les buffers audio, les états du réseau, la pile de la tâche, le microphone, ESPHome et le reste de l'application.

Le pipeline utilise un frontend `micro_speech` qui calcule 40 caractéristiques toutes les 10 ms à partir d'une fenêtre de 30 ms. Ce frontend intègre réduction de bruit, contrôle automatique de gain et PCAN. Le réseau MixConv réalise une nouvelle inférence toutes les 30 ms et conserve des ring buffers internes. Une moyenne glissante de plusieurs probabilités confirme la détection.

Comme OpenWakeWord, microWakeWord entraîne principalement sur des voix synthétiques Piper et de grands corpus négatifs. Le notebook de base ne génère que 1 000 positifs et avertit explicitement que son modèle sera probablement inutilisable en production sans beaucoup d'expérimentation.

Pour Eloquence, microWakeWord est donc moins un concurrent académique supplémentaire qu'une cible de déploiement crédible. Une trajectoire raisonnable serait :

1. conserver les modèles actuels pour l'étude et le benchmark sur ordinateur ;
2. entraîner une branche microWakeWord spécifique à « éloquence » ;
3. l'évaluer d'abord sur le même banc streaming ;
4. mesurer ensuite le modèle quantifié sur une véritable carte ESP32-S3 ;
5. ne lancer l'enregistrement complet et l'analyse vocale qu'après l'activation locale.

## 3. Architecture produit recommandée pour Eloquence

Le petit appareil n'a pas besoin d'exécuter toute l'analyse de prise de parole. Il peut agir comme un premier étage toujours actif :

```text
Microphone 16 kHz
       ↓
microWakeWord sur microcontrôleur
       ↓ « éloquence » détecté
démarrage de l'enregistrement / réveil du système principal
       ↓
transcription et analyse de la présentation
       ↓
feedback Eloquence
```

Cette séparation est favorable à l'embarqué :

- le microcontrôleur écoute en local avec un réseau de quelques dizaines de kilo-octets ;
- aucun flux audio permanent ne doit être envoyé au cloud ;
- le processeur principal ou le réseau n'est réveillé qu'après la détection ;
- la transcription, l'analyse des hésitations et le feedback restent sur une machine plus puissante.

La confidentialité est aussi améliorée : avant le wake word, le système peut ne conserver qu'un buffer circulaire très court nécessaire à la détection. Il faudra préciser dans le produit si ce buffer est entièrement volatil et quelle portion d'audio antérieure est jointe à l'enregistrement déclenché.

## 4. Comment microWakeWord obtient les répétitions du wake word

### 4.1 Pas d'interface officielle de collecte manuelle

Le dépôt ne propose pas une console comparable à ViolaWake pour enregistrer dix ou cinquante répétitions. Le notebook demande d'abord le texte ou une écriture phonétique du wake word, puis utilise [Piper Sample Generator](https://github.com/rhasspy/piper-sample-generator).

La cellule de démonstration génère une prise unique pour écoute humaine, puis la recette de base génère 1 000 fichiers. Le notebook insiste sur plusieurs points :

- écouter le premier fichier avant de lancer la génération ;
- tester éventuellement plusieurs orthographes phonétiques ;
- varier les paramètres de bruit du synthétiseur ;
- créer des phrases négatives proches ;
- générer beaucoup plus de positifs pour un vrai modèle.

Pour « éloquence », il faut impérativement comparer la graphie correcte `éloquence` à d'éventuelles formulations phonétiques acceptées par la voix Piper choisie. Le nom technique du fichier peut rester `eloquence`, mais le texte prononcé ne doit pas perdre l'accent ni modifier les phonèmes français.

Source : [notebook d'entraînement](https://github.com/OHF-Voice/micro-wake-word/blob/main/notebooks/basic_training_notebook.ipynb)

### 4.2 Les enregistrements réels doivent servir de vérité terrain

Le notebook reconnaît que valider et tester avec le même générateur TTS que celui de l'entraînement peut produire des scores trop optimistes. Il recommande des prises réelles ou un autre moteur TTS pour obtenir un benchmark plus représentatif.

Les voix réelles déjà enregistrées dans Eloquence sont donc précieuses. Elles devraient être réparties par locuteur et par source entre :

- un ensemble de calibration pour choisir le seuil ;
- une validation réelle pour sélectionner les réglages ;
- un test final verrouillé pour le résultat présenté au jury ;
- éventuellement, plus tard, un complément d'entraînement distinct.

Les augmentations provenant d'une même prise doivent rester dans le même split. Une augmentation d'un fichier de test ne doit jamais apparaître dans l'entraînement.

### 4.3 Pourquoi le TTS reste utile

Pour un modèle très compact, le TTS permet de couvrir rapidement :

- de nombreux timbres ;
- plusieurs vitesses de parole ;
- différentes hauteurs ;
- des réalisations acoustiques variées après augmentation ;
- plusieurs positions du mot dans la fenêtre.

Il ne remplace cependant pas la diversité des accents, microphones, distances et intentions humaines. Un utilisateur qui appelle volontairement l'application ne prononce pas nécessairement le mot comme une voix synthétique neutre.

## 5. Prétraitement adapté au microcontrôleur

### 5.1 Format audio

Le frontend reçoit un flux mono à 16 kHz. Il génère 40 caractéristiques à partir de fenêtres de 30 ms, avec une nouvelle tranche toutes les 10 ms. Deux fenêtres successives réutilisent donc une partie du même signal.

Le traitement est apparenté à un spectrogramme mel, mais il est spécifiquement dérivé de l'exemple TensorFlow Lite Micro `micro_speech`. Il inclut notamment :

- réduction de bruit ;
- contrôle automatique de gain ;
- PCAN, une normalisation dynamique adaptée aux énergies spectrales ;
- sortie compacte compatible avec les calculs entiers du microcontrôleur.

Source : [README microWakeWord](https://github.com/OHF-Voice/micro-wake-word/blob/main/README.md) et [exemple micro_speech de TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech)

### 5.2 Conséquence pour la comparaison avec Eloquence

Le CNN Eloquence actuel utilise son propre calcul log-mel sur ordinateur. Un modèle peut être performant avec ce frontend et perdre sa qualité si on lui fournit les caractéristiques du microcontrôleur.

Il faut donc considérer le frontend comme une partie du modèle embarqué, et tester la chaîne entière :

```text
microphone réel → frontend micro_speech → modèle int8 → logique événementielle
```

Comparer uniquement les poids ou le nombre de paramètres ne suffit pas. La robustesse de l'AGC, le bruit du microphone MEMS et le niveau électrique du signal peuvent modifier fortement les scores.

### 5.3 Compatibilité du manifeste

Le manifeste ESPHome version 2 contient `feature_step_size`. Les modèles officiels actuels utilisent 10 ms. Les caractéristiques pré-calculées, le modèle, le manifeste et le firmware doivent employer la même configuration.

Une incompatibilité de frontend peut produire un modèle qui se charge correctement mais donne des probabilités sans signification. Le format constitue donc un contrat à versionner avec le modèle.

## 6. Architecture neuronale MixConv

### 6.1 Réseau temporel compact

microWakeWord utilise une architecture issue des travaux Google sur le keyword spotting streaming. Le réseau traite le temps avec des convolutions depthwise et des convolutions ponctuelles `1 × 1`.

Une couche MixConv sépare les canaux en groupes et leur applique des noyaux temporels de longueurs différentes. Elle peut ainsi observer simultanément des phénomènes phonétiques courts et une structure plus longue sans le coût d'une convolution dense classique.

La recette du notebook configure par exemple :

```text
Convolution initiale : 32 filtres, noyau temporel 5, stride 3
Bloc 1 : 64 filtres, noyau 5
Bloc 2 : 64 filtres, noyaux 7 et 11
Bloc 3 : 64 filtres, noyaux 9 et 15
Bloc 4 : 64 filtres, noyau 23
Sortie : dense 1 + sigmoïde
```

Source : [mixednet.py](https://github.com/OHF-Voice/micro-wake-word/blob/main/microwakeword/mixednet.py)

### 6.2 Entraînement non streaming, export streaming

Pendant l'entraînement, le réseau voit le spectrogramme complet. Les convolutions utilisent un padding `valid`, ce qui évite de dépendre d'informations futures artificielles. Après entraînement, le modèle est converti en version streaming avec états internes.

Sur l'appareil, chaque couche conserve uniquement l'historique nécessaire dans un ring buffer. Lorsqu'une nouvelle tranche arrive, le modèle ne recalcule pas toute la seconde et demie précédente. Ce point explique une grande partie de son intérêt énergétique et de sa faible latence.

### 6.3 Cadence d'inférence

Les caractéristiques arrivent toutes les 10 ms, tandis que la première couche a un stride de trois dans la recette. Le réseau produit ainsi une nouvelle probabilité toutes les 30 ms.

Cette cadence est plus rapide que celle du détecteur Eloquence actuel à pas de 125 ms et que les embeddings OpenWakeWord à 80 ms. Elle ne signifie pas que la détection complète prend 30 ms : le réseau doit avoir observé assez de contexte et la moyenne glissante doit confirmer le score.

## 7. Données, augmentations et négatifs

### 7.1 Augmentations positives

La recette comprend notamment :

- égalisation sept bandes ;
- distorsion `tanh` ;
- variation de hauteur jusqu'à environ ±3 demi-tons ;
- filtre coupe-bande ;
- bruit coloré ;
- bruit ou musique de fond ;
- variation et transition de gain ;
- réponses impulsionnelles de pièces ;
- normalisation en cas de saturation.

Le notebook crée des exemples de 3,2 secondes afin que le frontend avec PCAN et réduction de bruit observe un historique plus long. Le modèle est ensuite entraîné sur une durée maximale de wake word configurée à 1,5 seconde. La position du mot est décalée près de la fin de la fenêtre, avec environ 200 ms de jitter dans la recette.

Sources : [augmentation.py](https://github.com/OHF-Voice/micro-wake-word/blob/main/microwakeword/audio/augmentation.py) et [notebook](https://github.com/OHF-Voice/micro-wake-word/blob/main/notebooks/basic_training_notebook.ipynb)

### 7.2 Corpus négatifs

Le projet fournit des caractéristiques précalculées pour différents types de négatifs :

- parole avec LibriSpeech et VOiCES ;
- conversations et parole lointaine avec CHiME6 et DiPCo ;
- musique avec FMA ;
- événements sonores avec FSD50K ;
- bruit avec WHAM! ;
- réponses impulsionnelles avec BIRD ou le corpus MIT utilisé par le notebook.

Le dataset disponible sur Hugging Face représente actuellement environ 9,7 Go. Cela illustre un paradoxe classique de TinyML : le modèle final ne fait que quelques dizaines de kilo-octets, mais sa robustesse dépend d'un dataset de préparation beaucoup plus volumineux.

Sources : [documentation des données](https://github.com/OHF-Voice/micro-wake-word/blob/main/documentation/data_sources.md) et [dataset kahrendt/microwakeword](https://huggingface.co/datasets/kahrendt/microwakeword)

### 7.3 Pondération contre les faux positifs

Chaque source peut recevoir :

- un `sampling_weight`, qui contrôle sa fréquence dans les batches ;
- un `penalty_weight`, qui augmente le coût de ses erreurs ;
- un poids de classe positif ou négatif global.

La recette de base donne un poids de classe négatif de 20. Cette valeur ne doit pas être copiée aveuglément : elle montre que les faux positifs sont traités comme une priorité métier, mais l'optimum dépend du dataset « éloquence ».

### 7.4 Négatifs français indispensables

Les données génériques apportent bruit, musique et parole variée, mais elles ne garantissent pas une bonne séparation phonétique de « éloquence ». Il faut ajouter :

- conversations françaises longues ;
- présentations orales, qui correspondent précisément au futur usage d'Eloquence ;
- mots et fragments proches ;
- télévision, radio et podcasts francophones ;
- vrais faux positifs récoltés sur le prototype ;
- occurrences de « éloquence » dans une phrase si seules les commandes isolées doivent activer le système.

Le dernier point doit être décidé fonctionnellement. Si toute occurrence du mot doit réveiller Eloquence, elle est positive. Si l'utilisateur doit interpeller l'application, certaines occurrences conversationnelles deviennent des négatifs contextuels difficiles.

## 8. Entraînement et sélection des poids

### 8.1 Notebook de départ, pas recette de production

Le projet se décrit comme une version précoce destinée aux utilisateurs avancés. Le notebook avertit que le modèle produit sera probablement trop difficile à déclencher ou sujet à de nombreux faux positifs.

La recette de démonstration utilise notamment :

- 1 000 positifs Piper ;
- 10 000 étapes ;
- batch de 128 ;
- Adam avec taux d'apprentissage de 0,001 ;
- fenêtre maximale de 1,5 seconde ;
- négatifs précalculés ;
- export et test du modèle streaming quantifié.

Ces nombres permettent de faire fonctionner le pipeline. Ils ne constituent pas une recommandation garantie pour « éloquence ».

### 8.2 Validation séparée des bruits ambiants

Les données sont organisées en quatre familles de validation et de test :

- `validation` et `testing` pour positifs et négatifs découpés ;
- `validation_ambient` et `testing_ambient` pour de longs flux entièrement négatifs.

Pendant l'entraînement, le modèle estime les faux positifs par heure en découpant les flux ambiants avec un pas de 100 ms. Cette approximation sert à choisir les poids. Après export, le modèle streaming est testé sur le flux ambiant.

Cette méthode est particulièrement pertinente pour Eloquence. Une accuracy élevée sur des clips équilibrés peut cacher un détecteur inutilisable en écoute continue.

### 8.3 Sélection à deux niveaux

Le framework peut sélectionner les poids en deux temps :

1. atteindre une cible sur une métrique à minimiser, par exemple les faux positifs par heure ;
2. parmi les modèles satisfaisants, maximiser une autre métrique telle que le rappel.

Il s'agit d'une formulation plus proche du produit qu'un simple `val_loss` minimal. Pour l'application, on peut fixer d'abord un budget maximal de faux réveils par heure, puis chercher le meilleur rappel sous cette contrainte.

## 9. Quantification et export TinyML

### 9.1 Quantification entière

Le convertisseur produit un modèle TensorFlow Lite avec :

- opérations entières `int8` ;
- entrée `int8` ;
- sortie `uint8` ;
- dataset représentatif de 500 spectrogrammes ;
- quantification expérimentale des variables internes du modèle streaming.

Quantifier aussi les états évite des opérations répétées de quantification et déquantification autour des ring buffers. C'est un détail technique important pour la vitesse et la mémoire sur microcontrôleur.

Source : [utils.py](https://github.com/OHF-Voice/micro-wake-word/blob/main/microwakeword/utils.py)

### 9.2 Mesurer après quantification

La documentation indique que la perte est généralement faible et parfois sans baisse d'accuracy. Cette observation ne doit pas remplacer une mesure propre à « éloquence ».

Il faut comparer au minimum :

- le modèle Keras non streaming ;
- le modèle TFLite streaming flottant, si exporté ;
- le modèle TFLite streaming quantifié ;
- le modèle réellement exécuté par TensorFlow Lite Micro sur la carte.

Les probabilités, le rappel et les faux positifs par heure peuvent se déplacer après chaque transformation. Le seuil du modèle final doit être calibré sur la sortie quantifiée, pas repris du modèle Keras.

## 10. Taille mémoire observée sur les modèles officiels

Les fichiers de la collection officielle ESPHome version 2 donnent un ordre de grandeur concret :

| Modèle | Taille `.tflite` | Arène déclarée | Seuil | Fenêtre glissante |
|---|---:|---:|---:|---:|
| Alexa | 55 856 octets | 22 348 octets | 0,90 | 5 |
| Hey Jarvis | 52 272 octets | 22 860 octets | 0,97 | 5 |
| Hey Mycroft | 57 248 octets | 23 628 octets | 0,95 | 5 |
| Okay Nabu | 60 264 octets | 26 080 octets | 0,97 | 5 |
| VAD | 34 328 octets | 16 772 octets | 0,50 | 5 |

Sources : [collection officielle de modèles](https://github.com/esphome/micro-wake-word-models) et [documentation ESPHome](https://esphome.io/components/micro_wake_word/)

Ces valeurs ne sont pas une garantie pour « éloquence ». Sa taille dépendra des filtres et de l'architecture choisis. L'arène doit également être testée sur l'appareil.

Le budget complet doit inclure :

```text
flash = firmware + frontend + modèle(s) + VAD éventuelle + ressources produit
RAM   = buffers micro + frontend + arène TFLM + états + pile + ESPHome + application
```

Le manifeste ne mesure ni les buffers I2S/DMA ni l'ensemble du runtime. Un modèle de 60 Ko n'implique donc pas un produit entier de 60 Ko.

## 11. Détection sur ESPHome

### 11.1 Manifeste du modèle

Un modèle personnalisé est livré avec un fichier JSON contenant notamment :

- le type `micro` ;
- le texte du wake word ;
- l'auteur ;
- le fichier `.tflite` ;
- les langues d'entraînement ;
- le seuil ;
- la taille de la fenêtre glissante ;
- le pas du frontend ;
- la taille minimale de l'arène ;
- la version minimale d'ESPHome.

Pour « éloquence », `trained_languages` devra contenir `fr`. Cette métadonnée documente les données ; elle ne rend pas automatiquement le modèle multilingue.

### 11.2 Politique de décision

ESPHome calcule la moyenne des probabilités récentes et la compare au seuil. Avec une inférence toutes les 30 ms et une fenêtre de cinq valeurs, le lissage porte nominalement sur environ 150 ms, même si les prédictions partagent une grande partie de leur contexte acoustique.

Un seuil plus élevé réduit normalement les fausses acceptations mais augmente les faux rejets. Une fenêtre plus grande stabilise la décision mais ajoute de la latence et peut diluer un pic bref.

Une VAD optionnelle peut bloquer une activation en l'absence de parole. Elle vise surtout les sons non vocaux ; elle ne sépare pas un mot français proche du wake word.

### 11.3 Plusieurs modèles

ESPHome permet de configurer plusieurs wake words et de les activer ou désactiver. Chaque modèle supplémentaire augmente toutefois la flash et peut augmenter la charge ou la mémoire selon son état d'activation. Eloquence devrait commencer avec un seul wake word et une VAD mesurée séparément.

### 11.4 Appareil cible

L'intégration officielle est particulièrement crédible sur ESP32-S3, plateforme utilisée par plusieurs appareils vocaux ESPHome. Une première carte de validation devrait comporter :

- ESP32-S3 ;
- PSRAM, même si le modèle seul est compact ;
- microphone numérique I2S/PDM correctement supporté ;
- accès aux logs et aux scores ;
- mesure du courant ;
- possibilité de rejouer un corpus audio reproductible, en plus des essais au microphone.

Si la cible finale est un petit ordinateur Linux, par exemple un Raspberry Pi, OpenWakeWord peut rester pertinent. Si l'objectif est un véritable microcontrôleur toujours actif, microWakeWord est nettement plus naturel.

## 12. Apple Silicon, TensorFlow Metal et durée d'entraînement

### 12.1 Comportement explicite du projet sur Mac ARM

Le code actuel désactive le GPU TensorFlow par défaut sur les Mac ARM, avec le commentaire qu'il est plus lent que le CPU. Le pipeline officiel ne cherche donc pas à accélérer l'entraînement MixConv avec TensorFlow Metal dans cette configuration.

Source : [model_train_eval.py](https://github.com/OHF-Voice/micro-wake-word/blob/main/microwakeword/model_train_eval.py)

Cette décision est différente de l'ancien problème Eloquence attribué à des gradients ou scores incorrects sous Metal : ici, la justification écrite dans le code est une question de performance. Elle renforce néanmoins l'intérêt d'un benchmark CPU/Metal contrôlé avant de réactiver le GPU.

### 12.2 MPS pour Piper n'est pas TensorFlow Metal

Sur macOS, le notebook récupère une branche `mps-support` du générateur Piper. Cette accélération concerne PyTorch/MPS et la génération des fichiers TTS. L'entraînement du réseau microWakeWord reste TensorFlow et suit une autre chaîne d'accélération.

Il faut donc mesurer séparément :

| Phase | Backend possible sur Mac |
|---|---|
| Génération Piper | PyTorch MPS via branche adaptée |
| Augmentation et frontend | CPU principalement |
| Entraînement microWakeWord | TensorFlow CPU par défaut sur ARM Mac |
| Conversion TFLite | CPU |
| Inférence finale | TensorFlow Lite Micro sur ESP32-S3 |

### 12.3 Mesures à consigner

Le temps total doit être ventilé entre :

- téléchargement et préparation des corpus ;
- génération TTS ;
- augmentation ;
- calcul et stockage des spectrogrammes ;
- entraînement ;
- validation ambiante ;
- conversion et quantification ;
- benchmark sur appareil.

Pour chaque phase, noter le processeur, le backend, le nombre d'exemples, le nombre d'étapes et la durée murale. Le modèle final étant petit, la préparation des données et les évaluations longues peuvent coûter plus de temps que la boucle neuronale elle-même.

## 13. Comparaison des quatre approches

| Dimension | Eloquence CNN | OpenWakeWord | ViolaWake | microWakeWord |
|---|---|---|---|---|
| Représentation | log-mel spécifique | embeddings préentraînés | embeddings OpenWakeWord | frontend `micro_speech` |
| Tête | CNN Keras volumineux | petit DNN/RNN | TemporalCNN | MixConv depthwise |
| Streaming | fenêtres recalculées | embeddings toutes les 80 ms | séquence temporelle | états internes, score toutes les 30 ms |
| Quantification MCU | non démontrée | export TFLite mais architecture plus lourde | non prioritaire | objectif natif int8 TFLite Micro |
| Taille observée | plusieurs millions de paramètres | ordre de 100 k paramètres pour certains modèles | quelques dizaines de milliers de paramètres | environ 52–60 Ko pour modèles officiels v2 |
| Positifs | réels français + TTS | TTS massif | prises utilisateur + TTS | TTS Piper, réels recommandés au test |
| Cible naturelle | ordinateur ou SBC | ordinateur/SBC | ordinateur/SBC léger | microcontrôleur ESP32-S3 |
| Atout pour Eloquence | maîtrise des données françaises | transfert d'apprentissage | collecte et tête temporelle | déploiement embarqué réel |

Cette comparaison suggère deux questions distinctes :

1. quelle architecture donne le meilleur compromis rappel/faux positifs sur le banc Eloquence ?
2. quelle architecture respecte le budget flash, RAM, latence et énergie de l'appareil final ?

Le meilleur modèle sur ordinateur peut ne pas être le meilleur produit embarqué.

## 14. Protocole recommandé pour le prototype embarqué « éloquence »

### Étape 1 — Définir le matériel et les budgets avant l'entraînement

Documenter :

- microcontrôleur exact ;
- flash et RAM/PSRAM disponibles ;
- microphone et fréquence audio ;
- latence maximale acceptable ;
- faux réveils maximaux par heure ;
- consommation moyenne cible ;
- nécessité ou non d'une connexion réseau.

Sans ces budgets, « petit » reste trop vague pour arbitrer l'architecture.

### Étape 2 — Créer un corpus positif français contrôlé

Générer plusieurs milliers de « éloquence » avec plusieurs voix françaises et variantes phonétiques validées à l'écoute. Conserver séparément les vraies voix de personnes non vues.

Le notebook à 1 000 exemples sert de test technique. Un modèle destiné à un usage quotidien demandera probablement plusieurs itérations et davantage de diversité.

### Étape 3 — Construire un banc négatif correspondant au produit

Ajouter aux corpus génériques des heures de présentations françaises, conversations, podcasts, télévision, musique et bruit de pièce. Enregistrer aussi le bruit propre du microphone et de la carte cible.

### Étape 4 — Entraîner et sélectionner sur les faux positifs par heure

Utiliser une validation ambiante longue pour choisir les poids. Réserver un autre flux long et verrouillé au test final.

### Étape 5 — Comparer avant et après quantification

Mesurer le modèle Keras, le TFLite streaming quantifié et le modèle sur ESP32-S3. Recalibrer le seuil sur la version quantifiée.

### Étape 6 — Uniformiser le comptage des événements

Appliquer des règles communes aux modèles comparés : tolérance temporelle, fusion de détections voisines, cooldown et définition d'un faux positif. Présenter séparément les réglages natifs ESPHome.

### Étape 7 — Mesurer la carte réelle

Relever :

- taille du firmware et du modèle ;
- RAM libre minimale, plus grande allocation et arène ;
- temps du frontend et de chaque inférence ;
- charge CPU et dépassements de trames ;
- latence entre la fin du mot et l'événement ;
- consommation au repos, en écoute et après activation ;
- température et stabilité sur plusieurs heures ;
- faux positifs par heure lors d'une écoute continue.

### Étape 8 — Récolter les erreurs et réentraîner

Conserver avec consentement les faux positifs et faux rejets, les annoter et les réinjecter uniquement dans le prochain cycle d'entraînement. Le test final initial doit rester inchangé pour mesurer le progrès sans déplacer la cible.

## 15. Risques et limites

### 15.1 Entraînement difficile

Le dépôt prévient lui-même que l'entraînement d'un modèle utilisable reste difficile. Un fichier `.tflite` généré avec succès ne prouve pas la robustesse du détecteur.

### 15.2 Domaine anglophone

Les modèles officiels version 2 observés sont déclarés entraînés en anglais. Le framework n'interdit pas le français, mais il n'apporte pas un modèle « éloquence » déjà validé. Le TTS, les négatifs et les tests français restent à construire.

### 15.3 Écart simulation–matériel

Le test sur fichiers propres ne reproduit pas le microphone MEMS, le boîtier, les vibrations, l'AGC et la réverbération réelle. Un passage sur la carte est obligatoire avant de conclure.

### 15.4 Confusion entre petit modèle et petit système

Le wake word peut tenir sur un microcontrôleur, mais l'analyse complète d'Eloquence — transcription, répétitions, hésitations, mots-clés et feedback — demandera vraisemblablement un SBC, un téléphone, un ordinateur ou un serveur. L'architecture à deux étages doit être assumée.

### 15.5 Maintenance et maturité

Le package se présente encore comme une version `0.1.0` précoce pour utilisateurs avancés. Le dépôt évolue et ses dépendances actuelles incluent TensorFlow récent, NumPy 2 et LiteRT. L'environnement d'entraînement, le commit et le manifeste doivent être figés pour reproduire une expérience.

## 16. Licences

Le code microWakeWord est sous licence Apache 2.0. La collection officielle de modèles ESPHome est également publiée dans un dépôt Apache 2.0.

Les données utilisées pour entraîner un modèle personnalisé obéissent toutefois à leurs propres conditions. Le notebook avertit que son mélange de sources comporte diverses restrictions et considère le résultat approprié à un usage personnel non commercial. Le dataset de caractéristiques Hugging Face est indiqué sous CC BY-NC 4.0.

Pour une future version commerciale d'Eloquence, il faudra donc auditer séparément :

- les voix et modèles Piper ;
- les corpus audio ;
- les caractéristiques précalculées ;
- les réponses impulsionnelles ;
- les conditions de redistribution du modèle dérivé.

La licence permissive du code n'efface pas les restrictions des données.

## 17. Intérêt pour le projet

microWakeWord apporte une dimension très forte au dossier : le passage d'un prototype Deep Learning à une contrainte TinyML réelle.

Le notebook de synthèse peut montrer :

- la définition métier du wake word ;
- le pipeline de données françaises ;
- la différence entre frontend log-mel et frontend microcontrôleur ;
- l'architecture MixConv et les convolutions depthwise ;
- le passage non streaming vers streaming stateful ;
- la quantification entière ;
- la comparaison avant/après quantification ;
- les faux positifs par heure ;
- les budgets flash, RAM, latence et énergie ;
- la mesure finale sur appareil.

Une formulation possible devant le jury est :

> Le modèle initial démontre la détection de « éloquence » sur ordinateur. J'ai ensuite étudié une architecture TinyML destinée à rester active sur un ESP32-S3. Elle calcule un frontend acoustique toutes les 10 ms, exécute un MixConv quantifié toutes les 30 ms et conserve ses états internes. J'ai comparé la qualité événementielle, la quantification, la mémoire, la latence et les faux positifs par heure afin de valider non seulement le réseau, mais aussi sa faisabilité dans le produit Eloquence.

## 18. Conclusion

microWakeWord est le projet le plus aligné avec l'objectif d'un petit détecteur toujours actif. Sa véritable valeur ne vient pas seulement de la petite taille du fichier : le frontend, les convolutions depthwise, les états internes, la quantification et l'intégration ESPHome forment ensemble une chaîne pensée pour le microcontrôleur.

Il ne fournit cependant pas un modèle français prêt à l'emploi. Le notebook officiel est volontairement minimal, l'entraînement robuste demande de nombreux essais et les modèles publiés sont anglophones. Pour « éloquence », la qualité reposera sur les voix françaises, les négatifs propres aux présentations orales et un test de longue durée sur le microphone final.

La recommandation est de faire de microWakeWord la branche « déploiement TinyML » du projet, sans supprimer immédiatement les autres modèles. Eloquence CNN, OpenWakeWord et ViolaWake peuvent rester des références expérimentales ; microWakeWord doit prouver qu'il respecte à la fois la qualité de détection et le budget réel de l'appareil embarqué.

## 19. Références principales

- [Dépôt OHF-Voice/micro-wake-word](https://github.com/OHF-Voice/micro-wake-word)
- [README et processus de détection](https://github.com/OHF-Voice/micro-wake-word/blob/main/README.md)
- [Notebook d'entraînement](https://github.com/OHF-Voice/micro-wake-word/blob/main/notebooks/basic_training_notebook.ipynb)
- [Architecture MixConv](https://github.com/OHF-Voice/micro-wake-word/blob/main/microwakeword/mixednet.py)
- [Entraînement et sélection des poids](https://github.com/OHF-Voice/micro-wake-word/blob/main/microwakeword/train.py)
- [Conversion streaming et quantification](https://github.com/OHF-Voice/micro-wake-word/blob/main/microwakeword/utils.py)
- [Augmentations audio](https://github.com/OHF-Voice/micro-wake-word/blob/main/microwakeword/audio/augmentation.py)
- [Sources des données](https://github.com/OHF-Voice/micro-wake-word/blob/main/documentation/data_sources.md)
- [Dataset de caractéristiques précalculées](https://huggingface.co/datasets/kahrendt/microwakeword)
- [Documentation du composant ESPHome](https://esphome.io/components/micro_wake_word/)
- [Collection officielle des modèles ESPHome](https://github.com/esphome/micro-wake-word-models)
- [TensorFlow Lite Micro `micro_speech`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech)
- [Article « Streaming Keyword Spotting on Mobile Devices »](https://arxiv.org/abs/2005.06720)
- [Article MixConv](https://arxiv.org/abs/1907.09595)
- [Piper Sample Generator](https://github.com/rhasspy/piper-sample-generator)


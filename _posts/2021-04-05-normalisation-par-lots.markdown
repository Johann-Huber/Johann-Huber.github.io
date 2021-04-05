---
layout: post
title:  "Normalisation par Lots (ou Batch Normalization)"
date:   2021-04-05 15:37:00 +0200
categories: Apprentissage-profond
---




Il est surprenant de constater le manque de contenu en français que l’on peut trouver sur internet à propos de ce concept, pourtant largement utilisé en apprentissage profond. Le foisonnement d’articles traitant du sujet n’est d’ailleurs par toujours éclairant ; beaucoup s’appuyant sur les premières explications proposées par les inventeurs de la technique, qui ont été très largement remises en question depuis la publication de l’article original.


Objectifs de cet article : 
- Permettre d’appréhender le concept de Normalisation par lots selon 3 niveaux de complexité :  en 30 secondes, en 3 minutes, et dans une exploration plus détaillée ;
- Aborder les éléments clefs à avoir à l’esprit pour exploiter efficacement la couche BN ;
- Proposer une implémentation simple de la couche BN sous PyTorch, pour voir en détail sa mise en pratique ;
- Faire le point sur le niveau de compréhension actuel que l’on a de ce concept.





| Nom français          | Nom anglais         | Abréviation courante |
|-----------------------|---------------------|----------------------|
| Normalization par lot | Batch Normalization | BN                   |






## A) En 30 secondes


La **Normalisation par lots** (en anglais ***Batch-Normalization*** - notée ***BN***) est une méthode algorithmique qui permet d’entraîner un réseau de neurones profond de manière plus rapide et plus stable. 

Cette méthode consiste à normaliser les vecteurs d’activation des couches cachées en utilisant les caractéristiques statistiques du lot (ou *batch*) - la moyenne et l’écart-type - juste avant (ou juste après) le passage dans la fonction non-linéaire.



<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/sbn_1a_fr.jpg">
  Schéma 1.a Perceptron multicouche <strong>sans normalisation par lots (BN)</strong>
</p>

	
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/sbn_1b_fr.jpg">
  Schéma 1.b Perceptron multicouche <strong>avec normalisation par lots (BN)</strong>
</p>


Toutes les infrastructures de développements (ou frameworks) populaires proposent des implémentations de cette méthode sous la forme de couche computationnelle, que l’on peut facilement insérer dans un réseau de neurones.




<ins>Article de référence :</ins> [“Batch-normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift”](url=https://arxiv.org/abs/1502.03167) (trad. “Normalisation par Lots : Accélération de l’entraînement des réseaux de neurones profonds par la réduction du décalage de covariable interne”).

<ins>Article (contribution significative dans la compréhension du concept) :</ins> [“How does batch normalization help optimization”](url=https://arxiv.org/pdf/1805.11604.pdf) (trad. “Comment la normalisation par lots facilite l’optimisation.”).


## B) En 3 minutes

### 1) Principe

La normalisation par lot s’articule différemment pendant la phase d’entraînement et la phase d’évaluation.

#### a) Phase d’entraînement

Pour chaque couche cachée, on calcule la normalisation par lot de la façon suivante :

AJOUTER LES EQUATIONS !!!!!!!!!!!!!!
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/bn_eq1.gif">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/bn_eq2.gif">
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mu&space;=&space;\frac{1}{n}*\sum_{i}Z^{(i)}" title="\mu = \frac{1}{n}*\sum_{i}Z^{(i)}" />
</p>


- On calcule d’abord la moyenne 𝜇 et l’écart-type σ des vecteurs d’activations à l’échelle du lot (1) et (2).
- En utilisant ces valeurs, on normalise le vecteur d’activation Z(i) (3). De cette façon, la distribution des valeurs d’activations associées à chaque exemple du lot suit une loi normale centrée réduite. (𝜀 est ici une constante de stabilisation numérique)

AJOUTER SCHEMA 2
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/sbn_2.jpg">
  <strong>Schéma 2 : 1ère étape de la normalisation par lots.</strong> Exemple d’une couche de 3 neurones, avec un lot de taille b. Pour chaque neurone, les valeurs à l’échelles du batch suivent une loi normal centrée réduite.
</p>


Finalement, on calcule les valeurs de **sortie de la couche de normalisation par lot** Ẑ(i) en appliquant une transformation linéaire avec deux paramètres à entraîner (4). Cette dernière opération permet au modèle de définir à chaque couche cachée la distribution optimale, en ajustant ces deux paramètres :
- 𝛾 permet de jouer sur l’étalement de la gaussienne ;
- 𝛽 joue le rôle de biais, décalant à gauche ou à droite la gaussienne.


AJOUTER SCHEMA 3
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/sbn_3.jpg">
  <strong>Schéma 3: Intérêt des paramètres 𝛾 et 𝛽.</strong> Les modifications sur la distribution (en haut) permettent d’exploiter différentes parties de la fonction non-linéaire (en bas).
</p>

<ins>Remarque :</ins> Les raisons qui rendent la couche BN efficace ont souvent fait l’objet d’incompréhensions et d’erreurs, jusque dans l’article officiel. Des recherches récentes ont écartées certaines hypothèses erronées, et ont permis une meilleure compréhension de cette technique. Ces aspects sont abordés plus largement dans la partie C.III : “Pourquoi la couche BN est-elle efficace ?” de cet article.


À chaque itération, le réseau calcule la moyenne 𝜇 et l’écart-type σ correspondant au lot en cours. Les paramètres 𝛾 et 𝛽 sont ajustés via la rétropropagation des gradients, en appliquant une [moyenne mobile](https://fr.wikipedia.org/wiki/Moyenne_mobile). De cette façon, l’ajustement des paramètres 𝛾 et 𝛽 tiennent davantage compte des dernières itérations que des premières. 

#### b) Phase d’évaluation

Contrairement à la phase d’entraînement, **on ne dispose pas forcément d’un lot complet à inférer lors de l’évaluation.**

Pour s’affranchir de ce problème, on détermine (𝜇pop , σpop), tel que :
- 𝜇pop : estimation de la moyenne de la population étudiée ;
- σpop : estimation de l’écart-type de la population étudiée.

Ces valeurs sont déterminées à partir des (𝜇lot , σlot) rencontrés pendant l'entraînement, et appliquée systématiquement dans l’équation (3), au lieu d’avoir recours aux équations (1) et (2).

<ins>Remarque :</ins> Cet aspect est plus largement décrit dans la partie C.II.3 : Paramètres statistiques lors de la phase d’évaluation”.


### 2) Principe

En pratique, on considère la normalisation par lots comme une couche à part entière, au même titre qu’un perceptron, qu’une couche de convolution, qu’une fonction d’activation ou qu’un dropout.

On trouve la couche de normalisation par lots (ou couche BN) dans les infrastructures de développements (ou frameworks) populaires.

| Librairie          | Couches BN
|--------------------|------------------------------------------------------------------|
| Pytorch            | torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d |
| Tensorflow / Keras | tf.nn.batch_normalization, tf.keras.layers.BatchNormalization    |

<ins>Remarque :</ins> Il est très facile de trouver la documentation de la couche BN pour votre infrastructure de développement, qu’il s’agisse de Mxnet, Matlab, Caffe …  


Toutes donnent la possibilités de modifier les paramètres que cette méthode fait intervenir ; dans la pratique, **le paramètre le plus important est la taille du vecteur d’entrée**, à savoir :
- Le nombre de neurones de la couche cachée, dans le cas d’un perceptron multicouche ;
- Le nombre de filtres de la couche cachée, dans le cas d’un réseau convolutif.


### 3) Un coup d’oeil aux résultats

Si l’on est loin d’avoir compris tous les mécanismes sous-jacents à la couche BN (voir C.III), il y a un point sur lequel tout le monde s’accorde : ça marche.

En guise de mise en bouche, regardons rapidement les résultats obtenus dans l’article officiel [1] :

AJOUTER GRAPHIQUE 1
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/gbn_1.png">
  <strong>Graphique 1 : Efficacité de la couche BN en entraînement</strong> (source : [1]). Précision sur le jeu de validation ImageNet (2012) en fonction du nombre d’itération d'entraînement, pour des réseaux Inception avec ou sans BN, en augmentant les taux d’apprentissage pour les réseaux BN (1 fois, 5 fois, 30 fois le taux optimal du réseau Inception).
</p>

Le résultat est net : en ajoutant des couches BN, le **réseau s’entraîne plus vite et plus efficacement**.


Voilà de quoi comprendre le principe des couches BN, leur intérêt, et d’être en mesure de les utiliser en pratique. une compréhension un peu plus approfondie est cependant nécessaire pour ne pas tomber des nues devant le comportement d’un réseau de neurone.


## C) Comprendre la Normalisation par lots (BN)

### I) Implémentation

J’ai ré-implémenté cette méthode sous Pytorch, de manière à retrouver les résultats de l’article officiel. Vous pourrez le trouver dans [ce repo git](https://github.com/Johann-Huber/batchnorm_pytorch/blob/main/batch_normalization_in_pytorch.ipynb).

Je vous invite à parcourir les diverses implémentations de la couche BN disponible en ligne (presque toujours en anglais), en premier lieu celle de l'infrastructure avec laquelle vous travaillez.

### II) La couche BN en pratique

#### 1) Résultats de l’article original

J’ai décidé de commencer par présenter les résultats obtenus avec la couche de normalisation par lots car **c’est le point sur lequel tout s’accorde** la concernant : **Elle est efficace en pratique.**

L’article officiel [1] a réalisé 3 expériences pour évaluer l’efficacité de leur méthode. 

La première a pour but de montrer l’efficacité de la normalisation par lots sur un exemple simple : Il s’agit d’entraîner un classificateur sur le jeu de donnée MNIST (reconnaissance de chiffres écrits à la main, issue du célèbre article de Y. Lecun). Le modèle consiste en une succession de 3 couches entièrement connectées de 100 neurones, suivis de fonctions sigmoïdes. On entraîne le tout sur 50 000 itérations en utilisant un algorithme de gradient stochastique (en anglais Stochastic Gradient Descent - notée SGD), avec ou sans couche de normalisation par lots pour comparer.

Ce résultat peut être reproduit rapidement sans GPU, je vous invite à essayer par vous-même pour vous faire la main.

AJOUTER GRAPHIQUE 2
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/gbn_2.png">
  <strong></strong> 
</p>


Bonne nouvelle, la normalisation par lots améliore les performances du réseau.

Pour la deuxième expérience, regardons l’impact de cette méthode sur l’activation des neurones au niveau des couches cachées. Voici les valeurs d’activations obtenues sur la dernière couche cachée, juste avant le passage dans la fonction non-linéaire :

AJOUTER GRAPHIQUE 3
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/gbn_3.png">
  <strong></strong> 
</p>

Sans la normalisation par lot, les valeurs d’activations varient fortement au cours des premières itérations. En revanche, les courbes d’activations ne présentent pas d’à-coups avec l’utilisation de couches BN. 


AJOUTER GRAPHIQUE 4
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/gbn_4.png">
  <strong></strong> 
</p>


Le signal est d’ailleurs moins bruité, lorsque l’on utilise la normalisation par lots. On constate que l’optimiseur (en anglais optimizer) fait converger les poids beaucoup plus facilement.

Cet exemple simple ne montre cependant pas toute l’étendue de l’impact de cette méthode.

L’article officiel explore une troisième expérience. Il s’agit d’évaluer les performances de la couche BN sur un modèle classificateur plus complexe, appliqué à la base de donnée ImageNet (2012). Pour cela, les auteurs adaptent un réseau de neurone très performant (pour l’époque) intitulé [Inception](https://arxiv.org/abs/1409.4842), en lui ajoutant des couches de normalisation par lot. Ils comparent ensuite des résultats du réseau original avec plusieurs versions modifiées. 

Ils obtiennent les résultats suivant :

AJOUTER GRAPHIQUE 5
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/gbn_1.png">
  <strong>Graphique 1 : Efficacité de la couche BN en entraînement</strong> (source : [1]). Précision sur le jeu de validation ImageNet (2012) en fonction du nombre d’itération d'entraînement, pour des réseaux Inception avec ou sans BN, en augmentant les taux d’apprentissage pour les réseaux BN (1 fois, 5 fois, 30 fois le taux optimal du réseau Inception).
</p>


AJOUTER GRAS ET COULEURS

Avec :
- BN-Baseline : Même réseau qu’Inception, avec des couches de BN
- BN-x5 : Même réseau qu’Inception, avec des couches de BN, et un taux d’apprentissage (learning rate - noté LR) multiplié par 5
- BN-x30 : Même réseau qu’Inception, avec des couches de BN, et un taux d’apprentissage multiplié par 30
- BN-x5-Sigmoid : Même réseau qu’Inception, avec des couches de BN, un taux d’apprentissage multiplié par 5, et des fonction sigmoïdes à la place des ReLU

Voici ce qu’on peut conclure de ces courbes :

- Ajouter des couches de BN permet de converger plus vite vers une meilleure solution (précision plus élevée) que lorsque l’on n’en utilise pas ;

L’amélioration est d’ailleurs bien plus nette que dans notre exemple du petit jeu de donné MNIST.

- Ajouter des couches de BN permet d’utiliser au taux d’apprentissage beaucoup plus important (à noter qu’avec un taux d’apprentissage 5 fois supérieur à celui initial, le réseau Inception diverge déjà).

On en conclut qu’il est plus facile de trouver un taux d’apprentissage “acceptable”, dans la mesure où l’intervalle de valeur entre le sous-entraînement et l’explosion de gradient est plus large. 

En outre, un plus grand taux d’apprentissage permet à l’optimiseur d’éviter de s’arrêter dans un minimum local. Incité à l’exploration, l’optimiseur converge vers de meilleures solutions.

- Le modèle qui ne repose que sur des sigmoïdes atteint des résultats compétitifs avec les modèles qui utilisent des ReLU.

Ce dernier point est davantage intéressant pour ce qu’il représente, que pour les résultats obtenus avec la sigmoïde - qui de toutes évidences, sont moins bons qu’avec la ReLU. 

Pour montrer la valeur de ce résultats, je me permets de paraphraser/reformuler les propos de Yann Goodfellow, référence dans le monde de l’apprentissage profond (inventeur des réseaux GANs, auteur de l’ouvrage de référence “Deep learning handbook”, … la liste est longue !)  : 
MODIFIER FORMULATION


INSERER CITATION FORME
Avant la BN, les chercheurs pensaient qu’il était presqu’impossible d’entraîner efficacement des modèles qui ne reposent que sur des sigmoïdes au niveau des couches cachées. Plusieurs approches ont été envisagées pour résoudre les problèmes d’instabilité à l’entraînement, cherchant des méthodes plus optimales d’initialisation des poids ; les embryons de solutions reposaient sur des découvertes heuristiques, fragiles, et peu satisfaisantes. L’arrivée de la BN a rendu exploitables des réseaux que l’on n’arrivaient pas à entraîner efficacement ; Cet exemple en est une preuve. 
Yann Goodfellow 
(source: la video)

Ces résultats donnent un aperçu de l’efficacité remarquable de la normalisation par lots. Mais cette technique implique quelques effets qu’il est important d’avoir à l’esprit pour l’exploiter pleinement.


#### 2) Régularisation, effet de bord de la normalisation par lots

La normalisation par lots repose sur les valeurs de moyenne et de variance de chaque lot (ou *batch*). Les valeurs d’activations de chaque couche cachée dépendent donc du lot actuellement traité par le réseau. Cette transformation ajoute donc du bruit lié aux distributions des exemples du lot au niveau de chaque couche cachée.

Ajouter un peu de bruit dans un réseau pour éviter le sur-apprentissage … cela ressemble à un processus de régularisation, non ? ;)

En pratique, on ne compte pas sur la normalisation par lot pour éviter le sur-apprentissage d’un réseau, pour des raisons d’[orthogonalités](https://en.wikipedia.org/wiki/Orthogonality_(programming)). Pour faire simple, on s’assure que chacun des modules de notre réseau remplissent un rôle précis, au lieu de compter sur plusieurs modules pour gérer différents problèmes en même temps (ce qui est le meilleur moyen de ne pas aboutir à un solution optimale).

Néanmoins, il est intéressant d’avoir conscience de ce phénomène, puisqu’il peut expliquer un comportement imprévu du réseau (notamment lorsque l’on fait du débogage).

<ins>Remarque :</ins> Plus le lot est grand, moins l’effet de régularisation sera important (minimisation de l’impact du bruit).


#### 3) Paramètres statistiques lors de la phase d’évaluation

Le modèle est appelé en phase d’évaluation dans deux contextes :
Dans le cadre d’un processus de validation / de test, réalisée au cours du développement et de l’entraînement du modèle ;
Lors du déploiement de ce dernier en conditions réelles (phase d’inférence).

Si dans le premier cas, on peut appliquer la normalisation par lot comme en entraînement dans un souci de commodité de calcul, l’appliquer en inférence n’a vraiment pas de sens. Pourquoi ? Parce que l’on a pas nécessairement un lot entier à prédire. Si notre modèle fonctionne en temps réel, pour une caméra embarquée sur un robot par exemple, on peut n’avoir qu’à traiter une image à la fois. Si la taille du lot d’entraînement est N, que faire des (N - 1) autres valeurs à fournir en entré pour réaliser l’inférence ? 

On peut imaginer que l’on choisit des valeurs arbitraires pour combler le lot. En fournissant le lot n°1 au modèle, on obtient un certain résultat pour l’image qui nous intéresse. Constituons à présent un nouveau lot n°2, à partir d’autres valeurs arbitraires ; on obtiendrait un résultat différent en sortie. Deux résultats différents pour une même image fournie en entrée du modèle n’est certainement pas souhaitable.

Néanmoins, il est nécessaire d’avoir des valeurs 𝜇 et σ pour chacune de nos couches BN, dans la mesure où les paramètres 𝛽 et 𝛾 ont été entraînées à partir de signaux normalisées. 

L’astuce consiste à définir 𝜇pop et σpop, qui sont respectivement l’estimation de la moyenne et l’écart-type de la population étudiée. Ces paramètres sont calculés comme la moyenne sur l’ensemble des (𝜇lot, σlot) rencontrés lors des itérations.

Cependant, cette astuce peut être à l’origine d’instabilité lors de la phase d’évaluation ; voyons cela dans la partie suivante.

#### 4) Stabilité de la couche BN

A REFOMULER : DE FAIT DE ... MAL DIT
Si la normalisation par lots marche généralement très bien, il arrive parfois que les choses se compliquent. Du fait de la façon dont cette couche est implémentée, il arrive que le réseau diverge durant la phase d’évaluation.

On a mentionné plus haut comment sont calculés 𝜇pop et σpop, de façon à estimer les paramètres de normalisation des valeurs d’activation au cours de l’évaluation : on fait la moyenne des (𝜇lot, σlot) vus lors des précédentes itérations.

Imaginons que l’on entraîne un réseau à partir d'images ne contenant que des chaussures de sport. Comment réagirait le réseau s'il rencontre des images contenant des chaussures de villes ?

AJOUTER IMAGES
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/car_n_shoes2.jpg">
  Si la distribution d'entrée durant la phase de test est trop différente de celle de la phase d'entraînement, le modèle peut surréagir à certains signaux, entraînant les couches d'activations à diverger.
</p>

On devine que les valeurs d’activation au niveau des couches cachées risquent de suivre des distributions tout à fait différentes - trop, sans doute. Dans ce cas, la paire (𝜇pop, σpop) estimée au cours de l’entraînement n’est pas représentative de la population réelle que rencontre le réseau en phase de test. Appliquer (𝜇pop, σpop) risque d’éloigner le signal de la loi normale centrée réduite désirée, pouvant mener à une surestimation des valeurs d’activation. 

Ce phénomène est amplifié par une propriété connue de la couche BN : au cours de l’entraînement, les valeurs d'activation sont normalisées en tenant compte de leur propre valeur. Au moment de l’inférence, on applique la normalisation à partir des coefficients (𝜇pop, σpop) calculé pendant l’entraînement : les coefficients utilisés pour la normalisation ne tiennent alors pas compte des valeurs d’activations elles-même.

En général, on s’assure que les jeux de données d’entraînement et de tests soient suffisamment proches pour que (𝜇pop, σpop) soient cohérents. Dans le cas inverse, on pourrait penser que le jeu d’entraînement n’est pas suffisamment large et de bonne qualité pour entraîner notre modèle sur la tâche désirée.

Mais il existe des cas où ce problème survient (lien : https://discuss.pytorch.org/t/model-eval-gives-incorrect-loss-for-model-with-batchnorm-layers/7561/38 ), j’en ai moi même fait les frais : Au cours de la compétition Kaggle de prédiction de l’évolution de la maladie de fibrose pulmonaire (lien : https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression), nous disposions d’un petit jeu de donnée d’entraînement contenant - entres autres - des scanners 3D des poumons de chaque patient. Le contenu était si riche et si divers (pour une petite centaine d’exemples), que le réseau convolutif avec lequel je comptais faire de l’extraction de caractéristiques m’a fait la fâcheuse surprise de retourner des valeurs astronomiques sitôt que l’entraînement se trouvait en phase de validation...un régale à déboguer. ;)

Dans ce genre de contexte où les jeux de données d’entraînement sont limités, il faut faire avec les moyens du bord. 

Ajouter systématiquement des BN dans notre réseau - en pensant que cela n’aura que des effets positifs - n’est certainement pas la meilleure stratégie !

#### 5) Réseaux récurrents, et normalisation par couches

En pratique, il est largement admis le principe suivant :
Pour les réseaux convolutifs (CNN) : utiliser de préférence la Normalisation par Lots (Batch Normalization, notée BN)
Pour les réseaux récurrents (RNN) : utiliser de préférence la Normalisation par Couches (Layer Normalization, notée LN)

Si la BN normalise à l’échelle des exemples de chaque lot, la LN normalise à l’échelle des couches cachées. Cette deuxième solution s’avère être plus efficace avec des réseaux récurrents. Une piste d’intuition réside dans la difficulté à définir une stratégie cohérente avec ce type de neurones, qui repose sur la multiplication d’une même matrice de poids de nombreuses fois successivement. Faut-il normaliser indépendamment chaque étape ? Ou au contraire, en faire la moyenne, puis appliquer la normalisation récursivement ?


Je ne m’attarderai pas davantage sur ce point, ce n’est pas précisément l’objet de cet article.


#### 6) Avant ou après la fonction non-linéaire ?

Historiquement, la couche BN est positionnée juste avant la fonction non-linéaire. Ceci étant cohérent avec les objectifs et les hypothèses des auteurs à l’époque. 

Dans leur article, ils déclarent :

AJOTUER CITATION
“Notre voudrions être certains que le réseau produise toujours une activation avec une distribution statistique désirée.”
Sergey Ioffe & Christian Szegedy
source : [1]


En revanche, des expérimentations ont montré que la couche BN positionnée après la fonction non-linéaire donne de meilleurs résultats.

Cette petite expérience en est [un exemple](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu).


François Chollet, créateur de Keras et actuellement ingénieur chez Google, a d’ailleurs prétendu à ce sujet que :
AJOUTER CITATION

“Je n’ai pas vérifié ce qui est suggéré dans l’article original, mais je peux garantir avoir vu dans du code écrit récemment par Christian [Szegedy] que la ReLU est appliquée avant la BN. Mais c’est parfois encore sujet à débat.”
François Chollet
source :  https://github.com/keras-team/keras/issues/1802



Même si le vent semble tourner, beaucoup d’architectures communément utilisées pour de l’apprentissage par transfert (ResNet, mobilenet-v2, ...) placent toujours BN avant.

Remarquez que l’article [2] - qui remet en question les intuitions défendues par l’article original [1] pour expliquer l’efficacité de la couche BN (voir C.III.3) - ont placé la couche BN avant la fonction d’activation. Ils n’apportent toutefois aucun élément d’explication sur cet aspect.

À ma connaissance, cette question est donc toujours en discussion. 


<ins>Pour en savoir plus :</ins> [Conversation  intéressante](https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/dgqaksn/) (hélas en anglais) sur reddit - même si certains arguments sont fragiles - avec une grosse tendance en faveur de la BN après l’activation.


### III) Pourquoi la couche BN est-elle efficace ?

#### 1) Première hypothèse - Confusion autour du décalage de covariable interne

Bien que fondamental, la normalisation par lots est un concept souvent mal compris. Cela tient plus d’une erreur longtemps propagée, que de la complexité de la notion.

Dans l’article officiel, les auteurs introduisent la BN comme suit : 

AJOUTER CITATION
“Nous appelons Décalage de Covariable Interne (en anglais Internal Covariate Shift) la modification au cours de l’entraînement de la distribution statistique des noeuds internes d’un réseau profond. [...] Nous proposons un nouveau mécanisme, que l’on appelle Normalisation par Lots (Batch Normalization), qui résout en partie le problème du décalage de covariable interne, et se faisant accélère significativement l’entraînement des réseaux de neurones profonds.”
Sergey Ioffe & Christian Szegedy
source : [1] article officiel


Autrement dit, l’efficacité de la couche BN réside dans sa résolution (partielle) du problème de décalage de covariable interne.

AJOUTER LIEN
Ce point à été remis en question dans des recherches postérieures (liens).

Pour comprendre ce qui a suscité cette confusion, intéressons-nous à ce qu’est le décalage de covariable, et aux effet de la normalisation par lot sur un réseau de neurones profond.

Notation : L’abréviation ICS fait référence au Décalage de Covariable Interne (venant de l’anglais Internal Covariate Shift). 

BONNE TAILLE DE TITRE ?
##### Qu’est-ce que le décalage de covariable (au sens de la distribution) ?

Les auteurs l’ont dit : le décalage de covariable, au sens de la distribution, décrit la modification de distribution statistique au cours de l’entraînement d’un modèle, et, par extension, le décalage de covariable interne décrit ce phénomène à l’intérieur d’un réseau de neurone profond.

Voyons en quoi cela pourrait poser problème avec un exemple.

Supposons que l’on cherche à entraîner un réseau classificateur qui puisse répondre à la question suivante : Cette image contient-elle une voiture ? Si l’on voulait extraire toutes les images de voiture d’une immense base de donnée non-étiquetée, un tel réseau serait très efficace. 

On aurait bien-sûr une image RGB en entrée, un ensemble de couches de neurones convolutifs, suivis de quelques couches entièrement connectées (perceptrons). On souhaite obtenir en sortie une seule valeur flottante comprise entre 0 et 1, décrivant la probabilité que l’image contienne effectivement une voiture.

AJOUTER SCHÉMA 5
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/sbn_5fr.jpg">
  <strong>Schéma 5 : Réseau convolutif simple pour réaliser une tâche de classification. </strong>
</p>


Pour entraîner un tel modèle, il nous faudrait un nombre conséquent d’images étiquetées (1 : “Cette image contient une voiture.”, ou 0 : “Cette image ne contient pas de voiture).

Mais imaginons que nous ne disposions que de voiture “classiques” (de ville, ou de sport) pour l'entraînement. Comment le modèle réagirait si nous lui demandions de classifier une image contenant une formule 1 ?

AJOUTER IMAGES
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/car_n_shoes.jpg">
  Comme évoqué dans la section (section C.2.4), le décalage de distribution peut détériorer les performances du réseau, voir provoquer une explosion des valeurs d'activation.
</p>


Dans cet exemple, il y a un décalage entre la distribution statistique associées aux images de voitures utilisées pour l’entraînement, et la distribution statistique associées aux images de voitures de test. Plus généralement, il suffit d’une autre orientation, forme, luminosité ou condition climatique que celles vues pendant la phase d’entraînement pour que nos performances se gâtent. On dit alors que notre modèle ne généralise pas efficacement.


Si on représentait les caractéristiques extraites par notre modèle dans l’espace de caractéristique, on aurait sans doute quelque chose comme ça :

AJOUTER SCHÉMA 6.a
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/sbn_6afr.jpg">
  <strong>Schéma 6.a : Pourquoi faut-il normaliser les valeur d’entrée d’un modèle ? cas non-normalisé.</strong> À l’entraînement, les valeurs d’entrée sont très éparses : la fonction approximée sera précise là où la densité de points est forte. Au contraire, elle sera imprécise là où la densité est faible (pouvant prendre l’une des courbes tracées à titre d’exemple).
</p>


Considérons que le symbole croix corresponde aux caractéristiques associées à une image ne contenant pas une voiture, et que le symbole rond corresponde aux caractéristiques associées à une image contenant une voiture. On peut voir qu’une même fonction séparerait efficacement les deux ensembles. Mais il y a fort à parier que notre modèle déduise du jeu d’entraînement une fonction moins précise pour la partie supérieure du graphique, puisqu’il n’y a pas de valeur d’entraînement qui se situe dans cette zone pour servir de repère à l’optimiseur. Ce dernier approximera la fonction du mieux qu’il pourra, poussant le classificateur à faire beaucoup d’erreurs. 

Entraîner efficacement notre réseau nécessiterait beaucoup d’images de voitures, de sorte que notre jeu d’entraînement contiennent à peu prêt toutes les variations de positions et de contexte imaginable. Même si dans les faits, c’est de cette façon que l’on entraîne de bons réseaux de neurones aujourd’hui, on aimerait bien que nos modèles puisse généraliser à partir du plus petit nombre d’exemple possible.

Le problème pourrait être résumé ainsi :

AJOUTER CITATION

Du point de vu du modèle, les images sont trop différentes les unes des autres. Autrement dit, leurs paramètres statistiques sont trop différents. 

On dit qu’il y a décalage de covariable [au sens de la distribution] (en anglais covariate shift). 


On retrouve ce même problème dans des cas plus simples que celui des réseaux de neurones profonds, comme lors de régressions linéaires. Il est apparu beaucoup plus facile de résoudre des problèmes de régression lorsque le jeu d’entraînement suit une loi normale centrée réduite (moyenne = 0, écart-type = 1) ; c’est pourquoi il est très fréquent de normaliser les valeurs d’entrées d’un modèle.

AJOUTER SCHÉMA 6.b
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/sbn_6bfr.jpg">
  <strong>Schéma 6.b : Pourquoi faut-il normaliser les valeur d’entrée d’un modèle ? cas normalisé.</strong> Le signal d’entré normalisé rend les valeurs moins éparses à l’entraînement : il sera plus facile de trouver une fonction généralisante. 
</p>

Cette solution était déjà connue et mise en pratique avant la publication de l’article qui nous intéresse ici. La couche de BN, elle, considère ce problème au niveau des couches cachées.

BONNE TAILLE DE TITRE ?
##### Le décalage de covariable interne, hypothèse défendue par l’article original


AJOUTER SCHÉMA 7
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/sbn_7fr.jpg">
  <strong>Schéma 7 : Principe du décalage de covariable (ICS)</strong> au sens de la distribution (ICSdistrib).
</p>

Dans notre exemple du classificateur de voiture, on peut envisager les couches cachées comme des unités qui s’activent lorsqu’elle identifient certaines caractéristiques “conceptuelles” associées à la voiture : par exemple une roue, un pneu, ou une portière. On peut supposer que le même phénomène précédemment décrit a lieu au niveau des couches cachées. Un pneu orienté d’une certaine façon activera un neurone selon une certaine distribution. On souhaite alors qu’un autre pneu, même orienté différemment, puisse activer le même neurone avec une distribution statistique comparable, afin que le réseau puisse en tirer des conclusions sur la probabilité que l’image de départ contienne une voiture.

Si le signal d’entré présente un grand décalage de covariable (c’est à dire si sa distribution statistique varie beaucoup d’un passage à l’autre), l’optimiseur aura plus de difficulté à généraliser - autrement dit à apprendre - à partir de caractéristiques communes. À l’inverse, en suivant une distribution proche de la loi centrée réduite, l’optimiseur pourra plus facilement approximer une fonction généralisante. Les auteurs appliquent donc la même stratégie à l’échelle des couches cachées pour aider le réseau à généraliser à des niveaux de caractéristiques plus “conceptuels”.


Néanmoins, il n’est pas souhaitable que tous nos signaux d’activations suivent une loi normal centrée réduite. Cela limiterait sa capacité de représentativité, et pour cause :


AJOUTER SCHEMA 8
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/sbn_8.jpg">
  <strong>Schéma 8 : Pourquoi il n’est pas souhaitable de contraindre l’activation à une loi normale centrée réduite.</strong> La sigmoïde ne fonctionne ici qu’en régime linéaire.
</p>


Si l’on prend l’exemple donné de l’article original, la sigmoïde, un signal d’entré compris en 0 et 1 limiterait la fonction non-linéaire à son régime … linéaire. 

Pour pallier à ce problème, les auteurs ont alors ajoutés deux paramètres, 𝛽 et 𝛾, pour permettre à l’optimiseur de définir lui même la moyenne (via 𝛽) et l’écart type (via 𝛾) optimal pour une tâche donné.

**⚠Nous arrivons au point qui est souvent l’objet de confusion.** Pendant quelques années après la sortie de l’article original, on a déduit de l’efficacité de la couche BN l’explication suivante :

TROUVER UNE MISE EN PAGE 
Hypothèse 1 : 

BN -> normalisation du signal à chaque couche cachée -> ajout de deux paramètres à ajuster pour profiter de tous les régimes d’activations -> facilite l’entraînement

Ce qui situe d’intérêt de la BN dans le fait que cette couche assure une distribution proche d’un loi normale centrée réduite, facilitant la généralisation. Ceci a été remis en question, préférant une autre explication que l’on pourrait énoncer comme suit :


Hypothèse 2 :

BN -> normalisation du signal à chaque couche cachée -> diminue l’interdépendance des couches cachées entre elles sur les paramètres statistiques -> facilite l’entraînement

Ce n’est plus tout à fait la même chose. Ici, le passage à la loi normale centrée réduite n’est plus qu’un moyen de réduire l'interdépendance des couches les unes avec les autres. Étudions cette nouvelle hypothèse.



#### 2) Deuxième hypothèse : limiter l’interdépendance de distributions 

*Note de rédaction : Ne disposant pas de preuves irréfutables, je me permets de m’appuyer très largement sur les explications de Yann Goodfellow à ce sujet (qu’il exprime dans cet brillante vidéo (https://www.youtube.com/watch?v=Xogn6veSyxA)), et sur quelques discussions en ligne citées en références.*

Considérons l’exemple suivant :

AJOUTER SCHÉMA 9
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/sbn_9fr.jpg">
  <strong>Schéma 9 : Principe simplifié d’un réseau de neurone profond,</strong> composé uniquement de transformation linéaires.
</p>



Où (a), (b), (c), (d) et (e) sont les couches successives d’un réseau de neurones. Notre cas est très simple, il s’agit d’un réseau constitué d’une succession de transformations linéaires. On cherche à entraîner ce réseau avec la méthode de descente de gradient (Stochastic Gradient Descent, SGD).

Pour calculer la mise à jour des poids de (a), on calcule le gradient en partant de la fin. On obtient :
grad(a) = b * c * d * e

On se place d’abord dans le cas d’un réseau sans couche BN. On conclut de l’équation établie ci-dessus que si tous les gradients sont grands, grad(a) aura une valeur très élevée. À l’inverse, des gradients très petits sur les couches suivantes forceront grad(a) vers une valeur presque négligeable. 

Si l’on s’intéresse au distributions statistiques qui se présentent à l’entrée de chacune de ces couches, on s’aperçoit de l’interdépendance entre les couches du réseau : Une modification des poids de (a) modifiera la distribution du signal entrant dans (b), qui aura à terme des conséquences sur celles des signaux entrant dans (d) et (e). Ceci est problématique pour la stabilisation de l’entraînement : pour ajuster la distribution statistique d’une couche, il faut tenir compte de l’ensemble de la chaîne. 

Or, la SGD est une méthode qui s’intéresse aux relations du 1er ordre (dérivée première appliquée à une couche par rapport à la précédente). Elle ne tient donc pas compte des interactions mentionnées précédemment !


AJOUTER ENCART

Approfondissement : Il existe des algorithmes qui utilisent des relations du 2e ordre pour tenir compte de ces effets. On peut par exemple prendre les dérivées 2nd pour voir comment interagit (a) avec l’ensemble des paramètres. On peut alors exploiter l’inverse de la matrice hessienne pour normaliser l’impact de chaque paramètre sur les autres. On obtient des résultats acceptables si l’on n’a que des interactions par paires, autrement dit si (a) est directement affecté par (b), (c), (d) ou (e). 

Mais si l’on calcule les dérivées de (b) par rapport aux autres paramètres, on a : 

grad(b) = c * d * e

Or, les paramètres de ces trois couches peuvent eux-même interagir entre eux !

Si on considère des méthodes du 2e ordre, comme la méthode de Newton, qui nécessite de calculer la matrice hessienne, l’algorithme d’apprentissage devient beaucoup plus lourd en calculs : la complexité devient quadratique avec le nombre de paramètres pour calculer la hessienne, on est bien loin de la complexité linéaire avec le nombre de paramètre lors de la descente de gradient. 

Tout cela pour finalement ne tenir compte que des interactions du 2e ordre. Dans notre exemple pourtant simple, nous avons des interactions du 5e ordre ! Il n’y aurait pas même de moyens directs de résoudre des interactions du 5e ordre avec un résolveur algébrique linéaire (linear algebric solver) pour gérer toute ces interactions. 

Imaginez pour des réseaux très profonds ...


AJOUTER SCHÉMA 10
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/sbn_10fr.jpg">
  <strong>Schéma 10 : Principe de l’hypothèse n°2.</strong> En normalisant puis ajustant le signal avec 𝛽 et 𝛾, la couche BN simplifie le contrôle du signal au niveau de chaque couche cachée.
</p>



Ajouter la couche BN atténue très largement l’interdépendance entre les couches pendant l’apprentissage. La normalisation agit comme une porte que l’optimiseur peut ajuster à partir des seuls paramètres 𝛽 et 𝛾. Il n’est alors plus nécessaire de tenir compte de tous les paramètres du réseau pour avoir des informations statistique sur une couche cachée.

<ins>Remarque :</ins> L’optimiseur peut alors se permettre de faire de bien plus grosses modifications de poids sur chacune des couches, sans que cela n’altère le travail réalisé sur les couches successives. Il est donc beaucoup plus facile de déterminer des hyperparamètres qui convergeront vers une solution optimale.

AJOUTER ENCARTS
Cet exemple met de côté l’hypothèse dans laquelle la BN servirait à faire tendre les valeurs d’activations des couches cachées vers une loi normale centrée réduite. 

Ici, il s’agit de faciliter le travail de l’optimiseur en lui permettant d’ajuster les distributions statistiques internes en jouant sur seulement deux paramètres à la fois.

Il s’agit néanmoins d’intuitions autour du fonctionnement de la normalisation par lot, et il n’existe pas, à ma connaissance, de solides preuves de ces hypothèses. 

Un article paru en 2019 par une équipe du MIT a apportée une contribution intéressante à la compréhension de l’efficacité de la couche BN. Les auteurs remettent très fortement en question le lien entre l’efficacité de la couche BN et la réduction du décalage de covariable interne, au sens de la distribution (première hypothèse) !


#### 3) Troisième hypothèse - lissage du paysage d’optimisation :

Note de rédaction : Dans cette partie, je m’efforce de synthétiser l’article [2], pour présenter leurs principales conclusions quant aux propriétés de la couche BN. Cet article est dense, je vous invite à vous y pencher avec plus d’attention si ces concepts vous intéressent.  

Intéressons-nous directement à la deuxième expérience de cet article. Les auteurs entraînent trois réseaux VGG (sur CIFAR-10) :
Le premier sans couche BN ;
Le deuxième avec des couches BN ;
Le troisième est identique au deuxième, à ceci prêt qu’ils ajoutent explicitement de l’ICS au niveau des couches cachées en ajoutant du bruit (valeurs aléatoires ajoutées/multipliées à la moyenne/variance) ; 

Ils observent ensuite la précision obtenue par chaque modèle, ainsi que l’évolution des distributions d’activations au niveau des couches cachées. Voici les résultats obtenus :


AJOUTER GRAPHIQUE 6
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/gbn_6.png">
  <strong>Graphique 6 : Impact de la couche BN sur l’ICSdistrib</strong> (source : [2]). Les deux réseaux qui utilisent la couche BN s’entraînent plus vite que le réseau standard ; ajouter explicitement de l’ICSdistrib sur un réseau normalisé ne détériore pas ces propriétés.
</p>


On observe que le 3e réseau a, comme prévu, un très fort ICS. Pourtant, cela ne l’empêche pas d’être entraîné de manière plus rapide et plus stable que le réseau standard. Les performances sont assez similaires au réseau avec des couches BN mais sans ajout explicit d’ICS, suggérant que l’efficacité de la BN n’est pas lié à la diminution de l’ICS, comme le soutient l’hypothèse 1.

N’écartons pas l’ICS trop vite : la définition du décalage de covariable interne (donnée dans l’article original de la couche BN) liée à la distribution est peut-être insatisfaisante. Les auteurs de [2] ont explorés une autre définition de l’ICS, cette fois-ci exprimant les propriété d’optimisation du modèle. En voici une définition :


AJOUTER ENCART
Considérons une entrée fixe à notre modèle, notée X. 

On définit le décalge de covariable interne d’un point de vu de l’optimisation (noté ICSopti ), la différence entre le gradient calculé au niveau d’une couche k après avoir rétropropagé l’erreur L(X)It, et le gradient calculé au niveau de la même couche k après la mise à jour des poids des couches précédentes L(X)It+1.


Cette définition a pour but de focaliser l’attention sur le gradient de l’erreur, plus que sur la distribution des valeurs d’activation. On cherche ainsi à s’intéresser directement au problème d’optimisation sous-jacent pour comprendre l’efficacité de la couche BN, et voir le lien que peut avoir l’ICS sur l’entraînement.

L’expérience suivante évalue cette nouvelle approche de l’ICS. Pour cela, les auteurs évaluent l’impact de la normalisation par lots sur l’ICSopti en regardant son évolution au cours de l’entraînement d’un réseau avec / sans couche BN. Pour quantifier la différence entre les gradients évoquées dans la définition de l’ICSopti , les auteurs calculent :
La différence L2 : Les gradients ont-ils une norme proche avant et après la mise à jour des poids ? Idéalement : L2-diff = 0 ;
Le cosinus de l’angle orienté : Les gradients ont-ils une direction similaire avant et après la mise à jour des poids ? Idéalement: cos(grad(k)It , grad(k)It+1) = 1 .

AJOUTER GRAPHIQUE 7
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/gbn_7.png">
  <strong>Graphique 7 : Impact de la couche BN sur l’ICSopti</strong> (source : [2]). Les différence de normes et d’angles de gradient suggère qu’elle n’empêche pas le décalage ; le phénomène semble au contraire s’aggraver.
</p>

Les résultats sont surprenants : Le réseau qui repose sur des couches de normalisation par lots a un décalage de covariable interne similaire, voir supérieur, au réseau standard. Rappelons-le, le réseau qui utilise des couche de BN (courbe bleue) s’entraîne beaucoup plus vite et converge vers une meilleure solution (courbe rouge) !

Décidément, l’ICS - dans les définitions qu’on en a donné - n’a pas l’air lié aux performances d’entraînement.

La normalisation par lots aurait donc d’autres effets sur l’entraînement, qui aboutissent à une convergence plus rapide vers une meilleure solution.

Intéressons nous directement au problème de l’optimisation : quel est l’impact de la couche BN sur le paysage d’optimisation (en anglais : optimization landscape) ?

Voici la dernière expérience que nous allons aborder dans cet article :

AJOUTER SCHÉMA 11
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/sbn_11.jpeg">
  <strong>Schéma 11 : Exploration du paysage d’optimisation</strong> dans la direction du gradient. Expérience menée dans l’article [2].
</p>

À partir d’un même gradient, on réalise la mise à jour des poids pour différents pas d’optimisation (comparable à une augmentation du taux d’apprentissage !). Intuitivement, on définit une direction à partir d’un certain point de l’hyperplan dans l’espace des paramètres, puis on explore le paysage d’optimisation en suivant cette direction de plus en plus loin. 

À chaque pas, on relève le gradient et la perte. On peut donc comparer les différents point du paysage d’optimisation avec le point de départ. Si l’on relève de fortes variations, le paysage est très instable et le gradient est incertain : de grands pas risquerait de détériorer notre optimisation. Au contraire, si les variations relevées sont petites, le paysage est stable et le gradient est plus sûr : on peut alors se permettre de plus grands pas sans compromettre l’optimisation ! Autrement dit, on peut appliquer un plus grand taux d’apprentissage, et atteindre une convergence plus rapide. Ceci étant des propriétés bien connus des utilisateurs de la couche BN…

Place aux résultats :


AJOUTER GRAPHIQUE 8
<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/gbn_8.png">
  <strong>Graphique 8 : Impact de la couche BN sur le lissage du paysage d’optimisation</strong> (source : [2]). Avec la normalisation par lots, on constate l’atténuation des fortes variations du gradient.
</p>

On peut voir très distinctement que le paysage d’optimisation est bien plus lisse avec la couche BN que sans. 

Nous tenons enfin une piste d’explication : Par un moyen ou un autre, la couche de normalization lisse de paysage d’optimisation. Le travail de l’optimiseur en est grandement facilité : on peut définir un taux d’apprentissage plus important, étant moins soumis au risque de disparition de gradient (poids bloqués sur une hypersurface plate) ou à l’explosion de gradient (poids entraîné par un minimum local abrupt).

Nous sommes à présent en mesure de formuler la troisième hypothèse, défendue par cet article [2] :


Hypothèse 3 :

BN -> normalisation du signal à chaque couche cachée -> lisse le paysage d’optimisation -> entraînement plus stable et plus rapide.


Une nouvelle interrogation s’impose : par quel moyen la normalisation par lots lisse-t-elle le paysage d’optimisation ?

Pour finir, les auteurs ont constaté que cet effet n’est pas unique à la normalisation par lots, obtenant des performances d’entraînement comparable avec d’autres formes de normalisation (par exemple la normalisation L1 ou L2). Les bonnes performances de la normalisation par lot seraient donc fortuites, mettant en oeuvre un mécanisme dont nous n’avons pas encore saisi tous les ressorts. Par ailleurs, leur article explore d’un point de vu théorique les conséquences de la normalisation par lots sur les propriétés de continuités de la fonction de coût. Ils montrent que la normalisation rend la fonction Lipschitzienne.

En définitive, cet article bat en brèche l’idée communément admise que l’efficacité de la couche BN reposerait sur l’atténuation du décalage de covariable interne (au sens de la distribution comme au sens de l’optimisation). En revanche, il souligne l’effet de lissage du paysage d’optimisation que la normalisation implique. 

Si cet article énonce une hypothèse quant à la raison pour laquelle l’entraînement est plus rapide, il n’apporte pas d’élément à propos des propriétés généralisantes de la couche BN. 

Une hypothèse à ce sujet est brièvement évoqué en fin d’article, soutenant que le lissage du paysage d’optimisation permettrait au modèle de converger vers des minimums plats, ayant de meilleures propriétés généralisantes.

Soulignons cependant que leur principal contribution est la remise en question de la vision communément admise depuis la sortie de l’article officiel - ce qui est, déjà, significatif.


#### 4) Bilan : Pourquoi la BN est efficace ? Ce que l’on sait aujourd’hui


A VERIFIER
| Hypothèse n°1 :                                                                                                            | Hypothèse n°2 :                                                                                                                                                                                            |                                                                                                      Hypothèse n°3                                                                                                      |   |   |
|----------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|---|---|
| La couche BN atténue le décalage de covariable interne (ICS)                                                               | La couche BN facilite la tâche de l’optimiseur en lui permettant d’ajuster la distribution des couches cachées à partir de 2 paramètres seulement                                                          |                                                         La couche BN reparamétrise le problème d’optimisation intrinsèque, le rendant plus stable et plus lisse                                                         |   |   |
| Faux : [2] a montré qu’en pratique, on ne distingue pas de corrélation entre cet effet et les performances d’entraînement. | Peut-être : Cette hypothèse met l’accent sur l’interdépendance des paramètres du réseau entre eux, rendant difficile l’optimisation des poids vers une solution optimale.  Pas de preuve solide néanmoins. | Encore très incertain : Leurs résultats n’ont pas encore été bousculés. Les preuves semblent encore fragiles (reposant principalement sur quelques expériences, et sur quelques éléments de démonstrations théoriques). |   |   |
|                                                                                                                            |                                                                                                                                                                                                            |                                                                                                                                                                                                                         |   |   |


De nombreuses questions demeurent, donc, et la couche BN est toujours l’objet de recherches à l’heure où j’écris ces lignes. Mais l'évaluation de ces hypothèses nous donnent une meilleur compréhension de la couche de normalisation par lots, nous éloignant des justifications erronées que l’on a eu longtemps à l’esprit. 

Ces questions ouvertes ne nous empêche cependant pas de profiter de l’efficacité des couches BN dans un réseau !


### VI) En résumé

AJOUTER ENCART
La normalization par lot (ou Batch-normalization - notée BN) constitue une des plus grande avancée liées à l’émergence de l’apprentissage profond. 

Reposant sur la succession de deux transformations linéaires, cette méthode rend les entraînements de réseaux de neurones profonds (perceptrons multicouches ou réseaux convolutifs) plus rapides et plus stables. L’intérêt majeur de cette technique réside dans le fait qu’elle atténue très largement l’impact de l’interdépendance entre les poids du réseau sur les paramètres statistiques au niveau des couches cachées. 

À l’heure où j’écris cet article, toutes les méthodes de l’état de l’art exploitent massivement cette méthode, que ce soit pour l’extraction de caractéristiques (EfficientNet(lien de l’article)), la détection d’objets (EfficientDet (lien de l’article)), la segmentation (? (lien)), … .

Si vous êtes intéressés par l’apprentissage profond, vous ne pourrez pas y couper !



### VII) Les questions en suspent

Même si la normalisation par lots a montré son efficacité en pratique depuis des années, ce concept est encore mal compris. Et si certains articles ont bousculé la compréhension largement admise pendant des années par la communauté scientifique, les mécanismes intrinsèques qui régissent ce concept restent très incertains.

En particulier, on se demande :
- Comment la normalisation par lots d’aide le réseau à généraliser plus efficacement ?
- La couche BN est-elle la meilleure solution de normalization pour faciliter l’optimisation ?
- Dans quelle mesure les paramètres 𝛽 et 𝛾 influencent le lissage du paysage d’optimisation ?
- Les expérimentations montrant l’effet de lissage de la couche BN sur le paysage d’optimisation ont réalisées dans des conditions de court-terme ; on a regardé l’évolution du gradient et de la fonction de coût à partir d’une seule itération, testant différentes longueurs de pas. Au delà de l’impact direct que ces expériences mettent en lumière, qu’en est-il sur le long terme ? L’interdépendances des poids provoque-t-elle d’autres effets remarquables sur le paysage d’optimisation ?

Cette liste n’est bien entendu pas exhaustive, et beaucoup de mystères demeurent autour de la Normalisation par lots. À suivre, donc … ;)


##### Sources et références


<ins>Articles :</ins>

[1] “Normalisation par Lots : Accélération de l’entraînement des réseaux de neurones profonds par la réduction du décalage de covariable interne”, l’article original : [article](https://arxiv.org/abs/1502.03167) 

[2] “Comment la normalisation par lots aide l’optimisation.” : [article](https://arxiv.org/pdf/1805.11604.pdf) 


Réseau Inception : [article](https://arxiv.org/abs/1409.4842 ) 


<ins>Liens :</ins>

Brillante présentation de Ian Goodfellow (malgré la qualité sonore), dont le début traite de la normalisation par lot :
https://www.youtube.com/watch?v=Xogn6veSyxA

Présentation de l’article “Comment la normalisation par lots aide l’optimisation ?” par l’un des auteurs chez Microsoft ; l’audience est incisive sur les questions, les débats déclenchés sont passionnants :
https://www.microsoft.com/en-us/research/video/how-does-batch-normalization-help-optimization/


À propos de la moyenne mobile : [lien](https://fr.wikipedia.org/wiki/Moyenne_mobile)


Expérimentation - BN après l’activation donne de meilleurs résultats qu’avant : [lien](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu)


Positionnement de la BN avant ou après l’activation : [lien](https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout)


Citation de F. Chollet à propos de la place de la BN : [lien](https://github.com/keras-team/keras/issues/1802)



-----------



###

### head3
#### head4
##### head5
###### head6




You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

Jekyll requires blog post files to be named according to the following format:

`YEAR-MONTH-DAY-title.MARKUP`

Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `MARKUP` is the file extension representing the format used in the file. After that, include the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/

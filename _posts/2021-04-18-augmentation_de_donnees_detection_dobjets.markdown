---
layout: post
title:  "Visualisation des méthodes d'Augmentation de Données pour la Détection d'objets"
date:   2021-04-18 21:50:00 +0200
categories: Apprentissage-profond
---


L'augmentation de données est une étape indispensable sitôt que l'on souhaite utiliser un réseau de neurones pour de la détection d'objet. Il s'agit d'appliquer des transformations plus ou moins complexes pour générer de nouvelles images d'entraînement à partir d'un jeu de donnée initial. Cela a pour effet de diminuer les risques de surrapprentissage, et de faciliter le travail de généralisation du réseau. Mais si l'augmentation de données est très communément exploitée, **sa mise en oeuvre repose souvent sur un processus empirique** : On réfléchit à la cohérence des transformations sur le problème ciblé, et l'on applique celles qui nous semblent les plus appropriées pour produire un contenu diversifié.

Le choix des transformations passent donc souvent par leur visualisation sur quelques exemples. Cet article s'inscrit dans le sens de cette démarche.

**Objectifs de cet article** : 
- Répertorier les **librairies** python les plus importantes pour l'augmentation de données adaptées à la détection d'objet ;
- Donner un **aperçu des méthodes** les plus utilisées ;
- Proposer une cahier jupyter permettant de manipuler les méthodes évoquées (*En cours de nettoyage*).


Bien que cet article se concentre sur la détection d'objet, il apparaît que beaucoup des méthodes évoquées sont tout aussi valable sur d'autres tâches de vision, comme la segmentation d'image, ou la classification.

<ins>Note de rédaction :</ins> La plupart des visuels ont été réalisée à partir du jeu de donnée [Global Wheat Dataset](http://www.global-wheat.com/) (que l'on nommera indifféremment GWD).

<br/>


## Tour d'horizon des librairies utiles (Python)

[Albumentation](https://github.com/albumentations-team/albumentations) : Contient une grande parties des transformations les plus couramment utilisées. Conçue pour fonctionner avec Pytorch (mais fonctionne très bien avec Keras & tf). C'est une référence, et à ma connaissance la lib [la mieux optimisée](https://github.com/albumentations-team/albumentations#benchmarking-results).

[imgaug](https://imgaug.readthedocs.io/en/latest/) : Faite pour la parallélisation sur plusieurs CPUs, mais à ma connaissance, rien de ce que fait imgaug n'est pas fait aussi bien par albumentation.

Transformations incluses dans les environnement de développement : [Pytorch](https://pytorch.org/vision/stable/transforms.html), [Keras](https://keras.io/api/preprocessing/image/), [Mxnet](https://mxnet.apache.org/versions/1.8.0/api/python/docs/api/mxnet/image/index.html#mxnet.image.Augmenter)

[Augmentor](https://github.com/mdbloice/Augmentor) : Librairie surtout axée sur les transormations géométriques. Elle n'est cependant pas assez complète / rapide / modulable pour être très populaire.

[Deepaugment](https://github.com/barisozmen/deepaugment) : Implémentation de l'approche [AutoAugment](https://arxiv.org/pdf/1805.09501.pdf), qui est une sorte d'autoML pour l'augmentation du données : l'algorithme recherche dans un espace d'augmentation possible la combinaison qui maximise une certaine métrique (la précision, par exemple). Peut utilisé, car les gains de performances sont souvent minimes et très coûteux.

À noter que Pytorch et MxNet ont chacun une libraire intitulée "Transforms" (respectivement dans torchvision et gluon.data.vision) qui contient un certain nombre de transformation très rapide à l'exécution. Les possibilitiés sont cependant limitées si l'on ne s'en tient qu'à ces opérations.

<br/>

## Arborescence des catégories de transformations


<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/augmentation_dimages.jpg" alt="arbo_augmentation_dimages" width="500">
  Adapté depuis : https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/augmentation_dimages.jpg" alt="arbo_augmentation_dimages" width="1000">
  Adapté depuis : https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/augmentation_dimages.jpg" alt="arbo_augmentation_dimages" width="1200">
  Adapté depuis : https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0
</p>




<br/>

## 1. Manipulations d'images  classiques 


### 1.1 Transformations géométriques

- **Transformation horizontale** (*HorizontalFlip* dans albumentation)

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/transfo_horiz_ex.png" alt="transfo_horiz" width="500">
</p>

<br/>


- **Transformation verticale** (*VerticalFlip* dans albumentation)

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/transfo_verti_ex.png" alt="transfo_verti" width="500">	
</p>

<br/>


- **Rotation aléatoire (pas de 90°)** (*RandomRotate90* dans albumentation)

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/rotation_aleatoire_ex.png" alt="transfo_verti" width="500">
</p>

<br/>



- **Translation, rotation, échelle** (*ShiftScaleRotate* dans albumentation)

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/translat_rota_echelle_ex.png" alt="transfo_verti" width="500">
</p>

<br/>



- **Compression d'image** (*ImageCompression* dans albumentation) (dans cette section ?)

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/compression_dimage_ex.png" alt="transfo_verti" width="500">
</p>

<br/>




### 1.2 Transformations d'espace colorimétrique



- **Décalage RVB** (*RGBShift* dans albumentation)

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/decalage_rvb_ex.png" alt="transfo_verti" width="500">
</p>

<br/>


- **Gamma aléatoire** (*RandomGamma* dans albumentation) ( ? )

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/gamma_alea_ex.png" alt="transfo_verti" width="500">
</p>

<br/>


- **Contraste de luminosité aléatoire** (*RandomBrightnessContrast* dans albumentation) (dans cette section ?)

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/contraste_lumino_alea_ex.png" alt="transfo_verti" width="500">
</p>

<br/>


- **Saturation de teinte aléatoire** (*HueSaturationValue* dans albumentation)

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/saturation_teinte_alea_ex.png" alt="transfo_verti" width="500">
</p>

<br/>


- **Transfert de couleur** (article [ColorTransfer](https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf))

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/transfert_couleur_ex.png">
</p>

<br/>


### 1.3 Mélange d'image

Ces méthodes récentes s'avèrent être très efficaces sur certaines tâches. On peut voir que les images générées reproduisent un certain nombre de contraintes liées au contexte de détection en champs de blé (notamment des scènes très denses avec occlusions).


- **Patchwork d'images** (article [cutmix](https://arxiv.org/abs/1905.04899))

Il existe également des variantes avec deux images (juxtaposition horizontale, ou verticale).

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/patchword_dimage_ex.png">
</p>

<br/>


- **Mélange d'images** (article [mixup](https://arxiv.org/abs/1710.09412))

Il serait pertinent de diminuer le taux de confiance associé à la rétropropagation de ces exemples. En pratique, j'ai toujours vu cette augmentation être utilisée au cours des premières phases d'entraînement. On s'assure alors que les dernières phases ne contiennent que des augmentations n'alterant pas la vraisemblance des images pour affiner l'apprentissage sur des exemples de la meilleure qualité possible.


<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/melange_dimage_ex.png">
</p>

<br/>


- **Mosaïque d'images** (article [yolo-v4](https://arxiv.org/pdf/2004.10934.pdf))

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/mosaique_dimage_ex.png">
</p>

<br/>



### 1.4 Effacement aléatoire


- **Décrochage grossier** (*CoarseDropout* dans albumentation) ( corriger ? )





- **Découpage avec nombre minimal de boite englobante** (vu pour la première fois [ici](https://www.kaggle.com/c/global-wheat-detection/discussion/172569); s'il existe une source antérieure, merci de me la faire parvenir !)

Approche intéressante pour les scènes très denses, comme dans GWD. Le découpage permet d'obtenir des images tout à fait cohérentes. Combinée avec d'autres augmentations, le contenu généré est très intéressant pour aider à la généralisation.

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/decoupe_nboites_min_ex.png" alt="transfo_verti" width="500">
</p>

<br/>


### 1.5 Noyaux

- **Bruit gaussien** (*GaussNoise* dans albumentation)

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/bruit_gauss_ex.png" alt="transfo_verti" width="500">
</p>

<br/>


- **Flou** (*Blur* dans albumentation)

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/flou_ex.png" alt="transfo_verti" width="500">
</p>

<br/>


- **Flou cinétique** (*MotionBlur* dans albumentation)

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/flou_cinetique_ex.png" alt="transfo_verti" width="500">
</p>

<br/>



## 2. Apprentissage profond


### 2.1 Exemples adverses

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/adversarial_attack_paper_img.png">
  <br/>
  <a href="https://arxiv.org/pdf/1412.6572.pdf">(Source)</a>
</p>

<br/>


Les attaques adverses ([articles](https://arxiv.org/pdf/1312.6199.pdf) [pionniers](https://arxiv.org/pdf/1412.6572.pdf)) pour les réseaux de neurones consistent à modifier très légèrement nos données d'entrées (dans notre cas une image) de sorte à provoquer de très fortes erreurs de prédiction sur un réseau. Une idée d'augmentation de donnée apparaît alors clairement : si l'on soumet un réseau à des exemples adverses, sans doute y sera-t-il plus robuste. Il devrait donc pouvoir en tirer des conclusions sur les données pour généraliser plus efficacement.

Si l'idée est attrayante, elle ne fonctionne pas très bien en pratique. Yann Goodfellow a lui-même affirmé que cette approche ne pouvait apporter que des améliorations marginales (dans [cette interview](https://www.youtube.com/watch?v=Z6rxFNMGdn0), me semble-t-il).

Une meilleure compréhension des attaques adverses dans les réseaux de neurones pourrait peut-être rendre cette piste plus viable.



### 2.2 Transfert de style

Il s'agit d'appliquer un [transfert de style neuronal](https://arxiv.org/pdf/1508.06576.pdf) sur les images de notre jeu de données.

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/transfert_style_neuronal_ex.png">
</p>


<br/>


Note : [Cette étude](https://arxiv.org/pdf/1705.04058.pdf) explore des différentes approches de transfert de style par réseaux de neurones (évoque également quelques approches de type GAN) : [Jing, Y., Yang, Y., Feng, Z., Ye, J., Yu, Y., & Song, M. (2019). Neural style transfer: A review. IEEE transactions on visualization and computer graphics, 26(11), 3365-3385.


### 2.3 GAN

Article de la méthode *pix2pix* : [Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1125-1134).](https://arxiv.org/pdf/1611.07004.pdf)


Piste prometteuse, qui a fait l'objet de [nombreux](https://arxiv.org/pdf/1711.04340.pdf) [travaux](https://arxiv.org/pdf/1801.05401.pdf) [de](https://arxiv.org/pdf/1803.01229.pdf) recherches et a déjà été [utilisée](https://arxiv.org/pdf/1907.12902.pdf) avec [succès](https://www.nature.com/articles/s41598-019-52737-x.pdf). Cette approche nécessite cependant des GANs bien entraîné, faute de quoi les images produites ne seront pas d'une qualité suffisante pour aider à la généralisation. 



Voici quelques examples d'images générées par la méthodes pix2pix sur le jeu de donnée GWD :

*Crédit : Ces images pix2pix ont été publiées par [bendang sur Kaggle](https://www.kaggle.com/bendang/synthetic-wheat-images)*


<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/single_pix2pix_gwd.jpg" alt="mosaic_pix2pix_gwd" width="1000">
  À partir d'images simples
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/mosaic_pix2pix_gwd.jpg" alt="mosaic_pix2pix_gwd" width="1000">
  À partir d'images mosaïques (voir plus haut)
</p>


Comme on peut le voir, les résultats peuvent être d'une qualité variable. Mais les résultats spectaculaires obtenus par les GANs ces dernières années laissent supposer que cette piste sera de plus en plus (et de mieux en mieux) exploitée à l'avenir.



## Conclusion

Il existe donc une multitude de méthodes pour générer de nouvelles données à partir d'un jeu initial, dans des problèmes de détection d'objets. En combinant des méthodes fondées sur les réseaux de neurones profonds (souvent en amont de l'entraîenement) et des méthodes de traitement d'images plus classiques (moins coûteuses en calcul, donc souvent mises en oeuvre au moment de l'entraîenement), on peut obtenir un jeu de donnée d'entraînement suffisamment divers pour augmenter les capacités généralisantes d'un réseau de neurones.

On aurait pu mentionner les augmentations par auto-encodeurs, avec lesquels on peut augmenter la résolution des images. Les augmentations de données dans l'espace des caractéristiques serait une autre pise à considérer à l'avenir ; bien que [certaines travaux](https://arxiv.org/pdf/1702.05538.pdf) explorent cette approche, elle n'est pas encore couramment employée.



<br/>

## Sources

**Liens :**

[https://neptune.ai/blog/data-augmentation-in-python](https://neptune.ai/blog/data-augmentation-in-python)


**Articles :**

[Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. Journal of Big Data, 6(1), 1-48.](https://link.springer.com/article/10.1186/s40537-019-0197-0?code=a6ae644c-3bfc-43d9-b292-82d77d5890d5)

[Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576.](https://arxiv.org/pdf/1508.06576.pdf)

[Jing, Y., Yang, Y., Feng, Z., Ye, J., Yu, Y., & Song, M. (2019). Neural style transfer: A review. IEEE transactions on visualization and computer graphics, 26(11), 3365-3385.](https://arxiv.org/pdf/1705.04058.pdf)








---
layout: post
title:  "Gradient de la politique"
date:   2021-04-18 21:50:00 +0200
categories: Apprentissage-profond
---


L'augmentation de données est une étape indispensable sitôt que l'on souhaite utiliser un réseau de neurones pour de la détection d'objet sur une tâche précise. Cette méthode permet générer de nouvelles images d'entraînement à partir d'un jeu de donnée initial. Un jeu de donnée plus étendu rend le travail de généralisation du réseau plus facile, en diminuant les risques de surrapprentissage. Mais si l'augmentation de données est très communément exploitée, **sa mise en oeuvre repose souvent sur un processus empirique** : On réfléchit à la cohérence des transformations sur le problème ciblé, et on applique celles qui nous semblent les plus à même de produire un contenu diversifié.

Le choix des transformations passent donc souvent par les yeux : on visualise sur quelques exemples, et choisit. Dans cet article, nous allons voir **à quoi ressemble chaque transformation**, **comment les implémenter**, ou quelles sont les **librairies** qui peuvent nous faciliter la tâche.

Notons enfin que cet article se concentre sur la détection d'objet, mais que beaucoup de ces opérations sont tout aussi valable sur d'autres tâches de vision, comme la segmentation d'image, ou la classification.

Note de rédaction : La plupart des visuels a été réalisée à partir du jeu de donnée [Global Wheat Dataset](http://www.global-wheat.com/). J'ajouterai le code sitôt que je l'aurais nettoyé.

</br>


## Tour d'horizon des librairies utiles (Python)

[Albumentation](https://github.com/albumentations-team/albumentations) : Contient une grande parties des transformations les plus couramment utilisées. Conçue pour fonctionner avec Pytorch (mais fonctionne très bien avec Keras & tf). C'est une référence, et à ma connaissance la lib [la mieux optimisée](https://github.com/albumentations-team/albumentations#benchmarking-results).

[imgaug](https://imgaug.readthedocs.io/en/latest/) : Faite pour la parallélisation sur plusieurs CPUs, mais à ma connaissance, rien de ce que fait imgaug n'est pas fait aussi bien par albumentation.

Transformations incluses dans les environnement de développement : [Pytorch](https://pytorch.org/vision/stable/transforms.html), [Keras](https://keras.io/api/preprocessing/image/), [Mxnet](https://mxnet.apache.org/versions/1.8.0/api/python/docs/api/mxnet/image/index.html#mxnet.image.Augmenter)

[Augmentor](https://github.com/mdbloice/Augmentor) : Librairie surtout axée sur les transormations géométriques. Elle n'est cependant pas assez complète / rapide / modulable pour être très populaire.

[Deepaugment](https://github.com/barisozmen/deepaugment) : Implémentation de l'approche [AutoAugment](https://arxiv.org/pdf/1805.09501.pdf), qui est une sorte d'autoML pour l'augmentation du données : l'algorithme recherche dans un espace d'augmentation possible la combinaison qui maximise une certaine métrique (la précision, par exemple). Peut utilisé, car les gains de performances sont souvent minimes et très coûteux.

À noter que Pytorch et MxNet ont chacun une libraire intitulée "Transforms" (respectivement dans torchvision et gluon.data.vision) qui contient un certain nombre de transformation très rapide à l'exécution. Les possibilitiés sont cependant limitées si l'on ne s'en tient qu'à ces opérations.

</br>

## Arborescence des catégories de transformations


(Schéma en français de : https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0)



</br>

## 1) Manipulations classiques d'images 


### Transformation géométriques

- **Transormation horizontale** (*HorizontalFlip* dans albumentation)

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/augmentation_dimages/transfo_horiz_ex.png">
</p>



- **Transormation verticale** (*VerticalFlip* dans albumentation)


- **Rotation aléatoire** (*RandomRotate90* dans albumentation)


- **Translation, rotation, échelle** (*ShiftScaleRotate* dans albumentation)


- **Compression d'image** (*ImageCompression* dans albumentation) (dans cette section ?)





### Tranformation d'espace colorimétrique


- **Contraste de luminosité aléatoire** (*RandomBrightnessContrast* dans albumentation) (dans cette section ?)


- **Saturation de teinte aléatoire** (*HueSaturationValue* dans albumentation)


- **Décalage RVB** (*RGBShift* dans albumentation)


- **Gamma aléatoire** (*RandomGamma* dans albumentation) ( ? )


- **Égalisation adaptative d'histogramme sous contrainte de contraste** (*CLAHE* dans albumentation) ( int8? )
CLAHE (Contrast Limited Adaptive Histogram Equalization)


- **Égalisation adaptative d'histogramme sous contrainte de contraste** (article [ColorTransfer](https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf))



### Mélange d'image


- **Patchwork d'images** (article [cutmix](https://arxiv.org/abs/1905.04899))


- **Mélange d'images** (article [mixup](https://arxiv.org/abs/1710.09412))


- **Mosaïque d'images** (article [yolo-v4](https://arxiv.org/pdf/2004.10934.pdf))




### Effacement aléatoire


- **Bruit gaussien** (*GaussNoise* dans albumentation)


- **Flou** (*Blur* dans albumentation)


- **Décrochage grossier** (*CoarseDropout* dans albumentation) ( corriger ? )


- **Découpage avec nombre minimal de boite englobante** (vu pour la première fois [ici](https://www.kaggle.com/c/global-wheat-detection/discussion/172569); s'il existe une source antérieure, merci de me la faire parvenir !)


</br>

## 2) Apprentissage profond



### Attaques adverses

(ajouter img)

Les attaques adverses ([articles](https://arxiv.org/pdf/1312.6199.pdf) [pionniers](https://arxiv.org/pdf/1412.6572.pdf)) pour les réseaux de neurones consistent à modifier très légèrement nos données d'entrées (dans notre cas une image) de sorte de provoquer de très fortes erreurs de prédiction sur un réseau. Une idée d'augmentation de donnée apparaît alors clairement : si l'on soumet un réseau à des exemples adverses, sans doute y sera-t-il plus robuste. Il devrait donc pouvoir en tirer des conclusions sur les données pour généraliser plus efficacement.

Si l'idée est attrayante, elle ne fonctionne pas très bien en pratique. Yann Goodfellow a lui-même affirmé que cette approche ne pouvait apporter que des améliorations marginales (dans [cette interview](https://www.youtube.com/watch?v=Z6rxFNMGdn0), me semble-t-il).

Une meilleure compréhension des attaques adverses dans les réseaux de neurones pourrait peut-être rendre cette piste plus viable.



### Transfert de style neuronal

Article original : [Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576.](https://arxiv.org/pdf/1508.06576.pdf)

Étude des différentes approches de transfert de style par réseaux de neurones (évoque également quelques approches de type GAN) : [Jing, Y., Yang, Y., Feng, Z., Ye, J., Yu, Y., & Song, M. (2019). Neural style transfer: A review. IEEE transactions on visualization and computer graphics, 26(11), 3365-3385.](https://arxiv.org/pdf/1705.04058.pdf)



		


### Augmentation de données GAN

Article de la méthode *pix2pix* : [Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1125-1134).](https://arxiv.org/pdf/1611.07004.pdf)


Piste très prometteuse.



</br>

## Source 

[https://neptune.ai/blog/data-augmentation-in-python](https://neptune.ai/blog/data-augmentation-in-python)


**Articles :**







---
layout: post
title:  "Expériementations IA : Bonnes pratiques"
date:   2021-04-09 21:07:00 +0200
categories: Apprentissage-profond
---


Mémento sur les bonnes pratiques glanées ça et là, pour l'expérimentation en apprentissage automatique / profond (ML, DL).



## Visualisation et suivi (Plotting & Logging)

### 1) L'entraînement d'un modèle et la visualisation des résultats doivent faire l'objet de scripts séparés

En ML/DL, on est amené à mettre en oeuvre des expérientations coûteuses en temps, et a acquérir un volume de donnée conséquent. Il est fréquent que l'on ait besoin de manipuler un peu les données mesurées pour les visualiser, et en tirer le maximum d'information. L'idéal est donc de séparer les scripts d'entraînement et de visualisation.


### 2) Surrestimer la quantité de données à exporter en log

Difficile de prévoir à l'avance les informations dont on va avoir besoin pour cerner un problème, distinguer un phénomène, ou plus généralement comprendre ce qu'il se passe à l'entraînement d'un réseau. Il est donc conseillé d'établir un logging clair et lisible, qui comprend les éléments suivants :

- Les données clefs à exporter, variant selon le domaine et la problème étudié (Vision, Apprentissage par renforcement, ...) :
	- Métriques principales : Erreur / Précision et Rappels / Récompense moyenne à chaque itération, ...
	- Métriques secondaires : Magnitude des gradients, Erreur de bellman, ...
	- Quelques examples de prédictions / trajectoires (pour en permettre la visualisation)

- Accompagnées de méta-informations :
	- Date et heure
	- Taux d'apprentissage, ... 





## Stochasticité


### 3) Comparer les résultats obtenus avec plusieurs graines d'aléatoire sitôt qu'une composante stochastique entre en jeu dans un modèle

...






## Sources :

[CS294-112 Deep Reinforcement Learning Plottingand Visualization Handout (UC Berkeley)](http://rail.eecs.berkeley.edu/deeprlcourse/static/misc/viz.pdf)







---
layout: post
title:  "Revue d'article : Mesures de la qualité de saisie en robotique"
date:   2021-09-16 08:07:00 +0200
categories: Robotique
---


Revue de l'article : Roa, M. A., & Suárez, R. (2015). Grasp quality measures: review and performance. Autonomous robots, 38(1), 65-88.



#### Résumé

Pour qu’un algorithme obtienne une bonne saisie, il faut **déterminer de bons points de contacts avec l’objet**, et **déterminer une bonne configuration du préhenseur**.

La quantification d’une “bonne” saisie nécessite la définition de métriques appropriées.

Dans la littérature, on distingue deux groupes de métriques, selon l’aspect évalué : **localisation des points de contact sur l’objet**, et **configuration du préhenseur**.

Cette article présente : 
- une étude de la littérature
- descriptions des méthodes issues des deux groupes de métriques, ainsi que celles qui exploitent les deux
- des études relatives à la main humaine, et aux performances de saisie


<ins>Note de rédaction :</ins> On utilisera indifféremment les mots **préhenseur** et **main** dans cet article.

<br/>

### 1) Introduction

#### 1.1) Comment déterminer une bonne saisie ?

Pour déterminer les bons points de contacts et la bonne configuraztion de préhenseur nécessaires à la saisie, on distingue deux approches :
* **Approche empirique**
	* Approche physiologique
	* Tente d’imiter le comportement de la main humaine
	* Utilise : apprentissage par démonstrations, NN, fuzzy logic (i), systèmes basées connaissances

* **Approche analytique**
	* Approche mécanique
	* Considère les propriétés mécaniques et physiques impliquées dans le geste de saisie.
	* Utilise : modèles mathématiques de l’interaction objet / main	


#### 1.2) Quelles sont les propriétés d'une bonne saisie ?

Les algorithmes de saisie tiennent compte des propriétés suivantes :

* **Résistances aux perturbations** (*Disturbance resistance*) → Traité par l'article

Une saisie doit pouvoir résister à des perturbations dans n’importe quelle directions à partir du moment où l’immobilité de l’objet est garantie (saisie réalisée), grâce à la positions des doigts, ou - jusqu’à une certaine magnitude - grâce aux forces appliquées par les doigts.
<ins>Pb :</ins> détermination des points de contacts sur l’objet

* **Dextérité** (*Dexterity*) → Traité par l'article

Il y a dextérité dans la saisie si la main peut déplacer l’objet autant que nécessaire pour la réalisation de la tâche (ou dans n’importe quelle direction s’il n’y a pas assez de spécifications).
<ins>Pb :</ins> détermination de la configuration de la main

* **Équilibre** (*Equilibrium*)

Une saisie est à l’équilibre si la résultante des forces et des couples appliqués sur l’objet (par les doigts ou par n’importe quelle perturbation) est nulle.
<ins>Pb :</ins> détermination et contrôle des forces de contact appropriées

* **Stabilité** (*Stablity*)

Une saisie est stable si n’importe quelle erreur causée dans la position de l’objet par une perturbation extérieure disparaît après la fin de la perturbation.
<ins>Pb :</ins> contrôle des forces vouées à corriger l’erreur


En général, il existe plusieurs saisies possible d'un objet par un préhenseur.
Le choix d'une saisie optimale nécessite une **métrique de qualité** (*quality measure*).


#### 1.3) Plan
```
2) Formalisme requis
3) Mesures de qualité associés aux positions de points de contact
4) Mesures de qualité associés aux configurations du préhenseur / de la main
5) Approche combinant points de contacts et configuration de main
6) Autres approches
7) Discussion
```

<br/>

### 2) Formalisme requis

#### 2.1) Modélisation des contacts, positions, forces et vitesses

Les forces aux points de contacts ne peuvent agir que contre l'objet (contraites positives).
Le nombre r de composantes indépendantes des torseurs appliqués aux points de contacts dépendent du type de chaque contrainte.

* **Contact ponctuel sans friction** : les forces appliquées sont normales à la surface de contact. r=1
* **Contact ponctuel avec frictions** (*hard contact*) : composante normale, et peuvent avoir une composante tangentielle. r=2 (2D) ou r=3 (3D)
* **Contact doux** (*soft contact*) : Idem que prec., plus un couple autour de la direction normale à la surface de contact. r=4

Une force <img src="https://latex.codecogs.com/svg.image?F_i"/> appliquée sur un objet en un point <img src="https://latex.codecogs.com/svg.image?p_i"/> génère un couple <img src="https://latex.codecogs.com/svg.image?\tau_i&space;=&space;p_i&space;\times&space;F_i"/> par rapport au centre de masse de l'objet (CM). La force et le couple sont groupés dans un torseur d'efforts <img src="https://latex.codecogs.com/svg.image?\omega_i"/>, de dimension d=3 en 2D et d=6 en 3D.

Le mouvement d'un objet est décrit par la vitesse de translation <img src="https://latex.codecogs.com/svg.image?v"/> de son CM, et par sa vitesse de rotation <img src="https://latex.codecogs.com/svg.image?w"/>. Chaque vitesse est resprésentée par un vecteur cinématique <img src="https://latex.codecogs.com/svg.image?\dot{x}=(v,w)^T&space;\in&space;\mathbb{R}^d"/>.

La force <img src="https://latex.codecogs.com/svg.image?f_i"/> au bout du doigt i est produit par les couples <img src="https://latex.codecogs.com/svg.image?T_{ij}"/>, <img src="https://latex.codecogs.com/svg.image?j=1,...,m"/>, m étant le nombre d'articulations. Pour une main de n doigts, un vecteur <img src="https://latex.codecogs.com/svg.image?T=[T_{1j}^{T}...T_{nj}^{T}]&space;\in&space;\mathbb{R}^{nm}"/> est défini pour grouper les couples appliqués sur chaque articulations de la main. Les vitesse au niveau des articulations de la main <img src="https://latex.codecogs.com/svg.image?\dot{\theta}_{ij}"/>, sont aussi groupées dans une unique vecteur <img src="https://latex.codecogs.com/svg.image?\dot{\theta}=[\dot{\theta}_{1j}^{T}...\dot{\theta}_{nj}^{T}]^T&space;\in&space;\mathbb{R}^{nm}"/>.

Les forces et les vitesses aux bouts des doigts peuvent être exprimées dans un référentiel local. De plus, le vecteur <img src="https://latex.codecogs.com/svg.image?f=[f_{1k}^T...f_{nk}^T]\in\mathbb{R}^{nr}(k=1,...,r)"/> regroupe toutes les composantes de forces appliquées aux points de contacts, et le vecteur <img src="https://latex.codecogs.com/svg.image?\nu&space;=[\nu_{1k}^T...\nu_{nk}^T]\in\mathbb{R}^{nr}"/> contient toutes les composantes de vitesses aux bouts des doigts. 


#### 2.2) Relation forces / vitesses

Les forces <img src="https://latex.codecogs.com/svg.image?f"/> et les vitesses <img src="https://latex.codecogs.com/svg.image?\nu"/> aux bouts des doigts sont reliés aux couples <img src="https://latex.codecogs.com/svg.image?T"/> et aux vitesses <img src="https://latex.codecogs.com/svg.image?\dot{\theta}"/> aux articulations des doigts par la jacobienne de la main, <img src="https://latex.codecogs.com/svg.image?J_h&space;=&space;diag[J_1,...,J_i]&space;\in&space;\mathbb{R}^{nr\times&space;nm}"/>, où <img src="https://latex.codecogs.com/svg.image?J_i&space;\in&space;\mathbb{R}^{n\times&space;m},&space;i=1,...,n"/> est la jacobienne pour le doigt i qui relie les variables des articulations des doigts avec les variables du bout des doigts :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?(1):&space;\nu=J_h\dot{\theta}"/>
</p>
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?(2):&space;T&space;=&space;J_h&space;f"/>
</p>

La relation entre les forces <img src="https://latex.codecogs.com/svg.image?f"/> au bout des doigts et le torseur d'effort appliqué à l'objet d'une part, et la relation entre les vitesse <img src="https://latex.codecogs.com/svg.image?\nu"/> aux points de contacts et le vecteur cinématique <img src="https://latex.codecogs.com/svg.image?\dot{x}"/> d'autre part, sont données par la matrice de saisie <img src="https://latex.codecogs.com/svg.image?G\in\mathbb{R}^{d&space;\times&space;nr}"/> :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?(3):&space;\nu=G^T\dot{x}"/>
</p>
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?(4):&space;\omega=Gf"/>
</p>


À partir de (1) et de (3), on peut relier les vitesses articulaires et les vitesses de l'objet sous la forme suivante (**contrainte fondamentale de saisie**) :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?(5):&space;J_h\dot{\theta}&space;=&space;G^T\dot{x}"/>
</p>

Et à partir de (3), on peut obtenir la relation entre la vitesse de l'objet en fonction des vitesse aux points de contact :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?(6):&space;\dot{x}=(G^T)^&plus;\nu&plus;N(G^T)\nu_0" />
</p>

Où <img src="https://latex.codecogs.com/svg.image?(G^T)^&plus;"/> est la pseudoinverse de <img src="https://latex.codecogs.com/svg.image?G^T"/>, <img src="https://latex.codecogs.com/svg.image?N(G^T)"/> est la matrice dont les colonnes forment une base pour le noyau de <img src="https://latex.codecogs.com/svg.image?G^T"/>, et <img src="https://latex.codecogs.com/svg.image?\nu_0"/> est un vecteur arbitraire qui paramétrise l'ensemble des solutions. La pseudoinverse est nécessaire, car <img src="https://latex.codecogs.com/svg.image?G^T\in&space;\mathbb{R}^{nr\times&space;d}"/> n'est généralement pas une matrice carré. Pour produire n'importe quel effort ou torsion **(i)** sur un objet, il est nécessaire d'avoir <img src="https://latex.codecogs.com/svg.image?N(G^T)=0"/>, ou <img src="https://latex.codecogs.com/svg.image?rang(G)=d"/> (ce qui simplifie (6) pour réduire l'expression à : <img src="https://latex.codecogs.com/svg.image?\dot{x}=(G^T)^&plus;\nu"/>).

La transformation directe dans le domaine des vitesses d'un espace articulaire de la main en grandes dimensions vers un espace de l'objet en plus petites dimensions peut être obtenu par la jacobienne <img src="https://latex.codecogs.com/svg.image?H"/> de l'objet-main :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?(7):&space;\dot{x}=H\dot{\theta}"/>
</p>

Où <img src="https://latex.codecogs.com/svg.image?H=(G^T)^&plus;J_h\in&space;\mathbb{R}^{d&space;\times&space;nm}"/>.


<ins>Note :</ins> On se place dans une approche quasi-statique, étant donné que les dynamiques ne sont pas considérées comme  jouant un rôle majeur dans les tâches de saisies. (Certains travaux explorent des cas de saisies / manipulations dynamiques.)

<ins>Note :</ins> On suppose ici que chaque doigt a un mobilité totale dans l'espace de la tâche : on ne se place pas dans le cas de système défectueux (*defective systems*) à mobilité réduites.



<p align="center">
	<img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/mesure_qualite_saisie/schema_relation_forces_vitesses.png">
	Schéma 1. Relations entre les forces de saisies et les domaines de vitesses
</p>



<br/>

### 3) Mesures de qualité associés aux positions de points de contact

#### 3.1) Mesures fondées sur des propriétés algébriques de la matrice de saisie G

**3.1.1) Valeur singulière minimale de G**

Une matrice de rang complet <img src="https://latex.codecogs.com/svg.image?G&space;\in&space;\mathbb{R}^{6&space;\times&space;r}"/> a 6 valeurs singulières dominées par les racines carrés des valeurs propres de <img src="https://latex.codecogs.com/svg.image?GG^T"/>.

Lorsqu'une saisie est en configuration singulière, au moins une des valeurs de G vaut 0. Dans ce cas, la saisie perd la capacité à resister à une contrainte extérieure dans au moins une direction.

La plus petite valeur singulière de la matrice de saisie G, <img src="https://latex.codecogs.com/svg.image?\sigma_{min}(G)" title="\sigma_{min}(G)" />, est une mesure de qualité qui indique si la configuration de saisie est proche d'une configuration singulière ou non. 

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?(8):&space;Q_{MSV}&space;=&space;\sigma_{min}(G)"/>
</p>

**Plus <img src="https://latex.codecogs.com/svg.image?Q_{MSV}"/> est élevé, meilleure est la saisie.**

Plus <img src="https://latex.codecogs.com/svg.image?Q_{MSV}"/> est élevé, plus grande est la contribution minimum (gain de transmission) des forces <img src="https://latex.codecogs.com/svg.image?f_i" /> aux points de contact vers le torseur <img src="https://latex.codecogs.com/svg.image?\omega" /> sur l'objet - aussi un critère d'optimisation de saisie (voir [1]).

(+) Indicateur critique : la saisie risque t-elle de ne pas pouvoir être réalisée en pratique ?

(-) <img src="https://latex.codecogs.com/svg.image?Q_{MSV}"/> n'est pas invariant aux changements de référentiels utilisés pour calculer les couples.

<br/>

**3.1.2) Volume de l'ellipsoïde de l'espace du torseur**

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?(4):&space;\omega=Gf"/>
</p>

Les effets de G sur (4) peuvent être visualisés ainsi : (4) projette une sphère de rayon unitaire du domaine des forces aux points de contact vers une ellispoïde dans l'espace des efforts.

La contribution globale de toutes les forces de contact peut être considérée en utilisant le volume de cette ellipsoïde comme mesure de qualité.

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?Q_{VEW}&space;=&space;\sqrt{det(GG')}&space;=&space;\sigma_1&space;\sigma_2&space;...&space;\sigma_d"/>
</p>

Où <img src="https://latex.codecogs.com/svg.image?\sigma_1&space;\sigma_2&space;...&space;\sigma_d"/> sont les valeurs singulières de G (toutes sont donc considérées avec le même poids).

**<img src="https://latex.codecogs.com/svg.image?Q_{VEW}"/> doit être maximisé pour obtenir une saisie optimale.**

(+) Invariant aux changements de référentiels pour les couples

(-) Ne précise pas la contribution relative de chaque doigt

<br/>

**3.1.3) Indice d'isotropie de la saisie**

Ce critère cherche une contribution uniforme des forces de contact au torseur global appliqué à l'objet.

L'objectif est d'obtenir une saisie isotropique, dans laquelle chaque force de contact contribue aux forces internes de l'objet d'une manière similaire.

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?(10):&space;Q_{GII}&space;=&space;\frac{\sigma_{min}(G)}{\sigma_{max}(G)}"/>
</p>

Où sont utilisées les valeur singulières minimales et maximales de G.

La saisie est isotropique si <img src="https://latex.codecogs.com/svg.image?Q_{GII}&space;\approx&space;1&space;"/>, et est proche d'une configuration singuluère si <img src="https://latex.codecogs.com/svg.image?Q_{GII}&space;\approx&space;0"/>.

(+) <img src="https://latex.codecogs.com/svg.image?Q_{GII}"/> indique si une saisie a un comportement équivalent dans n'importe quelle direction : saisies robustes pour n'importe quel usage.

(+) Indicateur critique : la saisie risque t-elle de ne pas pouvoir être réalisée en pratique ?

<br/>

#### 3.2) Mesures fondées sur les relations géométriques

**3.2.1) Forme du polygône de saisie**

En saisie planaire (lorsque les points de contact sont coplanaires), on veut que les points soient uniformément distribués sur la surface de l'objet pour améliorer la stabilité de saisie.

La moyenne des différences à la moyenne des angles internes du polygone de saisie est un critère de qualité de saisie.

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?(11):&space;Q_{SGP}=\frac{1}{\theta_{max}}\sum_{i=1}^{n}\left|&space;\theta_i&space;-&space;\bar{\theta}&space;\right|" title="(11): Q_{SGP}=\frac{1}{\theta_{max}}\sum_{i=1}^{n}\left| \theta_i - \bar{\theta} \right|" />
</p>

Avec <img src="https://latex.codecogs.com/svg.image?\bar{\theta}&space;=&space;\frac{180(n-2)}{n}"/>, et <img src="https://latex.codecogs.com/svg.image?\theta_{max}&space;=&space;(n-2)(180-\bar{\theta})&space;&plus;&space;2&space;\bar{\theta}"/>. <img src="https://latex.codecogs.com/svg.image?\theta_{max}" title="\theta_{max}" /> correspond à la somme des angles internes lorsque le polygône a la configuration la moins favorable (dégénérescence du polygône en ligne, et <img src="https://latex.codecogs.com/svg.image?\theta_{i}=\begin{Bmatrix}0&space;\\&space;\pi\end{Bmatrix}"/>.

**La saisie est optimale lorsque <img src="https://latex.codecogs.com/svg.image?Q_{SGP}"/> est minimum (cas d'un polygône régulier).**

(+) Interprétation simple, léger en calculs.

(-) Limité aux saisies planaires

(-) Peut parfois mener à des saisies inadéquates d'un point de vue pratique, pour les objets allongés par exemple.

<br/>

**3.2.2) Aire du polygône de saisie**

* Cas saisie 3 doigts : 

Plus le triangle des points de contact est large, meilleure est la saisie. Une même force fournie par les doigts aboutirait à une saisie plus robuste (c-à-d résistante à des couples externes plus importants).

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?(12):&space;Q_{AGP}&space;=&space;Aire(Triangle(p_1,p_2,p_3))"/>
</p>

(+) Interprétation simple, léger en calculs.

* Extension à plus de 3 doigts : 

Dans un premier temps, définir un polygône avec 3 doigts. Puis, projeter la surface pour déterminer les autres points de contact possibles. Enfin, choisir parmi ces points la positions des doigts restant en veillant à maximiser l'air du polygône.

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?(13):&space;Q_{AGP'}=&space;Aire(Polygone(p_1,p_2,p_3,p_4,...,p_n))&space;" />
</p>

(-) Comme <img src="https://latex.codecogs.com/svg.image?Q_{SGP}"/>, <img src="https://latex.codecogs.com/svg.image?Q_{AGP}"/> et <img src="https://latex.codecogs.com/svg.image?Q_{AGP'}"/> peuvent mener à des saisies inapplicables.

<ins>En pratique :</ins> Ces méthodes doivent être combinées à d'autres mesures directement liées aux propriétés de saisie.

<br/>

**3.2.3) Distance entre le centre de gravité du polygône de pts de contact et le CM de l'objet**



**3.2.4) Orthogonalité**


**3.2.5) Marges d'incertitudes dans les positions des doigts**


**3.2.6) Régions de contact indépendantes**




#### 3.3) Mesures considérant les limites de forces de saisie







<br/>

### 4) Mesures de qualité associés aux configurations de la main

<br/>

### 5) Approche combinant points de contacts et configuration de main

<br/>

### 6) Autres approches

<br/>

### 7) Discussion

<br/>



#### Sources

**Article original :**

Roa, M. A., & Suárez, R. (2015). Grasp quality measures: review and performance. Autonomous robots, 38(1), 65-88.

**Articles cités :**

[1] (Kim et al. 2001)


_____________




Le Gradient de la Politique (ou *Policy Gradient*) est une approche de résolution de problèmes en Apprentissage par Renforcement. Dans ce paradigme d'Apprentissage Automatique, il s'agit de trouver une stratégie de comportement optimale pour un ou plusieurs agents, de manière à maximiser les récompenses obtenues. Les méthodes de **gradient de la politique** visent à modéliser et à optimiser la politique directement. En général, cette dernière est modélisée par une fonction paramétrique de <img src="https://latex.codecogs.com/svg.image?\theta"/>, notée <img src="https://latex.codecogs.com/svg.image?\pi_\theta(a|s)"/>. Les valeurs de la fonction de récompenses (fonction objectif) dépendent de cette politique. Plusieurs algorithmes peuvent être appliqués pour optimiser <img src="https://latex.codecogs.com/svg.image?\theta"/> de sorte à maximiser les performances de l’agent sur une tâche donnée.


*<ins>Note de rédaction :</ins> Je mettrai à jour régulièrement cette liste pour qu'elle contiennent l'essentiel des informations pour appréhender les algorithmes de l'État de l'Art sans avoir à plonger trop en détail dans les articles originaux. J'ajouterai par ailleurs, autant que possible, une implémentation simple en python.*

*<ins>Crédit :</ins> Ces notes s'appuient très largement sur le [blog de Lilian Weng](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#off-policy-policy-gradient) et sur les [notes de cours de Serguey Levine, CS182 à l'UC Berkeley](https://cs182sp21.github.io/) (que je recommande aux lecteurs anglophones). L'ensemble des ressources utiliées sont listées à la fin de l'article.*



<br/>


### Comment optimiser la politique d'un agent ?

La fonction de récompense (ou fonction de performance) est définie par :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?J(\theta)=\sum_{s\in&space;S}&space;\mu(s)v_\pi(s)&space;=&space;\sum_{s\in&space;S}&space;\mu(s)\sum_{a\in&space;A}\pi_\theta(a|s)q_\pi(s)"/>
</p>

Avec <img src="https://latex.codecogs.com/svg.image?\mu(s)"/>: la distribution stationnaire de la chaïne markovienne sous la politique <img src="https://latex.codecogs.com/svg.image?\pi_\theta"/>.

On cherche donc à trouver les valeurs de <img src="https://latex.codecogs.com/svg.image?\theta"/> qui maximisent la récompense. Ce problème peut être résolu via la méthode d'ascension de gradient :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\theta_{t&plus;1}=&space;\theta_t&space;&plus;&space;\alpha*&space;\widehat{\nabla_\theta&space;J(\theta_t)}"/>
</p>

Où <img src="https://latex.codecogs.com/svg.image?\widehat{\nabla_\theta&space;J(\theta_t)}"/> est une estimation stochastique, dont l'espérence est le gradient de la performance mesurée par rapport à <img src="https://latex.codecogs.com/svg.image?\theta_t"/>.

En dérivant, on obtient :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\nabla_\theta&space;J(\theta)=\sum_{s\in&space;S}&space;\nabla&space;_\theta&space;\mu(s)&space;v_\pi(s)&space;&plus;&space;\mu(s)&space;\nabla_\theta&space;v_\pi(s)"/>
</p>


Comment estimer <img src="https://latex.codecogs.com/svg.image?\nabla_\theta&space;\mu(s)"/>, dans les cas où l'on ne connait pas les dynamiques qui régissent l'environnement dans lequel l'agent évolue ?

Il existe un moyen de contourner le problème, en écrivant le gradient de la performance sous une forme simplifiée. C'est ce que permet de le théorème du Gradient de la Politique.

<br/>

### Théorème du Gradient de la Politique


Le théorème du Gradient la Politique s'énonce de la façon suivante :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla_\theta&space;J(\theta)&space;&=&space;\nabla_\theta&space;\sum_{s\in&space;S}&space;\mu(s)&space;\sum_{a&space;\in&space;A}&space;q_\pi(s,a)&space;\pi_\theta(a|s)&space;\\&&space;\propto&space;\sum_{s\in&space;S}&space;\mu(s)&space;\sum_{a&space;\in&space;A}&space;q_\pi(s,a)&space;\nabla_\theta&space;\pi_\theta(a|s)\end{align*}"/>
</p>

Cette formulation permet d'estimer le gradient de la performance en s'afranchissant du terme <img src="https://latex.codecogs.com/svg.image?\nabla&space;\mu(s)"/>.

#### Preuve


On distingue le **cas épisodique** du **cas continue**, pour lesquels la fonction de performance ne s'exprime pas exactement de la même manière.

<ins>Remarque :</ins> On notera implicitement : <img src="https://latex.codecogs.com/svg.image?\nabla&space;\doteq&space;\nabla_\theta"/>, et <img src="https://latex.codecogs.com/svg.image?\pi(a|s)\doteq\pi_\theta(a|s)"/>.

**Cas épisodique**
 
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla&space;v_\pi(s)&space;&=&space;\nabla&space;\sum_{a&space;\in&space;A}&space;\pi(a|s)&space;q_\pi(s,a)&space;\\&=&space;\sum_{a&space;\in&space;A}&space;(\nabla&space;\pi(a|s)&space;q_\pi(s,a)&space;&plus;&space;\pi(a|s)&space;\nabla&space;q_\pi(s,a))&space;\\&=&space;\sum_{a&space;\in&space;A}&space;(\nabla&space;\pi(a|s)&space;q_\pi(s,a)&space;&plus;&space;\pi(a|s)&space;\nabla&space;(\sum_{r,s^\prime}p(r,s^\prime|s,a)(r&plus;v_\pi(s^\prime))))&space;\\&=&space;\sum_{a&space;\in&space;A}&space;(\nabla&space;\pi(a|s)&space;q_\pi(s,a)&space;&plus;&space;\pi(a|s)&space;\nabla&space;\sum_{s^\prime}p(s^\prime|s,a)(r&plus;v_\pi(s^\prime)))&space;\\\nabla&space;v_\pi(s)&space;&=&space;\sum_{a&space;\in&space;A}&space;(\nabla&space;\pi(a|s)&space;q_\pi(s,a)&space;&plus;&space;\pi(a|s)&space;\sum_{s^\prime}p(s^\prime|s,a)&space;\nabla&space;v_\pi(s^\prime))\end{align*}"/>
</p>


On obtient une forme recursive, reliant l'état s à l'état suivant s'.


Pour alléger l'écriture, posons : 
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\phi(s)&space;\doteq&space;\sum_{a&space;\in&space;A}&space;\nabla&space;\pi(a|s)q_\pi(s,a)"/>.
</p>

Soit <img src="https://latex.codecogs.com/svg.image?p_\pi(s\rightarrow&space;s^\prime,&space;k)"/> la probabilité de transitionner d'un état s à s' en suivant la politique <img src="https://latex.codecogs.com/svg.image?\pi_\theta"/>. On notera <img src="https://latex.codecogs.com/svg.image?p_\pi(s\rightarrow&space;s^\prime,&space;k)&space;=&space;p(s\rightarrow&space;s^\prime,&space;k)"/> par commodité d'écriture.

Cette probabilité s'exprime comme produit de la probabilité de choisir l'action a à partir de s (liée à la politique), et de la probabilité d'atteindre l'état s' en partant de l'état s et de l'action a (probabilité liée aux dynamiques de l'environnement). On somme les probabilités sur chaque action pour obtenir la probabilité de transition : <img src="https://latex.codecogs.com/svg.image?p(s\rightarrow&space;s^\prime,&space;k=1)=\sum_a&space;\pi(a|s)p(s^\prime|s,a)"/>.

Notons par ailleurs que l'on peut exprimer la probabilité de transitionner d'un état vers un autre sur plusieurs pas sous la forme d'un produit des probabilités des transitions intermédiaires :

<img src="https://latex.codecogs.com/svg.image?\forall&space;(s,s^\prime,s^{\prime\prime})&space;\in&space;S^3"/>, et <img src="https://latex.codecogs.com/svg.image?\forall&space;k\in\mathbb{N}^{*}"/>, on a : <img src="https://latex.codecogs.com/svg.image?p_\pi(s\rightarrow&space;s^{\prime\prime},&space;k)&space;=&space;p(s\rightarrow&space;s^\prime,&space;k-1)&space;p(s^\prime&space;\rightarrow&space;s^{\prime\prime},&space;1)"/>.

Grâce à ces expression, on peut dérouler la récursion :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla&space;v_\pi(s)&space;&=&space;\phi(s)&plus;&space;\sum_{a&space;\in&space;A}&space;\pi(a|s)&space;\sum_{s^\prime}p(s^\prime|s,a)&space;\nabla&space;v_\pi(s^\prime)&space;\\&=&space;\phi(s)&space;&plus;&space;p(s\rightarrow&space;s^\prime,1)&space;\nabla&space;v_\pi(s^\prime)&space;\\&=&space;\phi(s)&space;&plus;&space;p(s\rightarrow&space;s^\prime,1)(\phi(s^\prime)&space;&plus;&space;p(s^\prime&space;\rightarrow&space;s^{\prime&space;\prime},1)\nabla&space;v_\pi(s^{\prime&space;\prime}))&space;\\&=&space;\phi(s)&space;&plus;&space;p(s\rightarrow&space;s^\prime,1)\phi(s^\prime)&space;&plus;&space;p(s&space;\rightarrow&space;s^{\prime&space;\prime},2)\nabla&space;v_\pi(s^{\prime&space;\prime})&space;\\&=&space;\phi(s)&space;&plus;&space;p(s\rightarrow&space;s^\prime,1)\phi(s^\prime)&space;&plus;&space;p(s&space;\rightarrow&space;s^{\prime&space;\prime},2)\phi(s^{\prime&space;\prime})&space;&plus;&space;p(s&space;\rightarrow&space;s^{\prime&space;\prime&space;\prime},3)\phi(s^{\prime&space;\prime&space;\prime})&space;&plus;&space;...\\\nabla&space;v_\pi(s)&space;&=&space;\sum_{x&space;\in&space;S}&space;\sum_{k=0}^{\infty}p(s\rightarrow&space;x,&space;k)\phi(s)\end{align*}"/>
</p>


Le théorème du gradient de la politique fait intervenir la distribution stationnaire des états <img src="https://latex.codecogs.com/svg.image?\mu_\pi(s)"/> - notée <img src="https://latex.codecogs.com/svg.image?\mu(s)"/> - définit de la façon suivante :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\mu(s)&space;\doteq&space;\frac{\eta(s)&space;}{\sum_{s^\prime&space;\in&space;S}\eta(s^\prime)}"/>
</p>
Où <img src="https://latex.codecogs.com/svg.image?\eta(s)"/> est l'espérence du nombre de visite de s sur un épisode, soit : <img src="https://latex.codecogs.com/svg.image?\eta(s)&space;\doteq&space;\sum_{k=0}^{\infty}p(s_0\to&space;s&space;|k)"/>.

Cette dernière forme peut nous permettre d'exprimer les probabilités de transition sous forme d'espérence du nombre de visite, pour faire apparaître la distribution stationnaire. 

Rappelons enfin que la performance correspond à la récompense espérée sur l'épisode en suivant <img src="https://latex.codecogs.com/svg.image?\pi"/> à partir de l'état initial, soit : <img src="https://latex.codecogs.com/svg.image?J(\theta)&space;\doteq&space;v(s_0)"/>. 

Nous avons maintenant tous les éléments pour finir la démonstration :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla&space;J(\theta)&space;&=&space;\nabla&space;v(s_0)\\&=&space;\sum_{x&space;\in&space;S}&space;\sum_{k=0}^{\infty}p(s\rightarrow&space;x,&space;k)\sum_{a&space;\in&space;A}&space;\nabla&space;\pi(a|x)q_\pi(x,a)&space;\\&=&space;\sum_{x&space;\in&space;S}&space;\eta&space;(x)&space;\sum_{a&space;\in&space;A}&space;\nabla&space;\pi(a|x)&space;q_\pi(x,a)&space;\\&=&space;\sum_{x&space;\in&space;S}&space;(\sum_{s^\prime}&space;\eta(s^\prime))\frac{\eta&space;(x)}{\sum_{s^\prime}&space;\eta(s^\prime)}&space;\sum_{a&space;\in&space;A}&space;\nabla&space;\pi(a|x)&space;q_\pi(x,a)&space;\\\nabla&space;J(\theta)&space;&=&space;\sum_{s^\prime}&space;\eta(s^\prime)&space;\sum_{x&space;\in&space;S}&space;\mu(s)&space;\sum_{a&space;\in&space;A}&space;\nabla&space;\pi(a|x)&space;q_\pi(x,a)&space;\\\nabla&space;J(\theta)&space;&\propto&space;\sum_{x&space;\in&space;S}&space;\mu(s)&space;\sum_{a&space;\in&space;A}&space;\nabla&space;\pi(a|x)&space;q_\pi(x,a)\end{align*}"/>
</p>


 
**Cas continue**

Dans le cas continue, on définit la performance sous la forme de récompense moyenne par pas de temps :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;J(\theta)&space;&\doteq&space;r(\pi)&space;\doteq&space;\displaystyle&space;\lim_{h&space;\to&space;\infty}&space;\sum_{t=1}^{h}\mathop{\mathbb{E}}[R_t|S_0,A_{0&space;:&space;t-1}\sim\pi]&space;\\&&space;=&space;\lim_{h&space;\to&space;\infty}&space;\mathop{\mathbb{E}}[R_t|S_0,A_{0&space;:&space;t-1}\sim\pi]&space;\\&&space;=&space;\sum_s&space;\mu(s)&space;\sum_a&space;\pi(a|s)&space;\sum_{s^\prime,r}&space;p(^\prime,r|s,a)r\end{align*}"/>
</p>


En outre, <img src="https://latex.codecogs.com/svg.image?v_{\pi}"/> et <img src="https://latex.codecogs.com/svg.image?q_{\pi}"/> sont fonctions du retour différentiel, définit comme la différence entre la récompense obtenue à chaque pas, et la récompense moyenne : 
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?G_t&space;=&space;R_{t&plus;1}&space;-&space;r(\pi)&space;&plus;&space;R_{t&plus;2}&space;-&space;r(\pi)&space;&plus;&space;..."/>
</p>

On procède d'une façon analogue au cas épisodique :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla&space;v_\pi(s)&space;&&space;=&space;\nabla&space;\sum_a&space;\pi(a|s)q_\pi(s,a)&space;\\&&space;=&space;\sum_a&space;(\nabla\pi(a|s)&space;q_\pi(s,a)&plus;\pi(a|s)&space;\nabla&space;q_\pi(a,s)&space;)&space;\\&&space;=&space;\sum_a&space;(\nabla\pi(a|s)&space;q_\pi(s,a)&plus;\pi(a|s)&space;\nabla\sum_{s^\prime,r}&space;p(s^\prime,r|s,a)(r-r(\theta)&plus;v_\pi(s)))\\&=&space;\sum_a&space;(\nabla\pi(a|s)&space;q_\pi(s,a)&plus;\pi(a|s)&space;\sum_{s^\prime,r}&space;p(s^\prime,r|s,a)(-\nabla&space;r(\theta)&plus;\nabla&space;v_\pi(s^\prime)))&space;\\&=&space;\sum_a&space;(\nabla\pi(a|s)&space;q_\pi(s,a)&plus;\pi(a|s)&space;\sum_{s^\prime}&space;p(s^\prime|s,a)(-\nabla&space;r(\theta)&plus;\nabla&space;v_\pi(s^\prime)))&space;\\&&space;=&space;\sum_a&space;(\nabla\pi(a|s)&space;q_\pi(s,a)&space;-&space;\pi(a|s)\nabla&space;r(\theta)&space;&plus;&space;\pi(a|s)&space;\sum_{s^\prime}&space;p(s^\prime|s,a)\nabla&space;v_\pi(s^\prime))&space;\end{align*}"/>
</p>

Ce qui nous permet d'isoler <img src="https://latex.codecogs.com/svg.image?\nabla&space;r(\theta)&space;"/> :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\nabla&space;r(\theta)&space;=&space;\sum_a&space;\nabla\pi(a|s)q_\pi(s,a)&space;&plus;&space;\sum_a&space;\pi(a|s)\sum_{s^\prime}p(s^\prime|s,a)\nabla&space;v_\pi(s^\prime)-\nabla&space;v_\pi(s)"/>
</p>

Par définition, <img src="https://latex.codecogs.com/svg.image?J(\theta)=r(\theta)"/>. Or <img src="https://latex.codecogs.com/svg.image?r(\theta)"/> est indépendant de s. L'équation du gradient de la performance est donc toujours juste si l'on multiplie le terme de droite par <img src="https://latex.codecogs.com/svg.image?\sum_s&space;\mu(s)"/>, puisque:
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\sum_s&space;\mu(s)&space;\doteq&space;1" title="\sum_s&space;\mu(s)&space;\doteq&space;1" />
</p>

Ainsi :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla&space;J(\theta)&space;&=&space;\sum_s&space;\mu(s)&space;(\sum_a&space;(\nabla\pi(a|s)q_\pi(s,a)&space;&plus;&space;\pi(a|s)\sum_{s^\prime}p(s^\prime|s,a)\nabla&space;v_\pi(s^\prime))-\nabla&space;v_\pi(s))&space;\\&=&space;\sum_s&space;\mu(s)&space;\sum_a&space;\nabla\pi(a|s)q_\pi(s,a)&space;&plus;&space;\sum_{s^\prime}&space;\sum_{s}&space;\mu(s)&space;\sum_a&space;\pi(a|s)p(s^\prime|s,a)\nabla&space;v_\pi(s^\prime)&space;-&space;\sum_s&space;\mu(s)&space;\nabla&space;v_\pi(s)&space;\\&=&space;\sum_s&space;\mu(s)&space;\sum_a&space;\nabla\pi(a|s)q_\pi(s,a)&space;&plus;&space;\sum_{s^\prime}&space;\mu(s^\prime)\nabla&space;v_\pi(s^\prime)&space;-&space;\sum_s&space;\mu(s)&space;\nabla&space;v_\pi(s)&space;\\\nabla&space;J(\theta)&space;&=&space;\sum_s&space;\mu(s)&space;\sum_a&space;\nabla\pi(a|s)q_\pi(s,a)\end{align*}"/>
</p>

On retrouve la même forme que dans le cas épisodique. 



**En résumé**

Dans les deux cas, on a donc :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\nabla&space;J(\theta)&space;\propto&space;\sum_{s\in&space;S}&space;\mu(s)&space;\sum_{a&space;\in&space;A}&space;q_\pi(s,a)&space;\nabla_\theta&space;\pi_\theta(a|s)"/>
</p>

Avec pour coefficient de proportionnalité:
- <img src="https://latex.codecogs.com/svg.image?\sum_{s}&space;\eta(s)"/> dans le cas épisodique ;
- 1 dans le cas continue.



<br/>

### Généralisation aux algorithmes du gradient de la politique

Le théorème du gradient de la politique nous permet d'exprimer le gradient de la performance d'une manière simple et élégante :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla&space;J(\theta)&space;&\propto&space;\sum_{s&space;\in&space;S}&space;\mu(s)&space;\sum_{a&space;\in&space;A}&space;q_\pi(s,a)&space;\nabla&space;\pi(a|s)&space;\\&&space;=&space;\sum_{s&space;\in&space;S}&space;\mu(s)&space;\sum_{a&space;\in&space;A}&space;\pi(a|s)&space;q_\pi(s,a)&space;\frac{\nabla&space;\pi(a|s)}{\pi(a|s)}&space;&space;\\\nabla&space;J&space;(\theta)&space;&=&space;\mathop{\mathbb{E}}_{s\sim&space;\mu_\pi,&space;a\sim&space;\pi_0}&space;[q_\pi(s,a)&space;\nabla&space;\ln\pi(a|s)]\\\end{align*}"/>
</p>

Cette forme constitue les fondements de la plupart des algorithmes du gradient de la politique. Elle a pour particularité de ne **pas** avoir **de biais**, mais d'être soumis à une **forte variance**. Les algorithmes évoqués dans cet article proposent des solutions pour réduire la variance sans (trop) affecter le bias.

L'article *Estimation de l'Avantage Généralisé* (GAE) [Schulman et al., 2016](https://arxiv.org/pdf/1506.02438.pdf) propose une forme générale du gradient de la performance, mettant en lumière les différentes déclinaisons de cette forme que l'on peut trouver dans la littérature :

En posant g, le gradient de la performance, tel que <img src="https://latex.codecogs.com/svg.image?g&space;\doteq&space;\nabla_\theta&space;\mathop{\mathbb{E}}[\sum_{t=0}^\infty&space;r_t]"/>, on a la forme générale : 
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?g&space;=&space;\mathop{\mathbb{E}}[\sum_{t=0}^\infty&space;\Psi_t&space;\nabla_\theta&space;log&space;\pi_\theta(a_t|s_t)]"/>
</p>

Avec <img src="https://latex.codecogs.com/svg.image?\Psi_t"/>, l'une des fonctions suivantes :
- <img src="https://latex.codecogs.com/svg.image?\sum_{t=0}^\infty&space;r_{t}"/> : retour total de la trajectoire
- <img src="https://latex.codecogs.com/svg.image?\sum_{t^\prime=t}^\infty&space;r_{t^\prime}"/> : retour suivant l'action <img src="https://latex.codecogs.com/svg.image?a_t"/>
- <img src="https://latex.codecogs.com/svg.image?\sum_{t^\prime=t}^\infty&space;r_{t^\prime}-b(s_t)"/> : formule précédente, avec valeurs de références
- <img src="https://latex.codecogs.com/svg.image?q_\pi(s_t,a_t)"/> : fonction de valeur d'état-action
- <img src="https://latex.codecogs.com/svg.image?A_\pi(s_t,a_t)"/> : fonction d'avantage
- <img src="https://latex.codecogs.com/svg.image?r_t&space;&plus;&space;v_\pi(s_{t&plus;1})-v_\pi(s_t)"/> : résidu TD (différence temporelle)


Avec les fonctions de valeurs : 
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?v_\pi&space;(s_t)&space;\doteq&space;\mathop{\mathbb{E}}_{s_{t&plus;1}:\infty,a_t:\infty}[\sum_{l=0}^\infty&space;r_{t&plus;l}]"/>
	<img src="https://latex.codecogs.com/svg.image?q_\pi&space;(s_t,a_t)&space;\doteq&space;\mathop{\mathbb{E}}_{s_{t&plus;1}:\infty,a_{t&plus;1}:\infty}[\sum_{l=0}^\infty&space;r_{t&plus;l}]"/>
</p>

Et la fonction d'avantage : 
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?A_\pi(s_t,a_t)&space;\doteq&space;q_\pi(s_t,a_t)-v_\pi(s_t)"/>
</p>

<br/>

## Algorithmes du Gradient de la Politique

Les algorithmes présentés font tous mention du paramètre <img src="https://latex.codecogs.com/svg.image?\gamma&space;\in&space;&space;\]0;1\]"/>, le facteur d'atténuation. Sa définition sera donc implicite, afin d'éviter les redondances.

<br/>

### REINFORCE

L'algorithme **REINFORCE** (gradient de la Politique avec méthode Monte-Carlo) repose sur l'expression du gradient de la performance obtenue dans le Théorème du Gradient de la Politique, appliqué aux épisodes prélevés (i.e. obtenus par interaction directe avec l'environnement). En constatant que <img src="https://latex.codecogs.com/svg.image?q_\pi(s_t,a_t)=&space;\mathop{\mathbb{E}}_\pi[G_t&space;|s_t,&space;a_t]"/>, on trouve :


<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla_\theta&space;J(\theta)&space;&=&space;\mathop{\mathbb{E}}_\pi[q_\pi(s,a)&space;\nabla_\theta\ln\pi_\theta(a,s)]&space;\\&=&space;\mathop{\mathbb{E}}_\pi[G_t&space;\nabla_\theta\ln\pi_\theta(a,s)]\end{align*}"/>
</p>

Autrement dit, on peut optimiser <img src="https://latex.codecogs.com/svg.image?\theta"/> à partir du retour obtenu au cours d'un épisode. Cette approche exploite la trajectoire observée sur l'épisode entier pour faire ses mises à jours ; c'est pourquoi on parle de méthode de type Monte Carlo.

---

**Algorithme : REINFORCE (épisodique)**

<ins>Initialisation :</ins>
- Définir <img src="https://latex.codecogs.com/svg.image?\alpha"/>, le pas d'apprentissage associé à la politique
- Initialiser aléatoirement <img src="https://latex.codecogs.com/svg.image?\theta&space;\in&space;\mathbb{R}^{d}"/>, les poids associés aux caractéristiques définissant la politique

<ins>Exécution :</ins>
- Pour chaque épisode :
	- Générer la trajectoire <img src="https://latex.codecogs.com/svg.image?s_1,a_1,r_2,s_2,a_2,&space;...&space;,&space;a_{T-1},&space;s_T"/> en suivant <img src="https://latex.codecogs.com/svg.image?\pi_\theta"/>
	- Pour chaque étape de l'épisode <img src="https://latex.codecogs.com/svg.image?t=0,1,...,T-1,T"/> :
		- <img src="https://latex.codecogs.com/svg.image?G\leftarrow&space;\sum_{k=t&plus;1}^T&space;\gamma^{k-t-1}r_k"/>
		- <img src="https://latex.codecogs.com/svg.image?\theta&space;\leftarrow&space;\theta&space;&plus;&space;\alpha&space;\gamma^t&space;\nabla\ln\pi(a_t|s_t,\theta)"/>

---

<br/>

Avec Reinforce, <img src="https://latex.codecogs.com/svg.image?\theta"/> est mis-à-jour en utilisant directement le retour observé lors d'une interaction avec l'environnement. Il n'y a donc **pas de biais** : la montée de gradient fera toujours évoluer <img src="https://latex.codecogs.com/svg.image?\theta"/> vers des valeurs qui augmenteront l'espérence de retour. En revanche, cette méthode est très dépendante des valeurs de récompenses obtenues, introduisant une **forte variance**, rendant l'entraînement instable.



### REINFORCE avec valeurs de référence

L'algorithme **REINFORCE avec valeurs de référence** est une variante bien connue de l'algorithme REINFORCE. Il s'agit simplement de retrancher au retour de l'épisode une valeur de référence dans le calcul du gradient de la performance. Cette modification a pour effet de **réduire la variance** tout en assurant l'absence de biais.

On utilise souvent la **valeur d'état** en guise de valeur de référence, de sorte que l'on utilise la **fonction d'avantage** dans la mise à jour du gradient.

Il s'avère qu'il est possible de montrer que le Théorème du Gradient de la Politique peut être étendu à la forme suivante :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\nabla&space;J(\theta)&space;\propto&space;\sum_s&space;\mu(s)&space;\sum_a&space;(q_\pi(s,a)&space;-&space;b(s))\nabla&space;\pi(a|s,\theta)"/>
</p>

Où <img src="https://latex.codecogs.com/svg.image?b(s)"/> est une fonction quelconque qui **ne dépend pas des actions** prises par l'agent.

#### Preuve

L'expression du théorème avec la fonction <img src="https://latex.codecogs.com/svg.image?b(s)"/> peut être reformulée de la façon suivante :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla&space;J(\theta)&space;&\propto&space;\sum_s&space;\mu(s)&space;\sum_a&space;(q_\pi(s,a)&space;-&space;b(s))\nabla&space;\pi(a|s,\theta)&space;\\&&space;=&space;\sum_s&space;\mu(s)&space;(\sum_a&space;q_\pi(s,a)\nabla&space;\pi(a|s,\theta)&space;-&space;\sum_a&space;b(s)\nabla&space;\pi(a|s,\theta))\end{align*}"/>
</p>

Or :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\sum_a&space;b(s)\nabla&space;\pi(a|s,\theta)&space;=&space;b(s)&space;\nabla&space;\sum_a&space;\pi(a|s,\theta)=&space;b(s)&space;\nabla&space;1&space;=&space;0&space;"/>
</p>

Insérer <img src="https://latex.codecogs.com/svg.image?b(s)"/> dans l'expression ne l'invalide donc pas, puisque cette opération revient à une soustraction par zéro.


#### Intuition

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/gradient_de_la_politique/trajectoires_max_perf.png"/> 
  <br/>
  Example tiré du <a href="http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_4_policy_gradient.pdf">cours de S. Levine à l'UC Berkeley</a>
</p>


Pour comprendre en quoi cette idée permet de faciliter grandement l'entraînement, considérons un [MDP](https://fr.wikipedia.org/wiki/Processus_de_d%C3%A9cision_markovien) à récompenses positives. Imaginons que trois essais nous donne les trois trajectoires, représentées ci-dessus (en z se trouve la performance, que l'on cherche à maximiser); nous aurions trois récomponses positives, plus ou moins grandes selon la qualité de la trajectoire. Pourtant, il serait souhaitable d'augmenter la probabilité de choisir la meilleure trajectoire, et de réduire celle de choisir la moins bonne.

L'intuition derrière l'utilisation de la valeur de référence est la suivante : en soustrayant la récompense moyenne à la récompose obtenue, on incitera la politique à **choisir plus souvent des trajectoires qui ont permis l'obtention d'une récompense plus élevée que la moyenne**, tout en l'incitant à **moins choisir les trajectoires ayant abouties à une récompense inférieure à la moyenne**.


---

**Algorithme : REINFORCE avec valeurs de référence (épisodique)**

<ins>Initialisation :</ins>
- Définir :
	- <img src="https://latex.codecogs.com/svg.image?\alpha^w&space;\in&space;\mathbb{R}^{&plus;*}"/>, le pas d'apprentissage associé aux valeurs d'états
	- <img src="https://latex.codecogs.com/svg.image?\alpha^\theta&space;\in&space;\mathbb{R}^{&plus;*}"/>, le pas d'apprentissage associé à la politique
- Initialiser aléatoirement :
	- <img src="https://latex.codecogs.com/svg.image?w&space;\in&space;\mathbb{R}^{d^\prime}"/>, les poids associés aux valeurs d'états
	- <img src="https://latex.codecogs.com/svg.image?\theta&space;\in&space;\mathbb{R}^{d}"/>, les poids associés aux caractéristiques définissant la politique
	
<ins>Exécution :</ins>
- Pour chaque épisode :
	- Générer la trajectoire <img src="https://latex.codecogs.com/svg.image?s_1,a_1,r_2,s_2,a_2,&space;...&space;,&space;a_{T-1},&space;s_T"/> en suivant <img src="https://latex.codecogs.com/svg.image?\pi_\theta"/>
	- Pour chaque étape de l'épisode <img src="https://latex.codecogs.com/svg.image?t=0,1,...,T-1,T"/> :
		- <img src="https://latex.codecogs.com/svg.image?G\leftarrow&space;\sum_{k=t&plus;1}^T&space;\gamma^{k-t-1}r_k"/>
		- <img src="https://latex.codecogs.com/svg.image?\delta&space;\leftarrow&space;G&space;-&space;\hat{v}(s_t,w)"/>
		- <img src="https://latex.codecogs.com/svg.image?w&space;\leftarrow&space;w&space;&plus;&space;\alpha^w&space;\delta&space;\nabla&space;\hat{v}(s_t,w)"/>
		- <img src="https://latex.codecogs.com/svg.image?\theta&space;\leftarrow&space;\theta&space;&plus;&space;\alpha^\theta&space;\gamma^t&space;\delta&space;\nabla\ln\pi(a_t|s_t,\theta)"/>

---



<br/>

### Acteur-Critique

L'algorithme **Acteur-Critique** ressemble beaucoup à l'algorithme REINFORCE avec valeurs de référence : il s'agit de calculer une différence entre un retour, et une valeur de référence. En revanche, deux différences importantes les distinguent : la possibilité d'effectuer des mises-à-jours en ligne, et la mécanique d'évaluation des actions prises par l'agent.

Comme toutes les méthodes de type Monte-Carlo, REINFORCE ne fait pas de mise-à-jour avant la fin de l'épisode. Par ailleurs, la valeur de référence ne tient compte que de la valeur de l'état initial (avant de prendre l'action), et ne permet par conséquent pas de juger de la qualité de l'action choisie. Par cette approche, on répond à la question : **"L'agent a-t-il bien fait de se trouver à cette position au temps t ?"**, en tenant compte de l'épisode entier.

Dans le cas le plus simple de la méthode Acteur-Critique, le retour utilisé dans la mise-à-jour des paramètres de la politique calcule la différence entre la valeur de l'état au temps t, et celle de l'état au temps t+1 (en tenant compte du facteur d'atténuation) ; c'est le retour 1-pas, noté <img src="https://latex.codecogs.com/svg.image?G_{t:t&plus;1}"/> (comme dans les méthodes TD(0), SARSA(0) ou Q-apprentissage). Cette approche permet donc d'évaluer la différence de valeur entre l'état initial et le nouvel état, autrement dit de juger de la qualité de l'action prise par l'agent. On répond ici à la question : **"L'agent a-t-il bien fait de choisir cette action au temps t?"**, en ne tenant compte que de la transition entre les temps t et t+1.

En résumé : la politique agit, et la méthode de retour intermédiaire critique.


Il existe de nombreuses variantes autour des méthodes de type Acteur-Critique, impliquant entre autres : la fonction d'avanntage, les valeurs-Q, la méthode SARSA 1-pas, le retour n-pas, et différentes approches d'entraîements (séquentiel, asynchronisé). Toutes ces méthodes ont les deux composantes qui caractérisent ce type d'approche, à savoir 1) la possibilité de réaliser les mises-à-jours sans avoir à attendre la fin de l'épisode (retour n-pas), et 2) la mécanique d'évaluation des décisions prises dans le processus d'optimisation. 

L'algorithme présenté dans cette section correspond donc à une certaine variante de la méthode Acteur-Critique : cas épisodique, 1-étape, pas de mémoire tampon, et utilisation de la fonction d'avantage.

---

**Algorithme : Acteur-critique (épisodique)**

<ins>Initialisation :</ins>
- Définir :
	- <img src="https://latex.codecogs.com/svg.image?\alpha^w&space;\in&space;\mathbb{R}^{&plus;*}"/>, le pas d'apprentissage associé aux valeurs d'états
	- <img src="https://latex.codecogs.com/svg.image?\alpha^\theta&space;\in&space;\mathbb{R}^{&plus;*}"/>, le pas d'apprentissage associé à la politique
- Initialiser aléatoirement :
	- <img src="https://latex.codecogs.com/svg.image?w&space;\in&space;\mathbb{R}^{d^\prime}"/>, les poids associés aux valeurs d'états
	- <img src="https://latex.codecogs.com/svg.image?\theta&space;\in&space;\mathbb{R}^{d}"/>, les poids associés aux caractéristiques définissant la politique
	
<ins>Exécution :</ins>
- Pour chaque épisode :
	- Initialiser <img src="https://latex.codecogs.com/svg.image?s"/> (premier état de l'épisode)
	- <img src="https://latex.codecogs.com/svg.image?I&space;\leftarrow&space;1"/> (coefficient de réduction cumulée)
	- Tant que <img src="https://latex.codecogs.com/svg.image?s"/> n'est pas terminal (pour chaque pas de temps) :
		- <img src="https://latex.codecogs.com/svg.image?a&space;\sim&space;\pi(\cdot|s,\theta)"/>
		- Appliquer l'action <img src="https://latex.codecogs.com/svg.image?a"/>, observer <img src="https://latex.codecogs.com/svg.image?(s^\prime,r)"/>
		- <img src="https://latex.codecogs.com/svg.image?\delta&space;\leftarrow&space;r&space;&plus;&space;\gamma&space;\hat{v}(s^\prime,w)&space;-&space;\hat{v}(s,w)"/>
		- <img src="https://latex.codecogs.com/svg.image?w&space;\leftarrow&space;w&space;&plus;&space;\alpha^w&space;\delta&space;\nabla&space;\hat{v}(s,w)"/>
		- <img src="https://latex.codecogs.com/svg.image?\theta&space;\leftarrow&space;\theta&space;&plus;&space;\alpha^\theta&space;I&space;\delta&space;\nabla\ln\pi(a|s,\theta)"/>
		- <img src="https://latex.codecogs.com/svg.image?I&space;\leftarrow&space;\gamma&space;I"/>
		- <img src="https://latex.codecogs.com/svg.image?s&space;\leftarrow&space;s^\prime"/>


Par convention, on a <img src="https://latex.codecogs.com/svg.image?\hat{v}(s^\prime,w)&space;\doteq&space;0"/> si <img src="https://latex.codecogs.com/svg.image?s^\prime"/> est terminal. 

---


Cette version ne tient compte que d'une transition pour réaliser les mises-à-jours de poids, ce qui a toutes les chances de rendre l'optimisation difficile en raison de la large variance dans les données d'entraînement d'une itération à l'autre. L'objectif de cette section n'est pas donner le meilleur algorithme acteur-critique, mais d'expliciter les principales composantes de cette approche.

<ins>Remarque :</ins> En pratique, on peut simplement étendre cet algorithme aux itérations sur des lots, en accumulant un nombre <img src="https://latex.codecogs.com/svg.image?n"/> de transitions, et en appliquant les mêmes étapes mentionnées ci-dessus. À noter qu'il n'y aucune nécessité de lien entre les transitions (i.e. elles n'ont pas à provenir d'une même trajectoire) ; tant que l'on a des quadruplets <img src="https://latex.codecogs.com/svg.image?(s,a,s^\prime,r)"/>, on est en mesure d'appliquer l'algorithme.


<br/>


### Gradient de la politique en contexte Hors-Politique


Tous les algorithmes jusqu'ici présentés optimisent la politique qui a été utilisée pour obtenir des trajectoires. Dans cette section, nous considérons les cas dans lesquels **la politique d'exploration n'est pas la même que la politique cible**. 

Jetons à nouveau un coup d'oeil à l'estimateur utilisé par les algorithmes de gradient de la politique :

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\nabla_\theta&space;J(\theta)&space;=&space;\mathop{\mathbb{E}}_{\tau&space;\sim&space;\pi_\theta(\tau)}[\nabla_\theta&space;\ln&space;\pi_\theta(\tau)r(\tau)]"/>
</p>

Le gradient de la performance est calculé à partir de l'espérence sur <img src="https://latex.codecogs.com/svg.image?\tau&space;\sim&space;\pi_\theta(\tau)"/>. Une trajectoire relevée à partir d'un certain <img src="https://latex.codecogs.com/svg.image?\theta"/> n'est donc plus valable après application d'une itération de l'algorithme de monté de gradient sur <img src="https://latex.codecogs.com/svg.image?\theta"/>. Autrement dit, il est **nécessaire de prélever une trajectoire après chaque mise-à-jours de** <img src="https://latex.codecogs.com/svg.image?\theta"/>, rendant caduques toutes les mesures précédentes. Le problème est de taille, sitôt que l'on se trouve sur des tâches pour lesquelles les mesures sont lentes et coûteuses (comme en robotique, par exemple).

Pour cette raison, il est souhaitable de définir des algorithmes permettant d'optimiser une politique à partir de mesures réalisées *Hors-Politique*, c'est-à-dire avec une autre politique que celle que l'on optimise.

Par ailleurs, une telle approche permet d'utiliser une politique la plus efficace pour l'exploration, sans avoir à contraindre la politique que l'on est en train d'optimiser pour l'inciter à explorer de nouvelles trajectoires.


## Échantillonnage préférentiel 

Pour rappel, on cherche :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\theta^*&space;=&space;\operatorname*{argmax}_\theta&space;J(\theta)" title="\theta^* = \operatorname*{argmax}_\theta J(\theta)" />
</p>

que l'on optimise à partir de : 

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?J(\theta)&space;=&space;\mathop{\mathbb{E}}_{\tau\sim\pi_\theta(\tau)}&space;[r(\tau)]" title="J(\theta) = \mathop{\mathbb{E}}_{\tau\sim\pi_\theta(\tau)} [r(\tau)]" />	
</p>

Supposons que nous ne disposons pas d'exemples de trajectoires correspondant à <img src="https://latex.codecogs.com/svg.image?\tau\sim\pi_\theta(\tau)" title="\tau\sim\pi_\theta(\tau)" />, mais que nous ayons à la place des trajectoires <img src="https://latex.codecogs.com/svg.image?\tau\sim\pi_{\theta^\prime}(\tau)" title="\tau\sim\pi_{\theta^\prime}(\tau)" /> pour un autre jeu de paramtère <img src="https://latex.codecogs.com/svg.image?\theta^\prime" title="\theta^\prime" />.

On a :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}\mathop{\mathbb{E}}_{\tau\sim\pi_\theta}&space;&=&space;\int&space;\pi_\theta(\tau)r(\tau)d\tau&space;\\&=&space;\int&space;\pi_{\theta^\prime}(\tau)&space;\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}&space;r(\tau)d\tau&space;\\&=&space;\mathop{\mathbb{E}}_{\tau\sim\pi_{\theta^\prime}}[\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}r(\tau)]\end{align*}" title="\begin{align*}\mathop{\mathbb{E}}_{\tau\sim\pi_\theta} &= \int \pi_\theta(\tau)r(\tau)d\tau \\&= \int \pi_{\theta^\prime}(\tau) \frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)} r(\tau)d\tau \\&= \mathop{\mathbb{E}}_{\tau\sim\pi_{\theta^\prime}}[\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}r(\tau)]\end{align*}" />
</p>

On appelle **poids d'importance** le coefficient <img src="https://latex.codecogs.com/svg.image?\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}" title="\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}" />. On parle de poids d'importance au pluriel, dans le sens où chaque terme en <img src="https://latex.codecogs.com/svg.image?\tau" title="\tau" /> comprend implicitement le produit des termes en <img src="https://latex.codecogs.com/svg.image?(s_i,&space;a_i)" title="(s_i, a_i)" /> associés à la trajectoire.

Ainsi : 
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\nabla&space;J(\theta)&space;=&space;\mathop{\mathbb{E}}_{\tau\sim\pi_{\theta^\prime}}[\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}r(\tau)]" title="\nabla J(\theta) = \mathop{\mathbb{E}}_{\tau\sim\pi_{\theta^\prime}}[\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}r(\tau)]" />
</p>

Or :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\pi_\theta(\tau)&space;\doteq&space;p(s_1)&space;\prod_{t=1}^{T}\pi_\theta(s_t|s_t)p(s_{t&plus;1}|s_t,a_t)" title="\pi_\theta(\tau) \doteq p(s_1) \prod_{t=1}^{T}\pi_\theta(s_t|s_t)p(s_{t+1}|s_t,a_t)" />
</p>

Avec <img src="https://latex.codecogs.com/svg.image?p(s_1)" title="p(s_1)" /> la probabilité de démarrer l'épisode à l'état <img src="https://latex.codecogs.com/svg.image?s_1" title="s_1" />, et <img src="https://latex.codecogs.com/svg.image?T" title="T" /> le nombre d'étape dans l'épisode avant que l'état terminal n'ait été atteint.

Les poids d'importance s'expriment donc de la façon suivante :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}&space;=&space;\frac{p(s_1)\prod_{t=1}^{T}\pi_\theta(a_t|s_t)p(s_{t&plus;1}|s_t,a_t)}{p(s_1)\prod_{t=1}^{T}\pi_{\theta^\prime}(a_t|s_t)p(s_{t&plus;1}|s_t,a_t)}" title="\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)} = \frac{p(s_1)\prod_{t=1}^{T}\pi_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)}{p(s_1)\prod_{t=1}^{T}\pi_{\theta^\prime}(a_t|s_t)p(s_{t+1}|s_t,a_t)}" />
</p>

Il apparaît clairement que les facteurs relatifs aux dynamiques du système s'annulent :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}&space;=&space;\frac{\prod_{t=1}^{T}\pi_\theta(a_t|s_t)}{\prod_{t=1}^{T}\pi_{\theta^\prime}(a_t|s_t)}" title="\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)} = \frac{\prod_{t=1}^{T}\pi_\theta(a_t|s_t)}{\prod_{t=1}^{T}\pi_{\theta^\prime}(a_t|s_t)}" />
</p>

De la même manière que pour le théorème du gradient de la politique, on obtient une forme qui ne dépend que des politiques, et qui est donc calculable même sans connaître les dynamiques du système.

Ce coefficient nous permet donc d'estimer le gradient de la performance pour estimer de nouveaux paramètres <img src="https://latex.codecogs.com/svg.image?\theta" title="\theta" /> à partir de nos anciens paramètres <img src="https://latex.codecogs.com/svg.image?\theta^\prime" title="\theta^\prime" />.

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?J(\theta^\prime)&space;=&space;\mathop{\mathbb{E}_{\tau\sim&space;\pi_{\theta^\prime}(\tau)}[r(\tau)]" title="J(\theta^\prime) = \mathop{\mathbb{E}_{\tau\sim \pi_{\theta^\prime}(\tau)}[r(\tau)]" />
</p>

Soit :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}J(\theta)&space;&=&space;\mathop{\mathbb{E}_{\tau\sim&space;\pi_{\theta^\prime}}(\tau)}[\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}r(\tau)]&space;\\\nabla_\theta&space;J(\theta)&space;&=&space;\mathop{\mathbb{E}_{\tau\sim&space;\pi_{\theta^\prime}}(\tau)}&space;\left[&space;{\frac{\nabla_\theta&space;\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}r(\tau)}&space;\right]&space;\\&=&space;\mathop{\mathbb{E}_{\tau\sim&space;\pi_{\theta^\prime}}(\tau)}&space;\left[&space;{\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}}&space;\nabla_\theta&space;\ln\pi_\theta(\tau)&space;&space;r(\tau)&space;\right]\end{align*}" title="\begin{align*}J(\theta) &= \mathop{\mathbb{E}_{\tau\sim \pi_{\theta^\prime}}(\tau)}[\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}r(\tau)] \\\nabla_\theta J(\theta) &= \mathop{\mathbb{E}_{\tau\sim \pi_{\theta^\prime}}(\tau)} \left[ {\frac{\nabla_\theta \pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}r(\tau)} \right] \\&= \mathop{\mathbb{E}_{\tau\sim \pi_{\theta^\prime}}(\tau)} \left[ {\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}} \nabla_\theta \ln\pi_\theta(\tau) r(\tau) \right]\end{align*}" />
</p>


pour <img src="https://latex.codecogs.com/svg.image?\theta&space;\neq&space;\theta^\prime" title="\theta \neq \theta^\prime" />. 

En explicitant les trajectoires, on trouve la forme suivante :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\nabla_\theta&space;J(\theta)&space;=&space;\mathop{\mathbb{E}_{\tau\sim&space;\pi_{\theta^\prime}}(\tau)}&space;\left[&space;(\prod_{t=1}^{T}\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta^\prime}(a_t|s_t)})&space;(\sum_{t=1}^{T}\nabla_\theta&space;\ln\pi_\theta(a_t|s_t))(\sum_{t=1}^{T}r(s_t,a_t))&space;\right]" title="\nabla_\theta J(\theta) = \mathop{\mathbb{E}_{\tau\sim&space;\pi_{\theta^\prime}}(\tau)} \left[ (\prod_{t=1}^{T}\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta^\prime}(a_t|s_t)}) (\sum_{t=1}^{T}\nabla_\theta \ln\pi_\theta(a_t|s_t))(\sum_{t=1}^{T}r(s_t,a_t)) \right]" />
</p>

Puisque <img src="https://latex.codecogs.com/svg.image?r(\tau)&space;\doteq&space;\sum_{t=1}^{T}r(s_t,a_t)" title="r(\tau) \doteq \sum_{t=1}^{T}r(s_t,a_t)" />, et que <img src="https://latex.codecogs.com/svg.image?\nabla_\theta&space;\ln&space;(\prod_{t=1}^{T}\pi_\theta(a_t|s_t))=\sum_{t=1}^{T}\nabla_\theta&space;\ln\pi_\theta(a_t|s_t)" title="\nabla_\theta \ln (\prod_{t=1}^{T}\pi_\theta(a_t|s_t))=\sum_{t=1}^{T}\nabla_\theta \ln\pi_\theta(a_t|s_t)" />.

<ins>Principe de causalité :</ins> En remarquant que la politique au temps <img src="https://latex.codecogs.com/svg.image?t" title="t" /> ne peut affecter les récompenses au temps <img src="https://latex.codecogs.com/svg.image?t^\prime" title="t^\prime" /> si <img src="https://latex.codecogs.com/svg.image?t&space;<&space;t^\prime" title="t < t^\prime" />, on peut récrire <img src="https://latex.codecogs.com/svg.image?\nabla_\theta&space;J(\theta)" title="\nabla_\theta J(\theta)" /> pour obtenir la forme finale avec laquelle travailler dans un contexte hors-politique :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\nabla_\theta&space;J(\theta)&space;=&space;\mathop{\mathbb{E}_{\tau\sim&space;\pi_{\theta^\prime}}(\tau)}&space;\left[&space;\sum_{t=1}^{T}\nabla_\theta&space;\ln\pi_\theta(a_t|s_t)&space;(\prod_{t^\prime=1}^{t}&space;\frac{\pi_\theta(s_{t^\prime}|a_{t^\prime})}{\pi_{\theta^\prime}(a_{t^\prime}|s_{t^\prime})})&space;(\sum_{t^\prime=t}^{T}r(s_{t^\prime},a_{t^\prime}))&space;\right]" title="\nabla_\theta J(\theta) = \mathop{\mathbb{E}_{\tau\sim \pi_{\theta^\prime}}(\tau)} \left[ \sum_{t=1}^{T}\nabla_\theta \ln\pi_\theta(a_t|s_t) (\prod_{t^\prime=1}^{t} \frac{\pi_\theta(s_{t^\prime}|a_{t^\prime})}{\pi_{\theta^\prime}(a_{t^\prime}|s_{t^\prime})}) (\sum_{t^\prime=t}^{T}r(s_{t^\prime},a_{t^\prime})) \right]" />
</p>

Cette forme convient aux deux cas évoqués en début de partie : on peut remplacer <img src="https://latex.codecogs.com/svg.image?\pi_{\theta^\prime}" title="\pi_{\theta^\prime}" /> par une politique d'exploration <img src="https://latex.codecogs.com/svg.image?b" title="b" />, ou considérer le cas de mises-à-jours asynchronisées.

Les méthodes Hors-Politique en Apprentissage par Renforcement (et en particulier appliqués aux méthodes du gradient de la politique) font l'objet de recherches actives, et de nombreux articles y sont consacré dans les grandes revues scientifiques du domaine. Cette section pose les fondement de l'approche Hors-Politique ; nous entretrons dans davantage de détails au cas-par-cas si le besoin s'en fait sentir pour les méthodes présentées ci-dessous.


---

<br/>

*<ins>En rédaction :</ins> A3C, TRPO, PPO*

<br/>

---

<br/>

#### Sources

**Cours et articles blogs :**

[Article très complet de Lilian Weng](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#off-policy-policy-gradient) sur les méthodes de gradient de la politique.

[CS182 à UC Berkeley, notes de cours de Serguey Levine](https://cs182sp21.github.io/), en particulier le cours magistral n°15. 

[Article de Daniel Seita](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/) sur les méthodes de gradient de la politique, élaborant des pistes d'intuitions à propos de REINFORCE avec valeurs de références.

["Introduction à l'apprentissage par renforcement", 2e édition](http://incompleteideas.net/book/the-book-2nd.html), l'ouvrage de référence de Sutton et Barto.

["Transparents de Jie-Han Chen, DASI spring 2018, National Cheng Kung University, Taiwan"](https://fr.slideshare.net/zhihua98/policy-gradient-98034864)


**Articles originaux, ayant proposés les méthodes :**

[Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4), 229-256.](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf)

[Degris, T., White, M., & Sutton, R. S. (2012). Off-policy actor-critic. arXiv preprint arXiv:1205.4839.](https://arxiv.org/pdf/1205.4839.pdf)



<br/>

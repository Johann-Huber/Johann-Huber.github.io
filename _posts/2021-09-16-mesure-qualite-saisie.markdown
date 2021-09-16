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


#### 2.3) Visualisation

Explicitons ces formalisations par un exemple visuel :

<p align="center">
	<img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/mesure_qualite_saisie/formalisation_saisie_domaine_vitesse.png">
	Schéma 2. Visualisation du formalisme dans le domaine des vitesses
</p>



<p align="center">
	<img src="https://raw.githubusercontent.com/Johann-Huber/Johann-Huber.github.io/master/assets/mesure_qualite_saisie/formalisation_saisie_domaine_efforts.png">
	Schéma 3. Visualisation du formalisme dans le domaine des efforts
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

**La saisie est isotropique si <img src="https://latex.codecogs.com/svg.image?Q_{GII}&space;\approx&space;1&space;"/>, et est proche d'une configuration singuluère si <img src="https://latex.codecogs.com/svg.image?Q_{GII}&space;\approx&space;0"/>.**

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

**3.2.3) Distance entre le centre de gravité du polygône de points de contact et le CM de l'objet**

Plus la distance entre le centre de gravité du polygône de contact et le centre de masse de l'objet diminue, plus les effets des forces gravitationnelles et inertielles diminuent.

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?(14):&space;Q_{DCC}=dist(CM,C)" title="(14): Q_{DCC}=dist(CM,C)"/>
</p>

Où C est le centre de gravité du polygône (2D) ou du polyhèdre (3D) de contact, et CM est le centre de masse de l'objet.

(+) Interprétation simple, léger en calculs si le CM est connu.

(-) En pratique, CM n'est jamais connu précisément /!\

(-) Le nombre de points de contact n'influence pas <img src="https://latex.codecogs.com/svg.image?Q_{DCC}"/> (alors que la stabilité est améliorée ...)


<br/>

**3.2.4) Orthogonalité**

Les êtres humains ont tendance à aligner leur main avec l'axe principal d'inertie à saisir.

Soit z le vecteur normal à la paume de la main, et soit u l'axe principal d'inertie de l'objet. L'angle entre ces deux vecteurs peut être calculé comme <img src="https://latex.codecogs.com/svg.image?\delta&space;=&space;\arccos(z&space;\cdot&space;u)"/>. La mesure est alors :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}(15):&space;Q_O&space;=&space;\left\{\begin{matrix}\delta,&space;&&space;\textrm{if}&space;\hspace{0.1cm}&space;\delta&space;<&space;\pi/4&space;\\\pi/2-\delta,&space;&&space;\textrm{if}&space;\hspace{0.1cm}&space;\pi/4&space;<&space;\delta&space;<&space;\pi/2&space;\\\delta-\pi/2,&space;&&space;\textrm{if}&space;\hspace{0.1cm}&space;\pi/2&space;<&space;\delta&space;<&space;3\pi/4&space;\\\pi&space;-&space;\delta,&space;&&space;\textrm{if}&space;\hspace{0.1cm}&space;\delta&space;>&space;3\pi/4\end{matrix}\right&space;.\end{align*}&space;"/>
</p>

Avec <img src="https://latex.codecogs.com/svg.image?\left\{\begin{matrix}max(Q_0)&space;=&space;\pi/4&space;\\&space;min(Q_0)&space;=&space;0\end{matrix}\right.&space;"/>. 

**La saisie est optimale lorsque <img src="https://latex.codecogs.com/svg.image?Q_0&space;\equiv&space;0"/>.** (u et z colinéaires ou coplanaires).

<br/>

**3.2.5) Marges d'incertitudes dans les positions des doigts**

<ins>Espace de saisie (ou espace de contact) :</ins> Espace defini par les n paramètres représentant les points de contact possibles de n doigts sur le contour 2D de l'objet.

<ins>Espace des forces de saisie :</ins> Sous-espace de l'espace de saisie représentant les force appliquées aux points de contact.

FCS est l'union d'un ensemble de polyhèdres convexes <img src="https://latex.codecogs.com/svg.image?CP_i" />, et est utilisé dans plusieurs travaux pour calculer le FCS d'objets polygonaux et n'importe quel nombre de doigts, avec ou sans fonctions.

Une plus grande distance avec le contour de FCS implique une saisie plus sûre.

Soit p un point de l'espace de saisie. Le rayon de l'hypersphère la plus large qui a pour centre p et qui est entièrement contenue dans l'un des polyhèdres convexes <img src="https://latex.codecogs.com/svg.image?CP_i" /> qui forment le FCS est une métrique de qualité de saisie.

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?(16):&space;Q_{MUF}=\min_{p_j\in\partial&space;CP_i}&space;\left\|&space;p&space;-&space;p_j&space;\right\|"/>
</p>

Où <img src="https://latex.codecogs.com/svg.image?\partial&space;CP_i"/> est le contour de <img src="https://latex.codecogs.com/svg.image?CP_i" />.

(+) : Maximise l'effet de l'incertitude sur la position des doigts pendant la saisie.

(-) : Difficile à appliquer à des objets non-polygonaux (2D ou 3D), de fait de la complexité et la grande dimensionnalité de l'espace de saisie qui en résulte.

<br/>

**3.2.6) Régions de contact indépendantes**

<ins>Variante 1 :</ins> Objets polynomiaux

Régions de contact indépendantes (*independent contact regions*) : ensemble de régions <img src="https://latex.codecogs.com/svg.image?ICR_i"/> sur les contours de l'objet tels qu'elles produisent une force de saisie indépendante des points de contacts. On nomme cet ensemble ICRS (pour *ICR Set*).

Il s'agit d'une région fermée dans l'espace de saisie contenue entièrement dans l'espace des forces de saisie.

Pour des objets 2D et n doigts, la région est un parallélépipède B aligné avec l'axe de référence.

De plus grandes régions de B impliquent un plus grand ensemble de saisies FC possibles, et saisir en plaçant chaque doigt au centre de chaque ICR permet de plus larges erreurs de positionnement autorisées pour chaque doigt.

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?(17):&space;Q_{ICR}=L_{min}"/>
</p>

Avec <img src="https://latex.codecogs.com/svg.image?L_{min}"/> étant la taille de la région indépendante <img src="https://latex.codecogs.com/svg.image?ICR_i"/> la plus petite (c-à-d la taille de la plus petite arrête de B).

Un <img src="https://latex.codecogs.com/svg.image?Q_{ICR}" title="Q_{ICR}"/> élevé implique de meilleures possibilités de trouver un ensemble de positions de contact permettant la saisie.

(+) : Interprétable physiquement, utile dans le cas d'incertitude sur la position des doigts.

(-) : Lourd en calculs (en particulier l'ICRS, c-à-d B)

<ins>Variante 2 :</ins> Objets 2D non-polynomiaux (pince : 2 doigts)

Dans ce cas, l'espace des forces de saisies sont limités par des courbes.

L'ICRS est obtenu en maximisant l'aire de B.

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?(18):&space;Q_{ICR'}=aire(B)"/>
</p>

À noter que <img src="https://latex.codecogs.com/svg.image?Q_{ICR}"/> et <img src="https://latex.codecogs.com/svg.image?Q_{ICR'}"/> sont adaptés pour des objets 2D discrétisés de n'importe quelle forme (contours représentés par un nombre fini de points). La qualité de saisie est alors associée au nombre de points sur le contour de B (<img src="https://latex.codecogs.com/svg.image?Q_{ICR}"/>) ou dans B (<img src="https://latex.codecogs.com/svg.image?Q_{ICR'}"/>).


<ins>Variante 3 :</ins> Objets polyhédriques

La mesure de qualité est basée sur un ensemble ICRS : somme des distances entre chacuns des i points de contact <img src="https://latex.codecogs.com/svg.image?(x_{i},y_{i},z_{i})"/> et le centre <img src="https://latex.codecogs.com/svg.image?(x_{i0},y_{i0},z_{i0})"/> de la région de contact indépendant correspondante.

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?(19):&space;Q_{ICR^n}=\frac{1}{n}\sum_{i=1}^{n}\sqrt{(x_i-x_{i0})^2&plus;(y_i-y_{i0})^2&plus;(z_i-z_{i0})^2}"/>
</p>

On nomme <img src="https://latex.codecogs.com/svg.image?Q_{ICR^n}"/> l'indice d'incertitude de saisie (*uncertainty grasp index*), ou la marge de saisie (*grasp margin*).

La saisie est optimale lorsque <img src="https://latex.codecogs.com/svg.image?Q_{ICR^n}&space;=&space;0"/>, c-à-d lorsque les doigts sont localisés au centre de chaque <img src="https://latex.codecogs.com/svg.image?ICR_i"/>.


<br/>

#### 3.3) Mesures considérant les limites de forces de saisie



**3.3.1) La plus grande contrainte minimale resisté**

**3.3.2) Volume de l'espace d'efforts de saisie (volume de P)**

**3.3.3) Découplage des forces et des couples**

**3.3.4) Composantes normales des forces**

**3.3.5) Mesure adaptée à la tâche**


<br/>

#### 3.4) Exemples



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




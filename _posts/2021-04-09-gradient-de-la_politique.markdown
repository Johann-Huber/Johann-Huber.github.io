---
layout: post
title:  "Gradient de la politique"
date:   2021-04-09 21:07:00 +0200
categories: Apprentissage-profond
---



Le Gradient de la Politique (ou *Policy Gradient*) est une approche de résolution de problèmes en Apprentissage par Renforcement.

Dans ce paradigme d'apprentissage automatique, il s'agit de trouver une stratégie de comportement optimale pour un ou plusieurs agents, de manière à maximiser les récompenses obtenues. Les méthodes de **gradient de la politique** visent à modéliser et à optimiser la politique directement. En général, cette dernière est modélisée par une fonction paramétrique de <img src="https://latex.codecogs.com/svg.image?\theta"/>, notée <img src="https://latex.codecogs.com/svg.image?\pi_\theta(a|s)"/>. Les valeurs de la fonction de récompenses (fonction objectif) dépendent de cette politique. Plusieurs algorithmes peuvent être appliqués pour optimiser <img src="https://latex.codecogs.com/svg.image?\theta"/> de sorte à maximiser les performances de l’agent sur une tâche donnée.


<ins>Note de rédaction :</ins> Je mettrai à jour régulièrement cette liste pour qu'elle contiennent l'essentiel des informations pour appréhender les algorithmes de l'État de l'Art sans avoir à plonger trop en détail dans les articles originaux. J'ajouterai par ailleurs, autant que possible, une implémentation python sous la forme de script unique autant que possible.

<ins>Crédit :</ins> Ces notes s'appuient très largement sur le [blog de lilian weng](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#off-policy-policy-gradient) et sur les [notes de cours de Serguey Levine, CS182 à UC Berkeley](https://cs182sp21.github.io/). L'ensemble des sources utiliées sont listés à la fin de l'article.



<br/>


### Comment optimiser la politique d'un agent ?

La fonction de récompense (ou fonction de performance) est définie par :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?J(\theta)=\sum_{s\in&space;S}&space;\mu(s)v_\pi(s)&space;=&space;\sum_{s\in&space;S}&space;\mu(s)\sum_{a\in&space;A}\pi_\theta(a|s)q_\pi(s)"/>
</p>

Avec <img src="https://latex.codecogs.com/svg.image?\mu(s)"/>: la distribution stationnaire de la chaïne markovienne sous la politique <img src="https://latex.codecogs.com/svg.image?\pi_\theta"/>.

On cherche donc à trouver les valeurs de <img src="https://latex.codecogs.com/svg.image?\theta"/> qui maximisent la récompense. Ce problème peut être résolu via la méthode d'ascension de gradient :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\theta_{t&plus;1}=&space;\theta_t&space;&plus;&space;\alpha*&space;\widehat{\nabla&space;J(\theta_t)}"/>
</p>

Où <img src="https://latex.codecogs.com/svg.image?\widehat{\nabla&space;J(\theta_t)}" /> est une estimation stochastique, dont l'espérence est le gradient de la performance mesurée par rapport à <img src="https://latex.codecogs.com/svg.image?\theta_t"/>.

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


Pour alléger l'écriture, posons : <img src="https://latex.codecogs.com/svg.image?\phi(s)&space;\doteq&space;\sum_{a&space;\in&space;A}&space;\nabla&space;\pi(a|s)q_\pi(s,a)"/>.

Soit <img src="https://latex.codecogs.com/svg.image?p_\pi(s\rightarrow&space;s^\prime,&space;k)"/> la probabilité de transitionner d'un état s à s' en suivant la politique <img src="https://latex.codecogs.com/svg.image?\pi_\theta"/>. On notera <img src="https://latex.codecogs.com/svg.image?p_\pi(s\rightarrow&space;s^\prime,&space;k)&space;=&space;p(s\rightarrow&space;s^\prime,&space;k)"/> par commodité d'écriture.

Cette probabilité s'exprime comme produit de la probabilité de choisir l'action a à partir de s (liée à la politique), et de la probabilité d'atteindre l'état s' en partant de l'état s et de l'action a (probabilité liée aux dynamiques de l'environnement). On somme les probabilités sur chaque action pour obtenir la probabilité de transition : <img src="https://latex.codecogs.com/svg.image?p(s\rightarrow&space;s^\prime,&space;k=1)=\sum_a&space;\pi(a|s)p(s^\prime|s,a)"/>.

Notons par ailleurs que l'on peut exprimer la probabilité de transitionner d'un état vers un autre sur plusieurs pas sous la forme d'un produit des probabilités des transitions intermédiaires. Pour <img src="https://latex.codecogs.com/svg.image?\forall&space;(s,s^\prime,s^{\prime\prime})&space;\in&space;S^3"/>, et <img src="https://latex.codecogs.com/svg.image?\forall&space;k\in\mathbb{N}^{*}"/>, on a : <img src="https://latex.codecogs.com/svg.image?p_\pi(s\rightarrow&space;s^{\prime\prime},&space;k)&space;=&space;p(s\rightarrow&space;s^\prime,&space;k-1)&space;p(s^\prime&space;\rightarrow&space;s^{\prime\prime},&space;1)"/>.

Grâce à ces expression, on peut dérouler la récursion :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla&space;v_\pi(s)&space;&=&space;\phi(s)&plus;&space;\sum_{a&space;\in&space;A}&space;\pi(a|s)&space;\sum_{s^\prime}p(s^\prime|s,a)&space;\nabla&space;v_\pi(s^\prime)&space;\\&=&space;\phi(s)&space;&plus;&space;p(s\rightarrow&space;s^\prime,1)&space;\nabla&space;v_\pi(s^\prime)&space;\\&=&space;\phi(s)&space;&plus;&space;p(s\rightarrow&space;s^\prime,1)(\phi(s^\prime)&space;&plus;&space;p(s^\prime&space;\rightarrow&space;s^{\prime&space;\prime},1)\nabla&space;v_\pi(s^{\prime&space;\prime}))&space;\\&=&space;\phi(s)&space;&plus;&space;p(s\rightarrow&space;s^\prime,1)\phi(s^\prime)&space;&plus;&space;p(s&space;\rightarrow&space;s^{\prime&space;\prime},2)\nabla&space;v_\pi(s^{\prime&space;\prime})&space;\\&=&space;\phi(s)&space;&plus;&space;p(s\rightarrow&space;s^\prime,1)\phi(s^\prime)&space;&plus;&space;p(s&space;\rightarrow&space;s^{\prime&space;\prime},2)\phi(s^{\prime&space;\prime})&space;&plus;&space;p(s&space;\rightarrow&space;s^{\prime&space;\prime&space;\prime},3)\phi(s^{\prime&space;\prime&space;\prime})&space;&plus;&space;...\\\nabla&space;v_\pi(s)&space;&=&space;\sum_{x&space;\in&space;S}&space;\sum_{k=0}^{\infty}p(s\rightarrow&space;x,&space;k)\phi(s)\end{align*}"/>
</p>


Le théorème du gradient de la politique fait intervenir la distribution stationnaire des états <img src="https://latex.codecogs.com/svg.image?\mu_\pi(s)"/> - notée <img src="https://latex.codecogs.com/svg.image?\mu(s)"/>) - définit de la façon suivante :
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

Par définition, <img src="https://latex.codecogs.com/svg.image?J(\theta)=r(\theta)"/>. Or <img src="https://latex.codecogs.com/svg.image?r(\theta)"/> est indépendant de s. L'équation du gradient de la performance est donc toujours juste si l'on multiplie le terme de droite par <img src="https://latex.codecogs.com/svg.image?\sum_s&space;\mu(s)"/>, puisque <img src="https://latex.codecogs.com/svg.image?\sum_s&space;\mu(s)&space;=&space;1"/>.

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

L'article Estimation de l'Avantage Généralisée (GAE) [Schulman et al., 2016](https://arxiv.org/pdf/1506.02438.pdf) ((i) trad ?) propose une forme générale du gradient de la performance, mettant en lumière les différentes déclinaisons de cette forme que l'on peut trouver dans la littérature :

En posant g, le gradient de la performance, tel que <img src="https://latex.codecogs.com/svg.image?g&space;\doteq&space;\nabla_\theta&space;\mathop{\mathbb{E}}[\sum_{t=0}^\infty&space;r_t]"/>

On a la forme générale : 
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

Les algorithmes présentés font tous mention du paramètre <img src="https://latex.codecogs.com/svg.image?\gamma&space;\in&space;&space;\]0;1\]"/>, le facteur d'atténuation. Sa définition est implicite, afin d'éviter les redondances.

<br/>

### REINFORCE

L'algorithme **REINFORCE** (gradient de la Politique avec méthode Monte-Carlo) repose sur l'expression du gradient de la performance obtenue dans le Théorème du Gradient de la Politique, appliqué aux épisodes prélevés (i.e. obtenus par interaction directe avec l'environnement). En constatant que <img src="https://latex.codecogs.com/svg.image?q_\pi(s_t,a_t)=&space;\mathop{\mathbb{E}}_\pi[G_t&space;|s_t,&space;a_t]"/>, on trouve :


<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla_\theta&space;J(\theta)&space;&=&space;\mathop{\mathbb{E}}_\pi[q_\pi(s,a)&space;\nabla_\theta\ln\pi_\theta(a,s)]&space;\\&=&space;\mathop{\mathbb{E}}_\pi[G_t&space;\nabla_\theta\ln\pi_\theta(a,s)]\end{align*}"/>
</p>

Autrement dit, on peut optimiser <img src="https://latex.codecogs.com/svg.image?\theta"/> à partir du retour obtenu au cours d'un épisode. Cette approche exploite la trajectoire observée sur l'épisode entier pour faire ses mises à jours, c'est pourquoi on parle de méthode de type Monte Carlo.

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

Avec Reinforce, <img src="https://latex.codecogs.com/svg.image?\theta"/> est mis-à-jour en utilisant directement le retour observé lors d'une interaction avec l'environnement. Il n'y a donc **pas de biais** : la montée de gradient fera toujours évoluer <img src="https://latex.codecogs.com/svg.image?\theta"/> vers des valeurs qui augmenteront l'espérence de retour. En revanche, cette méthode introduit une **forte variance** : de trop grands pas sont réalisés en fonction de l'échantillon de trajectoire considéré, rendant la convergence vers une configuration optimale plus difficile.



### REINFORCE avec valeurs de référence

L'algorithme **REINFORCE avec valeurs de référence** est variante bien connue de l'algorithme REINFORCE. Il s'agit simplement de retrancher au retour de l'épisode une valeur de référence dans le calcul du gradient de la performance. Cette modification a pour effet de **réduire la variance** tout en assurant l'absence de biais.

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
  (AJOUTER IMAGE TRAJECTOIRE MAX PERFS)
  <br/>
  Example tiré du <a href="http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_4_policy_gradient.pdf">cours de S. Levine à l'UC Berkeley</a>
</p>


Pour comprendre en quoi cette idée permet de faciliter grandement l'entraînement, imaginons un [MDP](https://fr.wikipedia.org/wiki/Processus_de_d%C3%A9cision_markovien) à récompenses positives. Imaginons que trois essais nous donne les trois trajectoires, représentées ci-dessus (en z se trouve la performance, que l'on cherche à maximiser); nous aurions trois récomponses positives, plus ou moins grandes selon la qualité de la trajectoire. Pourtant, il serait souhaitable d'augmenter la probabilité de choisir la meilleure trajectoire, et de réduire celle de choisir la moins bonne.

L'intuition derrière l'utilisation de la valeur de référence est la suivante : en soustrayant la récompense à la récompose moyenne, on incitera la politique à **choisir plus souvent des trajectoires qui ont permis l'obtention d'une récompense plus élevée que la moyenne**, tout en l'incitant à **moins choisir les trajectoires ayant abouties à une récompense inférieure à la moyenne**.


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

L'algorithme **Acteur-Critique** ressemble beaucoup à l'algorithme REINFORCE avec valeurs de référence : il s'agit de calculer une différence entre un retour, et une valeur de référence. En revanche, deux différences importantes les distinguent.

Comme toutes les méthodes de type Monte-Carlo, REINFORCE ne fait pas de mise-à-jour avant la fin de l'épisode. Par ailleurs, la valeur de référence ne tient compte que de la valeur de l'état initial (avant de prendre l'action), et ne permet par conséquent pas de juger de la qualité de l'action choisie. Par cette approche, on répond à la question : **"L'agent a-t-il bien fait de se trouver à cette position au temps t ?"**, en tenant compte de l'épisode entier.

Dans le cas le plus simple de la méthode Acteur-Critique, le retour utilisé dans la mise-à-jour des paramètres de la politique calcule la différence entre la valeur de l'état au temps t, et celle de l'état au temps t+1 (en tenant compte du facteur d'atténuation) ; c'est le retour 1-pas, noté <img src="https://latex.codecogs.com/svg.image?G_{t:t&plus;1}"/> (comme dans les méthodes TD(0), SARSA(0) ou Q-apprentissage). Cette approche permet donc d'évaluer la différence de valeur entre l'état initial et le nouvel état, autrement dit de juger de la qualité de l'action prise par l'agent. On répond ici à la question : **"L'agent a-t-il bien fait de choisir cette action au temps t?"**, en ne tenant compte que de la transition entre les temps t et t+1.

En résumé : la politique agit, et la méthode de retour intermédiaire critique.


Il existe de nombreuses variantes, impliquant entre autres : la fonction d'avanntage, les valeurs-Q, la méthode SARSA 1-étape, le retour n-étapes, et différentes approches d'entraîements (séquentiel, asynchronisé).

L'algorithme présenté dans cette section correspond donc à une certaine variante de la méthode Acteur-Critique : cas épisodique, 1-étape, entièrement en ligne (i.e. sans avoir recourt à une mémoire tampon), avec fonction d'avantage.

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

<ins>Remarque :</ins> En pratique, on peut simplement étendre cet algorithme aux itérations sur des lots, en accumulant un nombre <img src="https://latex.codecogs.com/svg.image?n"/> de transitions, et en appliquant les mêmes étapes mentionnées ci-dessus. À noter qu'il n'y aucune nécessité de lien entre les transitions (i.e. elles n'ont pas à provenir d'une même trajectoire) ; tant que l'on a des quadruplets <img src="https://latex.codecogs.com/svg.image?(s,a,s^\prime,r)"/>, nous serons en mesure d'appliquer l'algorithme.


<br/>


### Gradient de la politique en contexte Hors-Politique


Tous les algorithmes jusqu'ici présentés optimisent la politique qui a été utilisée pour obtenir des trajectoires. Dans cette section, nous considérons les cas dans lesquels **la politique d'exploration n'est pas la même que la politique cible**. 

Jetons à nouveau un coup d'oeil à l'estimateur utilisé par les algorithmes de gradient de la politique :

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\nabla_\theta&space;J(\theta)&space;=&space;\mathop{\mathbb{E}}_{\tau&space;\sim&space;\pi_\theta(\tau)}[\nabla_\theta&space;\ln&space;\pi_\theta(\tau)r(\tau)]"/>
</p>

Le gradient de la performance est calculé à partir de l'espérence sur <img src="https://latex.codecogs.com/svg.image?\tau&space;\sim&space;\pi_\theta(\tau)"/>. Une trajectoire relevée à partir d'un certain <img src="https://latex.codecogs.com/svg.image?\theta"/> n'est donc plus valable après application d'une itération de l'algorithme de monté de gradient sur <img src="https://latex.codecogs.com/svg.image?\theta"/>. Autrement dit, il est **nécessaire de prélever une trajectoire après chaque mise-à-jours de** <img src="https://latex.codecogs.com/svg.image?\theta"/>, rendant caduques toutes les mesures précédentes. Le problème est de taille, sitôt que l'on se trouve sur des tâches pour lesquelles les mesures sont lentes et coûteuses (comme en robotique, par exemple).

Pour cette raison, il est souhaitable de définir des algorithmes permettant d'optimiser une politique à partir de mesures réalisées *Hors-Politique*, c'est-à-dire avec une autre politique que celle que l'on optimise.

Par ailleurs, une telle approche permet d'utiliser une politique la plus efficace pour l'exploration, sans avoir à contraindre la politique que l'on est en train d'optimiser pour qu'inciter à explorer de nouvelles trajectoires.


## Échantillonnage préférentiel 

Pour rappel, on cherche <img src="https://latex.codecogs.com/svg.image?\theta^*&space;=&space;\operatorname*{argmax}_\theta&space;J(\theta)" title="\theta^* = \operatorname*{argmax}_\theta J(\theta)" />, que l'on optimise à partir de : 

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?J(\theta)&space;=&space;\mathop{\mathbb{E}}_{\tau\sim\pi_\theta(\tau)}&space;[r(\tau)]" title="J(\theta) = \mathop{\mathbb{E}}_{\tau\sim\pi_\theta(\tau)} [r(\tau)]" />	
</p>

Supposons que nous ne disposons pas d'exemples de trajectoires correspondant à <img src="https://latex.codecogs.com/svg.image?\tau\sim\pi_\theta(\tau)" title="\tau\sim\pi_\theta(\tau)" />, mais que nous ayons à la place des trajectoires <img src="https://latex.codecogs.com/svg.image?\tau\sim\pi_{\theta^\prime}(\tau)" title="\tau\sim\pi_{\theta^\prime}(\tau)" /> pour un autre jeu de paramtère <img src="https://latex.codecogs.com/svg.image?\theta^\prime" title="\theta^\prime" />.

On a :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}\mathop{\mathbb{E}}_{\tau\sim\pi_\theta}&space;&=&space;\int&space;\pi_\theta(\tau)r(\tau)d\tau&space;\\&=&space;\int&space;\pi_{\theta^\prime}(\tau)&space;\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}&space;r(\tau)d\tau&space;\\&=&space;\mathop{\mathbb{E}}_{\tau\sim\pi_{\theta^\prime}}[\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}r(\tau)]\end{align*}" title="\begin{align*}\mathop{\mathbb{E}}_{\tau\sim\pi_\theta} &= \int \pi_\theta(\tau)r(\tau)d\tau \\&= \int \pi_{\theta^\prime}(\tau) \frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)} r(\tau)d\tau \\&= \mathop{\mathbb{E}}_{\tau\sim\pi_{\theta^\prime}}[\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}r(\tau)]\end{align*}" />
</p>

On appelle **poids d'importance** le coefficient <img src="https://latex.codecogs.com/svg.image?\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}" title="\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}" />. On parle des poids d'importance au pluriels, dans le sens où chaque terme en <img src="https://latex.codecogs.com/svg.image?\tau" title="\tau" /> comprend implicitement le produit des termes en <img src="https://latex.codecogs.com/svg.image?(s_i,&space;a_i)" title="(s_i, a_i)" /> associés à la trajectoire.

Ainsi : 
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\nabla&space;J(\theta)&space;=&space;\mathop{\mathbb{E}}_{\tau\sim\pi_{\theta^\prime}}[\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}r(\tau)]" title="\nabla J(\theta) = \mathop{\mathbb{E}}_{\tau\sim\pi_{\theta^\prime}}[\frac{\pi_\theta(\tau)}{\pi_{\theta^\prime}(\tau)}r(\tau)]" />
</p>

Or :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\pi_\theta(\tau)&space;\doteq&space;p(s_1)&space;\prod_{t=1}^{T}\pi_\theta(s_t|s_t)p(s_{t&plus;1}|s_t,a_t)" title="\pi_\theta(\tau) \doteq p(s_1) \prod_{t=1}^{T}\pi_\theta(s_t|s_t)p(s_{t+1}|s_t,a_t)" />
</p>

Avec <img src="https://latex.codecogs.com/svg.image?p(s_1)" title="p(s_1)" /> la probabilité de démarrer l'épisode à l'état <img src="https://latex.codecogs.com/svg.image?s_1" title="s_1" />, et <img src="https://latex.codecogs.com/svg.image?T" title="T" /> le nombre d'étape dans l'épisode avant que l'état terminal n'ait été atteint.

Les poids d'importance s'expriment donc de la façon suivant :
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


Pour <img src="https://latex.codecogs.com/svg.image?\theta&space;\neq&space;\theta^\prime" title="\theta \neq \theta^\prime" />. 

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

Les méthodes Hors-Politique en apprentissage par renforcement (et en particulier appliqués aux méthodes du gradient de la politique) font l'objet de recherches actives, et une importante quantité d'articles publiés dans les grandes revues scientifiques du domaine y sont consacré. Cette section pose les fondement de l'approche Hors-Politique ; nous entretrons dans davantage de détails au cas-par-cas si le besoin s'en fait sentir pour les méthodes présentées ci-dessous.


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



**Articles originaux, ayant proposés les méthodes :**

[Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4), 229-256.](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf)

[Degris, T., White, M., & Sutton, R. S. (2012). Off-policy actor-critic. arXiv preprint arXiv:1205.4839.](https://arxiv.org/pdf/1205.4839.pdf)



<br/>

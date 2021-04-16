---
layout: post
title:  "Gradient de la politique"
date:   2021-04-09 21:07:00 +0200
categories: Apprentissage-profond
---





Les résultats spectaculaires obtenus depuis 2015 (DQN, GO, DOTA, Alphafold) grâce à l'apprentissage par renforcement proviennent de deux raisons majeurs : D'une part, l'augmentation toujours croissante de la capacité de calcul (SOURCE?), et d'autre part, l'émergence de méthodes d'apprentissages capables d'approximer des fonctions dans des espaces de très grandes dimensions. On regroupe ces méthodes sous le nom de d'algorithmes de **gradient de la politique**.

<ins>Note de rédaction :</ins> Je mettrai à jour régulièrement cette liste pour qu'elle contiennent l'essentiel des informations pour appréhender les algorithmes de l'État de l'Art sans avoir à plonger trop en détail dans les articles originaux. J'ajouterai par ailleurs, autant que possible, une implémentation python sous la forme de script unique autant que possible.

<ins>Crédit :</ins> : Ces notes s'appuient très largement sur le [blog de lilian weng](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#off-policy-policy-gradient) et sur les [notes de cours de Serguey Levine, CS182 à UC Berkeley](https://cs182sp21.github.io/). L'ensemble des sources utiliées sont listés à la fin de l'article.



<br/>

## Qu’est-ce que le Gradient de la Politique ?

Le Gradient de la Politique (ou *Policy Gradient*) est une approche de résolution de problèmes en Apprentissage par Renforcement.

Dans cette branche de l'IA, l’objectif est de trouver une stratégie de comportement optimale pour un agent, de sorte qu’il puisse maximiser ses récompenses. Les méthodes de **gradient de la politique** visent à modéliser et à optimiser la politique directement. En général, cette dernière est modélisée par une fonction paramétrique de <img src="https://latex.codecogs.com/svg.image?\theta" title="\theta" />, notée <img src="https://latex.codecogs.com/svg.image?\pi_\theta(a|s)" title="\pi_\theta(a|s)" />. Les valeurs de la fonction de récompenses (fonction objectif) dépendent de cette politique. Plusieurs algorithmes peuvent être appliqués pour optimiser <img src="https://latex.codecogs.com/svg.image?\theta" title="\theta" /> de sorte à maximiser les performances de l’agent sur une tâche donnée.

<br/>


### Notations

(Ajouter tableau notations)


La fonction de récompense (ou fonction de performance) est définie par :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?J(\theta)=\sum_{s\in&space;S}&space;\mu(s)v_\pi(s)&space;=&space;\sum_{s\in&space;S}&space;\mu(s)\sum_{a\in&space;A}\pi_\theta(a|s)q_\pi(s)"/>
</p>

Avec <img src="https://latex.codecogs.com/svg.image?\mu(s)" title="\mu(s)" />: la distribution stationnaire de la chaïne markovienne sous la politique <img src="https://latex.codecogs.com/svg.image?\pi_\theta" title="\pi_\theta" />.

On cherche donc à trouver les valeurs de <img src="https://latex.codecogs.com/svg.image?\theta" title="\theta" /> qui maximisent la récompense. Ce problème peut être résolu via la méthode d'ascension de gradient :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\theta_{t&plus;1}=&space;\theta_t&space;&plus;&space;\alpha*&space;\widehat{\nabla&space;J(\theta_t)}" title="\theta_{t+1}= \theta_t + \alpha* \widehat{\nabla J(\theta_t)}" />
</p>

Où <img src="https://latex.codecogs.com/svg.image?\widehat{\nabla&space;J(\theta_t)}" title="\widehat{\nabla J(\theta_t)}" /> est une estimation stochastique, dont l'espérence est le gradient de la performance mesurée par rapport à <img src="https://latex.codecogs.com/svg.image?\theta_t" title="\theta_t" />.

En dérivant, on obtient :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\nabla_\theta&space;J(\theta)=\sum_{s\in&space;S}&space;\nabla&space;_\theta&space;\mu(s)&space;v_\pi(s)&space;&plus;&space;\mu(s)&space;\nabla_\theta&space;v_\pi(s)" title="\nabla_\theta J(\theta)=\sum_{s\in S} \nabla _\theta \mu(s) v_\pi(s) + \mu(s) \nabla_\theta v_\pi(s)" />
</p>


Comment estimer <img src="https://latex.codecogs.com/svg.image?\nabla_\theta&space;\mu(s)" title="\nabla_\theta \mu(s)" />, dans les cas où l'on ne connait pas les dynamiques qui régissent l'environnement dans lequel l'agent évolue ?

Il existe un moyen de contourner le problème, en écrivant le gradient de la performance sous une forme simplifiée. C'est ce que permet de le théorème du Gradient de la Politique.

<br/>

### Théorème du Gradient de la Politique


Le théorème du Gradient la Politique s'énonce de la façon suivante :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla_\theta&space;J(\theta)&space;&=&space;\nabla_\theta&space;\sum_{s\in&space;S}&space;\mu(s)&space;\sum_{a&space;\in&space;A}&space;q_\pi(s,a)&space;\pi_\theta(a|s)&space;\\&&space;\propto&space;\sum_{s\in&space;S}&space;\mu(s)&space;\sum_{a&space;\in&space;A}&space;q_\pi(s,a)&space;\nabla_\theta&space;\pi_\theta(a|s)\end{align*}" title="\begin{align*} \nabla_\theta J(\theta) &= \nabla_\theta \sum_{s\in S} \mu(s) \sum_{a \in A} q_\pi(s,a) \pi_\theta(a|s) \\& \propto \sum_{s\in S} \mu(s) \sum_{a \in A} q_\pi(s,a) \nabla_\theta \pi_\theta(a|s)\end{align*}" />
</p>

Cette formulation permet d'estimer le gradient de la performance en s'afranchissant du terme <img src="https://latex.codecogs.com/svg.image?\nabla&space;\mu(s)" title="\nabla \mu(s)" />.

#### Preuve


On distingue le **cas épisodique** du **cas continue**, pour lesquels la fonction de performance ne s'exprime pas exactement de la même manière.

<ins>Remarque :</ins> On notera implicitement : <img src="https://latex.codecogs.com/svg.image?\nabla&space;\doteq&space;\nabla_\theta" title="\nabla \doteq \nabla_\theta" />, et <img src="https://latex.codecogs.com/svg.image?\pi(a|s)\doteq\pi_\theta(a|s)" title="\pi(a|s)\doteq\pi_\theta(a|s)" />.

**Cas épisodique**
 
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla&space;v_\pi(s)&space;&=&space;\nabla&space;\sum_{a&space;\in&space;A}&space;\pi(a|s)&space;q_\pi(s,a)&space;\\&=&space;\sum_{a&space;\in&space;A}&space;(\nabla&space;\pi(a|s)&space;q_\pi(s,a)&space;&plus;&space;\pi(a|s)&space;\nabla&space;q_\pi(s,a))&space;\\&=&space;\sum_{a&space;\in&space;A}&space;(\nabla&space;\pi(a|s)&space;q_\pi(s,a)&space;&plus;&space;\pi(a|s)&space;\nabla&space;(\sum_{r,s^\prime}p(r,s^\prime|s,a)(r&plus;v_\pi(s^\prime))))&space;\\&=&space;\sum_{a&space;\in&space;A}&space;(\nabla&space;\pi(a|s)&space;q_\pi(s,a)&space;&plus;&space;\pi(a|s)&space;\nabla&space;\sum_{s^\prime}p(s^\prime|s,a)(r&plus;v_\pi(s^\prime)))&space;\\\nabla&space;v_\pi(s)&space;&=&space;\sum_{a&space;\in&space;A}&space;(\nabla&space;\pi(a|s)&space;q_\pi(s,a)&space;&plus;&space;\pi(a|s)&space;\sum_{s^\prime}p(s^\prime|s,a)&space;\nabla&space;v_\pi(s^\prime))\end{align*}" title="\begin{align*} \nabla v_\pi(s) &= \nabla \sum_{a \in A} \pi(a|s) q_\pi(s,a) \\&= \sum_{a \in A} (\nabla \pi(a|s) q_\pi(s,a) + \pi(a|s) \nabla q_\pi(s,a)) \\&= \sum_{a \in A} (\nabla \pi(a|s) q_\pi(s,a) + \pi(a|s) \nabla (\sum_{r,s^\prime}p(r,s^\prime|s,a)(r+v_\pi(s^\prime)))) \\&= \sum_{a \in A} (\nabla \pi(a|s) q_\pi(s,a) + \pi(a|s) \nabla \sum_{s^\prime}p(s^\prime|s,a)(r+v_\pi(s^\prime))) \\\nabla v_\pi(s) &= \sum_{a \in A} (\nabla \pi(a|s) q_\pi(s,a) + \pi(a|s) \sum_{s^\prime}p(s^\prime|s,a) \nabla v_\pi(s^\prime))\end{align*}" />
</p>


On obtient une forme recursive, reliant l'état s à l'état suivant s'.


Pour alléger l'écriture, posons : <img src="https://latex.codecogs.com/svg.image?\phi(s)&space;\doteq&space;\sum_{a&space;\in&space;A}&space;\nabla&space;\pi(a|s)q_\pi(s,a)" title="\phi(s) \doteq \sum_{a \in A} \nabla \pi(a|s)q_\pi(s,a)" />.

Soit <img src="https://latex.codecogs.com/svg.image?p_\pi(s\rightarrow&space;s^\prime,&space;k)" title="p_\pi(s\rightarrow s^\prime, k)" /> la probabilité de transitionner d'un état s à s' en suivant la politique <img src="https://latex.codecogs.com/svg.image?\pi_\theta" title="\pi_\theta" />. On notera <img src="https://latex.codecogs.com/svg.image?p_\pi(s\rightarrow&space;s^\prime,&space;k)&space;=&space;p(s\rightarrow&space;s^\prime,&space;k)" title="p_\pi(s\rightarrow s^\prime, k) = p(s\rightarrow s^\prime, k)" /> par commodité d'écriture.

Cette probabilité s'exprime comme produit de la probabilité de choisir l'action a à partir de s (liée à la politique), et de la probabilité d'atteindre l'état s' en partant de l'état s et de l'action a (probabilité liée aux dynamiques de l'environnement). On somme les probabilités sur chaque action pour obtenir la probabilité de transition : <img src="https://latex.codecogs.com/svg.image?p(s\rightarrow&space;s^\prime,&space;k=1)=\sum_a&space;\pi(a|s)p(s^\prime|s,a)" title="p(s\rightarrow s^\prime, k=1)=\sum_a \pi(a|s)p(s^\prime|s,a)" />.

Notons par ailleurs que l'on peut exprimer la probabilité de transitionner d'un état vers un autre sur plusieurs pas sous la forme d'un produit des probabilités des transitions intermédiaires. Pour <img src="https://latex.codecogs.com/svg.image?\forall&space;(s,s^\prime,s^{\prime\prime})&space;\in&space;S^3" title="\forall (s,s^\prime,s^{\prime\prime}) \in S^3" />, et <img src="https://latex.codecogs.com/svg.image?\forall&space;k\in\mathbb{N}^{*}" title="\forall k\in\mathbb{N}^{*}" />, on a : <img src="https://latex.codecogs.com/svg.image?p_\pi(s\rightarrow&space;s^{\prime\prime},&space;k)&space;=&space;p(s\rightarrow&space;s^\prime,&space;k-1)&space;p(s^\prime&space;\rightarrow&space;s^{\prime\prime},&space;1)" title="p_\pi(s\rightarrow s^{\prime\prime}, k) = p(s\rightarrow s^\prime, k-1) p(s^\prime \rightarrow s^{\prime\prime}, 1)" />.

Grâce à ces expression, on peut dérouler la récursion :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla&space;v_\pi(s)&space;&=&space;\phi(s)&plus;&space;\sum_{a&space;\in&space;A}&space;\pi(a|s)&space;\sum_{s^\prime}p(s^\prime|s,a)&space;\nabla&space;v_\pi(s^\prime)&space;\\&=&space;\phi(s)&space;&plus;&space;p(s\rightarrow&space;s^\prime,1)&space;\nabla&space;v_\pi(s^\prime)&space;\\&=&space;\phi(s)&space;&plus;&space;p(s\rightarrow&space;s^\prime,1)(\phi(s^\prime)&space;&plus;&space;p(s^\prime&space;\rightarrow&space;s^{\prime&space;\prime},1)\nabla&space;v_\pi(s^{\prime&space;\prime}))&space;\\&=&space;\phi(s)&space;&plus;&space;p(s\rightarrow&space;s^\prime,1)\phi(s^\prime)&space;&plus;&space;p(s&space;\rightarrow&space;s^{\prime&space;\prime},2)\nabla&space;v_\pi(s^{\prime&space;\prime})&space;\\&=&space;\phi(s)&space;&plus;&space;p(s\rightarrow&space;s^\prime,1)\phi(s^\prime)&space;&plus;&space;p(s&space;\rightarrow&space;s^{\prime&space;\prime},2)\phi(s^{\prime&space;\prime})&space;&plus;&space;p(s&space;\rightarrow&space;s^{\prime&space;\prime&space;\prime},3)\phi(s^{\prime&space;\prime&space;\prime})&space;&plus;&space;...\\\nabla&space;v_\pi(s)&space;&=&space;\sum_{x&space;\in&space;S}&space;\sum_{k=0}^{\infty}p(s\rightarrow&space;x,&space;k)\phi(s)\end{align*}" title="\begin{align*} \nabla v_\pi(s) &= \phi(s)+ \sum_{a \in A} \pi(a|s) \sum_{s^\prime}p(s^\prime|s,a) \nabla v_\pi(s^\prime) \\&= \phi(s) + p(s\rightarrow s^\prime,1) \nabla v_\pi(s^\prime) \\&= \phi(s) + p(s\rightarrow s^\prime,1)(\phi(s^\prime) + p(s^\prime \rightarrow s^{\prime \prime},1)\nabla v_\pi(s^{\prime \prime})) \\&= \phi(s) + p(s\rightarrow s^\prime,1)\phi(s^\prime) + p(s \rightarrow s^{\prime \prime},2)\nabla v_\pi(s^{\prime \prime}) \\&= \phi(s) + p(s\rightarrow s^\prime,1)\phi(s^\prime) + p(s \rightarrow s^{\prime \prime},2)\phi(s^{\prime \prime}) + p(s \rightarrow s^{\prime \prime \prime},3)\phi(s^{\prime \prime \prime}) + ...\\\nabla v_\pi(s) &= \sum_{x \in S} \sum_{k=0}^{\infty}p(s\rightarrow x, k)\phi(s)\end{align*}" />
</p>


Le théorème du gradient de la politique fait intervenir la distribution stationnaire des états <img src="https://latex.codecogs.com/svg.image?\mu_\pi(s)" title="\mu_\pi(s)" /> - notée <img src="https://latex.codecogs.com/svg.image?\mu(s)" title="\mu(s)" />) - définit de la façon suivante :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\mu(s)&space;\doteq&space;\frac{\eta(s)&space;}{\sum_{s^\prime&space;\in&space;S}\eta(s^\prime)}" title="\mu(s) \doteq \frac{\eta(s) }{\sum_{s^\prime \in S}\eta(s^\prime)}" />
</p>
Où <img src="https://latex.codecogs.com/svg.image?\eta(s)" title="\eta(s)" /> est l'espérence du nombre de visite de s sur un épisode, soit : <img src="https://latex.codecogs.com/svg.image?\eta(s)&space;\doteq&space;\sum_{k=0}^{\infty}p(s_0\to&space;s&space;|k)" title="\eta(s) \doteq \sum_{k=0}^{\infty}p(s_0\to s |k)" />.

Cette dernière forme peut nous permettre d'exprimer les probabilités de transition sous forme d'espérence du nombre de visite, pour faire apparaître la distribution stationnaire. 

Rappelons enfin que la performance correspond à la récompense espérée sur l'épisode en suivant <img src="https://latex.codecogs.com/svg.image?\pi" title="\pi" /> à partir de l'état initial, soit : <img src="https://latex.codecogs.com/svg.image?J(\theta)&space;\doteq&space;v(s_0)" title="J(\theta) \doteq v(s_0)" />. 

Nous avons maintenant tous les éléments pour finir la démonstration :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla&space;J(\theta)&space;&=&space;\nabla&space;v(s_0)\\&=&space;\sum_{x&space;\in&space;S}&space;\sum_{k=0}^{\infty}p(s\rightarrow&space;x,&space;k)\sum_{a&space;\in&space;A}&space;\nabla&space;\pi(a|x)q_\pi(x,a)&space;\\&=&space;\sum_{x&space;\in&space;S}&space;\eta&space;(x)&space;\sum_{a&space;\in&space;A}&space;\nabla&space;\pi(a|x)&space;q_\pi(x,a)&space;\\&=&space;\sum_{x&space;\in&space;S}&space;(\sum_{s^\prime}&space;\eta(s^\prime))\frac{\eta&space;(x)}{\sum_{s^\prime}&space;\eta(s^\prime)}&space;\sum_{a&space;\in&space;A}&space;\nabla&space;\pi(a|x)&space;q_\pi(x,a)&space;\\\nabla&space;J(\theta)&space;&=&space;\sum_{s^\prime}&space;\eta(s^\prime)&space;\sum_{x&space;\in&space;S}&space;\mu(s)&space;\sum_{a&space;\in&space;A}&space;\nabla&space;\pi(a|x)&space;q_\pi(x,a)&space;\\\nabla&space;J(\theta)&space;&\propto&space;\sum_{x&space;\in&space;S}&space;\mu(s)&space;\sum_{a&space;\in&space;A}&space;\nabla&space;\pi(a|x)&space;q_\pi(x,a)\end{align*}" title="\begin{align*} \nabla J(\theta) &= \nabla v(s_0)\\&= \sum_{x \in S} \sum_{k=0}^{\infty}p(s\rightarrow x, k)\sum_{a \in A} \nabla \pi(a|x)q_\pi(x,a) \\&= \sum_{x \in S} \eta (x) \sum_{a \in A} \nabla \pi(a|x) q_\pi(x,a) \\&= \sum_{x \in S} (\sum_{s^\prime} \eta(s^\prime))\frac{\eta (x)}{\sum_{s^\prime} \eta(s^\prime)} \sum_{a \in A} \nabla \pi(a|x) q_\pi(x,a) \\\nabla J(\theta) &= \sum_{s^\prime} \eta(s^\prime) \sum_{x \in S} \mu(s) \sum_{a \in A} \nabla \pi(a|x) q_\pi(x,a) \\\nabla J(\theta) &\propto \sum_{x \in S} \mu(s) \sum_{a \in A} \nabla \pi(a|x) q_\pi(x,a)\end{align*}" />
</p>


 
**Cas continue**

Dans le cas continue, on définit la performance sous la forme de récompense moyenne par pas de temps :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;J(\theta)&space;&\doteq&space;r(\pi)&space;\doteq&space;\displaystyle&space;\lim_{h&space;\to&space;\infty}&space;\sum_{t=1}^{h}\mathop{\mathbb{E}}[R_t|S_0,A_{0&space;:&space;t-1}\sim\pi]&space;\\&&space;=&space;\lim_{h&space;\to&space;\infty}&space;\mathop{\mathbb{E}}[R_t|S_0,A_{0&space;:&space;t-1}\sim\pi]&space;\\&&space;=&space;\sum_s&space;\mu(s)&space;\sum_a&space;\pi(a|s)&space;\sum_{s^\prime,r}&space;p(^\prime,r|s,a)r\end{align*}" title="\begin{align*} J(\theta) &\doteq r(\pi) \doteq \displaystyle \lim_{h \to \infty} \sum_{t=1}^{h}\mathop{\mathbb{E}}[R_t|S_0,A_{0 : t-1}\sim\pi] \\& = \lim_{h \to \infty} \mathop{\mathbb{E}}[R_t|S_0,A_{0 : t-1}\sim\pi] \\& = \sum_s \mu(s) \sum_a \pi(a|s) \sum_{s^\prime,r} p(^\prime,r|s,a)r\end{align*}" />
</p>


En outre, <img src="https://latex.codecogs.com/svg.image?v_{\pi}" title="v_{\pi}" /> et <img src="https://latex.codecogs.com/svg.image?q_{\pi}" title="q_{\pi}" /> sont fonctions du retour différentiel, définit comme la différence entre la récompense obtenue à chaque pas, et la récompense moyenne : 
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?G_t&space;=&space;R_{t&plus;1}&space;-&space;r(\pi)&space;&plus;&space;R_{t&plus;2}&space;-&space;r(\pi)&space;&plus;&space;..." title="G_t = R_{t+1} - r(\pi) + R_{t+2} - r(\pi) + ..." />
</p>

On procède d'une façon analogue au cas épisodique :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla&space;v_\pi(s)&space;&&space;=&space;\nabla&space;\sum_a&space;\pi(a|s)q_\pi(s,a)&space;\\&&space;=&space;\sum_a&space;(\nabla\pi(a|s)&space;q_\pi(s,a)&plus;\pi(a|s)&space;\nabla&space;q_\pi(a,s)&space;)&space;\\&&space;=&space;\sum_a&space;(\nabla\pi(a|s)&space;q_\pi(s,a)&plus;\pi(a|s)&space;\nabla\sum_{s^\prime,r}&space;p(s^\prime,r|s,a)(r-r(\theta)&plus;v_\pi(s)))\\&=&space;\sum_a&space;(\nabla\pi(a|s)&space;q_\pi(s,a)&plus;\pi(a|s)&space;\sum_{s^\prime,r}&space;p(s^\prime,r|s,a)(-\nabla&space;r(\theta)&plus;\nabla&space;v_\pi(s^\prime)))&space;\\&=&space;\sum_a&space;(\nabla\pi(a|s)&space;q_\pi(s,a)&plus;\pi(a|s)&space;\sum_{s^\prime}&space;p(s^\prime|s,a)(-\nabla&space;r(\theta)&plus;\nabla&space;v_\pi(s^\prime)))&space;\\&&space;=&space;\sum_a&space;(\nabla\pi(a|s)&space;q_\pi(s,a)&space;-&space;\pi(a|s)\nabla&space;r(\theta)&space;&plus;&space;\pi(a|s)&space;\sum_{s^\prime}&space;p(s^\prime|s,a)\nabla&space;v_\pi(s^\prime))&space;\end{align*}" title="\begin{align*} \nabla v_\pi(s) & = \nabla \sum_a \pi(a|s)q_\pi(s,a) \\& = \sum_a (\nabla\pi(a|s) q_\pi(s,a)+\pi(a|s) \nabla q_\pi(a,s) ) \\& = \sum_a (\nabla\pi(a|s) q_\pi(s,a)+\pi(a|s) \nabla\sum_{s^\prime,r} p(s^\prime,r|s,a)(r-r(\theta)+v_\pi(s)))\\&= \sum_a (\nabla\pi(a|s) q_\pi(s,a)+\pi(a|s) \sum_{s^\prime,r} p(s^\prime,r|s,a)(-\nabla r(\theta)+\nabla v_\pi(s^\prime))) \\&= \sum_a (\nabla\pi(a|s) q_\pi(s,a)+\pi(a|s) \sum_{s^\prime} p(s^\prime|s,a)(-\nabla r(\theta)+\nabla v_\pi(s^\prime))) \\& = \sum_a (\nabla\pi(a|s) q_\pi(s,a) - \pi(a|s)\nabla r(\theta) + \pi(a|s) \sum_{s^\prime} p(s^\prime|s,a)\nabla v_\pi(s^\prime)) \end{align*}" />
</p>

Ce qui nous permet d'isoler <img src="https://latex.codecogs.com/svg.image?\nabla&space;r(\theta)&space;" title="\nabla r(\theta) " /> :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\nabla&space;r(\theta)&space;=&space;\sum_a&space;\nabla\pi(a|s)q_\pi(s,a)&space;&plus;&space;\sum_a&space;\pi(a|s)\sum_{s^\prime}p(s^\prime|s,a)\nabla&space;v_\pi(s^\prime)-\nabla&space;v_\pi(s)" title="\nabla r(\theta) = \sum_a \nabla\pi(a|s)q_\pi(s,a) + \sum_a \pi(a|s)\sum_{s^\prime}p(s^\prime|s,a)\nabla v_\pi(s^\prime)-\nabla v_\pi(s)" />
</p>

Par définition, <img src="https://latex.codecogs.com/svg.image?J(\theta)=r(\theta)" title="J(\theta)=r(\theta)" />. Or <img src="https://latex.codecogs.com/svg.image?r(\theta)" title="r(\theta)" /> est indépendant de s. L'équation du gradient de la performance est donc toujours juste si l'on multiplie le terme de droite par <img src="https://latex.codecogs.com/svg.image?\sum_s&space;\mu(s)" title="\sum_s \mu(s)" />, puisque <img src="https://latex.codecogs.com/svg.image?\sum_s&space;\mu(s)&space;=&space;1" title="\sum_s \mu(s) = 1" />.

Ainsi :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla&space;J(\theta)&space;&=&space;\sum_s&space;\mu(s)&space;(\sum_a&space;(\nabla\pi(a|s)q_\pi(s,a)&space;&plus;&space;\pi(a|s)\sum_{s^\prime}p(s^\prime|s,a)\nabla&space;v_\pi(s^\prime))-\nabla&space;v_\pi(s))&space;\\&=&space;\sum_s&space;\mu(s)&space;\sum_a&space;\nabla\pi(a|s)q_\pi(s,a)&space;&plus;&space;\sum_{s^\prime}&space;\sum_{s}&space;\mu(s)&space;\sum_a&space;\pi(a|s)p(s^\prime|s,a)\nabla&space;v_\pi(s^\prime)&space;-&space;\sum_s&space;\mu(s)&space;\nabla&space;v_\pi(s)&space;\\&=&space;\sum_s&space;\mu(s)&space;\sum_a&space;\nabla\pi(a|s)q_\pi(s,a)&space;&plus;&space;\sum_{s^\prime}&space;\mu(s^\prime)\nabla&space;v_\pi(s^\prime)&space;-&space;\sum_s&space;\mu(s)&space;\nabla&space;v_\pi(s)&space;\\\nabla&space;J(\theta)&space;&=&space;\sum_s&space;\mu(s)&space;\sum_a&space;\nabla\pi(a|s)q_\pi(s,a)\end{align*}" title="	" />
</p>

On retrouve la même forme que dans le cas épisodique. 



**En résumé**

Dans les deux cas, on a donc :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\nabla&space;J(\theta)&space;\propto&space;\sum_{s\in&space;S}&space;\mu(s)&space;\sum_{a&space;\in&space;A}&space;q_\pi(s,a)&space;\nabla_\theta&space;\pi_\theta(a|s)" title="\nabla J(\theta) \propto \sum_{s\in S} \mu(s) \sum_{a \in A} q_\pi(s,a) \nabla_\theta \pi_\theta(a|s)" />
</p>

Avec pour coefficient de proportionnalité:
- <img src="https://latex.codecogs.com/svg.image?\sum_{s}&space;\eta(s)" title="\sum_{s} \eta(s)" /> dans le cas épisodique ;
- 1 dans le cas continue.



<br/>

### Généralisation aux algorithmes du gradient de la politique

Le théorème du gradient de la politique nous permet d'exprimer le gradient de la performance d'une manière simple et élégante :

<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla&space;J(\theta)&space;&\propto&space;\sum_{s&space;\in&space;S}&space;\mu(s)&space;\sum_{a&space;\in&space;A}&space;q_\pi(s,a)&space;\nabla&space;\pi(a|s)&space;\\&&space;=&space;\sum_{s&space;\in&space;S}&space;\mu(s)&space;\sum_{a&space;\in&space;A}&space;\pi(a|s)&space;q_\pi(s,a)&space;\frac{\nabla&space;\pi(a|s)}{\pi(a|s)}&space;&space;\\\nabla&space;J&space;(\theta)&space;&=&space;\mathop{\mathbb{E}}_{s\sim&space;\mu_\pi,&space;a\sim&space;\pi_0}&space;[q_\pi(s,a)&space;\nabla&space;\ln\pi(a|s)]\\\end{align*}" title="\begin{align*} \nabla J(\theta) &\propto \sum_{s \in S} \mu(s) \sum_{a \in A} q_\pi(s,a) \nabla \pi(a|s) \\& = \sum_{s \in S} \mu(s) \sum_{a \in A} \pi(a|s) q_\pi(s,a) \frac{\nabla \pi(a|s)}{\pi(a|s)} \\\nabla J (\theta) &= \mathop{\mathbb{E}}_{s\sim \mu_\pi, a\sim \pi_0} [q_\pi(s,a) \nabla \ln\pi(a|s)]\\\end{align*}" />
</p>

Cette forme constitue les fondements de la plupart des algorithmes du gradient de la politique. Elle a pour particularité de ne **pas** avoir **de biais**, mais d'être soumis à une **forte variance**. Les algorithmes évoqués dans cet article proposent des solutions pour réduire la variance sans (trop) affecter le bias.

L'article Estimation de l'Avantage Généralisée (GAE) [Schulman et al., 2016](https://arxiv.org/pdf/1506.02438.pdf) ((i) trad ?) propose une forme générale du gradient de la performance, mettant en lumière les différentes déclinaisons de cette forme que l'on peut trouver dans la littérature :

En posant g, le gradient de la performance, tel que <img src="https://latex.codecogs.com/svg.image?g&space;\doteq&space;\nabla_\theta&space;\mathop{\mathbb{E}}[\sum_{t=0}^\infty&space;r_t]" title="g \doteq \nabla_\theta \mathop{\mathbb{E}}[\sum_{t=0}^\infty r_t]" />

On a la forme générale : 
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?g&space;=&space;\mathop{\mathbb{E}}[\sum_{t=0}^\infty&space;\Psi_t&space;\nabla_\theta&space;log&space;\pi_\theta(a_t|s_t)]" title="g = \mathop{\mathbb{E}}[\sum_{t=0}^\infty \Psi_t \nabla_\theta log \pi_\theta(a_t|s_t)]" />
</p>

Avec <img src="https://latex.codecogs.com/svg.image?\Psi_t" title="\Psi_t" />, l'une des fonctions suivantes :
- <img src="https://latex.codecogs.com/svg.image?\sum_{t=0}^\infty&space;r_{t}" title="\sum_{t=0}^\infty r_{t}" /> : retour total de la trajectoire
- <img src="https://latex.codecogs.com/svg.image?\sum_{t^\prime=t}^\infty&space;r_{t^\prime}" title="\sum_{t^\prime=t}^\infty r_{t^\prime}" /> : retour suivant l'action <img src="https://latex.codecogs.com/svg.image?a_t" title="a_t" />
- <img src="https://latex.codecogs.com/svg.image?\sum_{t^\prime=t}^\infty&space;r_{t^\prime}-b(s_t)" title="\sum_{t^\prime=t}^\infty r_{t^\prime}-b(s_t)" /> : formule précédente, avec valeurs de références
- <img src="https://latex.codecogs.com/svg.image?q_\pi(s_t,a_t)" title="q_\pi(s_t,a_t)" /> : fonction de valeur d'état-action
- <img src="https://latex.codecogs.com/svg.image?A_\pi(s_t,a_t)" title="A_\pi(s_t,a_t)" /> : fonction d'avantage
- <img src="https://latex.codecogs.com/svg.image?r_t&space;&plus;&space;v_\pi(s_{t&plus;1})-v_\pi(s_t)" title="r_t + v_\pi(s_{t+1})-v_\pi(s_t)" /> : résidu TD (différence temporelle)


Avec les fonctions de valeurs : 
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?v_\pi&space;(s_t)&space;\doteq&space;\mathop{\mathbb{E}}_{s_{t&plus;1}:\infty,a_t:\infty}[\sum_{l=0}^\infty&space;r_{t&plus;l}]" title="v_\pi (s_t) \doteq \mathop{\mathbb{E}}_{s_{t+1}:\infty,a_t:\infty}[\sum_{l=0}^\infty r_{t+l}]" />
	<img src="https://latex.codecogs.com/svg.image?q_\pi&space;(s_t,a_t)&space;\doteq&space;\mathop{\mathbb{E}}_{s_{t&plus;1}:\infty,a_{t&plus;1}:\infty}[\sum_{l=0}^\infty&space;r_{t&plus;l}]" title="q_\pi (s_t,a_t) \doteq \mathop{\mathbb{E}}_{s_{t+1}:\infty,a_{t+1}:\infty}[\sum_{l=0}^\infty r_{t+l}]" />
</p>

Et la fonction d'avantage : 
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?A_\pi(s_t,a_t)&space;\doteq&space;q_\pi(s_t,a_t)-v_\pi(s_t)" title="A_\pi(s_t,a_t) \doteq q_\pi(s_t,a_t)-v_\pi(s_t)" />
</p>

<br/>

## Algorithmes du Gradient de la Politique

Les algorithmes présentés font tous mention du paramètre <img src="https://latex.codecogs.com/svg.image?\gamma&space;\in&space;&space;\]0;1\]" title="\gamma \in \]0;1\]" />, le facteur d'atténuation. Sa définition est implicite, afin d'éviter les redondances.

<br/>

### REINFORCE

L'algorithme **REINFORCE** (gradient de la Politique avec méthode Monte-Carlo) repose sur l'expression du gradient de la performance obtenue dans le Théorème du Gradient de la Politique, appliqué aux épisodes prélevés (i.e. obtenus par interaction directe avec l'environnement). En constatant que <img src="https://latex.codecogs.com/svg.image?q_\pi(s_t,a_t)=&space;\mathop{\mathbb{E}}_\pi[G_t&space;|s_t,&space;a_t]" title="q_\pi(s_t,a_t)= \mathop{\mathbb{E}}_\pi[G_t |s_t, a_t]" />, on trouve :


<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla_\theta&space;J(\theta)&space;&=&space;\mathop{\mathbb{E}}_\pi[q_\pi(s,a)&space;\nabla_\theta\ln\pi_\theta(a,s)]&space;\\&=&space;\mathop{\mathbb{E}}_\pi[G_t&space;\nabla_\theta\ln\pi_\theta(a,s)]\end{align*}" title="\begin{align*} \nabla_\theta J(\theta) &= \mathop{\mathbb{E}}_\pi[q_\pi(s,a) \nabla_\theta\ln\pi_\theta(a,s)] \\&= \mathop{\mathbb{E}}_\pi[G_t \nabla_\theta\ln\pi_\theta(a,s)]\end{align*}" />
</p>

Autrement dit, on peut optimiser <img src="https://latex.codecogs.com/svg.image?\theta" title="\theta" /> à partir du retour obtenu au cours d'un épisode. Cette approche exploite la trajectoire observée sur l'épisode entier pour faire ses mises à jours, c'est pourquoi on parle de méthode de type Monte Carlo.

---

**Algorithme : REINFORCE (épisodique)**

<ins>Initialisation :</ins>
- Définir <img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" />, le pas d'apprentissage associé à la politique
- Initialiser aléatoirement <img src="https://latex.codecogs.com/svg.image?\theta&space;\in&space;\mathbb{R}^{d}" title="\theta \in \mathbb{R}^{d}" />, les poids associés aux caractéristiques définissant la politique

<ins>Exécution :</ins>
- Pour chaque épisode :
	- Générer la trajectoire <img src="https://latex.codecogs.com/svg.image?S_1,A_1,R_2,S_2,A_2,&space;...&space;,&space;A_{T-1},&space;S_T" title="S_1,A_1,R_2,S_2,A_2, ... , A_{T-1}, S_T" /> en suivant <img src="https://latex.codecogs.com/svg.image?\pi_\theta" title="\pi_\theta" />
	- Pour chaque étape de l'épisode <img src="https://latex.codecogs.com/svg.image?t=0,1,...,T-1,T" title="t=0,1,...,T-1,T" /> :
		- <img src="https://latex.codecogs.com/svg.image?G\leftarrow&space;\sum_{k=t&plus;1}^T&space;\gamma^{k-t-1}R_k" title="G\leftarrow \sum_{k=t+1}^T \gamma^{k-t-1}R_k" />
		- <img src="https://latex.codecogs.com/svg.image?\theta&space;\leftarrow&space;\theta&space;&plus;&space;\alpha&space;\gamma^t&space;\nabla\ln\pi(A_t|S_t,\theta)" title="\theta \leftarrow \theta + \alpha \gamma^t \nabla\ln\pi(A_t|S_t,\theta)" />

---

<br/>

Avec Reinforce, <img src="https://latex.codecogs.com/svg.image?\theta" title="\theta" /> est mis-à-jour en utilisant directement le retour observé lors d'une interaction avec l'environnement. Il n'y a donc **pas de biais** : la montée de gradient fera toujours évoluer <img src="https://latex.codecogs.com/svg.image?\theta" title="\theta" /> vers des valeurs qui augmenteront l'espérence de retour. En revanche, cette méthode introduit une **forte variance** : de trop grands pas sont réalisés en fonction de l'échantillon de trajectoire considéré, rendant la convergence vers une configuration optimale plus difficile.



### REINFORCE avec valeurs de référence

L'algorithme **REINFORCE avec valeurs de référence** est variante bien connue de l'algorithme REINFORCE. Il s'agit simplement de retrancher au retour de l'épisode une valeur de référence dans le calcul du gradient de la performance. Cette modification a pour effet de **réduire la variance** tout en assurant l'absence de biais.

On utilise souvent la **valeur d'état** en guise de valeur de référence, de sorte que l'on utilise la **fonction d'avantage** dans la mise à jour du gradient.

Il s'avère qu'il est possible de montrer que le Théorème du Gradient de la Politique peut être étendu à la forme suivante :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\nabla&space;J(\theta)&space;\propto&space;\sum_s&space;\mu(s)&space;\sum_a&space;(q_\pi(s,a)&space;-&space;b(s))\nabla&space;\pi(a|s,\theta)" title="\nabla J(\theta) \propto \sum_s \mu(s) \sum_a (q_\pi(s,a) - b(s))\nabla \pi(a|s,\theta)" />
</p>

Où <img src="https://latex.codecogs.com/svg.image?b(s)" title="b(s)" /> est une fonction quelconque qui **ne dépend pas des actions** prises par l'agent.

#### Preuve

L'expression du théorème avec la fonction <img src="https://latex.codecogs.com/svg.image?b(s)" title="b(s)" /> peut être reformulée de la façon suivante :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\begin{align*}&space;\nabla&space;J(\theta)&space;&\propto&space;\sum_s&space;\mu(s)&space;\sum_a&space;(q_\pi(s,a)&space;-&space;b(s))\nabla&space;\pi(a|s,\theta)&space;\\&&space;=&space;\sum_s&space;\mu(s)&space;(\sum_a&space;q_\pi(s,a)\nabla&space;\pi(a|s,\theta)&space;-&space;\sum_a&space;b(s)\nabla&space;\pi(a|s,\theta))\end{align*}" title="\begin{align*} \nabla J(\theta) &\propto \sum_s \mu(s) \sum_a (q_\pi(s,a) - b(s))\nabla \pi(a|s,\theta) \\& = \sum_s \mu(s) (\sum_a q_\pi(s,a)\nabla \pi(a|s,\theta) - \sum_a b(s)\nabla \pi(a|s,\theta))\end{align*}" />
</p>

Or :
<p align="center">
	<img src="https://latex.codecogs.com/svg.image?\sum_a&space;b(s)\nabla&space;\pi(a|s,\theta)&space;=&space;b(s)&space;\nabla&space;\sum_a&space;\pi(a|s,\theta)=&space;b(s)&space;\nabla&space;1&space;=&space;0&space;" title="\sum_a b(s)\nabla \pi(a|s,\theta) = b(s) \nabla \sum_a \pi(a|s,\theta)= b(s) \nabla 1 = 0 " />
</p>

Insérer <img src="https://latex.codecogs.com/svg.image?b(s)" title="b(s)" /> dans l'expression ne l'invalide donc pas, puisque cette opération revient à une soustraction par zéro.


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
	- <img src="https://latex.codecogs.com/svg.image?\alpha^w&space;\in&space;\mathbb{R}^{&plus;*}" title="\alpha^w \in \mathbb{R}^{+*}" />, le pas d'apprentissage associé aux valeurs d'états
	- <img src="https://latex.codecogs.com/svg.image?\alpha^\theta&space;\in&space;\mathbb{R}^{&plus;*}" title="\alpha^\theta \in \mathbb{R}^{+*}" />, le pas d'apprentissage associé à la politique
- Initialiser aléatoirement :
	- <img src="https://latex.codecogs.com/svg.image?w&space;\in&space;\mathbb{R}^{d^\prime}" title="w \in \mathbb{R}^{d^\prime}" />, les poids associés aux valeurs d'états
	- <img src="https://latex.codecogs.com/svg.image?\theta&space;\in&space;\mathbb{R}^{d}" title="\theta \in \mathbb{R}^{d}" />, les poids associés aux caractéristiques définissant la politique
	
<ins>Exécution :</ins>
- Pour chaque épisode :
	- Générer la trajectoire <img src="https://latex.codecogs.com/svg.image?S_1,A_1,R_2,S_2,A_2,&space;...&space;,&space;A_{T-1},&space;S_T" title="S_1,A_1,R_2,S_2,A_2, ... , A_{T-1}, S_T" /> en suivant <img src="https://latex.codecogs.com/svg.image?\pi_\theta" title="\pi_\theta" />
	- Pour chaque étape de l'épisode <img src="https://latex.codecogs.com/svg.image?t=0,1,...,T-1,T" title="t=0,1,...,T-1,T" /> :
		- <img src="https://latex.codecogs.com/svg.image?G\leftarrow&space;\sum_{k=t&plus;1}^T&space;\gamma^{k-t-1}R_k" title="G\leftarrow \sum_{k=t+1}^T \gamma^{k-t-1}R_k" />
		- <img src="https://latex.codecogs.com/svg.image?\delta&space;\leftarrow&space;G&space;-&space;\hat{v}(S_t,w)" title="\delta \leftarrow G - \hat{v}(S_t,w)" />
		- <img src="https://latex.codecogs.com/svg.image?w&space;\leftarrow&space;w&space;&plus;&space;\alpha^w&space;\delta&space;\nabla&space;\hat{v}(S_t,w)" title="w \leftarrow w + \alpha^w \delta \nabla \hat{v}(S_t,w)" />
		- <img src="https://latex.codecogs.com/svg.image?\theta&space;\leftarrow&space;\theta&space;&plus;&space;\alpha^\theta&space;\gamma^t&space;\delta&space;\nabla\ln\pi(A_t|S_t,\theta)" title="\theta \leftarrow \theta + \alpha^\theta \gamma^t \delta \nabla\ln\pi(A_t|S_t,\theta)" />

---



<br/>

### Acteur-Critique

L'algorithme **Acteur-Critique** ressemble beaucoup à l'algorithme REINFORCE avec valeurs de référence : il s'agit de calculer une différence entre un retour, et une valeur de référence. En revanche, deux différences importantes les distinguent.

Comme toutes les méthodes de type Monte-Carlo, REINFORCE ne fait pas de mise-à-jour avant la fin de l'épisode. Par ailleurs, la valeur de référence ne tient compte que de la valeur de l'état initial (avant de prendre l'action), et ne permet par conséquent pas de juger de la qualité de l'action choisie. Par cette approche, on répond à la question : **"L'agent a-t-il bien fait de se trouver à cette position au temps t ?"**, en tenant compte de l'épisode entier.

Dans la méthode Acteur-Critique, le retour utilisé dans la mise-à-jour des paramètres de la politique calcule la différence entre la valeur de l'état au temps t, et celle de l'état au temps t+1 (en tenant compte du facteur d'atténuation) ; c'est le retour 1-pas, noté <img src="https://latex.codecogs.com/svg.image?G_{t:t&plus;1}" title="G_{t:t+1}" /> (comme dans les méthodes TD(0), SARSA(0) ou Q-apprentissage). Cette approche permet donc d'évaluer la différence de valeur entre l'état initial et le nouvel état, autrement dit de juger de la qualité de l'action prise par l'agent. On répond ici à la question : **"L'agent a-t-il bien fait de choisir cette action au temps t?"**, en ne tenant compte que de la transition entre les temps t et t+1.

En résumé : la politique agit, et le retour 1-pas critique.

---

**Algorithme : Acteur-critique (épisodique)**

<ins>Initialisation :</ins>
- Définir :
	- <img src="https://latex.codecogs.com/svg.image?\alpha^w&space;\in&space;\mathbb{R}^{&plus;*}" title="\alpha^w \in \mathbb{R}^{+*}" />, le pas d'apprentissage associé aux valeurs d'états
	- <img src="https://latex.codecogs.com/svg.image?\alpha^\theta&space;\in&space;\mathbb{R}^{&plus;*}" title="\alpha^\theta \in \mathbb{R}^{+*}" />, le pas d'apprentissage associé à la politique
- Initialiser aléatoirement :
	- <img src="https://latex.codecogs.com/svg.image?w&space;\in&space;\mathbb{R}^{d^\prime}" title="w \in \mathbb{R}^{d^\prime}" />, les poids associés aux valeurs d'états
	- <img src="https://latex.codecogs.com/svg.image?\theta&space;\in&space;\mathbb{R}^{d}" title="\theta \in \mathbb{R}^{d}" />, les poids associés aux caractéristiques définissant la politique
	
<ins>Exécution :</ins>
- Pour chaque épisode :
	- Initialiser <img src="https://latex.codecogs.com/svg.image?s" title="s" /> (premier état de l'épisode)
	- <img src="https://latex.codecogs.com/svg.image?I&space;\leftarrow&space;1" title="I \leftarrow 1" /> (coefficient de réduction cumulée)
	- Tant que <img src="https://latex.codecogs.com/svg.image?s" title="s" /> n'est pas terminal (pour chaque pas de temps) :
		- <img src="https://latex.codecogs.com/svg.image?a&space;\sim&space;\pi(\cdot|s,\theta)" title="a \sim \pi(\cdot|s,\theta)" />
		- Appliquer l'action <img src="https://latex.codecogs.com/svg.image?a" title="a" />, observer <img src="https://latex.codecogs.com/svg.image?(s^\prime,r)" title="(s^\prime,r)" />
		- <img src="https://latex.codecogs.com/svg.image?\delta&space;\leftarrow&space;r&space;&plus;&space;\gamma&space;\hat{v}(s^\prime,w)&space;-&space;\hat{v}(s,w)" title="\delta \leftarrow r + \gamma \hat{v}(s^\prime,w) - \hat{v}(s,w)" />
		- <img src="https://latex.codecogs.com/svg.image?w&space;\leftarrow&space;w&space;&plus;&space;\alpha^w&space;\delta&space;\nabla&space;\hat{v}(s,w)" title="w \leftarrow w + \alpha^w \delta \nabla \hat{v}(s,w)" />
		- <img src="https://latex.codecogs.com/svg.image?\theta&space;\leftarrow&space;\theta&space;&plus;&space;\alpha^\theta&space;I&space;\delta&space;\nabla\ln\pi(a|s,\theta)" title="\theta \leftarrow \theta + \alpha^\theta I \delta \nabla\ln\pi(a|s,\theta)" />
		- <img src="https://latex.codecogs.com/svg.image?I&space;\leftarrow&space;\gamma&space;I" title="I \leftarrow \gamma I" />
		- <img src="https://latex.codecogs.com/svg.image?s&space;\leftarrow&space;s^\prime" title="s \leftarrow s^\prime" />


Par convention, on a <img src="https://latex.codecogs.com/svg.image?\hat{v}(s^\prime,w)&space;\doteq&space;0" title="\hat{v}(s^\prime,w) \doteq 0" /> si <img src="https://latex.codecogs.com/svg.image?s^\prime" title="s^\prime" /> est terminal. 

Cette version ne tient compte que d'une transition pour réaliser les mises-à-jours de poids, ce qui a toutes les chances de rendre l'optimisation difficile en raison de la large variance dans les données d'entraînement d'une itération à l'autre. Il existe des approches plus sophistiquées qui s'appuient sur la parallélisation pour rendre cet algorithme plus stable (voir [A3C](https://arxiv.org/pdf/1602.01783.pdf)).

En pratique, on peut simplement étendre cet algorithme aux itérations sur des lots, en accumulant un nombre <img src="https://latex.codecogs.com/svg.image?n" title="n" /> de transitions, et en appliquant les mêmes étapes mentionnées ci-dessus. À noter qu'il n'y aucune nécessité de lien entre les transitions (i.e. elles n'ont pas à provenir d'une même trajectoire) ; tant que l'on a des quadruplets <img src="https://latex.codecogs.com/svg.image?(s,a,s^\prime,r)" title="(s,a,s^\prime,r)" />, nous serons en mesure d'appliquer l'algorithme.


	

---


<br/>


### Acteur-Critique Hors-Politique


Tous les algorithmes jusqu'ici présentés optimisent la politique qui a été utilisée pour obtenir des trajectoires. Dans cette section, nous abordons une variante de l'algorithme Acteur-Critique dans laquelle **la politique d'exploration n'est pas la même que la politique cible**. 

Jetons à nouveau un coup d'oeil à l'estimateur utilisé par les algorithmes de gradient de la politique :

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\nabla_\theta&space;J(\theta)&space;=&space;\mathop{\mathbb{E}}_{\tau&space;\sim&space;\pi_\theta(\tau)}[\nabla_\theta&space;\ln&space;\pi_\theta(\tau)r(\tau)]" title="\nabla_\theta J(\theta) = \mathop{\mathbb{E}}_{\tau \sim \pi_\theta(\tau)}[\nabla_\theta \ln \pi_\theta(\tau)r(\tau)]" />
</p>

Le gradient de la performance est calculé à partir de l'espérence sur <img src="https://latex.codecogs.com/svg.image?\tau&space;\sim&space;\pi_\theta(\tau)" title="\tau \sim \pi_\theta(\tau)" />. Une trajectoire relevée à partir d'un certain <img src="https://latex.codecogs.com/svg.image?\theta" title="\theta" /> n'est donc plus valable après application d'une itération de l'algorithme de monté de gradient sur <img src="https://latex.codecogs.com/svg.image?\theta" title="\theta" />. Autrement dit, il est **nécessaire de prélever une trajectoire après chaque mise-à-jours de** <img src="https://latex.codecogs.com/svg.image?\theta" title="\theta" />, rendant caduques toutes les mesures précédentes. Le problème est de taille, sitôt que l'on se trouve sur des tâches pour lesquelles les mesures sont lentes et coûteuses (comme en robotique, par exemple).

Pour cette raison, il est souhaitable de définir des algorithmes permettant d'optimiser une politique à partir de mesures réalisées *Hors-Politique*, c'est-à-dire avec une autre politique que celle que l'on optimise.

Par ailleurs, une telle approche permet d'utiliser une politique la plus efficace pour l'exploration, sans avoir à contraindre la politique que l'on est en train d'optimiser pour qu'inciter à explorer de nouvelles trajectoires.


---
**Algorithme : Acteur-critique Hors-Politique (épisodique)**

<ins>Initialisation :</ins>
- Définir :
	- <img src="https://latex.codecogs.com/svg.image?e_v&space;\leftarrow&space;0" title="e_v \leftarrow 0" />, (trace d'éligibilité sur les valeurs ?)
	- <img src="https://latex.codecogs.com/svg.image?e_u&space;\leftarrow&space;0" title="e_u \leftarrow 0" />, (trace d'éligibilité sur la politique ?)
	- <img src="https://latex.codecogs.com/svg.image?w&space;\leftarrow&space;0" title="w \leftarrow 0" />, (? utilisation de w (i) ?)
	- <img src="https://latex.codecogs.com/svg.image?S&space;\leftarrow&space;S_0" title="S \leftarrow S_0" />, état initial
- Initialiser aléatoirement :
	- v, les poids associés aux valeurs d'états
	- u, les poids associés aux caractéristiques définissant la politique
	
<ins>Exécution :</ins>
- Pour chaque étape :
	- <img src="https://latex.codecogs.com/svg.image?a&space;\sim&space;b(\cdot|s)" title="a \sim b(\cdot|s)" />
	- Appliquer l'action a, observer (s',r)
	- <img src="https://latex.codecogs.com/svg.image?\delta&space;\leftarrow&space;r&space;&plus;&space;\gamma(s^\prime)&space;v^Tx_{s^\prime}&space;-&space;v^Tx_s" title="\delta \leftarrow r + \gamma(s^\prime) v^Tx_{s^\prime} - v^Tx_s" />
	- <img src="https://latex.codecogs.com/svg.image?\rho&space;\leftarrow&space;\frac{\pi_u(a|s)}{b(a|s)}" title="\rho \leftarrow \frac{\pi_u(a|s)}{b(a|s)}" />
	- Mettre à jour le critique :
		- <img src="https://latex.codecogs.com/svg.image?e_v&space;\leftarrow&space;\rho(x_s&plus;\gamma(s)\lambda&space;e_v)" title="e_v \leftarrow \rho(x_s+\gamma(s)\lambda e_v)" />
		- <img src="https://latex.codecogs.com/svg.image?v&space;\leftarrow&space;v&space;&plus;&space;\alpha_v[\delta&space;e_v&space;-&space;\gamma(s^\prime)(1-\lambda)(w^Te_v)x_s]" title="v \leftarrow v + \alpha_v[\delta e_v - \gamma(s^\prime)(1-\lambda)(w^Te_v)x_s]" />
		- <img src="https://latex.codecogs.com/svg.image?w&space;\leftarrow&space;w&space;&plus;&space;\alpha_w[\delta&space;e_v&space;-&space;(w^Tx_s)x_s]" title="w \leftarrow w + \alpha_w[\delta e_v - (w^Tx_s)x_s]" />
	- Mettre à jour l'acteur :
		- <img src="https://latex.codecogs.com/svg.image?e_u&space;\leftarrow&space;\rho&space;[\frac{\nabla_u\pi_u(a|s)}{\pi_u(a|s)}&plus;\gamma(s)\lambda&space;e_u]" title="e_u \leftarrow \rho [\frac{\nabla_u\pi_u(a|s)}{\pi_u(a|s)}+\gamma(s)\lambda e_u]" />
		- <img src="https://latex.codecogs.com/svg.image?u&space;\leftarrow&space;u&space;&plus;&space;\alpha_u&space;\delta&space;e_u" title="u \leftarrow u + \alpha_u \delta e_u" />
	
	- <img src="https://latex.codecogs.com/svg.image?s&space;\leftarrow&space;s^\prime" title="s \leftarrow s^\prime" />
	

Avec <img src="https://latex.codecogs.com/svg.image?x_s" title="x_s" />, le vecteur de caractéristique correspondant à l'état observé <img src="https://latex.codecogs.com/svg.image?s" title="s" />.

(à faire 3 : corriger l'algo à partir de l'article)

---


<ins>En rédaction :</ins> TRPO, PPO





#### Sources

**Cours et articles blogs :**

[Article très complet de Lilian Weng sur les méthodes de gradient de la politique](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#off-policy-policy-gradient) 

[CS182 à UC Berkeley, notes de cours de Serguey Levine, en particulier le cours magistral n°15](https://cs182sp21.github.io/). L'ensemble des sources utiliées sont listés à la fin de l'article.

[Article de Daniel Seita sur les méthodes de gradient de la politique, élaborant des pistes d'intuitions à propos de REINFORCE avec valeurs de références](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/)

["Introduction à l'apprentissage par renforcement", 2e édition, l'ouvrage de référence de Sutton et Barto](http://incompleteideas.net/book/the-book-2nd.html)



**Articles originaux des méthodes :**

[Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4), 229-256.](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf)

[Degris, T., White, M., & Sutton, R. S. (2012). Off-policy actor-critic. arXiv preprint arXiv:1205.4839.](https://arxiv.org/pdf/1205.4839.pdf)





[1] [Ioffe, S., & Szegedy, C. (2015, June). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning (pp. 448-456). PMLR.](https://arxiv.org/abs/1502.03167) 

[2] [Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). How does batch normalization help optimization?. arXiv preprint arXiv:1805.11604.](https://arxiv.org/pdf/1805.11604.pdf)

[3] [Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., … & Rabinovich, A. (2015). Going deeper with convolutions, Proceedings of the IEEE conference on computer vision and pattern recognition](https://arxiv.org/abs/1409.4842) 

[4] [He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition](https://arxiv.org/abs/1512.03385)

[5] [Tan, M., & Le, Q. V. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks, arXiv preprint arXiv:1905.11946.](https://arxiv.org/abs/1905.11946)

[6] [Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. Bengio, Y. (2014), Generative adversarial nets, Advances in neural information processing systems](https://proceedings.neurips.cc/paper/2014/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html)

<br/>

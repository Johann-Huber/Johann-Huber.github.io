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






## En 30 secondes


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




Article de référence : [“Batch-normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift”](url=https://arxiv.org/abs/1502.03167) (trad. “Normalisation par Lots : Accélération de l’entraînement des réseaux de neurones profonds par la réduction du décalage de covariable interne”).

Article (contribution significative dans la compréhension du concept) : [“How does batch normalization help optimization”](url=https://arxiv.org/pdf/1805.11604.pdf) (trad. “Comment la normalisation par lots aide l’optimisation.”).







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

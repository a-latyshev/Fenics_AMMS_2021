# README pour TP MES01-CR1

Cette archive contient quatre fichiers *.py :
 - EshelbyInhomogeneity.py
 - EshelbyStudies.py
 - EshelbyUser.py
 - EshelbyEllipse.py

## EshelbyInhomogeneity.py

Ce fichier contient la fonctionnalité basique du project. Il comporte :
 * classe `materio` - l'information sur les propriétés de matériaux
 * classe `model`, qui contient les paramétrés de la taille du domaine et du maillage et le degrés de liberté de l'espace de la solution. Elle génère le maillage à partir de la fonction `mesh_generator`
 * classe `solution`, qui construit la formulation variationnelle dans la fonction `variational_problem`, résout le problème et fait des calculs dans la fonction  `solution_maker` et calcule une erreur exacte dans `behind_solution`

Il y a un paramètre pour initializer la classe  `solution` - `mod`. Quand on veut trouver la solution pour le cas incompressible, il faut met son valeur égale à `"incompressible"`.

## EshelbyStudies.py

Ce fichier contient les fonctions nécessaires pour résoudre la deuxième partie du TP. Chaque fonction fait des calculs, plot des résultats et les sauvegarde sous la forme d'image.
 * fonction `element_order_analysis` fait l'analyse de convergence de la solution numérique ver la solution théorique pour les normes - L2 et H1 et calcule le taux de convergence
 * fonction `nu_analysis` fait l'analyse de convergence de la solution en changeant la valeur du coefficient de Poisson de l'inclusion 
 * fonction `matrix_expansion` fait l'analyse de l'influence du rayon externe 'R_out' sur les valeurs de déformation à l'intérieur de l'inclusion
 * fonction `material_analysis` fait l'analyse de convergence pour différents rapports entre les modules de young de la matrice et de l'inclusion

## EshelbyUser.py

Ce fichier est exécutable et génère des figures présentées dans le rapport. Ci-dessous vous voyez l'information d'exploitation. Les numéros des images correspondent ceux de rapport.

Figure 2 :
  python3 EshelbyUser.py element_order
  ou utiliser la fonction
  order_analysis()

Figure 3
  python3 EshelbyUser.py E
  ou utiliser la fonction
  plot_E_ratio_influence()

Figure 4
  python3 EshelbyUser.py poisson_standard
  ou utiliser la fonction
  plot_poisson_influence('standard')

Figure 5 
  python3 EshelbyUser.py poisson_incompressible
  ou utiliser la fonction
  plot_poisson_influence('incompressible')

Figure 6
  python3 EshelbyUser.py density
  ou utiliser la fonction
  inclusion_density_influence()
  
## EshelbyEllipse.py

Ce fichier est basé sur EshelbyInhomogeneity.py. Il fait la modélisation de l'inclusion sous la forme elliptique. Il comporte :
 * classe `materio` se n'est pas changée
 * nouveaux paramètres dans la classe `model`. `A_in` et `B_in` sont les deux axes de l'ellipse et `theta` est l'angle de rotation de l'inclusion par rapport à l'axe X du système 
 * classe `solution` a les même méthodes sauf `behind_solution`
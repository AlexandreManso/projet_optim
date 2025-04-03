# # Partie 1

# Question 2: 
# On cherche à minimiser le risque sous contrainte d'un certain rendement r et du fait que le 1T*x= 1
# D'où, le problème d'optimisation se traduit par:
#
# min_(c_eq(x) = 0) (xT*SIGMA*x)
#
# ou c_eq = (c1,c2) et 
# c1 = p_moy*x - r
# c2 = 1T*x -1
#

# $\sqrt{xyz}$

# #Question 3
# Cette contrainte équivaut à dire que la somme des investissements négatifs ne doit pas dépasser un certain seuil s_m.
#
# D'un point de vue d'optimisation, cela revient à ajouter une contrainte de la forme:
# c_3 = 1T*max(-x,O) - s_m <=0 
#
# Cela ajoute de la complexité car on introduit une contrainte d'inégalité non différenciable au problème
#

# #Question 4
# Avec cette nouvelle contrainte, on autorise une position short maximale sur l'action i de s_i avec s>=-x et s>=0. tout en s'assurant que le total des investissements en short ne dépasse pas la valeur s_M de la question précédente. 
# ainsi, on peut limiter les positions de short en introduisant à la question 2 une contrainte linéaire cette fois de la forme :
# c_3 = -(s+x)
# (ou s vérifie s>=-x et s>=0)
#

# # Partie 2

# #Question 5
# Il s'agit d'un problème d'optimisation d'une fonction convexe sous contraintes d'égalités et d'inégalité linéaire.
# En effet, H_f = SIGMA qui est symmétrique définie positive donc f est convexe
#
# Analytiquement, on peut utiliser les algorithmes d'Uzawa, l'algorithme d'élimination des variables pour des contraintes linéaires et l'algorithme de contraintes actives QP.
#

# +
#Question 6
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

one = np.ones(3)
p = [0.05, 0.15, 0.3]
sigma_1 = 0.1
sigma_2 = 0.3
sigma_3 = 0.8
rho = 0.1
r = 0.1
sig = np.array([[sigma_1**2,           rho*sigma_1*sigma_2, 0],
                [rho*sigma_1*sigma_2, sigma_2**2,           0],
      ,         [0,                   0,           sigma_3**2]])

f = lambda x: np.dot(np.dot(np.transpose(x), sig),x)

c_1 = lambda x: np.dot(one, 


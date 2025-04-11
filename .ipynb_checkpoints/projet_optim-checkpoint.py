# # Partie 1

# ### Question 1
#
# Ce problème consiste à trouver la répartition optimale des investissements représentée au vecteur x, qui permet de minimiser les risques. 
# Le risque est lié à la matrice de covariance des variable de variation des prix des acitfs.
#
# Il s'agit donc d'un problème d'optimisation sous contrainte. On cherche à minimiser le risque sous contrainte d'un rendement donné

# ### Question 2: 
#
# On cherche à minimiser le risque sous contrainte d'un certain rendement  $\bar{p}^\intercal  x = r$ et du fait que $\textbf{1}^\intercal x= \textbf{1}$
#
# Ainsi, le problème d'optimisation se traduit par:
# $$
# \left\{
# \begin{array}{l}
# \text{min}(x^\intercal \Sigma  x) \\
# \textit{s.c.} \;\; c_{eq}(x) = 0
# \end{array}
# \right.
# $$
#
#
# ou 
# $$
# c_{eq}(x) = \left( 
# \begin{array}{c}
# c_1(x) = \bar{p} \cdot x - r \\ 
# c_2(x) = \mathbf{1}^\intercal \cdot x - \mathbf{1}
# \end{array}
# \right)
# $$
#
#

# ### Question 3
# Cette contrainte équivaut à dire que les investissements négatifs totaux ne doit pas dépasser un certain seuil $s_m$.
#
# D'un point de vue d'optimisation, cela revient à ajouter à $c_{eq}$ une contrainte de la forme:
# $$
# c_{3,non\;linéaire}(x) = \mathbf{1}^\intercal \cdot max(-x,O) - s_M \leq 0 
# $$
#
# Cela ajoute de la complexité car on introduit une contrainte d'inégalité non différenciable au problème
#

# ### Question 4
# Avec cette nouvelle contrainte, on autorise une position short maximale sur l'action i de $s_i$ avec $s\geq-x$ et $s\geq 0$. tout en s'assurant que le total des investissements en short ne dépasse pas la valeur $s_M$ de la question précédente. 
# ainsi, on peut limiter les positions de short en introduisant à la question 2 une contrainte linéaire cette fois de la forme :
# $$c_3(x) = -(s+x)$$
# (où s vérifie $s\geq -x$ et $s\geq 0$)
#

# # Partie 2

# ### Question 5
# Il s'agit d'un problème d'optimisation d'une fonction convexe sous contraintes d'égalités et d'inégalité linéaire.
# En effet, $\nabla ^2 f = \Sigma$ qui est symmétrique définie positive donc f est convexe
#
# Analytiquement, on peut utiliser les algorithmes d'Uzawa, l'algorithme d'élimination des variables pour des contraintes linéaires et l'algorithme de contraintes actives QP.
#

# ### Question 6

# +
import numpy as np
from scipy import optimize

p1, p2, p3 = 0.05, 0.15, 0.30
sigma1, sigma2, sigma3 = 0.10, 0.30, 0.80
x0 = (1/3)*np.ones(3)
p = np.array([[p1],[p2],[p3]])
rho = 0.5

def optimisation(r,rho) :
    
    def f(x):
        return np.transpose(x)@Sigma@x
   
    def c1(x) :
        n = x.size
        return np.ones(n)@x - 1

    def c2(x) :
        return np.transpose(p)@x - r
   
    Sigma = np.array([[sigma1**2,rho*sigma1*sigma2,0],[rho*sigma1*sigma2,sigma2**2,0],[0,0,sigma3**2]])
    print(optimize.minimize(f,x0,method='SLSQP', constraints=[{'type':'eq', 'fun':c1},{'type':'eq','fun':c2}]))

optimisation(0.1,0.1)

# -

# ## Question 7
# 7.a) $\rho$ représente le coefficient de corrélation des actions. Plus il est grand, plus on considère que les actions sont corrélées et cela infulencera le risque. 
#
# Comme $\Sigma$ est définie positive, on a:
#
# $det(\Sigma) = (\sigma_1 \sigma_2 \sigma_3)^2 \cdot (1-\rho^2)\geq 0$
#
# Donc:
#
# $|\rho| \leq 1$
#
# D'où $\rho \in [-1, 1]$

# *7.b)*

# +
import numpy as np
import matplotlib.pyplot as plt

sigma1 = 0.3
sigma2 = 0.2
sigma3 = 0.1
p = np.array([1, 0, 0]) 
RHO = [0.1, 0.5, -0.5]
n = 100

def f1(rho, x):
    Sigma = np.array([[sigma1**2, rho*sigma1*sigma2, 0],
                      [rho*sigma1*sigma2, sigma2**2, 0],
                      [0, 0, sigma3**2]])
    return np.sqrt(np.dot(np.dot(x.T, Sigma), x))

def f2(x):
    p_reduced = p[:2]  
    return np.dot(p_reduced, x[:2]) 

colors = ['red', 'blue', 'green']


for idx, rho in enumerate(RHO):
    x1v = np.linspace(0, 1, n)
    x2v = np.ones(x1v.size) - x1v 
    x = np.array([[x1v[i], x2v[i], 0] for i in range(x1v.size)])
    X = np.array([f1(rho, xi) for xi in x])
    Y = np.array([f2(xi) for xi in x]) 
    plt.plot(X, Y, marker='o', color=colors[idx], linestyle='none', label=f'Rho = {rho}')

plt.legend()
plt.title("tracé de ((<Σx,x>, p¯x)) sous la contrainte de 1.x = 1 pour différentes valeurs de rho")
plt.show()



# -

# # Partie 3

# ### Question 10
# Avec cette nouvelle fonction coût $g(x) =  -\bar{p}^\intercal x + \mu \, x^\intercal \Sigma x $, on cherche à minimiser le risque (représenté par $x^\intercal \cdot \Sigma  \cdot x$ auquel on soustrait le rendement. Il s'agit alors d'optimiser les "revenus" qu'on pourrait désigner par le rendement moins le risque potentiel. 
# En effet, minimiser $-\bar{p}^\intercal \cdot x$ revient à maximiser le rendement
#
# Si $\mu = 0$ le problème revient à maximiser le rendement des investissements.
#

# ### Question 11
# $\bullet$$*$Si $\mu = 0$ et que $ x \geq 0$ alors, en notant $p_{max} = max(p_i , p_i \geq 0)$:
#
# $g(x) = - \sum_{i=1}^{m} \bar{p}_i x_i \geq -p_{max} \sum_{i=1}^{m} x_i =-p_{max} \;\;\;$     car     $\;\;\; \textbf{1}^\intercal x= \textbf{1}$
#
# Donc en passant au min
#
# $$min_{c(x) = 0}(g(x)) \geq -p_{max} $$
#
# $*$d'autre part, en notant $i_0$ l'indice tel que $p_{i_0} = p_{max}$, avec $x_0 = (\delta (i,i_0))_{(i \in [\llbracket 1,m \rrbracket])}$, on a 
#
# $g(x_0) = - p_{max}$
#
# Donc : 
#
# $$
# min_{c(x) = 0}(g(x)) \leq -p_{max}
# $$
#
# Ainsi, par 5 et 6, on a :
#
# $$
# min_{c(x) = 0}(g(x)) = -p_{max}
# $$
#
# et ce minimum est atteint pour $x_0$ décrit ci dessus. 
# Ce comportement correspond au comportement d'un investisseur ignorant tout risque ($\mu = 0$). Assez intuitivement, dans cette situation, la stratégie la plus rentable est d'investir tout son argent sur l'action la plus rentable. 
#
#
# $\bullet$ pour tout $x\neq 0$:
#
# $\lim_{\mu \to \infty}  -\bar{p}^\intercal  \cdot x + \mu x^\intercal \cdot \Sigma  \cdot x = +\infty$ Car $\Sigma$ est définie positive
#
# et $g(0) = 0$
#
# Ainsi, dans le cas où $\mu \to \infty$, correspondant à la situation d'un investisseur beaucoup trop prudent surestimant tout les risques, la stratégie optimale serait de ne pas investir du tout dans la bourse.
#





from EshelbyInhomogeneity import *
from EshelbyStudies import * 
import sys


def plot_E_ratio_influence() :

  k_list = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
  materiaux = [[i*0.8, 0.8, 0.3, 0.3] for i in [1e-3, 0.5] + list(np.linspace(0.5,1.5, 5)) + list(np.linspace(2,25, 5))[:-1] +list(np.linspace(25, 100, 5))]#[[75*0.8, 0.8, 0.3, 0.3],[30*0.8, 0.8, 0.3, 0.3],[20*0.8, 0.8, 0.3, 0.3],[6*0.8, 0.8, 0.3, 0.3]]
  material_study = material_analysis(materiaux, k_list)


def plot_poisson_influence(mode = 'standard') :
  nu_list = list(np.linspace(0.2, 0.4, 5))[:-1] + list(np.linspace(0.4, 0.45, 5))[:-1] + list(np.linspace(0.45, 0.499999, 10))
  k_list = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15][::2]
  error = nu_analysis(nu_list, k_list, 'standard')

def order_analysis() : 

  k_list = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
  element_order_analysis(k_list)

def inclusion_density_influence() :
  Rext_list = np.linspace(2, 10, 9)
  matrix_expansion(Rext_list)



if __name__ == "__main__":
  

  print(sys.argv[1])

  if (len(sys.argv) == 1) :
    print("nothing to do, try again")

  elif (sys.argv[1] == "E") : plot_E_ratio_influence()
  elif (sys.argv[1] == "poisson_standard") : plot_poisson_influence()
  elif (sys.argv[1] == "poisson_incompressible") : plot_poisson_influence('incompressible')
  elif (sys.argv[1] == "element_order") : order_analysis()
  elif (sys.argv[1] == "density") : inclusion_density_influence()

  else : print("option not supported")



"""
READ_ME.txt:
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
  


"""

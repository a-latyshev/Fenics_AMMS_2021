from EshelbyInhomogeneity import *

import matplotlib.pyplot as plt
import numpy as np
import sys

def nu_analysis(nu_list, k_list, mod='standard') :

  error_list1 = []
  error_list2 = []
  error_list3 = []
  
  for nu in nu_list : #[0.3, 0.35, 0.4, 0.45, 0.49999999] :
      
    material1 = materio(nu, nu)
    material2 = materio(nu, 0.3)
    
    for k in k_list : #[0.35, 0.30, 0.25, 0.2, 0.15, 0.10] :

      modl1 = model(1, 3.9, 1, k)
      modl1.mesh_generator(material1)

      eshelbyEF1 = solution(material1, modl1)

      error_list1.append(eshelbyEF1.error)
     
      if mod == 'standard' :
        modl2 = model(1, 3.9, 1, k)
        modl2.mesh_generator(material2)

        eshelbyEF2 = solution(material2, modl2, mod)

        error_list2.append(eshelbyEF2.error)
      
      modl3 = model(1, 3.9, 2, k)
      modl3.mesh_generator(material1)

      eshelbyEF3 = solution(material1, modl3, mod)

      error_list3.append(eshelbyEF3.error)
  
  error_array1 = np.reshape(error_list1, (len(nu_list), len(k_list)))
  if mod == 'standard' : error_array2 = np.reshape(error_list2, (len(nu_list), len(k_list)))
  error_array3 = np.reshape(error_list3, (len(nu_list), len(k_list)))

  pente1 = [np.polyfit(np.log(k_list), np.log(error_array1[i]), 1)[0] for i in range(len(error_array1))]
  if mod == 'standard' : pente2 = [np.polyfit(np.log(k_list), np.log(error_array2[i]), 1)[0] for i in range(len(error_array1))]
  pente3 = [np.polyfit(np.log(k_list), np.log(error_array3[i]), 1)[0] for i in range(len(error_array1))]

  if mod == 'standard' : 
    plt.plot(nu_list, pente1, marker = "+", label=r'$\nu = \nu_i = nu_m$, P1', c='r')
    plt.plot(nu_list, pente2, marker = "+", label=r'$\nu = \nu_i$, $\nu_m=0.3$, P1')
    plt.plot(nu_list, pente3, marker = "+", label=r'$\nu = \nu_i = \nu_m$, P2', c='purple')
  else : 
    plt.plot(nu_list, pente1, marker = "+", label=r'formulation standard', c='r')
    plt.plot(nu_list, pente3, marker = "+", label='formulation mixte')
  plt.title(r"Vitesse de convergence en fonction de $\nu$")
  plt.grid(True)
  plt.legend()
  plt.xlabel(r'$\nu$')
  plt.ylabel(r'$\tau$')
  plt.savefig("convergence_nu.png", format='png')
  plt.show()

  return error_list1


def material_analysis(materiaux, k_list) :

  strain_inc_list = []
  E_ratio_list = []

  error_list = []

  un_index = 0

  for l,materiau in enumerate(materiaux) :
    material = materio(materiau[2], materiau[3], materiau[0], materiau[1])
    E_ratio_list.append(materiau[0]/materiau[1])

    if abs(materiau[0]/materiau[1] -1) < 0.0001 : un_index = l

    for k in k_list : #[0.35, 0.30, 0.25, 0.2, 0.15, 0.10] :

      modl = model(1, 3.9, 1, k)
      modl.mesh_generator(material)

      eshelbyEF = solution(material, modl)

      error_list.append(eshelbyEF.error)

      if k == k_list[-1] : strain_inc_list.append(eshelbyEF.strain_moy_inc[-1])
  
  error_array = np.reshape(error_list, (len(E_ratio_list), len(k_list)))

  pente = [np.polyfit(np.log(k_list), np.log(error_array[i]), 1)[0] for i in range(len(error_array))]
  
  fig, (ax1, ax2) = plt.subplots(2, 1)

  fig.suptitle(r'Analyse de convergence $E_i/E_m$ dans le matériau')

  fig.subplots_adjust(hspace=0.5)

  ax2.plot(k_list, error_array[un_index])
  ax2.set_xlabel(r'$k$')
  ax2.set_ylabel('erreur')
  ax2.set_title(r'erreur = f($E_i/E_m$)')
  ax2.grid(True)

  ax1.semilogx(E_ratio_list, pente, marker = "+", label=r'$\tau = f(E_i/E_m)$', c='r')
  ax1.set_title(r'Vitesse de convergence en fonction du $E_i/E_m$ dans le matériau')
  ax1.grid(True)
  ax1.legend()
  ax1.set_xlabel(r'$E_i/E_m$')
  ax1.set_ylabel(r'$\tau$')
  
  plt.savefig("E_ratio_convergence.png", format='png')
  plt.show()

  """
  plt.plot(E_ratio_list, strain_inc_list, marker = "+", label=r'$<\varepsilon_{xy}> = f(E_i/E_m)$', c='r')
  plt.title(r'$<\varepsilon_{xy}>$ en fonction du rapport de rigidité entre la matrice et le matériau')
  plt.grid(True)
  plt.legend()
  plt.xlabel(r'$E_i/E_m$')
  plt.ylabel(r'$<\varepsilon_{xy}>$')
  plt.show()"""


def element_order_analysis(k_list) :

  error_list1 = []
  error_list2 = []
  error_list3 = []
  error_list4 = []

  for k in k_list : 
      
    material = materio()

    modl1 = model(1, 3.9, 1, k)
    modl1.mesh_generator(material)

    eshelbyEF11 = solution(material, modl1, 'H1')
    eshelbyEF12 = solution(material, modl1, 'L2')

    error_list1.append(eshelbyEF11.error)
    error_list3.append(eshelbyEF12.error)
      
    modl2 = model(1, 3.9, 2, k)
    modl2.mesh_generator(material)

    eshelbyEF21 = solution(material, modl2, 'H1')
    eshelbyEF22 = solution(material, modl2, 'L2')

    error_list2.append(eshelbyEF21.error)
    error_list4.append(eshelbyEF22.error)
  
  error_array1 = np.array(error_list1)
  error_array2 = np.array(error_list2)
  error_array3 = np.array(error_list3)
  error_array4 = np.array(error_list4)

  pente1 = np.polyfit(np.log(k_list), np.log(error_array1), 1)
  pente2 = np.polyfit(np.log(k_list), np.log(error_array2), 1)
  pente3 = np.polyfit(np.log(k_list), np.log(error_array3), 1)
  pente4 = np.polyfit(np.log(k_list), np.log(error_array4), 1)

  plt.loglog(k_list, error_list1, marker = "+", label='P1, norm H1 : erreur = {:.2f}*k + {:.2f}'.format(pente1[0], pente1[1]), c='r')
  plt.loglog(k_list, error_list2, marker = "+", label='P2, norm H1 : erreur = {:.2f}*k + {:.2f}'.format(pente2[0], pente2[1]))
  plt.loglog(k_list, error_list3, marker = "+", label='P1, norm L2 : erreur = {:.2f}*k + {:.2f}'.format(pente3[0], pente3[1]), c='purple')
  plt.loglog(k_list, error_list4, marker = "+", label='P2, norm L2 : erreur = {:.2f}*k + {:.2f}'.format(pente4[0], pente3[1]), c='green')
  plt.title(r"Etude de convergence en fonction de la discrétisation")
  plt.grid(True)
  plt.legend()
  plt.xlabel(r'$k$')
  plt.ylabel('erreur')
  plt.savefig("convergence_h.png", format='png')
  plt.show()
    

def matrix_expansion(R_list) :

  error_hill_list = []

  material = materio()

  for R in R_list :
    
    modl = model(1, R, 1, 0.1)
    modl.mesh_generator(material)

    eshelbyEF = solution(material, modl)

    error_hill_list.append(abs((eshelbyEF.hill-eshelbyEF.strain_moy_inc[-1])/eshelbyEF.hill))

  plt.plot(R_list, error_hill_list, marker='+', c='r')
  plt.title("Comparaison entre la solution proposée par Hill et la solution EF")
  plt.grid(True)
  plt.xlabel('R')
  plt.ylabel('erreur relative')
  plt.savefig("radius_hill.png", format='png')
  plt.show()



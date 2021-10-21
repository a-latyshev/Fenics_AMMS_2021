import dolfin
import matplotlib.pyplot as plt
import mshr
import numpy as np

#INPUT PARAMETERS

class materio :

  E_m = 0.8 # Young's modulus for the matrix
  nu_m = 0.35 # Poisson ratio for the matrix

  E_i = 11.0 # Young's modulus for the inclusion
  #nu_i = 0.3 # Poisson ratio for the inclusion

  def lame_from_E_nu(self, E, nu) : #return \mu et \lambda dans cet ordre à partir de E et \nu
    return 0.5*E/(1+nu), 2*0.5*E/(1+nu)*nu/(1-2*nu)

  def __init__(self, nu_i=0.3, nu_m = 0.35, E_i=11.0, E_m=0.8) : 
    
    self.nu_i = nu_i 
    self.nu_m = nu_m
    self.E_i = E_i
    self.E_m = E_m

    self.mu_m, self.lamb_m = self.lame_from_E_nu(self.E_m, self.nu_m) 
    self.mu_i, self.lamb_i = self.lame_from_E_nu(self.E_i, self.nu_i) 


class model : 

  # R_in = 1.0 # radius of the inclusion
  # R_out = 3.9 # radius of the outter matrix
  
  origin = dolfin.Point(0., 0.)
  
  # h = 0.35*R_in # Size of elements

  MATRIX_ID = 1
  INCLUSION_ID = 2

  # degreeFE = 1 # Degree of the finite elements

  u_boundary = dolfin.Expression(("-x[1]", "-x[0]"), degree=1)

  def __init__(self, R_in=1.0, R_out=3.9, degreeFE=1, k=0.35) : 
    self.R_in = R_in
    self.R_out = R_out
    self.degreeFE = degreeFE
    self.h = k*R_in

  def mesh_generator(self, material) :

    L_in = 2*np.pi*self.R_in # perimeter of the inclusion
    L_out = 2*np.pi*self.R_out # perimeter of the matrix

    N_in = int(L_in/self.h) # number of mesh points on the perimeter 
                       # of the inclusion
    N_out = int(L_out/self.h) # number of mesh points on the perimeter 
                         # of the matrix


    Omega_i = mshr.Circle(self.origin, self.R_in, segments=N_in)
    Omega = mshr.Circle(self.origin, self.R_out, segments=N_out)


    Omega.set_subdomain(self.MATRIX_ID, Omega-Omega_i) # we are putting tags in parts of the mesh
    Omega.set_subdomain(self.INCLUSION_ID, Omega_i)    # we will use them later


    self.mesh = mshr.generate_mesh(Omega, resolution=2*self.R_out/self.h)

    # dolfin.plot(self.mesh)
    #plt.show()

    # we define a function = 1 in the matrix and = 2 in the inclusion
    subdomain_data_2d = dolfin.MeshFunction("size_t", # the function returns a positive integer
                                            self.mesh, # it is define over the entire mesh
                                            dim=2, # the function is defined on the cells (not edges nor vertices)
                                            value=self.mesh.domains() # the function value is in fact
                                                             # given by the tag we have put while creating the mesh
                                           ) 

    # plt.colorbar(dolfin.plot(subdomain_data_2d)) # we plot this function, note the added color scale on the side


    # we need to be able to integrate over the matrix only or the inclusion only
    # so in addition of the classical dolfin measure dx, we define dx(1) and dx(2)
    self.dx = dolfin.Measure("dx", domain=self.mesh, subdomain_data=subdomain_data_2d)

#BEHAVIOR

#Hook's law
def stress(eps, lamb, mu):
  return lamb*dolfin.tr(eps)*dolfin.Identity(2) + 2*mu*eps

# eps(u)
def strain(u):
  grad_u = dolfin.nabla_grad(u)
  return 0.5*(grad_u+grad_u.T)

def strain_dev(u):
  
  eps = strain(u)
  return eps - 1/3.*dolfin.tr(eps)*dolfin.Identity(2)


class solution :

  def variational_problem(self, material, modl, mod) :

    element_u = dolfin.VectorElement('P', 
                                   cell=modl.mesh.ufl_cell(), 
                                   degree=modl.degreeFE, 
                                   dim=2)


    if mod != 'incompressible' :
      
      self.V = dolfin.FunctionSpace(modl.mesh, element_u)
      u = dolfin.TrialFunction(self.V)
      v = dolfin.TestFunction(self.V)

      a_i = dolfin.inner(stress(strain(u), material.lamb_i, material.mu_i), strain(v))*modl.dx(modl.INCLUSION_ID)
      a_m = dolfin.inner(stress(strain(u), material.lamb_m, material.mu_m), strain(v))*modl.dx(modl.MATRIX_ID)
    
      self.linear_form = dolfin.inner(dolfin.Constant((0, 0)), v)*modl.dx

    else :

      element_p = dolfin.FiniteElement('P', 
                                       cell=modl.mesh.ufl_cell(), 
                                       degree=modl.degreeFE)
      self.V = dolfin.FunctionSpace(modl.mesh, element_u*element_p)
      
      (u, p) = dolfin.TrialFunctions(self.V)
      (v, q) = dolfin.TestFunctions(self.V)

      kappa_i = material.lamb_i+2/2*material.mu_i
      kappa_m = material.lamb_m+2/2*material.mu_m

      n = dolfin.FacetNormal(modl.mesh)
      ds = dolfin.Measure("ds", subdomain_data="on_boundary")

      a_i = (2*material.mu_i*dolfin.inner(strain_dev(u), strain_dev(v))+q*(dolfin.div(u)-p/kappa_i)+p*dolfin.div(v))*modl.dx(modl.INCLUSION_ID)
      a_m = (2*material.mu_m*dolfin.inner(strain_dev(u), strain_dev(v))+q*(dolfin.div(u)-p/kappa_m)+p*dolfin.div(v))*modl.dx(modl.MATRIX_ID) 
      #a_i = dolfin.inner(stress(strain(u), material.lamb_i, material.mu_i), strain(v))*modl.dx(modl.INCLUSION_ID)
      #a_m = dolfin.inner(stress(strain(u), material.lamb_m, material.mu_m), strain(v))*modl.dx(modl.MATRIX_ID)

      self.linear_form = dolfin.inner(dolfin.Constant((0, 0)), v)*modl.dx #dolfin.inner(0*n, v)*ds

    self.bilinear_form = a_i + a_m


  def solution_maker(self, modl, mod) :
  
    if mod != 'incompressible' :
      boundary_conditions = dolfin.DirichletBC(self.V, modl.u_boundary, "on_boundary")

      self.usol = dolfin.Function(self.V)
      dolfin.solve(self.bilinear_form == self.linear_form, self.usol, boundary_conditions)

    else :
      boundary_conditions = dolfin.DirichletBC(self.V.sub(0), modl.u_boundary, "on_boundary")

      mix_solution = dolfin.Function(self.V)
      dolfin.solve(self.bilinear_form == self.linear_form, mix_solution, boundary_conditions)

      (self.usol, ballec) = mix_solution.split()


  def vector_av(self, vector, subdomain, dx) :
    
    if (subdomain != 1) & (subdomain != 2):
      return [dolfin.assemble(vector[i]*dx)/dolfin.assemble(dolfin.Constant(1.)*dx) for i in range(len(vector))] 
      
    else :
      return [dolfin.assemble(vector[i]*dx(subdomain))/dolfin.assemble(dolfin.Constant(1.)*dx(subdomain)) for i in range(len(vector))]
    
  
  def behind_solution(self, material, modl, norm) :
    """
    strain_field_space = dolfin.FunctionSpace(modl.mesh, 'DG', 0)
    strain_field = [dolfin.project(strain(self.usol)[min(i,1), i%2], strain_field_space) for i in range(3)]

    self.strain_moy_inc = self.vector_av(strain_field, modl.INCLUSION_ID, modl.dx)
    self.strain_moy_mat = self.vector_av(strain_field, modl.MATRIX_ID, modl.dx)
    self.strain_moy = self.vector_av(strain_field, 'all', modl.dx)

    deviation_inc = [dolfin.assemble(abs(strain_field[i]-self.strain_moy_inc[i])*modl.dx(modl.INCLUSION_ID))/self.strain_moy_inc[i] for i in range(3)]
    deviation_mat = [dolfin.assemble(abs(strain_field[i]-self.strain_moy_mat[i])*modl.dx(modl.MATRIX_ID))/self.strain_moy_mat[i] for i in range(3)]
    deviation_tot = [dolfin.assemble(abs(strain_field[i]-self.strain_moy[i])*modl.dx)/self.strain_moy[i] for i in range(3)]

    q = (3-4*material.nu_m)/(8*material.mu_m*(1-material.nu_m))
    b = 1/(1+2*q*(material.mu_i-material.mu_m))

    self.hill = -b
     """
    from eshelby import EshelbyDisk

    solution_ref = EshelbyDisk(modl.R_out/modl.R_in, material.E_i/material.E_m, material.nu_i, material.nu_m)

    u_ref = solution_ref.to_expression(modl.R_in, degree=2)

    # Execute this to obtain the plot of the analytical solution
    V_ref = dolfin.VectorFunctionSpace(modl.mesh, 'P', 1)
    u_ref_num = dolfin.interpolate(u_ref, V_ref)
    # dolfin.plot(0.15*u_ref_num, mode="displacement")
    
    self.error = dolfin.errornorm(u_ref, self.usol, norm)

    # print("LOG")
    # print("average strain in inclusion = ", strain_moy_inc)
    # print("average strain in matrix = ", strain_moy_mat)
    # print("deviation in inclustion = ", deviation_inc)
    # print("deviation in matrix = ", deviation_mat)
    # print('eps_xy_inclusion = ',-b)
    # print("erreur relative = ", (strain_moy_inc[2] + b)/b)
    print("erreur L2 : ", self.error)

    return self.error
    # eps = 0.1
    # liste_x = np.linspace(-modl.R_out+eps, modl.R_out-eps, num=100)

    # u_formule = 0.0*liste_x
    # u_solution = 0.0*liste_x

    # for k, x_k in enumerate(liste_x):
    #   u_formule[k] = u_ref([x_k,0.0])[1]
    #   u_solution[k] = self.usol(x_k, 0.)[1]
  
    # plt.plot(liste_x, u_formule)
    # plt.plot(liste_x, u_solution)
    # plt.show()

  def __init__(self, material, modl, mod='standard', norm='L2') :
       
    self.variational_problem(material, modl, mod)
    self.solution_maker(modl, mod)
    self.behind_solution(material, modl, norm)

"""
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


  plt.plot(E_ratio_list, strain_inc_list, marker = "+", label=r'$<\varepsilon_{xy}> = f(E_i/E_m)$', c='r')
  plt.title(r'$<\varepsilon_{xy}>$ en fonction du rapport de rigidité entre la matrice et le matériau')
  plt.grid(True)
  plt.legend()
  plt.xlabel(r'$E_i/E_m$')
  plt.ylabel(r'$<\varepsilon_{xy}>$')
  plt.show()


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


"""


"""
k_list = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
materiaux = [[i*0.8, 0.8, 0.3, 0.3] for i in [1e-3, 0.5] + list(np.linspace(0.5,1.5, 5)) + list(np.linspace(2,25, 5))[:-1] +list(np.linspace(25, 100, 5))]#[[75*0.8, 0.8, 0.3, 0.3],[30*0.8, 0.8, 0.3, 0.3],[20*0.8, 0.8, 0.3, 0.3],[6*0.8, 0.8, 0.3, 0.3]]
material_study = material_analysis(materiaux, k_list)
"""

"""
nu_list = list(np.linspace(0.2, 0.4, 5))[:-1] + list(np.linspace(0.4, 0.45, 5))[:-1] + list(np.linspace(0.45, 0.499999, 10))
k_list = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15][::2]
error = nu_analysis(nu_list, k_list, 'incompressible')
"""


"""
k_list = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
element_order_analysis(k_list)
"""


"""
Rext_list = np.linspace(2, 10, 9)
matrix_expansion(Rext_list)
"""

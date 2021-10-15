#!/usr/bin/env python
# coding: utf-8

# # FEniCS simulation of Eshelby's circular inclusion problem

# The aim of this notebook is to setup a FEniCS simulation of Eshelby's inclusion problem. The framework is linear, plane strain elasticity. We have a matrix in a disk around the origin (radius $R_m$) with an inclusion having the shape of another disk around the origin, with a smaller radius ($R_i < R_m$). The matrix and the inclusion have different elastic modulus ($E$: Young modulus; $\nu$: Poisson ratio) but are both isotropic and linearly elastic:
# 
# \begin{equation}
# \sigma_{ij} = \lambda\varepsilon_{kk}\delta_{ij}+2\mu\varepsilon_{ij},
# \end{equation}
# 
# where indices $i, j, k$ are restricted to $\{1, 2\}$ and $\lambda$, $\mu$ are the plane strain Lamé coefficients :
# 
# \begin{equation*}
# \mu=\frac{E}{2\bigl(1+\nu\bigr)}
# \quad\text{and}\quad
# \lambda=\frac{2\mu\nu}{1-2\nu}.
# \end{equation*}
# 
# The variational formulation of the problem is the following:
# 
# Find $u\in \mathcal{C}\equiv\{u: H^1(\Omega), \; u(x_1,x_2)|_{x_1^2+x_2^2=R_m^2}
# %\text{border}
# =(-x_2,-x_1)\}$ such that 
# $\forall v\in \mathcal{C}_0\equiv \mathcal{C}$
# 
# 
# \begin{equation}
# \int_\Omega \sigma(\varepsilon(u)):\varepsilon(v)\,\mathrm{d}x\,\mathrm{d}y =
# -\int_{\Omega} b \cdot v\,\mathrm{d} x\,\mathrm{d} y,
# \end{equation}
# 
# where the body force $b=0$ and $\sigma(\varepsilon)$ is the constitutive equation and $\varepsilon(u)=\mathrm{sym} (\nabla u)$  

# ![shema](./inclusion_shear.png)

# In[614]:


import dolfin
import matplotlib.pyplot as plt
import mshr
import numpy as np

#INPUT PARAMETERS

class materio :

  E_m = 0.8 # Young's modulus for the matrix
  nu_m = 0.35 # Poisson ratio for the matrix

  E_i = 11.0 # Young's modulus for the inclusion
  nu_i = 0.3 # Poisson ratio for the inclusion

  def lame_from_E_nu(E, nu) : #return \mu et \lambda dans cet ordre à partir de E et \nu
    return 0.5*E/(1+nu), 2*0.5*E/(1+nu)*nu/(1-2*nu)

  # def __init__(self, nu_i=0.3) : 
  #   self.nu_i = nu_i


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


class solution :

  def variational_problem(self, material, modl) :

    element = dolfin.VectorElement('P', 
                                   cell=modl.mesh.ufl_cell(), 
                                   degree=modl.degreeFE, 
                                   dim=2)
    self.V = dolfin.FunctionSpace(modl.mesh, element)

    u = dolfin.TrialFunction(self.V)
    v = dolfin.TestFunction(self.V)

    a_i = dolfin.inner(stress(strain(u), material.lamb_i, material.mu_i), strain(v))*modl.dx(modl.INCLUSION_ID)
    a_m = dolfin.inner(stress(strain(u), material.lamb_m, material.mu_m), strain(v))*modl.dx(modl.MATRIX_ID)

    self.bilinear_form = a_i + a_m
    self.linear_form = dolfin.inner(dolfin.Constant((0, 0)), v)*modl.dx


  def solution_maker(self, modl) :
  
    boundary_conditions = dolfin.DirichletBC(self.V, modl.u_boundary, "on_boundary")

    self.usol = dolfin.Function(self.V)
    dolfin.solve(self.bilinear_form == self.linear_form, self.usol, boundary_conditions)

  def vector_av(self, vector, subdomain, dx) :
    
    if (subdomain != 1) & (subdomain != 2):
      return [dolfin.assemble(vector[i]*dx)/dolfin.assemble(dolfin.Constant(1.)*dx) for i in range(len(vector))] 
      
    else :
      return [dolfin.assemble(vector[i]*dx(subdomain))/dolfin.assemble(dolfin.Constant(1.)*dx(subdomain)) for i in range(len(vector))]
    
  
  def behind_solution(self, material, modl) :

    strain_field_space = dolfin.FunctionSpace(modl.mesh, 'DG', 0)
    strain_field = [dolfin.project(strain(self.usol)[min(i,1), i%2], strain_field_space) for i in range(3)]

    strain_moy_inc = self.vector_av(strain_field, modl.INCLUSION_ID, modl.dx)
    strain_moy_mat = self.vector_av(strain_field, modl.MATRIX_ID, modl.dx)
    strain_moy = self.vector_av(strain_field, 'all', modl.dx)

    deviation_inc = [dolfin.assemble(abs(strain_field[i]-strain_moy_inc[i])*modl.dx(modl.INCLUSION_ID))/strain_moy_inc[i] for i in range(3)]
    deviation_mat = [dolfin.assemble(abs(strain_field[i]-strain_moy_mat[i])*modl.dx(modl.MATRIX_ID))/strain_moy_mat[i] for i in range(3)]
    deviation_tot = [dolfin.assemble(abs(strain_field[i]-strain_moy[i])*modl.dx)/strain_moy[i] for i in range(3)]

    q = (3-4*material.nu_m)/(8*material.mu_m*(1-material.nu_m))
    b = 1/(1+2*q*(material.mu_i-material.mu_m))

    from eshelby import EshelbyDisk

    solution_ref = EshelbyDisk(modl.R_out/modl.R_in, material.E_i/material.E_m, material.nu_i, material.nu_m)

    u_ref = solution_ref.to_expression(modl.R_in)

    # Execute this to obtain the plot of the analytical solution
    V_ref = dolfin.VectorFunctionSpace(modl.mesh, 'P', 1)
    u_ref_num = dolfin.interpolate(u_ref, V_ref)
    # dolfin.plot(0.15*u_ref_num, mode="displacement")
    
    self.error = dolfin.errornorm(u_ref, self.usol, 'L2')

    print("LOG")
    print("average strain in inclusion = ", strain_moy_inc)
    # print("average strain in matrix = ", strain_moy_mat)
    # print("deviation in inclustion = ", deviation_inc)
    # print("deviation in matrix = ", deviation_mat)
    # print('eps_xy_inclusion = ',-b)
    # print("erreur relative = ", (strain_moy_inc[2] + b)/b)
    print("erreur L2 : ", self.error)

    liste_x = np.linspace(-modl.R_out, modl.R_out, num=100)

    u_formule = 0.0*liste_x
    u_solution = 0.0*liste_x

    for k, x_k in enumerate(liste_x):
      u_formule[k] = u_ref([x_k,0.0])[1]
      u_solution[k] = self.usol(x_k, 0.)[1]
  
    # plt.plot(liste_x, u_formule)
    # plt.plot(liste_x, u_solution)
    # plt.show()

  def __init__(self, material, modl) :
    
    material.mu_m, material.lamb_m = material.lame_from_E_nu(material.E_m, material.nu_m) 
    material.mu_i, material.lamb_i = material.lame_from_E_nu(material.E_i, material.nu_i) 
   
    self.variational_problem(material, modl)
    self.solution_maker(modl)
    self.behind_solution(material, modl)


def study() :
  
  modl0 = model(1, 3.9, 0.35, 1)
  modl0.mesh_generator(materio)

  eshelbyEF = solution(materio, modl0)

# study()

# # We now consider the strain tensor of the solution

# In[642]:




# ## We want to plot eps_xx, eps_yy, eps_xy using the dolfin.plot() command but eps has no FunctionSpace, so we need to define one and dolfin.project() each of the four tensor components of eps on this new function space and then plot it with the dolfin.plot() command

# In[643]:


"""element2 = dolfin.TensorElement('P', 
                               cell=mesh.ufl_cell(), 
                               degree=degreeFE, 
                               dim=2)
T = dolfin.TensorFunctionSpace(mesh, element2)"""


"""for i in range(3) :
    plt.colorbar(dolfin.plot(strain_field[i]))
    plt.show()"""

# ## 10) Compare the strains in the inclusion with the strains in the matrix: are they uniform? What are the dominant components?

#  La composante principale du tenseur de déformation est la composante hors diagonale eps_xy

# # 11)Compute average strains 
# 
# \begin{equation}
# <\varepsilon_{ij}>=\frac{\int_\Omega \varepsilon_{ij}\,\mathrm{d}x\,\mathrm{d}y }{ \int_\Omega 1 \, \mathrm{d}x\,\mathrm{d}y }
# \end{equation}
# 
# # over the matix subdomain and the inclusion subdomain

# In[644]:



    


# # 12)Show that the shear strain in the inclusion is 'almost' uniform. Quantify the term 'almost' by computing
# 
# \begin{equation}
# deviation = \frac{\int_\Omega Abs(\varepsilon_{ij} - <\varepsilon_{ij}>) \,\mathrm{d}x\,\mathrm{d}y }{ <\varepsilon_{ij}> }
# \end{equation}

# In[645]:




# # 13)Does $<\varepsilon_{xy}>$ follow the formula of the Hill tensor?

# In[646]:




# Non on trouve -0.127 contre -0.109 avec la formule de Hill, soit une erreur relative de 17%

# # Comparison with exact solution

# The closed-form expression of the solution is derived in the companion notebook *Circular inhomogeneity — Shear*. This solution is implemented in the module `eshelby`, that can be imported. It builds the exact displacement field as a `dolfin.Expression`, which will allow easy comparison with the approximate solution found here.






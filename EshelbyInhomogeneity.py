import dolfin
import matplotlib.pyplot as plt
import mshr
import numpy as np

#INPUT PARAMETERS

class materio :

  E_m = 0.8 # Young's modulus for the matrix
  nu_m = 0.35 # Poisson ratio for the matrix

  E_i = 11.0 # Young's modulus for the inclusion

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

  origin = dolfin.Point(0., 0.)

  MATRIX_ID = 1
  INCLUSION_ID = 2

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

    # we define a function = 1 in the matrix and = 2 in the inclusion
    subdomain_data_2d = dolfin.MeshFunction("size_t", # the function returns a positive integer
                                            self.mesh, # it is define over the entire mesh
                                            dim=2, # the function is defined on the cells (not edges nor vertices)
                                            value=self.mesh.domains() # the function value is in fact
                                                             # given by the tag we have put while creating the mesh
                                           ) 

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
      
      a_i = (2*material.mu_i*dolfin.inner(strain_dev(u), strain_dev(v))+q*(dolfin.div(u)-p/kappa_i)+p*dolfin.div(v))*modl.dx(modl.INCLUSION_ID)
      a_m = (2*material.mu_m*dolfin.inner(strain_dev(u), strain_dev(v))+q*(dolfin.div(u)-p/kappa_m)+p*dolfin.div(v))*modl.dx(modl.MATRIX_ID) 

      self.linear_form = dolfin.inner(dolfin.Constant((0, 0)), v)*modl.dx

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
    from eshelby import EshelbyDisk

    solution_ref = EshelbyDisk(modl.R_out/modl.R_in, material.E_i/material.E_m, material.nu_i, material.nu_m)

    u_ref = solution_ref.to_expression(modl.R_in, degree=2)
    self.error = dolfin.errornorm(u_ref, self.usol, norm)
    print("erreur L2 : ", self.error)
    return self.error


  def __init__(self, material, modl, mod='standard', norm='L2') :
    self.variational_problem(material, modl, mod)
    self.solution_maker(modl, mod)
    self.behind_solution(material, modl, norm)

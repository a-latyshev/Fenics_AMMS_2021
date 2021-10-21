#!/usr/bin/env python
# coding: utf-8

# # FEniCS simulation of Eshelby's inclusion problem

import dolfin
import mshr
import numpy as np

#Hook's law
def stress(eps, lamb, mu):
  return lamb*dolfin.tr(eps)*dolfin.Identity(2) + 2*mu*eps
 
# eps(u)
def strain(u):
  grad_u = dolfin.nabla_grad(u)
  return 0.5*(grad_u+grad_u.T)

def vector_av(self, vector, subdomain, dx) :
  
  if (subdomain != 1) & (subdomain != 2):
    return [dolfin.assemble(vector[i]*dx)/dolfin.assemble(dolfin.Constant(1.)*dx) for i in range(len(vector))] 
    
  else :
    return [dolfin.assemble(vector[i]*dx(subdomain))/dolfin.assemble(dolfin.Constant(1.)*dx(subdomain)) for i in range(len(vector))]

#INPUT PARAMETERS

class materio :

  E_m = 0.8 # Young's modulus for the matrix
  nu_m = 0.35 # Poisson ratio for the matrix

  E_i = 11.0 # Young's modulus for the inclusion
  nu_i = 0.3 # Poisson ratio for the inclusion

  def lame_from_E_nu(E, nu) : #return \mu et \lambda dans cet ordre à partir de E et \nu
    return 0.5*E/(1+nu), 2*0.5*E/(1+nu)*nu/(1-2*nu)

class model : 
  
  MATRIX_ID = 1
  INCLUSION_ID = 2

  u_boundary = dolfin.Expression(("-x[1]", "-x[0]"), degree=1)

  def __init__(self, A_in=1.0, B_in=1.0, theta=0, R_out=3.9, degreeFE=1, h=0.35) : 
    self.A_in = A_in
    self.B_in = B_in
    self.R_out = R_out
    self.degreeFE = degreeFE
    self.h = h
    self.theta=theta

  def mesh_generator(self) :
    L_in = 4 * (np.pi * self.A_in * self.B_in  + (self.A_in - self.B_in)**2) / (self.A_in + self.B_in)
    L_out = 2*np.pi*self.R_out # perimeter of the matrix

    N_in = int(L_in/self.h) # number of mesh points on the perimeter 
                       # of the inclusion
    N_out = int(L_out/self.h) # number of mesh points on the perimeter 
                         # of the matrix

    origin = dolfin.Point(0., 0.)
    Omega_i = mshr.Ellipse(origin, self.A_in, self.B_in, segments=N_in)
    Omega_i = mshr.CSGRotation(Omega_i, self.theta)

    Omega = mshr.Circle(origin, self.R_out, segments=N_out)


    Omega.set_subdomain(self.MATRIX_ID, Omega - Omega_i) # we are putting tags in parts of the mesh
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

class solution :

  def variational_problem(self, material, model) :

    element = dolfin.VectorElement('P', 
                                   cell=model.mesh.ufl_cell(), 
                                   degree=model.degreeFE, 
                                   dim=2)
    self.V = dolfin.FunctionSpace(model.mesh, element)

    u = dolfin.TrialFunction(self.V)
    v = dolfin.TestFunction(self.V)

    a_i = dolfin.inner(stress(strain(u), material.lamb_i, material.mu_i), strain(v))*model.dx(model.INCLUSION_ID)
    a_m = dolfin.inner(stress(strain(u), material.lamb_m, material.mu_m), strain(v))*model.dx(model.MATRIX_ID)

    self.bilinear_form = a_i + a_m
    self.linear_form = dolfin.inner(dolfin.Constant((0, 0)), v)*model.dx
    self.usol = dolfin.Function(self.V)

  def solve(self, model) :
    boundary_conditions = dolfin.DirichletBC(self.V, model.u_boundary, "on_boundary")
    dolfin.solve(self.bilinear_form == self.linear_form, self.usol, boundary_conditions, solver_parameters={"linear_solver": "mumps"})

  def __init__(self, material, model) :
    
    material.mu_m, material.lamb_m = material.lame_from_E_nu(material.E_m, material.nu_m) 
    material.mu_i, material.lamb_i = material.lame_from_E_nu(material.E_i, material.nu_i) 
   
    self.variational_problem(material, model)

#!/usr/bin/env python
# coding: utf-8

# # FEniCS simulation of Eshelby's circular inclusion problem

import dolfin
import matplotlib.pyplot as plt
import mshr
import numpy as np
from eshelby import EshelbyDisk

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
    # print(len(self.usol.vector().get_local()))

  def solve(self, model) :
    boundary_conditions = dolfin.DirichletBC(self.V, model.u_boundary, "on_boundary")
    dolfin.solve(self.bilinear_form == self.linear_form, self.usol, boundary_conditions, solver_parameters={"linear_solver": "mumps"})
    
  def getExactError(self, material, model) :
    solution_ref = EshelbyDisk(model.R_out/model.A_in, material.E_i/material.E_m, material.nu_i, material.nu_m)
    u_ref = solution_ref.to_expression(model.A_in)
    self.error = dolfin.errornorm(u_ref, self.usol, 'L2')

  # def getError(self, material, model) : 
  #   self.error = dolfin.errornorm(usol_fine_mesh, self.usol, 'L2')

  # def solveFineMesh(self, model_fine) :
  #   model_tmp = model(A_in=model.A_in, B_in=model.B_in, h=h, degreeFE=model.degreeFE)
  #   boundary_conditions = dolfin.DirichletBC(self.V, model.u_boundary, "on_boundary")
  #   dolfin.solve(self.bilinear_form == self.linear_form, self.usol, boundary_conditions, solver_parameters={"linear_solver": "mumps"})
  
  def post_processing(self, material, modl) :

    # strain_field_space = dolfin.FunctionSpace(modl.mesh, 'DG', 0)
    # self.strain_field = [dolfin.project(strain(self.usol)[min(i,1), i%2], strain_field_space) for i in range(3)]

    # self.strain_moy_inc = self.vector_av(self.strain_field, modl.INCLUSION_ID, modl.dx)
    # strain_moy_mat = self.vector_av(self.strain_field, modl.MATRIX_ID, modl.dx)
    # strain_moy = self.vector_av(self.strain_field, 'all', modl.dx)

    # self.deviation_inc = [dolfin.assemble(abs(self.strain_field[i]-self.strain_moy_inc[i])*modl.dx(modl.INCLUSION_ID))/self.strain_moy_inc[i] for i in range(3)]
    # deviation_mat = [dolfin.assemble(abs(self.strain_field[i]-strain_moy_mat[i])*modl.dx(modl.MATRIX_ID))/strain_moy_mat[i] for i in range(3)]
    # deviation_tot = [dolfin.assemble(abs(self.strain_field[i]-strain_moy[i])*modl.dx)/strain_moy[i] for i in range(3)]

    # q = (3-4*material.nu_m)/(8*material.mu_m*(1-material.nu_m))
    # b = 1/(1+2*q*(material.mu_i-material.mu_m))

    solution_ref = EshelbyDisk(modl.R_out/modl.A_in, material.E_i/material.E_m, material.nu_i, material.nu_m)
    u_ref = solution_ref.to_expression(modl.A_in)

    # Execute this to obtain the plot of the analytical solution
    # V_ref = dolfin.VectorFunctionSpace(modl.mesh, 'P', degree=modl.degreeFE)
    # u_ref_num = dolfin.interpolate(u_ref, V_ref)
    # dolfin.plot(0.15*u_ref_num, mode="displacement")
    
    # self.error = dolfin.errornorm(u_ref, self.usol, 'L2')
    self.error = dolfin.errornorm(u_ref, self.usol, 'L2', degree_rise=5)

    # print("LOG")
    # print("average strain in inclusion = ", strain_moy_inc)
    # print("average strain in matrix = ", strain_moy_mat)
    # print("deviation in inclustion = ", deviation_inc)
    # print("deviation in matrix = ", deviation_mat)
    # print('eps_xy_inclusion = ',-b)
    # print("erreur relative = ", (strain_moy_inc[2] + b)/b)
    # print("erreur L2 : ", self.error)

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

  def __init__(self, material, model) :
    
    material.mu_m, material.lamb_m = material.lame_from_E_nu(material.E_m, material.nu_m) 
    material.mu_i, material.lamb_i = material.lame_from_E_nu(material.E_i, material.nu_i) 
   
    self.variational_problem(material, model)


def study() :
  
  modl0 = model(1, 3.9, 0.35, 1)
  modl0.mesh_generator(materio)

  eshelbyEF = solution(materio, modl0)

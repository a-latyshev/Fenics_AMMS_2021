import dolfin
import matplotlib.pyplot as plt
import mshr
import numpy as np
from ufl import nabla_grad

def stress(eps, lamb, mu):
     return lamb*dolfin.tr(eps)*dolfin.Identity(2) + 2*mu*eps

def strain(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

R_in = 1.0 # radius of the inclusion
R_out = 3.9 # radius of the outter matrix

E_m = 0.8 # Young's modulus for the matrix
nu_m = 0.35 # Poisson ratio for the matrix
E_i = 11.0 # Young's modulus for the inclusion
nu_i = 0.3 # Poisson ratio for the inclusion

h = 0.35*R_in # Size of elements
degreeFE = 1 # Degree of the finite elements

ONE = dolfin.Constant(1.)

MATRIX_ID = 1
INCLUSION_ID = 2

L_in = 2*np.pi*R_in # perimeter of the inclusion
L_out = 2*np.pi*R_out # perimeter of the matrix

N_in = int(L_in/h) # number of mesh points on the perimeter 
                   # of the inclusion
N_out = int(L_out/h) # number of mesh points on the perimeter 
                     # of the matrix

origin = dolfin.Point(0., 0.)

Omega_i = mshr.Circle(origin, R_in, segments=N_in)
Omega = mshr.Circle(origin, R_out, segments=N_out)
Omega.set_subdomain(MATRIX_ID, Omega-Omega_i) # we are putting tags in parts of the mesh
Omega.set_subdomain(INCLUSION_ID, Omega_i)    # we will use them later
mesh = mshr.generate_mesh(Omega, resolution=2*R_out/h)

# we define a function = 1 in the matrix and = 2 in the inclusion
subdomain_data_2d = dolfin.MeshFunction("size_t", # the function returns a positive integer
                                        mesh, # it is define over the entire mesh
                                        dim=2, # the function is defined on the cells (not edges nor vertices)
                                        value=mesh.domains() # the function value is in fact
                                                             # given by the tag we have put while creating the mesh
                                       ) 

# we need to be able to integrate over the matrix only or the inclusion only
# so in addition of the classical dolfin measure dx, we define dx(1) and dx(2)
dx = dolfin.Measure("dx", domain=mesh, subdomain_data=subdomain_data_2d)

mu_m = 0.5 * E_m / (1 + nu_m)
lamb_m = 2 * mu_m * nu_m / (1 - 2*nu_m)
mu_i = 0.5 * E_i / (1 + nu_i)
lamb_i = 2 * mu_i * nu_i / (1 - 2*nu_i)

element = dolfin.VectorElement('P', 
                               cell=mesh.ufl_cell(), 
                               degree=1, 
                               dim=mesh.geometric_dimension())
V = dolfin.FunctionSpace(mesh, element)
u = dolfin.TrialFunction(V)
v = dolfin.TestFunction(V)

# The system comprises two subdomains
# The total bilinear form is the addition of integrals on each subdomain
a_i = dolfin.inner(stress(strain(u), lamb_i, mu_i), strain(v))*dx(INCLUSION_ID)
a_m = dolfin.inner(stress(strain(u), lamb_m, mu_m), strain(v))*dx(MATRIX_ID)

bilinear_form = a_i + a_m

linear_form = dolfin.dot(dolfin.Constant((0., 0.)), v)*dx  

u_D = dolfin.Expression(('-x[1]', '-x[0]'), degree=1)

boundary_conditions = dolfin.DirichletBC(V, u_D, 'on_boundary')

# we note usol the solution
usol = dolfin.Function(V)

dolfin.solve(bilinear_form == linear_form, usol, boundary_conditions)
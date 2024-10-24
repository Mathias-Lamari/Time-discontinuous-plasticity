# -*- coding: utf-8 -*-
"""
Time-discontinuous plasticity

@author: Pierre Kerfriden, Mathias Lamari
@affiliation: Centre des materiaux, Mines Paris - PSL, CNRS, Evry, France
@date: 2023-2024

Associated video: https://www.youtube.com/watch?v=nW0KEG-W4jk
Associated data: https://doi.org/10.5281/zenodo.13823869
"""

# FEniCSx can be installed using script https://fem-on-colab.github.io/packages.html
import dolfinx
from dolfinx import mesh, fem, io, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl

from petsc4py import PETSc
import numpy as np
from mpi4py import MPI
import meshio

##########################################
################ Parameters ##############
##########################################

# Simulation parameters
t = 0. # Start time
T = 60 # End time
num_steps = 6000 # Number of time steps
dt = (T-t)/num_steps # Time step size
length = 10  # specimen located within x \in [-length, length]
load_amp = 0.002 *length/10     # 0.01 for 10 cm specimen

# Material parameters
#Elasticity
E = 200000
nu = 0.3

#Plasticity
sigma_Y = 100
H_hard = 10000

#Geometry
Mesh = "Flat_specimen_refined_01.msh"

#SaveName
root = "results_plasticity"

#Special Plasticity ? 
dPmin=0.000195

#Boundary condition
MixedCondition = False


##########################################
################ Geometry ################
##########################################
import os

def create_mesh(mesh, cell_type, prune_z=True):
    """returns meshio mesh including physucal markers for ther given type
    prune_z argument is when we want to use 2d meshes"""
    cells = mesh.get_cells_type(cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells})
    return out_mesh

proc = MPI.COMM_WORLD.rank
if proc == 0:
    msh = meshio.read(Mesh)
    # Create and save one file for the mesh, and one file for the facets 
    triangle_mesh = create_mesh(msh, "tetra", prune_z=False)
    line_mesh = create_mesh(msh, "line", prune_z=True)
    meshio.write("mesh.xdmf", triangle_mesh)
    meshio.write("mt.xdmf", line_mesh)

with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
    pass 

domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)

import os
cmd = 'rm -r ' + root
os.system(cmd)
fic = io.XDMFFile(domain.comm, root + "/res.xdmf", "w")
fic.write_mesh(domain)

##########################################
################ Spaces   ################
##########################################

gdim = domain.topology.dim
V = fem.functionspace(domain,("CG",1,(gdim,)))
W = fem.functionspace(domain, ('DG', 0))
WT = fem.functionspace(domain, ('DG', 0, (gdim,gdim)))
x = ufl.SpatialCoordinate(domain)

t_paraview = 0

##########################################
############## Dirichlet BC ##############
##########################################
def dirichlet_bcs(domain, space, disp_value):
  boundary_left = lambda x: x[0]<= (-length + 1e-8)
  boundary_right = lambda x: x[0]>= (+length - 1e-8)
  facet_dim = domain.topology.dim - 1
  boundary_left_facets = mesh.locate_entities_boundary(domain, facet_dim, boundary_left)
  boundary_right_facets = mesh.locate_entities_boundary(domain, facet_dim, boundary_right)
  u_left = fem.Constant(domain, -disp_value)
  u_right = fem.Constant(domain, disp_value)
  boundary_left_dofs = fem.locate_dofs_topological(space, facet_dim, boundary_left_facets)
  boundary_right_dofs = fem.locate_dofs_topological(space, facet_dim, boundary_right_facets)
  bc_left = fem.dirichletbc(u_left, boundary_left_dofs, space)
  bc_right = fem.dirichletbc(u_right, boundary_right_dofs, space)
  bcs = [bc_left, bc_right]
  return bcs

def dirichlet_bcs_mixed(domain, space, disp_value):
  boundary_left = lambda x: x[0]<= (-length + 1e-8)
  boundary_right = lambda x: x[0]>= (+length - 1e-8)
  facet_dim = domain.topology.dim - 1
  boundary_left_facets = mesh.locate_entities_boundary(domain, facet_dim, boundary_left)
  boundary_right_facets = mesh.locate_entities_boundary(domain, facet_dim, boundary_right)
  u_left = fem.Constant(domain, -disp_value)
  u_right = fem.Constant(domain, disp_value)
  boundary_left_dofs = fem.locate_dofs_topological(space.sub(0), facet_dim, boundary_left_facets)
  boundary_right_dofs = fem.locate_dofs_topological(space.sub(0), facet_dim, boundary_right_facets)
  bc_left = fem.dirichletbc(u_left, boundary_left_dofs, space.sub(0))
  bc_right = fem.dirichletbc(u_right, boundary_right_dofs, space.sub(0))
  bcs = [bc_left, bc_right]
  return bcs

if MixedCondition:
  disp_value_0 = 0.
  bcs = dirichlet_bcs_mixed(domain, V, disp_value = disp_value_0)
else:
  disp_value_0 = np.array((0,) * domain.geometry.dim, dtype=PETSc.ScalarType) 
  bcs = dirichlet_bcs(domain, V, disp_value = disp_value_0)

# mecanique
mu_lame = E / (2.0 * (1.0 + nu))
lambda_lame = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

def sigma_hooke(eps):
    """Return an expression for the stress given a displacement field"""
    return lambda_lame * ufl.tr(eps) * ufl.Identity(tdim) + 2.0 * mu_lame * eps 

def epsilon(v):
    """Return an expression for the stress given a displacement field"""
    return (0.5 * (ufl.grad(v) + ufl.grad(v).T))

def deviator(sigma):
    """Return an expression of stress's deviatoric stress"""
    return sigma - (1./3.)*ufl.tr(sigma) * ufl.Identity(tdim) 
    
def vonMisesStress(sigma):
    """Return von Mises stress """
    s_dev=deviator(sigma)
    return ufl.sqrt((3./2.)*ufl.inner(s_dev, s_dev))

def func(sigma, p): 
    R_p = H_hard*p
    return vonMisesStress(sigma) - sigma_Y - R_p

def dfunc_Over_dP(sigma,p):
    R_prime = H_hard 
    return -R_prime - 3*mu_lame

def normal(sigma):
    return 3./2. * deviator(sigma) / vonMisesStress(sigma)
    
# definition of a random field for use as perturbation in plastic update
quant_param = fem.Function(W)
#quant_param.interpolate(lambda x: dPmin * (0*( (x[0] + x[1])) + 1.)) 
epsx = length/20.
lengthDefect=length/120.
# To introduce defects change zero factor by a small value like in commented expression
quant_param.interpolate(lambda x: dPmin * ( 1. - 0.00 * np.exp(-(x[0]**2+(x[1]-3)**2)/lengthDefect**2) )) 
# quant_param.interpolate(lambda x: dPmin * ( 1. - 0.1*np.exp(-(x[0]**2)/lengthDefect**2) )) 
quant_param.name = "quant_param"
fic.write_function(quant_param)

def update_intern_var(Param, eps):

    sigma_star = sigma_hooke( eps - Param.eps_p_old )
    f_func = func( sigma_star , Param.p_old )
    n = normal(sigma_star)
    f_prime= dfunc_Over_dP(sigma_star, Param.p_old)
    delta_p_0 = - (1./f_prime) * func(sigma_star, Param.p_old)
    delta_p_min = quant_param 
    delta_p = ufl.conditional( ufl.ge( delta_p_0 , delta_p_min), delta_p_0, 0)
    delta_eps_p = ufl.conditional( ufl.ge(f_func, 0.), delta_p*n , 0.*n )
    eps_p = Param.eps_p_old + delta_eps_p
    delta_eps_p_dev = deviator(delta_eps_p)
    p = Param.p_old + ufl.sqrt( 2./3. * ufl.inner(delta_eps_p,delta_eps_p))

    return p, eps_p

def sigma(Param,u):
    _ , eps_p = update_intern_var( Param, epsilon(u) )
    return sigma_hooke( epsilon(u) - eps_p )

def project(v, target_func, bcs=[]):
    # Ensure we have a mesh and attach to measure
    V = target_func.function_space
    metadata = {"quadrature_degree": 2}
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)
    # Define variational problem for projection
    w = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)
    a = fem.form(ufl.inner(Pv, w) * dx)
    L = fem.form(ufl.inner(v, w) * dx)

    # Assemble linear system
    A = fem.petsc.assemble_matrix(a, bcs)
    A.assemble()
    b = fem.petsc.assemble_vector(L)
    fem.petsc.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_func.vector)
    return target_func

# Define solution variable, and interpolate initial solution for visualization in Paraview
tdim = domain.topology.dim
fdim = tdim - 1
#interns var:
class intern_var():
    def __init__( self , p_init , eps_p_init ):
        self.p_old = p_init
        self.eps_p_old = eps_p_init

p_old = fem.Function(W)
eps_p_old = fem.Function(WT)
Param = intern_var(p_old,eps_p_old)

metadata = {"quadrature_degree": 1}
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

uh = fem.Function(V)  
F = ufl.inner(sigma(Param,uh), ufl.sym(ufl.grad(v))) * dx   

# Create the Jacobian matrix, dF/du
Jacobian_form = ufl.derivative(F, uh, du)

if MixedCondition:
    disp_value =  load_amp 
    bcs = dirichlet_bcs_mixed(domain, V, disp_value = disp_value)
else:
    disp_value = np.array((load_amp,0,0,) , dtype=PETSc.ScalarType) 
    bcs = dirichlet_bcs(domain, V, disp_value = disp_value)


problem = NonlinearProblem(F, uh, bcs, Jacobian_form)

#Create matrix A and vector b and sets functions from Nonlinearproblem    
solver = NewtonSolver(domain.comm, problem)
solver.atol = 1e-8
solver.rtol = 1e-8
solver.max_it = 500
solver.convergence_criterion = "incremental"

print("___________________________________")
print("___________________________________")
print("___________________________________")

for i in range(num_steps+1):
    print("_______ load step = {0:5d} / {1:d}".format(i,num_steps))
    t += dt
    if MixedCondition:
        bcs = dirichlet_bcs_mixed(domain, V, disp_value*t)
    else:
        bcs = dirichlet_bcs(domain, V, disp_value*t)
    problem.bcs = bcs
    print("disp_value*t",disp_value*t)
    log.set_log_level(log.LogLevel.INFO)
    num_its, converged = solver.solve(uh)
    log.set_log_level(log.LogLevel.WARNING)
    uh.name = "displacement"
    fic.write_function(uh, t_paraview) 


    #Epsilon:
    eps = epsilon(uh)
    p, eps_p = update_intern_var(Param, eps)
    
    epsXX=eps[0,0]
    epsilon_proj = fem.Function(W)
    epsilon_proj.interpolate( fem.Expression( epsXX , W.element.interpolation_points() ) )
    epsilon_proj.name = "Epsilon"
    fic.write_function(epsilon_proj, t_paraview)

    #Epsilon_p:
    eps_pXX=eps_p[0,0]
    epsilon_p_proj = fem.Function(W)
    epsilon_p_proj.interpolate( fem.Expression( eps_pXX , W.element.interpolation_points() ) )
    epsilon_p_proj.name = "Epsilon_p"
    fic.write_function(epsilon_p_proj, t_paraview)

    #sigma:
    stress = sigma_hooke(eps-eps_p)
    stressXX = stress[0,0]
    stress_proj = fem.Function(W)
    stress_proj.interpolate( fem.Expression( stressXX , W.element.interpolation_points() ) )
    stress_proj.name = "Stress"
    fic.write_function(stress_proj, t_paraview)
    

    #Von Mises stress:
    von_Mises = vonMisesStress(stress)
    von_Mises_proj = fem.Function(W)
    von_Mises_proj.interpolate( fem.Expression( von_Mises , W.element.interpolation_points() ) )
    von_Mises_proj.name = "Von Mises stress"
    fic.write_function(von_Mises_proj, t_paraview)


    #Cumulative plastic strain
    p_proj = fem.Function(W)
    p_proj.interpolate( fem.Expression( p , W.element.interpolation_points() ) )
    p_proj.name = "Cumulative plastic strain"
    fic.write_function(p_proj, t_paraview)

    delta_eps_p = eps_p - Param.eps_p_old
    delta_eps_p_proj = fem.Function(WT)
    delta_eps_p_proj.interpolate( fem.Expression( delta_eps_p , WT.element.interpolation_points() ) )


    delta_p = p - Param.p_old
    delta_p_proj = fem.Function(W)
    delta_p_proj.interpolate( fem.Expression( delta_p , W.element.interpolation_points() ) )
    delta_p_proj.name = "Cumulative plastic strain increment"
    fic.write_function(delta_p_proj, t_paraview)

    Param.p_old.interpolate( fem.Expression( p , W.element.interpolation_points() ) )
    Param.eps_p_old.x.array[:] = delta_eps_p_proj.x.array[:] + Param.eps_p_old.x.array[:]
    
        
    t_paraview += 1
    print("t_paraview = ",t_paraview)
    # Scatter forward the solution vector to update ghost values (https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_elasticity.html) 

print(" It's finished!!!!!")
fic.close()

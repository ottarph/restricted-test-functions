from mpi4py import MPI
import dolfinx.fem.petsc
import numpy as np
import ufl

# mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 12)


def fluid(x, tol=1e-14):
    return x[0] >= 0.5 - tol


def interface(x, tol=1e-14):
    return np.isclose(x[0], 0.5, atol=tol)


def left(x):
    return np.isclose(x[0], 0.0, atol=1e-14)


def right(x):
    return np.isclose(x[0], 1.0, atol=1e-14)


solid_marker = 1
fluid_marker = 2
cm = mesh.topology.index_map(mesh.topology.dim)
vec = dolfinx.la.vector(cm, solid_marker, dtype=np.int32)
vec.array[:] = solid_marker
values = vec.array
values[dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, fluid)] = fluid_marker
vec.scatter_forward()

ct = dolfinx.mesh.meshtags(
    mesh, mesh.topology.dim, np.arange(len(values), dtype=np.int32), values
)

V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

bcF = dolfinx.fem.dirichletbc(
    dolfinx.fem.Constant(mesh, 1.2), dolfinx.fem.locate_dofs_geometrical(V, right), V
)
bcS = dolfinx.fem.dirichletbc(
    dolfinx.fem.Constant(mesh, 0.3), dolfinx.fem.locate_dofs_geometrical(V, left), V
)
bcs = [bcF, bcS]

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
interface_facets = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim - 1, interface)
dofs_interface = dolfinx.fem.locate_dofs_topological(
    V, mesh.topology.dim - 1, interface_facets
)
bc_deactivate = dolfinx.fem.dirichletbc(
    dolfinx.fem.Constant(mesh, 0.0), dofs_interface, V
)


dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
dxF = dx(fluid_marker)
dxS = dx(solid_marker)

kernel = ufl.inner(ufl.grad(u), ufl.grad(v))

kernelS = kernel * dxS
compiled_solid = dolfinx.fem.form(kernelS)
As = dolfinx.fem.petsc.assemble_matrix(compiled_solid, bcs=bcs)
As.assemble()
compiled_fluid = dolfinx.fem.form(kernel * dxF)

deactivate_bc = True
if deactivate_bc:
    bcs_deac = [bcF, bcS, bc_deactivate]
else:
    bcs_deac = [bcF, bcS]

Af = dolfinx.fem.petsc.assemble_matrix(compiled_fluid, bcs=bcs_deac, diag=0.0)
dolfinx.cpp.fem.petsc.insert_diagonal(
    Af, V._cpp_object, [bc._cpp_object for bc in bcs], 1.0
)
Af.assemble()
Af.setValuesLocal([7], [6], [-12.0])
Af.assemble()

A = As + Af
print("As=", As[:, :])
print("Af=", Af[:, :])
print("A=", A[:, :])

x = ufl.SpatialCoordinate(mesh)
f_S = 2  # -(x[0] ** 2)
f_F = dolfinx.fem.Constant(mesh, 0.0)  # * x[0] ** 2
Ls = ufl.inner(f_S, v) * dxS
Lf = ufl.inner(f_F, v) * dxF

bs = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(Ls))
dolfinx.fem.petsc.apply_lifting(bs, [compiled_solid], bcs=[bcs])
[bc.set(bs.array_w) for bc in [bcS]]
bf = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(Lf))
# Zero out the disappearing basis function
bc_deactivate.set(bf.array_w, alpha=0.0)
dolfinx.fem.petsc.apply_lifting(bf, [compiled_fluid], bcs=[bcs])
[bc.set(bf.array_w) for bc in [bcF]]

b = bs + bf

from petsc4py import PETSc

ksp = PETSc.KSP().create(mesh.comm)
ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")
ksp.setErrorIfNotConverged(True)
ksp.setOperators(A)
x = dolfinx.fem.Function(V)
ksp.solve(b, x.x.petsc_vec)

fluid_mesh, fluid_map, _, _ = dolfinx.mesh.create_submesh(
    mesh, mesh.topology.dim, ct.find(fluid_marker)
)
Vf = dolfinx.fem.functionspace(fluid_mesh, ("Lagrange", 1))
uf = dolfinx.fem.Function(Vf)
num_fluid_cells = fluid_mesh.topology.index_map(fluid_mesh.topology.dim).size_local
parent_cells = fluid_map.sub_topology_to_topology(
    np.arange(num_fluid_cells, dtype=np.int32), False
)
uf.interpolate(
    x, cells0=parent_cells, cells1=np.arange(num_fluid_cells, dtype=np.int32)
)
uf.x.scatter_forward()
if_submesh = dolfinx.mesh.locate_entities(
    fluid_mesh, fluid_mesh.topology.dim - 1, interface
)
if_fluid_sm = dolfinx.mesh.locate_entities(
    fluid_mesh, fluid_mesh.topology.dim - 1, interface
)
bc_F_deac = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, 0.0), if_submesh, Vf)
bc_F_deac.set(uf.x.array, alpha=0.0)

solid_mesh, solid_map, _, _ = dolfinx.mesh.create_submesh(
    mesh, mesh.topology.dim, ct.find(solid_marker)
)

Vs = dolfinx.fem.functionspace(solid_mesh, ("Lagrange", 1))
us = dolfinx.fem.Function(Vs)
num_solid_cells = solid_mesh.topology.index_map(solid_mesh.topology.dim).size_local
us.interpolate(
    x, cells0=parent_cells, cells1=np.arange(num_solid_cells, dtype=np.int32)
)
us.x.scatter_forward()

with dolfinx.io.XDMFFile(mesh.comm, "output/solution.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(x)

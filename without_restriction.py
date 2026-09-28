import dolfinx.fem.petsc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scifem
import ufl
from mpi4py import MPI
from petsc4py import PETSc

mpl.rcParams["svg.hashsalt"] = "without-restricted-test-functions"

N = 36
mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, N)

u_f_strength = 0.25

degree = 1
element_specification = ("Lagrange", 1)


def u_s_exact(x):
    return -(x[0] - 0.5) * (x[0] - 0.5)


def u_f_exact(x):
    return (
        u_f_strength * 16.0 * (x[0] - 0.75) * (x[0] - 0.75)
        - u_f_strength * 16.0 * 0.25**2
    )


def u_exact(x, tol=1e-14):
    return np.where(x[0] >= 0.5 - tol, u_f_exact(x), u_s_exact(x))


alpha = dolfinx.fem.Constant(mesh, 1.0e0)


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

V = dolfinx.fem.functionspace(mesh, element_specification)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

bcF = dolfinx.fem.dirichletbc(
    dolfinx.fem.Constant(mesh, u_f_exact([1.0])),
    dolfinx.fem.locate_dofs_geometrical(V, right),
    V,
)
bcS = dolfinx.fem.dirichletbc(
    dolfinx.fem.Constant(mesh, u_s_exact([0.0])),
    dolfinx.fem.locate_dofs_geometrical(V, left),
    V,
)
bcs = [bcF, bcS]

interface_facets = scifem.find_interface(ct, solid_marker, fluid_marker)

total_interface_facets_found = MPI.COMM_WORLD.allreduce(
    interface_facets.size, op=MPI.SUM
)

assert total_interface_facets_found > 0, "Interface not found."
dofs_interface = dolfinx.fem.locate_dofs_topological(
    V, mesh.topology.dim - 1, interface_facets
)

ft = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, interface_facets, 10)

dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
dxF = dx(fluid_marker)
dxS = dx(solid_marker)
dS = ufl.Measure("dS", domain=mesh, subdomain_data=ft)
dS_interface = dS(ft.values[0])

kernel = ufl.inner(ufl.grad(u), ufl.grad(v))

kernelS = kernel * dxS
compiled_solid = dolfinx.fem.form(kernelS)
As = dolfinx.fem.petsc.assemble_matrix(compiled_solid, bcs=bcs)
As.assemble()

kernelF = alpha * kernel * dxF
kernelF -= (alpha * ufl.dot(ufl.grad(u), ufl.FacetNormal(mesh)) * v)("-") * dS_interface
compiled_fluid = dolfinx.fem.form(kernelF)
Af = dolfinx.fem.petsc.assemble_matrix(compiled_fluid, bcs=bcs)
Af.assemble()

A = As.duplicate(copy=True)  # once
A.axpy(1.0, Af)

if MPI.COMM_WORLD.size == 1:
    with np.printoptions(precision=2, linewidth=140):
        print("\nAs = \n", As[:, :])
        print("\nAf =\n", Af[:, :])
        print("\nA =\n", A[:, :])

        interface_x = V.tabulate_dof_coordinates()[dofs_interface[0]]
        print(f"{interface_x[0] = :.2f}")


f_S = dolfinx.fem.Constant(mesh, 2.0)
f_F = dolfinx.fem.Constant(mesh, -2.0 * 16.0 * u_f_strength)
Ls = ufl.inner(f_S, v) * dxS
Lf = alpha * ufl.inner(f_F, v) * dxF

bs = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(Ls))
dolfinx.fem.petsc.apply_lifting(bs, [compiled_solid], bcs=[bcs])
bs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
[bc.set(bs.array_w) for bc in [bcS]]
bf = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(Lf))
dolfinx.fem.petsc.apply_lifting(bf, [compiled_fluid], bcs=[bcs])
# Scatter reverse should happen after bc-treatment, which is local.
bf.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

[bc.set(bf.array_w) for bc in [bcF]]

b = bs + bf
# Final forward scatter, probably not needed.
# b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


ksp = PETSc.KSP().create(mesh.comm)
ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")
ksp.setErrorIfNotConverged(True)
ksp.setOperators(A)
x = dolfinx.fem.Function(V)
ksp.solve(b, x.x.petsc_vec)
x.x.scatter_forward()


writer = dolfinx.io.VTXWriter(mesh.comm, "output/solution_without_restriction.bp", [x])

x_mesh = ufl.SpatialCoordinate(mesh)
error_expression = dolfinx.fem.Expression(
    x - ufl.conditional(x_mesh[0] <= 0.5, u_s_exact(x_mesh), u_f_exact(x_mesh)),
    x.function_space.element.interpolation_points,
)
e = dolfinx.fem.Function(x.function_space, name="error")
L2_error_form = dolfinx.fem.form(e * e * ufl.dx)

tt = x.function_space.tabulate_dof_coordinates()[:, 0]
dof_sorting = np.argsort(tt)


# fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig, axs = plt.subplots(3, 1, figsize=(6, 12))


ax_sol = axs[0]
ax_err = axs[1]
ax_conv = axs[2]

ax_sol.plot(tt[dof_sorting], u_exact(tt[[dof_sorting]]), "k:", label="exact")

ax_sol.axvline(
    x=tt[dofs_interface[0]], color="black", alpha=0.4, lw=0.3, label="interface"
)
ax_err.axhline(y=0.0, color="black", alpha=0.6, lw=0.5)
ax_err.axvline(
    x=tt[dofs_interface[0]], color="black", alpha=0.4, lw=0.3, label="interface"
)

alpha_vals = np.array([1.0e0, 1.0e-1, 1.0e-2, 1.0e-3])
L2_errors = np.zeros_like(alpha_vals)
for i, alpha_val in enumerate(alpha_vals):
    alpha.value = alpha_val

    As.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(As, compiled_solid, bcs=bcs)
    As.assemble()

    Af.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(Af, compiled_fluid, bcs=bcs)
    Af.assemble()

    # after re-assembling As and Af:
    As.copy(A)  # A <- As (copies values into existing A)
    A.axpy(1.0, Af)  # A <- A + Af

    bs.zeroEntries()
    bs = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(Ls))
    dolfinx.fem.petsc.apply_lifting(bs, [compiled_solid], bcs=[bcs])
    bs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    [bc.set(bs.array_w) for bc in [bcS]]
    bf.zeroEntries()
    bf = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(Lf))
    dolfinx.fem.petsc.apply_lifting(bf, [compiled_fluid], bcs=[bcs])
    # Scatter reverse should happen after bc-treatment, which is local.
    bf.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    [bc.set(bf.array_w) for bc in [bcF]]

    b.waxpy(1.0, bs, bf)  # b <- 1.0 * bs + bf

    ksp.solve(b, x.x.petsc_vec)
    x.x.scatter_forward()

    e.interpolate(error_expression)
    L2_error = np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(L2_error_form)))
    print(f"{alpha_val = :.1e}, {L2_error = :.2e}")
    L2_errors[i] = L2_error

    ax_sol.plot(
        tt[dof_sorting], x.x.array[dof_sorting], "-", label=rf"$\alpha={alpha_val:.1e}$"
    )
    ax_err.plot(
        tt[dof_sorting],
        x.x.array[dof_sorting] - u_exact(tt[[dof_sorting]]),
        "-",
        label=rf"$\alpha={alpha_val:.1e}$",
    )

    writer.write(i)

writer.close()

ax_conv.loglog(alpha_vals, L2_errors, "o-")
ax_conv.set_xlabel(r"$\alpha$")
ax_conv.set_ylabel(r"$\|u_h - u\|_{L^2}$")

ax_sol.legend()
ax_err.legend()

fig.suptitle(f"{N = }, {degree = }")
ax_sol.set_title("solution")
ax_err.set_title("error")
ax_conv.set_title(r"$L^2$ against $\alpha$")

fig.savefig("output/solution_without_restriction.svg", metadata={"Date": None})
plt.show()

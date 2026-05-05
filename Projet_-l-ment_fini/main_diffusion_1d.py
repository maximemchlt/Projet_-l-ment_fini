#---------------------------------------MODIFIE---------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import gmsh

from gmsh_utils import gmsh_init, gmsh_finalize, build_1d_mesh, \
                       prepare_quadrature_and_basis, get_jacobians, \
                       end_dofs_from_nodes
from stiffness import assemble_stiffness_and_rhs, assemble_robin
from mass import assemble_mass
from dirichlet import theta_step_robin
from plot_utils import setup_interactive_figure, plot_fe_solution_high_order

# ── Paramètres matériau ──────────────────────────────────────────────────────
MATERIALS = {
    "steel":    {"rho": 7800, "cp": 500, "k": 50},
    "titanium": {"rho": 4500, "cp": 520, "k": 22},
}

# ── Paramètres fluide de trempe ──────────────────────────────────────────────
FLUIDS = {
    "water": {"h": 3000, "T_inf": 20.0},
    "oil":   {"h":  500, "T_inf": 60.0},
}

# ── Choix de la simulation ───────────────────────────────────────────────────
mat    = MATERIALS["steel"]
fluid  = FLUIDS["water"]

T0     = 1000.0   # température initiale [°C]
L      = 0.004    # demi-épaisseur de la lame [m]
dt     = 0.002    # pas de temps [s]
nsteps = 5000     # nombre de pas
theta  = 1.0      # 1=Euler implicite
order  = 1        # ordre des éléments

rho, cp, k = mat["rho"], mat["cp"], mat["k"]
h, T_inf   = fluid["h"], fluid["T_inf"]

# ── Maillage 1D ──────────────────────────────────────────────────────────────
gmsh_init("quench_1d")

_, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags = \
    build_1d_mesh(L=L, cl1=0.0002, cl2=0.0002, order=order)

unique_tags = np.unique(elemNodeTags)
num_dofs    = len(unique_tags)
max_tag     = int(np.max(nodeTags))
tag_to_dof  = np.full(max_tag + 1, -1, dtype=int)
for i, tag in enumerate(unique_tags):
    tag_to_dof[int(tag)] = i

xi, w, N, gN     = prepare_quadrature_and_basis(elemType, order)
jac, det, coords = get_jacobians(elemType, xi)

# ── Assemblage volume ─────────────────────────────────────────────────────────
def kappa(x): return float(k)
def source(x): return 0.0

K_vol, F_vol = assemble_stiffness_and_rhs(
    elemTags, elemNodeTags, jac, det, coords, w, N, gN,
    kappa, source, tag_to_dof
)
M_vol = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)

# ── Mise à l'échelle par rho*cp ───────────────────────────────────────────────
M_vol = M_vol * (rho * cp)

# ── Condition de Robin sur le bord gauche x=0 ────────────────────────────────
left_dof, right_dof = end_dofs_from_nodes(nodeCoords)

K_vol[left_dof, left_dof] += h
F_vol[left_dof]            += h * T_inf

K = K_vol.tocsr()
M = M_vol.tocsr()

# ── Condition initiale ────────────────────────────────────────────────────────
U = np.full(num_dofs, T0, dtype=float)

# ── Boucle en temps ───────────────────────────────────────────────────────────
fig, ax = setup_interactive_figure(xlim=(0, L), ylim=(T_inf - 10, T0 + 10))

try:
    for step in range(nsteps):
        U = theta_step_robin(M, K, F_vol, U, dt=dt, theta=theta)

        if step % 20 == 0:
            ax.clear()
            ax.set_xlim(0, L)
            ax.set_ylim(T_inf - 10, T0 + 10)
            plot_fe_solution_high_order(elemType, elemNodeTags, nodeCoords, U,
                                        M=100, ax=ax)
            ax.axhline(T_inf, color='blue',  linestyle='--', label='T bain')
            ax.axhline(T0,    color='red',   linestyle='--', label='T initiale')
            ax.set_title(f"t = {step*dt:.3f} s — acier dans eau")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("T [°C]")
            ax.legend()
            ax.grid(True)
            plt.pause(0.01)
finally:
    gmsh_finalize()

plt.ioff()
plt.show()

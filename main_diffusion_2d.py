#-----------------------------------------------------MODIFIE-------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import gmsh

from gmsh_utils import gmsh_init, gmsh_finalize, \
                       prepare_quadrature_and_basis, get_jacobians
from blade_geometry import create_blade_section
from stiffness import assemble_stiffness_and_rhs, assemble_robin
from mass import assemble_mass
from dirichlet import theta_step_robin
from plot_utils import plot_fe_solution_2d

# ── Paramètres matériau et fluide ────────────────────────────────────────────
MATERIALS = {
    "steel":    {"rho": 7800, "cp": 500, "k": 50},
    "titanium": {"rho": 4500, "cp": 520, "k": 22},
}
FLUIDS = {
    "water": {"h": 3000, "T_inf": 20.0},
    "oil":   {"h":  500, "T_inf": 60.0},
}

mat   = MATERIALS["steel"]
fluid = FLUIDS["water"]

rho, cp, k = mat["rho"], mat["cp"], mat["k"]
h, T_inf   = fluid["h"], fluid["T_inf"]

T0     = 1000.0   # température initiale [°C]
a      = 0.02     # demi-grand axe du losange [m]
b      = 0.004    # demi-petit axe du losange [m]
cl     = 0.001    # taille de maille [m]
dt     = 0.01     # pas de temps [s]
nsteps = 3000     # nombre de pas
theta  = 1.0      # 1=Euler implicite
order  = 1        # ordre des éléments

def main():
    

    gmsh_init("quench_2d")

    (elemType,     nodeTags,        nodeCoords,
    elemTags,     elemNodeTags,
    bnd_elemType, bnd_elemTags,    bnd_elemNodeTags,
    bnd_entityTag, surf_tag) = create_blade_section(a=a, b=b, cl=cl, order=order)

    unique_tags = np.unique(elemNodeTags)
    num_dofs    = len(unique_tags)
    max_tag     = int(np.max(nodeTags))
    tag_to_dof  = np.full(max_tag + 1, -1, dtype=int)
    for i, tag in enumerate(unique_tags):
        tag_to_dof[int(tag)] = i
    # Note: We create a mapping from Gmsh node tags to our dof indices, and we also store the coordinates of the dofs. This will be useful for assembling the system and for plotting.

    xi,  w,  N,  gN  = prepare_quadrature_and_basis(elemType,     order)  #-> elemtype -> triangles
    jac, det, coords  = get_jacobians(elemType, xi, tag=-1)   #-> tag = surf_tag pour obtenir les jacobiennes uniquement sur la surface  -> Robin

    xib, wb, Nb, gNb  = prepare_quadrature_and_basis(bnd_elemType, order) #->bnd_elemtype -> lignes
    jacb, detb, coordsb = get_jacobians(bnd_elemType, xib, tag=-1)  #->tag=-1 pour obtenir les jacobiennes sur tous les éléments de frontière   -> Neumann


    # ── Assemblage rigidité ──────────────────────────────────────────────────────
    K_lil, F = assemble_stiffness_and_rhs(
        elemTags, elemNodeTags,
        jac, det, coords,
        w, N, gN,
        lambda x: float(k),
        lambda x: 0.0,
        tag_to_dof
    )

    # ── Assemblage masse ─────────────────────────────────────────────────────────
    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
    M = M_lil.tocsr() * (rho * cp)

    # ── Assemblage Robin ─────────────────────────────────────────────────────────
    K_lil, F = assemble_robin(
        K_lil, F,
        bnd_elemTags, bnd_elemNodeTags,
        detb, coordsb,
        wb, Nb,
        h, T_inf,
        tag_to_dof
    )

    K = K_lil.tocsr()

    # ── Condition initiale ───────────────────────────────────────────────────────
    U = np.full(num_dofs, T0, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.ion()


    for step in range(nsteps):
        U = theta_step_robin(M, K, F, U, dt=dt, theta=theta)

        if step % 10 == 0:
            ax.clear()
            plot_fe_solution_2d(
                elemNodeTags=elemNodeTags,
                nodeTags=nodeTags,
                nodeCoords=nodeCoords,
                U=U,
                tag_to_dof=tag_to_dof,
                vmin=T_inf,
                vmax=T0,
                show_mesh=False,
                ax=ax
            )
            T_mean = U.mean()
            ax.set_title(f"Trempe acier/eau - t = {step*dt:.1f} s    T_moy = {T_mean:.0f} degC")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.axis('equal')
            ax.set_xlim(-a * 1.15, a * 1.15)
            ax.set_ylim(-b * 4,    b * 4)
            fig.tight_layout()
            plt.pause(0.01)

    gmsh_finalize()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()

# dirichlet.py
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.linalg import det


from stiffness import assemble_robin


def apply_dirichlet_by_reduction(K, F, dirichlet_dofs, dirichlet_values):
    """
    Reduce linear system with strong Dirichlet by elimination:
        K u = F, u_D fixed
    -> K_FF u_F = F_F - K_FD u_D

    K can be sparse (csr/lil/etc).
    """
    dirichlet_dofs = np.asarray(dirichlet_dofs, dtype=int)
    dirichlet_values = np.asarray(dirichlet_values, dtype=float)

    n = len(F)
    mask = np.ones(n, dtype=bool)
    mask[dirichlet_dofs] = False
    free_dofs = np.nonzero(mask)[0]

    K_FF = K[free_dofs, :][:, free_dofs]
    K_FD = K[free_dofs, :][:, dirichlet_dofs]

    F_F = F[free_dofs]
    F_red = F_F - K_FD.dot(dirichlet_values)

    U_full = np.zeros(n, dtype=float)
    U_full[dirichlet_dofs] = dirichlet_values

    return K_FF, F_red, free_dofs, U_full


def solve_dirichlet(K, F, dirichlet_dofs, dirichlet_values):
    K_red, F_red, free_dofs, U_full = apply_dirichlet_by_reduction(
        K, F, dirichlet_dofs, dirichlet_values
    )
    U_free = spsolve(K_red.tocsr(), F_red)
    U_full[free_dofs] = U_free
    U_full[dirichlet_dofs] = dirichlet_values
    return U_full


def theta_step(M, K, F_n, F_np1, U_n, dt, theta, dirichlet_dofs, dir_vals_np1):
    """
    One theta-scheme step for:
        M u_t + K u = F(t)

    (M + theta dt K) u^{n+1} = (M - (1-theta) dt K) u^n + dt*(theta F^{n+1} + (1-theta) F^n)
    with Dirichlet enforced at time n+1.
    """
    A = M + theta * dt * K
    B = M - (1.0 - theta) * dt * K
    rhs = B.dot(U_n) + dt * (theta * F_np1 + (1.0 - theta) * F_n)

    A_red, rhs_red, free_dofs, U_full = apply_dirichlet_by_reduction(
        A, rhs, dirichlet_dofs, dir_vals_np1
    )
    U_free = spsolve(A_red.tocsr(), rhs_red)
    U_full[free_dofs] = U_free
    U_full[dirichlet_dofs] = dir_vals_np1
    return U_full


# ajout de 2 fonctions (theta_step_robin et theta_step_robin_variable_h) pour le cas de conditions de Robin

def theta_step_robin(M, K, F, U_n, dt, theta):
    """
    schema de theta sans dirivhelt
    pour la condition de robin pure 
    return U_np1 : solution au temps n+1
     (M + theta dt K) u^{n+1} = (M - (1-theta) dt K) u^n + dt*(theta F^{n+1} + (1-theta) F^n)
    """
    A = M + theta * dt * K
    B = M - (1.0 - theta) * dt * K
    rhs = B.dot(U_n) + dt * F

    U_np1 = spsolve(A.tocsr(), rhs)
    return U_np1

def theta_step_robin_variable_h(M, K_vol, F_vol, U_n, dt, theta,
                                bnd_elemTags, bnd_elemNodeTags, detb, coordsb,
                                wb, Nb, h_fun, T_inf, tag_to_dof, surf_dofs = None):
    """
    schemas theta avec h variable en temperature pour la condition de robin
    param
    M : matrice de masse
    K_vol : matrice de rigidité volumique (sans contribution de Robin)
    F_vol : second membre volumique (sans contribution de Robin)
    U_n : solution au temps n
    dt : pas de temps
    theta : paramètre du schéma de theta
    bnd_elemTags, bnd_elemNodeTags, detb, coordsb, wb, Nb : données de quadrature et géométriques pour la frontière
    h_fun : fonction de convection (peut dépendre de la température)
    T_inf : température extérieure (pour la condition de Robin)
    tag_to_dof : mapping des tags de nœuds vers les dofs
    surf_dofs : liste des dofs situés sur la surface (optionnel, peut être déterminé à partir de bnd_elemNodeTags si non fourni)
    return U_np1 : solution au temps n+1
    (M + theta dt (K_vol + K_robin)) u^{n+1} = (M - (1-theta) dt (K_vol + K_robin)) u^n + dt*(theta F^{n+1} + (1-theta) F^n)
    """
    if surf_dofs is None:
        surf_dofs = np.unique(tag_to_dof[np.asarray(bnd_elemNodeTags, dtype=int)])
        surf_dofs = surf_dofs[surf_dofs >= 0]  # filtrer les dofs valides

    T_bnd_mean = np.mean(U_n[surf_dofs])  # moyenne des températures sur les dofs de surface
    H_eff = float(h_fun(T_bnd_mean))  # évaluer h à la température moyenne de la surface

    # assembler la contribution de Robin avec h_eff
    K_local, F_local = assemble_robin(
        K_vol, F_vol, bnd_elemTags, bnd_elemNodeTags,
        detb, coordsb, wb, Nb, H_eff, T_inf, tag_to_dof
    )
    K_csr = K_local.tocsr()
    A = M + theta * dt * K_csr
    B = M - (1.0 - theta) * dt * K_csr
    rhs = B.dot(U_n) + dt * F_local

    U_np1 = spsolve(A.tocsr(), rhs)
    return U_np1
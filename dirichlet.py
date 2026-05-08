# dirichlet.py
import numpy as np
from scipy.sparse.linalg import spsolve
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


def theta_step_robin(M, K, F, U_n, dt, theta):
    """
    Schéma theta sans Dirichlet — pour la condition de Robin pure.
    F contient déjà la contribution h*T_inf assemblée dans K et F.
    """
    # F est supposé constant en temps (assemblé une seule fois avant la boucle temporelle).
    A   = M + theta * dt * K    #matrice du système linéaire à résoudre (Theta = 1)
    B   = M - (1.0 - theta) * dt * K   #B = M si theta = 1
    rhs = B.dot(U_n) + dt * F

    return spsolve(A.tocsr(), rhs)


def theta_step_robin_variable_h(M, K_vol, F_vol,
                                U_n, dt, theta,
                                bnd_elemTags, bnd_elemNodeTags,
                                detb, coordsb, wb, Nb,
                                h_fun, T_inf, tag_to_dof,
                                surf_dofs=None):
    """
    Schéma theta avec h variable en température.

    À chaque pas :
      1. Calcule T_bnd = moyenne des températures sur les nœuds frontière
      2. Évalue h_eff = h_fun(T_bnd)
      3. Réassemble (K_vol + K_robin(h_eff)) et (F_vol + F_robin(h_eff))
      4. Résout (M + theta*dt*K_total) U_{n+1} = (M - (1-theta)*dt*K_total) U_n
                                                 + dt*F_total

    Paramètres
    ----------
    M, K_vol, F_vol : matrice de masse et terme volumique (constants en temps)
    U_n             : champ au pas n
    h_fun           : callable h(T) -> W/m²K
    surf_dofs       : optionnel, indices DOF des nœuds de surface ; sinon
                      reconstruit depuis bnd_elemNodeTags via tag_to_dof.

    Retourne
    --------
    U_np1 : champ au pas n+1
    h_eff : valeur scalaire de h utilisée à ce pas (pour log)
    """
    if surf_dofs is None:
        surf_dofs = np.unique(tag_to_dof[np.asarray(bnd_elemNodeTags, dtype=int)])
        surf_dofs = surf_dofs[surf_dofs >= 0]

    T_bnd_mean = float(U_n[surf_dofs].mean())
    h_eff      = float(h_fun(T_bnd_mean))

    # Réassemblage Robin sur une copie de K_vol et F_vol
    K_local = K_vol.copy()
    F_local = F_vol.copy()
    K_local, F_local = assemble_robin(
        K_local, F_local,
        bnd_elemTags, bnd_elemNodeTags,
        detb, coordsb,
        wb, Nb,
        h_eff, T_inf, tag_to_dof
    )

    K_csr = K_local.tocsr()
    A   = M + theta * dt * K_csr
    B   = M - (1.0 - theta) * dt * K_csr
    rhs = B.dot(U_n) + dt * F_local

    U_np1 = spsolve(A.tocsr(), rhs)
    return U_np1, h_eff
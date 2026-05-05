# stiffness.py
# donnée 
import numpy as np
from scipy.sparse import lil_matrix, coo_matrix


def assemble_stiffness_and_rhs(elemTags, conn, jac, det, xphys,
                                w, N, gN, kappa_fun, rhs_fun, tag_to_dof):
    """
    K_ij = ∫ kappa ∇N_i · ∇N_j dΩ
    F_i  = ∫ f N_i dΩ
    """
    ne   = len(elemTags)
    ngp  = len(w)
    nloc = int(len(conn) // ne)
    nn   = int(np.max(tag_to_dof) + 1)

    det   = np.asarray(det,   dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac   = np.asarray(jac,   dtype=np.float64).reshape(ne, ngp, 3, 3)
    conn  = np.asarray(conn,  dtype=np.int64  ).reshape(ne, nloc)
    N     = np.asarray(N,     dtype=np.float64).reshape(ngp, nloc)
    gN    = np.asarray(gN,    dtype=np.float64).reshape(ngp, nloc, 3)

    # Évaluation de kappa et f aux points de Gauss (boucle légère sur ne×ngp)
    kappa = np.fromiter(
        (kappa_fun(xphys[e, g]) for e in range(ne) for g in range(ngp)),
        dtype=float, count=ne * ngp).reshape(ne, ngp)
    f_val = np.fromiter(
        (rhs_fun(xphys[e, g]) for e in range(ne) for g in range(ngp)),
        dtype=float, count=ne * ngp).reshape(ne, ngp)

    # Jacobiens inverses : (ne, ngp, 3, 3)
    invjac = np.linalg.inv(jac)

    # Gradients physiques : gradN[e,g,a,i] = Σ_j invjac[e,g,i,j] * gN[g,a,j]
    gradN = np.einsum('egij,gaj->egai', invjac, gN)   # (ne, ngp, nloc, 3)

    # Matrices de rigidité élémentaires
    # Ke[e,a,b] = Σ_g w[g] * kappa[e,g] * det[e,g] * Σ_i gradN[e,g,a,i]*gradN[e,g,b,i]
    Ke = np.einsum('g,eg,eg,egai,egbi->eab', w, kappa, det, gradN, gradN)  # (ne, nloc, nloc)

    # Vecteurs élémentaires de charge
    Fe = np.einsum('g,eg,ga,eg->ea', w, f_val, N, det)   # (ne, nloc)

    # Assemblage global par COO (gère les doublons par sommation)
    dofs   = tag_to_dof[conn]                                          # (ne, nloc)
    rows_e = np.broadcast_to(dofs[:, :, None], (ne, nloc, nloc))      # (ne, nloc, nloc)
    cols_e = np.broadcast_to(dofs[:, None, :], (ne, nloc, nloc))      # (ne, nloc, nloc)

    K = coo_matrix((Ke.ravel(), (rows_e.ravel(), cols_e.ravel())),
                   shape=(nn, nn)).tolil()

    F = np.zeros(nn, dtype=np.float64)
    np.add.at(F, dofs.ravel(), Fe.ravel())

    return K, F


def assemble_rhs_neumann(F, elemTags, conn, jac, det, xphys,
                         w, N, gN, g_neu_fun, tag_to_dof):
    ne   = len(elemTags)
    ngp  = len(w)
    nloc = int(len(conn) // ne)

    det   = np.asarray(det,   dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    conn  = np.asarray(conn,  dtype=np.int64  ).reshape(ne, nloc)
    N     = np.asarray(N,     dtype=np.float64).reshape(ngp, nloc)

    g_vals = np.fromiter(
        (g_neu_fun(xphys[e, g]) for e in range(ne) for g in range(ngp)),
        dtype=float, count=ne * ngp).reshape(ne, ngp)

    Fe = np.einsum('g,eg,ga,eg->ea', w, g_vals, N, det)   # (ne, nloc)

    dofs = tag_to_dof[np.asarray(conn, dtype=np.int64)]
    np.add.at(F, dofs.ravel(), Fe.ravel())
    return F


def assemble_robin(K, F, elemTags, conn, det, _xphys, w, N, h, T_inf, tag_to_dof):
    """
    Condition de Robin :  -k ∂T/∂n = h(T - T_inf)
    Ajoute  h ∫ N_i N_j dΓ  à K  et  h T_inf ∫ N_i dΓ  à F.
    """
    ne   = len(elemTags)
    ngp  = len(w)
    nloc = int(len(conn) // ne)
    nn   = K.shape[0]

    det  = np.asarray(det,  dtype=np.float64).reshape(ne, ngp)
    conn = np.asarray(conn, dtype=np.int64  ).reshape(ne, nloc)
    N    = np.asarray(N,    dtype=np.float64).reshape(ngp, nloc)

    # Ke_robin[e,a,b] = h Σ_g w[g] N[g,a] N[g,b] det[e,g]
    Ke = h * np.einsum('g,ga,gb,eg->eab', w, N, N, det)    # (ne, nloc, nloc)
    # Fe_robin[e,a]   = h T_inf Σ_g w[g] N[g,a] det[e,g]
    Fe = h * T_inf * np.einsum('g,ga,eg->ea', w, N, det)   # (ne, nloc)

    dofs   = tag_to_dof[conn]
    rows_e = np.broadcast_to(dofs[:, :, None], (ne, nloc, nloc))
    cols_e = np.broadcast_to(dofs[:, None, :], (ne, nloc, nloc))

    K_robin = coo_matrix((Ke.ravel(), (rows_e.ravel(), cols_e.ravel())),
                          shape=(nn, nn)).tocsr()
    K = (K.tocsr() + K_robin).tolil()

    np.add.at(F, dofs.ravel(), Fe.ravel())
    return K, F

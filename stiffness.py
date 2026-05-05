# stiffness.py
# donnée 
import numpy as np
from scipy.sparse import lil_matrix, coo_matrix


def assemble_stiffness_and_rhs(elemTags, conn, jac, det, xphys, w, N, gN, kappa_fun, rhs_fun, tag_to_dof):
    """
    Assemble global stiffness matrix and load vector for:
        -d/dx (kappa(x) du/dx) = f(x)

    K_ij = ∫ kappa * grad(N_i)·grad(N_j) dx
    F_i  = ∫ f * N_i dx

    Notes:
    - gmsh gives gN in reference coordinates; we map with inv(J).
    - For 1D line embedded in 3D, gmsh provides a 3x3 Jacobian; we keep the same approach.

    Returns
    -------
    K : lil_matrix (nn x nn)
    F : ndarray (nn,)
    """
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)
    nn = int(np.max(tag_to_dof) + 1)

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac = np.asarray(jac, dtype=np.float64).reshape(ne, ngp, 3, 3)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    gN = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3)

    K = lil_matrix((nn, nn), dtype=np.float64)
    F = np.zeros(nn, dtype=np.float64)

    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]
        for g in range(ngp):
            xg = xphys[e, g]
            wg = w[g]
            detg = det[e, g]
            invjacg = np.linalg.inv(jac[e, g])

            kappa_g = float(kappa_fun(xg))
            f_g = float(rhs_fun(xg))

            for a in range(nloc):
                Ia = int(dof_indices[a])
                F[Ia] += wg * f_g * N[g, a] * detg

                gradNa = invjacg @ gN[g, a]
                for b in range(nloc):
                    Ib = int(dof_indices[b])
                    gradNb = invjacg @ gN[g, b]
                    K[Ia, Ib] += wg * kappa_g * float(np.dot(gradNa, gradNb)) * detg

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

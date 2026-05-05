# mass.py
import numpy as np
from scipy.sparse import coo_matrix


def assemble_mass(elemTags, conn, det, w, N, tag_to_dof):
    """
    M_ij = Σ_e ∫_e N_i N_j dΩ
    """
    ne   = len(elemTags)
    ngp  = len(w)
    nloc = int(len(conn) // ne)
    nn   = int(np.max(tag_to_dof) + 1)

    det  = np.asarray(det,  dtype=np.float64).reshape(ne, ngp)
    conn = np.asarray(conn, dtype=np.int64  ).reshape(ne, nloc)
    N    = np.asarray(N,    dtype=np.float64).reshape(ngp, nloc)

    # Me[e,a,b] = Σ_g w[g] N[g,a] N[g,b] det[e,g]
    Me = np.einsum('g,ga,gb,eg->eab', w, N, N, det)   # (ne, nloc, nloc)

    dofs   = tag_to_dof[conn]
    rows_e = np.broadcast_to(dofs[:, :, None], (ne, nloc, nloc))
    cols_e = np.broadcast_to(dofs[:, None, :], (ne, nloc, nloc))

    return coo_matrix((Me.ravel(), (rows_e.ravel(), cols_e.ravel())),
                      shape=(nn, nn)).tolil()

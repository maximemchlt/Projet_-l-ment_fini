# stiffness.py
import numpy as np
from scipy.sparse import lil_matrix


def assemble_stiffness_and_rhs(elemTags, conn, jac, det, xphys, w, N, gN, kappa_fun, rhs_fun, tag_to_dof):
    """
    Assemble global stiffness matrix and load vector for:
        -d/dx (kappa(x) du/dx) = f(x)

    K_ij = ∫ kappa * grad(N_i)·grad(N_j) dx
    F_i  = ∫ f * N_i dx

    Notes:
    - gmsh gives gN in reference coordinates; we map with inv(J).
    - For 1D line embedded in 3D, gmsh provides a 3x3 Jacobian; we keep the same approach.
    Parameters
    ----------
    elemTags : list of int
        Element tags (one per element).
    conn : ndarray (ne x nloc) 
        Connectivity: node tags for each element.
    jac : ndarray (ne x ngp x 3 x 3)
        Jacobian matrices at quadrature points.
    det : ndarray (ne x ngp)
        Determinants of the Jacobian at quadrature points.
    xphys : ndarray (ne x ngp x 3)
        Physical coordinates of quadrature points.
    w : ndarray (ngp,)
        Quadrature weights.
    N : ndarray (ngp x nloc)
        Basis functions evaluated at quadrature points.
    gN : ndarray (ngp x nloc x 3)
        Gradients of basis functions in reference coordinates at quadrature points.
    kappa_fun : function
        Function kappa(x) that returns the diffusivity at point x.
    rhs_fun : function
        Function f(x) that returns the source term at point x.
    tag_to_dof : ndarray
        Mapping from node tags to DoF indices.

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

def assemble_rhs_neumann(F, elemTags, conn, jac, det, xphys, w, N, gN, g_neu_fun, tag_to_dof):
    """
    Condition de Neumann : -k ∂T/∂n = g_neu(x)
    Ajoute ∫ g_neu * N_i dΓ  à F.

    Parameters
    ----------
    F : ndarray (nn,)
        Load vector to be modified.
    elemTags : list of int
        Element tags (one per element).
    conn : ndarray (ne x nloc)
        Connectivity: node tags for each element.
    jac : ndarray (ne x ngp x 3 x 3)
        Jacobian matrices at quadrature points.
    det : ndarray (ne x ngp)
        Determinants of the Jacobian at quadrature points.
    xphys : ndarray (ne x ngp x 3)
        Physical coordinates of quadrature points.
    w : ndarray (ngp,)
        Quadrature weights.
    N : ndarray (ngp x nloc)
        Basis functions evaluated at quadrature points.
    gN : ndarray (ngp x nloc x 3)
        Gradients of basis functions in reference coordinates at quadrature points.
    g_neu_fun : function
        Function g_neu(x) that returns the Neumann flux at point x.
    tag_to_dof : ndarray
        Mapping from node tags to DoF indices.
    
    Returns
    -------
    F : ndarray (nn,)
         Modified load vector with Neumann contributions.
    """

    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac = np.asarray(jac, dtype=np.float64).reshape(ne, ngp, 3, 3)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    gN = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3)

    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]
        for g in range(ngp):
            xg = xphys[e, g]
            wg = w[g]
            detg = det[e, g]

            g_neu_g = float(g_neu_fun(xg))

            for a in range(nloc):
                Ia = int(dof_indices[a])
                N_a = N[g, a]
                F[Ia] += wg * g_neu_g * N_a * detg

    return F

def assemble_robin(K, F, elemTags, conn, det, _xphys, w, N, h, T_inf, tag_to_dof):
    """
    Condition de Robin : -k ∂T/∂n = h(T - T_inf)
    Ajoute  h ∫ N_i N_j dΓ  à K  et  h T_inf ∫ N_i dΓ  à F.

    Parameters
    ----------
    K : lil_matrix (nn x nn)
        Stiffness matrix to be modified.
    F : ndarray (nn,)
        Load vector to be modified.
    elemTags : list of int
        Element tags (one per element).
    conn : ndarray (ne x nloc)
        Connectivity: node tags for each element.
    det : ndarray (ne x ngp)
        Determinants of the Jacobian at quadrature points.
    _xphys : ndarray (ne x ngp x 3)
        Physical coordinates of quadrature points (not used here).
    w : ndarray (ngp,)
        Quadrature weights.
    N : ndarray (ngp x nloc)
        Basis functions evaluated at quadrature points.
    h : float
        Heat transfer coefficient.
    T_inf : float
        Ambient temperature.
    tag_to_dof : ndarray
        Mapping from node tags to DoF indices.

    Returns
    -------
    K : lil_matrix (nn x nn)
        Modified stiffness matrix with Robin contributions.
    F : ndarray (nn,)
        Modified load vector with Robin contributions.
    """
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)

    det  = np.asarray(det,  dtype=np.float64).reshape(ne, ngp)
    conn = np.asarray(conn, dtype=np.int64  ).reshape(ne, nloc)
    N    = np.asarray(N,    dtype=np.float64).reshape(ngp, nloc)

    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]
        for g in range(ngp):
            wg   = w[g]
            detg = det[e, g]

            for a in range(nloc):
                Ia = int(dof_indices[a])
                F[Ia] += wg * h * T_inf * N[g, a] * detg

                for b in range(nloc):
                    Ib = int(dof_indices[b])
                    K[Ia, Ib] += wg * h * N[g, a] * N[g, b] * detg

    return K, F

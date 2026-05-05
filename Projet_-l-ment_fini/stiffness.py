#----------------------------------MODIFIE-------------------------------------------------------
import numpy as np
from scipy.sparse import coo_matrix, lil_matrix


def assemble_stiffness_and_rhs(elemTags, conn, jac, det, xphys, w, N, gN, kappa_fun, rhs_fun, tag_to_dof):
    """
    Assemble global stiffness matrix and load vector for:
        -d/dx (kappa(x) du/dx) = f(x)

    K_ij = ∫ kappa(x) * grad(N_i) · grad(N_j) dx
    F_i  = ∫ f(x) * N_i dx

    Parameters
    ----------
    elemTags  : array-like (ne,), element tags
    conn      : flattened connectivity (ne*nloc)
    jac       : flattened Jacobians (ne*ngp*3*3)
    det       : flattened det(J) values (ne*ngp)
    xphys     : flattened physical coordinates of Gauss points (ne*ngp*3)
    w         : quadrature weights (ngp,)
    N         : flattened basis values (ngp*nloc)
    gN        : flattened basis gradients in reference coords (ngp*nloc*3)
    kappa_fun : callable, diffusion coefficient kappa(x) -> float
    rhs_fun   : callable, source term f(x) -> float
    tag_to_dof: array-like (n_tags,), mapping from node tag to global dof index

    Returns
    -------
    K : coo_matrix (nn x nn), global stiffness matrix
    F : ndarray (nn,), global load vector
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


    F = np.zeros(nn, dtype=np.float64)

    kappa = np.array([[float(kappa_fun(xphys[e, g])) for g in range(ngp)] for e in range(ne)])
    # (ne, ngp) — valeur de kappa en chaque point de Gauss de chaque élément

    rhs = np.array([[float(rhs_fun(xphys[e, g])) for g in range(ngp)] for e in range(ne)])
    # (ne, ngp) — valeur de f en chaque point de Gauss de chaque élément

    invjac = np.linalg.inv(jac) # (ne, ngp, 3, 3) — inverse du Jacobien pour chaque élément et point de Gauss

    gN_phys = np.einsum('egij,gaj->egia', invjac, gN) # (ne, ngp, nloc, 3) — gradients physiques des fonctions de base à chaque point de Gauss
    
    wd_rhs   = w[None, :] * det * rhs        # (ne, ngp) — pour F 
    wd_stiff = w[None, :] * det * kappa      # (ne, ngp) — pour K

    # vecteur de charge
    Fe = np.einsum('eg,ga->ea', wd_rhs, N)   # (ne, nloc)

    # matrice de rigidité
    Ke = np.einsum('eg,egia,egib->eab', wd_stiff, gN_phys, gN_phys)

    dof_indices = tag_to_dof[conn]                                          # (ne, nloc)
    np.add.at(F, dof_indices, Fe)

    # assemblage de K
    rows = np.broadcast_to(dof_indices[:, :, None], (ne, nloc, nloc)).ravel()
    cols = np.broadcast_to(dof_indices[:, None, :], (ne, nloc, nloc)).ravel()
    vals = Ke.ravel()
    K = coo_matrix((vals, (rows, cols)), shape=(nn, nn))
    return K, F

def assemble_rhs_neumann(F, elemTags, conn, jac, det, xphys, w, N, gN, g_neu_fun, tag_to_dof):
    #Assemble le vecteur de charge pour les conditions de Neumann: (flux imposé g_neu sur la frontière)
    # F_i += ∫ g_neu * N_i ds
    """
    Assemble the Neumann boundary contribution into the load vector:
        F_i += ∫ g_neu(x) * N_i(x) dx

    Parameters
    ----------
    F         : ndarray (nn,), load vector to update in place
    elemTags  : array-like (ne,), element tags
    conn      : flattened connectivity (ne*nloc)
    jac       : flattened Jacobians (ne*ngp*3*3)
    det       : flattened det(J) values (ne*ngp)
    xphys     : flattened physical coordinates of Gauss points (ne*ngp*3)
    w         : quadrature weights (ngp,)
    N         : flattened basis values (ngp*nloc)
    gN        : flattened basis gradients in reference coords (ngp*nloc*3)
    g_neu_fun : callable, Neumann flux function g_neu(x) -> float
    tag_to_dof: array-like (n_tags,), mapping from node tag to global dof index

    Returns
    -------
    F : ndarray (nn,), load vector updated with Neumann contributions
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

    g_neu = np.array([[float(g_neu_fun)(xphys[e, g]) for g in range(ngp)] for e in range(ne)])  #-> valeur du flux de Neumann aux points de quadrature pour chaque élément
    wd = w[None, :] * det  * g_neu                                                              #-> poids de quadrature pondéré par le flux de Neumann et le déterminant du Jacobien pour chaque élément et point de quadrature
    Fe = np.einsum('eg,ga->ea', wd, N)                                                          #-> contribution locale au vecteur de charge pour chaque élément et nœud local, calculée en intégrant le flux de Neumann pondéré par les fonctions de base aux points de quadrature

    dof_indices = tag_to_dof[conn]
    np.add.at(F, dof_indices, Fe)                                                               #-> assemble les contributions locales au vecteur de charge global F en utilisant les indices de degrés de liberté globaux pour chaque élément et nœud local, en ajoutant les contributions locales à F aux positions correspondantes dans le vecteur global F.
                                                                                               #Prend en compte les doublons !
    return F


def assemble_robin(K, F, elemTags, conn, det, xphys, w, N, h, T_inf, tag_to_dof):
    """
    Assemble Robin boundary contributions into stiffness matrix and load vector:
        K_ij += ∫ h * N_i * N_j dx
        F_i  += ∫ h * T_inf * N_i dx

    Parameters
    ----------
    K         : lil_matrix (nn x nn), stiffness matrix to update in place
    F         : ndarray (nn,), load vector to update in place
    elemTags  : array-like (ne,), element tags
    conn      : flattened connectivity (ne*nloc)
    det       : flattened det(J) values (ne*ngp)
    xphys     : flattened physical coordinates of Gauss points (ne*ngp*3)
    w         : quadrature weights (ngp,)
    N         : flattened basis values (ngp*nloc)
    h         : float, heat transfer coefficient
    T_inf     : float, ambient temperature
    tag_to_dof: array-like (n_tags,), mapping from node tag to global dof index

    Returns
    -------
    K : sparse matrix (nn x nn), updated stiffness matrix
    F : ndarray (nn,), updated load vector
    """
    ne = len(elemTags) # number of elements
    ngp = len(w)    # number of quadrature points
    nloc = int(len(conn) // ne) # number of local nodes per element

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp) # reshape to (ne, ngp)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc) # reshape to (ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc) # reshape to (ngp, nloc) 


    wd = w[None, :] * det   

    #---Contribution sur K-----------------------
    Fe = h * T_inf * np.einsum('eg,ga->ea', wd, N)         
    dof_indices = tag_to_dof[conn]                          
    np.add.at(F, dof_indices, Fe)

    #---Contribution sur F-----------------------
    Ke = h * np.einsum('eg,ga,gb->eab', wd, N, N)          # (ne, nloc, nloc)


    return K, F # return the modified stiffness matrix and load vector

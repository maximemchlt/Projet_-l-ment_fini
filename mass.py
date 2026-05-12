# --------------------------------------------------MODIFIE-------------------------------------------------------------------------------
import numpy as np
from scipy.sparse import coo_matrix, lil_matrix


def assemble_mass(elemTags, conn, det, w, N, tag_to_dof):
    """
    Assemble global mass matrix:
        M_ij = sum_e ∫_e N_i N_j dx

    Parameters
    ----------
    elemTags : array-like, shape (ne,)
    conn     : flattened connectivity (ne*nloc)
    det      : flattened det(J) values (ne*ngp)
    w        : quadrature weights (ngp)
    N        : flattened basis values (ngp*nloc)

    Returns
    -------
    M : coo_matrix (nn x nn)
    """
    #---Dimensions------------------------------------------------
    ne = len(elemTags)                                           # nombre d'éléments dans le maillage
    ngp = len(w)                                              # nombre de points de Gauss par élément
    nloc = int(len(conn) // ne)                                  # nombre de nœuds locaux par élément (ex : 3 pour un triangle linéaire)
    nn = np.max(tag_to_dof[tag_to_dof >= 0]) + 1        # nombre total de degrés de liberté (DoFs) dans le système global

    #---Reshape des entrées-----------------------------------------
    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)

    #---Calcul de toutes les matrices locales de masse (ne, nloc, nloc)----------------------

    wd = w[None, :] * det                        # (ne, ngp)    ->modifie la taille de w pour qu'elle puisse être multipliée élément par élément avec det
    Me = np.einsum('eg,ga,gb->eab', wd, N, N)   # (ne, nloc, nloc)  -> calcule les contributions locales à la matrice de masse pour chaque élément en utilisant les poids de quadrature et les valeurs des fonctions de base aux points de Gauss


    #---Indicces globaux pour chaque (e,a,b)--------------------------------------

    dof_indices = tag_to_dof[conn]               # (ne, nloc) -> convertit les tags de nœuds locaux en indices de degrés de liberté globaux à l'aide du mapping tag_to_dof, ce qui permet d'assembler correctement les contributions locales dans la matrice globale M.
    rows = dof_indices[:, :, None]               # (ne, nloc, 1) -> prépare les indices de ligne pour l'assemblage en ajoutant une dimension supplémentaire pour permettre la diffusion lors de l'assemblage des contributions locales dans la matrice globale M.
    cols = dof_indices[:, None, :]               # (ne, 1, nloc) -> prépare les indices de colonne pour l'assemblage en ajoutant une dimension supplémentaire pour permettre la diffusion lors de l'assemblage des contributions locales dans la matrice globale M.
    rows = np.broadcast_to(rows, (ne, nloc, nloc)).ravel()
    cols = np.broadcast_to(cols, (ne, nloc, nloc)).ravel() #-> (ne*nloc*nloc,) -> diffuse les indices de ligne et de colonne pour correspondre à la taille des contributions locales à la matrice de masse, puis les aplatis en un tableau 1D pour l'assemblage dans la matrice globale M.
    vals = Me.ravel()                      # (ne*nloc*nloc,) -> aplatit les contributions locales à la matrice de masse en un tableau 1D pour l'assemblage dans la matrice globale M.


    #---Assemblage final dans la matrice globale M--------------------------------------

    M = coo_matrix((vals, (rows, cols)), shape=(nn, nn)) #-> crée une matrice creuse au format COO à partir des indices de ligne, de colonne et des valeurs, puis la convertit en format LIL pour une manipulation efficace lors de l'assemblage final dans la matrice globale M.

    return M

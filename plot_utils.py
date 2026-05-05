# plot_utils.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import gmsh


def plot_fe_solution_high_order(
    elemType, elemNodeTags, nodeCoords, U,
    M=80, show_nodes=False, ax=None, label=None
):
    """
    Plot 1D high-order FE solution by sampling each element and evaluating gmsh basis.
    Assumes U is aligned with gmsh's compact node ordering (0..nn-1).
    """
    _, _, _, nloc, _, _ = gmsh.model.mesh.getElementProperties(elemType)

    u = np.linspace(-1.0, 1.0, int(M))
    pts3 = np.zeros((len(u), 3), dtype=float)
    pts3[:, 0] = u
    uvw = pts3.reshape(-1).tolist()

    _, bf, _ = gmsh.model.mesh.getBasisFunctions(elemType, uvw, "Lagrange")
    N = np.asarray(bf, dtype=float).reshape(len(u), nloc)

    if ax is None:
        fig, ax = plt.subplots()

    ne = int(len(elemNodeTags) // nloc)
    _, _, coords_flat = gmsh.model.mesh.getJacobians(elemType, uvw)
    coords = np.asarray(coords_flat, dtype=float).reshape(ne, len(u), 3)

    for e in range(ne):
        tags_e = np.asarray(elemNodeTags[e * nloc:(e + 1) * nloc], dtype=int) - 1
        Ue = U[tags_e]

        x = coords[e, :, 0]
        uh = N @ Ue

        order = np.argsort(x)
        ax.plot(x[order], uh[order], label=label if (e == 0) else None)

    if show_nodes:
        Xn = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)[:, 0]
        ax.plot(Xn, U, "o", markersize=4)

    ax.set_xlabel("x")
    ax.set_ylabel("u_h")
    ax.grid(True)
    return ax


def setup_interactive_figure(xlim=None, ylim=None):
    plt.ion()
    fig, ax = plt.subplots()
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return fig, ax

def plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags, tag_to_index=None):

    coords = nodeCoords.reshape(-1, 3)
    x = coords[:, 0]
    y = coords[:, 1]

    if tag_to_index is None:
        max_node_tag = int(np.max(nodeTags))
        tag_to_index = np.zeros(max_node_tag + 1, dtype=int)
        for i, tag in enumerate(nodeTags):
            tag_to_index[int(tag)] = i

    num_elements = len(elemTags)
    nodes_per_elem = len(elemNodeTags) // num_elements

    # take only the first 3 nodes (=geometric nodes that form the triangles)    
    all_nodes = elemNodeTags.reshape(num_elements, nodes_per_elem)
    corner_nodes = all_nodes[:, :3] 
    
    # Map to indices
    tri_indices = tag_to_index[corner_nodes.astype(int)]
    # ---------------------------------------

    mesh_triang = tri.Triangulation(x, y, tri_indices)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot the skeleton
    ax.triplot(mesh_triang, color='black', lw=0.5, alpha=0.4)

    colors = ["red", "darkblue", "orange", "mediumpurple", "pink"]
    for i, (name, dim) in enumerate(bnds):
        tags = bnds_tags[i]
        indices = tag_to_index[tags.astype(int)]
        ax.scatter(x[indices], y[indices], label=name, s=15, zorder=3, 
                   marker="o", facecolor="None", edgecolor=colors[i % len(colors)])

    ax.set_aspect('equal')
    ax.legend(frameon=True, framealpha=1, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    plt.axis(False)
    plt.show()


def plot_fe_solution_2d(elemNodeTags, nodeCoords, nodeTags, U, tag_to_dof,
                         vmin=None, vmax=None,   # ← nouveau
                         show_mesh=False, ax=None, label=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    num_dofs = len(U)
    coords_mapped = np.zeros((num_dofs, 2))
    all_coords = nodeCoords.reshape(-1, 3)
    
    for i, tag in enumerate(nodeTags):
        dof_idx = tag_to_dof[int(tag)]
        if dof_idx != -1:
            coords_mapped[dof_idx] = all_coords[i, :2]

    x = coords_mapped[:, 0]
    y = coords_mapped[:, 1]

    for possible_n in [3, 6, 10, 15, 21]:
        if len(elemNodeTags) % possible_n == 0:
            nodes_per_elem = possible_n
            break

    conn_reshaped = elemNodeTags.reshape(-1, nodes_per_elem)
    triangles = tag_to_dof[conn_reshaped[:, :3].astype(int)]

    U = np.array(U).flatten()

    # Échelle FIXE basée sur les bornes passées (cohérente avec la colorbar fixe).
    # Si vmin/vmax ne sont pas fournis, on tombe sur l'échelle dynamique du champ.
    vmin_eff = float(np.min(U)) if vmin is None else float(vmin)
    vmax_eff = float(np.max(U)) if vmax is None else float(vmax)

    # Niveaux explicites — sinon tricontourf ignore vmin/vmax et fait son propre min/max
    levels = np.linspace(vmin_eff, vmax_eff, 100)
    # On clippe U pour éviter les NaN si une valeur sort des bornes
    U_clip = np.clip(U, vmin_eff, vmax_eff)

    contour = ax.tricontourf(x, y, triangles, U_clip, levels=levels,
                              cmap='hot',
                              vmin=vmin_eff,
                              vmax=vmax_eff,
                              extend='both')

    if show_mesh:
        ax.triplot(x, y, triangles, color='white', linewidth=0.2, alpha=0.3)

    return contour, ax

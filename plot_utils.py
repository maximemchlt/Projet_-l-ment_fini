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

    Parameters
    ----------
    elemType : int
        GMSH element type (e.g., 1 for line, 2 for triangle, etc.).
    elemNodeTags : array-like, shape (ne*nloc,)
        Element node tags.
    nodeCoords : array-like, shape (nn, 3)
        Node coordinates.
    U : array-like, shape (nn,)
        Solution values at nodes.
    M : int, optional
        Number of sampling points per element (default: 80).
    show_nodes : bool, optional
        Whether to show node locations (default: False).
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on.
    label : str, optional
        Label for the plot line.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object with the plot.
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
    """
    Set up an interactive Matplotlib figure with optional axis limits.

    Parameters
    ----------
    xlim : tuple of (float, float), optional
        Limits for the x-axis (default: None).
    ylim : tuple of (float, float), optional
        Limits for the y-axis (default: None).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax : matplotlib.axes.Axes
        The created axes object.
    """
    plt.ion()
    fig, ax = plt.subplots()
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return fig, ax

def plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags, tag_to_index=None):
    """
    Plot 2D mesh with boundary nodes highlighted.

    Parameters
    ----------
    elemType : int
        GMSH element type (e.g., 2 for triangle, etc.).
    nodeTags : array-like, shape (nn,)
        Node tags.
    nodeCoords : array-like, shape (nn, 3)
        Node coordinates.
    elemTags : array-like, shape (ne,)
        Element tags.
    elemNodeTags : array-like, shape (ne*nloc,)
        Element node tags.
    bnds : list of tuples
        List of boundary conditions, each tuple containing (name: str, dim: int).
    bnds_tags : list of arrays
        List of node tags for each boundary condition.
    tag_to_index : array-like, shape (max_tag + 1,), optional
        Mapping from node tags to indices (default: None).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax : matplotlib.axes.Axes
        The created axes object.
    """
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


def plot_fe_solution_2d(elemNodeTags, nodeCoords, nodeTags, U, tag_to_dof, show_mesh=False, ax=None, label=None, vmin = None, vmax = None):
    """
    Plot 2D FE solution using Matplotlib's tricontourf, dynamically determining the number of nodes per element and mapping GMSH tags to compact DoF indices.

    Parameters
    ----------
    elemNodeTags : array-like, shape (ne*nloc,)
        Element node tags.
    nodeCoords : array-like, shape (nn, 3)
        Node coordinates.
    nodeTags : array-like, shape (nn,)
        Node tags.
    U : array-like, shape (nn,)
        Solution values at each node.
    tag_to_dof : array-like, shape (max_tag + 1,)
        Mapping from node tags to DoF indices.
    show_mesh : bool, optional
        Whether to show the mesh skeleton (default: False).
    ax : matplotlib.axes.Axes, optional
        The axes object to plot on (default: None).
    label : str, optional
        Label for the colorbar (default: None).
    vmin : float, optional
        Minimum value for the colorbar (default: None).
    vmax : float, optional
        Maximum value for the colorbar (default: None).

    Returns
    -------
    contour : matplotlib.contour.QuadContourSet
        The contour plot object.
    ax : matplotlib.axes.Axes
        The axes object with the plot.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # 1. Map coordinates to our compact DoF indices
    num_dofs = len(U)
    coords_mapped = np.zeros((num_dofs, 2))
    all_coords = nodeCoords.reshape(-1, 3)
    
    for i, tag in enumerate(nodeTags):
        dof_idx = tag_to_dof[int(tag)]
        if dof_idx != -1:
            coords_mapped[dof_idx] = all_coords[i, :2]

    x = coords_mapped[:, 0]
    y = coords_mapped[:, 1]

    # 2. Determine nodes per element dynamically
    total_nodes_in_elems = len(elemNodeTags)
    # Standard Lagrange triangle node counts: Order 1=3, Order 2=6, Order 3=10, Order 4=15
    for possible_n in [3, 6, 10, 15, 21]:
        if total_nodes_in_elems % possible_n == 0:
            nodes_per_elem = possible_n
            break
    # 3. Reshape and extract ONLY the 3 corner nodes for Matplotlib
    conn_reshaped = elemNodeTags.reshape(-1, nodes_per_elem)
    # Map the GMSH tags to our 0...N-1 indices
    triangles = tag_to_dof[conn_reshaped[:, :3].astype(int)]
    # 4. Plotting
    U = np.array(U).flatten()
    vmin_eff = float(np.min(U)) if vmin is None else float(vmin)
    vmax_eff = float(np.max(U)) if vmax is None else float(vmax)

    levels = np.linspace(vmin_eff, vmax_eff, 100)
    U_clip = np.clip(U, vmin_eff, vmax_eff)
    
    contour = ax.tricontourf(x, y, triangles, U_clip, levels = levels, cmap='hot', vmin=vmin_eff, vmax=vmax_eff)
    
    if show_mesh:
        ax.triplot(x, y, triangles, color='white', linewidth=0.2, alpha=0.3)

    return contour, ax

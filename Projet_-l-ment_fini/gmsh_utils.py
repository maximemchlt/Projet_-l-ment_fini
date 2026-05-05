# gmsh_utils.py
import numpy as np
import gmsh


def gmsh_init(model_name="fem1d"):
    gmsh.initialize()
    gmsh.model.add(model_name)


def gmsh_finalize():
    gmsh.finalize()


def build_1d_mesh(L=1.0, cl1=0.02, cl2=0.10, order=1):
    """
    Build and mesh a 1D segment [0,L] with different characteristic lengths.
    Returns (line_tag, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags).
    """
    p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, cl1)
    p1 = gmsh.model.geo.addPoint(L, 0.0, 0.0, cl2)
    line = gmsh.model.geo.addLine(p0, p1)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(1)
    gmsh.model.mesh.setOrder(order)

    elemType = gmsh.model.mesh.getElementType("line", order)

    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)

    return line, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags


def prepare_quadrature_and_basis(elemType, order):
    """
    Returns:
      xi (flattened uvw), w (ngp), N (flattened bf), gN (flattened gbf)
    """
    rule = f"Gauss{2 * order}"
    xi, w = gmsh.model.mesh.getIntegrationPoints(elemType, rule)
    _, N, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "Lagrange")
    _, gN, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "GradLagrange")
    return xi, np.asarray(w, dtype=float), N, gN


def get_jacobians(elemType, xi, tag=-1):
    """
    Wrapper around gmsh.getJacobians.
    Returns (jacobians, dets, coords)
    """
    jacobians, dets, coords = gmsh.model.mesh.getJacobians(elemType, xi, tag=tag)
    return jacobians, dets, coords

#-------------------------------------------------------------------------------
# Modified by ourself to add a function that gets Jacobians
# for all entities in a named physical group, iterating over
# entity tags in the same order as blade_geometry uses for connectivity arrays.
#-------------------------------------------------------------------------------
def get_jacobians_physical(physical_name, elemType, xi):
    """
    Returns Jacobians for all entities in a named physical group, iterating over
    entity tags in the same order as blade_geometry uses for connectivity arrays.
    """
    entity_tags = gmsh.model.getEntitiesForPhysicalName(physical_name) # Get entity tags for the given physical group name
    jacobians_list = [] 
    dets_list = []
    coords_list = []
    
    for dim, tag in entity_tags:
        jacobians, dets, coords = gmsh.model.mesh.getJacobians(elemType, xi, tag=tag)
        jacobians_list.append(jacobians)
        dets_list.append(dets)
        coords_list.append(coords)
    
    # Concatenate results from all entities
    jacobians_all = np.concatenate(jacobians_list) 
    dets_all = np.concatenate(dets_list) 
    coords_all = np.concatenate(coords_list)
    
    return jacobians_all, dets_all, coords_all
#-------------------------------------------------------------------------------

def end_dofs_from_nodes(nodeCoords):
    """
    Robustly identify first/last node dofs from coordinates (x-min, x-max).
    nodeCoords is flattened [x0,y0,z0, x1,y1,z1, ...]
    Returns (left_dof, right_dof) as 0-based indices.
    """
    X = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)[:, 0]
    left = int(np.argmin(X))
    right = int(np.argmax(X))
    return left, right

def border_dofs_from_tags(l_tags, tag_to_dof):
    """
    Converts a list of GMSH node tags into the corresponding 
    compact matrix indices (DoFs).
    """
    # Ensure tags are integers
    l_tags = np.asarray(l_tags, dtype=int)
    
    # Filter out any tags that might not be in our DoF mapping (like geometry points)
    # then map them to our 0...N-1 indices
    valid_mask = (tag_to_dof[l_tags] != -1)
    l_dofs = tag_to_dof[l_tags[valid_mask]]
    return l_dofs

def getPhysical(name):
    """
    Get the physical group elements and nodes for a given name and dimension.
    """
    
    dimTags = gmsh.model.getEntitiesForPhysicalName(name)
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=dimTags[0][0], tag=dimTags[0][1])
    elemType = elemTypes[0]  # Assuming one element type per physical group
    elemTags = elemTags[0]
    elemNodeTags = elemNodeTags[0]
    entityTag = dimTags[0][1]
    return elemType, elemTags, elemNodeTags, entityTag
    

def open_2d_mesh(msh_filename, order=1):
    """
    Load a .msh file.

    Parameters
    ----------
    msh_filename : str
        Path to the .msh file
    order : int
        Polynomial order of elements

    Returns
    -------
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags
    """

    import gmsh

    # --- load geometry
    gmsh.open(msh_filename)

    # --- high order
    gmsh.model.mesh.setOrder(order)

    # --- element type (triangles)
    elemType = gmsh.model.mesh.getElementType("triangle", order)

    # --- nodes
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()

    # --- elements
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)

    surf = gmsh.model.getEntities(2)[0][1]

    curve_tags = gmsh.model.getBoundary([(2, surf)], oriented=False)
    
    #--------------------------------------------
    # Add physical groups for the boundaries
    #--------------------------------------------

    def _approx_curve_length(ctag):
        """
        Approximate the length of a curve given its tag by summing distances between its nodes.
        """
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes(dim=1, tag=ctag)
        if len(node_coords) == 0:
            return 0.0
        pts = np.asarray(node_coords, dtype=float).reshape(-1, 3)
        pts = pts[:, :2]  # We only care about x and y for length
        cen = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - cen[1], pts[:, 0] - cen[0])
        order = np.argsort(angles)
        pts_ordered = pts[order]
        segs = np.diff(np.vstack([pts_ordered, pts_ordered[0]]), axis=0)
        return float(np.sum(np.sqrt(segs[:, 0]**2 + segs[:, 1]**2)))

    lengths = [_approx_curve_length(ctag) for _, ctag in curve_tags]
    outer_idx = int(np.argmax(lengths))
    inner_idx = 1 - outer_idx

    """
    gmsh.model.addPhysicalGroup(1, [curve_tags[0][1]], tag=1)
    gmsh.model.setPhysicalName(1, 1, "OuterBoundary")

    gmsh.model.addPhysicalGroup(1, [curve_tags[1][1]], tag=2)
    gmsh.model.setPhysicalName(1, 2, "InnerBoundary")
    """
    gmsh.model.addPhysicalGroup(1, [abs(curve_tags[outer_idx][1])], tag=1)
    gmsh.model.setPhysicalName(1, 1, "OuterBoundary")

    gmsh.model.addPhysicalGroup(1, [abs(curve_tags[inner_idx][1])], tag=2)
    gmsh.model.setPhysicalName(1, 2, "InnerBoundary")

    #---------------------------------------------

    bnds = [('OuterBoundary', 1),('InnerBoundary', 1)]

    bnds_tags = []
    for name, dim in bnds:
        tag = -1
        for t in gmsh.model.getPhysicalGroups(dim):
            if gmsh.model.getPhysicalName(dim, t[1]) == name:
                tag = t[1]
                break
        if tag == -1:
            raise ValueError(f"Physical group '{name}' not found in mesh.")
        bnds_tags.append(gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)[0])

    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags

import gmsh
import numpy as np


def create_blade_3d(a=0.1, b=0.004,
                    L_forte=0.20, L_taper=0.40,
                    cl_base=0.004, cl_tip=0.0015,
                    order=1):
    """
    Crée et maille la lame 3D en deux parties :
      - Forte  (z = 0 … L_forte)      : section losange CONSTANTE  a × b
      - Effilée (z = L_forte … L)     : section losange qui décroît jusqu'à
                                         la pointe en z = L = L_forte + L_taper

    Paramètres
    ----------
    a, b      : demi-largeur / demi-épaisseur à la base [m]
    L_forte   : longueur de la partie à section constante [m]
    L_taper   : longueur de la partie effilée [m]
    cl_base   : taille de maille à la base [m]
    cl_tip    : taille de maille à la pointe [m]
    order     : ordre polynomial des éléments

    Retourne
    --------
    elemType, nodeTags, nodeCoords,
    elemTags, elemNodeTags,
    bnd_elemType, bnd_elemTags, bnd_elemNodeTags, bnd_entityTag, vol_tags
    """
    L = L_forte + L_taper

    # ── Sommets ───────────────────────────────────────────────────────────────
    # Base (z = 0)
    p1 = gmsh.model.geo.addPoint(0, -b, 0, cl_base)  # sud
    p2 = gmsh.model.geo.addPoint(a, 0, 0, cl_base)   # est
    p3 = gmsh.model.geo.addPoint(0, b, 0, cl_base)   # nord
    p4 = gmsh.model.geo.addPoint(-a, 0, 0, cl_base)  # ouest

    # Jonction forte / effilée (z = L_forte)
    p5 = gmsh.model.geo.addPoint( 0, -b, L_forte, cl_base)
    p6 = gmsh.model.geo.addPoint( a,  0, L_forte, cl_base)
    p7 = gmsh.model.geo.addPoint( 0,  b, L_forte, cl_base)
    p8 = gmsh.model.geo.addPoint(-a,  0, L_forte, cl_base)

    # Pointe (z = L)
    p9 = gmsh.model.geo.addPoint( 0,  0, L,       cl_tip)

    # ── Arêtes de la base losange ─────────────────────────────────────────────
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    # ── Arêtes du losange à la jonction ──────────────────────────────────────
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p8)
    l8 = gmsh.model.geo.addLine(p8, p5)

    # ── Arêtes verticales de la forte ────────────────────────────────────────
    l9  = gmsh.model.geo.addLine(p1, p5)
    l10 = gmsh.model.geo.addLine(p2, p6)
    l11 = gmsh.model.geo.addLine(p3, p7)
    l12 = gmsh.model.geo.addLine(p4, p8)

    # ── Arêtes vers la pointe ─────────────────────────────────────────────────
    l13 = gmsh.model.geo.addLine(p5, p9)
    l14 = gmsh.model.geo.addLine(p6, p9)
    l15 = gmsh.model.geo.addLine(p7, p9)
    l16 = gmsh.model.geo.addLine(p8, p9)

    # ── Surfaces ──────────────────────────────────────────────────────────────
    # Base (z = 0)
    s0  = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1,  l2,  l3,  l4])])

    # Losange à la jonction (face interne — partagée entre les deux volumes)
    s_mid = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l5,  l6,  l7,  l8])])

    # Faces latérales de la forte (rectangles)
    sf1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1,  l10, -l5,  -l9])])
    sf2 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l2,  l11, -l6, -l10])])
    sf3 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l3,  l12, -l7, -l11])])
    sf4 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l4,   l9, -l8, -l12])])

    # Faces latérales de la partie effilée (triangles)
    st1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l5,  l14, -l13])])
    st2 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l6,  l15, -l14])])
    st3 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l7,  l16, -l15])])
    st4 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l8,  l13, -l16])])

    # ── Volumes ───────────────────────────────────────────────────────────────
    v_forte = gmsh.model.geo.addVolume(
        [gmsh.model.geo.addSurfaceLoop([s0, sf1, sf2, sf3, sf4, s_mid])])

    v_taper = gmsh.model.geo.addVolume(
        [gmsh.model.geo.addSurfaceLoop([s_mid, st1, st2, st3, st4])])

    gmsh.model.geo.synchronize()

    # ── Groupes physiques ─────────────────────────────────────────────────────
    gmsh.model.addPhysicalGroup(3, [v_forte, v_taper], tag=1)
    gmsh.model.setPhysicalName(3, 1, "Blade")

    # Frontière extérieure uniquement (s_mid est intérieure)
    gmsh.model.addPhysicalGroup(2, [s0, sf1, sf2, sf3, sf4, st1, st2, st3, st4], tag=2)
    gmsh.model.setPhysicalName(2, 2, "BladeSurface")

    # ── Génération du maillage ────────────────────────────────────────────────
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(order)

    # ── Éléments volumiques (tétraèdres) ──────────────────────────────────────
    # getElementsByType sans tag → tous les tets (même ordre que getJacobians tag=-1)
    elemType = gmsh.model.mesh.getElementType("tetrahedron", order)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags  = gmsh.model.mesh.getElementsByType(elemType)

    # ── Éléments frontière (triangles) ────────────────────────────────────────
    bnd_elemType   = gmsh.model.mesh.getElementType("triangle", order)
    bnd_entityTags = gmsh.model.getEntitiesForPhysicalName("BladeSurface")
    bnd_entityTag  = bnd_entityTags[0][1]

    all_bnd_et, all_bnd_ent = [], []
    for dim, tag in bnd_entityTags:
        et, ent = gmsh.model.mesh.getElementsByType(bnd_elemType, tag=tag)
        all_bnd_et.append(et)
        all_bnd_ent.append(ent)

    bnd_elemTags     = np.concatenate(all_bnd_et)
    bnd_elemNodeTags = np.concatenate(all_bnd_ent)

    return (elemType, nodeTags, nodeCoords, elemTags, elemNodeTags,
            bnd_elemType, bnd_elemTags, bnd_elemNodeTags, bnd_entityTag,
            [v_forte, v_taper])

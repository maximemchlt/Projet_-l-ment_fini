import gmsh
import numpy as np

def create_blade_section(a=0.02, b=0.004, cl=0.001, order=1):
    """
    Crée et maille la section en losange de la lame.

    Paramètres
    ----------
    a   : demi-largeur [m]
    b   : demi-épaisseur [m]
    cl  : taille caractéristique du maillage [m]
    order : ordre des éléments

    Retourne
    --------
    elemType, nodeTags, nodeCoords,
    elemTags, elemNodeTags,
    bnd_elemType, bnd_elemTags, bnd_elemNodeTags, bnd_entityTag
    """

    # ── Étape 1 : définir les 4 coins du losange ──────────────────────────────
    p1 = gmsh.model.geo.addPoint( 0, -b, 0, cl)  # bas
    p2 = gmsh.model.geo.addPoint( a,  0, 0, cl)  # droite
    p3 = gmsh.model.geo.addPoint( 0,  b, 0, cl)  # haut
    p4 = gmsh.model.geo.addPoint(-a,  0, 0, cl)  # gauche

    # ── Étape 2 : relier les points par des lignes ────────────────────────────
    l1 = gmsh.model.geo.addLine(p1, p2)  # bas-droite
    l2 = gmsh.model.geo.addLine(p2, p3)  # droite-haut
    l3 = gmsh.model.geo.addLine(p3, p4)  # haut-gauche
    l4 = gmsh.model.geo.addLine(p4, p1)  # gauche-bas

    # ── Étape 3 : définir le contour fermé ───────────────────────────────────
    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

    # ── Étape 4 : définir la surface ─────────────────────────────────────────
    surf = gmsh.model.geo.addPlaneSurface([loop])

    # ── Étape 5 : synchroniser (obligatoire avant de mailler) ────────────────
    gmsh.model.geo.synchronize()

    # ── Étape 6 : définir les groupes physiques ───────────────────────────────
    # Surface du domaine
    gmsh.model.addPhysicalGroup(2, [surf], tag=1)
    gmsh.model.setPhysicalName(2, 1, "Blade")

    # Frontière complète = les 4 lignes
    gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4], tag=2)
    gmsh.model.setPhysicalName(1, 2, "BladeBoundary")

    # ── Étape 7 : générer le maillage ────────────────────────────────────────
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    # ── Étape 8 : récupérer les données du maillage ───────────────────────────
    elemType = gmsh.model.mesh.getElementType("triangle", order)

    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags  = gmsh.model.mesh.getElementsByType(elemType)

    # Éléments frontière (segments)
    bnd_elemType = gmsh.model.mesh.getElementType("line", order)
    bnd_elemTags, bnd_elemNodeTags = gmsh.model.mesh.getElementsByType(bnd_elemType)

    # Tag de l'entité frontière pour les jacobiens
    bnd_entityTags = gmsh.model.getEntitiesForPhysicalName("BladeBoundary")
    # On récupère tous les segments de toutes les lignes du contour
    all_bnd_elemTags     = []
    all_bnd_elemNodeTags = []
    bnd_entityTag        = bnd_entityTags[0][1]  # on garde le premier pour les jacobiens

    for dim, tag in bnd_entityTags:
        et, ent = gmsh.model.mesh.getElementsByType(bnd_elemType, tag=tag)
        all_bnd_elemTags.append(et)
        all_bnd_elemNodeTags.append(ent)

    bnd_elemTags     = np.concatenate(all_bnd_elemTags)
    bnd_elemNodeTags = np.concatenate(all_bnd_elemNodeTags)

    return (elemType, nodeTags, nodeCoords, elemTags, elemNodeTags,
        bnd_elemType, bnd_elemTags, bnd_elemNodeTags, bnd_entityTag,
        surf)  # ← tag de la surface physique

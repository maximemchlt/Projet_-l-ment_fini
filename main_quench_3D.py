import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.tri as mtri
import matplotlib.gridspec as gridspec
import scipy.sparse as sp
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
import gmsh

from gmsh_utils import (gmsh_init, gmsh_finalize,
                        prepare_quadrature_and_basis,
                        get_jacobians, get_jacobians_physical)
from blade_geometry_3d import create_blade_3d
from stiffness import assemble_stiffness_and_rhs, assemble_robin
from mass import assemble_mass
from dirichlet import theta_step_robin

# ----- Paramètres matériau / fluide -----------------------------------------
rho, cp, k  = 7800, 500, 50    # acier
h,   T_inf  = 3000, 20.0       # eau

# ----- Géométrie et simulation -----------------------------------------------
T0       = 1000.0
a        = 0.020   # demi-largeur (grande diagonale = 2a)  [m]
b        = 0.004   # demi-épaisseur (petite diagonale = 2b) [m]
L_forte  = 0.60    # partie à section constante             [m]
L_taper  = 0.20    # partie effilée jusqu'à la pointe       [m]
L        = L_forte + L_taper
cl_base  = 0.006   # taille maille à la base (diminuer pour affiner)
cl_tip   = 0.002   # taille maille à la pointe
dt       = 0.05    # pas de temps [s]
nsteps   = 600     # → 30 s de trempe
theta    = 1.0
order    = 5
z_cut    = L_forte * 0.5   # position de la coupe transversale

# ----- Maillage --------------------------------------------------------------
print("--- Génération du maillage ---")
gmsh_init("quench_3d")

(elemType, nodeTags, nodeCoords,
 elemTags, elemNodeTags,
 bnd_elemType, bnd_elemTags, bnd_elemNodeTags,
 bnd_entityTag, vol_tags) = create_blade_3d(
     a=a, b=b, L_forte=L_forte, L_taper=L_taper,
     cl_base=cl_base, cl_tip=cl_tip, order=order)

# ----- Mapping tags → DOF 0 … N-1 -------------------------------------------
unique_tags = np.unique(elemNodeTags)
num_dofs    = len(unique_tags)
max_tag     = int(np.max(nodeTags))
tag_to_dof  = np.full(max_tag + 1, -1, dtype=int)
for i, tag in enumerate(unique_tags):
    tag_to_dof[int(tag)] = i

all_node_tags   = np.asarray(nodeTags,   dtype=int)
all_node_coords = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)
dof_coords = np.zeros((num_dofs, 3))
for i, tag in enumerate(all_node_tags):
    d = tag_to_dof[tag]
    if d >= 0:
        dof_coords[d] = all_node_coords[i]

print(f"  {num_dofs} nœuds   {len(elemTags)} tétraèdres   {len(bnd_elemTags)} triangles frontière")

# ----- Quadrature ------------------------------------------------------------
xi,  w,  N,  gN  = prepare_quadrature_and_basis(elemType,     order)
xib, wb, Nb, gNb = prepare_quadrature_and_basis(bnd_elemType, order)

# ----- Jacobiens -------------------------------------------------------------
# Volume : tag=-1 → tous les tets, même ordre que getElementsByType sans tag
jac,  det,  coords  = get_jacobians(elemType,     xi,  tag=-1)

# Frontière : per-entité pour garantir l'alignement avec bnd_elemNodeTags
jacb, detb, coordsb = get_jacobians_physical("BladeSurface", bnd_elemType, xib)

# ----- Vérification géométrique ----------------------------------------------
ne_vol  = len(elemTags)
ngp_vol = len(w)
vol_num = np.sum(np.asarray(det).reshape(ne_vol, ngp_vol) * w)
# vol_th = 2ab(L_forte + L_taper/3) : section rhomboïde constante (prisme) + effilée (pyramide)
vol_th  = 2.0 * a * b * (L_forte + L_taper / 3.0)

ne_bnd  = len(bnd_elemTags)
ngp_bnd = len(wb)
surf_num = np.sum(np.asarray(detb).reshape(ne_bnd, ngp_bnd) * wb)
side      = np.sqrt(a**2 + b**2)
surf_th   = (2.0 * a * b                                           # base
             + 4.0 * side * L_forte                                # 4 rectangles forte
             + 2.0 * np.sqrt(L_taper**2 * (a**2 + b**2) + (a*b)**2))  # 4 triangles effilée

print(f"\n=== Vérification géométrique ===")
print(f"Volume    num : {vol_num:.4e}   théo : {vol_th:.4e}")
print(f"Surface   num : {surf_num:.4e}   théo : {surf_th:.4e}")
print(f"================================\n")

# ----- Assemblage (peut prendre ~ 30 s en Python pur) -----------------------
print("--- Assemblage ---")
print("Assemblage rigidité …")
K_lil, F = assemble_stiffness_and_rhs(
    elemTags, elemNodeTags, jac, det, coords, w, N, gN,
    lambda x: float(k), lambda x: 0.0, tag_to_dof)

print("Assemblage masse …")
M = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof).tocsr() * (rho * cp)

print("Assemblage Robin …")
K_lil, F = assemble_robin(
    K_lil, F, bnd_elemTags, bnd_elemNodeTags,
    detb, coordsb, wb, Nb, h, T_inf, tag_to_dof)
K = K_lil.tocsr()

L_carac = vol_th / surf_th
print(f"\n  tau_diff (b²ρc/k) = {rho*cp*b**2/k:.2f} s")
print(f"  tau_conv (V/S·ρc/h) = {rho*cp*L_carac/h:.2f} s")
print(f"  Bi = {h*L_carac/k:.4f}\n")

# ----- Condition initiale ----------------------------------------------------
U = np.full(num_dofs, T0, dtype=float)

# ----- Triangulations 2D pour les vues planes --------------------------------
def delaunay_tri(pts2d):
    d = Delaunay(pts2d)
    return mtri.Triangulation(pts2d[:, 0], pts2d[:, 1], d.simplices)

# Vue de face XZ : nœuds y ≥ 0 (demi-espace avant)
idx_f = np.where(dof_coords[:, 1] >= -1e-10)[0]
tri_f = delaunay_tri(dof_coords[idx_f][:, [0, 2]])   # (x, z)

# Coupe transversale à z = z_cut
tol_z = cl_base * 3.0
idx_c = np.where(np.abs(dof_coords[:, 2] - z_cut) < tol_z)[0]
tri_c = delaunay_tri(dof_coords[idx_c][:, :2]) if idx_c.size >= 4 else None

# Connectivité surface 3D
bnd_conn = np.asarray(bnd_elemNodeTags, dtype=int).reshape(-1, 3)
bnd_dofs = tag_to_dof[bnd_conn]   # (ne_bnd, 3)

# ----- Mise en place de la figure -------------------------------------------
norm = mcolors.Normalize(vmin=T_inf, vmax=T0)
cmap = cm.hot

fig = plt.figure(figsize=(22, 9))
# 3 colonnes : 3D large | vue de face large | section étroite
gs  = gridspec.GridSpec(2, 3, figure=fig,
                        width_ratios=[6, 5, 2],
                        left=0.04, right=0.91, top=0.91, bottom=0.07,
                        wspace=0.38, hspace=0.45)

ax3d = fig.add_subplot(gs[:, 0], projection='3d')   # colonne gauche entière
ax_f = fig.add_subplot(gs[:, 1])                    # colonne centrale entière
ax_c = fig.add_subplot(gs[1, 2])                    # bas-droit : section (petite)

# Colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.925, 0.10, 0.013, 0.78])
fig.colorbar(sm, cax=cbar_ax, label='T [°C]', format='%.0f')

# ----- Poly3DCollection créé UNE SEULE FOIS → seules les couleurs changent --
T_face_0 = U[bnd_dofs].mean(axis=1)
poly = Poly3DCollection(dof_coords[bnd_dofs],
                        facecolors=cmap(norm(T_face_0)),
                        edgecolors='none', shade=False, alpha=0.97)
ax3d.add_collection3d(poly)
ax3d.set_xlim(-a,     a    );  ax3d.set_xlabel('x [m]', fontsize=8, labelpad=2)
ax3d.set_ylim(-b*2,   b*2  );  ax3d.set_ylabel('y [m]', fontsize=8, labelpad=2)
ax3d.set_zlim( 0,     L    );  ax3d.set_zlabel('z [m]', fontsize=8, labelpad=2)
ax3d.set_box_aspect([2*a, 4*b, L])
ax3d.view_init(elev=18, azim=-50)
ax3d.tick_params(labelsize=7)

# Axes 2D : labels permanents (on utilise cla() pour effacer les artistes seuls)
fig.add_subplot(gs[0, 2]).set_visible(False)   # case haut-droit vide

for ax, xl, yl, ti in [
        (ax_f, 'x [m]', 'z [m]', 'Vue de face  (plan XZ, y ≥ 0)'),
        (ax_c, 'x [m]', 'y [m]', f'Section  z = {z_cut:.2f} m'),
]:
    ax.set_xlabel(xl, fontsize=8)
    ax.set_ylabel(yl, fontsize=8)
    ax.set_title(ti, fontsize=9)
    ax.tick_params(labelsize=7)

contour_f = contour_c = None   # références aux derniers contourf


def update(U, t_cur):
    global contour_f, contour_c

    T_mean = U.mean()

    # ── Vue 3D : mise à jour des couleurs uniquement ──────────────────────
    T_face = U[bnd_dofs].mean(axis=1)
    poly.set_facecolor(cmap(norm(T_face)))

    # ── Vue de face (XZ) ─────────────────────────────────────────────────
    if contour_f is not None:
        contour_f.remove()
    contour_f = ax_f.tricontourf(tri_f, U[idx_f],
                                  levels=50, cmap=cmap, norm=norm)
    ax_f.set_aspect('auto')

    # ── Section transversale ──────────────────────────────────────────────
    if contour_c is not None:
        contour_c.remove()
    if tri_c is not None and idx_c.size >= 4:
        contour_c = ax_c.tricontourf(tri_c, U[idx_c],
                                      levels=50, cmap=cmap, norm=norm)
    else:
        ax_c.scatter(dof_coords[idx_c, 0], dof_coords[idx_c, 1],
                     c=U[idx_c], cmap=cmap, norm=norm, s=15)
        contour_c = None
    ax_c.set_aspect('equal')

    fig.suptitle(
        f"Trempe acier / eau  —  t = {t_cur:.2f} s    T̄ = {T_mean:.0f} °C",
        fontsize=12)

    fig.canvas.draw_idle()   # force le rendu des nouvelles couleurs de Poly3DCollection
    plt.pause(0.001)


# ----- Boucle temporelle -----------------------------------------------------
print("Simulation en cours …\n")
plt.show(block=False)   # ouvre la fenêtre avant la boucle
plt.pause(0.1)          # laisse le temps au backend de l'afficher

try:
    for step in range(nsteps):
        U = theta_step_robin(M, K, F, U, dt=dt, theta=theta)

        if step % 5 == 0:
            update(U, step * dt)
finally:
    gmsh_finalize()

plt.ioff()
plt.show()

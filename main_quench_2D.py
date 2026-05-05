import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import scipy.sparse as sp
import gmsh

from gmsh_utils import gmsh_init, gmsh_finalize, \
                       prepare_quadrature_and_basis, get_jacobians
from blade_geometry import create_blade_section
from stiffness import assemble_stiffness_and_rhs, assemble_robin
from mass import assemble_mass
from dirichlet import theta_step_robin
from plot_utils import plot_fe_solution_2d

# ----- Parametres materiau et fluide ----------------------------------------
# delta_T_crit : différence de température centre-surface à ne pas dépasser
# pour éviter la fissuration thermique (critère simplifié σ = E α ΔT/(1-ν) < σ_rupture)
#   acier  : E=200 GPa, α=12e-6/°C, ν=0.3  → ΔT_crit ≈ 200°C
#   titane : E=110 GPa, α=8.6e-6/°C, ν=0.34 → ΔT_crit ≈ 500°C
MATERIALS = {
    "steel":    {"rho": 7800, "cp": 500, "k": 50,  "delta_T_crit": 200.0},
    "titanium": {"rho": 4500, "cp": 520, "k": 22,  "delta_T_crit": 500.0},
}
FLUIDS = {
    "water": {"h": 3000, "T_inf": 20.0},
    "oil":   {"h":  500, "T_inf": 60.0},
}

mat   = MATERIALS["steel"]
fluid = FLUIDS["water"]

rho, cp, k       = mat["rho"], mat["cp"], mat["k"]
delta_T_crit     = mat["delta_T_crit"]
h,   T_inf       = fluid["h"], fluid["T_inf"]

# ----- Verification immediate des parametres --------------------------------
print(f"\n=== Parametres lus ===")
print(f"rho={rho}  cp={cp}  k={k}")
print(f"h={h}  T_inf={T_inf}")
print(f"rho*cp = {rho*cp}")
print(f"======================\n")

T0     = 1000.0
a      = 0.02
b      = 0.004
cl     = 0.001
dt     = 0.01
nsteps = 3000
theta  = 1.0
order  = 1

# ----- Maillage -------------------------------------------------------------
gmsh_init("quench_2d")

(elemType,     nodeTags,        nodeCoords,
 elemTags,     elemNodeTags,
 bnd_elemType, bnd_elemTags,    bnd_elemNodeTags,
 bnd_entityTag, surf_tag) = create_blade_section(a=a, b=b, cl=cl, order=order)

# ----- Mapping tags Gmsh -> indices 0..N-1 ----------------------------------
unique_tags = np.unique(elemNodeTags)
num_dofs    = len(unique_tags)
max_tag     = int(np.max(nodeTags))
tag_to_dof  = np.full(max_tag + 1, -1, dtype=int)
for i, tag in enumerate(unique_tags):
    tag_to_dof[int(tag)] = i

# ----- Quadrature - volume et frontiere -------------------------------------
xi,  w,  N,  gN  = prepare_quadrature_and_basis(elemType,     order)
xib, wb, Nb, gNb = prepare_quadrature_and_basis(bnd_elemType, order)

# tag=-1 -> tous les éléments volumiques (triangles)
jac,  det,  coords  = get_jacobians(elemType,     xi,  tag=-1)

det_arr = np.asarray(det)
ngp    = len(w)
ne_vol = len(elemTags)
print(f"=== Inspection brute ===")
print(f"len(det)      : {len(det_arr)}")
print(f"len(elemTags) : {ne_vol}")
print(f"ngp           : {ngp}")
print(f"len(det) / ngp: {len(det_arr) / ngp}  (devrait = {ne_vol})")
print(f"det min/max   : {det_arr.min():.4e} / {det_arr.max():.4e}")
print(f"========================\n")

jacb, detb, coordsb = get_jacobians(bnd_elemType, xib, tag=-1)


# ----- Verification geometrique ---------------------------------------------
det_vol = np.asarray(det).reshape(ne_vol, ngp)
surface_numerique = np.sum(det_vol * w[np.newaxis, :])
# Aire d'un losange = (d1 * d2) / 2 = (2a * 2b) / 2 = 2*a*b
surface_theorique = 2.0 * a * b

ne_bnd  = len(bnd_elemTags)
ngp_bnd = len(wb)
det_bnd = np.asarray(detb).reshape(ne_bnd, ngp_bnd)
perimetre_numerique = np.sum(det_bnd * wb[np.newaxis, :])
perimetre_theorique = 4.0 * np.sqrt(a**2 + b**2)

print(f"=== Verification geometrique ===")
print(f"Surface numerique  : {surface_numerique:.6e}")
print(f"Surface theorique  : {surface_theorique:.6e}")
print(f"Perimetre numerique: {perimetre_numerique:.6e}")
print(f"Perimetre theorique: {perimetre_theorique:.6e}")
print(f"================================\n")


# ----- Assemblage rigidite --------------------------------------------------
K_lil, F = assemble_stiffness_and_rhs(
    elemTags, elemNodeTags,
    jac, det, coords,
    w, N, gN,
    lambda x: float(k),
    lambda x: 0.0,
    tag_to_dof
)

# ----- Assemblage masse + diagnostic ----------------------------------------
M_raw = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)

print(f"=== Diagnostic masse ===")
print(f"Somme de M_raw       : {M_raw.sum():.6e}  (attendu = surface = {surface_theorique:.4e})")
print(f"M_raw norme          : {sp.linalg.norm(M_raw):.4e}")
print(f"M_raw diagonal max   : {M_raw.tocsr().diagonal().max():.4e}")
print(f"rho*cp               : {rho * cp}")

M = M_raw.tocsr() * (rho * cp)
# Capacite thermique totale attendue : rho * cp * surface
M_sum_attendu = rho * cp * surface_theorique
print(f"Somme de M           : {M.sum():.4e}  (attendu = rho*cp*surface = {M_sum_attendu:.4e})")
print(f"M final diagonal max : {M.diagonal().max():.4e}")
print(f"========================\n")

# ----- Assemblage Robin + diagnostic ----------------------------------------
K_diag_avant = K_lil.tocsr().diagonal().copy()

K_lil, F = assemble_robin(
    K_lil, F,
    bnd_elemTags,    bnd_elemNodeTags,
    detb, coordsb,
    wb, Nb,
    h, T_inf,
    tag_to_dof
)

K = K_lil.tocsr()
delta_K_diag = K.diagonal() - K_diag_avant

# Somme attendue de F (terme source Robin) = h * T_inf * perimetre
F_sum_attendu = h * T_inf * perimetre_theorique
print(f"=== Diagnostic Robin ===")
print(f"Contribution Robin diagonale min : {delta_K_diag.min():.4e}")
print(f"Contribution Robin diagonale max : {delta_K_diag.max():.4e}")
print(f"Noeuds touches par Robin         : {np.count_nonzero(delta_K_diag)}")
print(f"F min / max                      : {F.min():.4e} / {F.max():.4e}")
print(f"Somme F                          : {F.sum():.4e}  (attendu = h*T_inf*perim = {F_sum_attendu:.4e})")
print(f"========================\n")

# ----- Diagnostic global ----------------------------------------------------
# Longueur caracteristique pour le bilan global (V/A pour le losange)
L_carac = surface_theorique / perimetre_theorique
print(f"=== Diagnostic global ===")
print(f"Noeuds                         : {num_dofs}")
print(f"Norme K (Frobenius)            : {sp.linalg.norm(K):.4e}")
print(f"Norme M (Frobenius)            : {sp.linalg.norm(M):.4e}")
print(f"Norme F                        : {np.linalg.norm(F):.4e}")
print(f"tau_diff  (b^2 rho c / k)      = {rho*cp*(b**2)/k:.2f} s   (epaisseur)")
print(f"tau_conv  (V/A * rho c / h)    = {rho*cp*L_carac/h:.2f} s   (losange)")
print(f"Bi        (V/A * h / k)        = {h*L_carac/k:.3f}")
print(f"=========================\n")

# ----- Nœud central et nœuds de surface ------------------------------------
# Coordonnées 2D de chaque DOF
dof_xy = np.zeros((num_dofs, 2))
coords_arr = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)
for i, tag in enumerate(nodeTags):
    d = tag_to_dof[int(tag)]
    if d >= 0:
        dof_xy[d] = coords_arr[i, :2]

center_dof = int(np.argmin(np.linalg.norm(dof_xy, axis=1)))

surf_dofs = np.unique(tag_to_dof[np.asarray(bnd_elemNodeTags, dtype=int)])
surf_dofs = surf_dofs[surf_dofs >= 0]

print(f"Noeud central : DOF {center_dof}  coords = {dof_xy[center_dof]}")
print(f"Noeuds de surface : {len(surf_dofs)}")
print(f"ΔT critique ({mat}) : {delta_T_crit:.0f} °C\n")

# ----- Condition initiale ---------------------------------------------------
U = np.full(num_dofs, T0, dtype=float)

# ----- Figure : champ de temperature (gauche) + ΔT centre-surface (droite) --
fig, (ax, ax_dt) = plt.subplots(1, 2, figsize=(14, 5))
plt.ion()

norm = mcolors.Normalize(vmin=T_inf, vmax=T0)
sm   = cm.ScalarMappable(cmap='hot', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label='T [degC]', fraction=0.03, pad=0.04)

# Graphe ΔT : ligne de mesure + seuil critique
t_hist  = []
dt_hist = []
line_dt, = ax_dt.plot([], [], 'b-', linewidth=1.5, label='ΔT centre – surface')
ax_dt.axhline(delta_T_crit, color='r', linestyle='--', linewidth=1.5,
              label=f'ΔT critique = {delta_T_crit:.0f} °C')
ax_dt.set_xlim(0, nsteps * dt)
ax_dt.set_ylim(0, T0 - T_inf)
ax_dt.set_xlabel('t [s]')
ax_dt.set_ylabel('ΔT [°C]')
ax_dt.set_title('Gradient thermique centre – surface')
ax_dt.legend(fontsize=9)
ax_dt.grid(True, alpha=0.3)
warning_text = ax_dt.text(
    0.98, 0.95, '', transform=ax_dt.transAxes,
    color='red', fontsize=10, fontweight='bold',
    ha='right', va='top')

# ----- Affichage initial + délai avant démarrage ----------------------------
plt.show(block=False)
plt.pause(0.1)   # délai minimal pour laisser le backend afficher la fenêtre

# ----- Boucle en temps ------------------------------------------------------
exceeded = False   # passage du seuil critique déjà signalé en console

try:
    for step in range(nsteps):
        U = theta_step_robin(M, K, F, U, dt=dt, theta=theta)

        # ΔT centre – surface (calculé à chaque pas)
        delta_T = float(U[center_dof] - U[surf_dofs].mean())
        t_cur   = step * dt
        t_hist.append(t_cur)
        dt_hist.append(delta_T)

        if delta_T > delta_T_crit and not exceeded:
            print(f"⚠  RISQUE DE FISSURE  t = {t_cur:.2f} s  ΔT = {delta_T:.1f} °C > {delta_T_crit:.0f} °C")
            exceeded = True
        if exceeded and delta_T <= delta_T_crit:
            print(f"   Seuil repassé sous la limite  t = {t_cur:.2f} s  ΔT = {delta_T:.1f} °C")
            exceeded = False

        if step % 10 == 0:
            # Champ de temperature
            ax.clear()
            plot_fe_solution_2d(
                elemNodeTags=elemNodeTags,
                nodeTags=nodeTags,
                nodeCoords=nodeCoords,
                U=U,
                tag_to_dof=tag_to_dof,
                vmin=T_inf,
                vmax=T0,
                show_mesh=False,
                ax=ax
            )
            T_mean = U.mean()
            ax.set_title(
                f"Trempe acier/eau - t = {t_cur:.1f} s    T_moy = {T_mean:.0f} °C",
                fontsize=11
            )
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.axis('equal')
            ax.set_xlim(-a * 1.15, a * 1.15)
            ax.set_ylim(-b * 1.5,  b * 1.5)

            # Courbe ΔT
            line_dt.set_data(t_hist, dt_hist)
            dt_max = max(dt_hist) if dt_hist else 0
            ax_dt.set_ylim(0, max(dt_max * 1.15, delta_T_crit * 1.3))
            # Zone critique en rouge si dépassement actuel
            if delta_T > delta_T_crit:
                ax_dt.set_facecolor('#fff0f0')
                warning_text.set_text('⚠ RISQUE DE FISSURE')
            else:
                ax_dt.set_facecolor('white')
                warning_text.set_text('')

            fig.tight_layout()
            plt.pause(0.01)

        if U.mean() <= T_inf + 5.0:
            print(f"   Pièce refroidie (T_moy ≤ T_inf+5)  t = {t_cur:.2f} s")
            break
finally:
    gmsh_finalize()

plt.ioff()
plt.show()

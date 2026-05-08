"""
Optimisation de la trempe — balayage paramétrique 1D
=====================================================
Pour chaque couple (T_inf, Bi) :
  - simulation EF 1D implicite (Euler, theta=1)
  - enregistrement du temps de refroidissement (T_centre → 200°C)
  - enregistrement du ΔT_max (centre − surface)
  - contrainte de rupture : ΔT_max > 200°C

Tracé : 4 graphes complémentaires.
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.sparse.linalg import splu

sys.path.insert(0, os.path.dirname(__file__))

import gmsh
from gmsh_utils import (gmsh_init, gmsh_finalize, build_1d_mesh,
                        prepare_quadrature_and_basis, get_jacobians,
                        end_dofs_from_nodes)
from stiffness import assemble_stiffness_and_rhs
from mass import assemble_mass

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Paramètres physiques
# ═══════════════════════════════════════════════════════════════════════════════
rho, cp, k = 7800.0, 500.0, 50.0   # acier  [kg/m³, J/(kg·K), W/(m·K)]
T0         = 1200.0                  # température initiale [°C]
T_target   = 200.0                   # critère d'arrêt : T_centre ≤ T_target [°C]
DT_crit    = 200.0                   # seuil de rupture : ΔT_max [°C]
L          = 0.004                   # demi-épaisseur [m]

dt         = 0.05    # pas de temps [s]
max_time   = 450.0   # durée maximale [s]  (couvre Bi=0.1, T_inf=5 °C)
theta      = 1.0     # Euler implicite

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Grille paramétrique
# ═══════════════════════════════════════════════════════════════════════════════
N_TINF = 30
N_BI   = 30

T_inf_arr = np.linspace(5.0, 190.0, N_TINF)           # température du bain [°C]
Bi_arr    = np.logspace(np.log10(0.05), np.log10(25),  # Bi = h·L/k
                        N_BI)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Maillage et assemblage (fait une seule fois — géométrie fixe)
# ═══════════════════════════════════════════════════════════════════════════════
print("Initialisation du maillage Gmsh...")
gmsh_init("quench_optim")

_, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags = \
    build_1d_mesh(L=L, cl1=0.0002, cl2=0.0002, order=1)

unique_tags = np.unique(elemNodeTags)
num_dofs    = len(unique_tags)
max_tag     = int(np.max(nodeTags))
tag_to_dof  = np.full(max_tag + 1, -1, dtype=int)
for i, tag in enumerate(unique_tags):
    tag_to_dof[int(tag)] = i

xi, w, N, gN     = prepare_quadrature_and_basis(elemType, 1)
jac, det, coords = get_jacobians(elemType, xi)

K_vol, F_vol_base = assemble_stiffness_and_rhs(
    elemTags, elemNodeTags, jac, det, coords, w, N, gN,
    lambda x: float(k), lambda x: 0.0, tag_to_dof)
M_vol = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
M_vol = M_vol * (rho * cp)

left_dof, right_dof = end_dofs_from_nodes(nodeCoords)
# left_dof  → surface x = 0  (Robin BC)
# right_dof → centre  x = L  (symétrie, Neumann = 0)

gmsh_finalize()
print(f"Maillage : {num_dofs} DDL  |  left_dof={left_dof}  right_dof={right_dof}")

M_csr = M_vol.tocsr()
nsteps_max = int(max_time / dt)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Balayage paramétrique
# ═══════════════════════════════════════════════════════════════════════════════
cooling_time = np.full((N_TINF, N_BI), np.nan)
max_deltaT   = np.zeros((N_TINF, N_BI))

print(f"\nBalayage {N_TINF}x{N_BI} = {N_TINF*N_BI} simulations...")

for i, T_inf in enumerate(T_inf_arr):
    for j, Bi in enumerate(Bi_arr):

        h = Bi * k / L      # [W/(m²·K)]

        # Matrice et vecteur avec condition de Robin
        K_case = K_vol.copy()
        K_case[left_dof, left_dof] += h
        F_case = F_vol_base.copy()
        F_case[left_dof] += h * T_inf

        K_csr  = K_case.tocsr()
        F_vec  = np.asarray(F_case).ravel()

        # Pré-factorisation LU de (M + dt·K) — constante sur toute la boucle temporelle
        A  = (M_csr + dt * K_csr).tocsc()
        lu = splu(A)
        # Pour theta=1 : rhs = M·U + dt·F
        # (B = M - 0·dt·K = M)

        U      = np.full(num_dofs, T0)
        dT_max = 0.0

        for step in range(nsteps_max):
            rhs = M_csr.dot(U) + dt * F_vec
            U   = lu.solve(rhs)

            T_centre  = U[right_dof]
            T_surface = U[left_dof]
            dT        = T_centre - T_surface
            if dT > dT_max:
                dT_max = dT

            if T_centre <= T_target:
                cooling_time[i, j] = (step + 1) * dt
                break

        max_deltaT[i, j] = dT_max

    pct = (i + 1) / N_TINF * 100
    print(f"  [{pct:5.1f}%]  T_inf = {T_inf:6.1f} C  |  "
          f"Bi in [{Bi_arr[0]:.3f}, {Bi_arr[-1]:.1f}]", flush=True)

feasible = max_deltaT <= DT_crit   # True ↔ pas de rupture

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Visualisation
# ═══════════════════════════════════════════════════════════════════════════════
Bi_mesh, Tinf_mesh = np.meshgrid(Bi_arr, T_inf_arr)

fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    "Optimisation de la trempe — Acier  "
    f"(ρ={rho:.0f} kg/m³, cp={cp:.0f} J/kg·K, k={k:.0f} W/m·K)\n"
    f"T₀ = {T0:.0f}°C,  demi-épaisseur L = {L*1e3:.1f} mm,  "
    f"critère arrêt T_centre ≤ {T_target:.0f}°C,  seuil rupture ΔT > {DT_crit:.0f}°C",
    fontsize=12, y=0.99)

# ── Colormap temps de refroidissement (NaN = gris) ───────────────────────────
cmap_ct = plt.cm.plasma_r.copy()
cmap_ct.set_bad(color='lightgrey')

ct_feasible = np.where(feasible, cooling_time, np.nan)
vmax_ct = np.nanpercentile(ct_feasible, 98) if np.any(~np.isnan(ct_feasible)) else max_time

# ── A. Carte 2D : temps de refroidissement ────────────────────────────────────
ax1 = fig.add_subplot(2, 2, 1)

im1 = ax1.pcolormesh(Bi_mesh, Tinf_mesh, ct_feasible,
                     cmap=cmap_ct, shading='auto',
                     norm=mcolors.Normalize(vmin=0, vmax=vmax_ct))
# Zone de rupture en rouge semi-transparent
ax1.pcolormesh(Bi_mesh, Tinf_mesh,
               np.where(~feasible, 1.0, np.nan),
               cmap=mcolors.ListedColormap(['#c0392b']),
               shading='auto', alpha=0.55)
# Contour de la frontière de rupture
cs1 = ax1.contour(Bi_mesh, Tinf_mesh, max_deltaT,
                  levels=[DT_crit], colors='white', linewidths=2, linestyles='--')
ax1.clabel(cs1, fmt=f'ΔT = {DT_crit:.0f}°C', fontsize=9, inline=True)

cb1 = fig.colorbar(im1, ax=ax1, extend='max')
cb1.set_label("Temps de refroidissement [s]", fontsize=9)
ax1.set_xscale('log')
ax1.set_xlabel("Nombre de Biot  Bi = h·L/k", fontsize=10)
ax1.set_ylabel("Température du bain  T∞ [°C]", fontsize=10)
ax1.set_title("Carte (Bi, T∞) — Temps de refroidissement\n"
              "(zone rouge = rupture, blanc pointillé = limite)", fontsize=10)
legend_patches = [
    Patch(facecolor='#c0392b', alpha=0.6, label=f'ΔT_max > {DT_crit:.0f}°C (rupture)'),
    Patch(facecolor='lightgrey', label='Temps dépassé (> {:.0f} s)'.format(max_time)),
]
ax1.legend(handles=legend_patches, fontsize=8, loc='upper right')

# ── B. Carte 2D : ΔT_max ────────────────────────────────────────────────────
ax2 = fig.add_subplot(2, 2, 2)

vmax_dt = max(DT_crit * 2, np.percentile(max_deltaT, 99))
im2 = ax2.pcolormesh(Bi_mesh, Tinf_mesh, max_deltaT,
                     cmap='hot_r', shading='auto',
                     norm=mcolors.Normalize(vmin=0, vmax=vmax_dt))
cs2 = ax2.contour(Bi_mesh, Tinf_mesh, max_deltaT,
                  levels=[DT_crit], colors='cyan', linewidths=2)
ax2.clabel(cs2, fmt=f'ΔT = {DT_crit:.0f}°C', fontsize=9, inline=True)

cb2 = fig.colorbar(im2, ax=ax2)
cb2.set_label("ΔT_max = T_centre − T_surface [°C]", fontsize=9)
ax2.set_xscale('log')
ax2.set_xlabel("Nombre de Biot  Bi = h·L/k", fontsize=10)
ax2.set_ylabel("Température du bain  T∞ [°C]", fontsize=10)
ax2.set_title("Gradient thermique maximal\n"
              "(contour cyan = seuil de rupture 200°C)", fontsize=10)

# ── C. Coupes à Bi fixé : temps vs T_inf ─────────────────────────────────────
ax3 = fig.add_subplot(2, 2, 3)

Bi_cuts   = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
palette_c = plt.cm.viridis(np.linspace(0.05, 0.95, len(Bi_cuts)))

for Bi_cut, col in zip(Bi_cuts, palette_c):
    j_idx = int(np.argmin(np.abs(Bi_arr - Bi_cut)))
    Bi_real = Bi_arr[j_idx]
    ct_col = cooling_time[:, j_idx].copy()
    feas   = feasible[:, j_idx]

    # Segment faisable
    ct_ok = np.where(feas, ct_col, np.nan)
    ax3.plot(T_inf_arr, ct_ok, '-', color=col, lw=1.8,
             label=f"Bi = {Bi_real:.2f}")
    # Segment infaisable (pointillé plus clair)
    ct_bad = np.where(~feas, ct_col, np.nan)
    ax3.plot(T_inf_arr, ct_bad, ':', color=col, lw=1.2, alpha=0.5)

ax3.set_xlabel("Température du bain  T∞ [°C]", fontsize=10)
ax3.set_ylabel("Temps de refroidissement [s]", fontsize=10)
ax3.set_title("Temps de refroidissement vs T∞\n"
              "(trait plein = faisable, pointillé = rupture)", fontsize=10)
ax3.legend(fontsize=8, title="Bi = h·L/k", ncol=2,
           title_fontsize=8, loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(T_inf_arr[0], T_inf_arr[-1])

# ── D. Coupes à T_inf fixé : temps vs Bi ─────────────────────────────────────
ax4 = fig.add_subplot(2, 2, 4)

Tinf_cuts = [20, 40, 60, 80, 100, 130, 160]
palette_d = plt.cm.coolwarm(np.linspace(0.0, 1.0, len(Tinf_cuts)))

for T_cut, col in zip(Tinf_cuts, palette_d):
    i_idx = int(np.argmin(np.abs(T_inf_arr - T_cut)))
    T_real = T_inf_arr[i_idx]
    ct_row = cooling_time[i_idx, :].copy()
    feas   = feasible[i_idx, :]

    ct_ok  = np.where(feas, ct_row, np.nan)
    ax4.semilogx(Bi_arr, ct_ok, '-', color=col, lw=1.8,
                 label=f"T∞ = {T_real:.0f}°C")
    ct_bad = np.where(~feas, ct_row, np.nan)
    ax4.semilogx(Bi_arr, ct_bad, ':', color=col, lw=1.2, alpha=0.5)

ax4.set_xlabel("Nombre de Biot  Bi = h·L/k", fontsize=10)
ax4.set_ylabel("Temps de refroidissement [s]", fontsize=10)
ax4.set_title("Temps de refroidissement vs Bi\n"
              "(trait plein = faisable, pointillé = rupture)", fontsize=10)
ax4.legend(fontsize=8, title="T∞ bain", ncol=2,
           title_fontsize=8, loc='upper right')
ax4.grid(True, alpha=0.3, which='both')
ax4.set_xlim(Bi_arr[0], Bi_arr[-1])

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = os.path.join(os.path.dirname(__file__), "quench_optimisation.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure sauvegardee -> {out_path}")
plt.show()

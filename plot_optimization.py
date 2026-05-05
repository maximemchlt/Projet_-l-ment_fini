"""
plot_optimization.py
====================
Visualisations dédiées à l'optimisation multi-objectif de la trempe.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from leidenfrost import h_leidenfrost, h_constant


# ─────────────────────────────────────────────────────────────────────────────
# 1. Carte de dureté 2D
# ─────────────────────────────────────────────────────────────────────────────
def plot_hardness_map(elemNodeTags, nodeCoords, nodeTags,
                      hardness, tag_to_dof, ax=None,
                      vmin=20.0, vmax=65.0, title=None):
    """
    Carte HRC sur la section 2D — vert = dur (bon), rouge = mou (mauvais).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    num_dofs = len(hardness)
    coords_mapped = np.zeros((num_dofs, 2))
    all_coords = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)
    for i, tag in enumerate(nodeTags):
        d = tag_to_dof[int(tag)]
        if d != -1:
            coords_mapped[d] = all_coords[i, :2]
    x = coords_mapped[:, 0]
    y = coords_mapped[:, 1]

    for possible_n in [3, 6, 10, 15, 21]:
        if len(elemNodeTags) % possible_n == 0:
            nodes_per_elem = possible_n
            break
    conn_reshaped = np.asarray(elemNodeTags).reshape(-1, nodes_per_elem)
    triangles = tag_to_dof[conn_reshaped[:, :3].astype(int)]

    H = np.clip(np.asarray(hardness).flatten(), vmin, vmax)
    levels = np.linspace(vmin, vmax, 80)

    contour = ax.tricontourf(x, y, triangles, H, levels=levels,
                             cmap="RdYlGn", vmin=vmin, vmax=vmax,
                             extend="both")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return contour, ax


# ─────────────────────────────────────────────────────────────────────────────
# 2. Profils h(T)
# ─────────────────────────────────────────────────────────────────────────────
def plot_h_profiles(params_list, labels, T_range=(20.0, 1000.0), ax=None):
    """
    Trace h(T) pour plusieurs jeux de paramètres sur le même graphe.
    Marque T_leid et T_nucl pour le premier jeu Leidenfrost trouvé.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    T = np.linspace(T_range[0], T_range[1], 500)
    annotated = False
    for p, label in zip(params_list, labels):
        if p is None or p.get("constant", False):
            h_val = p.get("h", 3000.0) if p else 3000.0
            ax.plot(T, h_constant(T, h=h_val), '--', linewidth=1.6, label=label)
        else:
            h = h_leidenfrost(
                T,
                h_max  = p.get("h_max",  8000.0),
                h_film = p.get("h_film",  200.0),
                T_leid = p.get("T_leid",  300.0),
                T_nucl = p.get("T_nucl",  100.0),
                h_conv = p.get("h_conv", 3000.0),
            )
            ax.plot(T, h, '-', linewidth=2, label=label)

            if not annotated:
                ax.axvline(p.get("T_leid", 300.0), color='orange',
                           linestyle=':', alpha=0.7)
                ax.text(p.get("T_leid", 300.0), ax.get_ylim()[1] * 0.9
                        if ax.get_ylim()[1] > 0 else 1.0,
                        ' T_leid', color='orange', fontsize=8)
                ax.axvline(p.get("T_nucl", 100.0), color='green',
                           linestyle=':', alpha=0.7)
                ax.text(p.get("T_nucl", 100.0), ax.get_ylim()[1] * 0.8
                        if ax.get_ylim()[1] > 0 else 1.0,
                        ' T_nucl', color='green', fontsize=8)
                annotated = True

    ax.set_xlabel("Temperature T [C]")
    ax.set_ylabel("h [W/m^2/K]")
    ax.set_title("Profils h(T) compares")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 3. Front de Pareto
# ─────────────────────────────────────────────────────────────────────────────
def plot_pareto_front(results_list, pareto_indices,
                      param_name="h_max", ax=None):
    """
    Scatter ΔT_max vs HRC_surface_mean — Pareto en rouge étoilé.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    dT  = np.array([r["delta_T_max"]            for r in results_list])
    HRC = np.array([r["hardness_surface_mean"]  for r in results_list])
    cvals = np.array([
        (r["params"] or {}).get(param_name, np.nan) for r in results_list
    ])

    sc = ax.scatter(dT, HRC, c=cvals, cmap="viridis", s=55,
                    edgecolor='gray', alpha=0.85)

    # Pareto
    p_idx = np.asarray(pareto_indices, dtype=int)
    if len(p_idx) > 0:
        # Tri pour relier les points Pareto
        order = np.argsort(dT[p_idx])
        ax.plot(dT[p_idx][order], HRC[p_idx][order],
                'r--', linewidth=1.0, alpha=0.7)
        ax.scatter(dT[p_idx], HRC[p_idx],
                   marker='*', s=240, color='red',
                   edgecolor='black', linewidth=1.0, zorder=5,
                   label='Front de Pareto')

    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(param_name)

    ax.set_xlabel("deltaT max centre-surface [C]  (a minimiser)")
    ax.set_ylabel("HRC surface moyen  (a maximiser)")
    ax.set_title("Front de Pareto -- compromis durete vs gradient thermique")
    ax.grid(True, alpha=0.3)

    # Flèche favorable vers bas-droite (ΔT faible, HRC haut) — en haut-gauche du plan
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.annotate(
        "favorable",
        xy=(xlim[0] + 0.10 * (xlim[1] - xlim[0]),
            ylim[0] + 0.92 * (ylim[1] - ylim[0])),
        xytext=(xlim[0] + 0.40 * (xlim[1] - xlim[0]),
                ylim[0] + 0.55 * (ylim[1] - ylim[0])),
        arrowprops=dict(arrowstyle="->", color='darkgreen', lw=1.5),
        color='darkgreen', fontsize=10, fontweight='bold'
    )
    ax.legend(fontsize=9, loc='lower right')
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dashboard final
# ─────────────────────────────────────────────────────────────────────────────
def plot_optimization_dashboard(best_params, best_results, baseline_results,
                                mesh, results_list, pareto_indices,
                                save_path=None):
    """
    Figure 2x3 comparant le cas optimisé (Leidenfrost) au cas de base (h cst).

    [0,0] Carte HRC base       [0,1] Carte HRC optimise   [0,2] gain HRC
    [1,0] h(T) compare         [1,1] dT(t) compare         [1,2] Pareto
    """
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    # [0,0] Carte HRC baseline
    cont1, _ = plot_hardness_map(
        mesh["elemNodeTags"], mesh["nodeCoords"], mesh["nodeTags"],
        baseline_results["hardness_field"], mesh["tag_to_dof"],
        ax=axes[0, 0], title="HRC -- baseline (h constant)"
    )
    plt.colorbar(cont1, ax=axes[0, 0], label="HRC")

    # [0,1] Carte HRC optimisé
    cont2, _ = plot_hardness_map(
        mesh["elemNodeTags"], mesh["nodeCoords"], mesh["nodeTags"],
        best_results["hardness_field"], mesh["tag_to_dof"],
        ax=axes[0, 1], title="HRC -- optimise (Leidenfrost)"
    )
    plt.colorbar(cont2, ax=axes[0, 1], label="HRC")

    # [0,2] Carte du gain HRC
    gain = best_results["hardness_field"] - baseline_results["hardness_field"]
    num_dofs = len(gain)
    coords_mapped = np.zeros((num_dofs, 2))
    all_coords = np.asarray(mesh["nodeCoords"], dtype=float).reshape(-1, 3)
    tag_to_dof = mesh["tag_to_dof"]
    for i, tag in enumerate(mesh["nodeTags"]):
        d = tag_to_dof[int(tag)]
        if d != -1:
            coords_mapped[d] = all_coords[i, :2]
    x = coords_mapped[:, 0]; y = coords_mapped[:, 1]
    nloc = int(len(mesh["elemNodeTags"]) // len(mesh["elemTags"]))
    conn_reshaped = np.asarray(mesh["elemNodeTags"]).reshape(-1, nloc)
    triangles = tag_to_dof[conn_reshaped[:, :3].astype(int)]
    vmax_gain = max(abs(gain.min()), abs(gain.max()), 1e-6)
    cont3 = axes[0, 2].tricontourf(
        x, y, triangles, gain,
        levels=np.linspace(-vmax_gain, vmax_gain, 40),
        cmap="RdBu_r", vmin=-vmax_gain, vmax=vmax_gain, extend="both"
    )
    plt.colorbar(cont3, ax=axes[0, 2], label="delta HRC")
    axes[0, 2].set_aspect("equal")
    axes[0, 2].set_title("Gain HRC (optimise - baseline)", fontsize=10)
    axes[0, 2].set_xlabel("x [m]"); axes[0, 2].set_ylabel("y [m]")

    # [1,0] h(T) comparé
    plot_h_profiles(
        [{"constant": True, "h": 3000.0}, best_params],
        ["baseline h=3000", "optimise (Leidenfrost)"],
        ax=axes[1, 0]
    )

    # [1,1] ΔT(t) comparé : centre - surface_mean au cours du temps
    def _compute_dT_history(res, mesh):
        center = mesh["center_dof"]
        surf   = mesh["surf_dofs"]
        Us = np.array(res["U_history"])
        ts = np.array(res["t_history"])
        dT = Us[:, center] - Us[:, surf].mean(axis=1)
        return ts, dT

    t_b, dT_b = _compute_dT_history(baseline_results, mesh)
    t_o, dT_o = _compute_dT_history(best_results,    mesh)
    axes[1, 1].plot(t_b, dT_b, 'k--', label="baseline")
    axes[1, 1].plot(t_o, dT_o, 'b-',  label="optimise")
    axes[1, 1].set_xlabel("t [s]"); axes[1, 1].set_ylabel("deltaT [C]")
    axes[1, 1].set_title("Gradient thermique centre-surface")
    axes[1, 1].grid(True, alpha=0.3); axes[1, 1].legend()

    # [1,2] Pareto
    plot_pareto_front(results_list, pareto_indices,
                      param_name="h_max", ax=axes[1, 2])

    fig.suptitle(
        f"Optimisation trempe -- compromis durete / gradient   "
        f"(meilleur HRC_surf={best_results['hardness_surface_mean']:.1f}, "
        f"deltaT_max={best_results['delta_T_max']:.0f} C)",
        fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dashboard sauvegarde -> {save_path}")
    return fig

"""
optimize_quench.py
==================
Pipeline d'optimisation paramétrique de la trempe en 2D.

Le maillage et les matrices volumiques (K_vol, M, F_vol) sont assemblés une
seule fois ; chaque simulation ne fait varier que la loi h(T) appliquée via
Robin réassemblé à chaque pas (theta_step_robin_variable_h).
"""
import os
import csv
import time
import numpy as np
import scipy.sparse as sp

import gmsh
from gmsh_utils import gmsh_init, gmsh_finalize, \
                       prepare_quadrature_and_basis, get_jacobians
from blade_geometry import create_blade_section
from stiffness import assemble_stiffness_and_rhs
from mass import assemble_mass
from dirichlet import theta_step_robin_variable_h
from leidenfrost import h_leidenfrost, h_constant
from metallurgy import (compute_martensite_fraction, martensite_to_hardness,
                        estimate_cooling_rates, CCT_DATA)


# ─────────────────────────────────────────────────────────────────────────────
# Paramètres physiques (fixes — cohérents avec main_quench_2D.py)
# ─────────────────────────────────────────────────────────────────────────────
RHO, CP, K_COND = 7800.0, 500.0, 50.0
T0     = 1000.0
T_INF  = 20.0
DT     = 0.01
THETA  = 1.0

A_GEOM = 0.02
B_GEOM = 0.004
CL     = 0.001
ORDER  = 1

NSTEPS_MAX = 5000


# ─────────────────────────────────────────────────────────────────────────────
# Pré-calcul du maillage et des matrices volumiques
# ─────────────────────────────────────────────────────────────────────────────
def build_mesh_data(a=A_GEOM, b=B_GEOM, cl=CL, order=ORDER):
    """
    Initialise Gmsh, génère le maillage, assemble K_vol, M, F_vol et
    pré-calcule toutes les structures nécessaires aux simulations.

    Retourne un dict `mesh` consommé par run_simulation_2d().
    L'appelant doit ensuite appeler gmsh_finalize() à la fin du balayage.
    """
    gmsh_init("quench_optim_2d")

    (elemType,     nodeTags,        nodeCoords,
     elemTags,     elemNodeTags,
     bnd_elemType, bnd_elemTags,    bnd_elemNodeTags,
     bnd_entityTag, surf_tag) = create_blade_section(a=a, b=b, cl=cl, order=order)

    unique_tags = np.unique(elemNodeTags)
    num_dofs    = len(unique_tags)
    max_tag     = int(np.max(nodeTags))
    tag_to_dof  = np.full(max_tag + 1, -1, dtype=int)
    for i, tag in enumerate(unique_tags):
        tag_to_dof[int(tag)] = i

    xi,  w,  N,  gN  = prepare_quadrature_and_basis(elemType,     order)
    xib, wb, Nb, gNb = prepare_quadrature_and_basis(bnd_elemType, order)

    jac,  det,  coords  = get_jacobians(elemType,     xi,  tag=-1)
    jacb, detb, coordsb = get_jacobians(bnd_elemType, xib, tag=-1)

    # K_vol et F_vol (sans Robin)
    K_vol_lil, F_vol = assemble_stiffness_and_rhs(
        elemTags, elemNodeTags,
        jac, det, coords,
        w, N, gN,
        lambda x: float(K_COND),
        lambda x: 0.0,
        tag_to_dof
    )

    # Matrice de masse * rho * cp
    M_raw = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
    M     = (M_raw.tocsr() * (RHO * CP)).tocsr()

    # Coordonnées 2D des DOFs
    dof_xy = np.zeros((num_dofs, 2))
    coords_arr = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)
    for i, tag in enumerate(nodeTags):
        d = tag_to_dof[int(tag)]
        if d >= 0:
            dof_xy[d] = coords_arr[i, :2]

    center_dof = int(np.argmin(np.linalg.norm(dof_xy, axis=1)))
    surf_dofs  = np.unique(tag_to_dof[np.asarray(bnd_elemNodeTags, dtype=int)])
    surf_dofs  = surf_dofs[surf_dofs >= 0]

    # Liste des arêtes uniques pour le calcul du gradient spatial maximal
    nloc = int(len(elemNodeTags) // len(elemTags))
    conn = np.asarray(elemNodeTags, dtype=np.int64).reshape(-1, nloc)[:, :3]
    dofs_tri = tag_to_dof[conn]                       # (ne, 3)
    edges = np.vstack([
        dofs_tri[:, [0, 1]],
        dofs_tri[:, [1, 2]],
        dofs_tri[:, [2, 0]],
    ])
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    edge_lengths = np.linalg.norm(dof_xy[edges[:, 0]] - dof_xy[edges[:, 1]], axis=1)
    # Filtrer arêtes nulles éventuelles
    keep = edge_lengths > 1e-14
    edges = edges[keep]
    edge_lengths = edge_lengths[keep]

    return {
        "elemType":         elemType,
        "nodeTags":         nodeTags,
        "nodeCoords":       nodeCoords,
        "elemTags":         elemTags,
        "elemNodeTags":     elemNodeTags,
        "bnd_elemTags":     bnd_elemTags,
        "bnd_elemNodeTags": bnd_elemNodeTags,
        "detb":             detb,
        "coordsb":          coordsb,
        "wb":               wb,
        "Nb":               Nb,
        "tag_to_dof":       tag_to_dof,
        "num_dofs":         num_dofs,
        "dof_xy":           dof_xy,
        "center_dof":       center_dof,
        "surf_dofs":        surf_dofs,
        "K_vol":            K_vol_lil,
        "F_vol":            F_vol,
        "M":                M,
        "edges":            edges,
        "edge_lengths":     edge_lengths,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Une simulation complète
# ─────────────────────────────────────────────────────────────────────────────
def _make_h_fun(params):
    """params dict -> callable h(T)."""
    if params is None or params.get("constant", False):
        h = params.get("h", 3000.0) if params else 3000.0
        return lambda T: h_constant(T, h=h)

    return lambda T: h_leidenfrost(
        T,
        h_max  = params.get("h_max",  8000.0),
        h_film = params.get("h_film",  200.0),
        T_leid = params.get("T_leid",  300.0),
        T_nucl = params.get("T_nucl",  100.0),
        h_conv = params.get("h_conv", 3000.0),
    )


def run_simulation_2d(params, mesh,
                      record_interval=20,
                      record_interval_dense=2,
                      verbose=False):
    """
    Lance une simulation 2D thermique avec la loi h(T) donnée par `params`.

    params : dict — paramètres Leidenfrost
        keys : h_max, h_film, T_leid, T_nucl, h_conv
        ou {"constant": True, "h": 3000.0} pour le cas de référence.
    mesh   : dict retourné par build_mesh_data()

    Retourne dict :
      hardness_surface_mean, hardness_center, hardness_field,
      delta_T_max, gradient_spatial_max, t_total,
      cooling_rates, U_history, t_history, h_history
    """
    h_fun = _make_h_fun(params)

    M           = mesh["M"]
    K_vol       = mesh["K_vol"]
    F_vol       = mesh["F_vol"]
    bnd_eT      = mesh["bnd_elemTags"]
    bnd_eNT     = mesh["bnd_elemNodeTags"]
    detb        = mesh["detb"]
    coordsb     = mesh["coordsb"]
    wb          = mesh["wb"]
    Nb          = mesh["Nb"]
    tag_to_dof  = mesh["tag_to_dof"]
    num_dofs    = mesh["num_dofs"]
    center_dof  = mesh["center_dof"]
    surf_dofs   = mesh["surf_dofs"]
    edges       = mesh["edges"]
    edge_len    = mesh["edge_lengths"]

    U = np.full(num_dofs, T0, dtype=float)

    U_history = [U.copy()]
    t_history = [0.0]
    h_history = []
    delta_T_max = 0.0
    grad_max    = 0.0

    Ms      = CCT_DATA["Ms"]
    T_start = CCT_DATA["T_austenitisation"]

    t_cur = 0.0
    last_step = 0
    for step in range(NSTEPS_MAX):
        U, h_eff = theta_step_robin_variable_h(
            M, K_vol, F_vol,
            U, DT, THETA,
            bnd_eT, bnd_eNT,
            detb, coordsb, wb, Nb,
            h_fun, T_INF, tag_to_dof,
            surf_dofs=surf_dofs
        )
        t_cur = (step + 1) * DT
        h_history.append(h_eff)

        # Métriques courantes
        delta_T = float(U[center_dof] - U[surf_dofs].mean())
        if delta_T > delta_T_max:
            delta_T_max = delta_T

        dT_edges = np.abs(U[edges[:, 0]] - U[edges[:, 1]]) / edge_len
        cur_grad = float(dT_edges.max())
        if cur_grad > grad_max:
            grad_max = cur_grad

        # Enregistrement adaptatif : dense dans la fenêtre critique [Ms-50, T_start+50]
        T_mean = U.mean()
        in_critical = (T_mean <= T_start + 50.0) and (T_mean >= Ms - 50.0)
        interval = record_interval_dense if in_critical else record_interval

        if (step + 1) % interval == 0:
            U_history.append(U.copy())
            t_history.append(t_cur)

        last_step = step + 1
        if U.mean() <= T_INF + 5.0:
            break

    # Garantir le dernier état dans l'historique
    if t_history[-1] != t_cur:
        U_history.append(U.copy())
        t_history.append(t_cur)

    if verbose:
        print(f"   simulation : {last_step} pas, t = {t_cur:.2f} s")

    # Vitesses de refroidissement → fraction martensitique → HRC
    # estimate_cooling_rates attend une liste de tuples (t, U)
    T_history_tuples = list(zip(t_history, U_history))
    cooling_rates = estimate_cooling_rates(
        T_history_tuples, None,
        Ms=Ms, T_start=T_start
    )
    f_m       = compute_martensite_fraction(cooling_rates)
    hardness  = martensite_to_hardness(f_m, C_content=0.8)

    return {
        "params":                params,
        "hardness_surface_mean": float(hardness[surf_dofs].mean()),
        "hardness_center":       float(hardness[center_dof]),
        "hardness_field":        hardness,
        "martensite_field":      f_m,
        "delta_T_max":           float(delta_T_max),
        "gradient_spatial_max":  float(grad_max),
        "t_total":               float(t_cur),
        "cooling_rates":         cooling_rates,
        "U_history":             U_history,
        "t_history":             t_history,
        "h_history":             np.asarray(h_history, dtype=float),
        "U_final":               U.copy(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Balayage paramétrique
# ─────────────────────────────────────────────────────────────────────────────
def parametric_sweep_leidenfrost(mesh,
                                 h_max_arr=None, T_leid_arr=None,
                                 T_nucl=100.0, h_conv=3000.0, h_film=200.0,
                                 verbose=True):
    """
    Balayage 2D sur (h_max, T_leid). Renvoie la liste des résultats.
    """
    if h_max_arr is None:
        h_max_arr = np.linspace(3000.0, 15000.0, 5)
    if T_leid_arr is None:
        T_leid_arr = np.linspace(200.0, 500.0, 5)

    results = []
    total = len(h_max_arr) * len(T_leid_arr)
    k = 0
    t0 = time.time()
    for h_max in h_max_arr:
        for T_leid in T_leid_arr:
            k += 1
            params = {
                "h_max":  float(h_max),
                "T_leid": float(T_leid),
                "T_nucl": float(T_nucl),
                "h_conv": float(h_conv),
                "h_film": float(h_film),
            }
            res = run_simulation_2d(params, mesh)
            results.append(res)

            if verbose:
                print(f"[sweep {k:>3}/{total}] "
                      f"h_max={h_max:>6.0f} T_leid={T_leid:>5.0f}  "
                      f"-> HRC_surf={res['hardness_surface_mean']:>5.1f}  "
                      f"deltaT_max={res['delta_T_max']:>5.0f} C  "
                      f"t={res['t_total']:>5.1f} s",
                      flush=True)

    if verbose:
        print(f"\nBalayage termine en {time.time() - t0:.1f} s")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pareto
# ─────────────────────────────────────────────────────────────────────────────
def compute_pareto_front(results_list):
    """
    Objectifs :
      - maximiser hardness_surface_mean
      - minimiser delta_T_max

    On ramène à 2 objectifs à minimiser :
      f1 = -hardness_surface_mean
      f2 =  delta_T_max
    """
    n = len(results_list)
    f1 = np.array([-r["hardness_surface_mean"] for r in results_list])
    f2 = np.array([ r["delta_T_max"]           for r in results_list])

    pareto_idx = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if j == i:
                continue
            # j domine i si j <= i sur les deux et < strictement sur un
            if (f1[j] <= f1[i] and f2[j] <= f2[i] and
                (f1[j] < f1[i] or f2[j] < f2[i])):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(i)
    return pareto_idx


def pick_best_compromise(results_list, pareto_idx):
    """
    Sélectionne le point Pareto le plus proche de l'utopie normalisée.
    """
    if not pareto_idx:
        return 0

    # Normalisation des deux objectifs sur l'ensemble Pareto
    f1 = np.array([-results_list[i]["hardness_surface_mean"] for i in pareto_idx])
    f2 = np.array([ results_list[i]["delta_T_max"]           for i in pareto_idx])

    f1_n = (f1 - f1.min()) / max(f1.max() - f1.min(), 1e-12)
    f2_n = (f2 - f2.min()) / max(f2.max() - f2.min(), 1e-12)

    dist = np.sqrt(f1_n**2 + f2_n**2)
    best_in_pareto = int(np.argmin(dist))
    return pareto_idx[best_in_pareto]


# ─────────────────────────────────────────────────────────────────────────────
# Export CSV
# ─────────────────────────────────────────────────────────────────────────────
def export_sweep_csv(results_list, filepath):
    """Sauvegarde les résultats en CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "h_max", "T_leid", "T_nucl", "h_conv",
            "HRC_surf_mean", "HRC_center",
            "delta_T_max", "gradient_max", "t_total"
        ])
        for r in results_list:
            p = r["params"] or {}
            writer.writerow([
                p.get("h_max",  ""),
                p.get("T_leid", ""),
                p.get("T_nucl", ""),
                p.get("h_conv", ""),
                f"{r['hardness_surface_mean']:.3f}",
                f"{r['hardness_center']:.3f}",
                f"{r['delta_T_max']:.2f}",
                f"{r['gradient_spatial_max']:.3e}",
                f"{r['t_total']:.3f}",
            ])
    print(f"CSV ecrit -> {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# Vérification — une simulation unique
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Verification optimize_quench : une simulation Leidenfrost par defaut.\n")
    mesh = build_mesh_data()
    try:
        params = {
            "h_max":  8000.0,
            "T_leid":  300.0,
            "T_nucl":  100.0,
            "h_conv": 3000.0,
            "h_film":  200.0,
        }
        res = run_simulation_2d(params, mesh, verbose=True)
        print("\nResultats :")
        print(f"  HRC surface (moyen) : {res['hardness_surface_mean']:.2f}")
        print(f"  HRC centre          : {res['hardness_center']:.2f}")
        print(f"  deltaT max          : {res['delta_T_max']:.1f} C")
        print(f"  gradient spatial max: {res['gradient_spatial_max']:.2e} C/m")
        print(f"  duree simulation    : {res['t_total']:.2f} s")
    finally:
        gmsh_finalize()

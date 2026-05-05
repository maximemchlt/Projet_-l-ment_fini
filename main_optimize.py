"""
main_optimize.py
================
Pipeline principal d'optimisation multi-objectif de la trempe en 2D.

Usage :
  python main_optimize.py --sweep                   # balayage complet (~15-30 min)
  python main_optimize.py --compare                 # baseline vs meilleur
  python main_optimize.py --single h_max=8000 T_leid=300

Etapes (--sweep) :
  1. Simulation de reference  (h constant = 3000 W/m^2/K)
  2. Balayage parametrique Leidenfrost (5 x 5 = 25 simulations)
  3. Calcul du front de Pareto
  4. Selection du meilleur compromis (distance min a l'utopie normalisee)
  5. Dashboard de comparaison
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from gmsh_utils import gmsh_finalize
from optimize_quench import (build_mesh_data,
                             run_simulation_2d,
                             parametric_sweep_leidenfrost,
                             compute_pareto_front,
                             pick_best_compromise,
                             export_sweep_csv)
from plot_optimization import (plot_hardness_map,
                               plot_h_profiles,
                               plot_pareto_front,
                               plot_optimization_dashboard)


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _parse_kv(args_kv):
    """['h_max=8000', 'T_leid=300'] -> {'h_max':8000, 'T_leid':300}"""
    out = {}
    for kv in args_kv:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def cmd_single(kv):
    """Une simulation Leidenfrost avec params override + carte HRC."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    params = {
        "h_max":  8000.0, "T_leid": 300.0, "T_nucl": 100.0,
        "h_conv": 3000.0, "h_film":  200.0,
    }
    params.update(kv)

    mesh = build_mesh_data()
    try:
        res = run_simulation_2d(params, mesh, verbose=True)
        print(f"\n--- Resultats ---")
        print(f"  HRC surface (moyen) : {res['hardness_surface_mean']:.2f}")
        print(f"  HRC centre          : {res['hardness_center']:.2f}")
        print(f"  deltaT max          : {res['delta_T_max']:.1f} C")
        print(f"  gradient max        : {res['gradient_spatial_max']:.2e} C/m")
        print(f"  duree               : {res['t_total']:.2f} s")

        fig, ax = plt.subplots(figsize=(9, 4))
        cont, _ = plot_hardness_map(
            mesh["elemNodeTags"], mesh["nodeCoords"], mesh["nodeTags"],
            res["hardness_field"], mesh["tag_to_dof"],
            ax=ax, title=f"HRC (h_max={params['h_max']:.0f}, "
                         f"T_leid={params['T_leid']:.0f})"
        )
        plt.colorbar(cont, ax=ax, label="HRC")
        out = os.path.join(RESULTS_DIR, "hardness_map_single.png")
        fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"\nFigure -> {out}")
        plt.show()
    finally:
        gmsh_finalize()


def cmd_compare():
    """Compare baseline (h constant) avec un cas Leidenfrost de reference."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    mesh = build_mesh_data()
    try:
        baseline = run_simulation_2d(
            {"constant": True, "h": 3000.0}, mesh, verbose=True
        )
        leid = run_simulation_2d(
            {"h_max": 8000.0, "T_leid": 300.0, "T_nucl": 100.0,
             "h_conv": 3000.0, "h_film": 200.0},
            mesh, verbose=True
        )

        print("\n--- Baseline (h=3000) ---")
        print(f"  HRC_surf={baseline['hardness_surface_mean']:.2f}  "
              f"deltaT={baseline['delta_T_max']:.0f}  "
              f"t={baseline['t_total']:.1f}")
        print("--- Leidenfrost ---")
        print(f"  HRC_surf={leid['hardness_surface_mean']:.2f}  "
              f"deltaT={leid['delta_T_max']:.0f}  "
              f"t={leid['t_total']:.1f}")

        fig = plot_optimization_dashboard(
            best_params={
                "h_max": 8000.0, "T_leid": 300.0,
                "T_nucl": 100.0, "h_conv": 3000.0, "h_film": 200.0
            },
            best_results=leid,
            baseline_results=baseline,
            mesh=mesh,
            results_list=[baseline, leid],
            pareto_indices=[1],
            save_path=os.path.join(RESULTS_DIR, "dashboard_comparison.png"),
        )
        plt.show()
    finally:
        gmsh_finalize()


def cmd_sweep():
    """Balayage complet + Pareto + dashboard."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    mesh = build_mesh_data()
    try:
        print("\n=== 1) Simulation de reference (h constant = 3000) ===")
        baseline = run_simulation_2d(
            {"constant": True, "h": 3000.0}, mesh, verbose=False
        )
        print(f"   HRC_surf={baseline['hardness_surface_mean']:.2f}  "
              f"deltaT={baseline['delta_T_max']:.0f}  "
              f"t={baseline['t_total']:.1f}")

        print("\n=== 2) Balayage parametrique Leidenfrost ===")
        h_max_arr  = np.linspace(3000.0, 15000.0, 5)
        T_leid_arr = np.linspace(200.0,   500.0, 5)
        results = parametric_sweep_leidenfrost(
            mesh, h_max_arr=h_max_arr, T_leid_arr=T_leid_arr
        )

        print("\n=== 3) Front de Pareto ===")
        pareto_idx = compute_pareto_front(results)
        print(f"   {len(pareto_idx)} points non-domines :")
        for i in pareto_idx:
            p = results[i]["params"]
            print(f"     h_max={p['h_max']:>5.0f} T_leid={p['T_leid']:>5.0f}  "
                  f"HRC_surf={results[i]['hardness_surface_mean']:.1f}  "
                  f"deltaT={results[i]['delta_T_max']:.0f}")

        print("\n=== 4) Meilleur compromis ===")
        best_i      = pick_best_compromise(results, pareto_idx)
        best        = results[best_i]
        best_params = best["params"]
        print(f"   selection -> h_max={best_params['h_max']:.0f}  "
              f"T_leid={best_params['T_leid']:.0f}  "
              f"HRC_surf={best['hardness_surface_mean']:.1f}  "
              f"deltaT={best['delta_T_max']:.0f}")

        # Export CSV
        export_sweep_csv(results, os.path.join(RESULTS_DIR, "sweep_results.csv"))

        # Figures individuelles
        fig_p = plt.figure(figsize=(8, 6))
        ax_p  = fig_p.add_subplot(1, 1, 1)
        plot_pareto_front(results, pareto_idx, param_name="h_max", ax=ax_p)
        fig_p.tight_layout()
        fig_p.savefig(os.path.join(RESULTS_DIR, "pareto_front.png"),
                      dpi=150, bbox_inches='tight')

        fig_h = plt.figure(figsize=(8, 5))
        ax_h  = fig_h.add_subplot(1, 1, 1)
        plot_h_profiles(
            [{"constant": True, "h": 3000.0}, best_params],
            ["baseline h=3000", "optimise (Leidenfrost)"],
            ax=ax_h
        )
        fig_h.tight_layout()
        fig_h.savefig(os.path.join(RESULTS_DIR, "h_profiles.png"),
                      dpi=150, bbox_inches='tight')

        fig_m = plt.figure(figsize=(9, 4))
        ax_m  = fig_m.add_subplot(1, 1, 1)
        cont, _ = plot_hardness_map(
            mesh["elemNodeTags"], mesh["nodeCoords"], mesh["nodeTags"],
            best["hardness_field"], mesh["tag_to_dof"],
            ax=ax_m, title="HRC -- meilleur compromis"
        )
        plt.colorbar(cont, ax=ax_m, label="HRC")
        fig_m.tight_layout()
        fig_m.savefig(os.path.join(RESULTS_DIR, "hardness_map_best.png"),
                      dpi=150, bbox_inches='tight')

        # Dashboard global
        plot_optimization_dashboard(
            best_params=best_params,
            best_results=best,
            baseline_results=baseline,
            mesh=mesh,
            results_list=results,
            pareto_indices=pareto_idx,
            save_path=os.path.join(RESULTS_DIR, "dashboard_comparison.png"),
        )

        print(f"\nFigures et CSV ecrits dans {RESULTS_DIR}")
        plt.show()
    finally:
        gmsh_finalize()


def main():
    ap = argparse.ArgumentParser(description="Optimisation trempe 2D")
    ap.add_argument("--sweep",   action="store_true",
                    help="balayage parametrique complet")
    ap.add_argument("--compare", action="store_true",
                    help="comparaison baseline vs Leidenfrost reference")
    ap.add_argument("--single",  nargs="*", default=None,
                    help="simulation unique : k=v ...")
    args = ap.parse_args()

    if args.sweep:
        cmd_sweep()
    elif args.compare:
        cmd_compare()
    elif args.single is not None:
        cmd_single(_parse_kv(args.single))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

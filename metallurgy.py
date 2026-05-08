"""
metallurgy.py
=============
Couplage métallurgique simplifié pour acier eutectoïde 1080 (C ~ 0.8%).

Permet, à partir d'une histoire thermique T(x, t), d'estimer :
  - la vitesse moyenne de refroidissement dans la plage [Ms, T_aust]
  - la fraction martensitique locale finale
  - la dureté HRC locale (loi Maynier simplifiée)
"""
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Diagramme TRC simplifié — acier 1080
# ─────────────────────────────────────────────────────────────────────────────
CCT_DATA = {
    "Vk_martensite":     150.0,   # [°C/s] vitesse critique 100% martensite
    "Vk_bainite":         20.0,   # [°C/s] vitesse limite bainite
    "Vk_perlite":          2.0,   # [°C/s] vitesse limite perlite
    "Ms":                250.0,   # [°C]   début transformation martensitique
    "Mf":                 50.0,   # [°C]   fin transformation martensitique
    "T_austenitisation": 850.0,   # [°C]   température de départ austénite
}


def compute_martensite_fraction(cooling_rates, cct_data=CCT_DATA):
    """
    Fraction martensitique finale par DOF.

    Modèle simplifié :
      cooling_rate >= Vk_martensite       → fraction = 1.0
      cooling_rate <= Vk_perlite          → fraction = 0.0
      sinon : interpolation linéaire entre les deux seuils
    """
    Vk_m = cct_data["Vk_martensite"]
    Vk_p = cct_data["Vk_perlite"]

    rates = np.asarray(cooling_rates, dtype=float)
    frac  = np.clip((rates - Vk_p) / (Vk_m - Vk_p), 0.0, 1.0)
    return frac


def martensite_to_hardness(martensite_fraction, C_content=0.8):
    """
    Dureté HRC à partir de la fraction martensitique.

    Loi empirique Maynier simplifiée pour acier C-pur :
      HRC = 64 * sqrt(C) * f_m + 20 * (1 - f_m)
    """
    f_m = np.clip(np.asarray(martensite_fraction, dtype=float), 0.0, 1.0)
    HRC_m = 64.0 * np.sqrt(C_content)
    HRC_p = 20.0
    return HRC_m * f_m + HRC_p * (1.0 - f_m)


def estimate_cooling_rates(T_history, times, Ms=250.0, T_start=850.0):
    """
    Pour chaque DOF, calcule la vitesse de refroidissement MAXIMALE
    dans la fenêtre critique [T_start=850°C → Ms=250°C].

    C'est cette vitesse qui détermine la microstructure finale,
    pas la vitesse moyenne sur toute la simulation.

    T_history : liste de tuples (t, U) enregistrés pendant la simulation
    times     : ignoré (les temps sont dans T_history[i][0])

    Retourne cooling_rates array (num_dofs,) en °C/s, valeurs positives.
    """
    if len(T_history) < 2:
        raise ValueError("T_history doit contenir au moins 2 snapshots")

    times_arr = np.array([snap[0] for snap in T_history])   # (n_snaps,)
    T_arr     = np.vstack([snap[1] for snap in T_history])  # (n_snaps, num_dofs)

    num_dofs = T_arr.shape[1]
    cooling_rates = np.zeros(num_dofs)

    for dof in range(num_dofs):
        T_dof = T_arr[:, dof]   # évolution temporelle de ce DOF

        # Indices où T est dans la fenêtre critique
        in_window = (T_dof <= T_start) & (T_dof >= Ms)

        if np.sum(in_window) < 2:
            # Pas assez de points dans la fenêtre → pas de transfo possible
            cooling_rates[dof] = 0.0
            continue

        t_win = times_arr[in_window]
        T_win = T_dof[in_window]

        # Pente max (en valeur absolue) sur paires consécutives dans la fenêtre
        dT = np.diff(T_win)   # négatif (refroidissement)
        dt = np.diff(t_win)   # positif

        valid = dt > 1e-10
        if not np.any(valid):
            cooling_rates[dof] = 0.0
            continue

        rates = np.abs(dT[valid] / dt[valid])   # °C/s positif
        cooling_rates[dof] = float(np.max(rates))

    return cooling_rates


if __name__ == "__main__":
    # ─── 1. Table cooling_rate -> HRC ────────────────────────────────────────
    test_rates = np.array([1.0, 10.0, 50.0, 200.0, 500.0])
    f_m = compute_martensite_fraction(test_rates)
    HRC = martensite_to_hardness(f_m)

    print("Table cooling_rate -> HRC --------------------------------------")
    print(f"{'V [C/s]':>10} | {'f_m':>8} | {'HRC':>8}")
    print("-" * 38)
    for v, f, h in zip(test_rates, f_m, HRC):
        print(f"{v:>10.1f} | {f:>8.3f} | {h:>8.2f}")
    print("----------------------------------------------------------------")

    # ─── 2. Test de cohérence sur T_history synthétique ─────────────────────
    print("\nTest de coherence estimate_cooling_rates ----------------------")
    dt_test = 0.05
    t_test  = np.arange(0, 60, dt_test)
    # Refroidissement linéaire 1000°C -> 20°C
    T_fast = np.maximum(1000 - 196 * t_test, 20.0)   # ~196 °C/s -> martensite
    T_slow = np.maximum(1000 -  16 * t_test, 20.0)   # ~16  °C/s -> perlite

    T_history_test = [(t, np.array([T_fast[i], T_slow[i]]))
                      for i, t in enumerate(t_test)]

    rates = estimate_cooling_rates(T_history_test, None)
    print(f"Vitesse DOF rapide : {rates[0]:6.1f} C/s  (attendu ~196)")
    print(f"Vitesse DOF lent   : {rates[1]:6.1f} C/s  (attendu ~16)")

    fracs = compute_martensite_fraction(rates)
    hrcs  = martensite_to_hardness(fracs)
    print(f"HRC DOF rapide     : {hrcs[0]:6.1f}      (attendu ~60-65)")
    print(f"HRC DOF lent       : {hrcs[1]:6.1f}      (attendu ~20-25)")
    print("----------------------------------------------------------------")

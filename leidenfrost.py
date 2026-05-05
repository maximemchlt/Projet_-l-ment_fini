"""
leidenfrost.py
==============
Loi h(T) physique pour la trempe à l'eau, prenant en compte les trois régimes
d'échange convectif :

  1. Ébullition en film (T élevée)  : h faible (vapeur isole la pièce)
  2. Transition Leidenfrost          : h croît brutalement
  3. Ébullition nucléée / convection : h fort puis stabilisé

La loi est vectorisée : T peut être un scalaire ou un array numpy.
"""
import numpy as np


def h_leidenfrost(T,
                  h_max=8000.0,
                  h_film=200.0,
                  T_leid=300.0,
                  T_nucl=100.0,
                  h_conv=3000.0):
    """
    h(T) en W/m²K à 5 paramètres.

    Régimes
    -------
    - T > T_leid          : régime film, h = h_film
    - T_nucl < T < T_leid : transition, h interpolé linéairement
                            de h_film (en T_leid) à h_max (en T_nucl)
    - T < T_nucl          : régime nucléé, h interpolé linéairement
                            de h_max (en T_nucl) à h_conv (en T_inf=20°C)

    Paramètres
    ----------
    h_max  : pic de h au point Leidenfrost effondré [W/m²K]
    h_film : h en régime film vapeur (haute T) [W/m²K]
    T_leid : température de Leidenfrost [°C]
    T_nucl : température de transition vers convection [°C]
    h_conv : h en convection forcée basse T [W/m²K]
    """
    T_arr = np.asarray(T, dtype=float)
    scalar_input = (T_arr.ndim == 0)
    T_arr = np.atleast_1d(T_arr)

    # Points de contrôle (T décroissant ⇒ T croissant pour np.interp)
    T_low = 20.0  # référence basse pour h_conv
    Tp = np.array([T_low, T_nucl, T_leid, max(T_leid + 200.0, 1500.0)])
    hp = np.array([h_conv, h_max, h_film, h_film])

    h = np.interp(T_arr, Tp, hp)

    if scalar_input:
        return float(h[0])
    return h


def h_constant(T, h=3000.0):
    """h constant — référence pour comparaison."""
    T_arr = np.asarray(T, dtype=float)
    if T_arr.ndim == 0:
        return float(h)
    return np.full_like(T_arr, float(h))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T = np.linspace(20.0, 1000.0, 500)
    h = h_leidenfrost(T)
    h_cst = h_constant(T)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T, h, 'b-', linewidth=2, label='Leidenfrost (defaut)')
    ax.plot(T, h_cst, 'k--', linewidth=1.5, label='h constant = 3000')

    # Annotations des seuils
    ax.axvline(300.0, color='orange', linestyle=':', alpha=0.7,
               label='T_leid = 300 C')
    ax.axvline(100.0, color='green',  linestyle=':', alpha=0.7,
               label='T_nucl = 100 C')

    ax.set_xlabel('Temperature T [C]')
    ax.set_ylabel('h [W/m^2/K]')
    ax.set_title('Loi h(T) -- effet Leidenfrost')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

"""
leidenfrost.py
==============
Loi h(T) physique pour la trempe à l'eau, prenant en compte les trois régimes
d'échange convectif :

  1. Ébullition en film (T élevée)  : h faible (vapeur isole la pièce)
  2. Transition Leidenfrost          : h croît brutalement
  3. Ébullition nucléée / convection : h fort puis stabilisé

On utilise une boucle pour parcourir les températures dans T. 

"""
import numpy as np

def h_constant(T, h=3000.0):
    """h constant — référence pour comparaison."""
    T_arr = np.asarray(T, dtype=float)
    if T_arr.ndim == 0:
        return float(h)
    return np.full_like(T_arr, float(h))

def h_leidenfrost(T, h_max=8000.0, h_film=200.0, T_leid=300.0, T_nucl=100.0, h_conv=3000.0):
    
    """
    h(T) en W/m²K — version scalaire.
    T : température de la pièce en °C
    """
    
    T_low = 20.0
    if T > T_leid:
        return h_film
    elif T > T_nucl:
        pente = (h_max - h_film) / (T_nucl - T_leid)
        return h_film + pente * (T - T_leid)
    else:
        pente = (h_conv - h_max) / (T_low - T_nucl)
        return h_max + pente * (T - T_nucl)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T = np.linspace(20.0, 1000.0, 500)
    h = np.array([h_leidenfrost(t) for t in T])  # ← boucle obligatoire
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

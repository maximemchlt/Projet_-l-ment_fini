# mechanical.py
"""
Modèle mécanique simplifié pour l'analyse des contraintes thermiques
lors de la trempe.

Hypothèse : contrainte thermique locale en "plane stress" isotrope.
Pour un point matériel contraint biaxialement par ses voisins,

        sigma_th(x) ~= - E*alpha / (1-nu) * ( T(x) - T_ref )

Avec T_ref = température moyenne instantanée de la pièce :
    - sigma > 0  (traction)    là où  T(x) < T_moy   -> typiquement la peau
    - sigma < 0  (compression) là où  T(x) > T_moy   -> typiquement le cœur

Les fissures de trempe nucléent en peau parce que les aciers durs cèdent
en traction bien avant de céder en compression.

Ce n'est PAS une vraie résolution d'élasticité (ça reste un post-traitement
nodal du champ de température), mais c'est cohérent en ordre de grandeur
avec un calcul élastique linéaire complet, et ça coûte zéro effort de calcul.
"""

import numpy as np


# ── Propriétés mécaniques typiques d'un acier à outils ───────────────────
STEEL_PROPS = {
    "E":         200.0e9,   # module d'Young            [Pa]
    "nu":        0.30,      # coefficient de Poisson    [-]
    "alpha":     12.0e-6,   # dilatation thermique      [1/K]
    "sigma_rup": 600.0e6,   # résistance en traction    [Pa]
}


def thermal_stress_coeff(E, nu, alpha):
    """
    Coefficient pré-calculé beta = E * alpha / (1 - nu).

    Multiplie l'écart de température (T - T_ref) pour donner
    la contrainte thermique en plane stress.
    """
    return E * alpha / (1.0 - nu)


def thermal_stress_field(U, T_ref=None, beta=None,
                         E=None, nu=None, alpha=None):
    """
    Champ de contrainte thermique nodal [Pa] :

        sigma(x) = - beta * ( T(x) - T_ref )

    Paramètres
    ----------
    U : ndarray
        Champ de température nodal (°C ou K — seul l'écart compte).
    T_ref : float, optionnel
        Température de référence. Par défaut : moyenne instantanée de U.
        On peut aussi passer la température initiale T0 si on veut une
        référence "état de départ sans contrainte".
    beta : float, optionnel
        Coefficient E*alpha/(1-nu). Si non fourni, calculé depuis E, nu, alpha.
    E, nu, alpha : float, optionnels
        Utilisés uniquement si beta n'est pas fourni.

    Retour
    ------
    sigma : ndarray
        Contrainte aux nœuds [Pa]. Convention : positif = traction.
    """
    if T_ref is None:
        T_ref = float(np.mean(U))
    if beta is None:
        if E is None or nu is None or alpha is None:
            raise ValueError(
                "Fournir soit beta, soit le triplet (E, nu, alpha)."
            )
        beta = thermal_stress_coeff(E, nu, alpha)
    return -beta * (U - T_ref)


def stress_indicators(sigma):
    """
    Indicateurs scalaires d'un champ de contrainte.

    Retour
    ------
    dict :
        sigma_max  : contrainte de traction maximale [Pa]   (>0 attendu en peau)
        sigma_min  : contrainte de compression min   [Pa]   (<0 attendu au cœur)
        idx_max    : indice du nœud le plus en traction
        idx_min    : indice du nœud le plus en compression
        sigma_amp  : amplitude max(sigma) - min(sigma) [Pa]
    """
    idx_max = int(np.argmax(sigma))
    idx_min = int(np.argmin(sigma))
    return {
        "sigma_max": float(sigma[idx_max]),
        "sigma_min": float(sigma[idx_min]),
        "idx_max":   idx_max,
        "idx_min":   idx_min,
        "sigma_amp": float(sigma[idx_max] - sigma[idx_min]),
    }


def crack_risk(sigma_max, sigma_rup):
    """
    True si la contrainte de traction max dépasse la limite admissible.
    Critère type Rankine (contrainte principale max) — adapté aux
    ruptures fragiles, donc cohérent avec un acier trempé qui casse net.
    """
    return sigma_max > sigma_rup


def safety_factor(sigma_max, sigma_rup):
    """
    Facteur de sécurité : sigma_rup / max(sigma_max, eps).
    < 1 -> rupture probable.
    """
    return sigma_rup / max(sigma_max, 1.0)

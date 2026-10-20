"""Planet catalog: the literature priors used by the paper-era rv_lc_gp_*.py scripts.

Every planet had the same prior in every script it appeared in, so one table
reproduces all five published configurations.
"""

# P [d]; logP_sigma [dex]; t0 [BJD]; t0_sigma [d] (period and phase priors, used by every run);
# mass [M_earth] and logM_sigma [dex]: the paper's log-mass prior, used only with
#   mass_prior="lognormal"; the suite's runs use the mass only as a starting point (as K);
# e_init: starting eccentricity (secosw starts at sqrt(e_init), sesinw at 0.01)
PLANETS = {
    "b": dict(P=3.0929385, logP_sigma=0.0001, t0=2459263.91706, t0_sigma=0.05, mass=4.3, logM_sigma=0.4, e_init=0.01),
    "c": dict(P=6.765047, logP_sigma=0.0001, t0=2459435.90931, t0_sigma=0.05, mass=4.0, logM_sigma=0.5, e_init=0.103),
    "f": dict(P=22.72, logP_sigma=0.05, t0=2455233.17, t0_sigma=10.0, mass=7.4, logM_sigma=0.5, e_init=0.2),
    "d": dict(P=46.73, logP_sigma=0.05, t0=2456744.56, t0_sigma=10.0, mass=14.6, logM_sigma=0.6, e_init=0.2),
    "g": dict(P=94.26, logP_sigma=0.05, t0=2455964.0, t0_sigma=20.0, mass=11.0, logM_sigma=0.6, e_init=0.2),
    "i": dict(P=192.32, logP_sigma=0.05, t0=2455167.8, t0_sigma=20.0, mass=15.0, logM_sigma=0.6, e_init=0.2),
    "h": dict(P=2180.0, logP_sigma=0.1, t0=2456840.0, t0_sigma=200.0, mass=146.0, logM_sigma=0.5, e_init=0.2),
}

# Transiting planets come first in every configuration
TRANSITING = {
    "b": dict(depth_ppt=0.358, b=0.9171),
    "c": dict(depth_ppt=0.318, b=0.801),
}

# Planets whose existence is disputed in the literature
DISPUTED = ("f", "g", "i")

# Published configurations (Table 6 of the draft)
CONFIGS = {
    "M4": ["b", "c", "d", "h"],            # Motalebi et al. 2015 (adopted)
    "G4": ["b", "c", "f", "d"],            # Gillon et al. 2017
    "P5": ["b", "c", "d", "i", "h"],       # 5-Planet
    "V6": ["b", "c", "f", "d", "g", "h"],  # Vogt et al. 2015
    "H6": ["b", "c", "f", "d", "i", "h"],  # Harada et al. 2025
    # Internal tests: the adopted model plus a single disputed planet
    "M4+f": ["b", "c", "f", "d", "h"],
    "M4+g": ["b", "c", "d", "g", "h"],
}


def injected_planet(P, t0):
    """Catalog entry for a synthetic planet 'x' used in injection-recovery runs.
    Same prior widths as planet f."""
    return dict(P=P, logP_sigma=0.05, t0=t0, t0_sigma=10.0, mass=7.4, logM_sigma=0.5, e_init=0.2)

"""Every run in the suite, in the order they are started.

Groups
  main   the five configurations reported in the paper
  test   internal checks that guide the wording, not reported: the adopted model plus one
         disputed planet, and injection-recovery at and away from the rotation period

All runs: K ~ U(0, 20) m/s for every planet; f, g, i (and the injected planet x) on circular
orbits; P_rot ~ N(22.56, 0.29) d.
"""
import pathlib
import zlib
from dataclasses import dataclass

import numpy as np

from .data import X_REF
from .model import K_from_mass
from .planets import CONFIGS, DISPUTED, PLANETS, injected_planet

SAMPLING = dict(chains=4, tune=1500, draws=1500, target_accept=0.98)
QUICK_SAMPLING = dict(chains=2, tune=10, draws=10, target_accept=0.9)

INIT_FILE = pathlib.Path(__file__).resolve().parent.parent / "data" / "init" / "m4_paper_posterior_draws.npz"
K_INJ = 2.3                     # m/s, about the K reported for planet f
T0_INJ = 2457000.0              # BJD, inside the dense HARPS-N/APF coverage


@dataclass(frozen=True)
class Run:
    id: str
    group: str
    planets: tuple
    note: str
    injection: tuple = None      # (P [d], K [m/s], t0 [BJD]) added to the RVs before fitting

    @property
    def circular(self):
        """Disputed and injected planets are fitted on circular orbits."""
        return tuple(p for p in self.planets if p in DISPUTED or p == "x")

    @property
    def seed(self):
        return zlib.crc32(self.id.encode())

    @property
    def catalog(self):
        if self.injection:
            return {**PLANETS, "x": injected_planet(self.injection[0], T0_INJ)}
        return PLANETS


def _x(planets):
    """Insert the synthetic planet 'x' after c (both test periods lie between c and d)."""
    i = planets.index("c") + 1
    return tuple(planets[:i]) + ("x",) + tuple(planets[i:])


M4 = CONFIGS["M4"]
RUNS = [
    Run("main_M4", "main", tuple(M4), "4-Planet-M: b c d h (adopted)"),
    Run("main_H6", "main", tuple(CONFIGS["H6"]), "6-Planet-H: b c f d i h"),
    Run("main_G4", "main", tuple(CONFIGS["G4"]), "4-Planet-G: b c f d"),
    Run("main_P5", "main", tuple(CONFIGS["P5"]), "5-Planet: b c d i h"),
    Run("main_V6", "main", tuple(CONFIGS["V6"]), "6-Planet-V: b c f d g h"),
    Run("test_M4+f", "test", tuple(CONFIGS["M4+f"]), "b c d h + f: does f alone improve the fit?"),
    Run("test_inject_22.56", "test", _x(M4), "K = 2.3 m/s planet injected AT P_rot = 22.56 d: does the GP absorb it?",
        injection=(22.56, K_INJ, T0_INJ)),
    Run("test_M4+g", "test", tuple(CONFIGS["M4+g"]), "b c d h + g: does g alone improve the fit?"),
    Run("test_inject_27.0", "test", _x(M4), "control: the same injection at 27.0 d, away from P_rot",
        injection=(27.0, K_INJ, T0_INJ)),
]
RUNS_BY_ID = {r.id: r for r in RUNS}
assert len(RUNS_BY_ID) == len(RUNS)


def chain_initvals(run, n_chains):
    """Start chain c from draw c of the converged paper 4-planet posterior.

    Planets in the 4-planet model start from that draw (its mass converted to K). Other
    planets start at their literature period, phase and mass (converted to K); the prior on
    K is uniform, so the starting point does not favour them.
    """
    init = np.load(INIT_FILE)
    m4 = list(init["planets"])
    out = []
    for c in range(n_chains):
        d = {k: init[k][c] for k in init.files if k != "planets"}
        per_planet = {}
        for p in run.planets:
            if p in m4:
                j = m4.index(p)
                e = d["secosw"][j] ** 2 + d["sesinw"][j] ** 2
                K = K_from_mass(10 ** d["log_m_pl"][j], 10 ** d["log_period"][j], d["m_star"], e)
                per_planet[p] = dict(t0=d["t0"][j], log_period=d["log_period"][j], secosw=d["secosw"][j],
                                     sesinw=d["sesinw"][j], K=K)
            else:
                cat = run.catalog[p]
                per_planet[p] = dict(t0=cat["t0"] - X_REF, log_period=np.log10(cat["P"]),
                                     secosw=np.sqrt(cat["e_init"]), sesinw=0.01,
                                     K=K_from_mass(cat["mass"], cat["P"], d["m_star"]))
        ecc_planets = [p for p in run.planets if p not in run.circular]
        iv = {k: v for k, v in d.items() if k not in ("t0", "log_period", "log_m_pl", "secosw", "sesinw")}
        iv["t0"] = np.array([per_planet[p]["t0"] for p in run.planets])
        iv["log_period"] = np.array([per_planet[p]["log_period"] for p in run.planets])
        iv["K_pl"] = np.array([per_planet[p]["K"] for p in run.planets])
        for k in ("secosw", "sesinw"):
            iv[k] = np.array([per_planet[p][k] for p in ecc_planets])
        out.append(iv)
    return out

"""Every run in the suite, in the order they are started (most informative first).

Groups
  paper   the five published configurations, same priors as the paper
  nested  the adopted model plus one disputed planet (direct tests of f and g)
  kprior  disputed planets on circular orbits with K ~ U(0, 10 m/s), so they can vanish
  prot    P_rot prior tests and injection-recovery around the rotation period
"""
import pathlib
import zlib
from dataclasses import dataclass

import numpy as np

from .model import TESS_PROT
from .planets import CONFIGS, PLANETS, injected_planet

SAMPLING = dict(chains=4, tune=1500, draws=1500, target_accept=0.98)
QUICK_SAMPLING = dict(chains=2, tune=10, draws=10, target_accept=0.9)

INIT_FILE = pathlib.Path(__file__).resolve().parent.parent / "data" / "init" / "m4_paper_posterior_draws.npz"
K_INIT = (0.5, 1.5, 2.5, 3.5)   # per-chain starting K [m/s] for K-prior planets: spread to expose multimodality
K_INJ = 2.3                     # m/s, about the K reported for planet f
T0_INJ = 2457000.0              # BJD, inside the dense HARPS-N/APF coverage
BROAD_PROT = ("uniform", 15.0, 50.0)
ALT_PROT = ("normal", 41.2, 2.8)  # the 41-d test runs (Table 9 of the draft)


@dataclass(frozen=True)
class Run:
    id: str
    group: str
    planets: tuple
    note: str
    k_planets: tuple = ()
    prot_prior: tuple = TESS_PROT
    prot_init: tuple = None      # per-chain initial P_rot; None = from the 4-planet posterior
    injection: tuple = None      # (P [d], K [m/s], t0 [BJD]) added to the RVs before fitting

    @property
    def seed(self):
        return zlib.crc32(self.id.encode())

    @property
    def catalog(self):
        if any(p == "x" for p in self.planets):
            P = self.injection[0] if self.injection else float(self.id.split("_")[-1])
            return {**PLANETS, "x": injected_planet(P, T0_INJ)}
        return PLANETS


def _x(planets):
    """Insert the synthetic planet 'x' after c (all test periods lie between c and d)."""
    i = planets.index("c") + 1
    return tuple(planets[:i]) + ("x",) + tuple(planets[i:])


M4 = CONFIGS["M4"]
RUNS = [
    Run("paper_M4", "paper", tuple(M4), "adopted 4-planet model (b c d h)"),
    Run("kprior_M4+f", "kprior", tuple(CONFIGS["M4+f"]), "M4 + f, f circular with K ~ U(0,10)", k_planets=("f",)),
    Run("nested_M4+f", "nested", tuple(CONFIGS["M4+f"]), "M4 + f, paper priors"),
    Run("inject_22.56", "prot", _x(M4), "inject K=2.3 at P=P_rot=22.56 d; can a planet AT P_rot be recovered?",
        k_planets=("x",), injection=(22.56, K_INJ, T0_INJ)),
    Run("prot_broad_M4", "prot", tuple(M4), "M4 with P_rot ~ U(15,50); chains start at 22.56 and 42",
        prot_prior=BROAD_PROT, prot_init=(22.56, 22.56, 42.0, 42.0)),
    Run("prot_broad_M4+f", "prot", tuple(CONFIGS["M4+f"]), "M4 + f (K prior) with P_rot ~ U(15,50)",
        k_planets=("f",), prot_prior=BROAD_PROT, prot_init=(22.56, 22.56, 42.0, 42.0)),
    Run("kprior_M4+g", "kprior", tuple(CONFIGS["M4+g"]), "M4 + g, g circular with K ~ U(0,10)", k_planets=("g",)),
    Run("nested_M4+g", "nested", tuple(CONFIGS["M4+g"]), "M4 + g, paper priors"),
    Run("inject_27.0", "prot", _x(M4), "inject K=2.3 at 27.0 d (away from P_rot)",
        k_planets=("x",), injection=(27.0, K_INJ, T0_INJ)),
    Run("null_27.0", "prot", _x(M4), "no injection, same model as inject_27.0 (false-positive check)",
        k_planets=("x",)),
    Run("kprior_P5", "kprior", tuple(CONFIGS["P5"]), "5-Planet (b c d i h), i with K prior", k_planets=("i",)),
    Run("kprior_G4", "kprior", tuple(CONFIGS["G4"]), "Gillon (b c f d), f with K prior", k_planets=("f",)),
    Run("paper_G4", "paper", tuple(CONFIGS["G4"]), "4-Planet-G (b c f d), paper priors"),
    Run("paper_P5", "paper", tuple(CONFIGS["P5"]), "5-Planet (b c d i h), paper priors"),
    Run("paper_V6", "paper", tuple(CONFIGS["V6"]), "6-Planet-V (b c f d g h), paper priors"),
    Run("paper_H6", "paper", tuple(CONFIGS["H6"]), "6-Planet-H (b c f d i h), paper priors"),
    Run("kprior_V6", "kprior", tuple(CONFIGS["V6"]), "6-Planet-V, f and g with K prior", k_planets=("f", "g")),
    Run("kprior_H6", "kprior", tuple(CONFIGS["H6"]), "6-Planet-H, f and i with K prior", k_planets=("f", "i")),
    Run("prot_41_M4", "prot", tuple(M4), "M4 with P_rot ~ N(41.2, 2.8)",
        prot_prior=ALT_PROT, prot_init=(41.2,) * 4),
    Run("prot_41_M4+f", "prot", tuple(CONFIGS["M4+f"]), "M4 + f (K prior) with P_rot ~ N(41.2, 2.8)",
        k_planets=("f",), prot_prior=ALT_PROT, prot_init=(41.2,) * 4),
    Run("inject_33.0", "prot", _x(M4), "inject K=2.3 at 33.0 d (away from P_rot)",
        k_planets=("x",), injection=(33.0, K_INJ, T0_INJ)),
    Run("null_33.0", "prot", _x(M4), "no injection, same model as inject_33.0 (false-positive check)",
        k_planets=("x",)),
]
RUNS_BY_ID = {r.id: r for r in RUNS}
assert len(RUNS_BY_ID) == len(RUNS)


def chain_initvals(run, n_chains):
    """Start chain c from draw c of the converged paper 4-planet posterior.

    Planets in the 4-planet model start from that draw; other planets start at their
    literature values (K-prior planets at K_INIT[c]).
    """
    init = np.load(INIT_FILE)
    m4 = list(init["planets"])
    out = []
    for c in range(n_chains):
        d = {k: init[k][c] for k in init.files if k != "planets"}
        per_planet = {}
        for p in run.planets:
            cat = run.catalog[p]
            if p in m4 and p not in run.k_planets:
                j = m4.index(p)
                per_planet[p] = {k: d[k][j] for k in ("t0", "log_period", "log_m_pl", "secosw", "sesinw")}
            else:
                per_planet[p] = dict(t0=cat["t0"] - 2450000.0, log_period=np.log10(cat["P"]),
                                     log_m_pl=np.log10(cat["mass"]), secosw=np.sqrt(cat["e_init"]), sesinw=0.01)
        kep = [p for p in run.planets if p not in run.k_planets]
        iv = {k: v for k, v in d.items() if k not in ("t0", "log_period", "log_m_pl", "secosw", "sesinw")}
        iv["t0"] = np.array([per_planet[p]["t0"] for p in run.planets])
        iv["log_period"] = np.array([per_planet[p]["log_period"] for p in run.planets])
        for k in ("log_m_pl", "secosw", "sesinw"):
            iv[k] = np.array([per_planet[p][k] for p in kep])
        if run.k_planets:
            iv["K_circ"] = np.full(len(run.k_planets), K_INIT[c % len(K_INIT)])
        if run.prot_init is not None:
            iv["P_rot"] = np.float64(run.prot_init[c % len(run.prot_init)])
        out.append(iv)
    return out

"""Joint TESS + RV model with a shared-period celerite2 RotationTerm GP.

With k_planets=() and the default P_rot prior this is exactly the model of the
paper-era rv_lc_gp_*.py scripts (tests/check_setup.py verifies the log-probability
matches them to machine precision). Options:

  k_planets   planets fitted on circular orbits with K ~ Uniform(0, k_max) m/s instead
              of a log-normal mass prior, so the planet can vanish (K = 0).
  prot_prior  ("normal", mu, sigma) or ("uniform", lower, upper) for P_rot [d].
"""
import dataclasses

import astropy.constants as const
import astropy.units as u
import exoplanet as xo
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from celerite2.pymc import GaussianProcess, terms

from .data import X_REF
from .planets import PLANETS, TRANSITING

G = const.G.cgs.value
M_SUN = const.M_sun.cgs.value
M_EARTH = const.M_earth.cgs.value
TEXP = 120.0 / 86400.0
TESS_PROT = ("normal", 22.56, 0.29)  # SAP periodogram, Section 4


def mass_from_K(K, P_days, m_star, ecc=0.0):
    """Minimum mass [M_earth] for semi-amplitude K [m/s] (m_planet << m_star)."""
    return (K * 100.0 * np.sqrt(1 - ecc ** 2) * (P_days * 86400.0 / (2 * np.pi * G)) ** (1 / 3)
            * (m_star * M_SUN) ** (2 / 3) / M_EARTH)


def inject_rv_signal(data, P, K, t0, m_star=0.763, r_star=0.792):
    """Return a copy of data with a circular Keplerian of semi-amplitude K added to the RVs.
    The signal is computed with exoplanet itself, so it follows the fitted model's conventions."""
    orbit = xo.orbits.KeplerianOrbit(
        period=P, t0=t0 - X_REF, b=0.0, m_star=m_star, r_star=r_star,
        m_planet=xo.units.with_unit(mass_from_K(K, P, m_star), u.Mearth))
    signal = orbit.get_radial_velocity(data.x_rv).eval()
    return dataclasses.replace(data, y_rv=data.y_rv + signal)


def build_model(data, planets, catalog=PLANETS, k_planets=(), prot_prior=TESS_PROT, k_max=10.0):
    names = list(planets)
    n = len(names)
    ntransit = sum(p in TRANSITING for p in names)
    assert all(p in TRANSITING for p in names[:ntransit]), "transiting planets must come first"
    kep = [i for i, p in enumerate(names) if p not in k_planets]
    circ = [i for i, p in enumerate(names) if p in k_planets]
    cat = [catalog[p] for p in names]
    col = lambda key: np.array([c[key] for c in cat], dtype=float)

    x, y, yerr = data.x, data.y, data.yerr
    x_rv, y_rv, yerr_rv, id_rv = data.x_rv, data.y_rv, data.yerr_rv, data.id_rv
    n_inst = len(data.inst_names)

    log_periods_lit = np.log10(col("P"))
    t0s_mu = col("t0") - X_REF
    mass_mu = np.log10(col("mass"))
    depths_ppt_lit = np.array([TRANSITING[p]["depth_ppt"] for p in names[:ntransit]])
    impact_p_lit = np.array([TRANSITING[p]["b"] for p in names[:ntransit]])

    with pm.Model() as model:
        # Stellar
        mean_flux = pm.Normal("mean_flux", mu=0.0, sigma=5)
        log_jitter_lc = pm.Normal("log_jitter_lc", mu=np.log(np.nanstd(y)), sigma=2.0)

        q1 = pm.Normal("q1", mu=0.4267, sigma=0.0924)
        q2 = pm.Normal("q2", mu=0.3705, sigma=0.0300)
        u1 = pm.Deterministic("u1", 2 * pt.sqrt(q1) * q2)
        u2 = pm.Deterministic("u2", pt.sqrt(q1) * (1 - 2 * q2))
        star = xo.LimbDarkLightCurve(pt.stack([u1, u2]))

        r_star = pm.Normal("r_star", mu=0.792, sigma=0.008)
        m_star = pm.Normal("m_star", mu=0.763, sigma=0.0212)  # Li et al. 2025

        # Planets
        t0 = pm.Normal("t0", mu=t0s_mu, sigma=col("t0_sigma"), shape=n, initval=t0s_mu)
        log_period = pm.Normal("log_period", mu=log_periods_lit, sigma=col("logP_sigma"), shape=n,
                               initval=log_periods_lit)
        period = pm.Deterministic("period", 10 ** log_period)

        nk = len(kep)
        log_m_pl = pm.Normal("log_m_pl", mu=mass_mu[kep], sigma=col("logM_sigma")[kep], shape=nk)
        secosw = pm.Uniform("secosw", lower=-1.0, upper=1.0, initval=np.sqrt(col("e_init")[kep]), shape=nk)
        sesinw = pm.Uniform("sesinw", lower=-1.0, upper=1.0, initval=[0.01] * nk, shape=nk)
        e_raw = pt.square(secosw) + pt.square(sesinw)

        if not circ:  # paper parameterization, kept literally identical
            m_pl = pm.Deterministic("m_pl", 10 ** log_m_pl)
            ecc = pm.Deterministic("ecc", pt.clip(e_raw, 0.0, 0.9999))
            omega = pm.Deterministic("omega", pt.arctan2(sesinw, secosw))
        else:
            K_circ = pm.Uniform("K_circ", lower=0.0, upper=k_max, shape=len(circ))
            m_circ = (K_circ * 100.0 * (period[circ] * 86400.0 / (2 * np.pi * G)) ** (1 / 3)
                      * (m_star * M_SUN) ** (2 / 3) / M_EARTH)
            zeros = pt.zeros(n)
            m_pl = pm.Deterministic("m_pl", pt.set_subtensor(
                pt.set_subtensor(zeros[kep], 10 ** log_m_pl)[circ], m_circ))
            ecc = pm.Deterministic("ecc", pt.set_subtensor(zeros[kep], pt.clip(e_raw, 0.0, 0.9999)))
            omega = pm.Deterministic("omega", pt.set_subtensor(zeros[kep], pt.arctan2(sesinw, secosw)))
        pm.Potential("e_bound", pm.math.log(pt.sigmoid((0.9999 - e_raw) * 500)))

        m_g, M_g = m_pl * M_EARTH, m_star * M_SUN
        pm.Deterministic("K", (2 * np.pi * G / (period * 86400.0)) ** (1 / 3) * m_g
                         / (M_g + m_g) ** (2 / 3) / pt.sqrt(1 - ecc ** 2) / 100.0)

        # Geometry
        b_transiting = pm.Uniform("b_transiting", lower=0.0, upper=1, shape=ntransit, initval=impact_p_lit)
        b_all = pm.Deterministic("b_all", pt.concatenate([b_transiting, pt.zeros(n - ntransit)]))

        log_depth = pm.Normal("log_depth", mu=np.log10(depths_ppt_lit), sigma=0.5, shape=ntransit)
        ror = pm.Deterministic("ror", star.get_ror_from_approx_transit_depth(1e-3 * 10 ** log_depth, b_transiting))
        r_pl = pm.Deterministic("r_pl", ror * r_star)

        # Light curve
        orbit_transit = xo.orbits.KeplerianOrbit(
            r_star=r_star, m_star=m_star,
            period=period[:ntransit], t0=t0[:ntransit], b=b_transiting,
            m_planet=xo.units.with_unit(m_pl[:ntransit], u.Mearth),
            ecc=ecc[:ntransit], omega=omega[:ntransit])
        light_curves = star.get_light_curve(orbit=orbit_transit, r=r_pl, t=x, texp=TEXP) * 1e3
        lc_model = pt.sum(light_curves, axis=-1) + mean_flux

        # Shared stellar rotation period
        kind, a, b = prot_prior
        if kind == "normal":
            P_rot = pm.Normal("P_rot", mu=a, sigma=b)
        elif kind == "uniform":
            P_rot = pm.Uniform("P_rot", lower=a, upper=b)
        else:
            raise ValueError(f"unknown P_rot prior {prot_prior}")

        # LC GP
        log_sigma_lc = pm.Normal("log_sigma_lc", mu=np.log(np.nanstd(y)), sigma=2.0)
        log_Q0_lc = pm.Normal("log_Q0_lc", mu=1.0, sigma=1.0)
        log_dQ_lc = pm.Normal("log_dQ_lc", mu=0.0, sigma=1.0)
        f_rot_lc = pm.Uniform("f_rot_lc", lower=0.05, upper=1.0)
        kernel_lc = terms.RotationTerm(sigma=pt.exp(log_sigma_lc), period=P_rot,
                                       Q0=pt.exp(log_Q0_lc), dQ=pt.exp(log_dQ_lc), f=f_rot_lc)
        gp_lc = GaussianProcess(kernel_lc, t=x, yerr=pt.sqrt(yerr ** 2 + pt.exp(2 * log_jitter_lc)), mean=lc_model)
        gp_lc.marginal("obs_lc", observed=y)

        # RV GP (own amplitude and quality factors, shared period)
        log_sigma_rv_gp = pm.Normal("log_sigma_rv_gp", mu=np.log(2.0), sigma=1.0)
        log_Q0_rv = pm.Normal("log_Q0_rv", mu=1.0, sigma=1.0)
        log_dQ_rv = pm.Normal("log_dQ_rv", mu=0.0, sigma=1.0)
        f_rot_rv = pm.Uniform("f_rot_rv", lower=0.05, upper=1.0)
        kernel_rv = terms.RotationTerm(sigma=pt.exp(log_sigma_rv_gp), period=P_rot,
                                       Q0=pt.exp(log_Q0_rv), dQ=pt.exp(log_dQ_rv), f=f_rot_rv)

        # RV orbits, offsets, jitters, activity
        orbit_rv = xo.orbits.KeplerianOrbit(
            r_star=r_star, m_star=m_star, period=period, t0=t0, b=b_all,
            m_planet=xo.units.with_unit(m_pl, u.Mearth), ecc=ecc, omega=omega)
        rv_model_sum = pt.sum(orbit_rv.get_radial_velocity(x_rv), axis=-1)
        rv_offsets = pm.Normal("rv_offsets", mu=0.0, sigma=25000.0, shape=n_inst, initval=data.rv_medians)
        rv_jitters = pm.HalfNormal("rv_jitters", sigma=10.0, shape=n_inst)

        C_activity = pm.Normal("C_activity", mu=0.0, sigma=5.0, initval=0.0)
        total_rv_mean = rv_model_sum + rv_offsets[id_rv] + C_activity * data.shk_rv_std

        gp_rv = GaussianProcess(kernel_rv, t=x_rv, yerr=pt.sqrt(yerr_rv ** 2 + rv_jitters[id_rv] ** 2 + 1e-4),
                                mean=total_rv_mean)
        gp_rv.marginal("obs_rv", observed=y_rv)

    return model

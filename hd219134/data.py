"""TESS photometry and RV loading (same cleaning as the paper-era rv_lc_gp_*.py scripts)."""
import pathlib
from dataclasses import dataclass

import numpy as np
from astropy.stats import sigma_clip

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
PHOTOMETRY_DIR = DATA_DIR / "photometry"
RV_DIR = DATA_DIR / "rv"
X_REF = 2450000.0

TESS_FILES = [
    "n20191008.TESS.TESS.TIC283722336.S17.0120.SPOC.dat",
    "n20200416.TESS.TESS.TIC283722336.S24.0120.SPOC.dat",
    "n20220930.TESS.TESS.TIC283722336.S57.0120.SPOC.dat",
    "n20240325.TESS.TESS.TIC283722336.S77.0120.SPOC.dat",
    "n20241001.TESS.TESS.TIC283722336.S84.0120.SPOC.dat",
]

# (name, file, instrument id). HARPS column 4 is log R'HK.
RV_FILES = [
    ("HARPS", "HD219134.HARPS.SHK.rv", 0),
    ("HJST", "HD219134.HJST_Phase3.rv", 1),
    ("HIRES-post", "HD219134.HARADA_HIRES-post.rv", 2),
    ("Levy", "HD219134.APF.Rosenthal2021.rv", 3),
    ("Hamilton", "HD219134.Hamilton.Howard2016a1994.rv", 4),
]


@dataclass
class Data:
    x: np.ndarray        # TESS time, BJD - X_REF
    y: np.ndarray        # flux, ppt
    yerr: np.ndarray
    x_rv: np.ndarray     # RV time, BJD - X_REF
    y_rv: np.ndarray     # m/s
    yerr_rv: np.ndarray
    id_rv: np.ndarray    # instrument index per RV
    inst_names: list
    rv_medians: list
    shk_rv_std: np.ndarray  # standardized HARPS activity index, 0 for other instruments

    @property
    def n_obs(self):
        return len(self.x) + len(self.x_rv)


def check_files():
    """Raise with a clear message if any data file is missing (all are required)."""
    missing = [PHOTOMETRY_DIR / f for f in TESS_FILES if not (PHOTOMETRY_DIR / f).exists()]
    missing += [RV_DIR / f for _, f, _ in RV_FILES if not (RV_DIR / f).exists()]
    if missing:
        raise FileNotFoundError("Missing data files:\n  " + "\n  ".join(map(str, missing)))


def load_data(verbose=True):
    check_files()

    # --- Light curve: drop NaNs, 4-sigma clip, drop flux > 1.002 ---
    all_time, all_flux, all_err = [], [], []
    for fname in TESS_FILES:
        d = np.loadtxt(PHOTOMETRY_DIR / fname, comments="#")
        t, f, e = d[:, 0], d[:, 1], d[:, 2]
        m = np.isfinite(f) & np.isfinite(t)
        t, f, e = t[m], f[m], e[m]
        keep = (~sigma_clip(f, sigma=4, maxiters=3).mask) & ~(f > 1.002)
        if verbose:
            print(f"  {fname}: {len(d)} raw, {keep.sum()} kept")
        all_time.append(t[keep]); all_flux.append(f[keep]); all_err.append(e[keep])

    x = np.concatenate(all_time) - X_REF
    y = (np.concatenate(all_flux) - 1) * 1e3
    yerr = np.concatenate(all_err) * 1e3 + 1e-6
    idx = np.argsort(x)
    x, y, yerr = x[idx], y[idx], yerr[idx]

    # --- RVs ---
    xs, ys, es, ids, shks, rv_medians, inst_names = [], [], [], [], [], [], []
    for name, fname, inst_id in RV_FILES:
        d = np.loadtxt(RV_DIR / fname, comments="#")
        t, v, e = d[:, 0] - X_REF, d[:, 1], d[:, 2]
        rv_medians.append(np.median(v))
        if name == "HARPS" and d.shape[1] >= 4:
            s = d[:, 3]
            m = np.isfinite(t) & np.isfinite(v) & np.isfinite(e) & np.isfinite(s)
            shks.append(s[m])
        else:
            m = np.isfinite(t) & np.isfinite(v) & np.isfinite(e)
            shks.append(np.zeros(m.sum()))
        xs.append(t[m]); ys.append(v[m]); es.append(e[m])
        ids.append(np.full(m.sum(), inst_id, dtype=int))
        inst_names.append(name)
        if verbose:
            print(f"  {name}: {m.sum()} RVs")

    x_rv, y_rv, yerr_rv, id_rv, shk_rv = (np.concatenate(a) for a in (xs, ys, es, ids, shks))
    o = np.argsort(x_rv)
    x_rv, y_rv, yerr_rv, id_rv, shk_rv = x_rv[o], y_rv[o], yerr_rv[o], id_rv[o], shk_rv[o]

    shk_rv_std = np.zeros_like(shk_rv)
    harps = id_rv == 0
    sd = np.std(shk_rv[harps]) or 1.0
    shk_rv_std[harps] = (shk_rv[harps] - np.mean(shk_rv[harps])) / sd

    return Data(x, y, yerr, x_rv, y_rv, yerr_rv, id_rv, inst_names, rv_medians, shk_rv_std)

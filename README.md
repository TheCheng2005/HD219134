# HD 219134 run suite

Joint TESS + radial-velocity modeling of HD 219134 with `exoplanet`, PyMC and a celerite2
rotation GP (Elliott et al., in prep.). This repository reruns every model the paper relies on,
plus the tests needed to settle the disputed planets f, g and i, unattended, on one machine.

One command does everything:

```
python run_suite.py
```

It checks the setup, then works through 22 runs, most informative first, a few at a time.
After every run it updates `results/summary/SUMMARY.md`. Finished runs are skipped, so
after a crash, reboot or Ctrl-C, just run the same command again.

---

## 1. Setting up (Windows)

1. **Install Miniforge** (conda from conda-forge): download the Windows installer from
   <https://github.com/conda-forge/miniforge/releases/latest> (`Miniforge3-Windows-x86_64.exe`).
   Install "just for me"; no admin rights needed. Then open **Miniforge Prompt** from the Start menu.
   Use it for every command below.
2. **Install git** if `git --version` fails: `conda install -c conda-forge git`.
3. **Download the repository** (it is private, so git will open a browser window to log in to GitHub):
   ```
   cd %USERPROFILE%
   git clone https://github.com/TheCheng2005/HD219134.git
   cd HD219134
   ```
4. **Create the environment** (10–20 minutes):
   ```
   conda env create -f environment.yml
   conda activate hd219134
   ```
5. **Check the setup** (2–5 minutes):
   ```
   python tests\check_setup.py
   ```
   This confirms the package versions, that pytensor found a C++ compiler, and that every
   model gives the *same* log-probability as on the reference Mac (to 10⁻³). It ends with
   "All checks passed" and a rough time per run for this machine.
   If the compiler check fails, run `conda install -c conda-forge "pytensor=2.31.7" gxx` and try again.
6. **Smoke test the whole pipeline** (about 15–30 minutes; tiny chains, results are meaningless):
   ```
   python run_suite.py --quick --only paper_M4,kprior_M4+f
   ```
   Output goes to `results_quick\` and can be deleted afterwards.

## 2. Running

```
conda activate hd219134
cd %USERPROFILE%\HD219134
python run_suite.py
```

| Command | What it does |
|---|---|
| `python run_suite.py` | run everything not yet finished |
| `python run_suite.py --list` | status of every run (done / running / pending / FAILED) |
| `python run_suite.py --group paper,nested` | only some groups |
| `python run_suite.py --only paper_M4,kprior_M4+f` | only some runs |
| `python run_suite.py --jobs 2` | runs at once (default: CPU cores ÷ 4) |
| `python run_suite.py --retry-failed` | also redo runs that failed |
| `python run_suite.py --summary` | just rebuild `results/summary/` |

**Keeping it running for days**
- Leave the Miniforge Prompt window open. Locking the screen is fine; **logging off or
  restarting stops the runs.** While it runs, the script asks Windows not to sleep.
- In Settings → System → Power, set "When plugged in, put my device to sleep after" to **Never**.
- In Settings → Windows Update, **pause updates** for the duration, so the machine doesn't restart.
- If it does stop, run `python run_suite.py` again. Finished runs are kept; the interrupted
  run starts over.

**Checking progress:** open `results\summary\SUMMARY.md` (any text editor, or VS Code's
preview), or each run's live log `results\<run>\log.txt`.

**Bringing results back:** zip the `results` folder (about 5–10 MB per run) and copy it over.
`results/` is git-ignored on purpose.

## 3. How long it takes

Each run samples 4 chains × (1500 tuning + 1500 kept) iterations on 4 cores. On the reference
Mac the converged 4-planet model took 12 h; the published 5- and 6-planet models took 35–37 h,
because the sampler hit its step limit on almost every iteration. With 16 cores (4 runs at once),
expect roughly 5–8 days for all 22 runs; with 8 cores, about twice that. The order puts the runs
that decide the paper's conclusions first, so useful results arrive within the first 1–2 days.

## 4. The runs

Planets: b, c (transiting), d, h (adopted); f, g, i (disputed); x (synthetic). P_rot prior:
N(22.56, 0.29) d from the TESS SAP periodogram unless stated otherwise.

| Run | Group | What it tests |
|---|---|---|
| `paper_M4` | paper | adopted model b c d h (4-Planet-M) |
| `paper_G4` `paper_P5` `paper_V6` `paper_H6` | paper | the other published configurations, same priors as the paper |
| `nested_M4+f`, `nested_M4+g` | nested | adopted model plus exactly one disputed planet: the direct test of f and of g |
| `kprior_M4+f`, `kprior_M4+g`, `kprior_P5`, `kprior_G4`, `kprior_V6`, `kprior_H6` | kprior | disputed planets on circular orbits with K ~ U(0, 10) m/s, so the data can switch them off (K = 0). Their log-normal mass priors in the paper could not reach zero. |
| `prot_broad_M4`, `prot_broad_M4+f` | prot | P_rot ~ U(15, 50) d; two chains start at 22.56 d, two at 42 d |
| `prot_41_M4`, `prot_41_M4+f` | prot | P_rot ~ N(41.2, 2.8) d (the 41-d test of Table 9) |
| `inject_22.56` | prot | adds a K = 2.3 m/s circular planet **at** P_rot. If it is not recovered, the GP absorbs real planets at the rotation period, and "f is activity" cannot be concluded from these fits. |
| `inject_27.0`, `inject_33.0` | prot | the same injection away from P_rot (should be recovered) |
| `null_27.0`, `null_33.0` | prot | same models as the injections, but with no injected signal (K should be consistent with 0) |

Every chain starts from a different draw of the converged paper 4-planet posterior
(`data/init/m4_paper_posterior_draws.npz`); planets not in that model start at their literature
values, and K-prior planets at K = 0.5, 1.5, 2.5 or 3.5 m/s depending on the chain, so chains
that end up in different modes show it.

## 5. What the numbers mean

- **ln L**: log-likelihood of the data only (TESS + RV; no priors).
- **ln L̂**: the best fit *inside the posterior's support*: the best posterior draw, or the
  posterior mode reached by optimizing from the best draw of each chain, whichever is higher.
  The likelihood-only maximum is **not** used. The light curve does not constrain P_rot in this
  model, so without priors the optimizer sends P_rot to thousands of days, and the GP then
  absorbs planets.
- **BIC** = k ln n − 2 ln L̂, with k the number of free parameters and n = 81,886 data points.
  The table reports ΔBIC relative to `paper_M4`; negative favours the other model.
- **median ln L**: shown for reference only. It sits about k/2 below ln L̂ even for perfect
  runs, and for unconverged runs it mostly reflects where the chains got stuck.
- **DIC** = D̄ + p_D with D = −2 ln L.
- **Converged** = R̂ ≤ 1.01, bulk ESS ≥ 400 and no divergences. Results from unconverged runs
  can still give ln L̂ (an optimum only needs one good point), but not posterior medians or
  uncertainties. The summary flags them.

## 6. Differences from the code used for the September draft

1. **Likelihood evaluation fixed.** The old `get_aic_bic_metric` read `rv.transform`, which is
   always `None` in PyMC 5, so bounded parameters (e, ω, b, f_rot, jitters) were left at their
   starting values, and a bare `except` hid the error. Every ln L and BIC in the draft's
   model-comparison tables is affected. `metrics.analyze` checks each run by rebuilding draws
   and matching the sampler's own log-probability.
2. **BIC uses ln L̂, not the median ln L.** DIC uses the likelihood only (before, it included the
   priors, with bounded parameters again at their starting values).
3. **One model builder** (`hd219134/model.py`) replaces five copied scripts. With the paper
   options it reproduces the old scripts' log-probability exactly. The one exception: the old
   6-planet script had `f_rot_lc ~ U(0, 1)`, the others `U(0.05, 1)`; the suite uses 0.05 throughout.
4. **No staged MAP optimization.** Chains start from the converged 4-planet posterior instead.
   The old MAP ended about 450 in ln L below the posterior median.
5. **Fixed random seeds** per run, and every output records the package versions used.

## 7. Layout

```
run_suite.py              runs everything (see section 2)
hd219134/data.py          TESS + RV loading and cleaning
hd219134/planets.py       literature priors for every planet; the published configurations
hd219134/model.py         the joint model; K-prior and P_rot-prior options; RV injection
hd219134/runs.py          the 22 runs and sampling settings
hd219134/run_one.py       samples and analyzes one run
hd219134/metrics.py       ln L, ln L-hat, BIC, DIC, convergence, planet tables
hd219134/summarize.py     results/summary/ (SUMMARY.md, model_comparison.csv, LaTeX table)
tests/check_setup.py      setup and reproducibility check
data/                     TESS SPOC light curves (5 sectors), RVs from 5 instruments, starting draws
```

RV data sources: HARPS-N (Gillon et al. 2017, private communication; includes Motalebi et al. 2015),
HIRES (Butler et al. 2017), Tull/HJST Phase III (Johnson et al. 2016), APF/Levy (Rosenthal et al. 2021),
Hamilton (Fischer et al. 2014, binned by Howard & Fulton 2016). **The HARPS-N velocities are not
public; keep this repository private.**

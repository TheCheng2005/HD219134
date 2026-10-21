# HD 219134 run suite

Joint TESS + radial-velocity modeling of HD 219134 with `exoplanet`, PyMC and a celerite2
rotation GP (Elliott et al., in prep.). This repository reruns the five planet configurations the
paper reports, plus four internal checks on the disputed planets f, g and i, unattended, on one
machine.

One command does everything:

```
python run_suite.py
```

It checks the setup, then works through 9 runs, a few at a time.
After every run it updates `results/summary/SUMMARY.md`. Finished runs are skipped and
unfinished or failed ones are redone, so after a crash, reboot or Ctrl-C, just run the same
command again.

---

## 1. Why this suite exists: problems in the September analysis

The September draft compared five planet configurations and concluded that the adopted model
(b, c, d, h) is preferred and that f, g and i are stellar activity. Checking the saved traces
turned up the problems below. Most of them are about **how the models were compared**; the
adopted 4-planet fit itself converged and holds up.

| # | Problem | Consequence | What the suite does |
|---|---|---|---|
| 1 | Likelihoods evaluated at the wrong parameter values | every ln L and BIC in the comparison tables is wrong | correct evaluation, checked against the sampler |
| 2 | BIC built from the *median* ln L | bigger models penalized twice; unstable for unconverged runs | BIC from the best fit, ln L̂ |
| 3 | A best fit without priors is meaningless here | the GP drifts to absurd periods and absorbs planets | ln L̂ keeps the priors |
| 4 | The 5- and 6-planet runs never converged | their medians and uncertainties describe no real solution | better starting points, better parameterization, automatic diagnostics |
| 5 | Log-mass priors centered on published masses | "no planet" is never allowed; the RVs count twice; causes problem 4 | K ~ U(0, 20) m/s for every planet; f, g, i circular |
| 6 | Planet f sits at the rotation period | "the GP explains f" is expected whether or not f is real | injection of a planet at P_rot, with a control |
| 7 | No direct test of single planets | a ΔBIC can't be pinned on one planet | internal runs b c d h + f, b c d h + g |
| 8 | Five copied scripts drifted apart | runs differ in ways the paper doesn't say | one model builder, verified |
| 9 | The MAP starting point wasn't a best fit | poor starts for the sampler | chains start from the converged 4-planet posterior |

### Problem 1: the likelihoods were evaluated at the wrong parameter values

To compare models, the old `get_aic_bic_metric` took each posterior draw and evaluated the
likelihood there. PyMC samples bounded parameters (√e cos ω, √e sin ω, impact parameter,
f_rot, jitters) on an internal, unbounded scale, so each draw must be converted to that scale
first. The code looked for the conversion in `rv.transform`, which is always `None` in PyMC 5.
The conversion therefore never happened, those parameters stayed at their **starting values** for
every draw, and a bare `except: pass` hid the error.

*Consequence:* every ln L and BIC in the draft's model-comparison tables is off by hundreds to
thousands, by a different amount for each model. The 4-planet median ln L is 49,755, not 48,692.
This is why larger models seemed to fit *worse* than the smaller models they contain, which
cannot happen when the extra planet can be switched off.

*Suite:* converts correctly, and before reporting anything checks that the rebuilt draws reproduce
the log-probability the sampler itself recorded (they agree to 10⁻¹¹; the run stops if not).

### Problem 2: BIC was built from the median ln L

BIC = k ln n − 2 ln L̂ is defined with the **best-fit** likelihood L̂. The draft used the median
over posterior draws instead. Even for a perfectly sampled posterior, the median sits well below
the best fit, by up to about k/2 (about 20 below for the 4-planet model, k = 50), and the gap grows
with the number of parameters. A model with more parameters is therefore penalized twice: once by
k ln n, and again by a lower median.

For unconverged runs it is worse: the median mostly reflects where the chains got stuck. Within a
single run, the four chains' medians differ by up to **85** (5-Planet) and **250** (6-Planet-H).
By contrast, all four chains of the converged 4-planet run reach the same best fit to 0.01.

*Suite:* BIC uses ln L̂ (problem 3). The median is still reported, for reference only.

### Problem 3: a best fit that ignores the priors is meaningless here

The natural ln L̂ would maximize the likelihood alone. Doing that sends the fit somewhere
absurd. The TESS light curve barely constrains the rotation period inside this model: its
ln L is **53,829.4 whether P_rot is 22 d or 21,000 d**. So nothing but the prior holds P_rot
near 22.6 d. Without the prior, the optimizer moves P_rot to thousands of days; the GP then
becomes a slow trend that absorbs planet h (Gillon configuration), or planet g's period runs off
to 5 × 10⁹ days so that g effectively disappears.

So a "bad" maximum really can look better. But the cause is dropping the priors, not using the maximum.

*Suite:* ln L̂ is the best fit **with the priors in place**: the best posterior draw, or the
posterior mode reached by optimizing from each chain's best draw, whichever is higher. For the
4-planet model, all four chains reach the same mode (ln L̂ = 49,775.38).

*Also note:* P_rot is set by its prior, not measured by the light curve, in this model. That
matters for problem 6.

### Problem 4: the 5- and 6-planet runs never converged, and longer runs would not help

| Published run | NUTS steps per draw | draws hitting the step limit | divergences | R̂ | wall time |
|---|---|---|---|---|---|
| 4-Planet-M | 63 | 0% | 0 | ≤ 1.01 | 12 h |
| 5-Planet | 255 | 48% | 52 | > 2 | 35 h |
| 6-Planet-V, 4-Planet-G, 6-Planet-H | 1023 (the limit) | 91–97% | 143–329 | 2.3–3.4 | 35–37 h |

In these runs the four chains settled in **different places**. In 4-Planet-G, f's period is
22.67, 26.04, 20.81 and 22.47 d in the four chains. The draft states R̂ ≤ 1.01 and N_eff ≥ 1000
for all parameters, which holds only for the 4-planet model.

Running longer does not fix this. Divergences and hitting the step limit mean the posterior has a
shape the sampler cannot move through: narrow funnels (problem 5) and separate modes. More draws
just add more samples from the same stuck chains. Effective sample size stays around 5, because
it counts how often chains move *between* regions, and they never do.

*Consequence:* the medians and uncertainties quoted for these models (f's periods, g's mass,
the median ln L) are averages over chains that disagree, so they describe no real solution.

*Suite:* each chain starts from a different draw of the converged 4-planet posterior; the new
priors and circular orbits for f, g and i remove the worst geometry (problem 5); and every run gets an automatic convergence
report (R̂, ESS, divergences, per-chain medians). An unconverged run can still give ln L̂, since an
optimum needs only one good point, but its posterior medians are flagged and should not be quoted.

### Problem 5: log-mass priors centered on published masses

Every planet had a prior on log₁₀ mass, centered on its published value. For f,
log₁₀ m ~ N(log₁₀ 7.4, 0.5). Zero mass is log m = −∞, so it is never reached. In practice this
prior puts 95% of its probability between about 0.8 and 70 M⊕ (K ≈ 0.2–19 m/s), and only about
0.3% below K = 0.1 m/s. It effectively states that f exists with roughly its published mass. This
has four consequences:

1. **No "no planet" answer.** The data can pull the mass down a few σ, but the posterior can
   never say "consistent with zero", so the test is not neutral.
2. **The comparison is not truly nested.** "b c d h + f" cannot exactly reproduce "b c d h".
3. **It causes problem 4.** When the data don't support the planet, the sampler drifts toward
   small log m. There the planet's period, phase and eccentricity stop affecting the fit, and the
   posterior narrows into a funnel that NUTS cannot step through: divergences, maximum tree depth,
   stuck chains.
4. **The RVs count twice.** The published masses (Motalebi et al. 2015, Gillon et al. 2017, Vogt
   et al. 2015) were measured from largely the same HARPS-N and HIRES velocities fitted here, so
   priors centered on them pull the results toward the published values.

*Suite:* every planet gets a **linear** prior on its RV semi-amplitude, K ~ U(0, 20) m/s, which
includes zero; masses are derived from K. If the data don't want a planet, its posterior piles up
near K = 0, and the 2.5th percentile of K answers the question directly. K rather than mass,
because K is what the RVs measure (at fixed period K ∝ m sin i, so the two are nearly equivalent).
The upper bound only needs to sit well above the largest signal (h, K ≈ 5.6 m/s). f, g and i are
also fitted on circular orbits, removing the eccentricity directions that a ~2 m/s signal cannot
constrain; b, c, d and h stay eccentric. b, c, d and h are detected at K/σ_K ≳ 20, so their
posteriors should move well under 1σ from the paper's; `main_M4` against the converged paper run
checks this. Two caveats: e = 0 for f, g, i is a modeling choice, and a uniform prior puts more
weight at large K than a log prior (this affects a Bayesian evidence, not ΔBIC).

### Problem 6: planet f sits at the rotation period, so the test is circular

Planet f's period (22.7 d) lies inside the P_rot prior, 22.56 ± 0.29 d. A GP whose period is
P_rot can absorb any signal at that period, so "the GP explains f, so f is activity" would come
out **whether or not f is real**. And by problem 3, P_rot is fixed by its prior rather than
measured. When the GP was moved to ~41 d (the draft's Table 9 runs), f came back at 22.80 d with
K ≈ 2.1–2.4 m/s.

*Suite:* two internal runs test this directly:
- `test_inject_22.56`: a synthetic K = 2.3 m/s planet is added to the RVs *at* 22.56 d and fitted
  with the GP at P_rot. If the GP swallows it, these fits cannot decide whether f is a planet.
- `test_inject_27.0`: the same injection at 27 d, away from P_rot, as a control that should be
  recovered.

Tests with a broad or a 41-d P_rot prior are planned for later; `hd219134/model.py` already
accepts other P_rot priors.

### Problem 7: no direct test of a single planet

The published configurations change several planets at once. For example, 4-Planet-G adds f
**and** drops h, so its poor BIC mostly measures losing h, not anything about f.

*Suite:* `test_M4+f` and `test_M4+g` add exactly one disputed planet to the adopted model.
These are the comparisons that actually test f and g. They guide the paper's wording but are not
reported.

### Problem 8: five copied scripts drifted apart

The five configurations lived in five copied scripts, and they differed in ways the paper does not
mention:
- The 6-planet script had `f_rot_lc ~ U(0, 1)`; the others had `U(0.05, 1)`.
- The P_rot prior was changed for test runs and changed back.
- Some table entries could not be traced to a specific run.

*Suite:* one model builder (`hd219134/model.py`). With the paper's priors (`mass_prior="lognormal"`)
it reproduces the four consistent scripts' log-probability **exactly**, and the 6-planet one exactly
once its bound is set to 0.05, which the suite uses throughout. The suite's runs then use the new
mass priors (problem 5). Every run records its full specification, random seed
and package versions, and the tables in `results/summary/` are generated from the runs, never typed.

### Problem 9: the MAP starting point was not a best fit

The staged `pmx.optimize` "MAP" used to start the chains ended about 450 in ln L **below** the
posterior median, so it was a poor start and not a best fit.

*Suite:* no MAP stage. Chains start from draws of the converged 4-planet posterior; f, g and i
start at their published period, phase and mass (as K). Under a uniform prior the starting point
does not favour a planet; it only saves tuning time.

---

## 2. Setting up (Windows)

1. **Install Miniforge** (conda from conda-forge): download the Windows installer from
   <https://github.com/conda-forge/miniforge/releases/latest> (`Miniforge3-Windows-x86_64.exe`).
   Install "just for me"; no admin rights needed. Then open **Miniforge Prompt** from the Start menu.
   Use it for every command below.
2. **Install git** if `git --version` fails: `conda install -c conda-forge git`.
3. **Download the repository**:
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
   model gives the *same* log-probability as on the reference Mac (to within 0.01 out of ~49,500,
   allowing for compiler rounding). It ends with
   "All checks passed" and a rough time per run for this machine.
   If the compiler check fails, run `conda install -c conda-forge "pytensor=2.31.7" gxx` and try again.
6. **Smoke test the whole pipeline** (about 15–30 minutes; tiny chains, results are meaningless):
   ```
   python run_suite.py --quick --only main_M4,test_M4+f
   ```
   Output goes to `results_quick\` and can be deleted afterwards.

## 3. Running

```
conda activate hd219134
cd %USERPROFILE%\HD219134
python run_suite.py
```

| Command | What it does |
|---|---|
| `python run_suite.py` | run everything not yet finished, including runs that failed before |
| `python run_suite.py --list` | status of every run (done / running / pending / FAILED) |
| `python run_suite.py --group main` | only the five reported configurations (`test` = internal checks) |
| `python run_suite.py --only main_M4,test_M4+f` | only some runs |
| `python run_suite.py --jobs 2` | runs at once (default: logical processors ÷ 4; see below) |
| `python run_suite.py --skip-failed` | leave runs that failed before alone |
| `python run_suite.py --summary` | just rebuild `results/summary/` |

**Keeping it running for days**
- Leave the Miniforge Prompt window open. Locking the screen is fine; **logging off or
  restarting stops the runs.** While it runs, the script asks Windows not to sleep.
- In Settings → System → Power, set "When plugged in, put my device to sleep after" to **Never**.
- In Settings → Windows Update, **pause updates** for the duration, so the machine doesn't restart.
- If it does stop, run `python run_suite.py` again. Finished runs are kept; interrupted and
  failed runs start over. Run only **one** copy at a time: two copies write to the same folders.
- **Don't click inside the console window.** Windows' QuickEdit mode freezes a console's output
  after a click until you press a key (or Esc). Sampling continues, but the suite cannot record a
  finished run until then. To turn it off for good: right-click the title bar → Properties →
  untick "QuickEdit Mode".

**How many runs at once (`--jobs`).** Each run uses 4 cores, one per chain. The default divides
the number of *logical* processors by 4. On a CPU with hyperthreading (most Intel desktops) that
is twice the number of *physical* cores ÷ 4, which still works but makes each run slower. If
Task Manager (Performance → CPU) shows, say, 8 cores and 16 logical processors, use `--jobs 2`.
Also watch memory in the first hour; if it is close to full, restart with a smaller `--jobs`.

**Checking progress:** open `results\summary\SUMMARY.md` (any text editor, or VS Code's
preview), or each run's live log `results\<run>\log.txt`. It gets a line per chain every
100 iterations (out of 3000) with the divergence count, the speed over those 100 iterations
(seconds and NUTS steps per iteration), the time left at that speed, and the elapsed time:

```
chain 0: iteration 1200/3000 (tuning), 0 divergences so far | last 100: 6.3 s/iteration, 64 steps/iteration, ~3.2 h left at this speed | 21.21 h elapsed
```

The first few hundred tuning iterations are by far the slowest: the sampler has not yet learned
the posterior's shape, so it takes tiny steps, up to 1023 per iteration. Once it adapts, steps per
iteration drop (to ~60 for the 4-planet model) and so does the time per iteration. A run finishes
when its slowest chain does. Don't keep the summary files open in Excel while the suite runs;
Excel locks them.

**Bringing results back:** zip the `results` folder (about 5–10 MB per run) and copy it over.
`results/` is git-ignored on purpose.

## 4. How long it takes

Each run samples 4 chains × (1500 tuning + 1500 kept) iterations on 4 cores. On the reference
Mac the converged 4-planet model took 12 h; the published 5- and 6-planet models took 35–37 h,
because the sampler hit its step limit on almost every iteration; the new priors should bring the
extended models closer to the 4-planet case. With 16 physical cores (4 runs at once) expect roughly
2–3 days for all 9 runs; with 8 (`--jobs 2`), about 4–5 days. The five reported configurations run
first.

## 5. The runs

Planets: b, c (transiting), d, h (adopted); f, g, i (disputed); x (synthetic, injected).

| Run | Group | What it is |
|---|---|---|
| `main_M4` | main | 4-Planet-M: b c d h, the adopted model |
| `main_H6` | main | 6-Planet-H: b c f d i h (Harada et al. 2025) |
| `main_G4` | main | 4-Planet-G: b c f d (Gillon et al. 2017) |
| `main_P5` | main | 5-Planet: b c d i h |
| `main_V6` | main | 6-Planet-V: b c f d g h (Vogt et al. 2015) |
| `test_M4+f` | test | b c d h + f: does f alone improve the fit? |
| `test_inject_22.56` | test | K = 2.3 m/s planet injected **at** P_rot. If it is not recovered, the GP absorbs real planets at the rotation period, and "f is activity" cannot be concluded from these fits. |
| `test_M4+g` | test | b c d h + g: does g alone improve the fit? |
| `test_inject_27.0` | test | the same injection at 27 d, away from P_rot; should be recovered |

The `main` runs are the paper's Table 6 configurations; `test` runs are internal and not reported.

**Model settings (all runs).** These differ from the September draft, and §5 should describe them:
- every planet: K ~ U(0, 20) m/s, masses derived from K (problem 5);
- f, g, i and x: circular orbits; b, c, d, h: free e and ω, as before;
- everything else as in the paper: P_rot ~ N(22.56, 0.29) d, Li et al. (2025) stellar mass,
  literature priors on periods and phases, the same GP, offsets, jitters and activity term.

**Starting points.** Every chain starts from a different draw of the converged paper 4-planet
posterior (`data/init/m4_paper_posterior_draws.npz`); f, g, i and x start at their published (or
injected-model) period, phase and mass, converted to K.

## 6. What the numbers mean

See problems 2 and 3 above for why these definitions were chosen.

- **ln L**: log-likelihood of the data only (TESS + RV; no priors). It also contains the tiny
  e < 1 boundary term, below 10⁻¹⁰⁰ for any e under 0.3.
- **ln L̂**: the best fit *inside the posterior's support*: the best posterior draw, or the
  posterior mode reached by optimizing from the best draw of each chain, whichever is higher.
  The likelihood-only maximum is **not** used. The light curve does not constrain P_rot in this
  model, so without priors the optimizer sends P_rot to thousands of days, and the GP then
  absorbs planets.
- **BIC** = k ln n − 2 ln L̂, with k the number of free parameters and n = 81,886 data points.
  The table reports ΔBIC relative to `main_M4`; negative favours the other model.
- **median ln L**: shown for reference only. It sits below ln L̂ by an amount that grows with k
  even for perfect runs, and for unconverged runs it mostly reflects where the chains got stuck.
- **DIC** = D̄ + p_D with D = −2 ln L.
- **Converged** = R̂ ≤ 1.01, bulk ESS ≥ 400 and no divergences, judged without the period,
  phase and K of the circular planets (f, g, i, x). When such a planet's K is near 0, those parameters are
  unconstrained, so their R̂ says nothing about the rest of the fit. The summary also shows R̂
  over all parameters. Unconverged runs can still give ln L̂ (an optimum only needs one good
  point), but not posterior medians or uncertainties. The summary flags them.
- Every run's `metrics.json` records its settings, random seed and package versions.

## 7. Layout

```
run_suite.py              runs everything (see section 3)
hd219134/data.py          TESS + RV loading and cleaning
hd219134/planets.py       literature priors for every planet; the published configurations
hd219134/model.py         the joint model (uniform-K or the paper's log-mass priors); RV injection
hd219134/runs.py          the 9 runs, starting points and sampling settings
hd219134/run_one.py       samples and analyzes one run
hd219134/metrics.py       ln L, ln L-hat, BIC, DIC, convergence, planet tables
hd219134/summarize.py     results/summary/ (SUMMARY.md, model_comparison.csv, LaTeX table)
tests/check_setup.py      setup and reproducibility check
data/                     TESS SPOC light curves (5 sectors), RVs from 5 instruments, starting draws
```

RV data sources: HARPS-N (Gillon et al. 2017, private communication; includes Motalebi et al. 2015),
HIRES (Butler et al. 2017), Tull/HJST Phase III (Johnson et al. 2016), APF/Levy (Rosenthal et al. 2021),
Hamilton (Fischer et al. 2014, binned by Howard & Fulton 2016). The HARPS-N velocities were provided
by M. Gillon by private communication; please do not reuse them without permission.

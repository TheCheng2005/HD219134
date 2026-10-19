"""Collect results/<run>/metrics.json into results/summary/ (SUMMARY.md, CSVs, LaTeX).

Safe to run at any time; runs that have not finished are listed as pending.
"""
import csv
import json

from .runs import RUNS

CONVERGED = dict(rhat=1.01, ess=400)
NAMES = {"paper_M4": "4-Planet-M", "paper_G4": "4-Planet-G", "paper_P5": "5-Planet",
         "paper_V6": "6-Planet-V", "paper_H6": "6-Planet-H"}


def status(base, run):
    d = base / run.id
    if (d / "done.json").exists():
        return "done"
    if (d / "failed.txt").exists():
        return "FAILED"
    if (d / "running.txt").exists():
        return "running"
    return "pending"


def converged(m):
    c = m["convergence"]
    return c["rhat_core"] <= CONVERGED["rhat"] and c["ess_core"] >= CONVERGED["ess"] and c["divergences"] == 0


def fmt(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def summarize(base):
    out = base / "summary"
    out.mkdir(parents=True, exist_ok=True)
    metrics = {r.id: json.loads((base / r.id / "metrics.json").read_text(encoding="utf-8"))
               for r in RUNS if status(base, r) == "done"}
    md = ["# HD 219134 run suite: summary", "",
          f"{len(metrics)} of {len(RUNS)} runs finished. Regenerated after every run; "
          "see README for how each number is defined.", ""]

    # 1. status
    md += ["## Status", "", table(
        ["run", "group", "status", "hours", "description"],
        [[r.id, r.group, status(base, r), fmt(metrics[r.id]["hours_total"], 1) if r.id in metrics else "", r.note]
         for r in RUNS]), ""]

    # 2. convergence
    rows = []
    for r in RUNS:
        if r.id in metrics:
            c = metrics[r.id]["convergence"]
            rows.append([r.id, fmt(c["rhat_core"], 3), f"{c['ess_core']:.0f}", fmt(c["rhat_max"], 3), c["divergences"],
                         fmt(c["frac_max_treedepth"]), f"{c['median_steps']:.0f}",
                         "yes" if converged(metrics[r.id]) else "**no**",
                         ", ".join(f"{p:.2f}" for p in c["per_chain_medians"]["P_rot"])])
    md += ["## Convergence",
           "",
           f"Converged = R-hat <= {CONVERGED['rhat']}, bulk ESS >= {CONVERGED['ess']} and no divergences, "
           "judged without the period, phase and K of K-prior planets (unconstrained when K is near 0; "
           "'R-hat all' includes them). Per-chain P_rot medians show whether chains agree.", "",
           table(["run", "R-hat", "ESS min", "R-hat all", "divergences", "frac max depth", "steps/draw",
                  "converged", "P_rot per chain"], rows), ""]

    # 3. model comparison (real data only)
    ref = metrics.get("paper_M4")
    comp = []
    for r in RUNS:
        m = metrics.get(r.id)
        if m is None or r.injection or "x" in r.planets:
            continue
        dll = m["ll_hat"] - ref["ll_hat"] if ref else None
        dbic = m["bic"] - ref["bic"] if ref else None
        comp.append(dict(run=r.id, model=NAMES.get(r.id, r.id), planets=" ".join(r.planets),
                         k_planets=" ".join(r.k_planets), prot_prior=str(r.prot_prior), k=m["k"],
                         ll_hat=m["ll_hat"], ll_median=m["ll_median"], d_ll_hat=dll, d_bic=dbic,
                         dic=m["dic"], d_dic=m["dic"] - ref["dic"] if ref else None,
                         converged=converged(m)))
    if comp:
        with open(out / "model_comparison.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(comp[0]))
            w.writeheader()
            w.writerows(comp)
    md += ["## Model comparison (relative to paper_M4)", "",
           "ln L-hat is the best fit inside the posterior's support. Negative dBIC favours the run over "
           "paper_M4. prot_* runs use a different P_rot prior: their rows say how well the data can be fit "
           "with the GP at that rotation period (see 'P_rot per chain' above), not which planets exist.", "",
           table(["run", "planets", "K-prior", "k", "ln L-hat", "d ln L-hat", "dBIC", "median ln L", "dDIC",
                  "converged"],
                 [[c["run"], c["planets"], c["k_planets"], c["k"], fmt(c["ll_hat"]), fmt(c["d_ll_hat"]),
                   fmt(c["d_bic"]), fmt(c["ll_median"]), fmt(c["d_dic"]), "yes" if c["converged"] else "**no**"]
                  for c in comp]), ""]

    # 4. disputed and synthetic planets
    rows = []
    for r in RUNS:
        m = metrics.get(r.id)
        if m is None:
            continue
        for p in r.planets:
            if p in ("f", "g", "i", "x"):
                q = m["planets"][p]
                rows.append([r.id, p, "K-prior" if p in r.k_planets else "paper",
                             f"{q['period'][1]:.3f}", f"{q['K'][1]:.2f} (+{q['K'][2] - q['K'][1]:.2f}/"
                             f"-{q['K'][1] - q['K'][0]:.2f})", fmt(q["K_2.5pct"]),
                             fmt(m["best_fit"]["K"][r.planets.index(p)])])
    md += ["## Disputed and injected planets", "",
           "K in m/s: posterior median (68% interval), the 2.5th percentile (a lower bound near 0 means the "
           "data allow no planet), and K at the best fit.", "",
           table(["run", "planet", "prior", "P median [d]", "K median", "K 2.5%", "K best fit"], rows), ""]

    # 5. injection-recovery
    rows = []
    for r in RUNS:
        m = metrics.get(r.id)
        if m is None or "x" not in r.planets:
            continue
        q = m["planets"]["x"]
        P_inj, K_inj = (r.injection[0], r.injection[1]) if r.injection else (float(r.id.split("_")[-1]), 0.0)
        rows.append([r.id, f"{P_inj:.2f}", f"{K_inj:.1f}", f"{q['period'][1]:.3f}",
                     f"{q['K'][1]:.2f} [{q['K'][0]:.2f}, {q['K'][2]:.2f}]", fmt(q["K_2.5pct"]),
                     fmt(m["best_fit"]["P_rot"])])
    md += ["## Injection-recovery", "",
           "A planet injected at P_rot that comes back with K well below the injected value means the GP "
           "absorbs real planets at the rotation period. Null runs should give K consistent with 0.", "",
           table(["run", "P injected", "K injected", "P recovered", "K recovered [68%]", "K 2.5%",
                  "P_rot at best fit"], rows), ""]

    # 6. LaTeX for the paper configurations
    paper = [c for c in comp if c["run"].startswith("paper_")]
    if paper:
        tex = [r"\begin{tabular}{llrrr}", r"\hline",
               r"Model & Planets & $k$ & $\Delta\ln\hat{L}$ & $\Delta$BIC \\", r"\hline"]
        tex += [f"{c['model']} & {', '.join(c['planets'].split())} & {c['k']} & {c['d_ll_hat']:+.1f} & "
                f"{c['d_bic']:+.1f} \\\\" for c in paper if c["d_bic"] is not None]
        tex += [r"\hline", r"\end{tabular}"]
        (out / "model_comparison_paper.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    (out / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out / "SUMMARY.md"

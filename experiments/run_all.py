"""Reproduce all results: python experiments/run_all.py [--quick]."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xynoise.core import (Qaoa, classical_baseline, metrics, random_feasible_baseline,  # noqa: E402
                          random_problem)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    q = ap.parse_args().quick
    seeds = range(1 if q else 5)
    n, k = (6, 3) if q else (8, 4)
    noises = [0, 1e-3, 1e-2] if q else [0, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
    depths = [1, 2] if q else [1, 2, 3]
    traj, shots, restarts = (8, 100, 2) if q else (60, 100, 6)
    out = ROOT / ("results/quick" if q else "results")
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    configs = [("penalty-X", "x", "auto"), ("XY-Dicke", "xy", "dicke"), ("XY-uniform-init", "xy", "uniform")]
    nan = float("nan")
    for s in seeds:
        prob = random_problem(n, k, 0.5, s)
        rng = np.random.default_rng(s)
        rows.append(dict(seed=s, method="classical-local-search", depth=0, p2=0.0, p_feasible=1.0,
                         ar_feasible=nan, best_of_shots=classical_baseline(prob, rng, 200)))
        rows.append(dict(seed=s, method="random-feasible", depth=0, p2=0.0, p_feasible=1.0,
                         ar_feasible=nan, best_of_shots=random_feasible_baseline(prob, rng, shots)))
        for name, mix, init in configs:
            for d in depths:
                if name == "XY-uniform-init" and d != 2:
                    continue
                qa = Qaoa(prob, mix, d, init)
                th = qa.train(restarts, s)
                for p2 in noises:
                    pr = qa.run(th, p2, traj, rng)
                    rows.append(dict(seed=s, method=name, depth=d, p2=p2, **metrics(prob, pr, shots, rng)))
        print(f"seed {s} done {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "raw.csv", index=False)
    cols = ["p_feasible", "ar_feasible", "best_of_shots"]
    g = df.groupby(["method", "depth", "p2"])[cols].agg(["mean", "std"])
    g.to_csv(out / "summary.csv")
    with open(out / "meta.json", "w") as f:
        json.dump({"n": n, "k": k, "seeds": len(seeds), "traj": traj, "shots": shots,
                   "runtime_s": time.time() - t0}, f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = ROOT / ("figures/quick" if q else "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    d2 = df[(df.depth == 2) & df.method.isin([c[0] for c in configs])]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    labs = [("p_feasible", "P(feasible)"), ("ar_feasible", "Feasible-weighted approx. ratio"),
            ("best_of_shots", f"Best feasible of {shots} shots / optimum")]
    for ai, (col, lab) in zip(ax, labs):
        for m, sub in d2.groupby("method"):
            st = sub.groupby("p2")[col].agg(["mean", "std"]).reset_index()
            x = st.p2.replace(0, noises[1] / 3)
            ai.errorbar(x, st["mean"], st["std"].fillna(0), marker="o", capsize=3, label=m)
        if col == "best_of_shots":
            for m, c in [("classical-local-search", "gray"), ("random-feasible", "k")]:
                ai.axhline(df[df.method == m][col].mean(), ls="--", color=c, lw=1, label=m)
        ai.set_xscale("log")
        ai.set_xlabel("two-qubit gate error p2 (p2=0 plotted at left)")
        ai.set_ylabel(lab)
        ai.grid(alpha=.3)
    ax[0].legend(fontsize=7)
    ax[2].legend(fontsize=7)
    fig.suptitle(f"Depth-2, n={n}, k={k}, {len(seeds)} graphs (mean ± std)")
    fig.tight_layout()
    fig.savefig(fig_dir / "noise_sweep.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m in ["penalty-X", "XY-Dicke"]:
        for p2, ls in [(0, "-"), (noises[-2], "--")]:
            st = df[(df.method == m) & (df.p2 == p2)].groupby("depth")["ar_feasible"].agg(["mean", "std"]).reset_index()
            ax.errorbar(st.depth, st["mean"], st["std"].fillna(0), ls=ls, marker="o", capsize=3, label=f"{m}, p2={p2}")
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("Feasible-weighted approx. ratio")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "depth.png", dpi=200)
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()

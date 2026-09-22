"""Charge and A280 constraints across a panel of backbones, not one protein.

Targets are set RELATIVE to each backbone's own unconstrained mean charge, so
"push 10 e away from what the model wants" means the same thing on a 20-mer and
a 150-mer.  An absolute target does not: -20 e is routine on ubiquitin and
unreachable on Trp-cage.

Every condition is sampled with N_DESIGNS independent sequences per backbone,
and infeasible requests are recorded as such rather than skipped silently --
the feasibility boundary is a result, not an error.

Writes benchmarks/results/panel_charge.csv and panel_e280.csv.
"""

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_lib import mean, run_design

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL_DIR = os.path.join(HERE, "structures", "panel")
OUT = os.path.join(HERE, "results")

DELTAS = [-15, -10, -5, 0, 5, 10, 15]
E280_FLOORS = [None, 1490.0, 5500.0, 11000.0]
MODEL = "protein_mpnn"
SEED = 2024


def backbones():
    return sorted(glob.glob(os.path.join(PANEL_DIR, "*.pdb")))


def main(n_designs, n_baseline):
    os.makedirs(OUT, exist_ok=True)
    charge_rows, e280_rows = [], []

    for pdb in backbones():
        name = os.path.splitext(os.path.basename(pdb))[0]
        base = run_design(pdb, MODEL, batch_size=n_baseline, n_batches=1, seed=SEED,
                          extra_args=["--report_properties", 1])
        bd = base["designs"]
        L = len(bd[0]["sequence"])
        base_nll = mean(d["nll"] for d in bd)
        base_q = mean(d["charge"] for d in bd)
        base_e = mean(d["e280"] for d in bd)
        frac_chromo = sum(1 for d in bd if d["e280"] >= 5500) / len(bd)
        centre = round(base_q)
        print(f"\n=== {name} L={L} baseline charge {base_q:+.2f} NLL {base_nll:.4f} "
              f"e280 {base_e:.0f} (Trp in {frac_chromo:.0%})", flush=True)

        for delta in DELTAS:
            target = centre + delta
            try:
                res = run_design(pdb, MODEL, batch_size=n_designs, n_batches=1,
                                 seed=SEED, extra_args=["--target_charge", target])
            except RuntimeError as exc:
                print(f"  delta {delta:+3d} (target {target:+3d}): INFEASIBLE", flush=True)
                charge_rows.append({"backbone": name, "length": L, "model": MODEL,
                                    "delta": delta, "target_charge": target,
                                    "design": -1, "achieved_charge": "",
                                    "exact": "", "nll": "", "baseline_nll": base_nll,
                                    "baseline_charge": base_q, "seq_rec": "",
                                    "feasible": 0, "sequence": ""})
                continue
            ds = res["designs"]
            n_exact = sum(1 for d in ds if d["charge"] == target)
            for i, d in enumerate(ds):
                charge_rows.append({"backbone": name, "length": L, "model": MODEL,
                                    "delta": delta, "target_charge": target,
                                    "design": i, "achieved_charge": d["charge"],
                                    "exact": int(d["charge"] == target), "nll": d["nll"],
                                    "baseline_nll": base_nll, "baseline_charge": base_q,
                                    "seq_rec": d["seq_rec"], "feasible": 1,
                                    "sequence": d["sequence"]})
            print(f"  delta {delta:+3d} (target {target:+3d}): exact {n_exact}/{len(ds)}  "
                  f"NLL {mean(d['nll'] for d in ds):.4f}  (+{mean(d['nll'] for d in ds)-base_nll:.4f})",
                  flush=True)

        for floor in E280_FLOORS:
            extra = ["--report_properties", 1] if floor is None else \
                    ["--min_extinction_280", floor]
            try:
                res = run_design(pdb, MODEL, batch_size=n_designs, n_batches=1,
                                 seed=SEED, extra_args=extra)
            except RuntimeError:
                continue
            ds = res["designs"]
            ref = 5500.0 if floor is None else floor
            for i, d in enumerate(ds):
                e280_rows.append({"backbone": name, "length": L, "model": MODEL,
                                  "floor": 0 if floor is None else floor,
                                  "constrained": int(floor is not None),
                                  "design": i, "e280": d["e280"],
                                  "meets": int(d["e280"] >= ref),
                                  "n_trp": d["sequence"].count("W"),
                                  "n_tyr": d["sequence"].count("Y"),
                                  "nll": d["nll"], "baseline_nll": base_nll,
                                  "seq_rec": d["seq_rec"], "sequence": d["sequence"]})

    for path, rows in (("panel_charge.csv", charge_rows), ("panel_e280.csv", e280_rows)):
        with open(os.path.join(OUT, path), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> results/{path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-designs", type=int, default=16)
    ap.add_argument("--n-baseline", type=int, default=32)
    a = ap.parse_args()
    main(a.n_designs, a.n_baseline)

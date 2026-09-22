"""Rejection sampling: the true cost of conditioning, measured not assumed.

Constrained decoding approximates sampling from p(sequence | net charge = c).
Rejection sampling realises that conditional exactly: draw a large
unconstrained pool and keep the designs that already have charge c.  Those are
by construction drawn from the true conditional, so their mean NLL is the
honest reference.  Any excess the constrained sampler shows above it is
distortion introduced by the controller, not the price of asking.

This is the experiment that showed the original mean-matching controller was
paying ~0.5 nats/residue of pure overhead on ubiquitin, and it is the yardstick
for the lookahead controller that replaced it.

    python benchmarks/bench_rejection.py --pool 1024 --n-designs 32

Writes benchmarks/results/rejection.csv (every pooled design) and
rejection_compare.csv (per backbone and charge: rejection vs constrained).
"""

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_lib import mean, run_design

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
PANEL_DIR = os.path.join(HERE, "structures", "panel")
# A spread of length and fold class; the full panel would cost ~3x this.
DEFAULT = ["1VII", "1ENH", "1PGA", "1SHG", "1UBQ", "1TEN"]
MODEL = "protein_mpnn"
SEED = 90210


def stdev(xs):
    xs = list(xs)
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main(names, pool, n_designs, min_bin):
    os.makedirs(OUT, exist_ok=True)
    pool_rows, cmp_rows = [], []

    for name in names:
        pdb = os.path.join(PANEL_DIR, f"{name}.pdb")
        if not os.path.exists(pdb):
            print(f"skip {name}: not in panel", flush=True)
            continue
        batch = min(pool, 64)
        res = run_design(pdb, MODEL, batch_size=batch, n_batches=max(1, pool // batch),
                         seed=SEED, extra_args=["--report_properties", 1])
        ds = res["designs"]
        L = len(ds[0]["sequence"])
        base_nll = mean(d["nll"] for d in ds)
        for i, d in enumerate(ds):
            pool_rows.append({"backbone": name, "length": L, "design": i,
                              "charge": d["charge"], "nll": d["nll"],
                              "e280": d["e280"], "seq_rec": d["seq_rec"],
                              "sequence": d["sequence"]})
        counts = {}
        for d in ds:
            counts[d["charge"]] = counts.get(d["charge"], 0) + 1
        print(f"\n=== {name} L={L}  pool {len(ds)}  mean charge "
              f"{mean(d['charge'] for d in ds):+.2f}  pool NLL {base_nll:.4f}", flush=True)

        for c in sorted(counts):
            if counts[c] < min_bin:
                continue
            acc = [d["nll"] for d in ds if d["charge"] == c]
            rej_nll, rej_sd, n = mean(acc), stdev(acc), len(acc)
            con = run_design(pdb, MODEL, batch_size=n_designs, n_batches=1, seed=SEED,
                             extra_args=["--target_charge", c])
            cd = con["designs"]
            con_nll = mean(d["nll"] for d in cd)
            n_exact = sum(1 for d in cd if d["charge"] == c)
            cmp_rows.append({"backbone": name, "length": L, "charge": c,
                             "n_rejection": n, "rejection_nll": rej_nll,
                             "rejection_sd": rej_sd,
                             "rejection_se": rej_sd / (n ** 0.5) if n > 1 else "",
                             "n_constrained": len(cd), "constrained_nll": con_nll,
                             "constrained_exact": n_exact,
                             "excess_nll": con_nll - rej_nll,
                             "pool_nll": base_nll})
            print(f"  charge {c:+5.0f}: rejection {rej_nll:.4f} (n={n:3d})   "
                  f"constrained {con_nll:.4f}  exact {n_exact}/{len(cd)}   "
                  f"excess {con_nll - rej_nll:+.4f}", flush=True)

    for path, rows in (("rejection.csv", pool_rows),
                       ("rejection_compare.csv", cmp_rows)):
        with open(os.path.join(OUT, path), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> results/{path}")
    if cmp_rows:
        ex = [r["excess_nll"] for r in cmp_rows]
        print(f"\nexcess over true conditional across {len(ex)} backbone/charge cells: "
              f"mean {mean(ex):+.4f}, max {max(ex):+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", nargs="*", default=DEFAULT)
    ap.add_argument("--pool", type=int, default=1024)
    ap.add_argument("--n-designs", type=int, default=32)
    ap.add_argument("--min-bin", type=int, default=20)
    a = ap.parse_args()
    main(a.panel, a.pool, a.n_designs, a.min_bin)

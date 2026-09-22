"""Benchmark 2: does --min_extinction_280 install a 280 nm chromophore, and
what else does it change?

Target: the villin headpiece subdomain HP36 (PDB 1VII, 36 residues), a small
folded three-helix peptide whose native sequence carries exactly one Trp.
The model is SolubleMPNN, which was chosen because it is reluctant to place
Trp here -- see the measured baseline in the output.

The test is two-sided.  The flag should (a) always reach the requested
extinction coefficient and (b) leave the rest of the amino-acid distribution
largely alone, rather than dragging the whole design toward aromatics.

Writes benchmarks/results/extinction.csv and extinction_composition.csv.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_lib import (
    AA20,
    STRUCTURES,
    composition,
    extinction_280,
    jensen_shannon,
    mean,
    run_design,
)

# 1490 = one Tyr; 5500 = one Trp; 11000 = two Trp equivalents.
THRESHOLDS = [None, 1490.0, 5500.0, 11000.0]
MODEL = "soluble_mpnn"
N_DESIGNS = 64
SEED = 2024
PDB = os.path.join(STRUCTURES, "1VII.pdb")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, comp_rows = [], []
    baseline_comp = None

    for thr in THRESHOLDS:
        extra = [] if thr is None else ["--min_extinction_280", thr]
        res = run_design(
            PDB, MODEL, batch_size=N_DESIGNS, n_batches=1, seed=SEED, extra_args=extra
        )
        designs = res["designs"]
        seqs = [d["sequence"] for d in designs]
        label = "unconstrained" if thr is None else f"{thr:.0f}"
        comp = composition(seqs)
        if thr is None:
            baseline_comp = comp

        ref = thr if thr is not None else 5500.0
        met = sum(1 for d in designs if d["e280"] >= ref) / len(designs)
        for i, d in enumerate(designs):
            rows.append(
                {
                    "threshold": label,
                    "threshold_value": 0.0 if thr is None else thr,
                    "design": i,
                    "e280": d["e280"],
                    "n_trp": d["sequence"].count("W"),
                    "n_tyr": d["sequence"].count("Y"),
                    "meets_threshold": int(d["e280"] >= ref),
                    "nll": d["nll"],
                    "seq_rec": d["seq_rec"],
                    "charge": d["charge"],
                    "spps_risk": d["spps_risk"],
                    "sequence": d["sequence"],
                }
            )
        js = jensen_shannon(comp, baseline_comp)
        for aa in AA20:
            comp_rows.append(
                {
                    "threshold": label,
                    "aa": aa,
                    "frequency": comp[aa],
                    "baseline_frequency": baseline_comp[aa],
                }
            )
        print(
            f"[{label:>13}] meets>={ref:.0f}: {met:6.1%}  "
            f"mean e280 {mean(d['e280'] for d in designs):8.0f}  "
            f"mean Trp {mean(s.count('W') for s in seqs):.2f}  "
            f"mean Tyr {mean(s.count('Y') for s in seqs):.2f}  "
            f"NLL {mean(d['nll'] for d in designs):.4f}  "
            f"JS vs baseline {js:.4f} bits",
            flush=True,
        )

    with open(os.path.join(OUT, "extinction.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "extinction_composition.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(comp_rows[0]))
        w.writeheader()
        w.writerows(comp_rows)
    print(f"\nwrote {len(rows)} design rows")
    print(f"native 1VII e280 = {extinction_280(res['native']):.0f}")


if __name__ == "__main__":
    main()

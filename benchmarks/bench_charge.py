"""Benchmark 1: does --target_charge hit the requested net charge exactly?

Target: ubiquitin (PDB 1UBQ, chain A, 76 residues), a small common globular
protein.  The full chain is redesigned.  The sweep runs well past the charge
the unconstrained model produces on its own, so the cost of an increasingly
unnatural request is visible.

Writes benchmarks/results/charge_sweep.csv and charge_baseline.csv.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_lib import (
    REPO,
    STRUCTURES,
    extinction_280,
    mean,
    net_charge,
    run_design,
    spps_risk,
)

TARGETS = list(range(-30, 35, 5))
MODELS = ["protein_mpnn", "soluble_mpnn", "ligand_mpnn"]
N_DESIGNS = 32
SEED = 2024
PDB = os.path.join(STRUCTURES, "1UBQ.pdb")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def tolerance_sweep():
    """How much of the likelihood cost is exactness itself, not the target?

    Pinning the charge to a single integer removes the spread the
    unconstrained model produces naturally, which costs likelihood even when
    the requested value sits right on the unconstrained mean.  Widening
    --charge_tolerance buys that spread back.
    """
    os.makedirs(OUT, exist_ok=True)
    rows = []
    model = "protein_mpnn"
    base = run_design(PDB, model, batch_size=N_DESIGNS, n_batches=1, seed=SEED)
    base_nll = mean(d["nll"] for d in base["designs"])
    base_charge = mean(d["charge"] for d in base["designs"])
    print(
        f"[{model}] unconstrained: charge {base_charge:+.2f}, NLL {base_nll:.4f}",
        flush=True,
    )
    for target in (-5, -15):
        for tol in (0, 1, 2, 4, 8):
            res = run_design(
                PDB,
                model,
                batch_size=N_DESIGNS,
                n_batches=1,
                seed=SEED,
                extra_args=[
                    "--target_charge",
                    target,
                    "--charge_tolerance",
                    tol,
                ],
            )
            got = [d["charge"] for d in res["designs"]]
            inside = sum(1 for q in got if abs(q - target) <= tol)
            for i, d in enumerate(res["designs"]):
                rows.append(
                    {
                        "model": model,
                        "target_charge": target,
                        "tolerance": tol,
                        "design": i,
                        "achieved_charge": d["charge"],
                        "inside_window": int(abs(d["charge"] - target) <= tol),
                        "nll": d["nll"],
                        "baseline_nll": base_nll,
                        "seq_rec": d["seq_rec"],
                        "sequence": d["sequence"],
                    }
                )
            print(
                f"[{model}] target {target:+3d} tol +-{tol}: "
                f"inside {inside}/{len(got)}, spread {min(got):+.0f}..{max(got):+.0f}, "
                f"NLL {mean(d['nll'] for d in res['designs']):.4f} "
                f"(baseline {base_nll:.4f})",
                flush=True,
            )
    with open(os.path.join(OUT, "charge_tolerance.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, base_rows = [], []

    for model in MODELS:
        base = run_design(
            PDB, model, batch_size=N_DESIGNS, n_batches=1, seed=SEED
        )
        native = base["native"]
        for i, d in enumerate(base["designs"]):
            base_rows.append(
                {
                    "model": model,
                    "design": i,
                    "charge": d["charge"],
                    "nll": d["nll"],
                    "seq_rec": d["seq_rec"],
                    "sequence": d["sequence"],
                }
            )
        base_charge = mean(d["charge"] for d in base["designs"])
        base_nll = mean(d["nll"] for d in base["designs"])
        print(
            f"[{model}] unconstrained: charge {base_charge:+.2f}, "
            f"NLL {base_nll:.4f}, native charge {net_charge(native):+.0f}",
            flush=True,
        )

        for target in TARGETS:
            res = run_design(
                PDB,
                model,
                batch_size=N_DESIGNS,
                n_batches=1,
                seed=SEED,
                extra_args=["--target_charge", target],
            )
            got = [d["charge"] for d in res["designs"]]
            n_exact = sum(1 for q in got if q == target)
            for i, d in enumerate(res["designs"]):
                rows.append(
                    {
                        "model": model,
                        "target_charge": target,
                        "design": i,
                        "achieved_charge": d["charge"],
                        "exact": int(d["charge"] == target),
                        "nll": d["nll"],
                        "baseline_nll": base_nll,
                        "seq_rec": d["seq_rec"],
                        "e280": d["e280"],
                        "spps_risk": d["spps_risk"],
                        "sequence": d["sequence"],
                    }
                )
            print(
                f"[{model}] target {target:+4d}: exact {n_exact}/{len(got)}, "
                f"NLL {mean(d['nll'] for d in res['designs']):.4f} "
                f"(baseline {base_nll:.4f})",
                flush=True,
            )

    with open(os.path.join(OUT, "charge_sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "charge_baseline.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(base_rows[0]))
        w.writeheader()
        w.writerows(base_rows)
    n_exact = sum(r["exact"] for r in rows)
    print(f"\nwrote {len(rows)} rows; exact on {n_exact}/{len(rows)} designs")


if __name__ == "__main__":
    section = sys.argv[1] if len(sys.argv) > 1 else "all"
    if section in ("all", "sweep"):
        main()
    if section in ("all", "tolerance"):
        tolerance_sweep()

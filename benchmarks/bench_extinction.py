"""Benchmark 2: final epsilon-280 values and composition changes.

The first output covers several backbones and all three standard MPNN models.
The second output keeps one backbone/model fixed and compares 50 designs with
and without an 8000 M^-1 cm^-1 floor.

Writes benchmarks/results/extinction.csv and extinction_composition.csv.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_lib import AA20, STRUCTURES, extinction_280, mean, run_design

THRESHOLDS = [None, 0.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
MODELS = ["protein_mpnn", "soluble_mpnn", "ligand_mpnn"]
PANEL = ["1L2Y", "1PGA", "1UBQ", "1VII"]
COMPOSITION_BACKBONE = "1VII"
COMPOSITION_MODEL = "soluble_mpnn"
N_DESIGNS = 50
SEED = 2024
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, comp_rows = [], []

    for stem in PANEL:
        pdb = os.path.join(STRUCTURES, f"{stem}.pdb")
        for model in MODELS:
            baseline_comp = None
            for threshold in THRESHOLDS:
                extra = [] if threshold is None else [
                    "--min_extinction_280",
                    threshold,
                ]
                result = run_design(
                    pdb,
                    model,
                    batch_size=N_DESIGNS,
                    n_batches=1,
                    seed=SEED,
                    extra_args=extra,
                )
                designs = result["designs"]
                sequences = [d["sequence"] for d in designs]
                label = "off" if threshold is None else f"{threshold:.0f}"
                composition = {
                    aa: sum(seq.count(aa) for seq in sequences)
                    / sum(len(seq) for seq in sequences)
                    for aa in AA20
                }
                if threshold is None:
                    baseline_comp = composition
                required = threshold if threshold is not None else 0.0
                for index, design in enumerate(designs):
                    rows.append(
                        {
                            "backbone": stem,
                            "model": model,
                            "threshold": label,
                            "threshold_value": required,
                            "design": index,
                            "e280": design["e280"],
                            "n_trp": sequences[index].count("W"),
                            "n_tyr": sequences[index].count("Y"),
                            "meets_threshold": int(design["e280"] >= required),
                            "seq_rec": design["seq_rec"],
                            "charge": design["charge"],
                            "spps_risk": design["spps_risk"],
                            "sequence": design["sequence"],
                        }
                    )
                for aa in AA20:
                    comp_rows.append(
                        {
                            "backbone": stem,
                            "model": model,
                            "threshold": label,
                            "aa": aa,
                            "frequency": composition[aa],
                            "baseline_frequency": baseline_comp[aa],
                        }
                    )
                print(
                    f"[{stem} {model} {label:>5}] "
                    f"mean e280 {mean(d['e280'] for d in designs):8.0f}  "
                    f"range {min(d['e280'] for d in designs):.0f}.."
                    f"{max(d['e280'] for d in designs):.0f}  "
                    f"meets {sum(d['e280'] >= required for d in designs)}/{len(designs)}",
                    flush=True,
                )
            print(
                f"native {stem} {model} e280 = "
                f"{extinction_280(result['native']):.0f}",
                flush=True,
            )

    with open(os.path.join(OUT, "extinction.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(OUT, "extinction_composition.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(comp_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comp_rows)
    print(f"wrote {len(rows)} design rows and {len(comp_rows)} composition rows")


if __name__ == "__main__":
    main()

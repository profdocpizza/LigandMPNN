"""Benchmark 3: what does --spps_bias do to synthesis-risk motifs, and at
what cost?

Two targets are used, because they exercise different rules.

``1VII`` -- the villin headpiece subdomain HP36, 36 residues, comfortably
inside routine Fmoc-SPPS range.  Designs on this backbone contain no Asp-Gly
at all, so here the score is driven almost entirely by beta-branched load.

``1PGA`` -- the B1 domain of protein G, 56 residues, sampled at T=0.3.  This
is the target where aspartimide-prone Asp-X motifs actually appear in
unconstrained designs, so it is the one that can say anything about those
rules.  At 56 residues it sits at the upper edge of routine SPPS.

Unlike the charge and extinction constraints this one has no guarantee
attached, so the benchmark reports the whole distribution and the individual
motif counts rather than a satisfaction rate, plus the NLL and
sequence-recovery cost of each setting.

Writes benchmarks/results/spps.csv and spps_components.csv.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_lib import STRUCTURES, mean, run_design, spps_risk

STRENGTHS = [0.0, 0.5, 1.0, 2.0, 4.0]
# (label, pdb stem, model, temperature)
TARGETS = [
    ("1VII", "1VII", "protein_mpnn", 0.1),
    ("1PGA", "1PGA", "protein_mpnn", 0.3),
]
N_DESIGNS = 64
SEED = 2024
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

COMPONENTS = [
    "asp_gly",
    "asp_other",
    "beta_branched",
    "beta_pair",
    "beta_run3",
    "aliphatic_window",
    "cys",
    "met",
]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, comp_rows = [], []

    for label, stem, model, temperature in TARGETS:
        pdb = os.path.join(STRUCTURES, f"{stem}.pdb")
        print(f"\n=== {label} ({model}, T={temperature}) ===", flush=True)
        for strength in STRENGTHS:
            extra = [] if strength == 0.0 else ["--spps_bias", strength]
            res = run_design(
                pdb,
                model,
                batch_size=N_DESIGNS,
                n_batches=1,
                seed=SEED,
                temperature=temperature,
                extra_args=extra,
            )
            designs = res["designs"]
            comps = [spps_risk(d["sequence"])[1] for d in designs]
            for i, (d, c) in enumerate(zip(designs, comps)):
                rows.append(
                    {
                        "target": label,
                        "model": model,
                        "temperature": temperature,
                        "spps_bias": strength,
                        "design": i,
                        "spps_risk": d["spps_risk"],
                        "nll": d["nll"],
                        "seq_rec": d["seq_rec"],
                        "charge": d["charge"],
                        "e280": d["e280"],
                        **{f"n_{k}": c[k] for k in COMPONENTS},
                        "sequence": d["sequence"],
                    }
                )
            frac_dg = sum(1 for c in comps if c["asp_gly"] > 0) / len(comps)
            for k in COMPONENTS:
                comp_rows.append(
                    {
                        "target": label,
                        "spps_bias": strength,
                        "component": k,
                        "mean_count": mean(c[k] for c in comps),
                    }
                )
            comp_rows.append(
                {
                    "target": label,
                    "spps_bias": strength,
                    "component": "frac_designs_with_asp_gly",
                    "mean_count": frac_dg,
                }
            )
            print(
                f"[bias {strength:4.1f}] risk {mean(d['spps_risk'] for d in designs):6.2f}  "
                f"DG {mean(c['asp_gly'] for c in comps):5.2f} ({frac_dg:5.1%} of designs)  "
                f"D-X {mean(c['asp_other'] for c in comps):5.2f}  "
                f"IVT {mean(c['beta_branched'] for c in comps):5.2f}  "
                f"pairs {mean(c['beta_pair'] for c in comps):5.2f}  "
                f"runs3 {mean(c['beta_run3'] for c in comps):5.2f}  "
                f"NLL {mean(d['nll'] for d in designs):.4f}  "
                f"rec {mean(d['seq_rec'] for d in designs):.3f}",
                flush=True,
            )
        native_total, native_comp = spps_risk(res["native"])
        print(f"native {label} risk = {native_total:.2f}  {native_comp}", flush=True)

    with open(os.path.join(OUT, "spps.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "spps_components.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(comp_rows[0]))
        w.writeheader()
        w.writerows(comp_rows)
    print(f"wrote {len(rows)} design rows")


if __name__ == "__main__":
    main()

"""Everything the claims need, and nothing else.

Run AFTER benchmarks/bench_panel.py has written panel_charge.csv and
panel_e280.csv (step 1 of run_all_gpu.sh).  This covers the remaining four
questions at the smallest size that can actually answer them:

  rejection   Does the lookahead controller reach the cost of exact
              conditioning?  Needs depth, but only in the charge bins the
              model populates anyway -- a 768-design pool puts ~90 hits in the
              modal bin, giving a standard error near 0.005 nats, far below
              the ~0.05 effect being resolved.  Three backbones spanning
              36-153 residues is enough to show it is not a single-protein
              fluke.
  models      Does it work for every model type?  Binary; 8 sequences per
              cell is plenty.
  tolerance   Does a tolerance actually reduce cost?
  e280        Does the A280 floor hold across model types?
  spps        The synthesis dial, which has no guarantee to test and so only
              needs its trend.

Roughly 3,000 designs instead of 35,000.  Writes rejection.csv,
rejection_compare.csv, charge_sweep.csv, charge_baseline.csv,
charge_tolerance.csv, extinction_models.csv and spps.csv.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_lib import mean, run_design, spps_risk

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "structures", "panel")
STRUCT = os.path.join(HERE, "structures")
OUT = os.path.join(HERE, "results")
SEED = 2024
MODELS = ["protein_mpnn", "soluble_mpnn", "ligand_mpnn"]
# short / medium / long, so any length dependence in the residual shows up
REJ_BACKBONES = [("1VII", 36), ("1ENH", 54), ("1UBQ", 76), ("1PIN", 153),
                 ("1EMA", 221)]
# SPPS only makes sense for peptides that can be made on resin.
PEPTIDES = ["1L2Y", "1VII", "1ENH", "1PGA", "1BDD"]
PEPTIDE_DIR = os.path.join(HERE, "structures", "peptides")


def sd(xs):
    xs = list(xs)
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def write(path, rows):
    with open(os.path.join(OUT, path), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows):5d} rows -> results/{path}", flush=True)


def rejection(pool, n_designs, min_bin):
    print(f"\n=== 1/5  rejection reference ({len(REJ_BACKBONES)} backbones) ===",
          flush=True)
    pool_rows, cmp_rows = [], []
    done = set()
    # Resume: a CUDA OOM or Ctrl-C part-way through should not cost the
    # backbones that already completed.
    for path, store in (("rejection.csv", pool_rows),
                        ("rejection_compare.csv", cmp_rows)):
        fp = os.path.join(OUT, path)
        if os.path.exists(fp):
            with open(fp) as fh:
                rows = list(csv.DictReader(fh))
            store.extend(rows)
            if path == "rejection_compare.csv":
                done = {r["backbone"] for r in rows
                        if abs(int(r.get("pool_size", 0)) - pool) <= 0.1 * pool}
                stale = {r["backbone"] for r in rows} - done
                if stale:
                    print(f"  ignoring rows from a different pool size: "
                          f"{', '.join(sorted(stale))}", flush=True)
                    store[:] = [r for r in rows if r["backbone"] in done]
    if done:
        print(f"  resuming; already have {', '.join(sorted(done))}", flush=True)
    for name, _ in REJ_BACKBONES:
        if name in done:
            continue
        pdb = os.path.join(PANEL, f"{name}.pdb")
        batch = min(pool, 128)
        res = run_design(pdb, "protein_mpnn", batch_size=batch,
                         n_batches=max(1, pool // batch), seed=SEED,
                         extra_args=["--report_properties", 1])
        ds = res["designs"]
        L = len(ds[0]["sequence"])
        pool_nll = mean(d["nll"] for d in ds)
        for i, d in enumerate(ds):
            pool_rows.append({"backbone": name, "length": L, "design": i,
                              "charge": d["charge"], "nll": d["nll"],
                              "e280": d["e280"], "sequence": d["sequence"]})
        counts = {}
        for d in ds:
            counts[d["charge"]] = counts.get(d["charge"], 0) + 1
        print(f"  {name} L={L}: pool {len(ds)}, mean charge "
              f"{mean(d['charge'] for d in ds):+.2f}, pool NLL {pool_nll:.4f}", flush=True)
        for c in sorted(k for k, v in counts.items() if v >= min_bin):
            acc = [d["nll"] for d in ds if d["charge"] == c]
            con = run_design(pdb, "protein_mpnn", batch_size=n_designs, n_batches=1,
                             seed=SEED, extra_args=["--target_charge", c])
            cd = con["designs"]
            cn, rn = mean(d["nll"] for d in cd), mean(acc)
            cmp_rows.append({"backbone": name, "length": L, "charge": c,
                             "n_rejection": len(acc), "rejection_nll": rn,
                             "rejection_sd": sd(acc),
                             "rejection_se": sd(acc) / len(acc) ** 0.5,
                             "n_constrained": len(cd), "constrained_nll": cn,
                             "constrained_exact": sum(1 for d in cd if d["charge"] == c),
                             "excess_nll": cn - rn, "pool_nll": pool_nll,
                             "pool_size": len(ds)})
            print(f"    charge {c:+5.0f}: rejection {rn:.4f} (n={len(acc):3d})  "
                  f"constrained {cn:.4f}  excess {cn - rn:+.4f}", flush=True)
        # Checkpoint now: an OOM on the next (longer) backbone must not
        # discard the ones that already finished.
        write("rejection.csv", pool_rows)
        write("rejection_compare.csv", cmp_rows)
    ex = [float(r["excess_nll"]) for r in cmp_rows]
    print(f"  excess over the true conditional: median {sorted(ex)[len(ex)//2]:+.4f}, "
          f"max {max(ex):+.4f}, over {len(ex)} cells", flush=True)


def models(n_designs, n_base):
    print("\n=== 2/5  all three model types (1UBQ) ===", flush=True)
    pdb = os.path.join(PANEL, "1UBQ.pdb")
    rows, base_rows = [], []
    for m in MODELS:
        b = run_design(pdb, m, batch_size=n_base, n_batches=1, seed=SEED,
                       extra_args=["--report_properties", 1])
        bn = mean(d["nll"] for d in b["designs"])
        for i, d in enumerate(b["designs"]):
            base_rows.append({"model": m, "design": i, "charge": d["charge"],
                              "nll": d["nll"], "seq_rec": d["seq_rec"],
                              "sequence": d["sequence"]})
        centre = round(mean(d["charge"] for d in b["designs"]))
        for delta in (-10, -5, 0, 5, 10):
            t = centre + delta
            r = run_design(pdb, m, batch_size=n_designs, n_batches=1, seed=SEED,
                           extra_args=["--target_charge", t])
            ds = r["designs"]
            for i, d in enumerate(ds):
                rows.append({"model": m, "target_charge": t, "design": i,
                             "achieved_charge": d["charge"],
                             "exact": int(d["charge"] == t), "nll": d["nll"],
                             "baseline_nll": bn, "seq_rec": d["seq_rec"],
                             "e280": d["e280"], "spps_risk": d["spps_risk"],
                             "sequence": d["sequence"]})
            print(f"  {m:14s} target {t:+4d}: exact {sum(1 for d in ds if d['charge']==t)}"
                  f"/{len(ds)}  +{mean(d['nll'] for d in ds)-bn:.4f}", flush=True)
    write("charge_sweep.csv", rows)
    write("charge_baseline.csv", base_rows)


def tolerance(n_designs):
    print("\n=== 3/5  tolerance (1UBQ, protein_mpnn) ===", flush=True)
    pdb = os.path.join(PANEL, "1UBQ.pdb")
    rows = []
    for target in (-15, -5):
        for tol in (0, 2, 8):
            r = run_design(pdb, "protein_mpnn", batch_size=n_designs, n_batches=1,
                           seed=SEED, extra_args=["--target_charge", target,
                                                  "--charge_tolerance", tol])
            ds = r["designs"]
            got = [d["charge"] for d in ds]
            for i, d in enumerate(ds):
                rows.append({"model": "protein_mpnn", "target_charge": target,
                             "tolerance": tol, "design": i,
                             "achieved_charge": d["charge"],
                             "inside_window": int(abs(d["charge"] - target) <= tol),
                             "nll": d["nll"], "seq_rec": d["seq_rec"],
                             "sequence": d["sequence"]})
            print(f"  target {target:+3d} tol +-{tol}: inside "
                  f"{sum(1 for q in got if abs(q-target)<=tol)}/{len(got)}  "
                  f"spread {min(got):+.0f}..{max(got):+.0f}  "
                  f"NLL {mean(d['nll'] for d in ds):.4f}", flush=True)
    write("charge_tolerance.csv", rows)


def extinction(n_designs):
    print("\n=== 4/5  A280 floor across model types (1VII) ===", flush=True)
    pdb = os.path.join(PANEL, "1VII.pdb")
    rows = []
    for m in MODELS:
        for floor in (None, 5500.0, 11000.0):
            extra = ["--report_properties", 1] if floor is None else \
                    ["--min_extinction_280", floor]
            r = run_design(pdb, m, batch_size=n_designs, n_batches=1, seed=SEED,
                           extra_args=extra)
            ds = r["designs"]
            ref = 5500.0 if floor is None else floor
            for i, d in enumerate(ds):
                rows.append({"backbone": "1VII", "model": m,
                             "floor": 0 if floor is None else floor,
                             "constrained": int(floor is not None), "design": i,
                             "e280": d["e280"], "meets": int(d["e280"] >= ref),
                             "n_trp": d["sequence"].count("W"),
                             "n_tyr": d["sequence"].count("Y"),
                             "nll": d["nll"], "sequence": d["sequence"]})
            lab = "off" if floor is None else f"{floor:.0f}"
            print(f"  {m:14s} floor {lab:>6s}: meets "
                  f"{sum(1 for d in ds if d['e280']>=ref)}/{len(ds)}  "
                  f"mean e280 {mean(d['e280'] for d in ds):7.0f}", flush=True)
    write("extinction_models.csv", rows)


def spps(n_designs):
    print("\n=== 5/5  SPPS dial ===", flush=True)
    rows = []
    for stem in PEPTIDES:
        label, temp = stem, 0.3   # T=0.3; at 0.1 the motifs never appear at all
        pdb = os.path.join(PEPTIDE_DIR, f"{stem}.pdb")
        for strength in (0.0, 0.5, 1.0, 2.0):
            extra = [] if strength == 0 else ["--spps_bias", strength]
            r = run_design(pdb, "protein_mpnn", batch_size=n_designs, n_batches=1,
                           seed=SEED, temperature=temp,
                           extra_args=extra + ["--report_properties", 1])
            ds = r["designs"]
            comps = [spps_risk(d["sequence"])[1] for d in ds]
            for i, (d, c) in enumerate(zip(ds, comps)):
                rows.append({"target": label, "model": "protein_mpnn",
                             "temperature": temp, "spps_bias": strength,
                             "design": i, "spps_risk": d["spps_risk"],
                             "nll": d["nll"], "seq_rec": d["seq_rec"],
                             **{f"n_{k}": c[k] for k in c},
                             "sequence": d["sequence"]})
            print(f"  {label} bias {strength:3.1f}: risk "
                  f"{mean(d['spps_risk'] for d in ds):6.2f}  "
                  f"DG {mean(c['asp_gly'] for c in comps):4.2f}  "
                  f"NLL {mean(d['nll'] for d in ds):.4f}", flush=True)
    write("spps.csv", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=384)
    ap.add_argument("--min-bin", type=int, default=25)
    ap.add_argument("--n-designs", type=int, default=24)
    ap.add_argument("--n-small", type=int, default=16)
    ap.add_argument("--n-baseline", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rejection(a.pool, a.n_designs, a.min_bin)
    models(a.n_small, a.n_baseline)
    tolerance(a.n_small)
    extinction(a.n_small)
    spps(a.n_designs)
    print("\nall done")

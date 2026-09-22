"""No-folding validation experiments for constrained decoding.

The reference for the distributional tests is an unconstrained sample from the
same backbone, model, temperature, and seed.  This is deliberately different
from treating MPNN's mean NLL as ground truth:

* rejection sampling is an exact sample from the unconstrained model
  conditional on an achieved charge;
* ``--bias_AA`` is compared at matched achieved-charge targets;
* composition changes are compared with a ten-seed unconstrained null;
* short/fixed/omitted/symmetric cases distinguish a clean infeasibility error
  from a sampled sequence that silently misses its target.

No structure prediction or folding is performed here.

Examples::

    # Full rejection experiment (20k pool, 200 constrained designs).
    python benchmarks/bench_validation.py rejection --pool-size 20000

    # Run a quick local smoke test when checkpoints are available.
    python benchmarks/bench_validation.py all --pool-size 256 --n-designs 32 \
        --min-bin 8 --panel benchmarks/structures/1UBQ.pdb

The default panel is the four checked-in benchmark structures.  Supply
``--panel`` repeatedly, or a directory with ``--panel-dir``, for the intended
25--30 backbone panel.  Panel discovery is explicit so adding a backbone does
not silently change a published result.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import sys
import tempfile
from collections import Counter
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_lib import (  # noqa: E402
    REPO,
    STRUCTURES,
    composition,
    jensen_shannon,
    mean,
    run_design,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DEFAULT_PANEL = sorted(
    os.path.join(STRUCTURES, name)
    for name in ("1L2Y.pdb", "1PGA.pdb", "1UBQ.pdb", "1VII.pdb")
)
TEMPERATURES = (0.1, 0.2, 0.3)
BIAS_MAGNITUDES = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
SEED_SET = tuple(range(2024, 2034))
CHARGE_OFFSETS = (-10, -5, 0, 5, 10)


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _write_csv(name, rows):
    if not rows:
        return
    os.makedirs(RESULTS, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(os.path.join(RESULTS, name), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _n_batches(n, batch_size):
    return max(1, math.ceil(n / batch_size))


class Runner:
    """Cache identical CLI runs while keeping every arm user-visible."""

    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size
        self.cache = {}

    def run(self, pdb, n, seed, temperature, extra=(), allow_failure=False):
        key = (
            os.path.abspath(pdb),
            self.model,
            int(n),
            int(seed),
            float(temperature),
            tuple(str(x) for x in extra),
            bool(allow_failure),
        )
        if key not in self.cache:
            self.cache[key] = run_design(
                pdb,
                self.model,
                batch_size=min(self.batch_size, n),
                n_batches=_n_batches(n, self.batch_size),
                seed=seed,
                temperature=temperature,
                extra_args=extra,
                allow_failure=allow_failure,
            )
        result = self.cache[key]
        if result["designs"]:
            result = dict(result)
            result["designs"] = result["designs"][:n]
        return result


def _pairwise_diversity(seqs):
    if len(seqs) < 2:
        return float("nan")
    length = len(seqs[0])
    if any(len(seq) != length for seq in seqs):
        raise ValueError("pairwise diversity requires equal-length sequences")
    return mean(
        sum(a != b for a, b in zip(x, y)) / length
        for x, y in combinations(seqs, 2)
    )


def _mean_se(xs):
    values = list(xs)
    if not values:
        return float("nan"), float("nan")
    average = mean(values)
    if len(values) < 2:
        return average, float("nan")
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return average, math.sqrt(variance / len(values))


def _design_rows(pdb, designs, **meta):
    rows = []
    for index, design in enumerate(designs):
        rows.append(
            {
                **meta,
                "protein": _stem(pdb),
                "design": index,
                "nll": design["nll"],
                "seq_rec": design["seq_rec"],
                "charge": design["charge"],
                "e280": design["e280"],
                "spps_risk": design["spps_risk"],
                "sequence": design["sequence"],
            }
        )
    return rows


def _charge_targets(designs, min_bin, max_targets=5):
    counts = Counter(round(d["charge"]) for d in designs)
    populated = sorted(q for q, count in counts.items() if count >= min_bin)
    if len(populated) <= max_targets:
        return populated
    # Keep evenly spaced bins, including both tails, rather than selecting
    # targets based on which one gives the most flattering NLL.
    if max_targets == 1:
        return [populated[len(populated) // 2]]
    indices = {
        round(i * (len(populated) - 1) / (max_targets - 1))
        for i in range(max_targets)
    }
    return [populated[i] for i in sorted(indices)]


def _bias_for_target(target, baseline_mean, magnitude):
    direction = 1.0 if target >= baseline_mean else -1.0
    return (
        "--bias_AA",
        f"D:{-direction * magnitude:g},E:{-direction * magnitude:g},"
        f"K:{direction * magnitude:g},R:{direction * magnitude:g}",
    )


def rejection(args, runner, panel):
    design_rows, summary_rows = [], []
    for temperature in args.temperatures:
        for pdb in panel:
            print(f"rejection { _stem(pdb) } T={temperature}", flush=True)
            baseline = runner.run(pdb, args.pool_size, args.seed, temperature)
            designs = baseline["designs"]
            if not designs:
                raise RuntimeError(f"no baseline designs for {pdb}")
            targets = _charge_targets(designs, args.min_bin, args.max_targets)
            counts = Counter(round(d["charge"]) for d in designs)
            for target in targets:
                accepted = [d for d in designs if round(d["charge"]) == target]
                constrained = runner.run(
                    pdb,
                    args.n_designs,
                    args.seed + 1,
                    temperature,
                    ("--target_charge", target),
                )["designs"]
                if len(accepted) < args.min_bin or not constrained:
                    continue
                rng = random.Random(args.seed + target + round(100 * temperature))
                accepted = rng.sample(accepted, min(len(accepted), len(constrained)))
                accepted_seqs = [d["sequence"] for d in accepted]
                constrained_seqs = [d["sequence"] for d in constrained[: len(accepted)]]
                accepted_comp = composition(accepted_seqs)
                constrained_comp = composition(constrained_seqs)
                accepted_nll, accepted_se = _mean_se(d["nll"] for d in accepted)
                constrained_nll, constrained_se = _mean_se(
                    d["nll"] for d in constrained[: len(accepted)]
                )
                summary_rows.append(
                    {
                        "protein": _stem(pdb),
                        "model": runner.model,
                        "temperature": temperature,
                        "target_charge": target,
                        "pool_size": len(designs),
                        "accepted_count": counts[target],
                        "acceptance_rate": counts[target] / len(designs),
                        "comparison_n": len(accepted),
                        "rejection_nll": accepted_nll,
                        "constrained_nll": constrained_nll,
                        "constrained_minus_rejection_nll": constrained_nll - accepted_nll,
                        "nll_difference_se": math.sqrt(
                            accepted_se * accepted_se + constrained_se * constrained_se
                        ),
                        "composition_js_bits": jensen_shannon(
                            accepted_comp, constrained_comp
                        ),
                        "rejection_diversity": _pairwise_diversity(accepted_seqs),
                        "constrained_diversity": _pairwise_diversity(constrained_seqs),
                    }
                )
                design_rows.extend(
                    _design_rows(
                        pdb,
                        accepted,
                        arm="rejection",
                        temperature=temperature,
                        target_charge=target,
                    )
                )
                design_rows.extend(
                    _design_rows(
                        pdb,
                        constrained[: len(accepted)],
                        arm="constrained",
                        temperature=temperature,
                        target_charge=target,
                    )
                )
    _write_csv("validation_rejection_designs.csv", design_rows)
    _write_csv("validation_rejection_summary.csv", summary_rows)
    print(f"wrote {len(summary_rows)} rejection comparisons", flush=True)


def bias(args, runner, panel):
    design_rows, summary_rows = [], []
    for temperature in args.temperatures:
        for pdb in panel:
            baseline = runner.run(pdb, args.n_designs, args.seed, temperature)
            base = baseline["designs"]
            base_mean = mean(d["charge"] for d in base)
            targets = [round(base_mean + offset) for offset in CHARGE_OFFSETS]
            for target in targets:
                constrained = runner.run(
                    pdb,
                    args.n_designs,
                    args.seed + 1,
                    temperature,
                    ("--target_charge", target),
                    allow_failure=True,
                )
                cdesigns = constrained["designs"]
                for magnitude in BIAS_MAGNITUDES:
                    extra = _bias_for_target(target, base_mean, magnitude)
                    result = runner.run(
                        pdb,
                        args.n_designs,
                        args.seed + 2,
                        temperature,
                        extra,
                        allow_failure=True,
                    )
                    designs = result["designs"]
                    if not designs:
                        continue
                    mean_charge = mean(d["charge"] for d in designs)
                    within = mean(abs(d["charge"] - target) <= 1 for d in designs)
                    n_to_95 = (
                        math.ceil(math.log(0.05) / math.log(1.0 - within))
                        if 0 < within < 1
                        else (1 if within == 1 else float("inf"))
                    )
                    summary_rows.append(
                        {
                            "protein": _stem(pdb),
                            "model": runner.model,
                            "temperature": temperature,
                            "target_charge": target,
                            "baseline_mean_charge": base_mean,
                            "method": "bias_AA",
                            "bias_magnitude": magnitude,
                            "bias": extra[1],
                            "mean_charge": mean_charge,
                            "mean_abs_charge_error": mean(
                                abs(d["charge"] - target) for d in designs
                            ),
                            "fraction_within_plus_minus_1": within,
                            "samples_for_95_percent_hit": n_to_95,
                            "mean_nll": mean(d["nll"] for d in designs),
                            "mean_seq_rec": mean(d["seq_rec"] for d in designs),
                        }
                    )
                    design_rows.extend(
                        _design_rows(
                            pdb,
                            designs,
                            method="bias_AA",
                            temperature=temperature,
                            target_charge=target,
                            bias_magnitude=magnitude,
                        )
                    )
                if cdesigns:
                    summary_rows.append(
                        {
                            "protein": _stem(pdb),
                            "model": runner.model,
                            "temperature": temperature,
                            "target_charge": target,
                            "baseline_mean_charge": base_mean,
                            "method": "constrained",
                            "bias_magnitude": "",
                            "bias": "",
                            "mean_charge": mean(d["charge"] for d in cdesigns),
                            "mean_abs_charge_error": mean(
                                abs(d["charge"] - target) for d in cdesigns
                            ),
                            "fraction_within_plus_minus_1": mean(
                                abs(d["charge"] - target) <= 1 for d in cdesigns
                            ),
                            "samples_for_95_percent_hit": 1,
                            "mean_nll": mean(d["nll"] for d in cdesigns),
                            "mean_seq_rec": mean(d["seq_rec"] for d in cdesigns),
                        }
                    )
    _write_csv("validation_bias_designs.csv", design_rows)
    _write_csv("validation_bias_summary.csv", summary_rows)
    print(f"wrote {len(summary_rows)} bias/constrained summaries", flush=True)


def panel(args, runner, panel_paths):
    rows = []
    for temperature in args.temperatures:
        for pdb in panel_paths:
            baseline = runner.run(pdb, args.n_designs, args.seed, temperature)
            base = baseline["designs"]
            base_nll = mean(d["nll"] for d in base)
            base_comp = composition([d["sequence"] for d in base])
            base_mean = mean(d["charge"] for d in base)
            rows.append(
                {
                    "protein": _stem(pdb),
                    "temperature": temperature,
                    "property": "baseline",
                    "target": "",
                    "satisfaction": 1.0,
                    "extra_nll": 0.0,
                    "composition_js_bits": 0.0,
                    "mean_value": base_mean,
                    "baseline_value": base_mean,
                    "baseline_nll": base_nll,
                    "length": len(base[0]["sequence"]),
                }
            )
            for target in (round(base_mean + offset) for offset in CHARGE_OFFSETS):
                result = runner.run(
                    pdb,
                    args.n_designs,
                    args.seed + 1,
                    temperature,
                    ("--target_charge", target),
                    allow_failure=True,
                )
                designs = result["designs"]
                if not designs:
                    rows.append(
                        {
                            "protein": _stem(pdb),
                            "temperature": temperature,
                            "property": "charge",
                            "target": target,
                            "status": "infeasible_or_failed",
                        }
                    )
                    continue
                rows.append(
                    {
                        "protein": _stem(pdb),
                        "temperature": temperature,
                        "property": "charge",
                        "target": target,
                        "satisfaction": mean(d["charge"] == target for d in designs),
                        "extra_nll": mean(d["nll"] for d in designs) - base_nll,
                        "composition_js_bits": jensen_shannon(
                            base_comp, composition([d["sequence"] for d in designs])
                        ),
                        "mean_value": mean(d["charge"] for d in designs),
                        "baseline_value": base_mean,
                        "baseline_nll": base_nll,
                        "length": len(designs[0]["sequence"]),
                    }
                )
            for threshold in (5500.0,):
                result = runner.run(
                    pdb,
                    args.n_designs,
                    args.seed + 1,
                    temperature,
                    ("--min_extinction_280", threshold),
                    allow_failure=True,
                )
                designs = result["designs"]
                if designs:
                    rows.append(
                        {
                            "protein": _stem(pdb),
                            "temperature": temperature,
                            "property": "e280",
                            "target": threshold,
                            "satisfaction": mean(d["e280"] >= threshold for d in designs),
                            "extra_nll": mean(d["nll"] for d in designs) - base_nll,
                            "composition_js_bits": jensen_shannon(
                                base_comp, composition([d["sequence"] for d in designs])
                            ),
                            "mean_value": mean(d["e280"] for d in designs),
                            "baseline_value": mean(d["e280"] for d in base),
                            "baseline_nll": base_nll,
                            "baseline_zero_e280_fraction": mean(
                                d["e280"] == 0 for d in base
                            ),
                            "length": len(designs[0]["sequence"]),
                        }
                    )
    _write_csv("validation_panel.csv", rows)
    print(f"wrote {len(rows)} backbone-panel rows", flush=True)


def seed_noise(args, runner, panel_paths):
    seed_rows, pair_rows, constrained_rows = [], [], []
    for temperature in args.temperatures:
        for pdb in panel_paths:
            compositions = {}
            for seed in SEED_SET:
                result = runner.run(pdb, args.n_designs, seed, temperature)
                designs = result["designs"]
                compositions[seed] = composition([d["sequence"] for d in designs])
                seed_rows.append(
                    {
                        "protein": _stem(pdb),
                        "temperature": temperature,
                        "seed": seed,
                        **compositions[seed],
                    }
                )
            values = []
            for seed_a, seed_b in combinations(SEED_SET, 2):
                value = jensen_shannon(compositions[seed_a], compositions[seed_b])
                values.append(value)
                pair_rows.append(
                    {
                        "protein": _stem(pdb),
                        "temperature": temperature,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "composition_js_bits": value,
                    }
                )
            reference = {
                aa: mean(compositions[seed][aa] for seed in SEED_SET)
                for aa in compositions[SEED_SET[0]]
            }
            constrained = runner.run(
                pdb,
                args.n_designs,
                args.seed + 1,
                temperature,
                ("--min_extinction_280", 5500.0),
                allow_failure=True,
            )["designs"]
            if constrained:
                constrained_js = jensen_shannon(
                    reference, composition([d["sequence"] for d in constrained])
                )
                constrained_rows.append(
                    {
                        "protein": _stem(pdb),
                        "temperature": temperature,
                        "constraint": "e280>=5500",
                        "composition_js_bits": constrained_js,
                        "seed_noise_mean_js_bits": mean(values),
                        "seed_noise_p95_js_bits": sorted(values)[int(0.95 * (len(values) - 1))],
                        "within_seed_noise_p95": int(
                            constrained_js <= sorted(values)[int(0.95 * (len(values) - 1))]
                        ),
                    }
                )
    _write_csv("validation_seed_compositions.csv", seed_rows)
    _write_csv("validation_seed_js.csv", pair_rows)
    _write_csv("validation_seed_summary.csv", constrained_rows)
    print(f"wrote {len(pair_rows)} seed-noise comparisons", flush=True)


def _residue_labels(path, max_residues=None):
    labels, seen = [], set()
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM") or line[21].strip() != "A":
                continue
            key = line[22:27].strip()
            if key not in seen:
                seen.add(key)
                labels.append("A" + key)
                if max_residues and len(labels) >= max_residues:
                    break
    return labels


def _truncate_chain_a(source, n, destination):
    keep = set(_residue_labels(source, n))
    with open(source) as src, open(destination, "w") as dst:
        for line in src:
            if line.startswith("ATOM") and line[21].strip() == "A":
                if "A" + line[22:27].strip() in keep:
                    dst.write(line)
            elif line.startswith(("TER", "END")):
                dst.write(line)


def _failure_text(result):
    text = (result.get("stderr", "") or "") + (result.get("stdout", "") or "")
    match = re.search(r"InfeasibleConstraint:.*", text)
    return match.group(0).strip() if match else text[-500:].replace("\n", " ")


def stress(args, runner):
    rows = []
    source = args.stress_source or os.path.join(STRUCTURES, "1UBQ.pdb")
    with tempfile.TemporaryDirectory(prefix="constraint_stress_") as tmp:
        for length in args.stress_lengths:
            truncated = os.path.join(tmp, f"chain_{length}.pdb")
            _truncate_chain_a(source, length, truncated)
            labels = _residue_labels(truncated)
            for fixed_fraction in args.fixed_fractions:
                fixed_n = round(len(labels) * fixed_fraction)
                fixed = labels[:fixed_n]
                designed = labels[fixed_n:]
                for omitted in ("", "DE", "KR"):
                    for group_size in args.symmetry_groups:
                        symmetry = ""
                        symmetry_weights = ""
                        if group_size:
                            if len(designed) < group_size:
                                continue
                            groups = [
                                designed[index : index + group_size]
                                for index in range(0, len(designed), group_size)
                                if len(designed[index : index + group_size])
                                == group_size
                            ]
                            symmetry = "|".join(
                                ",".join(group) for group in groups
                            )
                            symmetry_weights = "|".join(
                                ",".join("1.0" for _ in group) for group in groups
                            )
                        for direction in (-1, 1):
                            for step in range(args.stress_steps + 1):
                                target = direction * step
                                extra = ["--target_charge", target]
                                if fixed:
                                    extra += ["--fixed_residues", " ".join(fixed)]
                                if omitted:
                                    extra += ["--omit_AA", omitted]
                                if symmetry:
                                    extra += ["--symmetry_residues", symmetry]
                                    extra += ["--symmetry_weights", symmetry_weights]
                                result = runner.run(
                                    truncated,
                                    1,
                                    args.seed,
                                    args.temperature,
                                    extra,
                                    allow_failure=True,
                                )
                                status = "error" if not result["designs"] else "exact"
                                achieved = ""
                                if result["designs"]:
                                    achieved = result["designs"][0]["charge"]
                                    if achieved != target:
                                        status = "silent_miss"
                                rows.append(
                                    {
                                        "length": length,
                                        "fixed_fraction": fixed_fraction,
                                        "fixed_count": fixed_n,
                                        "omitted": omitted or "none",
                                        "symmetry_group_size": group_size,
                                        "direction": direction,
                                        "step": step,
                                        "target_charge": target,
                                        "achieved_charge": achieved,
                                        "status": status,
                                        "error": _failure_text(result) if status == "error" else "",
                                    }
                                )
                                if status != "exact":
                                    break
    _write_csv("validation_stress.csv", rows)
    counts = Counter(row["status"] for row in rows)
    print(f"wrote {len(rows)} stress rows: {dict(counts)}", flush=True)


def _panel_paths(args):
    if args.panel:
        return [os.path.abspath(path) for path in args.panel]
    if args.panel_dir:
        return sorted(
            os.path.join(os.path.abspath(args.panel_dir), name)
            for name in os.listdir(args.panel_dir)
            if name.lower().endswith(".pdb")
        )
    return DEFAULT_PANEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("rejection", "bias", "panel", "seeds", "stress", "all")
    )
    parser.add_argument("--model", default="protein_mpnn")
    parser.add_argument("--panel", action="append", help="PDB path; repeat for a panel")
    parser.add_argument("--panel-dir", default="")
    parser.add_argument("--pool-size", type=int, default=20000)
    parser.add_argument("--n-designs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-bin", type=int, default=200)
    parser.add_argument("--max-targets", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--temperatures",
        type=lambda value: tuple(float(x) for x in value.split(",")),
        default=TEMPERATURES,
    )
    parser.add_argument("--stress-source", default="")
    parser.add_argument(
        "--stress-lengths",
        type=lambda value: tuple(int(x) for x in value.split(",")),
        default=(15, 25, 40),
    )
    parser.add_argument(
        "--fixed-fractions",
        type=lambda value: tuple(float(x) for x in value.split(",")),
        default=(0.0, 0.25, 0.5, 0.75),
    )
    parser.add_argument(
        "--symmetry-groups",
        type=lambda value: tuple(int(x) for x in value.split(",")),
        default=(0, 2, 3, 4, 5, 6),
    )
    parser.add_argument("--stress-steps", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()
    panel_paths = _panel_paths(args)
    runner = Runner(args.model, args.batch_size)
    if args.phase in ("rejection", "all"):
        rejection(args, runner, panel_paths)
    if args.phase in ("bias", "all"):
        bias(args, runner, panel_paths)
    if args.phase in ("panel", "all"):
        panel(args, runner, panel_paths)
    if args.phase in ("seeds", "all"):
        seed_noise(args, runner, panel_paths)
    if args.phase in ("stress", "all"):
        stress(args, runner)


if __name__ == "__main__":
    main()

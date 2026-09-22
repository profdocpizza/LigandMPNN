"""Shared helpers for the constrained-decoding benchmarks.

Every benchmark shells out to ``run.py`` exactly as a user would, then parses
the FASTA headers.  Nothing is computed through a private code path, so the
numbers reported are the numbers the CLI produces.
"""

import math
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from constraints import (  # noqa: E402
    ALPHABET,
    extinction_280,
    net_charge,
    spps_risk,
)

PYTHON = os.environ.get("PYTHON", sys.executable)
STRUCTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "structures")

CHECKPOINTS = {
    "protein_mpnn": "proteinmpnn_v_48_020.pt",
    "soluble_mpnn": "solublempnn_v_48_020.pt",
    "ligand_mpnn": "ligandmpnn_v_32_010_25.pt",
}

AA20 = ALPHABET[:20]
_HEADER_FIELD = re.compile(r"([A-Za-z0-9_]+)=([^,]+)")


def native_sequence(pdb_path):
    """One-letter sequence of the input structure, from run.py's own parser."""
    designs = run_design(pdb_path, "protein_mpnn", batch_size=1, n_batches=1, seed=1)
    return designs["native"]


def run_design(
    pdb_path,
    model_type="protein_mpnn",
    batch_size=8,
    n_batches=1,
    seed=1,
    temperature=0.1,
    extra_args=(),
    out_folder=None,
    allow_failure=False,
):
    """Run one ``run.py`` invocation and return parsed designs.

    Returns ``{"native": str, "designs": [{sequence, nll, seq_rec, charge,
    e280, spps_risk}], "stderr": str}``.
    """
    tmp = out_folder or tempfile.mkdtemp(prefix="bench_")
    cmd = [
        PYTHON,
        "run.py",
        "--model_type",
        model_type,
        f"--checkpoint_{model_type}",
        os.path.join(REPO, "model_params", CHECKPOINTS[model_type]),
        "--pdb_path",
        pdb_path,
        "--out_folder",
        tmp,
        "--batch_size",
        str(batch_size),
        "--number_of_batches",
        str(n_batches),
        "--seed",
        str(seed),
        "--temperature",
        str(temperature),
        "--verbose",
        "0",
        "--report_properties",
        "1",
        *[str(a) for a in extra_args],
    ]
    env = dict(os.environ, PYTHONPATH=REPO)
    env.pop("PYTHONSAFEPATH", None)
    proc = subprocess.run(
        cmd, cwd=REPO, capture_output=True, text=True, env=env, timeout=7200
    )
    stem = os.path.splitext(os.path.basename(pdb_path))[0]
    fasta = os.path.join(tmp, "seqs", f"{stem}.fa")
    if not os.path.exists(fasta):
        if allow_failure:
            return {
                "native": None,
                "designs": [],
                "stderr": proc.stderr,
                "stdout": proc.stdout,
                "returncode": proc.returncode,
            }
        raise RuntimeError(
            f"run.py produced no FASTA (exit {proc.returncode}).\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )

    records = []
    with open(fasta) as fh:
        header = None
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:]
            else:
                records.append((header, line.replace(":", "").replace("/", "")))

    native = records[0][1]
    designs = []
    for header, seq in records[1:]:
        fields = dict(_HEADER_FIELD.findall(header))
        conf = float(fields["overall_confidence"])
        designs.append(
            {
                "sequence": seq,
                # run.py reports exp(-mean_NLL); invert for the NLL itself.
                "nll": -math.log(conf) if conf > 0 else float("nan"),
                "confidence": conf,
                "seq_rec": float(fields["seq_rec"]),
                "charge": float(fields["net_charge"]),
                "e280": float(fields["e280"]),
                "spps_risk": float(fields["spps_risk"]),
            }
        )
    return {
        "native": native,
        "designs": designs,
        "stderr": proc.stderr,
        "stdout": proc.stdout,
        "returncode": proc.returncode,
    }


def composition(seqs):
    """Fractional amino-acid composition over a list of sequences."""
    total = sum(len(s) for s in seqs)
    return {
        aa: sum(s.count(aa) for s in seqs) / total if total else 0.0 for aa in AA20
    }


def identity(a, b):
    """Fraction of positions at which two equal-length sequences agree."""
    if len(a) != len(b):
        raise ValueError("sequences differ in length")
    return sum(x == y for x, y in zip(a, b)) / len(a)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def jensen_shannon(p, q):
    """Jensen-Shannon divergence in bits between two composition dicts."""
    keys = sorted(set(p) | set(q))

    def kl(a, b):
        return sum(
            a[k] * math.log2(a[k] / b[k]) for k in keys if a.get(k, 0) > 0 and b.get(k, 0) > 0
        )

    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


__all__ = [
    "AA20",
    "CHECKPOINTS",
    "REPO",
    "STRUCTURES",
    "composition",
    "extinction_280",
    "identity",
    "jensen_shannon",
    "mean",
    "native_sequence",
    "net_charge",
    "run_design",
    "spps_risk",
]

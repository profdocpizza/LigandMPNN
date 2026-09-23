"""Shared helpers for the constrained-decoding benchmarks.

Every benchmark shells out to ``run.py`` exactly as a user would, then parses
the FASTA headers.  Nothing is computed through a private code path, so the
numbers reported are the numbers the CLI produces.
"""

import gc
import io
import math
import os
import re
import subprocess
from contextlib import redirect_stderr, redirect_stdout
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

# Driving run.main() in-process reuses the interpreter, the imported torch and
# the CUDA context, which on a GPU dominate the cost of a small design batch.
# It is the SAME code path as the CLI -- the arguments go through run.py's own
# parser and into its own main() -- so the numbers are still the CLI's numbers.
# Set BENCH_INPROC=0 to force the subprocess path; tests/test_inproc_matches.py
# checks the two agree bit for bit.
INPROC = os.environ.get("BENCH_INPROC", "1") not in ("0", "false", "False")
_RUN = None
_PARSER = None


#: Peak activation scales with batch_size * n_residues. 12000 keeps a 4080
#: comfortable; override with BENCH_MAX_BATCH_RESIDUES.
MAX_BATCH_RESIDUES = int(os.environ.get("BENCH_MAX_BATCH_RESIDUES", "12000"))
_LEN_CACHE = {}


def _n_residues(pdb_path):
    if pdb_path not in _LEN_CACHE:
        with open(pdb_path) as fh:
            _LEN_CACHE[pdb_path] = len(
                {l[21:27] for l in fh if l.startswith("ATOM") and l[12:16].strip() == "CA"}
            )
    return _LEN_CACHE[pdb_path] or 1


def _split_batches(pdb_path, batch_size, n_batches):
    """Shrink the batch and add batches so the design count is unchanged."""
    L = _n_residues(pdb_path)
    cap = max(1, MAX_BATCH_RESIDUES // L)
    if batch_size <= cap:
        return batch_size, n_batches
    total = batch_size * n_batches
    new_bs = cap
    new_nb = -(-total // new_bs)          # ceil, so never fewer designs
    return new_bs, new_nb


def _run_inproc(cli):
    """Call run.main() with CLI-parsed args; returns (stdout, stderr, rc)."""
    global _RUN, _PARSER
    if _RUN is None:
        cwd = os.getcwd()
        os.chdir(REPO)
        try:
            import run as _run_mod
        finally:
            os.chdir(cwd)
        _RUN = _run_mod
        _PARSER = _run_mod.build_parser()
    buf_out, buf_err = io.StringIO(), io.StringIO()
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            _RUN.main(_PARSER.parse_args(cli))
        rc = 0
    except SystemExit as exc:  # argparse on a bad flag
        rc = int(exc.code or 0)
    except Exception as exc:  # infeasible constraint, bad input
        buf_err.write(f"{type(exc).__name__}: {exc}")
        rc = 1
    finally:
        os.chdir(cwd)
        # run.main() builds a fresh model on the device every call. As a
        # subprocess, process exit reclaimed that; in-process it accumulates
        # until the GPU runs out, so drop it explicitly.
        _free_device_memory()
    return buf_out.getvalue(), buf_err.getvalue(), rc


def _free_device_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
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
    inproc=None,
):
    """Run one ``run.py`` invocation and return parsed designs.

    Returns ``{"native": str, "designs": [{sequence, nll, seq_rec, charge,
    e280, spps_risk}], "stderr": str}``.
    """
    tmp = out_folder or tempfile.mkdtemp(prefix="bench_")
    batch_size, n_batches = _split_batches(pdb_path, batch_size, n_batches)
    cli = [
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
    use_inproc = INPROC if inproc is None else inproc
    if use_inproc:
        out, err, rc = _run_inproc(cli)
    else:
        env = dict(os.environ, PYTHONPATH=REPO)
        env.pop("PYTHONSAFEPATH", None)
        proc = subprocess.run(
            [PYTHON, "run.py", *cli], cwd=REPO, capture_output=True, text=True,
            env=env, timeout=7200,
        )
        out, err, rc = proc.stdout, proc.stderr, proc.returncode
    stem = os.path.splitext(os.path.basename(pdb_path))[0]
    fasta = os.path.join(tmp, "seqs", f"{stem}.fa")
    if not os.path.exists(fasta):
        if allow_failure:
            return {
                "native": None,
                "designs": [],
                "stderr": err,
                "stdout": out,
                "returncode": rc,
            }
        raise RuntimeError(
            f"run.py produced no FASTA (exit {rc}).\n"
            f"{out[-2000:]}\n{err[-2000:]}"
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
        "stderr": err,
        "stdout": out,
        "returncode": rc,
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

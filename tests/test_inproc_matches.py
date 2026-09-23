"""The in-process benchmark driver must equal the command line, byte for byte.

benchmarks/bench_lib.py can either shell out to run.py or import it and call
main() directly.  The second is much faster -- one interpreter start and one
CUDA context for a whole benchmark instead of one per condition -- but it is
only legitimate if it is genuinely the same code path.  This test runs the
same configurations both ways at a fixed seed and compares the FASTA output.

    python tests/test_inproc_matches.py
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmarks"))

from bench_lib import STRUCTURES, run_design  # noqa: E402

CASES = [
    ("protein_mpnn", []),
    ("protein_mpnn", ["--target_charge", -8]),
    ("soluble_mpnn", ["--min_extinction_280", 5500]),
    ("protein_mpnn", ["--spps_bias", 1.0]),
    ("protein_mpnn", ["--target_charge", -4, "--charge_tolerance", 2]),
]
PDB = os.path.join(STRUCTURES, "panel", "1VII.pdb")


def _fasta(model, extra, inproc):
    out = tempfile.mkdtemp(prefix="xchk_")
    run_design(PDB, model, batch_size=4, n_batches=1, seed=31337,
               extra_args=extra, out_folder=out, inproc=inproc)
    with open(os.path.join(out, "seqs", "1VII.fa")) as fh:
        return fh.read()


def main():
    failures = 0
    for model, extra in CASES:
        label = f"{model} {' '.join(str(a) for a in extra) or '(no flags)'}"
        a = _fasta(model, extra, inproc=False)
        b = _fasta(model, extra, inproc=True)
        if a == b:
            print(f"IDENTICAL  {label}")
        else:
            failures += 1
            print(f"DIFFERS    {label}")
            for la, lb in zip(a.splitlines(), b.splitlines()):
                if la != lb:
                    print(f"    cli: {la[:90]}\n    inp: {lb[:90]}")
                    break
    print(f"\n{len(CASES) - failures}/{len(CASES)} identical")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

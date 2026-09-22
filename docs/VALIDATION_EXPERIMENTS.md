# No-Folding Validation Experiments

`benchmarks/bench_validation.py` adds the distributional checks that are
missing from the original feature benchmarks. It uses only sequence outputs and
the metrics already emitted by LigandMPNN: sequence recovery, net charge,
epsilon-280, SPPS score, composition, and sequence diversity. Raw per-residue
NLL remains in the CSVs as a model-diagnostic field, but is not used as a
quality claim or figure axis. It does not run a structure predictor.

## Why compare composition conditionally

The unconstrained model samples a distribution of charges. An exact target
keeps only one charge slice of that distribution, even when the target is near
the unconstrained mean. The mean charge is generally fractional, while an
exact request is an integer, and conditioning also removes the natural charge
spread. Comparing the constrained composition directly with all unconstrained
designs would therefore confound the requested property with the natural
charge distribution. This is not evidence that the sequence is less foldable;
it is why the primary plots report amino-acid frequency changes instead.

The rejection arm tests this directly. If constrained decoding is a good
approximation to `p(sequence | achieved charge = c)`, its composition and
pairwise diversity should agree with unconstrained designs rejected into the
same charge bin. The comparison is made separately for each backbone and
temperature, so no cross-protein charge bias is treated as universal.

## Experiments

The default checked-in panel contains four structures. For the paper panel,
pass 25--30 monomer PDBs explicitly with repeated `--panel` arguments or
`--panel-dir`; all arms then use the same model and temperature settings.

```bash
# Exact conditional-distribution check. Defaults: 20,000 pool, 200 designs.
python benchmarks/bench_validation.py rejection --pool-size 20000 --n-designs 200

# Bias flag head-to-head, with charge targets chosen relative to each backbone.
python benchmarks/bench_validation.py bias

# Per-backbone charge and A280 panel.
python benchmarks/bench_validation.py panel

# Ten-seed composition null and constrained-vs-null comparison.
python benchmarks/bench_validation.py seeds

# Short-chain/fixed/omitted/symmetry guarantee stress test.
python benchmarks/bench_validation.py stress
```

Use `--temperatures 0.1,0.2,0.3` to run the temperature panel. The stress
runner records `exact`, `error`, and `silent_miss` separately. A silent miss is
the result that must not be reported as a successful guarantee.

Outputs are written under `benchmarks/results/` with the prefix
`validation_`. The large rejection pool is intentionally configurable for
smoke tests, but final results should use the defaults or larger pools and
report the acceptance rate for every target.

Folding/self-consistency and wet-lab synthesis validation remain outside this
suite.

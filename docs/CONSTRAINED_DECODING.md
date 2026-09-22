# Constrained decoding: charge, A280 detectability, SPPS friendliness

Three deterministic constraints applied **while the model decodes**, so a design
can be *required* to have a given net charge, to carry a 280 nm chromophore, or
to avoid known solid-phase peptide synthesis (SPPS) failure motifs.

No weights are retrained and no checkpoint is modified. At each autoregressive
step MPNN emits 21 amino-acid logits; a deterministic layer re-weights and
prunes them before sampling. Two of the three are therefore not biases but
**guarantees**, checked on every sampled design.

---

## 1. What to use

| I want... | Flag | What you get | Typical composition change |
|---|---|---|---|
| A specific net charge | `--target_charge -8` | Exactly that charge, every design | D/E/K/R carry most of the shift |
| Roughly that charge | `--target_charge -8 --charge_tolerance 2` | Within ±2 e, every design | More freedom inside the window |
| To quantify protein by A280 | `--min_extinction_280 5500` | At least one Trp, every design | Mainly Trp/Tyr frequency |
| An easier peptide to synthesise | `--spps_bias 0.75` | Asp-Gly eliminated, fewer β-branched runs | Composition shifts with bias |

**If you only remember one thing:** `--min_extinction_280 5500` is nearly free
and stops you designing a protein you cannot measure. `--spps_bias 0.5`–`1.0` is
the useful range for anything you intend to have synthesised.

```bash
# Ubiquitin redesign that must carry exactly -8 net charge
python run.py --model_type soluble_mpnn \
    --checkpoint_soluble_mpnn ./model_params/solublempnn_v_48_020.pt \
    --pdb_path ./benchmarks/structures/1UBQ.pdb --out_folder ./outputs/charge \
    --batch_size 8 --target_charge -8

# A peptide guaranteed to have at least one Trp, so A280 can quantify it
python run.py --model_type soluble_mpnn \
    --checkpoint_soluble_mpnn ./model_params/solublempnn_v_48_020.pt \
    --pdb_path ./benchmarks/structures/1VII.pdb --out_folder ./outputs/uv \
    --batch_size 8 --min_extinction_280 5500

# All three at once
python run.py --model_type ligand_mpnn \
    --checkpoint_ligand_mpnn ./model_params/ligandmpnn_v_32_010_25.pt \
    --pdb_path ./inputs/1BC8.pdb --out_folder ./outputs/combined \
    --batch_size 8 --target_charge 4 --min_extinction_280 5500 --spps_bias 1.0
```

With any constraint active — or with `--report_properties 1` — the measured
values are written into the FASTA header, so baselines are directly comparable:

```
>1UBQ, id=1, T=0.1, seed=5, overall_confidence=0.2884, ligand_confidence=0.2884,
 seq_rec=0.4342, net_charge=-8, e280=11000, spps_risk=12.40
```

---

## 2. How to read the numbers

The plots report sequence-level changes. None of them measure whether a design
folds — see [what is not claimed](#7-what-is-not-claimed).

**Amino-acid frequency change** is the percentage-point difference between a
constrained arm and the unconstrained arm on the same backbone. Highlighted
residues show where the requested property is being satisfied; the remaining
residues show collateral composition changes. This is a descriptive sequence
metric, not a folding or activity predictor.

**Sequence recovery** — the fraction of redesigned positions matching the
native residue. Higher usually tracks "more natural", but a low value is not
automatically bad; MPNN often improves on natural sequences.

**SPPS risk score** — a dimensionless ranking heuristic summing documented
Fmoc-SPPS failure modes, described in [section 6](#6-exact-definitions). It is
**not** a predicted yield. Only differences between arms on the *same* target
are meaningful; do not compare absolute values between proteins of different
length.

---

## 3. Net charge: exact, on every design

![charge benchmark](../benchmarks/figures/charge_benchmark.png)

Ubiquitin (1UBQ, 76 aa, whole chain redesigned), targets swept from −30 to +30,
three model types, 32 designs each.

**Panel a — it works.** All **1248 / 1248** designs landed exactly on the
requested charge. Left to itself the model produces about −5 e on this backbone
(native ubiquitin is 0 e), so most of this range is far from what it would
choose on its own.

The figure overlays the exact arm as solid model-colored curves and the ±4 e
arm as lower-opacity dashed curves. The exact curves lie on the diagonal;
the tolerance curves spread around it while remaining inside their requested
windows.

---

## 4. Extinction coefficient: a chromophore you can actually measure

![extinction benchmark](../benchmarks/figures/extinction_benchmark.png)

Four backbones (1L2Y, 1PGA, 1UBQ and 1VII), three model types, and 50 designs
per arm are used for the final-value swarm. The composition comparison uses
1VII with SolubleMPNN and 50 designs per arm.

Native HP36 has exactly one Trp. In the 1VII SolubleMPNN comparison, the
unconstrained arm is often low in ε₂₈₀, illustrating why a design set without a
chromophore cannot be quantified by A280, the standard concentration check at
the bench.

**Panel a — final values.** The swarm plot shows the final ε₂₈₀ from 50 designs
per threshold, across four backbones and three models. The requested floor is
visible as a lower bound, while the unconstrained distribution differs by
backbone and model.

**Panel b — composition on 1VII.** The 50-design comparison at an 8000 floor
shows the Trp/Tyr change against the unconstrained arm and the movement of the
other residues. This supports a sequence-level statement only; it is not
structural validation.

---

## 5. SPPS risk: steering, with no guarantee

![SPPS benchmark](../benchmarks/figures/spps_benchmark.png)

Two targets, 64 designs per arm. **This constraint offers no guarantee**, and
the benchmark is reported accordingly — distributions and motif counts rather
than a satisfaction rate.

Why no guarantee: SPPS risk depends on sequence *neighbours*, and MPNN decodes
in a random order, so when a position is decided some of its neighbours do not
exist yet. The penalty therefore acts greedily on partial context.

Two targets are shown because they exercise different rules, and reporting only
the first would overstate the case:

- **1VII (36 aa)** — comfortably in routine Fmoc-SPPS range. Its designs contain
  **no Asp-Gly at all**, so here the score is driven almost entirely by
  β-branched load, which falls from 4.8 to 0.0 as bias goes 0 → 2. The headline
  aspartimide rule simply has nothing to do on this target.
- **1PGA (GB1, 56 aa, T = 0.3)** — the target where aspartimide motifs actually
  appear, and the only one that can say anything about those rules.

**Panel a — what it removes.** On 1PGA, Asp-Gly is present in 14.1 % of
unconstrained designs, 7.8 % at bias 0.5, and **0 % from bias 1.0 upward**;
adjacent Ile/Val/Thr pairs fall from 5.9 to 0.1 per design at bias 1.0.

**Panel b — the aggregate effect.** The total risk distribution falls sharply
by bias 1.0. GB1's *native* sequence scores 54.2 on this risk scale —
essentially the same as the unconstrained designs — because its β-sheet core
genuinely is β-branched-rich. Driving the risk to zero means removing the
residues that build that sheet. `--spps_bias` is a dial on a real trade-off, not
a free improvement. **0.5–1.0** removes the classic motifs while avoiding the
largest composition shifts.

---

## 6. Exact definitions

Read these before quoting a number — each is a specific convention, not a
measurement.

**Net charge** counts Asp and Glu as −1, Lys and Arg as +1. His is treated as
**neutral**: its side-chain pKa near 6.0 leaves it roughly 10 % protonated at
pH 7.4, and a fractional value would forfeit the exact-integer guarantee. Free
α-amino and α-carboxyl termini are **not** counted; for a single chain they
cancel. This is a design-time convention, not a predicted titration curve — if
you need pI or pH-dependent charge, compute it from the output sequence with a
proper tool.

**ε₂₈₀** is `5500·n_Trp + 1490·n_Tyr` M⁻¹cm⁻¹, from [Pace et al., Protein Sci
4:2411 (1995)](https://doi.org/10.1002/pro.5560041120), for a protein with all
cysteines **reduced**. Each cystine disulfide adds ≈125 M⁻¹cm⁻¹; disulfide
connectivity is not a function of sequence alone, so it is excluded and the
value is a lower bound for an oxidised protein. The flag guarantees an
extinction coefficient, **not an absorbance** — absorbance additionally depends
on concentration and path length. Useful values: `1490` ≥ one Tyr, `5500` ≥ one
Trp, `11000` ≥ two Trp equivalents.

**SPPS risk** sums documented Fmoc-SPPS failure modes: Asp-Gly and weaker
aspartimide-prone Asp-X motifs, β-branched Ile/Val/Thr adjacency and runs,
sliding-window aliphatic load, and Cys and Met counts. **The weights are
hyperparameters chosen to rank these modes in a sensible order; they are not
measured effect sizes from the literature.** They live in one dict
(`SPPS_DEFAULT_WEIGHTS` in `constraints.py`) so you can retune them for your
chemistry. Two rules were deliberately left out: penalising Arg in favour of
Lys, because Arg(Pbf) has been reported to *reduce* resin-bound aggregation
where Lys(Boc) increases it; and rewarding Pro, because poly-Pro did not
reliably suppress on-resin aggregation.

**Scope** (`--constraint_scope`) decides which residues a property is computed
over. The default `designed_chains` uses every residue of any chain containing a
designable position, so fixed native residues count toward the target — which is
what "this protein should have net charge −4" normally means.
`designed_positions` uses only redesigned positions. If a target is unreachable
given the fixed residues, the run fails immediately with the achievable range
rather than silently missing:

```
constraints.InfeasibleConstraint: net_charge: target -3 e is not reachable.
Given the fixed residues and the allowed alphabet, the achievable range is [-2, 2] e.
```

---

## 7. What is not claimed

- **No structural validation.** Nothing here was folded, and no design was
  expressed, synthesised or measured. Amino-acid frequency changes and
  sequence recovery describe sequence behavior, not whether a design folds. A
  self-consistency check (fold the designs, measure RMSD to the input
  backbone) is the obvious next step and has not been done.
- **The SPPS term is untested against real synthesis.** It encodes literature
  failure modes; it does not predict whether a peptide will couple.
- **Joint hard constraints can compete.** Reachability is computed per
  constraint, not jointly, so an exact charge target combined with a high
  extinction floor on a very short peptide could in principle over-restrict a
  position. `run.py` verifies every sampled design against every hard constraint
  and warns if any missed; no violation occurred in any benchmark run.
- **Discrete reachability matters for equality constraints.** Integer-valued
  equality targets, including tied symmetry groups and omitted residues, are
  checked against their discrete reachable set. If a target is impossible, the
  run raises an `InfeasibleConstraint` rather than silently returning a miss.

---

## 8. How it works

Charge and extinction coefficient are both **linear functionals of
composition**: `f(seq) = Σ v[seq_i]`. That shared structure means one controller
handles both, in two stages per decoding step.

1. **Soft steering.** A multiplier `λ` is found by bisection so the expected
   value contributed by this step matches the per-position requirement implied
   by what is still outstanding. This spreads the requirement over the whole
   chain instead of dumping it on whichever positions decode last. `E_λ[v]` is
   strictly increasing in `λ`, so the bisection is unconditionally safe. With a
   tolerance set, this becomes a deadband: inside the window, `λ = 0`.
2. **Hard reachability pruning.** Any amino acid whose selection would make the
   target unreachable by the positions not yet decoded is masked out. Integer
   equality constraints use an exact suffix reachability set, accounting for
   tied-group multiplicities, fixed residues and per-position omissions. Other
   linear constraints use the interval relaxation.

Stage 2 is what turns "approximately the requested charge" into "exactly the
requested charge". Stage 1 is what keeps the sequence sensible while doing so.

SPPS risk is not a linear functional, so it gets neither stage — just a greedy
penalty on already-decoded context.

**Naming.** The accurate term is **constrained decoding**, the protein-design
analogue of constrained text generation — "NeuroLogic Decoding: (Un)supervised
Neural Text Generation with Predicate Logic Constraints" and its successor
["NeuroLogic A*esque Decoding: Constrained Text Generation with Lookahead
Heuristics"](https://aclanthology.org/2022.naacl-main.57/) (Lu et al., NAACL
2022), where lookahead estimates of future constraint satisfaction guide an
autoregressive decoder. Applying constrained decoding to ProteinMPNN is not new
either: ["A novel decoding strategy for ProteinMPNN to design with less
visibility to cytotoxic
T-lymphocytes"](https://doi.org/10.1016/j.csbj.2025.07.055) (Computational and
Structural Biotechnology Journal, 2025) steers designs away from peptides
predicted to be presented by MHC class I. What is added here is the constraint
set and the exactness.

---

## 9. Non-destructive by construction

The hook is gated on `feature_dict.get("constraints")`, so with no new flags the
original arithmetic runs untouched. This is verified, not asserted:

```bash
bash tests/test_identical_to_upstream.sh
```

runs every model type at a fixed seed from both the base revision and the
working tree and compares FASTA output byte for byte — **10/10 identical**
(5 model types × 2 inputs).

`sample()` is one method shared by every model type, so the constraints work
with `protein_mpnn`, `soluble_mpnn`, `ligand_mpnn` and both membrane variants
without per-model code. They compose with `--fixed_residues`,
`--redesigned_residues`, `--omit_AA`, `--bias_AA`, `--symmetry_residues` and the
per-residue JSON options. `log_probs` is still computed from the raw logits, so
`overall_confidence` stays comparable to unconstrained runs.

Unit tests: `python tests/test_constraints.py` (15 tests).

---

## 10. All flags

| Flag | Default | Meaning |
|---|---|---|
| `--target_charge` | off | Net charge the design must have, e.g. `-4` |
| `--charge_tolerance` | `0` | Allowed deviation; 0 means exact |
| `--min_extinction_280` | off | Minimum ε₂₈₀ in M⁻¹cm⁻¹ |
| `--spps_bias` | `0` | SPPS penalty strength; 0.5–1.0 recommended |
| `--constraint_scope` | `designed_chains` | Or `designed_positions` |
| `--constraint_lambda_max` | `8.0` | Cap on the steering multiplier, in logit units |
| `--report_properties` | `0` | `1` adds measured properties to FASTA headers even with no constraint active |

---

## 11. Reproducing the benchmarks

```bash
python benchmarks/bench_charge.py       # charge sweep plus 0/+-4 tolerance arms
python benchmarks/bench_extinction.py   # ~3 min
python benchmarks/bench_spps.py         # ~8 min
python benchmarks/make_figures.py       # needs pandas + matplotlib only
```

Each benchmark shells out to `run.py` exactly as a user would and parses the
FASTA headers, so the numbers are the numbers the CLI produces. Raw per-design
results are in `benchmarks/results/*.csv`.

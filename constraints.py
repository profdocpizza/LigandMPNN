"""Deterministic sequence constraints applied at MPNN decoding time.

This module implements *constrained decoding* for the MPNN family: the network
still produces the per-position amino-acid logits, and a deterministic layer
re-weights and prunes them before sampling.  Nothing is retrained and no model
weights are touched.

Two kinds of constraint are provided.

``LinearFunctional``
    For any property that is a weighted count of amino acids -- net charge,
    molar extinction coefficient at 280 nm, residue counts -- the property is a
    linear functional of composition, ``f(seq) = sum_i v[seq_i]``.  Such a
    constraint is enforced in two stages at every decoding step:

    1.  *Soft steering.*  A multiplier ``lam`` is found by bisection so that the
        expected value contributed by the current step matches the per-position
        requirement implied by what is still outstanding.  This spreads the
        requirement over the whole chain instead of dumping it on whichever
        positions happen to decode last.
    2.  *Hard reachability pruning.*  Any amino acid whose selection would make
        the target unreachable by the positions that have not decoded yet is
        masked out.  The reachable interval is precomputed as a suffix sum over
        each batch row's own decoding order, accounting for fixed residues and
        for per-position omissions.

    Stage 2 is what turns "approximately the requested charge" into "exactly the
    requested charge".  See ``LinearFunctional.exactness_caveats`` for the one
    condition under which the interval relaxation is not tight.

``SPPSTerm``
    Solid-phase peptide synthesis risk is *not* a linear functional -- it
    depends on sequence neighbours (Asp-Gly, beta-branched runs, hydrophobic
    windows).  Because MPNN decodes in a random order, some neighbours of the
    position being decided have not been decided yet.  This term therefore
    penalises only the risk that is visible from already-decoded neighbours.
    It steers, it does not guarantee.

Method context: the soft-steering-plus-lookahead-pruning combination is the
protein-design analogue of constrained text decoding -- see "NeuroLogic
A*esque Decoding: Constrained Text Generation with Lookahead Heuristics" (Lu
et al., NAACL 2022, aclanthology.org/2022.naacl-main.57) and the earlier
"NeuroLogic Decoding: (Un)supervised Neural Text Generation with Predicate
Logic Constraints", where future-constraint-satisfaction estimates guide an
autoregressive decoder.  Constrained decoding has been applied to ProteinMPNN
before, for MHC class I immune visibility: "A novel decoding strategy for
ProteinMPNN to design with less visibility to cytotoxic T-lymphocytes"
(Comput Struct Biotechnol J 2025, doi:10.1016/j.csbj.2025.07.055).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# Must match ``data_utils.restype_str_to_int``.
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_INT = {aa: i for i, aa in enumerate(ALPHABET)}
N_TOKENS = len(ALPHABET)  # 21; index 20 ("X") is never sampled by MPNN
N_AA = 20

_OMIT_THRESHOLD = -1e7  # run.py encodes "omit this amino acid" as a -1e8 bias
_MASK_VALUE = -1e8  # same convention, so masked logits behave like omissions


class InfeasibleConstraint(ValueError):
    """Raised when a target cannot be reached from the given inputs."""


def _vector(weights: Dict[str, float]) -> torch.Tensor:
    """Build a length-21 value vector from a {one-letter code: value} mapping."""
    v = torch.zeros(N_TOKENS, dtype=torch.float32)
    for aa, value in weights.items():
        if aa not in AA_TO_INT:
            raise ValueError(f"unknown amino acid code {aa!r}")
        v[AA_TO_INT[aa]] = float(value)
    return v


# --------------------------------------------------------------------------- #
# Property definitions and standalone scorers
# --------------------------------------------------------------------------- #

#: Net charge at neutral pH.  Asp/Glu carry -1, Lys/Arg carry +1.  His is
#: treated as neutral: its side chain pKa near 6.0 leaves it ~10% protonated at
#: pH 7.4, and a fractional value would forfeit the exact-integer guarantee.
#: Free alpha-amino and alpha-carboxyl termini are not counted; for a single
#: chain they cancel.
CHARGE_WEIGHTS: Dict[str, float] = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0}

#: Molar extinction coefficient at 280 nm in M^-1 cm^-1, from Pace et al.,
#: Protein Sci 4:2411 (1995), for a protein with *all cysteines reduced*.  A
#: cystine disulfide adds ~125 M^-1 cm^-1 each; disulfide connectivity is not a
#: function of sequence alone, so it is excluded and the value returned is a
#: lower bound for an oxidised protein.
EXTINCTION_280_WEIGHTS: Dict[str, float] = {"W": 5500.0, "Y": 1490.0}

#: Default weights for the SPPS risk score.  These are *hyperparameters chosen
#: to rank the well-documented failure modes in a sensible order*, not measured
#: effect sizes from the literature.  Tune them for your chemistry.
SPPS_DEFAULT_WEIGHTS: Dict[str, float] = {
    "beta_branched": 1.0,  # per Ile/Val/Thr
    "beta_pair": 3.0,  # per adjacent beta-branched pair
    "beta_run3": 6.0,  # per position extending a beta-branched run to >=3
    "asp_gly": 8.0,  # per Asp-Gly (canonical aspartimide motif)
    "asp_other": 3.0,  # per other aspartimide-prone Asp-X
    "aliphatic_window": 2.0,  # per excess hydrophobic residue in a window
    "cys": 0.5,  # per Cys
    "met": 0.2,  # per Met
}

BETA_BRANCHED = "IVT"
#: Asp-X motifs prone to aspartimide formation under repeated Fmoc removal,
#: with Asp-Gly by far the worst.  Set per Mergler et al. and Subiros-Funosas
#: et al.; the ordering is well established, the magnitudes are not.
ASP_OTHER_NEXT = "DNSTRCQE"
#: Residues counted toward on-resin hydrophobic/aliphatic load.
ALIPHATIC = "AVILMF"
SPPS_WINDOW = 7
SPPS_WINDOW_MAX_HYDROPHOBIC = 5


def net_charge(seq: str) -> float:
    """Net charge of ``seq`` under :data:`CHARGE_WEIGHTS`."""
    return float(sum(CHARGE_WEIGHTS.get(aa, 0.0) for aa in seq))


def extinction_280(seq: str) -> float:
    """Molar extinction coefficient at 280 nm (M^-1 cm^-1), Cys fully reduced."""
    return float(sum(EXTINCTION_280_WEIGHTS.get(aa, 0.0) for aa in seq))


def spps_risk(
    seq: str, weights: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, float]]:
    """Fmoc-SPPS synthesis-risk score for ``seq``.

    Returns ``(total, components)``.  Higher is worse.  The score is a
    dimensionless ranking heuristic over documented failure modes, not a
    predicted synthesis yield.
    """
    w = dict(SPPS_DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    n = len(seq)
    comp = {
        "beta_branched": 0.0,
        "beta_pair": 0.0,
        "beta_run3": 0.0,
        "asp_gly": 0.0,
        "asp_other": 0.0,
        "aliphatic_window": 0.0,
        "cys": 0.0,
        "met": 0.0,
    }
    for i, aa in enumerate(seq):
        if aa in BETA_BRANCHED:
            comp["beta_branched"] += 1.0
            if i + 1 < n and seq[i + 1] in BETA_BRANCHED:
                comp["beta_pair"] += 1.0
            if (
                i >= 2
                and seq[i - 1] in BETA_BRANCHED
                and seq[i - 2] in BETA_BRANCHED
            ):
                comp["beta_run3"] += 1.0
        if aa == "D" and i + 1 < n:
            nxt = seq[i + 1]
            if nxt == "G":
                comp["asp_gly"] += 1.0
            elif nxt in ASP_OTHER_NEXT:
                comp["asp_other"] += 1.0
        if aa == "C":
            comp["cys"] += 1.0
        if aa == "M":
            comp["met"] += 1.0
    for start in range(0, max(1, n - SPPS_WINDOW + 1)):
        window = seq[start : start + SPPS_WINDOW]
        hydrophobic = sum(1 for aa in window if aa in ALIPHATIC)
        comp["aliphatic_window"] += max(
            0.0, hydrophobic - SPPS_WINDOW_MAX_HYDROPHOBIC
        )
    total = sum(w[k] * v for k, v in comp.items())
    return float(total), comp


# --------------------------------------------------------------------------- #
# Step schedule: a uniform view of both decoding branches
# --------------------------------------------------------------------------- #


class StepSchedule:
    """Which positions each batch row decides at each decoding step.

    ``positions`` is ``[B, K, G]`` padded with -1, where ``K`` is the number of
    decoding steps and ``G`` the largest tied group (1 when no symmetry is
    used).  This one structure covers both branches of
    ``ProteinMPNN.sample()``: the plain branch gives every batch row its own
    order of singleton groups, the symmetric branch gives all rows a shared
    order of tied groups that are assigned a single residue together.
    """

    def __init__(self, positions: torch.Tensor):
        if positions.dim() != 3:
            raise ValueError("positions must be [B, K, G]")
        self.positions = positions
        self.B, self.K, self.G = positions.shape

    @classmethod
    def from_decoding_order(cls, decoding_order: torch.Tensor) -> "StepSchedule":
        """Plain branch: ``decoding_order`` is ``[B, L]``, one position per step."""
        return cls(decoding_order.unsqueeze(-1))

    @classmethod
    def from_groups(
        cls, groups: Sequence[Sequence[int]], batch_size: int, device
    ) -> "StepSchedule":
        """Symmetric branch: a shared list of tied groups, one residue each."""
        g_max = max(len(g) for g in groups) if groups else 1
        pos = torch.full((1, len(groups), g_max), -1, dtype=torch.long, device=device)
        for k, group in enumerate(groups):
            for j, p in enumerate(group):
                pos[0, k, j] = int(p)
        return cls(pos.repeat(batch_size, 1, 1))


# --------------------------------------------------------------------------- #
# Linear-functional constraints (charge, extinction coefficient, ...)
# --------------------------------------------------------------------------- #


class LinearFunctional:
    """A constraint on ``sum_i v[seq_i]`` over the positions in scope.

    Parameters
    ----------
    name
        Short identifier used in messages and output headers.
    weights
        ``{one-letter code: value}``; all other residues contribute zero.
    target
        Required value of the functional.
    mode
        ``"eq"`` for an equality target (charge), ``"ge"`` for a lower bound
        (extinction coefficient), ``"le"`` for an upper bound.
    tolerance
        Half-width of the accepted window in ``"eq"`` mode.  Zero means exact.
    lambda_max
        Cap on the soft-steering multiplier, in logit units per unit of the
        normalised value vector.  The hard mask, not this cap, is what enforces
        the target; the cap only limits how hard the distribution is pushed
        before the mask has to intervene.
    unit
        Human-readable unit, for messages only.
    """

    def __init__(
        self,
        name: str,
        weights: Dict[str, float],
        target: float,
        mode: str = "eq",
        tolerance: float = 0.0,
        lambda_max: float = 8.0,
        unit: str = "",
    ):
        if mode not in ("eq", "ge", "le"):
            raise ValueError(f"mode must be eq, ge or le, got {mode!r}")
        self.name = name
        self.values = _vector(weights)
        self.target = float(target)
        self.mode = mode
        self.tolerance = float(tolerance)
        self.lambda_max = float(lambda_max)
        self.unit = unit
        # Normalise so lambda_max means the same thing whether values are +-1
        # (charge) or thousands (extinction coefficient).
        scale = float(self.values.abs().max())
        self.scale = scale if scale > 0 else 1.0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        rel = {"eq": "==", "ge": ">=", "le": "<="}[self.mode]
        tol = f" +-{self.tolerance:g}" if self.mode == "eq" and self.tolerance else ""
        return f"<{self.name} {rel} {self.target:g}{tol} {self.unit}>"


class _LinearState:
    """Per-``sample()``-call state for one :class:`LinearFunctional`."""

    def __init__(
        self,
        cfg: LinearFunctional,
        schedule: StepSchedule,
        S_true: torch.Tensor,
        chain_mask: torch.Tensor,
        bias: torch.Tensor,
        scope: torch.Tensor,
        temperature: float,
    ):
        self.cfg = cfg
        self.schedule = schedule
        self.temperature = max(float(temperature), 1e-6)
        device = S_true.device
        B, K, G = schedule.B, schedule.K, schedule.G
        self.v = cfg.values.to(device)
        self.v_norm = self.v / cfg.scale

        pos = schedule.positions  # [B,K,G]
        valid = pos >= 0
        pos_c = pos.clamp(min=0)

        # A position counts toward the functional only if it is in scope.
        in_scope = scope[pos_c] & valid  # [B,K,G]
        designable = (
            torch.gather(chain_mask, 1, pos_c.reshape(B, -1)).reshape(B, K, G) > 0
        )

        # Multiplicity: how many in-scope positions this step's single residue
        # choice will occupy (>1 only for tied groups spanning the scope).
        self.mult = (in_scope).float().sum(-1)  # [B,K]

        # Allowed alphabet at each step: intersection over the tied positions of
        # the amino acids not omitted by --omit_AA / --omit_AA_per_residue.
        flat_bias = torch.gather(
            bias, 1, pos_c.reshape(B, -1, 1).expand(-1, -1, N_TOKENS)
        ).reshape(B, K, G, N_TOKENS)
        allowed = flat_bias[..., :N_AA] > _OMIT_THRESHOLD  # [B,K,G,20]
        allowed = allowed | (~valid).unsqueeze(-1)  # padding never restricts
        self.allowed = allowed.all(dim=2)  # [B,K,20]

        # Per-step reachable contribution.
        step_designable = (designable & valid).any(-1)  # [B,K]
        native = torch.gather(S_true, 1, pos_c.reshape(B, -1)).reshape(B, K, G)
        native_contrib = (self.v[native] * in_scope.float()).sum(-1)  # [B,K]

        big = torch.full_like(self.v[:N_AA], float("inf"))
        vmin_allowed = torch.where(
            self.allowed, self.v[:N_AA].expand(B, K, N_AA), big
        ).min(-1).values
        vmax_allowed = torch.where(
            self.allowed, self.v[:N_AA].expand(B, K, N_AA), -big
        ).max(-1).values

        dmin = torch.where(step_designable, self.mult * vmin_allowed, native_contrib)
        dmax = torch.where(step_designable, self.mult * vmax_allowed, native_contrib)
        dmin = torch.where(self.mult > 0, dmin, torch.zeros_like(dmin))
        dmax = torch.where(self.mult > 0, dmax, torch.zeros_like(dmax))

        # Fixed residues still count toward the target; their contribution is
        # known up front and is excluded from what the designable steps must
        # supply.
        self.fixed_contrib = torch.where(
            step_designable, torch.zeros_like(native_contrib), native_contrib
        )
        self.n_designable = torch.where(
            step_designable, self.mult, torch.zeros_like(self.mult)
        )
        self.step_designable = step_designable

        # Suffix sums, with a trailing zero so step K-1 can look at index K.
        def suffix(x: torch.Tensor) -> torch.Tensor:
            z = torch.zeros(B, 1, device=device, dtype=x.dtype)
            return torch.cat([x.flip(-1).cumsum(-1).flip(-1), z], dim=-1)

        self.suf_min = suffix(dmin)  # [B,K+1]
        self.suf_max = suffix(dmax)
        self.suf_fixed = suffix(self.fixed_contrib)
        self.suf_ndes = suffix(self.n_designable)
        # Variance of a tied group of g positions scales as g^2, not g, so the
        # lookahead needs the sum of squared multiplicities as well.
        self.suf_ndes2 = suffix(self.n_designable**2)

        # Running estimate of the model's own per-position value moments, used
        # to predict what the undecoded positions can still supply. Updated
        # from the unbiased marginals at each step, so it costs nothing extra.
        self._m_sum = torch.zeros(B, device=device)
        self._var_sum = torch.zeros(B, device=device)
        self._n_obs = torch.zeros(B, device=device)

        self.accum = torch.zeros(B, device=device)
        self.feasible_at_start = self._window_overlaps(
            self.suf_min[:, 0], self.suf_max[:, 0]
        )
        self._init_exact_reachability()

    def _init_exact_reachability(self) -> None:
        """Build exact integer reachability for equality constraints.

        The interval relaxation is sufficient for ordinary singleton steps,
        but tied groups can restrict a charge to a lattice (for example, all
        groups of three can only change charge in multiples of three).  Keep a
        small suffix dynamic-programming table for integer-valued functionals
        so the hard mask cannot silently choose a value outside that lattice.
        """
        self.exact_reachability = None
        self.exact_targets = None
        if self.cfg.mode != "eq":
            return
        values = [float(v) for v in self.v[:N_AA].detach().cpu().tolist()]
        if not all(abs(v - round(v)) < 1e-6 for v in values):
            return
        if abs(self.cfg.target - round(self.cfg.target)) >= 1e-6:
            return
        target_lo = math.ceil(self.cfg.target - self.cfg.tolerance - 1e-6)
        target_hi = math.floor(self.cfg.target + self.cfg.tolerance + 1e-6)
        self.exact_targets = (target_lo, target_hi)
        values = [int(round(v)) for v in values]
        suffixes = []
        for b in range(self.schedule.B):
            row = [set() for _ in range(self.schedule.K + 1)]
            row[-1].add(0)
            for k in range(self.schedule.K - 1, -1, -1):
                if bool(self.step_designable[b, k]):
                    mult = int(round(float(self.mult[b, k])))
                    choices = [
                        mult * values[a]
                        for a in range(N_AA)
                        if bool(self.allowed[b, k, a])
                    ]
                else:
                    native = int(round(float(self.fixed_contrib[b, k])))
                    choices = [native]
                row[k] = {choice + rest for choice in choices for rest in row[k + 1]}
            suffixes.append(row)
        self.exact_reachability = suffixes

    def exact_feasible_at_start(self) -> bool:
        if self.exact_reachability is None:
            return True
        lo, hi = self.exact_targets
        return all(
            any(lo <= value <= hi for value in suffixes[0])
            for suffixes in self.exact_reachability
        )

    def _exact_candidate_ok(self, b: int, k: int, aa: int) -> bool:
        if self.exact_reachability is None:
            return True
        lo, hi = self.exact_targets
        current = int(round(float(self.accum[b])))
        mult = int(round(float(self.mult[b, k])))
        contribution = mult * int(round(float(self.v[aa])))
        return any(
            lo <= current + contribution + rest <= hi
            for rest in self.exact_reachability[b][k + 1]
        )

    # -- feasibility -------------------------------------------------------- #

    def _window_overlaps(self, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """Does the reachable interval ``[lo, hi]`` admit the target window?"""
        cfg = self.cfg
        if cfg.mode == "eq":
            return (lo <= cfg.target + cfg.tolerance) & (hi >= cfg.target - cfg.tolerance)
        if cfg.mode == "ge":
            return hi >= cfg.target
        return lo <= cfg.target

    def exactness_caveats(self) -> List[str]:
        """Conditions under which the interval fallback is not tight.

        Integer-valued equality constraints use an exact suffix set above. The
        warning remains for non-integer equality functionals, where the hard
        mask still uses a contiguous interval and omitting every zero-valued
        residue can make that relaxation non-tight.
        """
        if self.cfg.mode != "eq" or self.exact_reachability is not None:
            return []
        zero_ok = (self.v[:N_AA].abs() < 1e-12).unsqueeze(0).unsqueeze(0) & self.allowed
        bad = self.step_designable & (self.mult > 0) & (~zero_ok.any(-1))
        n_bad = int(bad.sum())
        if n_bad:
            return [
                f"{self.cfg.name}: {n_bad} designable position(s) have no "
                f"zero-valued residue left in their allowed alphabet, so the "
                f"exact target may be unreachable."
            ]
        return []

    # -- decoding-time interface ------------------------------------------- #

    def logit_delta(self, k: int, base: torch.Tensor) -> torch.Tensor:
        """Additive logit term for step ``k``.

        ``base`` is ``[B, 21]``, the logits plus any bias already applied.  The
        return value is added to the bias and the sum is then divided by the
        temperature, so a log-probability correction must be multiplied by the
        temperature here to act as a proper likelihood reweighting.
        """
        B = base.shape[0]
        out = torch.zeros_like(base)
        active = self.step_designable[:, k] & (self.mult[:, k] > 0)
        if not bool(active.any()):
            return out

        mult = self.mult[:, k]
        out = out + self._lookahead(k, base, mult, active)

        # Hard reachability mask, applied last so nothing can override it.
        contrib = mult.unsqueeze(-1) * self.v.unsqueeze(0)  # [B,21]
        lo = (self.accum + self.suf_min[:, k + 1]).unsqueeze(-1) + contrib
        hi = (self.accum + self.suf_max[:, k + 1]).unsqueeze(-1) + contrib
        ok = self._window_overlaps(lo, hi)
        if self.exact_reachability is not None:
            for b in range(B):
                if bool(active[b]):
                    for aa in range(N_AA):
                        ok[b, aa] = ok[b, aa] and self._exact_candidate_ok(b, k, aa)
        ok = ok & torch.cat(
            [self.allowed[:, k], torch.zeros(B, 1, dtype=torch.bool, device=base.device)],
            dim=-1,
        )
        # If pruning would leave nothing, leave the step alone rather than
        # emitting a degenerate distribution.
        keep = ok.any(-1) & active
        mask = torch.where(ok, torch.zeros_like(out), torch.full_like(out, _MASK_VALUE))
        out = out + torch.where(keep.unsqueeze(-1), mask, torch.zeros_like(mask))
        return torch.where(active.unsqueeze(-1), out, torch.zeros_like(out))

    def _lookahead(
        self,
        k: int,
        base: torch.Tensor,
        mult: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Reweight this step's logits by the chance of still hitting the target.

        The exact conditional distribution is
        ``p(s_t | s_<t, target) = p(s_t | s_<t) * P(target | s_<=t) / Z``.  The
        second factor is the probability that the positions *after* this one
        can still make up whatever the target demands.  Multiplying by it
        constrains only the endpoint, which is what was actually asked for.

        The earlier controller instead solved for a multiplier that forced
        each step's *expected* contribution to match the outstanding
        per-position requirement.  That pins the whole trajectory rather than
        the endpoint, and measured against rejection sampling it cost roughly
        0.5 nats per residue on ubiquitin even for targets the model would
        have hit on its own.

        ``P`` is estimated by a normal approximation to the sum over the
        remaining positions, whose per-position mean and variance are read off
        the model's own unbiased marginals as decoding proceeds.  The sum of
        many near-independent bounded terms is close to normal, and only
        differences across the 20 candidates matter, so the approximation is
        mild.  Nothing here affects correctness: the hard reachability mask
        applied afterwards is what guarantees the target is met.
        """
        v = self.v[:N_AA].unsqueeze(0)  # [1,20] in natural units
        eps = 1e-8

        # Model's own marginal at this step, at the sampling temperature.
        p = torch.softmax(base[:, :N_AA] / self.temperature, dim=-1)
        m_t = (p * v).sum(-1)
        var_t = (p * v * v).sum(-1) - m_t**2
        upd = active.float()
        self._m_sum = self._m_sum + upd * m_t
        self._var_sum = self._var_sum + upd * var_t.clamp(min=0.0)
        self._n_obs = self._n_obs + upd
        n_obs = self._n_obs.clamp(min=1.0)
        m_hat = self._m_sum / n_obs
        var_hat = (self._var_sum / n_obs).clamp(min=eps)

        # What the positions after this one would have to supply, per candidate.
        n_rem = self.suf_ndes[:, k + 1]
        n2_rem = self.suf_ndes2[:, k + 1]
        deficit = self.cfg.target - self.accum - self.suf_fixed[:, k + 1]
        need = deficit.unsqueeze(-1) - mult.unsqueeze(-1) * v  # [B,20]

        mu = (n_rem * m_hat).unsqueeze(-1)
        sigma = (n2_rem * var_hat).clamp(min=eps).sqrt().unsqueeze(-1)

        if self.cfg.mode == "eq":
            # Soft deadband: no penalty while the requirement sits inside the
            # tolerance window around what the remainder is expected to give.
            dist = ((need - mu).abs() - self.cfg.tolerance).clamp(min=0.0)
            logp = -0.5 * (dist / sigma) ** 2
        elif self.cfg.mode == "ge":
            logp = torch.special.log_ndtr((mu - need) / sigma)
        else:
            logp = torch.special.log_ndtr((need - mu) / sigma)
        logp = logp.clamp(min=-50.0)
        logp = logp - logp.max(dim=-1, keepdim=True).values

        # On the last designable step there is no remainder to look ahead to;
        # the mask alone decides, so do not let a degenerate sigma shout.
        logp = torch.where((n_rem > 0.5).unsqueeze(-1), logp, torch.zeros_like(logp))

        # The caller divides by the temperature, so scale to keep this a
        # likelihood reweighting rather than a temperature-dependent nudge.
        delta = (self.temperature * logp).clamp(min=-self.cfg.lambda_max)
        out = torch.zeros_like(base)
        out[:, :N_AA] = torch.where(
            active.unsqueeze(-1), delta, torch.zeros_like(delta)
        )
        return out

    def commit(self, k: int, S_t: torch.Tensor) -> None:
        """Record the residue chosen at step ``k``.

        ``S_t`` is the *final* choice, after fixed positions have been
        overwritten with their native residue, so this one line accounts for
        designable and fixed steps alike.
        """
        self.accum = self.accum + self.mult[:, k] * self.v[S_t]

    def value(self) -> torch.Tensor:
        return self.accum


# --------------------------------------------------------------------------- #
# SPPS risk steering
# --------------------------------------------------------------------------- #


class SPPSTerm:
    """Penalise Fmoc-SPPS-unfriendly local sequence patterns during decoding.

    Unlike :class:`LinearFunctional` this offers **no guarantee**.  The risk
    terms are defined over sequence neighbours, and MPNN's decoding order is
    random, so at the moment a position is decided only some of its neighbours
    exist.  The penalty therefore sees partial context and acts greedily.
    """

    def __init__(self, strength: float, weights: Optional[Dict[str, float]] = None):
        self.strength = float(strength)
        self.weights = dict(SPPS_DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.name = "spps_risk"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<spps_risk strength={self.strength:g}>"


class _SPPSState:
    def __init__(
        self,
        cfg: SPPSTerm,
        schedule: StepSchedule,
        chain_index: torch.Tensor,
        scope: torch.Tensor,
        L: int,
    ):
        self.cfg = cfg
        self.schedule = schedule
        B = schedule.B
        device = chain_index.device
        self.L = L
        self.chain_index = chain_index.tolist()
        self.scope = scope.tolist()
        self.S = [[-1] * L for _ in range(B)]
        self.device = device
        w = cfg.weights
        # Per-candidate unary cost, precomputed.
        self.unary = [0.0] * N_TOKENS
        for aa, idx in AA_TO_INT.items():
            c = 0.0
            if aa in BETA_BRANCHED:
                c += w["beta_branched"]
            if aa == "C":
                c += w["cys"]
            if aa == "M":
                c += w["met"]
            self.unary[idx] = c
        self.is_beta = [ALPHABET[i] in BETA_BRANCHED for i in range(N_TOKENS)]
        self.is_hydro = [ALPHABET[i] in ALIPHATIC for i in range(N_TOKENS)]
        self.half = SPPS_WINDOW // 2

    def _decoded(self, b: int, p: int) -> int:
        if p < 0 or p >= self.L:
            return -1
        if self.S[b][p] < 0:
            return -1
        return self.S[b][p]

    def _same_chain(self, p: int, q: int) -> bool:
        if q < 0 or q >= self.L:
            return False
        return self.chain_index[p] == self.chain_index[q]

    def _delta(self, b: int, p: int) -> List[float]:
        """Risk increase for each candidate residue at position ``p``."""
        w = self.cfg.weights
        prev_i = p - 1 if self._same_chain(p, p - 1) else -1
        next_i = p + 1 if self._same_chain(p, p + 1) else -1
        prev2_i = p - 2 if self._same_chain(p, p - 2) else -1
        next2_i = p + 2 if self._same_chain(p, p + 2) else -1
        prev_aa = self._decoded(b, prev_i) if prev_i >= 0 else -1
        next_aa = self._decoded(b, next_i) if next_i >= 0 else -1
        prev2_aa = self._decoded(b, prev2_i) if prev2_i >= 0 else -1
        next2_aa = self._decoded(b, next2_i) if next2_i >= 0 else -1

        # Hydrophobic load in the window around p, over decoded positions only.
        n_dec, n_hydro = 0, 0
        for q in range(p - self.half, p + self.half + 1):
            if q == p or not self._same_chain(p, q):
                continue
            aa = self._decoded(b, q)
            if aa >= 0:
                n_dec += 1
                if self.is_hydro[aa]:
                    n_hydro += 1
        allowance = SPPS_WINDOW_MAX_HYDROPHOBIC * (n_dec + 1) / SPPS_WINDOW

        out = [0.0] * N_TOKENS
        for a in range(N_AA):
            cost = self.unary[a]
            if self.is_beta[a]:
                if prev_aa >= 0 and self.is_beta[prev_aa]:
                    cost += w["beta_pair"]
                    if prev2_aa >= 0 and self.is_beta[prev2_aa]:
                        cost += w["beta_run3"]
                if next_aa >= 0 and self.is_beta[next_aa]:
                    cost += w["beta_pair"]
                    if next2_aa >= 0 and self.is_beta[next2_aa]:
                        cost += w["beta_run3"]
                if (
                    prev_aa >= 0
                    and next_aa >= 0
                    and self.is_beta[prev_aa]
                    and self.is_beta[next_aa]
                ):
                    cost += w["beta_run3"]
            # Aspartimide: p is the Asp, or p is the X following a decoded Asp.
            if ALPHABET[a] == "D" and next_aa >= 0:
                nxt = ALPHABET[next_aa]
                if nxt == "G":
                    cost += w["asp_gly"]
                elif nxt in ASP_OTHER_NEXT:
                    cost += w["asp_other"]
            if prev_aa >= 0 and ALPHABET[prev_aa] == "D":
                if ALPHABET[a] == "G":
                    cost += w["asp_gly"]
                elif ALPHABET[a] in ASP_OTHER_NEXT:
                    cost += w["asp_other"]
            excess = (n_hydro + (1.0 if self.is_hydro[a] else 0.0)) - allowance
            if excess > 0:
                cost += w["aliphatic_window"] * excess
            out[a] = cost
        return out

    def logit_delta(self, k: int, base: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(base)
        if self.cfg.strength == 0.0:
            return out
        pos = self.schedule.positions[:, k]  # [B,G]
        B = base.shape[0]
        rows = []
        any_row = False
        for b in range(B):
            acc = [0.0] * N_TOKENS
            hit = False
            for g in range(pos.shape[1]):
                p = int(pos[b, g])
                if p < 0 or not self.scope[p]:
                    continue
                d = self._delta(b, p)
                for a in range(N_AA):
                    acc[a] += d[a]
                hit = True
            any_row = any_row or hit
            rows.append(acc)
        if not any_row:
            return out
        delta = torch.tensor(rows, dtype=base.dtype, device=base.device)
        return -self.cfg.strength * delta

    def commit(self, k: int, S_t: torch.Tensor) -> None:
        pos = self.schedule.positions[:, k]
        s = S_t.tolist()
        for b in range(pos.shape[0]):
            for g in range(pos.shape[1]):
                p = int(pos[b, g])
                if p >= 0:
                    self.S[b][p] = int(s[b])


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #


class ConstraintSet:
    """A stateless bundle of constraints, plus the scope they apply to.

    ``scope`` is a boolean mask over the flattened residue axis selecting the
    positions a property is computed over -- normally every position of the
    chains being designed, so that fixed native residues inside those chains
    count toward the target.
    """

    def __init__(
        self,
        terms: Sequence[object],
        scope: torch.Tensor,
        chain_index: torch.Tensor,
    ):
        self.terms = list(terms)
        self.scope = scope.bool()
        self.chain_index = chain_index.long()

    def __bool__(self) -> bool:
        return bool(self.terms)

    def describe(self) -> str:
        return ", ".join(repr(t) for t in self.terms)

    def start(
        self,
        schedule: StepSchedule,
        S_true: torch.Tensor,
        chain_mask: torch.Tensor,
        bias: torch.Tensor,
        temperature: float,
    ) -> "_ActiveConstraints":
        return _ActiveConstraints(self, schedule, S_true, chain_mask, bias, temperature)


class _ActiveConstraints:
    """Per-``sample()``-call runtime for a :class:`ConstraintSet`."""

    def __init__(
        self,
        cs: ConstraintSet,
        schedule: StepSchedule,
        S_true: torch.Tensor,
        chain_mask: torch.Tensor,
        bias: torch.Tensor,
        temperature: float,
    ):
        device = S_true.device
        scope = cs.scope.to(device)
        L = S_true.shape[1]
        self.states: List[object] = []
        self.warnings: List[str] = []
        for term in cs.terms:
            if isinstance(term, LinearFunctional):
                st = _LinearState(
                    term, schedule, S_true, chain_mask, bias, scope, temperature
                )
                if not bool(st.feasible_at_start.all()):
                    raise InfeasibleConstraint(
                        f"{term.name}: target {term.target:g} {term.unit} is not "
                        f"reachable. Given the fixed residues and the allowed "
                        f"alphabet, the achievable range is "
                        f"[{float(st.suf_min[0, 0]):g}, {float(st.suf_max[0, 0]):g}] "
                        f"{term.unit}."
                    )
                if not st.exact_feasible_at_start():
                    raise InfeasibleConstraint(
                        f"{term.name}: target {term.target:g} {term.unit} is not "
                        f"reachable. Given the fixed residues and the allowed "
                        f"alphabet, the achievable interval is "
                        f"[{float(st.suf_min[0, 0]):g}, {float(st.suf_max[0, 0]):g}] "
                        f"but the discrete reachable values do not contain the "
                        f"requested target {term.unit}."
                    )
                self.warnings.extend(st.exactness_caveats())
                self.states.append(st)
            elif isinstance(term, SPPSTerm):
                self.states.append(
                    _SPPSState(term, schedule, cs.chain_index.to(device), scope, L)
                )
            else:  # pragma: no cover - guarded at construction
                raise TypeError(f"unsupported constraint {term!r}")

    def logit_delta(self, k: int, base: torch.Tensor) -> torch.Tensor:
        total = torch.zeros_like(base)
        for st in self.states:
            total = total + st.logit_delta(k, base + total)
        return total

    def commit(self, k: int, S_t: torch.Tensor) -> None:
        for st in self.states:
            st.commit(k, S_t)

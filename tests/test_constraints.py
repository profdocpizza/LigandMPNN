"""Tests for the deterministic decoding constraints.

Run from the repository root:

    python tests/test_constraints.py

Also collectable by pytest.  The decoding tests drive a faithful replica of
``ProteinMPNN.sample()``'s plain branch with random logits in place of the
network, which keeps them fast and independent of the checkpoints.  The
byte-identical regression against upstream needs the real model and lives in
``tests/test_identical_to_upstream.sh``.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraints import (  # noqa: E402
    ALPHABET,
    CHARGE_WEIGHTS,
    EXTINCTION_280_WEIGHTS,
    ConstraintSet,
    InfeasibleConstraint,
    LinearFunctional,
    SPPSTerm,
    StepSchedule,
    extinction_280,
    net_charge,
    spps_risk,
)


def _simulate(terms, L=60, B=8, n_fixed=0, temperature=0.1, seed=0, omit=None):
    """Replicate the plain branch of ProteinMPNN.sample() with random logits."""
    g = torch.Generator().manual_seed(seed)
    S_true = torch.randint(0, 20, (1, L), generator=g)
    chain_mask = torch.ones(1, L)
    chain_mask[0, :n_fixed] = 0.0
    bias = torch.zeros(1, L, 21)
    if omit:
        for aa in omit:
            bias[0, :, ALPHABET.index(aa)] = -1e8
    randn = torch.randn(B, L, generator=g)
    decoding_order = torch.argsort((chain_mask + 0.0001) * torch.abs(randn))
    S_true_b, chain_mask_b = S_true.repeat(B, 1), chain_mask.repeat(B, 1)
    bias_b = bias.repeat(B, 1, 1)
    cs = ConstraintSet(
        terms, torch.ones(L, dtype=torch.bool), torch.zeros(L, dtype=torch.long)
    )
    ac = cs.start(
        StepSchedule.from_decoding_order(decoding_order),
        S_true_b,
        chain_mask_b,
        bias_b,
        temperature,
    )
    S = torch.full((B, L), 20)
    logits_all = torch.randn(B, L, 21, generator=g) * 2.0
    for t_ in range(L):
        t = decoding_order[:, t_]
        logits = logits_all[torch.arange(B), t]
        bias_t = torch.gather(bias_b, 1, t[:, None, None].repeat(1, 1, 21))[:, 0, :]
        bias_t = bias_t + ac.logit_delta(t_, logits + bias_t)
        probs = torch.softmax((logits + bias_t) / temperature, -1)
        ps = probs[:, :20] / probs[:, :20].sum(-1, keepdim=True)
        S_t = torch.multinomial(ps, 1, generator=g)[:, 0]
        cm = torch.gather(chain_mask_b, 1, t[:, None])[:, 0]
        st = torch.gather(S_true_b, 1, t[:, None])[:, 0]
        S_t = (S_t * cm + st * (1 - cm)).long()
        ac.commit(t_, S_t)
        S[torch.arange(B), t] = S_t
    return ["".join(ALPHABET[i] for i in row) for row in S.tolist()]


# --------------------------------------------------------------------------- #
# Standalone scorers
# --------------------------------------------------------------------------- #


def test_net_charge_scorer():
    assert net_charge("DEKR") == 0.0
    assert net_charge("DDDD") == -4.0
    assert net_charge("KRKR") == 4.0
    assert net_charge("HHHH") == 0.0, "His is documented as neutral"
    assert net_charge("AGSTLV") == 0.0


def test_extinction_scorer():
    assert extinction_280("W") == 5500.0
    assert extinction_280("Y") == 1490.0
    assert extinction_280("WY") == 6990.0
    assert extinction_280("F" * 20) == 0.0, "Phe does not absorb at 280 nm"


def test_spps_scorer_components():
    _, c = spps_risk("DG")
    assert c["asp_gly"] == 1.0
    _, c = spps_risk("DS")
    assert c["asp_other"] == 1.0 and c["asp_gly"] == 0.0
    _, c = spps_risk("GDG")
    assert c["asp_gly"] == 1.0, "Asp-Gly is directional"
    _, c = spps_risk("IVT")
    assert c["beta_branched"] == 3.0 and c["beta_pair"] == 2.0
    assert c["beta_run3"] == 1.0
    _, c = spps_risk("IVTI")
    assert c["beta_run3"] == 2.0, "a run of 4 extends twice"
    assert spps_risk("GSGSGSGSGS")[0] == 0.0, "a benign sequence scores zero"
    assert spps_risk("IVIVIVIVDG")[0] > spps_risk("GSGSGSGSGS")[0]


# --------------------------------------------------------------------------- #
# Linear-functional guarantees
# --------------------------------------------------------------------------- #


def test_charge_is_exact():
    for target in (-20, -8, -1, 0, 4, 15):
        seqs = _simulate(
            [LinearFunctional("charge", CHARGE_WEIGHTS, target, "eq")],
            seed=abs(target) + 1,
        )
        got = [net_charge(s) for s in seqs]
        assert all(q == target for q in got), (target, got)


def test_charge_counts_fixed_native_residues():
    """Fixed positions contribute to the target; only designables can absorb."""
    seqs = _simulate(
        [LinearFunctional("charge", CHARGE_WEIGHTS, -6, "eq")], n_fixed=30, seed=7
    )
    assert all(net_charge(s) == -6 for s in seqs)


def test_charge_tolerance_widens_the_window():
    seqs = _simulate(
        [LinearFunctional("charge", CHARGE_WEIGHTS, -10, "eq", tolerance=3)], seed=3
    )
    assert all(abs(net_charge(s) + 10) <= 3 for s in seqs)


def test_tolerance_widens_the_achieved_spread():
    """A tolerance must actually buy back spread, not only widen the promise.

    Steering is a deadband controller: inside the window the model's own
    preference is left alone, so a wider window yields a wider distribution of
    achieved values rather than the same pinned value.
    """
    spreads = []
    for tol in (0, 2, 8):
        seqs = _simulate(
            [LinearFunctional("charge", CHARGE_WEIGHTS, -10, "eq", tolerance=tol)],
            B=16,
            seed=5,
        )
        got = [net_charge(s) for s in seqs]
        assert all(abs(q + 10) <= tol for q in got), (tol, got)
        spreads.append(max(got) - min(got))
    assert spreads[0] == 0, "an exact target must give a single value"
    assert spreads[-1] > spreads[0], f"tolerance should widen the spread: {spreads}"


def test_extinction_lower_bound_is_met():
    for thr in (1490.0, 5500.0, 11000.0, 22000.0):
        seqs = _simulate(
            [LinearFunctional("e280", EXTINCTION_280_WEIGHTS, thr, "ge")],
            seed=int(thr),
        )
        assert all(extinction_280(s) >= thr for s in seqs), thr


def test_extinction_forces_trp_when_tyr_is_omitted():
    """With Tyr omitted, a 5500 floor can only be met by a Trp."""
    seqs = _simulate(
        [LinearFunctional("e280", EXTINCTION_280_WEIGHTS, 5500.0, "ge")],
        omit="Y",
        seed=21,
    )
    assert all("W" in s for s in seqs)
    assert all("Y" not in s for s in seqs), "omit_AA must still be respected"


def test_charge_and_extinction_compose():
    seqs = _simulate(
        [
            LinearFunctional("charge", CHARGE_WEIGHTS, -5, "eq"),
            LinearFunctional("e280", EXTINCTION_280_WEIGHTS, 5500.0, "ge"),
        ],
        seed=11,
    )
    assert all(net_charge(s) == -5 for s in seqs)
    assert all(extinction_280(s) >= 5500.0 for s in seqs)


def test_unreachable_target_raises():
    for target in (-80.0, 80.0):
        try:
            _simulate([LinearFunctional("charge", CHARGE_WEIGHTS, target, "eq")], L=60)
        except InfeasibleConstraint as exc:
            assert "not reachable" in str(exc)
        else:
            raise AssertionError(f"target {target} should be unreachable at L=60")


def test_symmetric_charge_lattice_raises_instead_of_silent_miss():
    """Tied groups must reject charge values outside their discrete lattice."""
    cs = ConstraintSet(
        [LinearFunctional("charge", CHARGE_WEIGHTS, 1, "eq")],
        torch.ones(6, dtype=torch.bool),
        torch.zeros(6, dtype=torch.long),
    )
    order = StepSchedule.from_groups([[0, 1, 2], [3, 4, 5]], 2, "cpu")
    try:
        cs.start(
            order,
            torch.zeros(2, 6, dtype=torch.long),
            torch.ones(2, 6),
            torch.zeros(2, 6, 21),
            0.1,
        )
    except InfeasibleConstraint as exc:
        assert "discrete reachable values" in str(exc)
    else:
        raise AssertionError("charge +1 should be unreachable with groups of 3")


def test_omitting_all_neutral_residues_is_flagged():
    """Discrete charge reachability rejects an odd target without neutrals."""
    neutral = [aa for aa in ALPHABET[:20] if CHARGE_WEIGHTS.get(aa, 0.0) == 0.0]
    cs = ConstraintSet(
        [LinearFunctional("charge", CHARGE_WEIGHTS, 1.0, "eq")],
        torch.ones(8, dtype=torch.bool),
        torch.zeros(8, dtype=torch.long),
    )
    bias = torch.zeros(1, 8, 21)
    for aa in neutral:
        bias[0, :, ALPHABET.index(aa)] = -1e8
    try:
        cs.start(
            StepSchedule.from_decoding_order(torch.arange(8).unsqueeze(0)),
            torch.zeros(1, 8, dtype=torch.long),
            torch.ones(1, 8),
            bias,
            0.1,
        )
    except InfeasibleConstraint as exc:
        assert "discrete reachable values" in str(exc)
    else:
        raise AssertionError("odd charge should be unreachable without neutrals")


# --------------------------------------------------------------------------- #
# SPPS steering (directional, not guaranteed)
# --------------------------------------------------------------------------- #


def test_spps_bias_reduces_risk_monotonically():
    means = []
    for strength in (0.0, 0.5, 1.0, 2.0):
        seqs = _simulate([SPPSTerm(strength)], seed=42)
        means.append(sum(spps_risk(s)[0] for s in seqs) / len(seqs))
    assert all(
        b <= a + 1e-9 for a, b in zip(means, means[1:])
    ), f"risk should not increase with strength: {means}"
    assert means[-1] < means[0]


def test_spps_respects_chain_boundaries():
    """Asp at the end of one chain and Gly at the start of the next is not DG."""
    cs = ConstraintSet(
        [SPPSTerm(1.0)],
        torch.ones(4, dtype=torch.bool),
        torch.tensor([0, 0, 1, 1]),
    )
    order = torch.tensor([[1, 2, 0, 3]])
    ac = cs.start(
        StepSchedule.from_decoding_order(order),
        torch.zeros(1, 4, dtype=torch.long),
        torch.ones(1, 4),
        torch.zeros(1, 4, 21),
        0.1,
    )
    # Decide position 1 (last of chain 0) as Asp.
    ac.commit(0, torch.tensor([ALPHABET.index("D")]))
    # Position 2 is the first residue of chain 1, so Gly must not be penalised.
    delta = ac.logit_delta(1, torch.zeros(1, 21))
    assert float(delta[0, ALPHABET.index("G")]) == 0.0


def test_constraints_off_is_a_no_op():
    empty = ConstraintSet(
        [], torch.ones(4, dtype=torch.bool), torch.zeros(4, dtype=torch.long)
    )
    assert not empty


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)

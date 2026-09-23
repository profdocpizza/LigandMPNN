"""Render docs/CONSTRAINED_DECODING.md from the template and the results CSVs.

    python benchmarks/make_docs.py

Every number quoted in the documentation is computed here from
benchmarks/results/*.csv, never typed into the prose.  A hardcoded figure
survives a change of dataset and silently becomes a false claim -- that
happened once in this repo already -- so the template contains only
placeholders and this script fills them.

Edit docs/CONSTRAINED_DECODING.md.in, then re-run this.  Do not edit the
generated .md by hand.
"""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DOCS = os.path.join(os.path.dirname(HERE), "docs")


def load(name):
    return pd.read_csv(os.path.join(RESULTS, name))


def stats():
    s = {}
    pc = load("panel_charge.csv")
    ok = pc[pc.feasible == 1]
    d = ok.groupby("delta").nll.mean() - ok.baseline_nll.mean()
    s.update(
        charge_exact=f"{int(ok.exact.sum())}/{len(ok)}",
        charge_n_backbones=pc.backbone.nunique(),
        charge_len_lo=int(pc.length.min()), charge_len_hi=int(pc.length.max()),
        charge_infeasible=int((pc.feasible == 0).sum()),
        charge_cost_0=f"{d.loc[0]:+.2f}", charge_cost_15=f"{max(d.loc[-15], d.loc[15]):+.2f}",
        charge_conditions=ok.groupby(["backbone", "delta"]).ngroups,
    )
    # Cost tracks per-residue demand, not absolute shift.
    ok = ok.assign(excess=ok.nll - ok.baseline_nll)
    s["charge_r_abs"] = f"{ok.delta.abs().corr(ok.excess):.2f}"
    s["charge_r_perres"] = f"{(ok.delta.abs() / ok.length).corr(ok.excess):.2f}"

    tol = load("charge_tolerance.csv")
    g = tol.groupby(["target_charge", "tolerance"]).agg(
        nll=("nll", "mean"),
        spread=("achieved_charge", lambda x: x.max() - x.min())).reset_index()
    t0 = g[g.target_charge == g.target_charge.min()]
    s.update(
        tol_target=f"{g.target_charge.min():+.0f}",
        tol_nll_exact=f"{t0[t0.tolerance == 0].nll.iloc[0]:.2f}",
        tol_nll_wide=f"{t0[t0.tolerance == t0.tolerance.max()].nll.iloc[0]:.2f}",
        tol_wide=f"{t0.tolerance.max():.0f}",
        tol_inside=f"{int(tol.inside_window.sum())}/{len(tol)}",
    )

    pe = load("panel_e280.csv")
    base, con = pe[pe.constrained == 0], pe[pe.constrained == 1]
    frac = base.groupby("backbone").n_trp.apply(lambda x: (x > 0).mean())
    s.update(
        e280_met=f"{int(con.meets.sum())}/{len(con)}",
        e280_lowtrp=int((frac < 0.5).sum()), e280_notrp=int((frac == 0).sum()),
        e280_n=len(frac),
        e280_notrp_list=", ".join(sorted(frac[frac == 0].index)),
        e280_cost=f"{con[con.floor == 5500].nll.mean() - base.nll.mean():+.2f}",
    )
    if os.path.exists(os.path.join(RESULTS, "extinction_models.csv")):
        em = load("extinction_models.csv")
        s["e280_models"] = ", ".join(sorted(em.model.unique()))
        s["e280_models_met"] = (
            f"{int(em[em.constrained == 1].meets.sum())}/{len(em[em.constrained == 1])}")
        s["e280_models_base"] = (
            f"{int(em[em.constrained == 0].meets.sum())}/{len(em[em.constrained == 0])}")

    rc = load("rejection_compare.csv")
    g = rc.groupby(["backbone", "length"]).excess_nll.mean().reset_index().sort_values("length")
    s.update(
        rej_cells=len(rc), rej_backbones=rc.backbone.nunique(),
        rej_pool=int(rc.pool_size.median()),
        rej_median=f"{rc.excess_nll.median():+.3f}",
        rej_max=f"{rc.excess_nll.max():+.3f}",
        rej_exact=f"{int(rc.constrained_exact.sum())}/{int(rc.n_constrained.sum())}",
        rej_short=f"{g.iloc[0].excess_nll:+.3f}", rej_short_len=int(g.iloc[0].length),
        rej_short_bb=g.iloc[0].backbone,
        rej_long=f"{g.iloc[-1].excess_nll:+.3f}", rej_long_len=int(g.iloc[-1].length),
        rej_long_bb=g.iloc[-1].backbone,
        rej_r=f"{rc.length.corr(rc.excess_nll):.2f}",
        rej_table="\n".join(
            f"| {r.backbone} | {int(r.length)} | {r.excess_nll:+.3f} |"
            for _, r in g.iterrows()),
    )

    sw = load("charge_sweep.csv")
    s.update(
        models_exact=f"{int(sw.exact.sum())}/{len(sw)}",
        models_list=", ".join(f"`{m}`" for m in sorted(sw.model.unique())),
    )

    sp = load("spps.csv")
    sheet = {"1L2Y": 0.22, "1VII": 0.18, "1ENH": 0.23, "1PGA": 0.54, "1BDD": 0.09}
    risk = sp.groupby("spps_bias").spps_risk.mean()
    cost = (sp[sp.spps_bias == sp.spps_bias.max()].groupby("target").nll.mean()
            - sp[sp.spps_bias == 0].groupby("target").nll.mean())
    lens = sp.groupby("target").sequence.first().str.len()
    s.update(
        spps_risk0=f"{risk.loc[0.0]:.0f}", spps_risk1=f"{risk.loc[1.0]:.0f}",
        spps_bias_max=f"{sp.spps_bias.max():.0f}",
        spps_beta_cost=f"{cost['1PGA']:+.2f}",
        spps_helix_cost=f"{cost.drop('1PGA').max():+.2f}",
        spps_n_peptides=sp.target.nunique(),
        spps_len_lo=int(lens.min()), spps_len_hi=int(lens.max()),
        spps_dg0=f"{sp[sp.spps_bias == 0].n_asp_gly.mean():.2f}",
        spps_table="\n".join(
            f"| {t} | {int(lens[t])} | {sheet[t]:.0%} | {cost[t]:+.2f} |"
            for t in lens.sort_values().index),
    )
    return s


def main():
    tpl = os.path.join(DOCS, "CONSTRAINED_DECODING.md.in")
    out = os.path.join(DOCS, "CONSTRAINED_DECODING.md")
    s = stats()
    with open(tpl) as fh:
        text = fh.read()
    missing = [k for k in set(__import__("re").findall(r"{([a-z0-9_]+)}", text))
               if k not in s]
    if missing:
        raise SystemExit(f"template references unknown placeholders: {sorted(missing)}")
    with open(out, "w") as fh:
        fh.write(text.format(**s))
    print(f"wrote {os.path.relpath(out, os.path.dirname(HERE))} "
          f"({len(s)} values substituted)")


if __name__ == "__main__":
    main()

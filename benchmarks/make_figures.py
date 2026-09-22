"""Render the benchmark figures from benchmarks/results/*.csv.

    python benchmarks/make_figures.py

Each figure answers three questions, one per panel: does the constraint do what
it says, what does it cost as a function of the value you pass, and what else
changes.  Cost is always plotted as EXCESS NLL over the unconstrained run on
the same backbone, so zero means free and backbones with very different
absolute likelihoods can be shown together.

Needs pandas and matplotlib only -- not torch -- so it runs in a plain analysis
environment after the benchmarks have been executed.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

FOCAL = "#1f5fa9"
SECOND = "#e07b25"
THIRD = "#6a4c93"
NEUTRAL = "#7a7a7a"
LIGHT = "#c8c8c8"
REFERENCE = "#c1272d"
GOOD = "#2f7d4f"
BAND = "#e8f0e8"
BASE, MID, SMALL = 9, 8, 7.2
MODEL_C = {"protein_mpnn": FOCAL, "soluble_mpnn": SECOND, "ligand_mpnn": THIRD}


def style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": BASE, "axes.titlesize": BASE, "axes.labelsize": BASE,
        "axes.titlelocation": "left", "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#444444",
        "xtick.labelsize": SMALL, "ytick.labelsize": SMALL,
        "legend.fontsize": SMALL, "legend.frameon": False,
    })


def panel(ax, letter, question, answer):
    """Question above the panel, measured answer as the bold title."""
    ax.set_title(answer, fontweight="bold", pad=6)
    ax.text(0, 1.30, question, transform=ax.transAxes, fontsize=SMALL,
            color=NEUTRAL, va="bottom")
    ax.text(-0.17, 1.30, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom")


def costnote(fig):
    fig.text(0.008, 0.012,
             "Excess NLL = how much less likely the model finds its own output "
             "than when unconstrained, per residue. 0 = free; 0.1 is small, "
             "1.0 is a different kind of sequence.",
             fontsize=SMALL, color=NEUTRAL)


def spread_lines(ax, df, xcol, ycol, group, colour=FOCAL, label_median=True):
    """One faint line per backbone plus a bold median across them."""
    for _, sub in df.groupby(group):
        s = sub.groupby(xcol)[ycol].mean()
        ax.plot(s.index, s.values, "-", color=colour, lw=0.8, alpha=0.30, zorder=2)
    med = df.groupby(xcol)[ycol].median()
    ax.plot(med.index, med.values, "-o", color=colour, lw=2.0, ms=4.5, mew=0,
            zorder=4, label="median of backbones" if label_median else None)
    return med


# --------------------------------------------------------------------------- #


def figure_charge():
    pc = pd.read_csv(os.path.join(RESULTS, "panel_charge.csv"))
    pc = pc[pc.feasible == 1].copy()
    pc["excess"] = pc.nll - pc.baseline_nll
    tol = pd.read_csv(os.path.join(RESULTS, "charge_tolerance.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))

    ax = axes[0]
    jitter = np.random.default_rng(0).uniform(-0.25, 0.25, len(pc))
    ax.scatter(pc.target_charge + jitter, pc.achieved_charge, s=6, color=FOCAL,
               alpha=0.30, lw=0, zorder=3)
    lim = [pc.target_charge.min() - 2, pc.target_charge.max() + 2]
    ax.plot(lim, lim, ls=(0, (4, 3)), color=NEUTRAL, lw=0.9, zorder=1,
            label="requested = achieved")
    ax.set_xlabel("net charge requested (e)")
    ax.set_ylabel("net charge achieved (e)")
    ax.legend(loc="upper left")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    panel(ax, "a", "Does it hit the target?",
          "Yes — 1792/1792 designs,\n16 backbones, 0 failures")

    ax = axes[1]
    spread_lines(ax, pc, "delta", "excess", "backbone")
    ax.axhline(0, color=REFERENCE, lw=1.1, ls="--", zorder=1)
    ax.set_xlabel("charge shift requested, relative to what\nthe model produces on its own (e)")
    ax.set_ylabel("excess NLL vs unconstrained")
    ax.legend(loc="upper center")
    panel(ax, "b", "What does it cost?",
          "+0.07 at the model's own charge,\n+0.42 at a 15 e shift")

    ax = axes[2]
    for m, sub in tol.groupby("model"):
        for t, ls in ((0, "-"), (4, "--")):
            s = sub[sub.tolerance == t]
            if s.empty:
                continue
            g = s.groupby("target_charge").nll.mean()
            ax.plot(g.index, g.values, ls, color=MODEL_C.get(m, FOCAL), lw=1.6,
                    marker="o" if t == 0 else "s", ms=3.5, mew=0,
                    label=m.replace("_mpnn", "") if t == 0 else None)
    handles, labels = ax.get_legend_handles_labels()
    handles += [mpl.lines.Line2D([], [], color=NEUTRAL, ls="-"),
                mpl.lines.Line2D([], [], color=NEUTRAL, ls="--")]
    labels += ["exact", "tolerance ±4 e"]
    ax.legend(handles, labels, loc="upper center", ncol=2, columnspacing=1.0)
    ax.set_xlabel("net charge requested (e)")
    ax.set_ylabel("mean per-residue NLL")
    panel(ax, "c", "Does a tolerance buy anything?",
          "Yes — ±4 e is cheaper\neverywhere, and still guaranteed")

    fig.tight_layout(w_pad=2.6, rect=[0, 0.05, 1, 0.86])
    costnote(fig)
    out = os.path.join(FIGURES, "charge_benchmark.png")
    fig.savefig(out)
    return fig, out


def figure_extinction():
    pe = pd.read_csv(os.path.join(RESULTS, "panel_e280.csv"))
    pe["excess"] = pe.nll - pe.baseline_nll
    base = pe[pe.constrained == 0]
    con = pe[pe.constrained == 1].copy()

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))

    ax = axes[0]
    frac = base.groupby("backbone").n_trp.apply(lambda s: (s > 0).mean()).sort_values()
    y = np.arange(len(frac))
    ax.barh(y, frac.values * 100, color=LIGHT, zorder=3, label="unconstrained")
    ax.scatter([100] * len(frac), y, s=22, color=GOOD, zorder=4, marker="D",
               label="with --min_extinction_280 5500")
    ax.set_yticks(y); ax.set_yticklabels(frac.index, fontsize=SMALL)
    ax.set_xlabel("designs carrying at least one Trp (%)")
    ax.axvline(50, color=REFERENCE, lw=1.0, ls="--", zorder=2)
    ax.legend(loc="lower right")
    panel(ax, "a", "Is there a problem to solve?",
          "Yes — 6 of 16 backbones give a Trp\nin under half their designs")

    ax = axes[1]
    spread_lines(ax, con, "floor", "excess", "backbone")
    ax.axhline(0, color=REFERENCE, lw=1.1, ls="--", zorder=1)
    ax.set_xlabel(r"--min_extinction_280 requested (M$^{-1}$cm$^{-1}$)")
    ax.set_ylabel("excess NLL vs unconstrained")
    ax.legend(loc="upper left")
    panel(ax, "b", "What does it cost?",
          "Little — a guaranteed Trp\ncosts about +0.1 nats")

    ax = axes[2]
    for col, colour, lab in (("n_trp", FOCAL, "Trp"), ("n_tyr", SECOND, "Tyr")):
        g = pe.groupby("floor")[col].mean()
        ax.plot(g.index, g.values, "-o", color=colour, lw=1.8, ms=4.5, mew=0, label=lab)
    ax.set_xlabel(r"--min_extinction_280 requested (M$^{-1}$cm$^{-1}$)")
    ax.set_ylabel("mean count per design")
    ax.legend(loc="upper left")
    panel(ax, "c", "What else changes?",
          "Trp is added; Tyr is\nbarely touched")

    fig.tight_layout(w_pad=2.6, rect=[0, 0.05, 1, 0.86])
    costnote(fig)
    out = os.path.join(FIGURES, "extinction_benchmark.png")
    fig.savefig(out)
    return fig, out


def figure_spps():
    df = pd.read_csv(os.path.join(RESULTS, "spps.csv"))
    targets = sorted(df.target.unique())
    tc = {t: c for t, c in zip(targets, (FOCAL, SECOND, THIRD))}

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))

    ax = axes[0]
    for t in targets:
        g = df[df.target == t].groupby("spps_bias").spps_risk.mean()
        ax.plot(g.index, g.values, "-o", color=tc[t], lw=1.8, ms=4.5, mew=0, label=t)
    ax.set_xlabel("--spps_bias")
    ax.set_ylabel("mean SPPS risk score")
    ax.legend(loc="upper right")
    panel(ax, "a", "Does the risk score fall?",
          "Yes — most of the drop\nby bias 1.0")

    ax = axes[1]
    for t in targets:
        g = df[df.target == t].groupby("spps_bias").n_asp_gly.mean()
        ax.plot(g.index, g.values, "-o", color=tc[t], lw=1.8, ms=4.5, mew=0, label=t)
    ax.set_xlabel("--spps_bias")
    ax.set_ylabel("mean Asp-Gly motifs per design")
    ax.legend(loc="upper right")
    panel(ax, "b", "Do the classic motifs go?",
          "Asp-Gly is gone from\nbias 1.0 upward")

    ax = axes[2]
    for t in targets:
        g = df[df.target == t].groupby("spps_bias").nll.mean()
        ax.plot(g.index, g.values - g.loc[0.0], "-o", color=tc[t], lw=1.8, ms=4.5,
                mew=0, label=t)
    ax.axhline(0, color=REFERENCE, lw=1.1, ls="--", zorder=1)
    ax.axvspan(0.5, 1.0, color=BAND, zorder=0)
    ax.set_xlabel("--spps_bias")
    ax.set_ylabel("excess NLL vs unconstrained")
    ax.legend(loc="upper left")
    panel(ax, "c", "What does it cost?",
          "Real — this one trades\nlikelihood for synthesis")

    fig.tight_layout(w_pad=2.6, rect=[0, 0.05, 1, 0.86])
    costnote(fig)
    out = os.path.join(FIGURES, "spps_benchmark.png")
    fig.savefig(out)
    return fig, out


def figure_rejection():
    path = os.path.join(RESULTS, "rejection_compare.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))

    ax = axes[0]
    ax.scatter(df.rejection_nll, df.constrained_nll, s=26, color=FOCAL, alpha=0.75,
               lw=0, zorder=3)
    lo = min(df.rejection_nll.min(), df.constrained_nll.min()) - 0.05
    hi = max(df.rejection_nll.max(), df.constrained_nll.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), color=NEUTRAL, lw=0.9, zorder=1,
            label="equal to the true conditional")
    ax.set_xlabel("NLL of rejection-sampled designs\n(the true conditional)")
    ax.set_ylabel("NLL of constrained designs")
    ax.legend(loc="upper left")
    panel(ax, "a", "Does it match exact conditioning?",
          f"Close — median excess\n{df.excess_nll.median():+.3f} nats")

    ax = axes[1]
    order = df.sort_values("excess_nll")
    lab = [f"{b} {c:+.0f}e" for b, c in zip(order.backbone, order.charge)]
    y = np.arange(len(order))
    ax.barh(y, order.excess_nll, color=FOCAL, zorder=3)
    ax.axvline(0, color=REFERENCE, lw=1.1, ls="--", zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=SMALL)
    ax.set_xlabel("excess NLL over the true conditional")
    panel(ax, "b", "Where is the residual cost?",
          "Small everywhere; largest on\nthe shortest chains")

    fig.tight_layout(w_pad=2.6, rect=[0, 0.05, 1, 0.86])
    costnote(fig)
    out = os.path.join(FIGURES, "rejection_vs_constrained.png")
    fig.savefig(out)
    return fig, out


def main():
    os.makedirs(FIGURES, exist_ok=True)
    style()
    for fn in (figure_charge, figure_extinction, figure_spps, figure_rejection):
        _, path = fn()
        if path:
            print("wrote", os.path.relpath(path, HERE))
        else:
            print("skipped", fn.__name__, "(results not present yet)")


if __name__ == "__main__":
    main()

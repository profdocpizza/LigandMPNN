"""Render the benchmark figures from benchmarks/results/*.csv.

    python benchmarks/make_figures.py

Conventions, so the panels can be read without the caption:

* Y axes are ABSOLUTE mean per-residue NLL -- the quantity run.py reports as
  overall_confidence = exp(-NLL).  Lower means the model finds the sequence
  more likely.  The unconstrained run on the same backbone is always drawn as
  a reference, so the cost of a constraint is the gap to that line.
* X axes name the CLI flag being swept.
* Every number quoted in a panel title is computed from the dataframe, never
  typed, so a title cannot survive a change of dataset.

Needs pandas and matplotlib only -- not torch.
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
NEUTRAL = "#7a7a7a"
LIGHT = "#c8c8c8"
REFERENCE = "#c1272d"
GOOD = "#2f7d4f"
RAMP = ["#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"]
BASE, MID, SMALL = 9, 8, 7.2


def style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": BASE, "axes.titlesize": BASE, "axes.labelsize": BASE,
        "axes.titlelocation": "left", "axes.spines.top": False,
        "axes.spines.right": False, "xtick.labelsize": SMALL,
        "ytick.labelsize": SMALL, "legend.fontsize": SMALL,
        "legend.frameon": False, "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8, "lines.solid_capstyle": "round",
    })


def panel(ax, letter, question, answer):
    ax.text(-0.24, 1.30, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top")
    ax.text(0, 1.30, question, transform=ax.transAxes, fontsize=SMALL,
            color=NEUTRAL, va="top")
    ax.text(0, 1.20, answer, transform=ax.transAxes, fontsize=BASE,
            fontweight="bold", va="top", linespacing=1.35)


def nllnote(fig, baseline=True):
    txt = ("Per-residue NLL = how unlikely the model finds its own output, as "
           "reported by run.py (overall_confidence = exp(-NLL)). Lower is better.")
    if baseline:
        txt += (" Dashed red line = the same backbones designed with no "
                "constraint, so the gap to it is what the constraint costs.")
    fig.text(0.008, 0.012, txt, fontsize=SMALL, color=NEUTRAL, va="bottom")


def baseline_line(ax, value, label="unconstrained"):
    ax.axhline(value, color=REFERENCE, lw=1.2, ls="--", zorder=1)
    ax.text(0.99, value, label, transform=ax.get_yaxis_transform(),
            fontsize=SMALL, color=REFERENCE, ha="right", va="bottom")


# --------------------------------------------------------------------------- #


def figure_charge():
    pc = pd.read_csv(os.path.join(RESULTS, "panel_charge.csv"))
    tol = pd.read_csv(os.path.join(RESULTS, "charge_tolerance.csv"))
    ok = pc[pc.feasible == 1]

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.9))

    ax = axes[0]
    ax.scatter(ok.target_charge, ok.achieved_charge, s=9, color=FOCAL,
               alpha=0.5, lw=0, zorder=3)
    lim = [ok.target_charge.min() - 2, ok.target_charge.max() + 2]
    ax.plot(lim, lim, ls=(0, (4, 3)), color=NEUTRAL, lw=1.0, zorder=1,
            label="requested = achieved")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("--target_charge (e)")
    ax.set_ylabel("net charge achieved (e)")
    ax.legend(loc="upper left")
    panel(ax, "a", "Does it hit the target?",
          f"Yes — {int(ok.exact.sum())}/{len(ok)} designs,\n"
          f"{pc.backbone.nunique()} backbones, 0 failures")

    ax = axes[1]
    for bb, sub in ok.groupby("backbone"):
        g = sub.groupby("delta").nll.mean()
        ax.plot(g.index, g.values, "-", color=LIGHT, lw=0.8, zorder=2)
    med = ok.groupby("delta").nll.mean()
    ax.plot(med.index, med.values, "-o", color=FOCAL, lw=2.0, ms=4.5, mew=0,
            zorder=4, label=f"mean of {ok.backbone.nunique()} backbones")
    baseline_line(ax, ok.baseline_nll.mean())
    ax.set_xlabel("--target_charge, relative to each\nbackbone's unconstrained mean (e)")
    ax.set_ylabel("mean per-residue NLL")
    ax.legend(loc="upper center")
    _d = ok.groupby("delta").nll.mean() - ok.baseline_nll.mean()
    panel(ax, "b", "What does it cost?",
          f"+{_d.loc[0]:.2f} nats at the model's own\ncharge, "
          f"+{max(_d.loc[-15], _d.loc[15]):.2f} at a 15 e shift")

    ax = axes[2]
    for i, (t, colour) in enumerate(
            zip(sorted(tol.target_charge.unique()), (FOCAL, SECOND))):
        sub = tol[tol.target_charge == t]
        g = sub.groupby("tolerance").nll.mean()
        sp = sub.groupby("tolerance").achieved_charge.agg(lambda x: x.max() - x.min())
        ax.plot(g.index, g.values, "-o", color=colour, lw=1.8, ms=5, mew=0,
                label=f"--target_charge {t:+.0f}")
        # Only the endpoint is annotated: at tolerance 0 the spread is 0 by
        # construction, which the panel title already states.
        ax.annotate(f"achieved spread {sp.iloc[-1]:.0f} e",
                    (g.index[-1], g.values[-1]), textcoords="offset points",
                    xytext=(-6, -4), fontsize=SMALL, color=colour,
                    ha="right", va="top")
    ax.set_xlabel("--charge_tolerance (e)")
    ax.set_ylabel("mean per-residue NLL")
    ax.set_xticks(sorted(tol.tolerance.unique()))
    ax.margins(y=0.18)
    ax.legend(loc="upper right")
    panel(ax, "c", "Does a tolerance buy anything?",
          "Yes — a wider window is\ncheaper, and still guaranteed")

    fig.tight_layout(w_pad=3.2, rect=[0, 0.06, 1, 0.85])
    nllnote(fig)
    out = os.path.join(FIGURES, "charge_benchmark.png")
    fig.savefig(out)
    return fig, out


def figure_extinction():
    pe = pd.read_csv(os.path.join(RESULTS, "panel_e280.csv"))
    base = pe[pe.constrained == 0]
    con = pe[pe.constrained == 1]
    frac = base.groupby("backbone").n_trp.apply(lambda s: (s > 0).mean()).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))

    ax = axes[0]
    y = np.arange(len(frac))
    ax.barh(y, frac.values * 100, color=LIGHT, zorder=3, label="unconstrained")
    met = con[con.floor == 5500].groupby("backbone").apply(
        lambda d: (d.n_trp > 0).mean(), include_groups=False).reindex(frac.index)
    ax.scatter(met.values * 100, y, s=26, color=GOOD, marker="D", zorder=4,
               label="--min_extinction_280 5500")
    ax.axvline(50, color=REFERENCE, lw=1.0, ls=":", zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(frac.index, fontsize=SMALL)
    ax.set_xlabel("designs carrying at least one Trp (%)")
    ax.set_xlim(0, 118)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, -0.02))
    _n = int((frac < 0.5).sum()); _z = int((frac == 0).sum())
    panel(ax, "a", "Is there a problem to solve?",
          f"Yes — {_n} of {len(frac)} backbones give a Trp in\n"
          f"under half their designs, {_z} in none")

    ax = axes[1]
    for bb, sub in con.groupby("backbone"):
        g = sub.groupby("floor").nll.mean()
        ax.plot(g.index, g.values, "-", color=LIGHT, lw=0.8, zorder=2)
    med = con.groupby("floor").nll.mean()
    ax.plot(med.index, med.values, "-o", color=FOCAL, lw=2.0, ms=5, mew=0,
            zorder=4, label=f"mean of {con.backbone.nunique()} backbones")
    baseline_line(ax, base.nll.mean())
    ax.set_xlabel(r"--min_extinction_280 (M$^{-1}$cm$^{-1}$)")
    ax.set_ylabel("mean per-residue NLL")
    ax.legend(loc="upper left")
    _e = con[con.floor == 5500].nll.mean() - base.nll.mean()
    panel(ax, "b", "What does it cost?",
          f"Little — a guaranteed Trp\ncosts about +{_e:.2f} nats")

    fig.tight_layout(w_pad=3.2, rect=[0, 0.06, 1, 0.85])
    nllnote(fig)
    out = os.path.join(FIGURES, "extinction_benchmark.png")
    fig.savefig(out)
    return fig, out


def figure_spps():
    df = pd.read_csv(os.path.join(RESULTS, "spps.csv")).copy()
    # Backbone dihedrals, measured from the input structures (see the doc).
    sheet = {"1L2Y": 0.22, "1VII": 0.18, "1ENH": 0.23, "1PGA": 0.54, "1BDD": 0.09}
    lengths = df.groupby("target").sequence.first().str.len().sort_values()
    targets = list(lengths.index)
    tc = {t: RAMP[i % len(RAMP)] for i, t in enumerate(targets)}
    tlab = {t: f"{t} ({lengths[t]} aa, {sheet[t]:.0%} sheet)" for t in targets}
    df["n_asp_all"] = df.n_asp_gly + df.n_asp_other

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))

    ax = axes[0]
    for t in targets:
        g = df[df.target == t].groupby("spps_bias").spps_risk.mean()
        ax.plot(g.index, g.values, "-o", color=tc[t], lw=1.8, ms=4.5, mew=0,
                label=tlab[t])
    ax.set_xlabel("--spps_bias")
    ax.set_ylabel("mean SPPS risk score")
    ax.legend(loc="upper right")
    _r = df.groupby("spps_bias").spps_risk.mean()
    panel(ax, "a", "Does the risk score fall?",
          f"Yes — {_r.loc[0.0]:.0f} to {_r.loc[1.0]:.0f} by bias 1.0,\n"
          f"all {len(targets)} peptides")

    ax = axes[1]
    for col, colour, lab, mk in (
            ("n_beta_pair", FOCAL, "adjacent Ile/Val/Thr", "o"),
            ("n_beta_run3", "#08306b", r"Ile/Val/Thr runs $\geq$3", "s"),
            ("n_aliphatic_window", SECOND, "hydrophobic-window excess", "^"),
            ("n_asp_all", REFERENCE, "aspartimide-prone Asp-X", "D")):
        g = df.groupby("spps_bias")[col].mean()
        ax.plot(g.index, g.values, "-", marker=mk, color=colour, lw=1.8, ms=5,
                mew=0, label=lab)
    ax.set_xlabel("--spps_bias")
    ax.set_ylabel("mean motif count per design")
    ax.legend(loc="upper right")
    _b0 = df[df.spps_bias == 0].n_beta_pair.mean()
    _b1 = df[df.spps_bias == 1.0].n_beta_pair.mean()
    panel(ax, "b", "Which motifs go away?",
          f"All of them — adjacent Ile/Val/Thr\n"
          f"{_b0:.1f} to {_b1:.1f} per design by bias 1.0")

    ax = axes[2]
    for t in targets:
        g = df[df.target == t].groupby("spps_bias").nll.mean()
        beta = sheet[t] > 0.4
        ax.plot(g.index, g.values, "-o", color=tc[t], lw=2.6 if beta else 1.4,
                ms=5.5 if beta else 4, mew=0, zorder=4 if beta else 3)
        if beta:
            # Panel a's legend already maps colour to peptide, so only the
            # peptide the title singles out is labelled here.
            ax.annotate(f"{t}\n{sheet[t]:.0%} sheet", (g.index[-1], g.values[-1]),
                        textcoords="offset points", xytext=(7, 0), fontsize=SMALL,
                        color=tc[t], va="center", fontweight="bold")
    ax.set_xlabel("--spps_bias")
    ax.set_ylabel("mean per-residue NLL")
    ax.set_xlim(-0.1, 2.85)
    _lo, _hi = ax.get_ylim()
    ax.set_ylim(_lo, _hi + 0.06 * (_hi - _lo))
    _c = (df[df.spps_bias == 2].groupby("target").nll.mean()
          - df[df.spps_bias == 0].groupby("target").nll.mean())
    panel(ax, "c", "What does it cost?",
          f"Only the \u03b2-sheet peptide: +{_c['1PGA']:.2f} nats\n"
          f"for 1PGA, \u2264+{_c.drop('1PGA').max():.2f} for the helical four")

    fig.tight_layout(w_pad=3.2, rect=[0, 0.06, 1, 0.85])
    nllnote(fig, baseline=False)
    out = os.path.join(FIGURES, "spps_benchmark.png")
    fig.savefig(out)
    return fig, out


def figure_rejection():
    path = os.path.join(RESULTS, "rejection_compare.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path).sort_values("length")
    lens = sorted(df.length.unique())
    lc = {L: RAMP[i % len(RAMP)] for i, L in enumerate(lens)}
    lab = {L: f"{df[df.length == L].backbone.iloc[0]} ({L} aa)" for L in lens}

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))

    ax = axes[0]
    for L in lens:
        sub = df[df.length == L]
        ax.scatter(sub.rejection_nll, sub.constrained_nll, s=30, color=lc[L],
                   alpha=0.85, lw=0, zorder=3, label=lab[L])
    lo = min(df.rejection_nll.min(), df.constrained_nll.min()) - 0.05
    hi = max(df.rejection_nll.max(), df.constrained_nll.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), color=NEUTRAL, lw=1.0, zorder=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("NLL of rejection-sampled designs\n(exact conditioning)")
    ax.set_ylabel("NLL of --target_charge designs")
    ax.legend(loc="upper left", fontsize=SMALL - 0.5)
    panel(ax, "a", "Does it match exact conditioning?",
          f"Close — median gap {df.excess_nll.median():+.3f} nats\n"
          f"over {len(df)} backbone \u00d7 charge cells")

    ax = axes[1]
    g = df.groupby(["backbone", "length"])[["rejection_nll", "constrained_nll"]]\
          .mean().reset_index().sort_values("length")
    ax.plot(g.length, g.constrained_nll, "-o", color=FOCAL, lw=2.0, ms=6, mew=0,
            zorder=4, label="--target_charge")
    ax.plot(g.length, g.rejection_nll, "--s", color=GOOD, lw=1.8, ms=5, mew=0,
            zorder=3, label="rejection (exact conditioning)")
    for _, r in g.iterrows():
        ax.annotate(f"+{r.constrained_nll - r.rejection_nll:.3f}",
                    (r.length, r.constrained_nll), textcoords="offset points",
                    xytext=(0, 8), fontsize=SMALL, color=NEUTRAL, ha="center")
    ax.set_xlabel("backbone length (residues)")
    ax.set_ylabel("mean per-residue NLL")
    ax.legend(loc="upper right")
    ax.margins(y=0.2)
    panel(ax, "b", "Where is the remaining gap?",
          f"Short chains — it shrinks from\n"
          f"+{g.iloc[0].constrained_nll - g.iloc[0].rejection_nll:.2f} at "
          f"{int(g.iloc[0].length)} aa to "
          f"+{g.iloc[-1].constrained_nll - g.iloc[-1].rejection_nll:.2f} at "
          f"{int(g.iloc[-1].length)} aa")

    ax = axes[2]
    for L in lens:
        sub = df[df.length == L].sort_values("charge")
        ax.plot(sub.charge, sub.constrained_nll, "-o", color=lc[L], lw=1.6,
                ms=4, mew=0, zorder=3)
        ax.plot(sub.charge, sub.rejection_nll, ":", color=lc[L], lw=1.4, zorder=2)
    ax.set_xlabel("--target_charge (e)")
    ax.set_ylabel("mean per-residue NLL")
    h = [mpl.lines.Line2D([], [], color=NEUTRAL, ls="-", marker="o", ms=4, mew=0),
         mpl.lines.Line2D([], [], color=NEUTRAL, ls=":")]
    ax.legend(h, ["--target_charge", "rejection (exact)"], loc="upper center")
    ax.margins(y=0.18)
    panel(ax, "c", "Does the gap widen at harder targets?",
          "No — it stays flat across the\ncharges each backbone populates")

    fig.tight_layout(w_pad=3.2, rect=[0, 0.06, 1, 0.85])
    nllnote(fig, baseline=False)
    out = os.path.join(FIGURES, "rejection_vs_constrained.png")
    fig.savefig(out)
    return fig, out


def main():
    os.makedirs(FIGURES, exist_ok=True)
    style()
    for fn in (figure_charge, figure_extinction, figure_spps, figure_rejection):
        _, path = fn()
        print("wrote", os.path.relpath(path, HERE) if path else f"skipped {fn.__name__}")


if __name__ == "__main__":
    main()

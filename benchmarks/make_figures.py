"""Render the three benchmark figures from benchmarks/results/*.csv.

    python benchmarks/make_figures.py

Needs pandas and matplotlib only -- not torch -- so it can run in a plain
analysis environment after the benchmarks have been executed.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

# Focal / comparator palette. Blue-orange-purple stays distinguishable under
# deuteranopia; red is reserved for the "unconstrained" reference marks.
FOCAL = "#1f5fa9"
C_MODEL = {
    "protein_mpnn": "#1f5fa9",
    "soluble_mpnn": "#e07b25",
    "ligand_mpnn": "#6a4c93",
}
NEUTRAL = "#666666"
REFERENCE = "#c1272d"
BASE, MID, SMALL = 9, 8, 7


def style():
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": BASE,
            "axes.titlesize": BASE,
            "axes.labelsize": BASE,
            "axes.titlelocation": "left",
            "axes.titleweight": "regular",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#111111",
            "axes.edgecolor": "#444444",
            "xtick.labelsize": SMALL,
            "ytick.labelsize": SMALL,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "legend.fontsize": MID,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
        }
    )


def letter(ax, ch):
    ax.text(
        -0.16,
        1.06,
        ch,
        transform=ax.transAxes,
        fontsize=BASE + 2,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


# --------------------------------------------------------------------------- #
# Figure 1: charge
# --------------------------------------------------------------------------- #


def figure_charge():
    sweep = pd.read_csv(os.path.join(RESULTS, "charge_sweep.csv"))
    base = pd.read_csv(os.path.join(RESULTS, "charge_baseline.csv"))
    tol = pd.read_csv(os.path.join(RESULTS, "charge_tolerance.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
    ax = axes[0]
    lim = (sweep.target_charge.min() - 4, sweep.target_charge.max() + 4)
    ax.plot(
        lim,
        lim,
        color=NEUTRAL,
        lw=0.9,
        ls=(0, (4, 3)),
        zorder=1,
        label="requested = achieved",
    )
    for model, sub in sweep.groupby("model"):
        g = sub.groupby("target_charge").achieved_charge
        ax.plot(
            g.mean().index,
            g.mean().values,
            "o",
            ms=4.5,
            color=C_MODEL[model],
            label=model.replace("_mpnn", ""),
            zorder=3,
            mew=0,
        )
    for model, sub in base.groupby("model"):
        ax.axhline(
            sub.charge.mean(),
            color=C_MODEL[model],
            lw=0.8,
            ls=":",
            zorder=2,
        )
    ax.annotate(
        "unconstrained mean\n(all three models)",
        xy=(lim[0] + 1, base.charge.mean()),
        xytext=(-26, 16),
        textcoords="offset points",
        fontsize=SMALL,
        color=NEUTRAL,
        ha="left",
        arrowprops=dict(arrowstyle="-", lw=0.7, color=NEUTRAL),
    )
    n_exact = int(sweep.exact.sum())
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("requested net charge (e)")
    ax.set_ylabel("achieved net charge (e)")
    ax.set_title(f"Every design hits the target exactly\n({n_exact}/{len(sweep)})")
    ax.legend(loc="upper left", handletextpad=0.3, borderpad=0.2)
    ax.set_aspect("equal")
    letter(ax, "a")

    ax = axes[1]
    for model, sub in sweep.groupby("model"):
        g = sub.groupby("target_charge")
        ax.plot(
            g.nll.mean().index,
            g.nll.mean().values,
            "-o",
            ms=3.5,
            color=C_MODEL[model],
            mew=0,
            label=model.replace("_mpnn", ""),
        )
    ax.axhline(base.nll.mean(), color=REFERENCE, lw=1.0, ls="--")
    ax.text(
        sweep.target_charge.min(),
        base.nll.mean() - 0.055,
        "unconstrained",
        fontsize=SMALL,
        color=REFERENCE,
        ha="left",
    )
    ax.set_xlabel("requested net charge (e)")
    ax.set_ylabel("mean per-residue NLL")
    ax.set_title("Cost is lowest at net zero and\nrises toward both extremes")
    ax.text(
        0.5,
        0.97,
        "lower = better",
        transform=ax.transAxes,
        fontsize=SMALL,
        color=NEUTRAL,
        ha="center",
        va="top",
    )
    ax.margins(0.06)
    letter(ax, "b")

    ax = axes[2]
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    for i, (target, sub) in enumerate(tol.groupby("target_charge")):
        colour = FOCAL if i == 0 else C_MODEL["soluble_mpnn"]
        g = sub.groupby("tolerance")
        ax.plot(
            g.nll.mean().index,
            g.nll.mean().values,
            "-o",
            ms=3.5,
            mew=0,
            color=colour,
            label=f"target {target:+.0f} e",
        )
        spread = g.achieved_charge.agg(lambda s: s.max() - s.min())
        ax2.plot(
            spread.index, spread.values, "--s", ms=3, mew=0, color=colour, alpha=0.55
        )
    ax.axhline(tol.baseline_nll.iloc[0], color=REFERENCE, lw=1.0, ls="--")
    ax.set_xlabel("--charge_tolerance (e)")
    ax.set_ylabel("mean per-residue NLL")
    ax2.set_ylabel("achieved spread (e)", color=NEUTRAL)
    ax2.tick_params(axis="y", colors=NEUTRAL, labelsize=SMALL)
    ax.set_title("Relaxing exactness buys\nback likelihood and diversity")
    h, lab = ax.get_legend_handles_labels()
    h.append(mpl.lines.Line2D([], [], ls="--", marker="s", ms=3, color=NEUTRAL))
    lab.append("spread (right axis)")
    ax.legend(h, lab, loc="lower left", handletextpad=0.3, borderpad=0.2)
    ax.margins(0.08)
    letter(ax, "c")

    fig.tight_layout(w_pad=2.4)
    out = os.path.join(FIGURES, "charge_benchmark.png")
    fig.savefig(out)
    return fig, out


# --------------------------------------------------------------------------- #
# Figure 2: extinction coefficient
# --------------------------------------------------------------------------- #


def figure_extinction():
    df = pd.read_csv(os.path.join(RESULTS, "extinction.csv"))
    comp = pd.read_csv(os.path.join(RESULTS, "extinction_composition.csv"))
    order = ["unconstrained", "1490", "5500", "11000"]
    labels = {
        "unconstrained": "off",
        "1490": "1490\n(1 Tyr)",
        "5500": "5500\n(1 Trp)",
        "11000": "11000\n(2 Trp)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
    ax = axes[0]
    x = np.arange(len(order))
    trp = [df[df.threshold == t].n_trp.mean() for t in order]
    tyr = [df[df.threshold == t].n_tyr.mean() for t in order]
    ax.bar(x - 0.19, trp, 0.36, color=FOCAL, label="Trp", zorder=3)
    ax.bar(x + 0.19, tyr, 0.36, color=C_MODEL["soluble_mpnn"], label="Tyr", zorder=3)
    for xi, v in zip(x - 0.19, trp):
        if v < 0.02:
            ax.plot([xi], [0], marker="_", ms=9, color=FOCAL, zorder=4)
    ax.text(
        x[0] - 0.19,
        0.13,
        "0.00",
        fontsize=SMALL,
        color=FOCAL,
        ha="center",
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([labels[t] for t in order])
    ax.set_xlabel(r"--min_extinction_280 (M$^{-1}$cm$^{-1}$)")
    ax.set_ylabel("mean residues per design")
    ax.set_title("SolubleMPNN places no Trp here\nuntil the floor demands one")
    ax.legend(loc="upper left", handletextpad=0.4, borderpad=0.2)
    ax.grid(axis="y", zorder=0)
    ax.margins(y=0.16)
    letter(ax, "a")

    ax = axes[1]
    sub = comp[comp.threshold == "5500"].copy()
    sub["delta"] = sub.frequency - sub.baseline_frequency
    sub = sub.sort_values("delta")
    colours = [
        FOCAL if aa in ("W", "Y") else NEUTRAL for aa in sub.aa
    ]
    ax.barh(np.arange(len(sub)), sub.delta * 100, color=colours, zorder=3)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub.aa, fontsize=SMALL)
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_xlabel("change in frequency vs unconstrained (pp)")
    ax.set_title("At a 5500 floor, only Trp moves\nmuch; the rest is unchanged")
    ax.text(
        0.97,
        0.05,
        "Trp/Tyr highlighted",
        transform=ax.transAxes,
        fontsize=SMALL,
        color=FOCAL,
        ha="right",
    )
    ax.grid(axis="x", zorder=0)
    ax.margins(y=0.01)
    letter(ax, "b")

    ax = axes[2]
    nll = [df[df.threshold == t].nll.mean() for t in order]
    ax.plot(x, nll, "-o", ms=4.5, color=FOCAL, mew=0, zorder=3)
    ax.axhline(nll[0], color=REFERENCE, lw=1.0, ls="--", zorder=2)
    ax.text(
        x[-1],
        nll[0] - 0.012,
        "unconstrained",
        fontsize=SMALL,
        color=REFERENCE,
        ha="right",
        va="top",
    )
    for xi, v in zip(x, nll):
        ax.annotate(
            f"{v:.2f}",
            (xi, v),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=SMALL,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([labels[t] for t in order])
    ax.set_xlabel(r"--min_extinction_280 (M$^{-1}$cm$^{-1}$)")
    ax.set_ylabel("mean per-residue NLL")
    ax.set_title("A guaranteed chromophore costs\n~0.14 nats per residue")
    ax.text(
        0.04,
        0.97,
        "lower = better",
        transform=ax.transAxes,
        fontsize=SMALL,
        color=NEUTRAL,
        ha="left",
        va="top",
    )
    ax.margins(0.12)
    letter(ax, "c")

    fig.tight_layout(w_pad=2.4)
    out = os.path.join(FIGURES, "extinction_benchmark.png")
    fig.savefig(out)
    return fig, out


# --------------------------------------------------------------------------- #
# Figure 3: SPPS risk
# --------------------------------------------------------------------------- #


def figure_spps():
    df = pd.read_csv(os.path.join(RESULTS, "spps.csv"))
    comp = pd.read_csv(os.path.join(RESULTS, "spps_components.csv"))
    targets = list(dict.fromkeys(df.target))
    tcolour = {targets[0]: FOCAL, targets[1]: C_MODEL["soluble_mpnn"]}
    tlabel = {"1VII": "1VII, 36 aa", "1PGA": "1PGA, 56 aa"}

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
    ax = axes[0]
    strengths = sorted(df.spps_bias.unique())
    for ti, target in enumerate(targets):
        sub = df[df.target == target]
        for si, s in enumerate(strengths):
            vals = sub[sub.spps_bias == s].spps_risk.values
            pos = si + (ti - 0.5) * 0.3
            ax.plot(
                pos + np.random.default_rng(si).uniform(-0.055, 0.055, len(vals)),
                vals,
                ".",
                ms=2.6,
                color=tcolour[target],
                alpha=0.35,
                zorder=2,
            )
            ax.plot(
                [pos - 0.1, pos + 0.1],
                [np.median(vals)] * 2,
                lw=1.8,
                color=tcolour[target],
                zorder=4,
                solid_capstyle="butt",
            )
    ax.set_xticks(np.arange(len(strengths)))
    ax.set_xticklabels([f"{s:g}" for s in strengths])
    ax.set_xlabel("--spps_bias")
    ax.set_ylabel("SPPS risk score")
    ax.set_title("Risk falls steeply with bias\n(bars = median, n=64)")
    for target in targets:
        ax.plot([], [], "s", ms=5, color=tcolour[target], label=tlabel[target])
    ax.legend(loc="upper right", handletextpad=0.3, borderpad=0.2)
    ax.text(
        0.98,
        0.42,
        "lower = better",
        transform=ax.transAxes,
        fontsize=SMALL,
        color=NEUTRAL,
        ha="right",
    )
    ax.margins(0.06)
    letter(ax, "a")

    ax = axes[1]
    sub = comp[(comp.target == "1PGA")]
    keys = [
        ("asp_gly", "Asp-Gly motifs"),
        ("beta_pair", "adjacent Ile/Val/Thr pairs"),
        ("beta_run3", "Ile/Val/Thr runs " + r"$\geq$" + "3"),
    ]
    shades = [REFERENCE, FOCAL, "#6a4c93"]
    for (key, lab), colour in zip(keys, shades):
        s = sub[sub.component == key].sort_values("spps_bias")
        ax.plot(
            s.spps_bias,
            s.mean_count,
            "-o",
            ms=4,
            mew=0,
            color=colour,
            label=lab,
        )
        if key == "asp_gly":
            for xv, yv in zip(s.spps_bias, s.mean_count):
                if yv > 0:
                    ax.annotate(
                        f"{yv:.2f}",
                        (xv, yv),
                        textcoords="offset points",
                        xytext=(3, 7),
                        fontsize=SMALL,
                        color=colour,
                    )
    ax.set_xlabel("--spps_bias")
    ax.set_ylabel("mean count per design")
    ax.set_title("On 1PGA the aspartimide motif\nis gone by bias 1.0")
    ax.legend(loc="upper right", handletextpad=0.3, borderpad=0.2, labelspacing=0.3)
    ax.margins(0.08)
    letter(ax, "b")

    ax = axes[2]
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    for target in targets:
        sub = df[df.target == target].groupby("spps_bias")
        ax.plot(
            sub.nll.mean().index,
            sub.nll.mean().values,
            "-o",
            ms=4,
            mew=0,
            color=tcolour[target],
            label=tlabel[target],
        )
        ax2.plot(
            sub.seq_rec.mean().index,
            sub.seq_rec.mean().values,
            "--s",
            ms=3,
            mew=0,
            color=tcolour[target],
            alpha=0.55,
        )
    ax.set_xlabel("--spps_bias")
    ax.set_ylabel("mean per-residue NLL")
    ax2.set_ylabel("sequence recovery", color=NEUTRAL)
    ax2.tick_params(axis="y", colors=NEUTRAL, labelsize=SMALL)
    ax.set_title("The trade-off: likelihood and\nrecovery both decline")
    h, lab = ax.get_legend_handles_labels()
    h.append(mpl.lines.Line2D([], [], ls="--", marker="s", ms=3, color=NEUTRAL))
    lab.append("recovery (right axis)")
    ax.legend(h, lab, loc="center left", handletextpad=0.3, borderpad=0.2)
    ax.margins(0.08)
    letter(ax, "c")

    fig.tight_layout(w_pad=2.4)
    out = os.path.join(FIGURES, "spps_benchmark.png")
    fig.savefig(out)
    return fig, out


def main():
    os.makedirs(FIGURES, exist_ok=True)
    style()
    for fn in (figure_charge, figure_extinction, figure_spps):
        _, path = fn()
        print("wrote", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()

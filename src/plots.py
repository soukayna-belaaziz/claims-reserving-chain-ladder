"""Figures for the README (matplotlib, static PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs)
DIVERGING = LinearSegmentedColormap.from_list("bluegrayred", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#b8302f"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])


def set_style() -> None:
    """Recessive axes and grid, system sans-serif, light chart surface."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
            "lines.linewidth": 2,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def link_ratio_heatmaps(data: list[dict], path: Path) -> None:
    """Individual link ratios (annotated) coloured by Mack standardised residual."""
    fig, axes = plt.subplots(1, len(data), figsize=(6.2 * len(data), 4.6))
    for ax, d in zip(np.atleast_1d(axes), data):
        f, r, origins = d["factors"], d["residuals"], d["origins"]
        lim = np.nanmax(np.abs(r))
        im = ax.imshow(r, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                if not np.isnan(f[i, j]):
                    ax.text(j, i, f"{f[i, j]:.3f}", ha="center", va="center", fontsize=6.3, color=INK)
        ax.set_xticks(range(f.shape[1]), [f"{12 * (j + 1)}-{12 * (j + 2)}" for j in range(f.shape[1])], rotation=45)
        ax.set_yticks(range(len(origins)), origins)
        ax.set_title(d["title"], loc="left")
        ax.set_xlabel("Development age (months)")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("Mack standardised residual", fontsize=8, color=INK_2)
        cb.ax.tick_params(labelsize=7, colors=MUTED)
        cb.outline.set_visible(False)
    fig.suptitle("Paid age-to-age factors (text) and their standardised residuals (colour)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    _save(fig, path)


def reserve_error_bars(data: list[dict], path: Path) -> None:
    """Chain-ladder unpaid by accident year with +/- 1 and 2 Mack standard errors."""
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 3.8))
    for k, (ax, d) in enumerate(zip(np.atleast_1d(axes), data)):
        x, res, se = d["origins"], d["reserve"] / 1e3, d["se"] / 1e3
        ax.bar(x, res, color=SERIES[k], width=0.6, zorder=2)
        ax.errorbar(x, res, yerr=2 * se, fmt="none", ecolor=MUTED, elinewidth=1, capsize=0, zorder=3)
        ax.errorbar(x, res, yerr=se, fmt="none", ecolor=INK, elinewidth=2, capsize=0, zorder=4)
        ax.set_title(d["title"], loc="left")
        ax.set_ylabel("Unpaid, USD m")
        ax.set_xticks(x, [str(o) for o in x], rotation=45)
        ax.text(0.02, 0.95, f"Total {d['total'] / 1e3:,.1f}m, Mack s.e. {d['total_se'] / 1e3:,.1f}m "
                f"(CV {d['total_se'] / d['total']:.1%})", transform=ax.transAxes, fontsize=8, color=INK_2, va="top")
    fig.suptitle("Paid chain-ladder unpaid by accident year (incl. tail): thick bar = 1 s.e., thin = 2 s.e. (Mack 1993)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    _save(fig, path)


def bootstrap_histograms(data: list[dict], path: Path) -> None:
    """ODP bootstrap predictive distribution of total unpaid with mean / 75th / 99.5th percentiles."""
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 3.6))
    for k, (ax, d) in enumerate(zip(np.atleast_1d(axes), data)):
        s = d["samples"] / 1e3
        ax.hist(s, bins=70, color=SERIES[k], alpha=0.85, edgecolor=SURFACE, linewidth=0.3, zorder=2)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
        top = ax.get_ylim()[1]
        marks = [(s.mean(), "mean", "-", 0.97, "right"), (np.percentile(s, 75), "75th", "--", 0.97, "left"),
                 (np.percentile(s, 99.5), "99.5th", ":", 0.75, "left")]
        for val, lab, ls, h, side in marks:
            ax.axvline(val, color=INK, lw=1.2, ls=ls, zorder=3)
            pad = " " if side == "left" else ""
            ax.text(val, top * h, f"{pad}{lab}\n{pad}{val:,.1f}{'' if side == 'left' else ' '}", fontsize=7.5,
                    color=INK_2, va="top", ha=side, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0})
        ax.set_title(d["title"], loc="left")
        ax.set_xlabel("Total unpaid incl. tail, USD m")
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    fig.suptitle("ODP bootstrap (England & Verrall), 10,000 simulations, ultimate view", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    _save(fig, path)


def residual_panels(data: list[dict], path: Path) -> None:
    """Scaled Pearson residuals of the ODP model vs development, accident, calendar year and fitted value."""
    fig, axes = plt.subplots(len(data), 4, figsize=(15, 3.0 * len(data)), squeeze=False)
    for k, d in enumerate(data):
        r, m = d["residuals"], d["fitted"]
        i, j = np.indices(r.shape)
        ok = ~np.isnan(r) & ~np.isclose(r, 0.0)
        xs = [(j[ok] + 1) * 12, d["origins"][i[ok]], d["origins"][0] + i[ok] + j[ok], np.log(m[ok])]
        labels = ["Development age (months)", "Accident year", "Calendar year", "log fitted incremental"]
        for c, (x, lab) in enumerate(zip(xs, labels)):
            ax = axes[k, c]
            ax.axhline(0, color=AXIS, lw=1)
            ax.scatter(x, r[ok], s=14, color=SERIES[k], alpha=0.8, edgecolor=SURFACE, linewidth=0.5, zorder=3)
            if c < 3:
                ux = np.unique(x)
                means = [r[ok][x == u].mean() for u in ux]
                ax.plot(ux, means, color=INK, lw=1.2, zorder=4)
            ax.set_xlabel(lab)
            if c == 0:
                ax.set_ylabel(f"{d['title']}\nscaled residual", fontsize=8)
    fig.suptitle("ODP scaled Pearson residuals, paid triangles (black line = mean per period)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    _save(fig, path)


def backtest_figure(diag: list[dict], runoff: list[dict], path: Path) -> None:
    """(a) AvE on the held-out latest diagonal; (b) realised run-off vs predictive intervals."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    width = 0.27
    for k, d in enumerate(diag):
        x = np.arange(len(d["origins"])) + (k - 1) * width
        a.bar(x, 100 * d["ave_pct"], width=width, color=SERIES[k], zorder=2,
              label=f"{d['title']} (total {d['total_pct']:+.1f}%)")
    a.axhline(0, color=AXIS, lw=1)
    a.set_xticks(np.arange(len(diag[0]["origins"])), diag[0]["origins"], rotation=45)
    a.set_ylabel("Actual vs expected paid in CY 2007, % of expected")
    a.set_xlabel("Accident year")
    a.set_title("(a) Hold-out of the 2007 diagonal (in-sample back-test)", loc="left")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)

    for k, d in enumerate(runoff):
        y = len(runoff) - 1 - k
        lo, hi = d["boot_p0_5"], d["boot_p99_5"]
        b.plot([lo, hi], [y, y], color=SERIES[k], lw=8, alpha=0.35, solid_capstyle="butt", zorder=2)
        b.plot([d["mack_lo"], d["mack_hi"]], [y, y], color=SERIES[k], lw=2.5, zorder=3)
        b.scatter([100], [y], marker="|", s=250, color=INK, zorder=4)
        b.scatter([d["actual_pct"]], [y], marker="D", s=55, color=INK, zorder=5)
        b.text(d["actual_pct"], y + 0.22, f"{d['actual_pct']:.0f}%", ha="center", fontsize=8, color=INK)
    b.set_yticks(range(len(runoff)), [d["title"] for d in runoff][::-1], fontsize=8)
    b.set_xlabel("Paid emergence 2008-2016 to age 120, % of chain-ladder estimate")
    b.set_title("(b) Realised outcome (diamond) vs model ranges", loc="left")
    b.text(0.01, -0.28, "Wide band: bootstrap 0.5%-99.5%. Thin line: Mack +/- 2 s.e. Tick: chain-ladder estimate.",
           transform=b.transAxes, fontsize=7.5, color=INK_2)
    b.set_ylim(-0.6, len(runoff) - 0.4)
    fig.tight_layout()
    _save(fig, path)


def paid_incurred_figure(data: list[dict], path: Path) -> None:
    """Paid-to-incurred ratio development by accident year (upper triangle)."""
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 3.6), sharey=True)
    for k, (ax, d) in enumerate(zip(np.atleast_1d(axes), data)):
        ratio = d["ratio"]
        n = ratio.shape[0]
        shades = SEQUENTIAL(np.linspace(0.25, 1.0, n))
        for i in range(n):
            ax.plot(12 * np.arange(1, n + 1), ratio[i], color=shades[i], lw=1.4)
        ax.text(12 * (n - 1), ratio[0, -1], f"AY{d['origins'][0]}", fontsize=7, color=INK_2, ha="right", va="bottom")
        ax.text(12, ratio[-1, 0], f"AY{d['origins'][-1]}", fontsize=7, color=INK_2, ha="left", va="top")
        ax.set_title(d["title"], loc="left")
        ax.set_xlabel("Development age (months)")
    axes[0].set_ylabel("Paid / case incurred")
    fig.suptitle("Paid-to-incurred ratios by accident year (darker = more recent)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    _save(fig, path)


def one_year_vs_ultimate_figure(data: list[dict], path: Path) -> None:
    """Distribution of the one-year loss (-CDR) next to the ultimate deviation of the unpaid, both from the bootstrap."""
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 3.6))
    for ax, d in zip(np.atleast_1d(axes), data):
        ult, one = d["ultimate_dev"] / 1e3, d["one_year_loss"] / 1e3
        bins = np.linspace(min(ult.min(), one.min()), max(ult.max(), one.max()), 80)
        ax.hist(ult, bins=bins, histtype="stepfilled", color=SERIES[0], alpha=0.30, zorder=2)
        ax.hist(ult, bins=bins, histtype="step", color=SERIES[0], lw=1.5, zorder=3, label="Ultimate (run-off) view")
        ax.hist(one, bins=bins, histtype="stepfilled", color=SERIES[1], alpha=0.30, zorder=2)
        ax.hist(one, bins=bins, histtype="step", color=SERIES[1], lw=1.5, zorder=3, label="One-year view (-CDR)")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
        top = ax.get_ylim()[1]
        for val, color, lab, h in ((np.quantile(ult, 0.995), SERIES[0], "99.5%", 0.97), (np.quantile(one, 0.995), SERIES[1], "99.5%", 0.80)):
            ax.axvline(val, color=color, lw=1.2, ls=":", zorder=4)
            ax.text(val, top * h, f" {lab}: {val:,.1f}", fontsize=7.5, color=INK_2, va="top", ha="left",
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0}, zorder=5)
        ax.text(0.02, 0.97, f"MW s.e. {d['mw_se'] / 1e3:,.1f} vs Mack s.e. {d['mack_se'] / 1e3:,.1f}",
                transform=ax.transAxes, fontsize=7.5, color=INK_2, va="top")
        ax.set_title(d["title"], loc="left")
        ax.set_xlabel("Loss relative to best estimate, USD m (positive = adverse)")
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    np.atleast_1d(axes)[0].legend(loc="center left", fontsize=7.5)
    fig.suptitle("Paid chain ladder incl. tail: one-year CDR vs full run-off uncertainty (ODP bootstrap, 10,000 sims)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    _save(fig, path)


def loss_ratio_trend_figure(data: list[dict], path: Path) -> None:
    """Chain-ladder loss ratios by accident year, fitted trend and the BF priors; hold-out outcome for reference."""
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 3.8))
    for k, (ax, d) in enumerate(zip(np.atleast_1d(axes), data)):
        x = d["origins"]
        w = d["weights"] / d["weights"].max()
        used = d["used_in_fit"]
        ax.scatter(x[used], 100 * d["lr_cl"][used], s=20 + 90 * w[used], color=SERIES[0], edgecolor=SURFACE, lw=1.0,
                   zorder=4, label="Incurred CL loss ratio (size = used-up premium)")
        if (~used).any():
            ax.scatter(x[~used], 100 * d["lr_cl"][~used], s=40, facecolor="none", edgecolor=SERIES[0], lw=1.2, zorder=4,
                       label="Excluded from the fit")
        ax.plot(x, 100 * d["trend_line"], color=SERIES[0], lw=1.4, zorder=3,
                label="Fitted trend")
        ax.plot(x, 100 * np.full(len(x), d["elr_flat"]), color=MUTED, lw=1.4, ls="--", zorder=3, label="Original BF prior (flat)")
        ax.plot(x, 100 * d["elr_trended"], color=SERIES[1], lw=2.0, zorder=3, label="On-levelled BF prior")
        ax.scatter(x, 100 * d["lr_actual"], marker="D", s=22, color=INK, zorder=5,
                   label="Hold-out: case incurred LR at 120m (not used)")
        ax.set_title(f"{d['title']}: trend {100 * d['trend']:+.1f}% p.a. (t = {d['t']:.1f})", loc="left")
        ax.set_ylabel("Loss ratio, %")
        ax.set_xticks(x, [str(o) for o in x], rotation=45)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5)
    fig.suptitle("Loss-ratio trend estimated at 31.12.2007 and the on-levelled Bornhuetter-Ferguson prior",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save(fig, path)

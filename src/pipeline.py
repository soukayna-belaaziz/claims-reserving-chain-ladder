"""End-to-end reserving analysis. Run from the project root:  python -m src.pipeline

Valuation date 31.12.2007: only the upper triangles (calendar year <= 2007) are used to
estimate reserves. The lower triangles (calendar years 2008-2016) are held back and used
only at the end, to check the projections against what actually emerged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import plots
from .bootstrap import odp_bootstrap, summarise
from .data import PROJECT_ROOT, VALUATION_YEAR, Segment, load_segments, upper_triangle
from .deterministic import (
    apriori_elr_from_mature_years,
    bornhuetter_ferguson,
    cape_cod,
    chain_ladder,
    exponential_tail,
    volume_weighted_ldf,
)
from .diagnostics import calendar_year_test, development_correlation_test, diagonal_backtest, mack_standardised_residuals
from .mack import mack, mack_sigma2
from .triangle import link_ratios
from .validation import compare_with_package

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
TAB_DIR = PROJECT_ROOT / "reports" / "tables"
N_SIMS = 10_000
SEED = 2026
N_MATURE = 5  # accident years used for the BF a-priori loss ratio
N_BF_YEARS = 3  # most recent accident years where the selection switches to BF
TAIL_START_LAG = 2  # fit the exponential tail on factors from 36-48 months onwards


@dataclass(frozen=True)
class Tails:
    """Tail factor selection for one segment."""

    paid_curve: float
    incurred_curve: float
    incurred_to_paid_oldest: float
    paid: float
    incurred: float


def select_tails(tri_p: np.ndarray, tri_i: np.ndarray) -> Tails:
    """Incurred tail: exponential decay fit. Paid tail: at least (incurred/paid at the oldest age) x incurred tail.

    The exponential curve decays fast and tends to understate paid tails of long-tailed
    lines. At 120 months the oldest year still carries case reserves, so paid must at
    least grow to the incurred level (and the incurred tail on top of it).
    """
    tp, _, _ = exponential_tail(volume_weighted_ldf(tri_p), TAIL_START_LAG)
    ti, _, _ = exponential_tail(volume_weighted_ldf(tri_i), TAIL_START_LAG)
    ti = max(ti, 1.0)
    ratio = tri_i[0, -1] / tri_p[0, -1]
    return Tails(tp, ti, ratio, max(tp, ratio * ti), ti)


def method_ultimates(seg: Segment, tri_p: np.ndarray, tri_i: np.ndarray, tail_p: float, tail_i: float) -> tuple[pd.DataFrame, dict]:
    """Ultimates by accident year for chain ladder, BF and Cape Cod on paid and incurred data."""
    prem = seg.premium
    cp, ci = chain_ladder(tri_p, tail_p), chain_ladder(tri_i, tail_i)
    elr = apriori_elr_from_mature_years(ci.ultimate, prem, N_MATURE)
    cc_p, elr_cc_p = cape_cod(cp.latest, cp.cdf, prem)
    cc_i, elr_cc_i = cape_cod(ci.latest, ci.cdf, prem)
    df = pd.DataFrame(
        {
            "accident_year": seg.origins,
            "premium": prem,
            "paid_to_date": cp.latest,
            "incurred_to_date": ci.latest,
            "pct_paid": 1 / cp.cdf,
            "pct_reported": 1 / ci.cdf,
            "ult_paid_cl": cp.ultimate,
            "ult_incurred_cl": ci.ultimate,
            "ult_paid_bf": bornhuetter_ferguson(cp.latest, cp.cdf, prem, elr),
            "ult_incurred_bf": bornhuetter_ferguson(ci.latest, ci.cdf, prem, elr),
            "ult_paid_cc": cc_p,
            "ult_incurred_cc": cc_i,
        }
    )
    recent = df["accident_year"] > VALUATION_YEAR - N_BF_YEARS
    df["ult_selected"] = np.where(recent, df["ult_incurred_bf"], df["ult_incurred_cl"])
    return df, {"elr_apriori": elr, "elr_cape_cod_paid": elr_cc_p, "elr_cape_cod_incurred": elr_cc_i}


def analyse_segment(seg: Segment) -> dict:
    """Run every method and diagnostic for one segment."""
    tri_p, tri_i = upper_triangle(seg.paid), upper_triangle(seg.incurred)
    tails = select_tails(tri_p, tri_i)
    ults, elrs = method_ultimates(seg, tri_p, tri_i, tails.paid, tails.incurred)

    # Mack (no tail inside the triangle); with a constant tail: unpaid = tail * U - latest, s.e. scales by tail
    mp, mi = mack(tri_p), mack(tri_i)
    unpaid_cl = tails.paid * mp.ultimate - mp.latest

    boot = odp_bootstrap(tri_p, n_sims=N_SIMS, seed=SEED)
    latest = mp.latest
    unpaid_sims_tail = ((latest[None] + boot.reserves_by_origin) * tails.paid - latest[None]).sum(axis=1)

    by_ay = ults.copy()
    by_ay["unpaid_paid_cl"] = unpaid_cl
    by_ay["mack_se_paid"] = tails.paid * mp.se
    by_ay["mack_cv_paid"] = np.where(unpaid_cl > 0, by_ay["mack_se_paid"] / unpaid_cl, np.nan)
    by_ay["ibnr_incurred_cl"] = ults["ult_incurred_cl"] - ults["incurred_to_date"]
    by_ay["mack_se_incurred"] = tails.incurred * mi.se
    by_ay["boot_mean_unpaid"] = ((latest[None] + boot.reserves_by_origin) * tails.paid - latest[None]).mean(axis=0)
    by_ay["boot_std_unpaid"] = ((latest[None] + boot.reserves_by_origin) * tails.paid - latest[None]).std(axis=0, ddof=1)

    return {
        "seg": seg,
        "tri_p": tri_p,
        "tri_i": tri_i,
        "tails": tails,
        "by_ay": by_ay,
        "elrs": elrs,
        "mack_p": mp,
        "mack_i": mi,
        "boot": boot,
        "boot_unpaid_tail": unpaid_sims_tail,
        "tests": {
            "paid": (calendar_year_test(tri_p), development_correlation_test(tri_p)),
            "incurred": (calendar_year_test(tri_i), development_correlation_test(tri_i)),
        },
        "diag_bt": {"paid": diagonal_backtest(tri_p), "incurred": diagonal_backtest(tri_i)},
    }


def runoff_check(r: dict) -> pd.DataFrame:
    """Compare projections to age 120 (no tail) with the actual 2008-2016 emergence of the lower triangle."""
    seg, tri_p, tri_i = r["seg"], r["tri_p"], r["tri_i"]
    ults, _ = method_ultimates(seg, tri_p, tri_i, 1.0, 1.0)
    paid_latest, inc_latest = ults["paid_to_date"].sum(), ults["incurred_to_date"].sum()
    actual_paid = seg.paid[:, -1].sum() - paid_latest
    actual_inc = seg.incurred[:, -1].sum() - inc_latest
    boot_total = r["boot"].total
    rows = []
    for basis, actual, latest, se in [("paid", actual_paid, paid_latest, r["mack_p"].total_se),
                                      ("incurred", actual_inc, inc_latest, r["mack_i"].total_se)]:
        for meth in ("cl", "bf", "cc"):
            pred = ults[f"ult_{basis}_{meth}"].sum() - latest
            row = {"segment": seg.spec.key, "basis": basis, "method": meth, "predicted": pred, "actual": actual,
                   "actual_vs_predicted_pct": 100 * (actual / pred - 1)}
            if meth == "cl":
                row["mack_z"] = (actual - pred) / se
                if basis == "paid":
                    row["bootstrap_percentile"] = 100 * float((boot_total <= actual).mean())
            rows.append(row)
    sel_pred = (ults["ult_selected"].sum() - inc_latest)
    rows.append({"segment": seg.spec.key, "basis": "incurred", "method": "selected", "predicted": sel_pred,
                 "actual": actual_inc, "actual_vs_predicted_pct": 100 * (actual_inc / sel_pred - 1)})
    return pd.DataFrame(rows)


def write_tables(results: list[dict]) -> dict[str, pd.DataFrame]:
    """Write all result tables to reports/tables and return them."""
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}

    ldf_rows, tail_rows, summary_rows, boot_rows, test_rows, bt_rows, val_rows = [], [], [], [], [], [], []
    runoff = []
    for r in results:
        key, seg, t = r["seg"].spec.key, r["seg"], r["tails"]
        for basis, tri in (("paid", r["tri_p"]), ("incurred", r["tri_i"])):
            ldf = volume_weighted_ldf(tri)
            s2 = mack_sigma2(tri, ldf)
            for j, (f, s) in enumerate(zip(ldf, s2)):
                ldf_rows.append({"segment": key, "basis": basis, "age": f"{12 * (j + 1)}-{12 * (j + 2)}", "ldf": f, "sigma": np.sqrt(s)})
        tail_rows.append({"segment": key, "paid_exponential_curve": t.paid_curve, "incurred_exponential_curve": t.incurred_curve,
                          "incurred_to_paid_at_120m_oldest_ay": t.incurred_to_paid_oldest, "selected_paid_tail": t.paid,
                          "selected_incurred_tail": t.incurred})
        b = r["by_ay"]
        b.assign(segment=key).to_csv(TAB_DIR / f"by_accident_year_{key}.csv", index=False)
        paid_td = b["paid_to_date"].sum()
        summ = {"segment": key, "premium_1998_2007": b["premium"].sum(), "paid_to_date": paid_td,
                "case_reserves": b["incurred_to_date"].sum() - paid_td, **r["elrs"]}
        for col in ("paid_cl", "incurred_cl", "paid_bf", "incurred_bf", "paid_cc", "incurred_cc", "selected"):
            summ[f"unpaid_{col}"] = b[f"ult_{col}"].sum() - paid_td
        summ["mack_se_paid_cl"] = t.paid * r["mack_p"].total_se
        summ["mack_se_incurred_cl"] = t.incurred * r["mack_i"].total_se
        summ["mack_process_se_paid"] = t.paid * r["mack_p"].total_process_se
        summ["mack_parameter_se_paid"] = t.paid * r["mack_p"].total_parameter_se
        summary_rows.append(summ)

        s = summarise(r["boot_unpaid_tail"], (50, 75, 95, 99, 99.5))
        sims = r["boot_unpaid_tail"]
        s["es99"] = float(sims[sims >= np.percentile(sims, 99)].mean())
        s["p99.5_over_mean"] = s["p99.5"] / s["mean"]
        s["scale_phi"] = r["boot"].fit.scale_phi
        boot_rows.append({"segment": key, **s})

        for basis, (cy, dc) in r["tests"].items():
            test_rows.append({"segment": key, "basis": basis, "cy_Z": cy.z, "cy_E[Z]": cy.expected,
                              "cy_ci95_low": cy.ci[0], "cy_ci95_high": cy.ci[1], "cy_reject": cy.reject,
                              "dev_corr_T": dc.t, "dev_corr_ci50_low": dc.ci[0], "dev_corr_ci50_high": dc.ci[1],
                              "dev_corr_reject": dc.reject})
        for basis, bt in r["diag_bt"].items():
            for i, ay in enumerate(seg.origins[:-1]):
                bt_rows.append({"segment": key, "basis": basis, "accident_year": ay,
                                "actual": bt.actual_incremental[i], "expected": bt.expected_incremental[i],
                                "ave": bt.ave[i], "z": bt.z_score[i]})
        for basis, tri in (("paid", r["tri_p"]), ("incurred", r["tri_i"])):
            val_rows.append({"segment": key, "basis": basis,
                             **compare_with_package(tri, seg.origins, seg.premium, r["elrs"]["elr_apriori"])})
        runoff.append(runoff_check(r))

    out["development_factors"] = pd.DataFrame(ldf_rows)
    out["tail_selection"] = pd.DataFrame(tail_rows)
    out["reserve_summary"] = pd.DataFrame(summary_rows)
    out["bootstrap_summary"] = pd.DataFrame(boot_rows)
    out["assumption_tests"] = pd.DataFrame(test_rows)
    bt = pd.DataFrame(bt_rows)
    out["backtest_diagonal"] = bt
    out["backtest_diagonal_totals"] = (
        bt.dropna(subset=["expected"]).groupby(["segment", "basis"])[["actual", "expected"]].sum()
        .assign(ave_pct=lambda d: 100 * (d["actual"] / d["expected"] - 1)).reset_index()
    )
    out["runoff_validation"] = pd.concat(runoff, ignore_index=True)
    out["validation_vs_chainladder"] = pd.DataFrame(val_rows)
    for name, df in out.items():
        df.to_csv(TAB_DIR / f"{name}.csv", index=False, float_format="%.6g")
    return out


def make_figures(results: list[dict], tables: dict[str, pd.DataFrame]) -> None:
    """Create all figures under reports/figures."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plots.set_style()
    short = {"wkcomp": "Workers' comp", "comauto": "Commercial auto", "ppauto": "Private passenger auto"}
    plots.link_ratio_heatmaps(
        [{"title": short[r["seg"].spec.key], "factors": link_ratios(r["tri_p"]),
          "residuals": mack_standardised_residuals(r["tri_p"]), "origins": r["seg"].origins} for r in results],
        FIG_DIR / "link_ratio_heatmap.png",
    )
    plots.reserve_error_bars(
        [{"title": short[r["seg"].spec.key], "origins": r["seg"].origins,
          "reserve": r["by_ay"]["unpaid_paid_cl"].to_numpy(), "se": r["by_ay"]["mack_se_paid"].to_numpy(),
          "total": r["by_ay"]["unpaid_paid_cl"].sum(), "total_se": r["tails"].paid * r["mack_p"].total_se} for r in results],
        FIG_DIR / "reserve_mack_error_bars.png",
    )
    plots.bootstrap_histograms(
        [{"title": short[r["seg"].spec.key], "samples": r["boot_unpaid_tail"]} for r in results],
        FIG_DIR / "bootstrap_distribution.png",
    )
    plots.residual_panels(
        [{"title": short[r["seg"].spec.key], "origins": r["seg"].origins,
          "residuals": r["boot"].fit.unscaled_residuals / np.sqrt(r["boot"].fit.scale_phi),
          "fitted": r["boot"].fit.expected_incremental} for r in results],
        FIG_DIR / "residuals.png",
    )
    diag = []
    for r in results:
        bt = r["diag_bt"]["paid"]
        diag.append({"title": short[r["seg"].spec.key], "origins": r["seg"].origins[:-1],
                     "ave_pct": bt.ave / bt.expected_incremental,
                     "total_pct": 100 * (np.nansum(bt.actual_incremental[1:]) / np.nansum(bt.expected_incremental[1:]) - 1)})
    ro = tables["runoff_validation"]
    runoff = []
    for r in results:
        key = r["seg"].spec.key
        row = ro[(ro.segment == key) & (ro.basis == "paid") & (ro.method == "cl")].iloc[0]
        pred, se = row["predicted"], r["mack_p"].total_se
        sims = r["boot"].total
        runoff.append({"title": short[key], "actual_pct": 100 * row["actual"] / pred,
                       "boot_p0_5": 100 * np.percentile(sims, 0.5) / pred, "boot_p99_5": 100 * np.percentile(sims, 99.5) / pred,
                       "mack_lo": 100 * (pred - 2 * se) / pred, "mack_hi": 100 * (pred + 2 * se) / pred})
    plots.backtest_figure(diag, runoff, FIG_DIR / "backtest.png")
    plots.paid_incurred_figure(
        [{"title": short[r["seg"].spec.key], "origins": r["seg"].origins, "ratio": r["tri_p"] / r["tri_i"]} for r in results],
        FIG_DIR / "paid_to_incurred.png",
    )


def main() -> None:
    start = time.perf_counter()
    segments = load_segments()
    results = [analyse_segment(s) for s in segments.values()]
    tables = write_tables(results)
    make_figures(results, tables)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    for name in ("tail_selection", "reserve_summary", "bootstrap_summary", "assumption_tests",
                 "backtest_diagonal_totals", "runoff_validation", "validation_vs_chainladder"):
        print(f"\n== {name}")
        print(tables[name].round(4).to_string(index=False))
    print(f"\nPipeline finished in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()

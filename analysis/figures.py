"""
analysis/figures.py
===================
Post-processing and publication-quality figure generation..

Reads the JSON results files produced by src/experiment.py and generates
a comprehensive set of evaluation figures plus an extended statistics table.

Figures produced by generate_all_figures():
  fig_risk_comparison.pdf        -- Normalized Risk^ex per method (bar + CI)
  fig_resource.pdf               -- Avg CPU utilization bar chart
  fig_peak_vs_avg_util.pdf       -- Grouped bar: avg vs peak satellite utilization
  fig_migration_epochs.pdf       -- Avoidable migrations per epoch, per method
  fig_bound_tightness.pdf        -- Risk bound tightness (Risk^ex / Risk^UB)
  fig_runtime.pdf                -- Mean solver runtime per method
  fig_risk_epochs.pdf            -- Risk^ex vs. epoch, per method (line chart + CI)
  fig_runtime_epochs.pdf         -- Solver runtime vs. epoch, per method (line chart + CI)
  fig_risk_resource_tradeoff.pdf -- Risk vs. resource utilization scatter + Pareto front

Multi-scenario helpers (for scalability sweeps):
  plot_risk()       -- Risk vs users/slice line chart
  plot_scalability() -- Runtime vs users/slice line chart

All functions return gracefully when matplotlib is unavailable.
"""

from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.patches import Patch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not installed -- figures will not be generated.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

W_COL  = 3.45   # IEEE single-column width (inches)
W_FULL = 7.16   # IEEE double-column width (inches)

METHODS_ORDER = ["B1", "B2", "B3", "proposed_coarse", "proposed_exact"]

METHOD_STYLES: Dict[str, dict] = {
    "B1":              {"color": "#546e7a", "marker": "s", "ls": "--",
                        "label": "B1 Res-Min",       "hatch": "//"},
    "B2":              {"color": "#3366cc", "marker": "^", "ls": "--",
                        "label": "B2 Risk-Unaware",  "hatch": ".."},
    "B3":              {"color": "#e07b39", "marker": "D", "ls": "-",
                        "label": "B3 Greedy",        "hatch": "xx"},
    "proposed_coarse": {"color": "#2e7d32", "marker": "o", "ls": "-",
                        "label": "Proposed",         "hatch": ""},
    "proposed_exact":  {"color": "#c62828", "marker": "o", "ls": "None",
                        "label": "Proposed (exact)", "hatch": "++"},
}


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _setup_style() -> None:
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset":  "dejavuserif",
        "axes.linewidth":    0.7,
        "axes.labelsize":    10,
        "axes.titlesize":    10,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "legend.fontsize":   8,
        "legend.framealpha": 1.0,
        "legend.edgecolor":  "#999999",
        "grid.color":        "#cccccc",
        "grid.linewidth":    0.4,
        "lines.linewidth":   1.3,
        "lines.markersize":  4.5,
        "pdf.fonttype":      42,
    })


def _ci95(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return 1.96 * statistics.stdev(vals) / (len(vals) ** 0.5)


def _stats(vals: List[float]) -> Tuple[float, float, float]:
    """Return (mean, std, ci95). Safe for n=1."""
    if not vals:
        return float("nan"), 0.0, 0.0
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    ci = 1.96 * s / (len(vals) ** 0.5)
    return m, s, ci


def _save(fig: "plt.Figure", path: str) -> None:
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_path: str) -> dict:
    with open(results_path) as f:
        return json.load(f)


_NO_SOLUTION = {"infeasible", "timelimit_nofeas"}


def _group_by_method(records: List[dict]) -> Dict[str, List[dict]]:
    by_method: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        if r.get("status") not in _NO_SOLUTION:
            by_method[r["method"]].append(r)
    return by_method


# ---------------------------------------------------------------------------
# Figure 1: Risk comparison (normalized bar chart)
# ---------------------------------------------------------------------------

def plot_risk_bars(
    by_method: Dict[str, List[dict]],
    output_path: str,
    norm_base: Optional[float] = None,
) -> None:
    """
    Horizontal bar chart of mean normalized Risk^ex per method.

    norm_base: if None, uses B1 mean as normalization denominator.
    Error bars show 95% CI over instances x epochs.
    """
    if not HAS_MPL:
        return
    _setup_style()

    methods = [m for m in METHODS_ORDER if m in by_method]
    if not methods:
        return

    if norm_base is None:
        b1_risks = [r["risk_ex"] for r in by_method.get("B1", [])]
        norm_base = statistics.mean(b1_risks) if b1_risks else 1.0
    if norm_base < 1e-12:
        norm_base = 1.0

    means, cis, labels, colors = [], [], [], []
    for m in methods:
        vals = [r["risk_ex"] / norm_base for r in by_method[m]]
        mn, _, ci = _stats(vals)
        means.append(mn)
        cis.append(ci)
        labels.append(METHOD_STYLES[m]["label"])
        colors.append(METHOD_STYLES[m]["color"])

    fig, ax = plt.subplots(figsize=(W_COL, 1.6 + 0.35 * len(methods)))
    y = np.arange(len(methods))
    bars = ax.barh(y, means, xerr=cis, color=colors,
                   edgecolor="#333333", linewidth=0.6,
                   error_kw={"elinewidth": 0.8, "capsize": 3},
                   height=0.55, zorder=3)

    for bar, val, ci in zip(bars, means, cis):
        ax.text(val + ci + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Normalized cross-slice risk (B1 = 1.0)")
    ax.axvline(1.0, color="#999999", linestyle="--", linewidth=0.7)
    ax.grid(True, axis="x", zorder=0)
    ax.set_axisbelow(True)

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 2: Average CPU utilization (bar chart)
# ---------------------------------------------------------------------------

def plot_resource(
    method_utils: Dict[str, float],
    output_path: str,
) -> None:
    """Bar chart of average satellite CPU utilization per method."""
    if not HAS_MPL:
        return
    _setup_style()

    methods = [m for m in METHODS_ORDER if m in method_utils]
    if not methods:
        return

    labels = [METHOD_STYLES[m]["label"] for m in methods]
    colors = [METHOD_STYLES[m]["color"] for m in methods]
    values = [method_utils[m] for m in methods]

    fig, ax = plt.subplots(figsize=(W_COL, 1.90))
    bars = ax.bar(labels, values, color=colors, edgecolor="#333333",
                  linewidth=0.5, width=0.55, zorder=3)

    ymax = max(values) * 1.55
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + ymax * 0.03,
                f"{val:.2f}%", ha="center", va="bottom",
                fontsize=6.5, fontweight="bold")

    ax.set_ylabel("Avg. satellite CPU util. (%)")
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="lower"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=6.5)

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 3: Peak vs average satellite utilization (grouped bar)
# ---------------------------------------------------------------------------

def plot_peak_vs_avg_util(
    by_method: Dict[str, List[dict]],
    output_path: str,
) -> None:
    """
    Grouped bar chart: average and peak per-satellite CPU utilization per method.
    Reveals hot-spot behaviour not captured by the mean alone.
    """
    if not HAS_MPL:
        return
    _setup_style()

    methods = [m for m in METHODS_ORDER if m in by_method]
    if not methods:
        return

    avg_vals, peak_vals, labels = [], [], []
    for m in methods:
        recs = by_method[m]
        avg_vals.append(statistics.mean(r["cap_use_pct"]      for r in recs))
        peak_field = "peak_sat_util_pct"
        if any(peak_field in r for r in recs):
            peak_vals.append(statistics.mean(r[peak_field] for r in recs))
        else:
            peak_vals.append(float("nan"))
        labels.append(METHOD_STYLES[m]["label"])

    x = np.arange(len(methods))
    w = 0.35

    fig, ax = plt.subplots(figsize=(W_COL, 1.90))
    b1 = ax.bar(x - w / 2, avg_vals,  w, label="Avg util.",
                color=[METHOD_STYLES[m]["color"] for m in methods],
                edgecolor="#333", linewidth=0.5, zorder=3)
    b2 = ax.bar(x + w / 2, peak_vals, w, label="Peak util.",
                color=[METHOD_STYLES[m]["color"] for m in methods],
                edgecolor="#333", linewidth=0.5, alpha=0.45,
                hatch="//", zorder=3)

    valid_peaks = [v for v in peak_vals if not (isinstance(v, float) and v != v)]
    ymax = (max(valid_peaks) if valid_peaks else max(avg_vals)) * 1.35

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Satellite CPU util. (%)")
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="lower"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)

    for xi, val in zip(x - w / 2, avg_vals):
        ax.text(xi, val + ymax * 0.02, f"{val:.2f}%",
                ha="center", va="bottom", fontsize=5.5)
    for xi, val in zip(x + w / 2, peak_vals):
        if not (isinstance(val, float) and val != val):
            ax.text(xi, val + ymax * 0.02, f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=5.5)

    ax.legend(handles=[
        Patch(facecolor="#888", label="Avg util."),
        Patch(facecolor="#888", alpha=0.45, hatch="//", label="Peak util."),
    ], fontsize=6, loc="upper right")

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 4: Migration stability over epochs
# ---------------------------------------------------------------------------

def plot_migration_epochs(
    by_method: Dict[str, List[dict]],
    output_path: str,
) -> None:
    """
    Line chart of mean avoidable migrations per epoch per method.
    Aggregates across all instances for each epoch index.
    """
    if not HAS_MPL:
        return
    _setup_style()

    # group by (method, epoch) -> list of n_avoidable_mig values
    epoch_data: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for m, recs in by_method.items():
        for r in recs:
            epoch_data[m][r["epoch"]].append(r["n_avoidable_mig"])

    methods = [m for m in METHODS_ORDER if m in epoch_data]
    if not methods:
        return

    all_epochs = sorted({ep for m in methods for ep in epoch_data[m]})

    fig, ax = plt.subplots(figsize=(W_COL, 2.00))
    for m in methods:
        st = METHOD_STYLES[m]
        xs, ys, errs = [], [], []
        for ep in all_epochs:
            vals = epoch_data[m].get(ep, [])
            if vals:
                mn, _, ci = _stats(vals)
                xs.append(ep + 1)
                ys.append(mn)
                errs.append(ci)

        if xs:
            ax.errorbar(xs, ys, yerr=errs,
                        color=st["color"], marker=st["marker"],
                        linestyle=st["ls"], lw=st.get("lw", 1.3),
                        label=st["label"], capsize=2)

    tick_labels_mig = [e + 1 for e in all_epochs]
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avoidable migrations / epoch")
    ax.set_xlim(0.5, max(all_epochs) + 1.5)
    ax.set_xticks(tick_labels_mig)
    ax.set_xticklabels([str(t) if t % 2 == 1 else "" for t in tick_labels_mig],
                       fontsize=6.5)
    ax.grid(True)
    ax.legend(loc="upper right", ncol=1)

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 5: Delay compliance
# ---------------------------------------------------------------------------

def plot_delay_compliance(
    by_method: Dict[str, List[dict]],
    output_path: str,
) -> None:
    """
    Bar chart of mean E2E delay budget compliance (%) per method.
    Uses the delay_compliance_pct field; skips figure if field is absent.
    """
    if not HAS_MPL:
        return

    methods = [m for m in METHODS_ORDER if m in by_method]
    if not methods:
        return

    # Check field availability
    has_field = any(
        "delay_compliance_pct" in r
        for m in methods for r in by_method[m]
    )
    if not has_field:
        print("  [skip] delay_compliance_pct not in results -- re-run experiment to populate.")
        return

    _setup_style()
    means, cis, labels, colors = [], [], [], []
    for m in methods:
        vals = [r["delay_compliance_pct"] for r in by_method[m]
                if "delay_compliance_pct" in r]
        mn, _, ci = _stats(vals)
        means.append(mn)
        cis.append(ci)
        labels.append(METHOD_STYLES[m]["label"])
        colors.append(METHOD_STYLES[m]["color"])

    fig, ax = plt.subplots(figsize=(W_COL, 1.90))
    bars = ax.bar(labels, means, yerr=cis, color=colors,
                  edgecolor="#333333", linewidth=0.5, width=0.55,
                  error_kw={"elinewidth": 0.8, "capsize": 2}, zorder=3)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=6.5, fontweight="bold")

    ax.set_ylabel("Delay budget compliance (%)")
    ax.set_ylim(0, 115)
    ax.axhline(100, color="#999999", linestyle="--", linewidth=0.7)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=6.5)

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 6: Risk bound tightness
# ---------------------------------------------------------------------------

def plot_bound_tightness(
    by_method: Dict[str, List[dict]],
    output_path: str,
) -> None:
    """
    Grouped bar chart showing Risk^LB, Risk^ex, and Risk^UB per method.
    Validates Proposition 1 and reveals how tight the coarse approximation is.
    """
    if not HAS_MPL:
        return

    methods = [m for m in METHODS_ORDER if m in by_method]
    if not methods:
        return

    has_bounds = any(
        "risk_lb" in r and "risk_ub" in r
        for m in methods for r in by_method[m]
    )
    if not has_bounds:
        print("  [skip] risk_lb/risk_ub not in results.")
        return

    _setup_style()

    lb_means, ex_means, ub_means, labels = [], [], [], []
    for m in methods:
        recs = by_method[m]
        lb_means.append(statistics.mean(r["risk_lb"] for r in recs))
        ex_means.append(statistics.mean(r["risk_ex"] for r in recs))
        ub_means.append(statistics.mean(r["risk_ub"] for r in recs))
        labels.append(METHOD_STYLES[m]["label"])

    x = np.arange(len(methods))
    w = 0.25

    fig, ax = plt.subplots(figsize=(W_COL, 2.10))
    ax.bar(x - w, lb_means, w, label=r"Risk$^{LB}$",
           color="#90caf9", edgecolor="#333", linewidth=0.5, zorder=3)
    ax.bar(x,     ex_means, w, label=r"Risk$^{ex}$",
           color=[METHOD_STYLES[m]["color"] for m in methods],
           edgecolor="#333", linewidth=0.5, zorder=3)
    ax.bar(x + w, ub_means, w, label=r"Risk$^{UB}$",
           color="#ef9a9a", edgecolor="#333", linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Co-location risk")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)

    # Annotate where UB == EX (Proposition 1 validation)
    for i, (ex, ub) in enumerate(zip(ex_means, ub_means)):
        if abs(ex - ub) < 1e-6:
            ax.text(x[i] + w, ub + max(ub_means) * 0.02, "UB=EX",
                    ha="center", va="bottom", fontsize=5, color="#c62828")

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 7: Solver runtime
# ---------------------------------------------------------------------------

def plot_runtime_bars(
    by_method: Dict[str, List[dict]],
    output_path: str,
) -> None:
    """Bar chart of mean solver runtime (s) per method with 95% CI."""
    if not HAS_MPL:
        return
    _setup_style()

    methods = [m for m in METHODS_ORDER if m in by_method]
    if not methods:
        return

    # For proposed methods exclude epoch 0 (cold-start) so reported mean
    # matches the paper's warm-started per-epoch figure.
    _PROPOSED = {"proposed_coarse", "proposed_exact"}

    means, cis, labels, colors = [], [], [], []
    for m in methods:
        recs = by_method[m]
        if m in _PROPOSED:
            recs = [r for r in recs if r.get("epoch", 0) != 0]
        if not recs:
            recs = by_method[m]
        vals = [r["solve_time_s"] for r in recs]
        mn, _, ci = _stats(vals)
        means.append(mn)
        cis.append(ci)
        labels.append(METHOD_STYLES[m]["label"])
        colors.append(METHOD_STYLES[m]["color"])

    # Asymmetric error bars: lower bound clipped to avoid negative axis values
    yerr_lo = [min(ci, val) for val, ci in zip(means, cis)]
    yerr_hi = list(cis)

    fig, ax = plt.subplots(figsize=(W_COL, 1.90))
    ax.bar(labels, means,
           yerr=[yerr_lo, yerr_hi],
           color=colors,
           edgecolor="#333333", linewidth=0.5, width=0.55,
           error_kw={"elinewidth": 0.8, "capsize": 2}, zorder=3)

    for i, (val, hi) in enumerate(zip(means, yerr_hi)):
        ax.text(i, val + hi + max(means) * 0.02,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=6)

    ax.set_ylabel("Solver runtime (s)")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=6.5)

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Shared helper: group records by (method, epoch) for a scalar field
# ---------------------------------------------------------------------------

def _epoch_series(
    by_method: Dict[str, List[dict]],
    field: str,
) -> Tuple[List[str], List[int], Dict[str, Dict[int, List[float]]]]:
    """
    Return (present_methods, sorted_epochs, data[method][epoch] -> values).
    Only methods that have at least one record with `field` are included.
    """
    data: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for m, recs in by_method.items():
        for r in recs:
            ep = r.get("epoch")
            if ep is not None and field in r:
                data[m][ep].append(r[field])

    methods = [m for m in METHODS_ORDER if m in data]
    all_epochs = sorted({ep for m in methods for ep in data[m]})
    return methods, all_epochs, data


def _draw_epoch_lines(
    ax: "plt.Axes",
    methods: List[str],
    all_epochs: List[int],
    data: Dict[str, Dict[int, List[float]]],
) -> None:
    """Plot one line + CI band per method onto ax."""
    for m in methods:
        st = METHOD_STYLES[m]
        xs, ys, cis = [], [], []
        for ep in all_epochs:
            vals = data[m].get(ep, [])
            if vals:
                mn, _, ci = _stats(vals)
                xs.append(ep + 1)
                ys.append(mn)
                cis.append(ci)
        if not xs:
            continue
        ax.plot(xs, ys, color=st["color"], marker=st["marker"],
                linestyle=st["ls"], lw=1.3, label=st["label"])
        lo = [y - e for y, e in zip(ys, cis)]
        hi = [y + e for y, e in zip(ys, cis)]
        ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.15)


# ---------------------------------------------------------------------------
# Figure 8: Risk^ex vs. epoch
# ---------------------------------------------------------------------------

def plot_risk_epochs(
    by_method: Dict[str, List[dict]],
    output_path: str,
) -> None:
    """
    Line chart of mean Risk^ex per epoch per method.
    95% CI rendered as a shaded band around each line.
    Aggregates across all instances for each epoch index.
    """
    if not HAS_MPL:
        return

    methods, all_epochs, data = _epoch_series(by_method, "risk_ex")
    if not methods or not all_epochs:
        return

    _setup_style()
    fig, ax = plt.subplots(figsize=(W_COL, 2.10))
    _draw_epoch_lines(ax, methods, all_epochs, data)

    risk_ticks = [ep + 1 for ep in all_epochs]
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Risk$^{ex}$")
    ax.set_xlim(all_epochs[0] + 0.5, all_epochs[-1] + 1.5)
    ax.set_xticks(risk_ticks)
    ax.set_xticklabels([str(t) if t % 2 == 1 else "" for t in risk_ticks],
                       fontsize=6.5)
    ax.grid(True)
    ax.legend(loc="upper right", ncol=1, fontsize=6)

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 9: Solver runtime vs. epoch
# ---------------------------------------------------------------------------

def plot_runtime_epochs(
    by_method: Dict[str, List[dict]],
    output_path: str,
) -> None:
    """
    Line chart of mean solver runtime (s) per epoch per method.
    95% CI rendered as a shaded band around each line.
    Aggregates across all instances for each epoch index.
    """
    if not HAS_MPL:
        return

    methods, all_epochs, data = _epoch_series(by_method, "solve_time_s")
    if not methods or not all_epochs:
        return

    _setup_style()
    fig, ax = plt.subplots(figsize=(W_COL, 2.10))
    _draw_epoch_lines(ax, methods, all_epochs, data)

    rt_ticks = [ep + 1 for ep in all_epochs]
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Runtime (s)")
    ax.set_xlim(all_epochs[0] + 0.5, all_epochs[-1] + 1.5)
    ax.set_xticks(rt_ticks)
    ax.set_xticklabels([str(t) if t % 2 == 1 else "" for t in rt_ticks],
                       fontsize=6.5)
    ax.grid(True)
    ax.legend(loc="upper right", ncol=1, fontsize=6)

    # Annotate epoch 1 as cold-start for proposed methods
    if 0 in all_epochs:
        ax.annotate("cold\nstart", xy=(1, 0), xycoords=("data", "axes fraction"),
                    xytext=(1, -0.22), textcoords=("data", "axes fraction"),
                    ha="center", fontsize=7, color="#555555",
                    arrowprops=dict(arrowstyle="->", lw=0.5, color="#555555"))

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 10: Risk–resource utilization trade-off with Pareto front
# ---------------------------------------------------------------------------

def _pareto_front(
    points: List[Tuple[float, float, str]]
) -> List[Tuple[float, float, str]]:
    """
    Return the Pareto-optimal subset of (x, y, label) points.
    A point is dominated if another point has both x <= xi and y <= yi
    (with at least one strict).  Lower x and lower y are both preferred.
    """
    dominated = set()
    for i, (xi, yi, _) in enumerate(points):
        for j, (xj, yj, _) in enumerate(points):
            if i == j:
                continue
            if xj <= xi and yj <= yi and (xj < xi or yj < yi):
                dominated.add(i)
                break
    return [p for k, p in enumerate(points) if k not in dominated]


def plot_risk_resource_tradeoff(
    by_method: Dict[str, List[dict]],
    output_path: str,
) -> None:
    """
    Scatter plot of mean Risk^ex vs. mean satellite CPU utilization per method.

    Layout:
    - Semi-transparent per-record cloud in the background for each method.
    - Aggregated method means as solid markers with 95% CI error bars.
    - Non-dominated (Pareto-optimal) methods connected by a dashed line.
    - Dominated methods drawn with hollow markers to distinguish them.
    """
    if not HAS_MPL:
        return
    _setup_style()

    methods = [m for m in METHODS_ORDER if m in by_method]
    if not methods:
        return

    # Aggregated means and CIs
    agg: Dict[str, Tuple[float, float, float, float]] = {}  # method -> (x, y, ci_x, ci_y)
    for m in methods:
        recs = by_method[m]
        xs = [r["cap_use_pct"] for r in recs if "cap_use_pct" in r]
        ys = [r["risk_ex"]     for r in recs if "risk_ex"     in r]
        if not xs or not ys:
            continue
        mx, _, ci_x = _stats(xs)
        my, _, ci_y = _stats(ys)
        agg[m] = (mx, my, ci_x, ci_y)

    if not agg:
        return

    # Pareto analysis on aggregated means
    pts = [(v[0], v[1], m) for m, v in agg.items()]
    pareto_pts = _pareto_front(pts)
    pareto_methods = {p[2] for p in pareto_pts}

    fig, ax = plt.subplots(figsize=(W_COL, 2.30))

    # Background cloud
    for m in methods:
        if m not in agg:
            continue
        color = METHOD_STYLES[m]["color"]
        recs = by_method[m]
        xs = [r["cap_use_pct"] for r in recs if "cap_use_pct" in r]
        ys = [r["risk_ex"]     for r in recs if "risk_ex"     in r]
        if xs and ys:
            ax.scatter(xs, ys, color=color, s=8, alpha=0.25, zorder=1)

    # Method aggregated markers
    for m in methods:
        if m not in agg:
            continue
        st = METHOD_STYLES[m]
        mx, my, ci_x, ci_y = agg[m]
        on_pareto = m in pareto_methods

        ax.errorbar(
            mx, my,
            xerr=ci_x, yerr=ci_y,
            fmt=st["marker"],
            color=st["color"] if on_pareto else "none",
            markeredgecolor=st["color"],
            markeredgewidth=1.0,
            ecolor=st["color"],
            elinewidth=0.7,
            capsize=2,
            markersize=6,
            label=st["label"],
            zorder=3,
        )

    # Pareto front connecting line
    if len(pareto_pts) > 1:
        pareto_sorted = sorted(pareto_pts, key=lambda p: p[0])
        px = [p[0] for p in pareto_sorted]
        py = [p[1] for p in pareto_sorted]
        ax.plot(px, py, ls="--", lw=0.9, color="#555555",
                zorder=2, label="Pareto front")

    ax.set_xlabel("Avg. satellite CPU util. (%)")
    ax.set_ylabel(r"Cross-slice risk (Risk$^{ex}$)")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=6, ncol=1)

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Multi-scenario helpers (scalability sweeps)
# ---------------------------------------------------------------------------

def aggregate_by_method_and_users(
    records: List[dict],
    user_counts: List[int],
) -> Dict[str, Dict[int, dict]]:
    """Aggregate records by (method, users_per_slice) for scalability figures."""
    grouped: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for r in records:
        u = r.get("users_per_slice", user_counts[0] if user_counts else 40)
        grouped[(r["method"], u)].append(r)

    agg: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for (method, u), recs in grouped.items():
        valid = [r for r in recs if r.get("status") not in _NO_SOLUTION]
        if not valid:
            continue
        agg[method][u] = {
            "risk_ex":  _stats([r["risk_ex"]          for r in valid]),
            "cap_use":  _stats([r["cap_use"]           for r in valid]),
            "mig":      _stats([r["n_avoidable_mig"]   for r in valid]),
            "runtime":  _stats([r["solve_time_s"]      for r in valid]),
            "n_valid":  len(valid),
        }
    return agg


def plot_risk(
    data: Dict[str, List[Tuple[int, float, float]]],
    norm_base: float,
    output_path: str,
) -> None:
    """Line chart of normalized Risk^ex vs users/slice (multi-scenario)."""
    if not HAS_MPL:
        return
    _setup_style()
    fig, ax = plt.subplots(figsize=(W_COL, 2.10))

    for method, points in sorted(data.items()):
        if not points:
            continue
        st = METHOD_STYLES.get(method, {})
        xs   = [p[0] for p in points]
        ys   = [p[1] / norm_base for p in points]
        errs = [p[2] / norm_base for p in points]
        ax.errorbar(xs, ys, yerr=errs,
                    color=st.get("color", "gray"),
                    marker=st.get("marker", "o"),
                    linestyle=st.get("ls", "-"),
                    lw=st.get("lw", 1.3),
                    label=st.get("label", method),
                    capsize=2)

    ax.set_xlabel("Users per slice")
    ax.set_ylabel("Normalized cross-slice risk")
    ax.grid(True)
    ax.legend(loc="upper left")
    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_scalability(
    data: Dict[str, List[Tuple[int, float]]],
    output_path: str,
) -> None:
    """Solver runtime vs users/slice (multi-scenario)."""
    if not HAS_MPL:
        return
    _setup_style()
    fig, ax = plt.subplots(figsize=(W_COL, 2.10))

    for method, points in sorted(data.items()):
        if not points:
            continue
        st = METHOD_STYLES.get(method, {})
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys,
                color=st.get("color", "gray"),
                marker=st.get("marker", "o"),
                linestyle=st.get("ls", "-"),
                lw=st.get("lw", 1.3),
                label=st.get("label", method))

    time_limit = 300
    ax.axhline(time_limit, color="#e53935", linestyle="--", linewidth=0.8)
    ax.text(min(xs) if xs else 0, time_limit + 8,
            "time limit", fontsize=6, color="#e53935")
    ax.set_xlabel("Users per slice")
    ax.set_ylabel("Solver runtime (s)")
    ax.grid(True)
    ax.legend(loc="upper left")
    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Extended statistics table
# ---------------------------------------------------------------------------

def print_extended_statistics_table(records: List[dict]) -> None:
    """
    Print a comprehensive per-method statistics table covering all metrics.
    Columns: Risk^ex, Compliance%, AvgCPU%, ActiveCPU%, PeakCPU%, AvoidMig, Runtime(s), Gap(%), Bound-Tightness
    """
    by_method = _group_by_method(records)

    FIELDS = [
        ("risk_ex",              "Risk^ex",         ".4f"),
        ("delay_compliance_pct", "Comply%",          ".1f"),
        ("cap_use_pct",          "AvgCPU%",          ".1f"),
        ("active_sat_util_pct",  "ActiveCPU%",       ".1f"),
        ("peak_sat_util_pct",    "PeakCPU%",         ".1f"),
        ("max_isl_load",         "MaxISL",            ".1f"),
        ("avg_isl_load",         "AvgISL",            ".2f"),
        ("n_avoidable_mig",      "AvoidMig",         ".1f"),
        ("solve_time_s",         "Runtime(s)",       ".2f"),
        ("mip_gap_pct",          "Gap%",             ".2f"),
        ("risk_bound_tightness", "BoundTight",       ".3f"),
    ]

    col_w = 14
    header = f"{'Method':<20}" + "".join(f"{name:>{col_w}}" for _, name, _ in FIELDS)
    print(f"\n{'='*len(header)}")
    print(header)
    print("-" * len(header))

    for method in METHODS_ORDER:
        recs = by_method.get(method, [])
        if not recs:
            continue
        row = f"{method:<20}"
        for field, _, fmt in FIELDS:
            vals = [r[field] for r in recs if field in r]
            if not vals:
                row += f"{'N/A':>{col_w}}"
            else:
                m, s, _ = _stats(vals)
                cell = f"{m:{fmt}}±{s:{fmt}}"
                row += f"{cell:>{col_w}}"
        print(row)

    print("=" * len(header))
    print(f"  n = {len(records)} records  "
          f"({sum(1 for r in records if r.get('status')=='infeasible')} infeasible excluded)\n")


# ---------------------------------------------------------------------------
# Master automation function
# ---------------------------------------------------------------------------

def generate_all_figures(
    results_path: str,
    output_dir: str = "data/results",
    verbose: bool = True,
    methods: Optional[List[str]] = None,
) -> List[str]:
    """
    Load a results JSON and generate the complete evaluation figure set.

    Produces up to 10 PDF figures in output_dir and prints an extended stats
    table.  Returns a list of successfully written file paths.

    Works with both single-instance and multi-instance result files.
    Fields added after the prior session (delay_compliance_pct, peak_sat_util_pct,
    risk_bound_tightness) are used when present; figures that require absent
    fields are skipped with a diagnostic message.

    Parameters
    ----------
    methods : list of str, optional
        Subset of method names to include in every figure.  Accepted names:
        "B1", "B2", "B3", "proposed_coarse", "proposed_exact".
        If None (default) all methods present in the results file are shown.
    """
    if not HAS_MPL:
        print("[ERROR] matplotlib not installed -- cannot generate figures.")
        return []

    data    = load_results(results_path)
    records = data.get("records", [])
    sid     = data.get("scenario_id", "experiment")

    if verbose:
        n_inf = sum(1 for r in records if r.get("status") == "infeasible")
        print(f"\nLoaded {len(records)} records for '{sid}' "
              f"({n_inf} infeasible).")

    os.makedirs(output_dir, exist_ok=True)

    by_method = _group_by_method(records)
    if not by_method:
        print("[WARNING] No non-infeasible records -- nothing to plot.")
        return []

    if methods is not None:
        unknown = set(methods) - set(by_method)
        if unknown and verbose:
            print(f"[WARNING] Requested methods not found in results: {sorted(unknown)}")
        by_method = {m: v for m, v in by_method.items() if m in methods}
        if not by_method:
            print("[WARNING] No records remain after method filter -- nothing to plot.")
            return []

    generated: List[str] = []

    def _path(name: str) -> str:
        return os.path.join(output_dir, f"{sid}_{name}.pdf")

    # ── Figure 1: Risk comparison ──────────────────────────────────────────
    p = _path("fig_risk_comparison")
    plot_risk_bars(by_method, p)
    if os.path.exists(p):
        generated.append(p)

    # ── Figure 2: Average CPU utilization ─────────────────────────────────
    method_utils = {
        m: statistics.mean(r["cap_use_pct"] for r in recs)
        for m, recs in by_method.items()
    }
    p = _path("fig_resource")
    plot_resource(method_utils, p)
    if os.path.exists(p):
        generated.append(p)

    # ── Figure 3: Peak vs avg utilization ─────────────────────────────────
    p = _path("fig_peak_vs_avg_util")
    plot_peak_vs_avg_util(by_method, p)
    if os.path.exists(p):
        generated.append(p)

    # ── Figure 4: Migration over epochs ───────────────────────────────────
    p = _path("fig_migration_epochs")
    plot_migration_epochs(by_method, p)
    if os.path.exists(p):
        generated.append(p)

    # ── Figure 5: Delay compliance ────────────────────────────────────────
    p = _path("fig_delay_compliance")
    plot_delay_compliance(by_method, p)
    if os.path.exists(p):
        generated.append(p)

    # ── Figure 6: Risk bound tightness ────────────────────────────────────
    p = _path("fig_bound_tightness")
    plot_bound_tightness(by_method, p)
    if os.path.exists(p):
        generated.append(p)

    # ── Figure 7: Solver runtime ──────────────────────────────────────────
    p = _path("fig_runtime")
    plot_runtime_bars(by_method, p)
    if os.path.exists(p):
        generated.append(p)

    # ── Figure 8: Risk^ex vs. epochs ─────────────────────────────────────
    p = _path("fig_risk_epochs")
    plot_risk_epochs(by_method, p)
    if os.path.exists(p):
        generated.append(p)

    # ── Figure 9: Runtime vs. epochs ─────────────────────────────────────
    p = _path("fig_runtime_epochs")
    plot_runtime_epochs(by_method, p)
    if os.path.exists(p):
        generated.append(p)

    # ── Figure 10: Risk–resource trade-off scatter + Pareto front ────────
    p = _path("fig_risk_resource_tradeoff")
    plot_risk_resource_tradeoff(by_method, p)
    if os.path.exists(p):
        generated.append(p)

    # ── Extended stats table ──────────────────────────────────────────────
    print_extended_statistics_table(records)

    if verbose:
        print(f"\n{len(generated)} figure(s) written to {output_dir}/")

    return generated


# ---------------------------------------------------------------------------
# Legacy helper kept for backward compatibility
# ---------------------------------------------------------------------------

def print_statistics_table(records: List[dict]) -> None:
    """Condensed summary table (subset of print_extended_statistics_table)."""
    by_method = _group_by_method(records)

    header = f"{'Method':<20} {'Risk^ex':>10} {'Avoidable Mig':>15} {'Runtime(s)':>12} {'Gap(%)':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for method in METHODS_ORDER:
        recs = by_method.get(method, [])
        if not recs:
            continue

        def _fmt(vals: List[float]) -> str:
            if not vals:
                return "N/A"
            m, s, _ = _stats(vals)
            return f"{m:.3f}±{s:.3f}"

        risks = [r["risk_ex"]          for r in recs]
        migs  = [r["n_avoidable_mig"]  for r in recs]
        times = [r["solve_time_s"]     for r in recs]
        gaps  = [r["mip_gap_pct"]      for r in recs]
        print(f"{method:<20} {_fmt(risks):>10} {_fmt(migs):>15} {_fmt(times):>12} {_fmt(gaps):>8}")

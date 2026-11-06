"""
OpenStat Studio — Comprehensive Publication-Grade Statistical Plot Engine (18 Plot Types)
License: MIT
Supports full PlotStyle customization (palettes, themes, Korean CJK fonts, custom labels, dimensions, DPI, SVG/PDF vector export).
"""

import io
import base64
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch
import seaborn as sns
from scipy import stats

from .plot_style import (
    DEFAULT_FONT_STACK,
    get_active_style,
    apply_style_to_axes,
    finalize_and_serialize_figure,
)

# Default constants (dynamic helpers _pal() / _col() override per active PlotStyle)
JMV_BLUE = "#0D9488"
JMV_NAVY = "#1E3A5F"
JMV_ORANGE = "#D97706"
JMV_GREEN = "#059669"
JMV_RED = "#E11D48"
JMV_GRAY = "#64748B"
JMV_PALETTE = ["#0D9488", "#2563EB", "#D97706", "#E11D48", "#7C3AED", "#0284C7"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.9,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": DEFAULT_FONT_STACK,  # list form enables per-glyph fallback (e.g. em-dash, subscripts)
    "font.sans-serif": DEFAULT_FONT_STACK,
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
})


def _pal(n: Optional[int] = None) -> List[str]:
    cols = list(get_active_style()["colors"])
    if n is not None and n > 0:
        while len(cols) < n:
            cols.extend(cols)
        return cols[:n]
    return cols


def _col(idx: int = 0) -> str:
    cols = get_active_style()["colors"]
    return cols[idx % len(cols)]


def _hdr_col() -> str:
    return get_active_style()["heading_color"]


def fig_to_base64(fig: plt.Figure, dpi: int = 150) -> str:
    return finalize_and_serialize_figure(fig, default_dpi=dpi)


def _clean_ax(ax):
    apply_style_to_axes(ax)


def plot_descriptives_box(df: pd.DataFrame, var: str, group: Optional[str] = None, violin: bool = True) -> str:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    cols = [var] + ([group] if group and group in df.columns else [])
    clean = df[cols].dropna().copy()
    clean[var] = pd.to_numeric(clean[var], errors="coerce")
    clean = clean.dropna()

    if group and group in clean.columns and clean[group].nunique() <= 12:
        clean[group] = clean[group].astype(str)
        n_grp = clean[group].nunique()
        pal = _pal(n_grp)
        if violin:
            sns.violinplot(data=clean, x=group, y=var, hue=group, palette=pal,
                           inner=None, alpha=0.25, cut=0, legend=False, ax=ax)
        sns.boxplot(data=clean, x=group, y=var, hue=group, palette=pal,
                    width=0.26, boxprops=dict(alpha=0.85, edgecolor="#222"), legend=False, ax=ax)
        sns.stripplot(data=clean, x=group, y=var, color=_hdr_col(), size=4.5, jitter=0.14, alpha=0.55, ax=ax)
        means = clean.groupby(group, sort=False)[var].mean()
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        for idx, lbl in enumerate(tick_labels):
            if lbl in means.index:
                ax.scatter([idx], [means[lbl]], marker="s", s=45, color="#FFFFFF", edgecolor="#111111", zorder=10, label="Mean" if idx == 0 else "")
        ax.set_title(f"{var} by {group}", color=_hdr_col())
    else:
        if violin:
            sns.violinplot(y=clean[var], color=_col(0), inner=None, alpha=0.28, cut=0, ax=ax)
        sns.boxplot(y=clean[var], width=0.24, color=_col(0), boxprops=dict(alpha=0.82), ax=ax)
        sns.stripplot(y=clean[var], color=_hdr_col(), size=4.5, jitter=0.12, alpha=0.55, ax=ax)
        ax.scatter([0], [clean[var].mean()], marker="s", s=50, color="#FFFFFF", edgecolor="#111111", zorder=10)
        ax.set_title(f"Box & Violin Plot — {var}", color=_hdr_col())

    _clean_ax(ax)
    return fig_to_base64(fig)


def plot_descriptives_hist(df: pd.DataFrame, var: str, kde: bool = True) -> str:
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    clean = pd.to_numeric(df[var], errors="coerce").dropna()
    sns.histplot(clean, kde=kde, ax=ax, color=_col(0), edgecolor="white", alpha=0.72, bins="auto")
    ax.axvline(clean.mean(), color=_col(2), linestyle="--", linewidth=1.6, label=f"Mean = {clean.mean():.2f}")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"Histogram & Density — {var}", color=_hdr_col())
    ax.set_xlabel(var)
    ax.set_ylabel("Density / Count")
    _clean_ax(ax)
    return fig_to_base64(fig)


def plot_descriptives_bar(df: pd.DataFrame, var: str, group: Optional[str] = None) -> str:
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    if group and group in df.columns and pd.api.types.is_numeric_dtype(df[var]):
        clean = df[[var, group]].dropna()
        pal = _pal(clean[group].nunique())
        sns.barplot(data=clean, x=group, y=var, hue=group, palette=pal,
                    errorbar=("ci", 95), capsize=0.12, legend=False, ax=ax)
        ax.set_title(f"Bar Plot (Mean ± 95% CI) — {var} by {group}", color=_hdr_col())
    else:
        vc = df[var].dropna().astype(str).value_counts().head(15)
        pal = _pal(len(vc))
        sns.barplot(x=vc.index, y=vc.values, hue=vc.index, palette=pal, legend=False, ax=ax)
        ax.set_title(f"Frequency Bar Plot — {var}", color=_hdr_col())
        ax.set_ylabel("Counts")
        ax.tick_params(axis="x", rotation=20)
    _clean_ax(ax)
    return fig_to_base64(fig)


def plot_mean_ci(df: pd.DataFrame, dep: str, group: str, ci_level: float = 0.95) -> str:
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    clean = df[[dep, group]].dropna().copy()
    clean[dep] = pd.to_numeric(clean[dep], errors="coerce")
    clean = clean.dropna()

    agg = clean.groupby(group)[dep].agg(["mean", "median", "sem", "count"]).reset_index()
    alpha = 1.0 - ci_level
    agg["ci"] = agg.apply(lambda r: r["sem"] * stats.t.ppf(1 - alpha / 2, max(1, r["count"] - 1)), axis=1)

    x = np.arange(len(agg))
    ax.errorbar(x, agg["mean"], yerr=agg["ci"], fmt="o", color="#111111",
                ecolor=_col(0), elinewidth=2.0, capsize=6, capthick=1.6,
                markersize=7.5, markerfacecolor="white", markeredgewidth=1.8,
                label=f"Mean ({int(ci_level*100)}% CI)", zorder=5)
    ax.scatter(x + 0.06, agg["median"], marker="s", s=38, color=_col(2), zorder=6, label="Median")
    ax.set_xticks(x)
    ax.set_xticklabels(agg[group].astype(str))
    ax.set_xlabel(group)
    ax.set_ylabel(dep)
    ax.set_title(f"Descriptives Plot — {dep} by {group}", color=_hdr_col())
    ax.legend(frameon=False, fontsize=9)
    _clean_ax(ax)
    return fig_to_base64(fig)


def plot_interaction_emm(df: pd.DataFrame, dep: str, factor1: str, factor2: Optional[str] = None) -> str:
    if not factor2 or factor2 not in df.columns or factor1 == factor2:
        return plot_mean_ci(df, dep, factor1)

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    clean = df[[dep, factor1, factor2]].dropna().copy()
    clean[dep] = pd.to_numeric(clean[dep], errors="coerce")
    clean = clean.dropna()

    f1_levels = sorted(clean[factor1].unique(), key=lambda x: str(x))
    f2_levels = sorted(clean[factor2].unique(), key=lambda x: str(x))
    x = np.arange(len(f1_levels))

    for idx, lv2 in enumerate(f2_levels):
        sub = clean[clean[factor2] == lv2]
        means, cis = [], []
        for lv1 in f1_levels:
            vals = sub[sub[factor1] == lv1][dep]
            if len(vals) >= 2:
                m = vals.mean()
                ci = vals.sem() * stats.t.ppf(0.975, len(vals) - 1)
            elif len(vals) == 1:
                m, ci = vals.iloc[0], 0.0
            else:
                m, ci = np.nan, 0.0
            means.append(m)
            cis.append(ci)
        offset = (idx - (len(f2_levels) - 1) / 2) * 0.06
        color = _col(idx)
        ax.errorbar(x + offset, means, yerr=cis, fmt="-o", color=color, linewidth=1.8,
                    capsize=5, markersize=6.5, markerfacecolor="white", markeredgewidth=1.8,
                    label=f"{factor2}: {lv2}")

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in f1_levels])
    ax.set_xlabel(factor1)
    ax.set_ylabel(f"Estimated Marginal Means ({dep})")
    ax.set_title(f"Estimated Marginal Means — {factor1} * {factor2}", color=_hdr_col())
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
    _clean_ax(ax)
    return fig_to_base64(fig)


def plot_rm_profile(df: pd.DataFrame, rm_vars: List[str], between: Optional[str] = None) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    cols = rm_vars + ([between] if between and between in df.columns else [])
    clean = df[cols].dropna().copy()
    x = np.arange(len(rm_vars))

    if between and between in clean.columns:
        for idx, grp in enumerate(sorted(clean[between].unique(), key=str)):
            sub = clean[clean[between] == grp][rm_vars].apply(pd.to_numeric, errors="coerce")
            means = sub.mean().values
            cis = (sub.sem() * stats.t.ppf(0.975, max(1, len(sub) - 1))).values
            offset = (idx - 0.5) * 0.05
            color = _col(idx)
            ax.errorbar(x + offset, means, yerr=cis, fmt="-o", color=color, capsize=5,
                        linewidth=1.9, markersize=6.5, markerfacecolor="white", markeredgewidth=1.8,
                        label=f"{between}: {grp}")
        ax.legend(frameon=False, fontsize=9)
    else:
        sub = clean[rm_vars].apply(pd.to_numeric, errors="coerce")
        means = sub.mean().values
        cis = (sub.sem() * stats.t.ppf(0.975, max(1, len(sub) - 1))).values
        ax.errorbar(x, means, yerr=cis, fmt="-o", color=_col(0), capsize=5,
                    linewidth=2.0, markersize=7, markerfacecolor="white", markeredgewidth=1.8)

    ax.set_xticks(x)
    ax.set_xticklabels(rm_vars)
    ax.set_xlabel("Repeated Measures Factor")
    ax.set_ylabel("Mean (95% CI)")
    ax.set_title("Repeated Measures Profile Plot", color=_hdr_col())
    _clean_ax(ax)
    return fig_to_base64(fig)


def plot_qq(df: pd.DataFrame, var: str) -> str:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    clean = pd.to_numeric(df[var], errors="coerce").dropna()
    standardized = (clean - clean.mean()) / (clean.std(ddof=1) if clean.std(ddof=1) > 0 else 1.0)
    stats.probplot(standardized, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor("white")
    ax.get_lines()[0].set_markeredgecolor(_col(0))
    ax.get_lines()[0].set_markersize(5.5)
    ax.get_lines()[1].set_color("#333333")
    ax.get_lines()[1].set_linewidth(1.5)
    ax.set_title(f"Normal Q-Q Plot — {var}", color=_hdr_col())
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Standardized Residuals")
    _clean_ax(ax)
    return fig_to_base64(fig)


def plot_scatr(df: pd.DataFrame, x_var: str, y_var: str, group: Optional[str] = None,
               reg_line: str = "linear", marginal: str = "dens") -> str:
    """Scatterplot with regression line and marginal densities/boxplots."""
    cols = [x_var, y_var] + ([group] if group and group in df.columns else [])
    clean = df[cols].dropna().copy()
    clean[x_var] = pd.to_numeric(clean[x_var], errors="coerce")
    clean[y_var] = pd.to_numeric(clean[y_var], errors="coerce")
    clean = clean.dropna()

    fig = plt.figure(figsize=(6.5, 5.5))
    gs = fig.add_gridspec(4, 4, hspace=0.12, wspace=0.12)
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    if group and group in clean.columns and clean[group].nunique() <= 10:
        for idx, grp in enumerate(sorted(clean[group].unique(), key=str)):
            sub = clean[clean[group] == grp]
            color = _col(idx)
            ax_main.scatter(sub[x_var], sub[y_var], color=color, alpha=0.72, s=32, label=str(grp), edgecolors="white", linewidth=0.5)
            if reg_line == "linear" and len(sub) >= 3:
                sns.regplot(data=sub, x=x_var, y=y_var, scatter=False, ax=ax_main, color=color, line_kws={"linewidth": 1.6})
            if marginal == "dens" and len(sub) >= 3:
                sns.kdeplot(x=sub[x_var], ax=ax_top, color=color, fill=True, alpha=0.25)
                sns.kdeplot(y=sub[y_var], ax=ax_right, color=color, fill=True, alpha=0.25)
        ax_main.legend(title=group, frameon=True, fontsize=8.5)
    else:
        ax_main.scatter(clean[x_var], clean[y_var], color=_col(0), alpha=0.72, s=34, edgecolors="white", linewidth=0.5)
        if reg_line == "linear" and len(clean) >= 3:
            sns.regplot(data=clean, x=x_var, y=y_var, scatter=False, ax=ax_main, color=_col(1), line_kws={"linewidth": 1.8})
        elif reg_line == "smooth" and len(clean) >= 5:
            sns.regplot(data=clean, x=x_var, y=y_var, scatter=False, lowess=True, ax=ax_main, color=_col(2), line_kws={"linewidth": 1.8})
        if marginal == "dens" and len(clean) >= 3:
            sns.kdeplot(x=clean[x_var], ax=ax_top, color=_col(0), fill=True, alpha=0.3)
            sns.kdeplot(y=clean[y_var], ax=ax_right, color=_col(0), fill=True, alpha=0.3)
        elif marginal == "box":
            sns.boxplot(x=clean[x_var], ax=ax_top, color=_col(0), width=0.4)
            sns.boxplot(y=clean[y_var], ax=ax_right, color=_col(0), width=0.4)

    ax_top.axis("off")
    ax_right.axis("off")
    ax_main.set_xlabel(x_var)
    ax_main.set_ylabel(y_var)
    _clean_ax(ax_main)
    fig.suptitle(f"Scatterplot — {y_var} vs {x_var}", fontsize=11.5, fontweight="bold", color=_hdr_col(), y=0.98)
    return fig_to_base64(fig)


def plot_correlation_heatmap(df: pd.DataFrame, vars_list: List[str], method: str = "pearson") -> str:
    clean = df[vars_list].apply(pd.to_numeric, errors="coerce").dropna()
    corr = clean.corr(method=method)

    # Build annotation with significance stars
    annot = np.empty(corr.shape, dtype=object)
    for i in range(len(vars_list)):
        for j in range(len(vars_list)):
            if i == j:
                annot[i, j] = "1.00"
            else:
                r_v = corr.iloc[i, j]
                if method == "spearman":
                    _, p_v = stats.spearmanr(clean.iloc[:, i], clean.iloc[:, j])
                else:
                    _, p_v = stats.pearsonr(clean.iloc[:, i], clean.iloc[:, j])
                stars = "***" if p_v < 0.001 else ("**" if p_v < 0.01 else ("*" if p_v < 0.05 else ""))
                annot[i, j] = f"{r_v:.2f}{stars}"

    fig, ax = plt.subplots(figsize=(max(5.5, len(vars_list) * 1.05), max(4.5, len(vars_list) * 0.9)))
    cmap = sns.diverging_palette(245, 15, as_cmap=True)
    sns.heatmap(corr, annot=annot, fmt="", cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.8, ax=ax, cbar_kws={"shrink": 0.78})
    ax.set_title(f"Correlation Matrix ({method.title()})", color=JMV_NAVY, pad=12)
    return fig_to_base64(fig)


def plot_regression_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, cooks_d: Optional[np.ndarray] = None) -> str:
    """2x2 Regression Diagnostic Suite matching jamovi Linear Regression Assumption Plots."""
    resid = y_true - y_pred
    std_resid = (resid - np.mean(resid)) / (np.std(resid, ddof=1) if np.std(resid, ddof=1) > 0 else 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.8))

    # 1. Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(y_pred, resid, color=JMV_BLUE, alpha=0.7, edgecolors="white", s=32)
    ax.axhline(0, color="#333333", linestyle="--", linewidth=1.2)
    ax.set_title("Residuals vs Fitted", fontsize=10.5, color=JMV_NAVY)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    _clean_ax(ax)

    # 2. Normal Q-Q of Standardized Residuals
    ax = axes[0, 1]
    stats.probplot(std_resid, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor("white")
    ax.get_lines()[0].set_markeredgecolor(JMV_BLUE)
    ax.get_lines()[1].set_color(JMV_RED)
    ax.set_title("Q-Q Plot of Standardized Residuals", fontsize=10.5, color=JMV_NAVY)
    _clean_ax(ax)

    # 3. Scale-Location
    ax = axes[1, 0]
    sqrt_abs = np.sqrt(np.abs(std_resid))
    ax.scatter(y_pred, sqrt_abs, color=JMV_GREEN, alpha=0.7, edgecolors="white", s=32)
    ax.set_title("Scale-Location", fontsize=10.5, color=JMV_NAVY)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("√|Standardized Residuals|")
    _clean_ax(ax)

    # 4. Cook's Distance / Residual Histogram
    ax = axes[1, 1]
    if cooks_d is not None and len(cooks_d) == len(y_true):
        ax.stem(np.arange(len(cooks_d)), cooks_d, linefmt="-", markerfmt="o", basefmt=" ")
        plt.setp(ax.lines, color=JMV_BLUE, markersize=3.5)
        ax.axhline(4.0 / len(cooks_d), color=JMV_RED, linestyle="--", linewidth=1.1, label="4/n threshold")
        ax.set_title("Cook's Distance Influence", fontsize=10.5, color=JMV_NAVY)
        ax.set_xlabel("Observation")
        ax.set_ylabel("Cook's D")
        ax.legend(frameon=False, fontsize=8)
    else:
        sns.histplot(resid, kde=True, ax=ax, color=JMV_BLUE)
        ax.set_title("Residuals Distribution", fontsize=10.5, color=JMV_NAVY)
    _clean_ax(ax)

    fig.tight_layout()
    return fig_to_base64(fig)


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_val: float) -> str:
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot(fpr, tpr, color=_col(0), linewidth=2.2, label=f"ROC Curve (AUC = {auc_val:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.14, color=_col(0))
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1.2, label="Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title("ROC Curve — Predictive Discrimination", color=_hdr_col())
    ax.legend(loc="lower right", frameon=True, facecolor="white")
    _clean_ax(ax)
    return fig_to_base64(fig)


def plot_bayes_posterior(d_obs: float, se_d: float, bf10: float, prior_scale: float = 0.707) -> str:
    """JASP / jamovi jsq Prior & Posterior Plot with Bayes Factor Pizza Wheel & 95% Credible Interval."""
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    grid = np.linspace(-2.5, 2.5, 500)

    # Cauchy Prior
    prior_dens = stats.cauchy.pdf(grid, loc=0, scale=prior_scale)
    # Normal likelihood approximation on effect size delta * prior -> exact normalized posterior
    lik = stats.norm.pdf(grid, loc=d_obs, scale=max(se_d, 0.08))
    unnorm_post = prior_dens * lik
    post_dens = unnorm_post / np.trapezoid(unnorm_post, grid)

    cdf = np.cumsum(post_dens) * (grid[1] - grid[0])
    ci_low = grid[np.searchsorted(cdf, 0.025)]
    ci_high = grid[min(len(grid) - 1, np.searchsorted(cdf, 0.975))]
    median_d = grid[min(len(grid) - 1, np.searchsorted(cdf, 0.50))]

    ax.plot(grid, post_dens, color=JMV_NAVY, linewidth=2.2, label="Posterior")
    ax.plot(grid, prior_dens, color=JMV_GRAY, linestyle="--", linewidth=1.6, label=f"Cauchy Prior (r={prior_scale:.3f})")

    # Point at delta = 0 (Savage-Dickey density ratio illustration)
    prior_0 = stats.cauchy.pdf(0, loc=0, scale=prior_scale)
    post_0 = np.interp(0.0, grid, post_dens)
    ax.scatter([0, 0], [prior_0, post_0], s=48, color="white", edgecolor="#111111", linewidth=1.8, zorder=10)

    # 95% CI horizontal bar at top
    y_max = max(np.max(post_dens), np.max(prior_dens)) * 1.24
    ax.set_ylim(0, y_max * 1.22)
    ax.errorbar([median_d], [y_max * 1.03], xerr=[[median_d - ci_low], [ci_high - median_d]],
                fmt="o", color="#111111", capsize=4, linewidth=1.6, markersize=5)
    ax.text(median_d, y_max * 1.07, f"median = {median_d:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]",
            ha="center", va="bottom", fontsize=8.6, color="#222222")

    ax.set_xlabel("Effect Size δ")
    ax.set_ylabel("Density")
    ax.set_title(f"Prior and Posterior — BF₁₀ = {bf10:.3f} (BF₀₁ = {1/max(bf10, 1e-9):.3f})", color=JMV_NAVY)
    ax.legend(loc="center right", frameon=True, facecolor="white", framealpha=0.88, fontsize=8.6)
    _clean_ax(ax)

    # Inset Bayes Factor Wheel (Pizza Plot)
    ax_pie = fig.add_axes([0.14, 0.62, 0.18, 0.22])
    p_h1 = bf10 / (1.0 + bf10) if np.isfinite(bf10) else 1.0
    ax_pie.pie([p_h1, 1.0 - p_h1], colors=[JMV_RED, "#FFFFFF"], startangle=90,
               wedgeprops=dict(edgecolor="#222222", linewidth=1.1))
    ax_pie.set_title("data | H₁ vs H₀", fontsize=7.5, pad=2)

    return fig_to_base64(fig)


def plot_bayes_robustness(widths: List[float], bfs: List[float], user_r: float, user_bf: float) -> str:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    log_bfs = np.log10(np.maximum(bfs, 1e-6))
    ax.plot(widths, log_bfs, color=JMV_NAVY, linewidth=2.0)
    ax.axhline(0, color="#666666", linestyle=":", linewidth=1.1)
    ax.axhline(np.log10(3), color=JMV_GREEN, linestyle="--", alpha=0.5, linewidth=0.9)
    ax.axhline(np.log10(10), color=JMV_ORANGE, linestyle="--", alpha=0.5, linewidth=0.9)
    ax.scatter([user_r], [np.log10(max(user_bf, 1e-6))], s=65, color=JMV_RED, edgecolor="white", zorder=10,
               label=f"User Prior (r={user_r:.3f}, BF₁₀={user_bf:.3f})")
    ax.set_xlabel("Cauchy Prior Width (r)")
    ax.set_ylabel("log₁₀(BF₁₀)")
    ax.set_title("Bayes Factor Robustness Check", color=JMV_NAVY)
    ax.legend(frameon=False, fontsize=8.8)
    _clean_ax(ax)
    return fig_to_base64(fig)


def plot_scree(eigenvalues: np.ndarray, parallel_ev: Optional[np.ndarray] = None) -> str:
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    comps = np.arange(1, len(eigenvalues) + 1)
    ax.plot(comps, eigenvalues, "-o", color=JMV_BLUE, linewidth=2.0, markersize=6.5,
            markerfacecolor="white", markeredgewidth=1.8, label="Data Eigenvalues")
    if parallel_ev is not None and len(parallel_ev) == len(eigenvalues):
        ax.plot(comps, parallel_ev, "--^", color=JMV_ORANGE, linewidth=1.6, markersize=5.5,
                label="Simulated (Parallel Analysis 95th)")
    ax.axhline(1.0, color="#777777", linestyle=":", linewidth=1.2, label="Kaiser Criterion (EV = 1)")
    ax.set_xticks(comps)
    ax.set_xlabel("Component / Factor Number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Scree Plot", color=JMV_NAVY)
    ax.legend(frameon=False, fontsize=8.8)
    _clean_ax(ax)
    return fig_to_base64(fig)


def plot_sem_diagram(estimates_df: pd.DataFrame, title: str = "Structural Path Diagram") -> str:
    """Path / CFA diagram: latent variables as ellipses, observed variables as boxes.

    Rows use lavaan operators: ``F =~ x`` (loading, latent → indicator) and ``y ~ x`` (regression, x → y).
    Structural variables are laid out left → right by path depth; indicators of exogenous latents sit in a
    column on the left, indicators of the most endogenous latents on the right, others below their latent.
    Edge labels show the estimate (standardized when supplied) with significance stars."""
    rows = []
    for _, r in estimates_df.iterrows():
        op = str(r.get("op", ""))
        if op not in ("=~", "~"):
            continue
        est = r.get("Estimate", r.get("est", np.nan))
        pval = r.get("p-value", r.get("p", np.nan))
        lv, rv = str(r.get("lval", "")), str(r.get("rval", ""))
        if op == "=~":
            rows.append((lv, rv, "load", est, pval))       # latent -> indicator
        else:
            rows.append((rv, lv, "path", est, pval))       # predictor -> outcome
    if not rows:
        fig, ax = plt.subplots(figsize=(6.0, 2.0))
        ax.axis("off")
        ax.text(0.5, 0.5, "No paths to display", ha="center", va="center", transform=ax.transAxes)
        return fig_to_base64(fig)

    latents = list(dict.fromkeys(src for src, _, kind, _, _ in rows if kind == "load"))
    indicators: Dict[str, List[str]] = {L: [] for L in latents}
    for src, dst, kind, _, _ in rows:
        if kind == "load" and dst not in indicators[src]:
            indicators[src].append(dst)
    ind_of = {x: L for L, xs in indicators.items() for x in xs if x not in indicators}
    structural = [n for n in dict.fromkeys([a for a, *_ in rows] + [b for _, b, *_ in rows]) if n not in ind_of]
    paths = [(a, b) for a, b, kind, *_ in rows if kind == "path" and a in structural and b in structural]

    # path depth: exogenous variables at level 0
    level = {n: 0 for n in structural}
    for _ in range(len(structural)):
        changed = False
        for a, b in paths:
            if a != b and level[b] < level[a] + 1 <= len(structural):
                level[b] = level[a] + 1
                changed = True
        if not changed:
            break
    max_level = max(level.values()) if level else 0
    cols: Dict[int, List[str]] = {}
    for n in structural:
        cols.setdefault(level[n], []).append(n)

    STEP_IND, BOX_W, BOX_H, ELL_W, ELL_H = 0.46, 1.15, 0.34, 1.35, 0.62
    side = {}                                   # where the indicators of each latent go
    for L in latents:
        if L not in level:
            continue
        lev = level[L]
        side[L] = "left" if lev == 0 else ("right" if lev == max_level else "below")
    has_left = any(v == "left" for v in side.values())
    has_right = any(v == "right" for v in side.values()) and max_level > 0

    def slot_h(n):
        if side.get(n) in ("left", "right"):
            return max(0.95, len(indicators[n]) * STEP_IND + 0.2)
        if side.get(n) == "below":
            return 1.75
        return 0.95

    col_h = {lev: sum(slot_h(n) for n in ns) for lev, ns in cols.items()}
    H = min(14.0, max(3.6, max(col_h.values()) + 1.2))
    x0 = 1.0 + (2.4 if has_left else 0.0)
    k_below = max([len(indicators[L]) for L in side if side[L] == "below"] or [0])
    gap = max(2.5, k_below * (BOX_W + 0.12) / 2 + 1.4)
    W = min(14.0, x0 + max_level * gap + 1.0 + (2.4 if has_right else 0.0))
    pos: Dict[str, Tuple[float, float]] = {}
    top = (H - 0.5)
    for lev, ns in cols.items():
        x = x0 + lev * gap
        y = top - (top - 0.25 - col_h[lev]) / 2          # centre the column
        single_mid = (len(ns) == 1 and ns[0] not in indicators and 0 < lev < max_level
                      and any(level[a] < lev < level[b] for a, b in paths))
        for n in ns:
            h = slot_h(n)
            cy = y - h / 2 if side.get(n) != "below" else y - 0.45
            if single_mid:
                cy = min(top - 0.4, cy + 1.0)            # lift a lone mediator so the direct path stays visible
            pos[n] = (x, cy)
            y -= h
    for L, xs in indicators.items():
        if L not in pos:
            continue
        lx, ly = pos[L]
        k = len(xs)
        for i, x_name in enumerate(xs):
            if x_name in pos:
                continue
            if side.get(L) == "left":
                pos[x_name] = (0.85, ly + ((k - 1) / 2 - i) * STEP_IND)
            elif side.get(L) == "right":
                pos[x_name] = (W - 0.85, ly + ((k - 1) / 2 - i) * STEP_IND)
            else:
                pos[x_name] = (lx + (i - (k - 1) / 2) * (BOX_W + 0.12), ly - 1.0)

    is_lat = set(latents)

    # A straight path that would run through other variables is drawn as an arc above them.
    def arc_rad(src, dst):
        (x1, y1), (x2, y2) = pos[src], pos[dst]
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        if L2 == 0:
            return 0.0
        need = 0.0
        for n, (px, py) in pos.items():
            if n in (src, dst):
                continue
            t = ((px - x1) * dx + (py - y1) * dy) / L2
            if not 0.08 < t < 0.92:
                continue
            off = (py - (y1 + t * dy)) if abs(dx) >= abs(dy) else (px - (x1 + t * dx))
            half = (ELL_H if n in is_lat else BOX_H) / 2
            if abs(off) < half + 0.45:
                need = max(need, off + half + 0.45)        # clearance above (positive y) the variable
        if need <= 0:
            return 0.0
        # quadratic Bezier apex sits at half the control-point offset: offset = -rad * dx (for left-to-right)
        return -2.0 * need / dx if dx else 0.0

    rads = {}
    arc_top = 0.0
    for src, dst, kind, est, pval in rows:
        if kind == "path" and src in pos and dst in pos and src != dst:
            r_ = arc_rad(src, dst)
            if r_:
                rads[(src, dst)] = r_
                (x1, y1), (x2, y2) = pos[src], pos[dst]
                arc_top = max(arc_top, (y1 + y2) / 2 - 0.5 * r_ * (x2 - x1) + 0.3)
    H = max(H, arc_top + 0.7)
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def anchor(n, toward):
        cx, cy = pos[n]
        dx, dy = toward[0] - cx, toward[1] - cy
        if dx == 0 and dy == 0:
            return cx, cy
        if n in is_lat:
            a, b = ELL_W / 2, ELL_H / 2
            t = 1.0 / np.sqrt((dx / a) ** 2 + (dy / b) ** 2)
        else:
            hw, hh = BOX_W / 2, BOX_H / 2
            t = min(hw / abs(dx) if dx else np.inf, hh / abs(dy) if dy else np.inf)
        return cx + t * dx, cy + t * dy

    def stars(p):
        if p is None or pd.isna(p):
            return ""
        return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))

    for src, dst, kind, est, pval in rows:
        if src not in pos or dst not in pos or src == dst:
            continue
        p1 = anchor(src, pos[dst])
        p2 = anchor(dst, pos[src])
        st = stars(pval)
        color = JMV_NAVY if (st or pd.isna(pval)) else "#94A3B8"
        rad = rads.get((src, dst), 0.0)
        ax.annotate("", xy=p2, xytext=p1, zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, mutation_scale=11, connectionstyle=f"arc3,rad={rad}"))
        if est is None or pd.isna(est):
            continue
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        if rad:
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            mx, my = mx + 0.5 * rad * dy, my - 0.5 * rad * dx
        ax.text(mx, my, f"{float(est):.2f}{st}", fontsize=7.6, color=color, ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.9))

    for n, (nx, ny) in pos.items():
        if n in is_lat:
            ax.add_patch(Ellipse((nx, ny), ELL_W, ELL_H, facecolor="#E6F4F1", edgecolor=JMV_BLUE, lw=1.6, zorder=5))
        else:
            ax.add_patch(FancyBboxPatch((nx - BOX_W / 2, ny - BOX_H / 2), BOX_W, BOX_H, boxstyle="round,pad=0.02",
                                        facecolor="#FFFFFF", edgecolor="#334155", lw=1.2, zorder=5))
        lbl = n if len(n) <= 14 else n[:13] + "…"
        ax.text(nx, ny, lbl, ha="center", va="center", fontsize=8.2, fontweight="bold" if n in is_lat else "normal",
                color="#0F172A", zorder=6)

    ax.set_title(title, color=JMV_NAVY, fontsize=11, fontweight="bold")
    return fig_to_base64(fig)


def plot_moderation_slopes(x_grid: np.ndarray, slopes_dict: Dict[str, np.ndarray],
                           pred_name: str, mod_name: str, dep_name: str) -> str:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    styles = [("Average - 1SD", _col(0), "--"), ("Average", _col(1), "-"), ("Average + 1SD", _col(2), "-.")]
    for (lbl, y_vals), (_, col, ls) in zip(slopes_dict.items(), styles):
        ax.plot(x_grid, y_vals, linestyle=ls, color=col, linewidth=2.1, label=f"{mod_name}: {lbl}")
    ax.set_xlabel(pred_name)
    ax.set_ylabel(dep_name)
    ax.set_title(f"Simple Slope Plot — {pred_name} × {mod_name} on {dep_name}", color=_hdr_col())
    ax.legend(frameon=True, facecolor="white", fontsize=8.8)
    _clean_ax(ax)
    return fig_to_base64(fig)


def plot_panel_trends(df: pd.DataFrame, entity: str, time: str, dep: str) -> str:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    clean = df[[entity, time, dep]].dropna().copy()
    entities = sorted(clean[entity].unique(), key=str)[:10]
    for idx, ent in enumerate(entities):
        sub = clean[clean[entity] == ent].sort_values(time)
        ax.plot(sub[time], sub[dep], "-o", markersize=4, linewidth=1.5,
                color=_col(idx), alpha=0.82, label=str(ent))
    ax.set_xlabel(time)
    ax.set_ylabel(dep)
    ax.set_title(f"Panel Longitudinal Trajectories — {dep} across {time}", color=_hdr_col())
    ax.legend(title=entity, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
    _clean_ax(ax)
    return fig_to_base64(fig)


def plot_kaplan_meier(km_curves: List[Dict[str, Any]], time_col: str) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for idx, c in enumerate(km_curves):
        col = _col(idx)
        ax.step(c["timeline"], c["surv"], where="post", color=col, linewidth=2.0, label=c["label"])
        if "ci_low" in c and "ci_high" in c:
            ax.fill_between(c["timeline"], c["ci_low"], c["ci_high"], step="post", color=col, alpha=0.16)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel(f"Time ({time_col})")
    ax.set_ylabel("Survival Probability S(t)")
    ax.set_title("Kaplan-Meier Survival Curves (with 95% CI)", color=_hdr_col())
    ax.legend(frameon=False, fontsize=9)
    _clean_ax(ax)
    return fig_to_base64(fig)


# Backward-compatible alias
plot_regression_residuals = plot_regression_diagnostics


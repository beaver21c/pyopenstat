"""
OpenStat Studio — Comprehensive Standalone Graph Builder & Figure-to-Results Engine
License: MIT

Supports 15 standalone chart types independent of statistical hypothesis tests,
and outputs directly into the OpenStat Studio Analysis Results Sheet (with optional
summary data table, APA caption, Python `.py` syntax, and lossless `.svg`/`.pdf`/`.png` vectors):

1.  `bar`         : Bar / Grouped Bar Chart (mean, median, sum, count, pct + CI95/SE/SD error bars + data labels)
2.  `stacked_bar` : Stacked / 100% Proportional Stacked Bar Chart
3.  `line`        : Line / Multi-Group Time-Series Trend Chart (with CI95/SE/SD ribbon or error bars)
4.  `area`        : Area / Stacked Area Chart
5.  `pie`         : Pie Composition Chart (counts & percentages)
6.  `donut`       : Donut Composition Chart (with center total N)
7.  `histogram`   : Histogram + Kernel Density (KDE) + Normal Reference Curve
8.  `density`     : Multi-Group Overlapping Kernel Density Plot
9.  `box`         : Boxplot (with Mean ♦ marker & optional jittered points)
10. `violin`      : Violin + Inner Box Distribution Plot
11. `raincloud`   : Raincloud Plot (Half-KDE Cloud + Box + Jittered Raindrops)
12. `scatter`     : Scatter / Bubble Chart (Linear / Quadratic / Smooth LOESS + correlation r)
13. `pairplot`    : Multivariate Scatterplot Matrix (SPLOM)
14. `heatmap`     : Pivot Aggregation / Multi-Variable Correlation Heatmap
15. `errorbar`    : Mean & 95% Confidence Interval Forest / Dot-Whisker Chart
"""

from typing import Any, Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .base import BaseAnalysis, ResultsTable
from .plot_style import (
    apply_style_to_axes,
    finalize_and_serialize_figure,
    get_active_style,
    use_plot_style,
)

CHART_TYPES = (
    "bar",
    "stacked_bar",
    "line",
    "area",
    "pie",
    "donut",
    "histogram",
    "density",
    "box",
    "violin",
    "raincloud",
    "scatter",
    "pairplot",
    "heatmap",
    "errorbar",
)

CHART_TYPE_LABELS: Dict[str, str] = {
    "bar": "Bar Chart",
    "stacked_bar": "Stacked Bar Chart",
    "line": "Line & Trend Chart",
    "area": "Area Chart",
    "pie": "Pie Chart",
    "donut": "Donut Chart",
    "histogram": "Histogram & Density",
    "density": "Kernel Density Plot",
    "box": "Box Plot",
    "violin": "Violin Plot",
    "raincloud": "Raincloud Plot",
    "scatter": "Scatter / Bubble Plot",
    "pairplot": "Scatterplot Matrix (Pairplot)",
    "heatmap": "Heatmap",
    "errorbar": "Mean & Error-Bar Forest Plot",
}


def _agg_series(s: pd.Series, agg: str) -> float:
    clean = pd.to_numeric(s, errors="coerce").dropna()
    if len(clean) == 0:
        return np.nan
    a = agg.lower()
    if a == "sum":
        return float(clean.sum())
    if a == "median":
        return float(clean.median())
    if a == "count":
        return float(len(clean))
    return float(clean.mean())


def _err_val(s: pd.Series, err_mode: str) -> float:
    clean = pd.to_numeric(s, errors="coerce").dropna()
    n = len(clean)
    if n < 2 or err_mode == "none":
        return 0.0
    sd = float(clean.std(ddof=1))
    se = sd / np.sqrt(n)
    if err_mode == "sd":
        return sd
    if err_mode == "se":
        return se
    # ci95 default
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    return t_crit * se


class GraphBuilder(BaseAnalysis):
    """Standalone multi-purpose chart engine outputting directly to the Analysis Results Sheet."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        chart_type: str = "bar",
        x: Optional[str] = None,
        y: Optional[str] = None,
        vars: Optional[List[str]] = None,
        group: Optional[str] = None,
        size_var: Optional[str] = None,
        agg: str = "mean",
        error_bar: str = "ci95",
        horizontal: bool = False,
        stacked_100: bool = False,
        show_values: bool = False,
        jitter: bool = False,
        fit_line: str = "linear",
        bins: int = 18,
        show_summary_table: bool = True,
        caption: str = "",
        **kwargs,
    ):
        ctype = str(chart_type or "bar").strip().lower()
        if ctype not in CHART_TYPES:
            ctype = "bar"
        label = CHART_TYPE_LABELS.get(ctype, "Custom Graph")
        opts: Dict[str, Any] = {
            "chartType": ctype,
            "x": x,
            "y": y,
            "vars": vars or [],
            "group": group,
            "sizeVar": size_var,
            "agg": agg,
            "errorBar": error_bar,
            "horizontal": horizontal,
            "stacked100": stacked_100,
            "showValues": show_values,
            "jitter": jitter,
            "fitLine": fit_line,
            "bins": bins,
            "showSummaryTable": show_summary_table,
            "caption": caption,
        }
        opts.update(kwargs)
        super().__init__(name="graph", title=f"Graph Builder — {label}", dataset=dataset, options=opts)
        self.chart_type = ctype
        self.x = x if x in dataset.columns else None
        self.y = y if y in dataset.columns else None
        self.vars = [v for v in (vars or []) if v in dataset.columns]
        self.group = group if group in dataset.columns else None
        self.size_var = size_var if size_var in dataset.columns else None
        self.agg = str(agg or "mean").lower()
        self.error_bar = str(error_bar or "ci95").lower()
        self.horizontal = bool(horizontal)
        self.stacked_100 = bool(stacked_100)
        self.show_values = bool(show_values)
        self.jitter = bool(jitter)
        self.fit_line = str(fit_line or "linear").lower()
        self.bins = max(5, min(80, int(bins or 18)))
        self.show_summary_table = bool(show_summary_table)
        self.caption = str(caption or "").strip()

        self.syntax_python = (
            f"ost.graph(df, chart_type={self.chart_type!r}, x={self.x!r}, y={self.y!r}, "
            f"group={self.group!r}, agg={self.agg!r}, error_bar={self.error_bar!r})"
        )
        self.syntax_r = f"ggplot2::ggplot(data, aes(x={self.x or ''}, y={self.y or ''}, color={self.group or 'NULL'})) + geom_{self.chart_type}()"
        self.run()

    def _auto_pick_cols(self, df: pd.DataFrame):
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() <= 12]
        if not self.x:
            if self.chart_type in ("histogram", "density", "qq"):
                self.x = num_cols[0] if num_cols else df.columns[0]
            elif self.chart_type in ("scatter",):
                self.x = num_cols[0] if num_cols else df.columns[0]
            else:
                self.x = cat_cols[0] if cat_cols else df.columns[0]
        if not self.y and self.chart_type in ("bar", "line", "area", "box", "violin", "raincloud", "scatter", "errorbar"):
            rem_num = [c for c in num_cols if c != self.x]
            self.y = rem_num[0] if rem_num else (num_cols[0] if num_cols else None)
        if not self.vars and self.chart_type in ("pairplot", "heatmap"):
            self.vars = num_cols[:4] if len(num_cols) >= 2 else list(df.columns[:3])

    def run(self):
        df = self.dataset.copy()
        if df.empty:
            raise ValueError("Dataset is empty")
        self._auto_pick_cols(df)

        st = get_active_style()
        pal = st["colors"]
        ct = self.chart_type

        summary_tbl: Optional[ResultsTable] = None
        fig_title = st["title"] or f"{CHART_TYPE_LABELS.get(ct, 'Graph')}"

        if ct == "bar":
            fig, summary_tbl, fig_title = self._draw_bar(df, pal)
        elif ct == "stacked_bar":
            fig, summary_tbl, fig_title = self._draw_stacked_bar(df, pal)
        elif ct in ("line", "area"):
            fig, summary_tbl, fig_title = self._draw_line_or_area(df, pal, is_area=(ct == "area"))
        elif ct in ("pie", "donut"):
            fig, summary_tbl, fig_title = self._draw_pie_or_donut(df, pal, is_donut=(ct == "donut"))
        elif ct == "histogram":
            fig, summary_tbl, fig_title = self._draw_histogram(df, pal)
        elif ct == "density":
            fig, summary_tbl, fig_title = self._draw_density(df, pal)
        elif ct in ("box", "violin", "raincloud"):
            fig, summary_tbl, fig_title = self._draw_distribution(df, pal, mode=ct)
        elif ct == "scatter":
            fig, summary_tbl, fig_title = self._draw_scatter(df, pal)
        elif ct == "pairplot":
            fig, summary_tbl, fig_title = self._draw_pairplot(df, pal)
        elif ct == "heatmap":
            fig, summary_tbl, fig_title = self._draw_heatmap(df)
        elif ct == "errorbar":
            fig, summary_tbl, fig_title = self._draw_errorbar(df, pal)
        else:
            fig, summary_tbl, fig_title = self._draw_bar(df, pal)

        if self.show_summary_table and summary_tbl is not None and len(summary_tbl.rows) > 0:
            self.add_table(summary_tbl)

        png_src = finalize_and_serialize_figure(fig)
        self.add_plot(st["title"] or fig_title, png_src)
        self.apa_narrative = self.caption or f"{fig_title} generated and attached to Analysis Results."

    def _draw_bar(self, df: pd.DataFrame, pal: List[str]):
        x_col = self.x
        y_col = self.y
        grp_col = self.group if self.group and self.group != x_col else None
        fig, ax = plt.subplots(figsize=(6.4, 4.2))

        tbl = ResultsTable(name="graph_summary", title=f"Chart Data Summary ({x_col})")
        tbl.add_column("category", str(x_col), col_type="text")
        if grp_col:
            tbl.add_column("group", str(grp_col), col_type="text")
        tbl.add_column("n", "N", col_type="integer")
        tbl.add_column("value", f"{self.agg.upper()}({y_col or 'Count'})", col_type="number")
        if y_col and self.agg == "mean" and self.error_bar != "none":
            tbl.add_column("err", f"±{self.error_bar.upper()}", col_type="number")

        cats = [str(k) for k in df[x_col].dropna().unique()[:20]]
        if not grp_col:
            vals, errs = [], []
            for c in cats:
                sub = df[df[x_col].astype(str) == c]
                n_c = len(sub)
                if not y_col or self.agg == "count":
                    v = float(n_c)
                    e = 0.0
                elif self.agg == "pct":
                    v = 100.0 * n_c / max(1, len(df))
                    e = 0.0
                else:
                    v = _agg_series(sub[y_col], self.agg)
                    e = _err_val(sub[y_col], self.error_bar) if self.agg == "mean" else 0.0
                vals.append(0.0 if np.isnan(v) else v)
                errs.append(e)
                row = {"category": c, "n": n_c, "value": v}
                if "err" in [col["name"] for col in tbl.columns]:
                    row["err"] = e
                tbl.add_row(row)

            pos = np.arange(len(cats))
            bar_colors = [pal[i % len(pal)] for i in range(len(cats))]
            yerr = errs if any(e > 0 for e in errs) else None
            if self.horizontal:
                bars = ax.barh(pos, vals, xerr=yerr, color=bar_colors, alpha=0.88, edgecolor="#1E293B", linewidth=0.7, capsize=4)
                ax.set_yticks(pos)
                ax.set_yticklabels(cats)
                ax.set_xlabel(f"{self.agg.capitalize()} of {y_col}" if y_col and self.agg != "count" else "Count")
                ax.set_ylabel(str(x_col))
            else:
                bars = ax.bar(pos, vals, yerr=yerr, color=bar_colors, alpha=0.88, edgecolor="#1E293B", linewidth=0.7, capsize=4)
                ax.set_xticks(pos)
                ax.set_xticklabels(cats, rotation=20 if len(cats) > 5 else 0)
                ax.set_xlabel(str(x_col))
                ax.set_ylabel(f"{self.agg.capitalize()} of {y_col}" if y_col and self.agg != "count" else "Count")

            if self.show_values:
                for b, v in zip(bars, vals):
                    if self.horizontal:
                        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v:.2f}", va="center", fontsize=8.5)
                    else:
                        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
        else:
            grps = [str(g) for g in df[grp_col].dropna().unique()[:8]]
            pos = np.arange(len(cats))
            width = 0.8 / max(1, len(grps))
            for gi, g in enumerate(grps):
                vals, errs = [], []
                for c in cats:
                    sub = df[(df[x_col].astype(str) == c) & (df[grp_col].astype(str) == g)]
                    n_c = len(sub)
                    v = float(n_c) if (not y_col or self.agg == "count") else _agg_series(sub[y_col], self.agg)
                    e = _err_val(sub[y_col], self.error_bar) if (y_col and self.agg == "mean") else 0.0
                    vals.append(0.0 if np.isnan(v) else v)
                    errs.append(e)
                    row = {"category": c, "group": g, "n": n_c, "value": v}
                    if "err" in [col["name"] for col in tbl.columns]:
                        row["err"] = e
                    tbl.add_row(row)
                offset = (gi - (len(grps) - 1) / 2) * width
                yerr = errs if any(e > 0 for e in errs) else None
                if self.horizontal:
                    ax.barh(pos + offset, vals, height=width * 0.92, xerr=yerr, color=pal[gi % len(pal)], alpha=0.88, label=g, capsize=3)
                else:
                    ax.bar(pos + offset, vals, width=width * 0.92, yerr=yerr, color=pal[gi % len(pal)], alpha=0.88, label=g, capsize=3)

            if self.horizontal:
                ax.set_yticks(pos)
                ax.set_yticklabels(cats)
                ax.set_xlabel(f"{self.agg.capitalize()} of {y_col or 'Count'}")
                ax.set_ylabel(str(x_col))
            else:
                ax.set_xticks(pos)
                ax.set_xticklabels(cats, rotation=20 if len(cats) > 5 else 0)
                ax.set_xlabel(str(x_col))
                ax.set_ylabel(f"{self.agg.capitalize()} of {y_col or 'Count'}")
            ax.legend(title=str(grp_col), frameon=False)

        title = f"Bar Chart — {y_col or 'Count'} by {x_col}" + (f" ({grp_col})" if grp_col else "")
        ax.set_title(title, fontweight="bold")
        apply_style_to_axes(ax)
        fig.tight_layout()
        return fig, tbl, title

    def _draw_stacked_bar(self, df: pd.DataFrame, pal: List[str]):
        x_col = self.x
        grp_col = self.group or (self.y if self.y and self.y != x_col else x_col)
        ctab = pd.crosstab(df[x_col].astype(str), df[grp_col].astype(str))
        if self.stacked_100:
            plot_df = ctab.div(ctab.sum(axis=1).replace(0, 1), axis=0) * 100.0
        else:
            plot_df = ctab.astype(float)

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        bottom = np.zeros(len(plot_df))
        pos = np.arange(len(plot_df.index))

        tbl = ResultsTable(name="stacked_summary", title=f"Stacked Composition ({x_col} × {grp_col})")
        tbl.add_column("cat", str(x_col), col_type="text")
        for col_name in plot_df.columns:
            tbl.add_column(str(col_name), str(col_name), col_type="number", fmt="pc" if self.stacked_100 else None)

        for idx_r, cat_name in enumerate(plot_df.index):
            r_dict = {"cat": str(cat_name)}
            for col_name in plot_df.columns:
                r_dict[str(col_name)] = float(plot_df.loc[cat_name, col_name])
            tbl.add_row(r_dict)

        for ci, col_name in enumerate(plot_df.columns):
            vals = plot_df[col_name].values
            if self.horizontal:
                ax.barh(pos, vals, left=bottom, color=pal[ci % len(pal)], alpha=0.9, label=str(col_name), edgecolor="white", linewidth=0.6)
            else:
                ax.bar(pos, vals, bottom=bottom, color=pal[ci % len(pal)], alpha=0.9, label=str(col_name), edgecolor="white", linewidth=0.6)
            bottom += vals

        if self.horizontal:
            ax.set_yticks(pos)
            ax.set_yticklabels(plot_df.index)
            ax.set_xlabel("Percentage (%)" if self.stacked_100 else "Count")
            ax.set_ylabel(str(x_col))
        else:
            ax.set_xticks(pos)
            ax.set_xticklabels(plot_df.index, rotation=20 if len(pos) > 5 else 0)
            ax.set_xlabel(str(x_col))
            ax.set_ylabel("Percentage (%)" if self.stacked_100 else "Count")
        ax.legend(title=str(grp_col), bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
        title = f"{'100% Proportional ' if self.stacked_100 else ''}Stacked Bar — {x_col} × {grp_col}"
        ax.set_title(title, fontweight="bold")
        apply_style_to_axes(ax)
        fig.tight_layout()
        return fig, tbl, title

    def _draw_line_or_area(self, df: pd.DataFrame, pal: List[str], is_area: bool = False):
        x_col, y_col, grp_col = self.x, self.y, self.group
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        tbl = ResultsTable(name="trend_summary", title=f"{'Area' if is_area else 'Line'} Series Summary")
        tbl.add_column("x", str(x_col), col_type="text")
        if grp_col:
            tbl.add_column("group", str(grp_col), col_type="text")
        tbl.add_column("val", f"{self.agg.upper()}({y_col})", col_type="number")

        sub_cols = [c for c in [x_col, y_col, grp_col] if c]
        clean = df[sub_cols].dropna().copy()
        x_vals = list(dict.fromkeys(clean[x_col].tolist()))
        try:
            x_vals = sorted(x_vals)
        except Exception:
            pass
        x_pos = np.arange(len(x_vals))
        x_labels = [str(v) for v in x_vals]

        groups = [str(g) for g in clean[grp_col].unique()[:8]] if grp_col else [None]
        stack_base = np.zeros(len(x_vals))

        for gi, g in enumerate(groups):
            gdf = clean[clean[grp_col].astype(str) == g] if g is not None else clean
            ys, errs = [], []
            for xv in x_vals:
                cell = gdf[gdf[x_col] == xv]
                v = _agg_series(cell[y_col], self.agg) if y_col else float(len(cell))
                e = _err_val(cell[y_col], self.error_bar) if (y_col and self.agg == "mean" and not is_area) else 0.0
                ys.append(0.0 if np.isnan(v) else v)
                errs.append(e)
                r_d = {"x": str(xv), "val": v}
                if g is not None:
                    r_d["group"] = g
                tbl.add_row(r_d)

            ys_arr = np.array(ys, dtype=float)
            errs_arr = np.array(errs, dtype=float)
            col = pal[gi % len(pal)]
            if is_area:
                ax.fill_between(x_pos, stack_base, stack_base + ys_arr, color=col, alpha=0.55, label=g or str(y_col))
                ax.plot(x_pos, stack_base + ys_arr, color=col, linewidth=1.8)
                stack_base += ys_arr
            else:
                ax.plot(x_pos, ys_arr, marker="o", linewidth=2.1, markersize=5.5, color=col, label=g or str(y_col))
                if np.any(errs_arr > 0):
                    ax.fill_between(x_pos, ys_arr - errs_arr, ys_arr + errs_arr, color=col, alpha=0.16)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=25 if len(x_labels) > 7 else 0)
        ax.set_xlabel(str(x_col))
        ax.set_ylabel(f"{self.agg.capitalize()} of {y_col}")
        if grp_col:
            ax.legend(title=str(grp_col), frameon=False)
        title = f"{'Area' if is_area else 'Line'} Chart — {y_col} across {x_col}"
        ax.set_title(title, fontweight="bold")
        apply_style_to_axes(ax)
        fig.tight_layout()
        return fig, tbl, title

    def _draw_pie_or_donut(self, df: pd.DataFrame, pal: List[str], is_donut: bool = False):
        x_col = self.x
        if self.y and self.agg in ("sum", "mean"):
            vc = df.groupby(df[x_col].astype(str))[self.y].apply(lambda s: _agg_series(s, self.agg)).dropna()
            vc = vc[vc > 0].sort_values(ascending=False).head(10)
        else:
            vc = df[x_col].dropna().astype(str).value_counts().head(10)

        fig, ax = plt.subplots(figsize=(5.8, 4.5))
        total = float(vc.sum()) if len(vc) else 1.0
        colors = [pal[i % len(pal)] for i in range(len(vc))]
        wedge_props = {"width": 0.46, "edgecolor": "white", "linewidth": 1.6} if is_donut else {"edgecolor": "white", "linewidth": 1.4}

        ax.pie(
            vc.values,
            labels=[f"{k} ({v / total * 100:.1f}%)" for k, v in vc.items()],
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops=wedge_props,
            textprops={"fontsize": 9.5},
        )
        if is_donut:
            ax.text(0, 0, f"Total\n{int(round(total))}", ha="center", va="center", fontsize=11, fontweight="bold", color="#1E293B")

        tbl = ResultsTable(name="pie_summary", title=f"Category Proportions ({x_col})")
        tbl.add_column("category", str(x_col), col_type="text")
        tbl.add_column("value", "Value / Count", col_type="number")
        tbl.add_column("pct", "Share (%)", col_type="number", fmt="pc")
        for k, v in vc.items():
            tbl.add_row({"category": str(k), "value": float(v), "pct": float(v) / total * 100.0})

        title = f"{'Donut' if is_donut else 'Pie'} Chart — {x_col}"
        ax.set_title(title, fontweight="bold")
        fig.tight_layout()
        return fig, tbl, title

    def _draw_histogram(self, df: pd.DataFrame, pal: List[str]):
        x_col = self.x
        grp_col = self.group
        fig, ax = plt.subplots(figsize=(6.3, 4.1))
        tbl = ResultsTable(name="hist_summary", title=f"Distribution Summary ({x_col})")
        tbl.add_column("group", "Group", col_type="text")
        tbl.add_column("n", "N", col_type="integer")
        tbl.add_column("mean", "Mean", col_type="number")
        tbl.add_column("sd", "SD", col_type="number")

        if grp_col:
            for gi, (g, sub) in enumerate(df.groupby(grp_col)):
                vals = pd.to_numeric(sub[x_col], errors="coerce").dropna()
                if len(vals) < 2:
                    continue
                sns.histplot(vals, bins=self.bins, kde=True, stat="density", color=pal[gi % len(pal)], alpha=0.42, label=str(g), ax=ax)
                tbl.add_row({"group": str(g), "n": len(vals), "mean": float(vals.mean()), "sd": float(vals.std(ddof=1))})
            ax.legend(title=str(grp_col), frameon=False)
        else:
            vals = pd.to_numeric(df[x_col], errors="coerce").dropna()
            sns.histplot(vals, bins=self.bins, kde=True, stat="density", color=pal[0], alpha=0.58, edgecolor="white", ax=ax)
            if len(vals) >= 3 and vals.std(ddof=1) > 0:
                xs = np.linspace(vals.min(), vals.max(), 150)
                ax.plot(xs, stats.norm.pdf(xs, vals.mean(), vals.std(ddof=1)), linestyle="--", color=pal[1 % len(pal)], linewidth=1.6, label="Normal PDF")
                ax.legend(frameon=False)
            tbl.add_row({"group": "All", "n": len(vals), "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0})

        title = f"Histogram & Density — {x_col}"
        ax.set_xlabel(str(x_col))
        ax.set_ylabel("Density")
        ax.set_title(title, fontweight="bold")
        apply_style_to_axes(ax)
        fig.tight_layout()
        return fig, tbl, title

    def _draw_density(self, df: pd.DataFrame, pal: List[str]):
        x_col, grp_col = self.x, self.group
        fig, ax = plt.subplots(figsize=(6.3, 4.0))
        tbl = ResultsTable(name="density_summary", title=f"Kernel Density Summary ({x_col})")
        tbl.add_column("group", "Group", col_type="text")
        tbl.add_column("n", "N", col_type="integer")
        tbl.add_column("median", "Median", col_type="number")
        tbl.add_column("iqr", "IQR", col_type="number")

        if grp_col:
            for gi, (g, sub) in enumerate(df.groupby(grp_col)):
                vals = pd.to_numeric(sub[x_col], errors="coerce").dropna()
                if len(vals) >= 3 and vals.std(ddof=1) > 0:
                    sns.kdeplot(vals, fill=True, alpha=0.35, linewidth=2.0, color=pal[gi % len(pal)], label=str(g), ax=ax)
                    tbl.add_row({"group": str(g), "n": len(vals), "median": float(vals.median()), "iqr": float(vals.quantile(0.75) - vals.quantile(0.25))})
            ax.legend(title=str(grp_col), frameon=False)
        else:
            vals = pd.to_numeric(df[x_col], errors="coerce").dropna()
            if len(vals) >= 3:
                sns.kdeplot(vals, fill=True, alpha=0.42, linewidth=2.2, color=pal[0], ax=ax)
                tbl.add_row({"group": "All", "n": len(vals), "median": float(vals.median()), "iqr": float(vals.quantile(0.75) - vals.quantile(0.25))})

        title = f"Kernel Density Distribution — {x_col}"
        ax.set_xlabel(str(x_col))
        ax.set_ylabel("Density")
        ax.set_title(title, fontweight="bold")
        apply_style_to_axes(ax)
        fig.tight_layout()
        return fig, tbl, title

    def _draw_distribution(self, df: pd.DataFrame, pal: List[str], mode: str = "box"):
        # Supports box, violin, raincloud
        cat_col = self.x if (self.y and self.x != self.y) else self.group
        num_col = self.y or self.x
        fig, ax = plt.subplots(figsize=(6.5, 4.3))

        tbl = ResultsTable(name="dist_summary", title=f"{mode.capitalize()} Quartiles & Means ({num_col})")
        tbl.add_column("group", str(cat_col or "Overall"), col_type="text")
        tbl.add_column("n", "N", col_type="integer")
        tbl.add_column("mean", "Mean", col_type="number")
        tbl.add_column("median", "Median", col_type="number")
        tbl.add_column("iqr", "IQR", col_type="number")

        work = df[[c for c in [cat_col, num_col] if c]].copy()
        work[num_col] = pd.to_numeric(work[num_col], errors="coerce")
        work = work.dropna()

        if cat_col:
            groups = [str(k) for k in work[cat_col].unique()[:12]]
            data_arrays = [work[work[cat_col].astype(str) == g][num_col].values for g in groups]
        else:
            groups = [str(num_col)]
            data_arrays = [work[num_col].values]

        positions = np.arange(len(groups))
        rng = np.random.default_rng(42)

        for idx, (g, arr) in enumerate(zip(groups, data_arrays)):
            if len(arr) == 0:
                continue
            col = pal[idx % len(pal)]
            q75, q25 = np.percentile(arr, [75, 25]) if len(arr) >= 2 else (arr[0], arr[0])
            tbl.add_row({"group": g, "n": len(arr), "mean": float(np.mean(arr)), "median": float(np.median(arr)), "iqr": float(q75 - q25)})

            if mode in ("violin", "raincloud") and len(arr) >= 3 and np.std(arr) > 0:
                vp = ax.violinplot([arr], positions=[idx], widths=0.58 if mode == "violin" else 0.65, showmeans=False, showextrema=False)
                for b in vp["bodies"]:
                    if mode == "raincloud":
                        m = np.mean(b.get_paths()[0].vertices[:, 0])
                        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
                    b.set_facecolor(col)
                    b.set_edgecolor(col)
                    b.set_alpha(0.45 if mode == "raincloud" else 0.38)

            box_pos = idx - 0.08 if mode == "raincloud" else idx
            box_w = 0.14 if mode in ("violin", "raincloud") else 0.42
            bp = ax.boxplot(
                [arr],
                positions=[box_pos],
                widths=box_w,
                patch_artist=True,
                showfliers=(mode == "box" and not self.jitter),
                medianprops={"color": "#0F172A", "linewidth": 1.8},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(col if mode == "box" else "#FFFFFF")
                patch.set_alpha(0.75 if mode == "box" else 0.92)
                patch.set_edgecolor("#1E293B")

            ax.scatter([box_pos], [np.mean(arr)], marker="D", s=34, color="#DC2626", zorder=5, label="Mean" if idx == 0 else None)

            if mode == "raincloud" or self.jitter:
                jit_center = idx - 0.22 if mode == "raincloud" else idx
                jitter_x = jit_center + rng.uniform(-0.06, 0.06, size=len(arr))
                ax.scatter(jitter_x, arr, s=18, color=col, alpha=0.68, edgecolors="none", zorder=4)

        ax.set_xticks(positions)
        ax.set_xticklabels(groups, rotation=18 if len(groups) > 5 else 0)
        if cat_col:
            ax.set_xlabel(str(cat_col))
        ax.set_ylabel(str(num_col))
        title = f"{CHART_TYPE_LABELS[mode]} — {num_col}" + (f" by {cat_col}" if cat_col else "")
        ax.set_title(title, fontweight="bold")
        apply_style_to_axes(ax)
        fig.tight_layout()
        return fig, tbl, title

    def _draw_scatter(self, df: pd.DataFrame, pal: List[str]):
        x_col, y_col = self.x, self.y
        grp_col, sz_col = self.group, self.size_var
        fig, ax = plt.subplots(figsize=(6.4, 4.4))

        cols = [c for c in [x_col, y_col, grp_col, sz_col] if c]
        clean = df[cols].copy()
        clean[x_col] = pd.to_numeric(clean[x_col], errors="coerce")
        clean[y_col] = pd.to_numeric(clean[y_col], errors="coerce")
        clean = clean.dropna(subset=[x_col, y_col])

        sizes = 42.0
        if sz_col and sz_col in clean.columns:
            sv = pd.to_numeric(clean[sz_col], errors="coerce").fillna(clean[sz_col].median() if pd.api.types.is_numeric_dtype(clean[sz_col]) else 1.0)
            s_min, s_max = float(sv.min()), float(sv.max())
            sizes = 25.0 + 180.0 * ((sv - s_min) / max(1e-9, s_max - s_min))

        if grp_col:
            for gi, (g, sub) in enumerate(clean.groupby(grp_col)):
                sub_sz = sizes.loc[sub.index] if isinstance(sizes, pd.Series) else sizes
                ax.scatter(sub[x_col], sub[y_col], s=sub_sz, color=pal[gi % len(pal)], alpha=0.78, label=str(g), edgecolors="white", linewidth=0.5)
            ax.legend(title=str(grp_col), frameon=False)
        else:
            ax.scatter(clean[x_col], clean[y_col], s=sizes, color=pal[0], alpha=0.78, edgecolors="white", linewidth=0.5)

        if len(clean) >= 3 and self.fit_line != "none":
            xv = clean[x_col].values
            yv = clean[y_col].values
            order = np.argsort(xv)
            xs_sorted = xv[order]
            if self.fit_line == "quadratic" and len(clean) >= 4:
                poly = np.poly1d(np.polyfit(xv, yv, 2))
                gx = np.linspace(xv.min(), xv.max(), 120)
                ax.plot(gx, poly(gx), color="#0F172A", linewidth=1.9, linestyle="--", label="Quadratic fit")
            elif self.fit_line == "smooth":
                from statsmodels.nonparametric.smoothers_lowess import lowess
                sm_xy = lowess(yv, xv, frac=0.6)
                ax.plot(sm_xy[:, 0], sm_xy[:, 1], color="#0F172A", linewidth=1.9, label="LOESS smooth")
            else:
                sns.regplot(x=x_col, y=y_col, data=clean, scatter=False, ax=ax, color="#0F172A", line_kws={"linewidth": 1.8})

        r_val, p_val = (stats.pearsonr(clean[x_col], clean[y_col]) if len(clean) >= 3 else (np.nan, np.nan))
        tbl = ResultsTable(name="scatter_summary", title=f"Bivariate Association ({x_col} vs {y_col})")
        tbl.add_column("n", "N", col_type="integer")
        tbl.add_column("r", "Pearson r", col_type="number")
        tbl.add_column("p", "p-value", col_type="number", fmt="pvalue")
        tbl.add_row({"n": len(clean), "r": float(r_val), "p": float(p_val)})

        title = f"{'Bubble' if sz_col else 'Scatter'} Plot — {y_col} vs {x_col}"
        ax.set_xlabel(str(x_col))
        ax.set_ylabel(str(y_col))
        ax.set_title(title, fontweight="bold")
        apply_style_to_axes(ax)
        fig.tight_layout()
        return fig, tbl, title

    def _draw_pairplot(self, df: pd.DataFrame, pal: List[str]):
        v_list = self.vars[:5] if len(self.vars) >= 2 else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:4]
        k = len(v_list)
        fig, axes = plt.subplots(k, k, figsize=(2.1 * k, 2.0 * k))
        if k == 1:
            axes = np.array([[axes]])

        clean = df[v_list + ([self.group] if self.group and self.group not in v_list else [])].dropna()
        for i in range(k):
            for j in range(k):
                ax = axes[i, j]
                if i == j:
                    sns.histplot(clean[v_list[i]], kde=True, color=pal[i % len(pal)], ax=ax, bins=14)
                else:
                    if self.group and self.group in clean.columns:
                        for gi, (g, sub) in enumerate(clean.groupby(self.group)):
                            ax.scatter(sub[v_list[j]], sub[v_list[i]], s=14, alpha=0.65, color=pal[gi % len(pal)], label=str(g) if (i == 0 and j == 1) else None)
                    else:
                        ax.scatter(clean[v_list[j]], clean[v_list[i]], s=14, alpha=0.65, color=pal[0])
                if i == k - 1:
                    ax.set_xlabel(v_list[j], fontsize=8.5)
                else:
                    ax.set_xlabel("")
                if j == 0:
                    ax.set_ylabel(v_list[i], fontsize=8.5)
                else:
                    ax.set_ylabel("")
                apply_style_to_axes(ax)

        title = f"Scatterplot Matrix ({', '.join(v_list)})"
        fig.suptitle(title, fontweight="bold", y=1.02)
        fig.tight_layout()
        return fig, None, title

    def _draw_heatmap(self, df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        if self.x and self.group and not self.vars:
            if self.y and pd.api.types.is_numeric_dtype(df[self.y]):
                mat = df.pivot_table(index=self.x, columns=self.group, values=self.y, aggfunc=self.agg if self.agg in ("mean", "sum", "median", "count") else "mean")
                title = f"Heatmap ({self.agg.upper()} of {self.y}: {self.x} × {self.group})"
            else:
                mat = pd.crosstab(df[self.x], df[self.group])
                title = f"Frequency Heatmap ({self.x} × {self.group})"
            sns.heatmap(mat, annot=True, fmt=".2f" if self.y else "d", cmap="YlGnBu", ax=ax)
        else:
            v_list = self.vars if len(self.vars) >= 2 else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:6]
            mat = df[v_list].apply(pd.to_numeric, errors="coerce").corr()
            title = f"Correlation Heatmap ({len(v_list)} Variables)"
            sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0, ax=ax)

        ax.set_title(title, fontweight="bold")
        fig.tight_layout()
        return fig, None, title

    def _draw_errorbar(self, df: pd.DataFrame, pal: List[str]):
        x_col, y_col = self.x, self.y
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        tbl = ResultsTable(name="ci_summary", title=f"Mean & {self.error_bar.upper()} Intervals ({y_col} by {x_col})")
        tbl.add_column("group", str(x_col), col_type="text")
        tbl.add_column("n", "N", col_type="integer")
        tbl.add_column("mean", "Mean", col_type="number")
        tbl.add_column("lower", "Lower", col_type="number")
        tbl.add_column("upper", "Upper", col_type="number")

        cats = [str(k) for k in df[x_col].dropna().unique()[:16]]
        means, errs = [], []
        for c in cats:
            sub = pd.to_numeric(df[df[x_col].astype(str) == c][y_col], errors="coerce").dropna()
            m = float(sub.mean()) if len(sub) else np.nan
            e = _err_val(sub, self.error_bar if self.error_bar != "none" else "ci95")
            means.append(m)
            errs.append(e)
            tbl.add_row({"group": c, "n": len(sub), "mean": m, "lower": m - e, "upper": m + e})

        pos = np.arange(len(cats))
        if self.horizontal:
            ax.errorbar(means, pos, xerr=errs, fmt="o", color=pal[0], ecolor=pal[1 % len(pal)], elinewidth=2.0, capsize=5, markersize=7)
            ax.set_yticks(pos)
            ax.set_yticklabels(cats)
            ax.set_xlabel(f"Mean of {y_col} (±{self.error_bar.upper()})")
            ax.set_ylabel(str(x_col))
        else:
            ax.errorbar(pos, means, yerr=errs, fmt="o", color=pal[0], ecolor=pal[1 % len(pal)], elinewidth=2.0, capsize=5, markersize=7)
            ax.set_xticks(pos)
            ax.set_xticklabels(cats, rotation=18 if len(cats) > 5 else 0)
            ax.set_xlabel(str(x_col))
            ax.set_ylabel(f"Mean of {y_col} (±{self.error_bar.upper()})")

        title = f"Point & Interval Plot — {y_col} by {x_col}"
        ax.set_title(title, fontweight="bold")
        apply_style_to_axes(ax)
        fig.tight_layout()
        return fig, tbl, title


class CustomFigureResult(BaseAnalysis):
    """Wraps any user- or Claude-created Matplotlib Figure (or saved PNG/SVG) into an Analysis Results card."""

    def __init__(
        self,
        fig_or_src: Union[plt.Figure, str],
        title: str = "Custom Visualization",
        caption: str = "",
        summary_df: Optional[pd.DataFrame] = None,
        dataset: Optional[pd.DataFrame] = None,
        plot_style: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name="customFigure", title=title, dataset=dataset, options={"title": title, "caption": caption, "plotStyle": plot_style or {}})
        if summary_df is not None and not summary_df.empty:
            tbl = ResultsTable(name="figure_data", title=f"{title} — Summary Data")
            for col in summary_df.columns:
                ctype = "number" if pd.api.types.is_numeric_dtype(summary_df[col]) else "text"
                tbl.add_column(str(col), str(col), col_type=ctype)
            for _, row in summary_df.head(50).iterrows():
                tbl.add_row({str(c): row[c] for c in summary_df.columns})
            self.add_table(tbl)

        if isinstance(fig_or_src, plt.Figure):
            with use_plot_style(plot_style):
                png_src = finalize_and_serialize_figure(fig_or_src)
            self.add_plot(title, png_src)
        elif isinstance(fig_or_src, str):
            self.add_plot(title, fig_or_src)

        self.apa_narrative = caption or f"{title} exported directly to Analysis Results."
        self.syntax_python = f"sess.add_figure(fig, title={title!r}, caption={caption!r})"

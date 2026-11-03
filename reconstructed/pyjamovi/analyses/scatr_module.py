"""
pyjamovi scatr Visual Exploration Module (v2.0)
Implements jamovi's scatr Scatterplot (with linear/LOESS regression + marginal density/boxplot) and Pareto Chart.
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from .base import BaseAnalysis, ResultsTable
from .plots import plot_scatr, fig_to_base64, _clean_ax, JMV_BLUE, JMV_ORANGE, JMV_NAVY


class ScatrPlot(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        x: str,
        y: Optional[str] = None,
        group: Optional[str] = None,
        mode: str = "scatter",       # "scatter" or "pareto"
        reg_line: str = "linear",    # "none", "linear", "smooth"
        marginal: str = "dens"       # "none", "dens", "box"
    ):
        title = "Scatterplot (scatr)" if mode == "scatter" else "Pareto Chart (scatr)"
        super().__init__(
            name="scatr",
            title=title,
            dataset=data,
            options={"x": x, "y": y, "group": group, "mode": mode, "regLine": reg_line, "marginal": marginal}
        )
        self.x = x
        self.y = y
        self.group = group
        self.mode = mode
        self.reg_line = reg_line
        self.marginal = marginal
        self.syntax_python = f"jmv.scatr(df, x={x!r}, y={y!r}, group={group!r}, reg_line={reg_line!r}, marginal={marginal!r})"
        self.syntax_r = f"scatr::scat(data=data, x={x}, y={y}, group={group or 'NULL'}, line={reg_line!r}, marg={marginal!r})"
        self.run()

    def run(self):
        df = self.dataset
        if self.mode == "pareto" or not self.y:
            vc = df[self.x].dropna().astype(str).value_counts()
            tot = max(1, int(vc.sum()))
            cum_pct = np.cumsum(vc.values) / tot * 100.0

            tbl = ResultsTable(name="pareto", title=f"Pareto Frequency Breakdown — {self.x}")
            tbl.add_column("cat", self.x, col_type="text")
            tbl.add_column("cnt", "Counts", col_type="integer")
            tbl.add_column("pct", "% of Total", col_type="number", fmt="pc")
            tbl.add_column("cum", "Cumulative %", col_type="number", fmt="pc")
            for cat, cnt, cp in zip(vc.index, vc.values, cum_pct):
                tbl.add_row({"cat": str(cat), "cnt": int(cnt), "pct": (cnt / tot) * 100.0, "cum": float(cp)})
            self.add_table(tbl)

            fig, ax1 = plt.subplots(figsize=(6.4, 4.1))
            xs = np.arange(len(vc))
            ax1.bar(xs, vc.values, color=JMV_BLUE, edgecolor="white", width=0.68)
            ax1.set_xticks(xs)
            ax1.set_xticklabels(vc.index, rotation=20)
            ax1.set_ylabel("Counts", color=JMV_NAVY)
            ax1.set_title(f"Pareto Chart — {self.x}", color=JMV_NAVY)
            _clean_ax(ax1)

            ax2 = ax1.twinx()
            ax2.plot(xs, cum_pct, "-o", color=JMV_ORANGE, linewidth=2.0, markersize=5.5)
            ax2.set_ylim(0, 105)
            ax2.set_ylabel("Cumulative %", color=JMV_ORANGE)
            ax2.spines["top"].set_visible(False)

            self.add_plot(f"Pareto Chart — {self.x}", fig_to_base64(fig))
            self.apa_narrative = f"Pareto chart of {self.x} across {len(vc)} categories (N = {tot})."
            return

        clean = df[[self.x, self.y]].apply(pd.to_numeric, errors="coerce").dropna()
        r_val, p_val = stats.pearsonr(clean[self.x], clean[self.y]) if len(clean) >= 3 else (np.nan, np.nan)
        tbl = ResultsTable(name="scatr_summary", title=f"Bivariate Summary — {self.y} vs. {self.x}")
        tbl.add_column("n", "N", col_type="integer")
        tbl.add_column("r", "Pearson's r", col_type="number", fmt="zto")
        tbl.add_column("r2", "R²", col_type="number", fmt="zto")
        tbl.add_column("p", "p", col_type="number", fmt="pvalue")
        tbl.add_row({"n": len(clean), "r": float(r_val), "r2": float(r_val ** 2) if not np.isnan(r_val) else np.nan, "p": float(p_val)})
        self.add_table(tbl)

        self.add_plot(
            f"Scatterplot — {self.y} vs {self.x}",
            plot_scatr(df, self.x, self.y, group=self.group, reg_line=self.reg_line, marginal=self.marginal)
        )
        self.apa_narrative = f"Scatterplot (scatr) of {self.y} against {self.x} (N = {len(clean)}, r = {r_val:.3f})."

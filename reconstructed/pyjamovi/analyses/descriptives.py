"""
pyjamovi Descriptives Module (v2.0)
Faithful clean-room Python implementation of jamovi jmv::descriptives with integrated plots & frequency tables.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseAnalysis, ResultsTable
from .plots import plot_descriptives_box, plot_descriptives_hist, plot_descriptives_bar, plot_qq


class Descriptives(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        vars: Union[str, List[str]],
        split_by: Optional[str] = None,
        freq: bool = False,
        n: bool = True,
        missing: bool = True,
        mean: bool = True,
        median: bool = True,
        mode: bool = False,
        sd: bool = True,
        variance: bool = False,
        iqr: bool = False,
        se: bool = True,
        min: bool = True,
        max: bool = True,
        skew: bool = False,
        kurt: bool = False,
        sw: bool = False,
        hist: bool = False,
        box: bool = False,
        violin: bool = True,
        bar: bool = False,
        qq: bool = False
    ):
        v_list = [vars] if isinstance(vars, str) else list(vars)
        super().__init__(
            name="descriptives",
            title="Descriptives",
            dataset=data,
            options={
                "vars": v_list,
                "splitBy": split_by,
                "freq": freq,
                "n": n,
                "missing": missing,
                "mean": mean,
                "median": median,
                "mode": mode,
                "sd": sd,
                "variance": variance,
                "iqr": iqr,
                "se": se,
                "min": min,
                "max": max,
                "skew": skew,
                "kurt": kurt,
                "sw": sw,
                "hist": hist,
                "box": box,
                "violin": violin,
                "bar": bar,
                "qq": qq
            }
        )
        self.vars = v_list
        self.split_by = split_by
        split_arg = f", splitBy={split_by}" if split_by else ""
        self.syntax_python = f"jmv.descriptives(df, vars={v_list!r}, split_by={split_by!r})"
        self.syntax_r = f"jmv::descriptives(data=data, vars=vars({', '.join(v_list)}){split_arg})"
        self.run()

    def run(self):
        opts = self.options
        df = self.dataset

        table = ResultsTable(name="descriptives", title="Descriptives")
        table.add_column("var", "Statistic", col_type="text")

        if self.split_by and self.split_by in df.columns:
            groups = sorted(df[self.split_by].dropna().unique(), key=str)
            for v in self.vars:
                for g in groups:
                    col_id = f"{v}_{g}"
                    table.add_column(col_id, f"{v} ({g})", col_type="number")
        else:
            groups = [None]
            for v in self.vars:
                table.add_column(v, v, col_type="number")

        def _num(s: pd.Series) -> pd.Series:
            return pd.to_numeric(s, errors="coerce").dropna()

        stats_to_compute = [
            ("N", "n", lambda s: int(s.notna().sum())),
            ("Missing", "missing", lambda s: int(s.isna().sum())),
            ("Mean", "mean", lambda s: float(_num(s).mean()) if len(_num(s)) > 0 else np.nan),
            ("Std. error mean", "se", lambda s: float(_num(s).sem()) if len(_num(s)) > 1 else np.nan),
            ("Median", "median", lambda s: float(_num(s).median()) if len(_num(s)) > 0 else np.nan),
            ("Mode", "mode", lambda s: float(_num(s).mode().iloc[0]) if len(_num(s).mode()) > 0 else np.nan),
            ("Std. deviation", "sd", lambda s: float(_num(s).std(ddof=1)) if len(_num(s)) > 1 else np.nan),
            ("Variance", "variance", lambda s: float(_num(s).var(ddof=1)) if len(_num(s)) > 1 else np.nan),
            ("IQR", "iqr", lambda s: float(_num(s).quantile(0.75) - _num(s).quantile(0.25)) if len(_num(s)) > 0 else np.nan),
            ("Minimum", "min", lambda s: float(_num(s).min()) if len(_num(s)) > 0 else np.nan),
            ("Maximum", "max", lambda s: float(_num(s).max()) if len(_num(s)) > 0 else np.nan),
            ("Skewness", "skew", lambda s: float(stats.skew(_num(s), bias=False)) if len(_num(s)) >= 3 else np.nan),
            ("Kurtosis", "kurt", lambda s: float(stats.kurtosis(_num(s), bias=False)) if len(_num(s)) >= 4 else np.nan),
            ("Shapiro-Wilk W", "sw", lambda s: float(stats.shapiro(_num(s))[0]) if 3 <= len(_num(s)) <= 5000 else np.nan),
            ("Shapiro-Wilk p", "sw", lambda s: float(stats.shapiro(_num(s))[1]) if 3 <= len(_num(s)) <= 5000 else np.nan),
        ]

        for stat_name, opt_key, func in stats_to_compute:
            if not opts.get(opt_key, False):
                continue
            row = {"var": stat_name}
            if opt_key in ("n", "missing"):
                row["_rowfmt"] = "integer"   # counts print as whole numbers (60, not 60.000)
            for v in self.vars:
                if v not in df.columns:
                    continue
                series = df[v]
                if self.split_by and self.split_by in df.columns:
                    for g in groups:
                        sub = series[df[self.split_by] == g]
                        try:
                            row[f"{v}_{g}"] = func(sub)
                        except Exception:
                            row[f"{v}_{g}"] = np.nan
                else:
                    try:
                        row[v] = func(series)
                    except Exception:
                        row[v] = np.nan
            table.add_row(row)

        self.add_table(table)

        # Frequency tables for categorical or requested variables
        if opts.get("freq", False):
            for v in self.vars:
                if v in df.columns and df[v].nunique() <= 30:
                    ft = ResultsTable(name=f"freq_{v}", title=f"Frequencies of {v}")
                    ft.add_column("level", v, col_type="text")
                    ft.add_column("counts", "Counts", col_type="integer")
                    ft.add_column("pct_total", "% of Total", col_type="number", fmt="pc")
                    ft.add_column("pct_cum", "Cumulative %", col_type="number", fmt="pc")
                    vc = df[v].dropna().value_counts()
                    tot = max(1, int(vc.sum()))
                    cum = 0.0
                    for lvl, cnt in vc.items():
                        pct = (cnt / tot) * 100.0
                        cum += pct
                        ft.add_row({"level": str(lvl), "counts": int(cnt), "pct_total": pct, "pct_cum": cum})
                    self.add_table(ft)

        # Plots
        for v in self.vars:
            if v not in df.columns:
                continue
            is_num = len(_num(df[v])) >= 2
            if (opts.get("box", False) or opts.get("violin", False)) and is_num:
                self.add_plot(f"Box Plot — {v}", plot_descriptives_box(df, v, self.split_by, violin=opts.get("violin", True)))
            if opts.get("hist", False) and is_num:
                self.add_plot(f"Histogram — {v}", plot_descriptives_hist(df, v, kde=True))
            if opts.get("bar", False):
                self.add_plot(f"Bar Plot — {v}", plot_descriptives_bar(df, v, self.split_by))
            if opts.get("qq", False) and is_num:
                self.add_plot(f"Q-Q Plot — {v}", plot_qq(df, v))

        apa_notes = []
        for v in self.vars:
            if v in df.columns:
                sub = _num(df[v])
                if len(sub) > 0:
                    apa_notes.append(f"{v} (M = {sub.mean():.2f}, SD = {sub.std(ddof=1):.2f}, N = {len(sub)})")
                else:
                    apa_notes.append(f"{v} (N = {df[v].notna().sum()}, {df[v].nunique()} levels)")
        self.apa_narrative = f"Descriptive statistics: {', '.join(apa_notes)}." if apa_notes else ""

"""
pyjamovi One-Way ANOVA Module (v2.0)
Faithful clean-room Python implementation of jamovi jmv::anovaOneW with 5 Post-Hoc methods & Descriptives/Q-Q plots.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseAnalysis, ResultsTable
from .posthoc import compute_posthoc
from .plots import plot_mean_ci, plot_qq, plot_descriptives_box


class AnovaOneW(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        deps: Union[str, List[str]],
        group: str,
        welchs: bool = True,
        fishers: bool = False,
        desc: bool = True,
        norm: bool = False,
        eqv: bool = False,
        ph_method: Optional[str] = "tukey",  # "none", "tukey", "gamesHowell", "scheffe", "bonferroni", "holm"
        desc_plot: bool = False,
        qq: bool = False
    ):
        dep_vars = [deps] if isinstance(deps, str) else list(deps)
        super().__init__(
            name="anovaOneW",
            title="One-Way ANOVA",
            dataset=data,
            options={
                "deps": dep_vars,
                "group": group,
                "welchs": welchs,
                "fishers": fishers,
                "desc": desc,
                "norm": norm,
                "eqv": eqv,
                "phMethod": ph_method,
                "descPlot": desc_plot,
                "qq": qq
            }
        )
        self.dep_vars = dep_vars
        self.group = group
        self.syntax_python = f"jmv.anovaOneW(df, deps={dep_vars!r}, group={group!r}, welchs={welchs}, fishers={fishers}, ph_method={ph_method!r})"
        self.syntax_r = f"jmv::anovaOneW(formula={' + '.join(dep_vars)} ~ {group}, data=data, welchs={str(welchs).upper()}, fishers={str(fishers).upper()})"
        self.run()

    def run(self):
        df = self.dataset
        opts = self.options
        group_col = self.group

        if group_col not in df.columns:
            raise ValueError(f"Grouping variable '{group_col}' not found in dataset")

        groups = sorted(df[group_col].dropna().unique(), key=str)
        k = len(groups)
        if k < 2:
            raise ValueError(f"Grouping variable '{group_col}' must have at least 2 levels, got {k}")

        main_table = ResultsTable(name="anova", title="One-Way ANOVA")
        main_table.add_column("var", "Dependent Variable", col_type="text")
        main_table.add_column("stat_type", "Method", col_type="text")
        main_table.add_column("f", "F", col_type="number")
        main_table.add_column("df1", "df1", col_type="number")
        main_table.add_column("df2", "df2", col_type="number")
        main_table.add_column("p", "p", col_type="number", fmt="pvalue")

        desc_table = None
        if opts["desc"]:
            desc_table = ResultsTable(name="desc", title="Group Descriptives")
            desc_table.add_column("var", "Variable", col_type="text")
            desc_table.add_column("group", "Group", col_type="text")
            desc_table.add_column("n", "N", col_type="integer")
            desc_table.add_column("mean", "Mean", col_type="number")
            desc_table.add_column("sd", "SD", col_type="number")
            desc_table.add_column("se", "SE", col_type="number")

        eqv_table = None
        if opts["eqv"]:
            eqv_table = ResultsTable(name="eqv", title="Homogeneity of Variances Test (Levene's)")
            eqv_table.add_column("var", "Variable", col_type="text")
            eqv_table.add_column("f", "F", col_type="number")
            eqv_table.add_column("df1", "df1", col_type="integer")
            eqv_table.add_column("df2", "df2", col_type="integer")
            eqv_table.add_column("p", "p", col_type="number", fmt="pvalue")

        norm_table = None
        if opts["norm"]:
            norm_table = ResultsTable(name="norm", title="Normality Test (Shapiro-Wilk)")
            norm_table.add_column("var", "Variable", col_type="text")
            norm_table.add_column("w", "W", col_type="number")
            norm_table.add_column("p", "p", col_type="number", fmt="pvalue")

        posthoc_tables = []
        apa_statements = []

        for v in self.dep_vars:
            if v not in df.columns:
                continue

            sub_groups = [pd.to_numeric(df[df[group_col] == g][v], errors="coerce").dropna() for g in groups]
            valid_groups = [g for g in sub_groups if len(g) >= 2]
            if len(valid_groups) < 2:
                continue

            if opts["welchs"]:
                ns = np.array([len(g) for g in valid_groups], dtype=float)
                means = np.array([g.mean() for g in valid_groups], dtype=float)
                vars_ = np.maximum(np.array([g.var(ddof=1) for g in valid_groups], dtype=float), 1e-12)
                weights = ns / vars_
                sum_w = np.sum(weights)
                mean_w = np.sum(weights * means) / sum_w

                k_val = len(valid_groups)
                df1_w = k_val - 1
                q = np.sum((1.0 - weights / sum_w) ** 2 / (ns - 1.0)) / (k_val ** 2 - 1.0)
                df2_w = 1.0 / (3.0 * q) if q > 0 else np.nan
                f_welch = (np.sum(weights * (means - mean_w) ** 2) / (k_val - 1.0)) / (1.0 + 2.0 * (k_val - 2.0) * q)
                p_welch = 1.0 - stats.f.cdf(f_welch, df1_w, df2_w) if not np.isnan(df2_w) else np.nan

                main_table.add_row({
                    "var": v,
                    "stat_type": "Welch's",
                    "f": float(f_welch),
                    "df1": float(df1_w),
                    "df2": float(df2_w),
                    "p": float(p_welch)
                })
                if not opts["fishers"]:
                    p_str = "< .001" if p_welch < 0.001 else f"= {p_welch:.3f}"
                    apa_statements.append(
                        f"Welch's one-way ANOVA for {v} across {group_col} yielded F({df1_w:.0f}, {df2_w:.2f}) = {f_welch:.2f}, p {p_str}."
                    )

            if opts["fishers"]:
                f_stat, f_p = stats.f_oneway(*valid_groups)
                n_total = sum(len(g) for g in valid_groups)
                df1 = k - 1
                df2 = n_total - k
                main_table.add_row({
                    "var": v,
                    "stat_type": "Fisher's",
                    "f": float(f_stat),
                    "df1": float(df1),
                    "df2": float(df2),
                    "p": float(f_p)
                })
                sig = "statistically significant" if f_p < 0.05 else "not statistically significant"
                p_str = "< .001" if f_p < 0.001 else f"= {f_p:.3f}"
                apa_statements.append(f"A one-way ANOVA revealed a {sig} effect for {v}, F({df1}, {df2}) = {f_stat:.2f}, p {p_str}.")

            if desc_table:
                for g_name, g_data in zip(groups, sub_groups):
                    desc_table.add_row({
                        "var": v,
                        "group": str(g_name),
                        "n": len(g_data),
                        "mean": float(g_data.mean()) if len(g_data) > 0 else np.nan,
                        "sd": float(g_data.std(ddof=1)) if len(g_data) > 1 else np.nan,
                        "se": float(g_data.sem()) if len(g_data) > 1 else np.nan
                    })

            if eqv_table:
                lev_f, lev_p = stats.levene(*valid_groups, center="median")
                n_total = sum(len(g) for g in valid_groups)
                eqv_table.add_row({
                    "var": v,
                    "f": float(lev_f),
                    "df1": k - 1,
                    "df2": n_total - k,
                    "p": float(lev_p)
                })

            if norm_table:
                resids = pd.concat([g - g.mean() for g in valid_groups])
                if 3 <= len(resids) <= 5000:
                    w_val, p_w = stats.shapiro(resids)
                    norm_table.add_row({"var": v, "w": float(w_val), "p": float(p_w)})

            ph_m = opts.get("phMethod")
            if ph_m and str(ph_m).lower() != "none":
                ph_tbl = compute_posthoc(df, dep=v, group=group_col, method=str(ph_m))
                ph_tbl.title = f"{ph_tbl.title} [{v}]" if len(self.dep_vars) > 1 else ph_tbl.title
                posthoc_tables.append(ph_tbl)

            if opts.get("descPlot", False):
                self.add_plot(f"Descriptives Plot — {v} by {group_col}", plot_mean_ci(df, v, group_col))
            if opts.get("qq", False):
                self.add_plot(f"Q-Q Plot — {v}", plot_qq(df, v))

        self.add_table(main_table)
        if desc_table:
            self.add_table(desc_table)
        if eqv_table:
            self.add_table(eqv_table)
        if norm_table:
            self.add_table(norm_table)
        for pt in posthoc_tables:
            self.add_table(pt)

        self.apa_narrative = " ".join(apa_statements)

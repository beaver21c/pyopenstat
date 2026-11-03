"""
pyjamovi Independent Samples T-Test Module (v2.0)
Faithful clean-room Python implementation of jamovi jmv::ttestIS with Student's, Welch's, Mann-Whitney, CIs & Plots.
"""

from typing import List, Union
import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseAnalysis, ResultsTable
from .plots import plot_mean_ci, plot_qq, plot_descriptives_box


class TTestIS(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        vars: Union[str, List[str]],
        group: str,
        hypothesis: str = "different",
        students: bool = True,
        welchs: bool = False,
        mann: bool = False,
        mean_diff: bool = True,
        effect_size: bool = True,
        ci: bool = False,
        ci_width: float = 95.0,
        desc: bool = True,
        norm: bool = False,
        eqv: bool = False,
        plots: bool = False,
        qq: bool = False,
        raincloud: bool = False
    ):
        dep_vars = [vars] if isinstance(vars, str) else list(vars)
        super().__init__(
            name="ttestIS",
            title="Independent Samples T-Test",
            dataset=data,
            options={
                "vars": dep_vars,
                "group": group,
                "hypothesis": hypothesis,
                "students": students,
                "welchs": welchs,
                "mann": mann,
                "meanDiff": mean_diff,
                "effectSize": effect_size,
                "ci": ci,
                "ciWidth": ci_width,
                "desc": desc,
                "norm": norm,
                "eqv": eqv,
                "plots": plots,
                "qq": qq,
                "raincloud": raincloud
            }
        )
        self.dep_vars = dep_vars
        self.group = group
        self.syntax_python = f"jmv.ttestIS(df, vars={dep_vars!r}, group={group!r}, welchs={welchs}, mann={mann})"
        self.syntax_r = f"jmv::ttestIS(formula={' + '.join(dep_vars)} ~ {group}, data=data, welchs={str(welchs).upper()}, mann={str(mann).upper()})"
        self.run()

    def run(self):
        df = self.dataset
        opts = self.options
        group_col = self.group

        if group_col not in df.columns:
            raise ValueError(f"Grouping variable '{group_col}' not found in dataset")

        groups = sorted(df[group_col].dropna().unique(), key=str)
        if len(groups) != 2:
            raise ValueError(f"Grouping variable '{group_col}' must have exactly 2 levels, got {len(groups)}: {groups}")

        g1_name, g2_name = groups[0], groups[1]

        main_table = ResultsTable(name="ttest", title="Independent Samples T-Test")
        main_table.add_column("var", "Dependent Variable", col_type="text")
        main_table.add_column("test", "Test", col_type="text")
        main_table.add_column("stat", "Statistic", col_type="number")
        main_table.add_column("df", "df", col_type="number")
        main_table.add_column("p", "p", col_type="number", fmt="pvalue")

        if opts["meanDiff"]:
            main_table.add_column("md", "Mean difference", col_type="number")
            main_table.add_column("sed", "SE difference", col_type="number")
        if opts["ci"]:
            main_table.add_column("cil", f"{opts['ciWidth']:.0f}% CI Lower", col_type="number")
            main_table.add_column("ciu", f"{opts['ciWidth']:.0f}% CI Upper", col_type="number")
        if opts["effectSize"]:
            main_table.add_column("es", "Effect Size", col_type="number")

        desc_table = None
        if opts["desc"]:
            desc_table = ResultsTable(name="desc", title="Group Descriptives")
            desc_table.add_column("var", "Variable", col_type="text")
            desc_table.add_column("group", "Group", col_type="text")
            desc_table.add_column("n", "N", col_type="integer")
            desc_table.add_column("mean", "Mean", col_type="number")
            desc_table.add_column("med", "Median", col_type="number")
            desc_table.add_column("sd", "SD", col_type="number")
            desc_table.add_column("se", "SE", col_type="number")

        norm_table = None
        if opts["norm"]:
            norm_table = ResultsTable(name="norm", title="Normality Test (Shapiro-Wilk)")
            norm_table.add_column("var", "Variable", col_type="text")
            norm_table.add_column("w", "W", col_type="number")
            norm_table.add_column("p", "p", col_type="number", fmt="pvalue")

        eqv_table = None
        if opts["eqv"]:
            eqv_table = ResultsTable(name="eqv", title="Homogeneity of Variances Test (Levene's)")
            eqv_table.add_column("var", "Variable", col_type="text")
            eqv_table.add_column("f", "F", col_type="number")
            eqv_table.add_column("df1", "df1", col_type="integer")
            eqv_table.add_column("df2", "df2", col_type="integer")
            eqv_table.add_column("p", "p", col_type="number", fmt="pvalue")

        apa_statements = []
        alpha = 1.0 - float(opts["ciWidth"]) / 100.0

        for v in self.dep_vars:
            if v not in df.columns:
                continue

            sub1 = pd.to_numeric(df[df[group_col] == g1_name][v], errors="coerce").dropna()
            sub2 = pd.to_numeric(df[df[group_col] == g2_name][v], errors="coerce").dropna()
            n1, n2 = len(sub1), len(sub2)
            if n1 < 2 or n2 < 2:
                continue

            m1, m2 = sub1.mean(), sub2.mean()
            s1, s2 = sub1.std(ddof=1), sub2.std(ddof=1)
            mean_diff_val = m1 - m2

            df_stud = n1 + n2 - 2
            s_pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / df_stud)
            se_diff_stud = s_pooled * np.sqrt(1.0 / n1 + 1.0 / n2)
            cohen_d = mean_diff_val / s_pooled if s_pooled > 0 else np.nan

            alt_scipy = "two-sided"
            if opts["hypothesis"] == "oneGreater":
                alt_scipy = "greater"
            elif opts["hypothesis"] == "twoGreater":
                alt_scipy = "less"

            if opts["students"]:
                t_res = stats.ttest_ind(sub1, sub2, equal_var=True, alternative=alt_scipy)
                t_crit = stats.t.ppf(1.0 - alpha / 2.0, df_stud)
                row = {
                    "var": v,
                    "test": "Student's t",
                    "stat": float(t_res.statistic),
                    "df": float(df_stud),
                    "p": float(t_res.pvalue),
                    "md": float(mean_diff_val),
                    "sed": float(se_diff_stud),
                    "cil": float(mean_diff_val - t_crit * se_diff_stud),
                    "ciu": float(mean_diff_val + t_crit * se_diff_stud),
                    "es": float(cohen_d)
                }
                main_table.add_row(row)

                sig = "significant" if t_res.pvalue < 0.05 else "non-significant"
                p_str = "< .001" if t_res.pvalue < 0.001 else f"= {t_res.pvalue:.3f}"
                apa_statements.append(
                    f"For {v}, the difference between {g1_name} (M = {m1:.2f}, SD = {s1:.2f}) and {g2_name} "
                    f"(M = {m2:.2f}, SD = {s2:.2f}) was {sig}, t({df_stud}) = {t_res.statistic:.2f}, p {p_str}, d = {cohen_d:.2f}."
                )

            if opts["welchs"]:
                t_welch = stats.ttest_ind(sub1, sub2, equal_var=False, alternative=alt_scipy)
                se_welch = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
                num = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2
                den = (s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1)
                df_welch = num / den if den > 0 else np.nan
                t_crit_w = stats.t.ppf(1.0 - alpha / 2.0, df_welch) if not np.isnan(df_welch) else 1.96
                main_table.add_row({
                    "var": v,
                    "test": "Welch's t",
                    "stat": float(t_welch.statistic),
                    "df": float(df_welch),
                    "p": float(t_welch.pvalue),
                    "md": float(mean_diff_val),
                    "sed": float(se_welch),
                    "cil": float(mean_diff_val - t_crit_w * se_welch),
                    "ciu": float(mean_diff_val + t_crit_w * se_welch),
                    "es": float(cohen_d)
                })

            if opts["mann"]:
                u_res = stats.mannwhitneyu(sub1, sub2, alternative=alt_scipy)
                r_biserial = 1.0 - (2.0 * u_res.statistic) / (n1 * n2)
                main_table.add_row({
                    "var": v,
                    "test": "Mann-Whitney U",
                    "stat": float(u_res.statistic),
                    "df": np.nan,
                    "p": float(u_res.pvalue),
                    "md": float(sub1.median() - sub2.median()),
                    "sed": np.nan,
                    "cil": np.nan,
                    "ciu": np.nan,
                    "es": float(r_biserial)
                })

            if desc_table:
                for grp_val, sub_val in [(g1_name, sub1), (g2_name, sub2)]:
                    desc_table.add_row({
                        "var": v,
                        "group": str(grp_val),
                        "n": len(sub_val),
                        "mean": float(sub_val.mean()),
                        "med": float(sub_val.median()),
                        "sd": float(sub_val.std(ddof=1)),
                        "se": float(sub_val.sem())
                    })

            if norm_table:
                resid = pd.concat([sub1 - m1, sub2 - m2])
                if 3 <= len(resid) <= 5000:
                    w_val, p_w = stats.shapiro(resid)
                    norm_table.add_row({"var": v, "w": float(w_val), "p": float(p_w)})

            if eqv_table:
                lev_f, lev_p = stats.levene(sub1, sub2, center="median")
                eqv_table.add_row({
                    "var": v,
                    "f": float(lev_f),
                    "df1": 1,
                    "df2": int(df_stud),
                    "p": float(lev_p)
                })

            if opts.get("plots", False):
                self.add_plot(f"Descriptives Plot — {v}", plot_mean_ci(df, v, group_col, ci_level=float(opts["ciWidth"]) / 100.0))
            if opts.get("raincloud", False):
                self.add_plot(f"Raincloud / Violin Plot — {v}", plot_descriptives_box(df, v, group_col, violin=True))
            if opts.get("qq", False):
                self.add_plot(f"Q-Q Plot — {v}", plot_qq(df, v))

        if opts["mann"] and opts["effectSize"]:
            main_table.add_note("For Student's t and Welch's t, effect size is given by Cohen's d. For Mann-Whitney U, effect size is rank biserial correlation.")

        self.add_table(main_table)
        if norm_table:
            self.add_table(norm_table)
        if eqv_table:
            self.add_table(eqv_table)
        if desc_table:
            self.add_table(desc_table)
        self.apa_narrative = " ".join(apa_statements)

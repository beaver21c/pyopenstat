"""
pyjamovi Paired Samples & One Sample T-Test Modules (v2.0)
Faithful clean-room Python implementations of jamovi jmv::ttestPS and jmv::ttestOneS.
"""

from typing import List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseAnalysis, ResultsTable
from .plots import plot_rm_profile, plot_descriptives_box, plot_qq


class TTestPS(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        pairs: Optional[List[Tuple[str, str]]] = None,
        var1: Optional[str] = None,
        var2: Optional[str] = None,
        students: bool = True,
        wilcoxon: bool = False,
        hypothesis: str = "different",
        mean_diff: bool = True,
        effect_size: bool = True,
        desc: bool = True,
        norm: bool = False,
        plots: bool = False
    ):
        if pairs is None:
            if var1 and var2:
                resolved_pairs = [(var1, var2)]
            else:
                raise ValueError("Provide either 'pairs=[(v1, v2), ...]' or 'var1' and 'var2'")
        else:
            resolved_pairs = [(p[0], p[1]) for p in pairs if len(p) >= 2]

        super().__init__(
            name="ttestPS",
            title="Paired Samples T-Test",
            dataset=data,
            options={
                "pairs": resolved_pairs,
                "students": students,
                "wilcoxon": wilcoxon,
                "hypothesis": hypothesis,
                "meanDiff": mean_diff,
                "effectSize": effect_size,
                "desc": desc,
                "norm": norm,
                "plots": plots
            }
        )
        self.pairs = resolved_pairs
        self.syntax_python = f"jmv.ttestPS(df, pairs={resolved_pairs!r}, wilcoxon={wilcoxon})"
        self.syntax_r = f"jmv::ttestPS(data=data, pairs=list({', '.join(f'list(i1={a!r}, i2={b!r})' for a, b in resolved_pairs)}), wilcoxon={str(wilcoxon).upper()})"
        self.run()

    def run(self):
        df = self.dataset
        opts = self.options

        table = ResultsTable(name="ttest", title="Paired Samples T-Test")
        table.add_column("pair", "Measure 1 - Measure 2", col_type="text")
        table.add_column("test", "Test", col_type="text")
        table.add_column("stat", "Statistic", col_type="number")
        table.add_column("df", "df", col_type="number")
        table.add_column("p", "p", col_type="number", fmt="pvalue")

        if opts["meanDiff"]:
            table.add_column("md", "Mean difference", col_type="number")
            table.add_column("sed", "SE difference", col_type="number")
        if opts["effectSize"]:
            table.add_column("es", "Cohen's d", col_type="number")

        desc_table = ResultsTable(name="desc", title="Descriptives") if opts["desc"] else None
        if desc_table:
            desc_table.add_column("var", "Measure", col_type="text")
            desc_table.add_column("n", "N", col_type="integer")
            desc_table.add_column("mean", "Mean", col_type="number")
            desc_table.add_column("med", "Median", col_type="number")
            desc_table.add_column("sd", "SD", col_type="number")
            desc_table.add_column("se", "SE", col_type="number")

        norm_table = ResultsTable(name="norm", title="Normality Test (Shapiro-Wilk)") if opts["norm"] else None
        if norm_table:
            norm_table.add_column("pair", "Pair", col_type="text")
            norm_table.add_column("w", "W", col_type="number")
            norm_table.add_column("p", "p", col_type="number", fmt="pvalue")

        apa_statements = []
        seen_desc = set()

        for v1, v2 in self.pairs:
            if v1 not in df.columns or v2 not in df.columns:
                continue

            valid = df[[v1, v2]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(valid) < 2:
                continue

            diff = valid[v1] - valid[v2]
            n = len(diff)
            df_val = n - 1
            mean_d = diff.mean()
            sd_d = diff.std(ddof=1)
            se_d = sd_d / np.sqrt(n)
            cohen_d = mean_d / sd_d if sd_d > 0 else np.nan

            if opts["students"]:
                t_res = stats.ttest_rel(valid[v1], valid[v2])
                row = {
                    "pair": f"{v1} - {v2}",
                    "test": "Student's t",
                    "stat": float(t_res.statistic),
                    "df": float(df_val),
                    "p": float(t_res.pvalue),
                    "md": float(mean_d),
                    "sed": float(se_d),
                    "es": float(cohen_d)
                }
                table.add_row(row)

                sig = "statistically significant" if t_res.pvalue < 0.05 else "not statistically significant"
                p_str = "< .001" if t_res.pvalue < 0.001 else f"= {t_res.pvalue:.3f}"
                apa_statements.append(
                    f"A paired-samples t-test for {v1} and {v2} showed a {sig} difference, "
                    f"t({df_val}) = {t_res.statistic:.2f}, p {p_str}, d = {cohen_d:.2f}."
                )

            if opts["wilcoxon"]:
                try:
                    w_res = stats.wilcoxon(valid[v1], valid[v2])
                    table.add_row({
                        "pair": f"{v1} - {v2}",
                        "test": "Wilcoxon W",
                        "stat": float(w_res.statistic),
                        "df": np.nan,
                        "p": float(w_res.pvalue),
                        "md": float(diff.median()),
                        "sed": np.nan,
                        "es": np.nan
                    })
                except Exception:
                    pass

            if desc_table:
                for vn in (v1, v2):
                    if vn not in seen_desc:
                        seen_desc.add(vn)
                        s = valid[vn]
                        desc_table.add_row({
                            "var": vn, "n": len(s), "mean": float(s.mean()),
                            "med": float(s.median()), "sd": float(s.std(ddof=1)), "se": float(s.sem())
                        })

            if norm_table and 3 <= len(diff) <= 5000:
                w_v, p_v = stats.shapiro(diff)
                norm_table.add_row({"pair": f"{v1} - {v2}", "w": float(w_v), "p": float(p_v)})

            if opts.get("plots", False):
                self.add_plot(f"Paired Profile Plot — {v1} vs {v2}", plot_rm_profile(valid, [v1, v2]))

        self.add_table(table)
        if norm_table:
            self.add_table(norm_table)
        if desc_table:
            self.add_table(desc_table)
        self.apa_narrative = " ".join(apa_statements)


class TTestOneS(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        vars: Union[str, List[str]],
        test_value: float = 0.0,
        students: bool = True,
        wilcoxon: bool = False,
        hypothesis: str = "different",
        mean_diff: bool = True,
        effect_size: bool = True,
        desc: bool = True,
        plots: bool = False
    ):
        dep_vars = [vars] if isinstance(vars, str) else list(vars)
        super().__init__(
            name="ttestOneS",
            title="One Sample T-Test",
            dataset=data,
            options={
                "vars": dep_vars,
                "testValue": float(test_value),
                "students": students,
                "wilcoxon": wilcoxon,
                "hypothesis": hypothesis,
                "meanDiff": mean_diff,
                "effectSize": effect_size,
                "desc": desc,
                "plots": plots
            }
        )
        self.vars = dep_vars
        self.test_value = float(test_value)
        self.syntax_python = f"jmv.ttestOneS(df, vars={dep_vars!r}, test_value={self.test_value})"
        self.syntax_r = f"jmv::ttestOneS(data=data, vars=vars({', '.join(dep_vars)}), testValue={self.test_value})"
        self.run()

    def run(self):
        df = self.dataset
        opts = self.options
        mu0 = self.test_value

        table = ResultsTable(name="ttest", title=f"One Sample T-Test (Test Value = {mu0})")
        table.add_column("var", "Variable", col_type="text")
        table.add_column("test", "Test", col_type="text")
        table.add_column("stat", "Statistic", col_type="number")
        table.add_column("df", "df", col_type="number")
        table.add_column("p", "p", col_type="number", fmt="pvalue")

        if opts["meanDiff"]:
            table.add_column("md", "Mean difference", col_type="number")
        if opts["effectSize"]:
            table.add_column("es", "Cohen's d", col_type="number")

        desc_table = ResultsTable(name="desc", title="Descriptives") if opts["desc"] else None
        if desc_table:
            desc_table.add_column("var", "Variable", col_type="text")
            desc_table.add_column("n", "N", col_type="integer")
            desc_table.add_column("mean", "Mean", col_type="number")
            desc_table.add_column("med", "Median", col_type="number")
            desc_table.add_column("sd", "SD", col_type="number")
            desc_table.add_column("se", "SE", col_type="number")

        apa_statements = []

        for v in self.vars:
            if v not in df.columns:
                continue

            sub = pd.to_numeric(df[v], errors="coerce").dropna()
            n = len(sub)
            if n < 2:
                continue

            df_val = n - 1
            mean_val = sub.mean()
            sd_val = sub.std(ddof=1)
            mean_diff_val = mean_val - mu0
            cohen_d = mean_diff_val / sd_val if sd_val > 0 else np.nan

            if opts["students"]:
                t_res = stats.ttest_1samp(sub, popmean=mu0)
                table.add_row({
                    "var": v,
                    "test": "Student's t",
                    "stat": float(t_res.statistic),
                    "df": float(df_val),
                    "p": float(t_res.pvalue),
                    "md": float(mean_diff_val),
                    "es": float(cohen_d)
                })
                sig = "statistically significant" if t_res.pvalue < 0.05 else "not statistically significant"
                p_str = "< .001" if t_res.pvalue < 0.001 else f"= {t_res.pvalue:.3f}"
                apa_statements.append(
                    f"A one-sample t-test comparing {v} (M = {mean_val:.2f}, SD = {sd_val:.2f}) against {mu0} "
                    f"was {sig}, t({df_val}) = {t_res.statistic:.2f}, p {p_str}, d = {cohen_d:.2f}."
                )

            if opts["wilcoxon"]:
                try:
                    w_res = stats.wilcoxon(sub - mu0)
                    table.add_row({
                        "var": v,
                        "test": "Wilcoxon W",
                        "stat": float(w_res.statistic),
                        "df": np.nan,
                        "p": float(w_res.pvalue),
                        "md": float(sub.median() - mu0),
                        "es": np.nan
                    })
                except Exception:
                    pass

            if desc_table:
                desc_table.add_row({
                    "var": v, "n": n, "mean": float(mean_val),
                    "med": float(sub.median()), "sd": float(sd_val), "se": float(sub.sem())
                })

            if opts.get("plots", False):
                self.add_plot(f"Distribution — {v}", plot_descriptives_box(df, v))

        self.add_table(table)
        if desc_table:
            self.add_table(desc_table)
        self.apa_narrative = " ".join(apa_statements)

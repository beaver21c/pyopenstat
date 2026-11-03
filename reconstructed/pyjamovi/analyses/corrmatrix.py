"""
pyjamovi Correlation Matrix Module (v2.0)
Faithful clean-room Python implementation of jamovi jmv::corrMatrix (Pearson, Spearman, Kendall tau-b, Heatmap).
"""

from typing import List, Union
import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseAnalysis, ResultsTable
from .plots import plot_correlation_heatmap


class CorrMatrix(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        vars: Union[str, List[str]],
        pearson: bool = True,
        spearman: bool = False,
        kendall: bool = False,
        sig: bool = True,
        flag: bool = True,
        ci: bool = False,
        ci_width: float = 95.0,
        plots: bool = False
    ):
        v_list = [vars] if isinstance(vars, str) else list(vars)
        super().__init__(
            name="corrMatrix",
            title="Correlation Matrix",
            dataset=data,
            options={
                "vars": v_list,
                "pearson": pearson,
                "spearman": spearman,
                "kendall": kendall,
                "sig": sig,
                "flag": flag,
                "ci": ci,
                "ciWidth": ci_width,
                "plots": plots
            }
        )
        self.vars = v_list
        self.syntax_python = f"jmv.corrMatrix(df, vars={v_list!r}, pearson={pearson}, spearman={spearman}, kendall={kendall})"
        self.syntax_r = f"jmv::corrMatrix(data=data, vars=vars({', '.join(v_list)}), spearman={str(spearman).upper()}, kendall={str(kendall).upper()})"
        self.run()

    def run(self):
        df = self.dataset
        opts = self.options
        v_list = self.vars

        table = ResultsTable(name="matrix", title="Correlation Matrix")
        table.add_column("var", "", col_type="text")
        table.add_column("stat", "", col_type="text")
        for v in v_list:
            table.add_column(v, v, col_type="text")

        apa_highlights = []

        def _stars(pval: float) -> str:
            if not opts["flag"]:
                return ""
            if pval < 0.001:
                return " ***"
            elif pval < 0.01:
                return " **"
            elif pval < 0.05:
                return " *"
            return ""

        for i, row_var in enumerate(v_list):
            if row_var not in df.columns:
                continue

            if opts["pearson"]:
                r_row = {"var": row_var, "stat": "Pearson's r"}
                p_row = {"var": "", "stat": "p-value"}
                n_row = {"var": "", "stat": "N"}
                for j, col_var in enumerate(v_list):
                    if col_var not in df.columns or j > i:
                        r_row[col_var] = ""
                        p_row[col_var] = ""
                        n_row[col_var] = ""
                        continue
                    if i == j:
                        r_row[col_var] = "—"
                        p_row[col_var] = "—"
                        n_row[col_var] = str(int(df[row_var].notna().sum()))
                        continue
                    valid = df[[row_var, col_var]].apply(pd.to_numeric, errors="coerce").dropna()
                    if len(valid) >= 3:
                        r_val, p_val = stats.pearsonr(valid[row_var], valid[col_var])
                        r_row[col_var] = f"{r_val:.3f}{_stars(p_val)}"
                        p_row[col_var] = "< .001" if p_val < 0.001 else f"{p_val:.3f}"
                        n_row[col_var] = str(len(valid))
                        if abs(r_val) >= 0.3:
                            p_txt = "< .001" if p_val < 0.001 else f"= {p_val:.3f}"
                            apa_highlights.append(f"{row_var} and {col_var} (r = {r_val:.2f}, p {p_txt})")
                    else:
                        r_row[col_var] = "—"
                        p_row[col_var] = "—"
                        n_row[col_var] = str(len(valid))
                table.add_row(r_row)
                if opts["sig"]:
                    table.add_row(p_row)
                    table.add_row(n_row)

            if opts["spearman"]:
                rho_row = {"var": row_var if not opts["pearson"] else "", "stat": "Spearman's rho"}
                p_rho_row = {"var": "", "stat": "p-value"}
                for j, col_var in enumerate(v_list):
                    if col_var not in df.columns or j > i:
                        rho_row[col_var] = ""
                        p_rho_row[col_var] = ""
                        continue
                    if i == j:
                        rho_row[col_var] = "—"
                        p_rho_row[col_var] = "—"
                        continue
                    valid = df[[row_var, col_var]].apply(pd.to_numeric, errors="coerce").dropna()
                    if len(valid) >= 3:
                        rho_val, p_val = stats.spearmanr(valid[row_var], valid[col_var])
                        rho_row[col_var] = f"{rho_val:.3f}{_stars(p_val)}"
                        p_rho_row[col_var] = "< .001" if p_val < 0.001 else f"{p_val:.3f}"
                    else:
                        rho_row[col_var] = "—"
                        p_rho_row[col_var] = "—"
                table.add_row(rho_row)
                if opts["sig"]:
                    table.add_row(p_rho_row)

            if opts.get("kendall", False):
                tau_row = {"var": row_var if (not opts["pearson"] and not opts["spearman"]) else "", "stat": "Kendall's tau B"}
                p_tau_row = {"var": "", "stat": "p-value"}
                for j, col_var in enumerate(v_list):
                    if col_var not in df.columns or j > i:
                        tau_row[col_var] = ""
                        p_tau_row[col_var] = ""
                        continue
                    if i == j:
                        tau_row[col_var] = "—"
                        p_tau_row[col_var] = "—"
                        continue
                    valid = df[[row_var, col_var]].apply(pd.to_numeric, errors="coerce").dropna()
                    if len(valid) >= 3:
                        tau_val, p_val = stats.kendalltau(valid[row_var], valid[col_var])
                        tau_row[col_var] = f"{tau_val:.3f}{_stars(p_val)}"
                        p_tau_row[col_var] = "< .001" if p_val < 0.001 else f"{p_val:.3f}"
                    else:
                        tau_row[col_var] = "—"
                        p_tau_row[col_var] = "—"
                table.add_row(tau_row)
                if opts["sig"]:
                    table.add_row(p_tau_row)

        if opts["flag"]:
            table.add_note("* p < .05, ** p < .01, *** p < .001")

        self.add_table(table)
        if opts.get("plots", False) and len(v_list) >= 2:
            method = "spearman" if (opts["spearman"] and not opts["pearson"]) else "pearson"
            self.add_plot("Correlation Matrix Plot", plot_correlation_heatmap(df, v_list, method=method))

        self.apa_narrative = (
            f"Bivariate correlation analysis highlighted notable associations between {'; '.join(apa_highlights[:6])}."
            if apa_highlights else "Bivariate correlations showed no strong associations."
        )

"""
pyjamovi Contingency Tables Module (v2.0)
Faithful clean-room Python implementation of jamovi jmv::contTables (Chi-Square, Likelihood Ratio, Fisher's Exact, Odds Ratio, Phi/Cramer's V, Bar Plot).
"""

from typing import Optional
import io
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from .base import BaseAnalysis, ResultsTable
from .plots import fig_to_base64, _clean_ax, JMV_PALETTE, JMV_NAVY


class ContTables(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        rows: str,
        cols: str,
        counts: Optional[str] = None,
        chi_sq: bool = True,
        chi_sq_corr: bool = False,
        like_rat: bool = False,
        fisher: bool = False,
        phi_cra: bool = True,
        odds: bool = False,
        obs: bool = True,
        exp: bool = False,
        pc_row: bool = False,
        pc_col: bool = False,
        bar_plot: bool = False
    ):
        super().__init__(
            name="contTables",
            title="Contingency Tables",
            dataset=data,
            options={
                "rows": rows,
                "cols": cols,
                "counts": counts,
                "chiSq": chi_sq,
                "chiSqCorr": chi_sq_corr,
                "likeRat": like_rat,
                "fisher": fisher,
                "phiCra": phi_cra,
                "odds": odds,
                "obs": obs,
                "exp": exp,
                "pcRow": pc_row,
                "pcCol": pc_col,
                "barPlot": bar_plot
            }
        )
        self.rows_var = rows
        self.cols_var = cols
        self.counts_var = counts
        self.syntax_python = f"jmv.contTables(df, rows={rows!r}, cols={cols!r}, chi_sq={chi_sq}, fisher={fisher}, phi_cra={phi_cra})"
        self.syntax_r = f"jmv::contTables(formula=~{rows}:{cols}, data=data, phiCra={str(phi_cra).upper()}, fisher={str(fisher).upper()})"
        self.run()

    def run(self):
        df = self.dataset
        opts = self.options
        r_var = self.rows_var
        c_var = self.cols_var

        if r_var not in df.columns or c_var not in df.columns:
            raise ValueError(f"Variables '{r_var}' or '{c_var}' not found in dataset")

        if self.counts_var and self.counts_var in df.columns:
            clean_df = df[[r_var, c_var, self.counts_var]].dropna()
            crosstab = pd.pivot_table(clean_df, values=self.counts_var, index=r_var, columns=c_var, aggfunc="sum", fill_value=0)
        else:
            clean_df = df[[r_var, c_var]].dropna()
            crosstab = pd.crosstab(clean_df[r_var], clean_df[c_var])

        if crosstab.size == 0:
            raise ValueError("Insufficient observations for contingency table")

        chi2, p_val, dof, expected = stats.chi2_contingency(crosstab, correction=False)

        ct_table = ResultsTable(name="crosstab", title=f"Contingency Tables — {r_var} × {c_var}")
        ct_table.add_column("row_val", r_var, col_type="text")
        ct_table.add_column("stat_type", "", col_type="text")
        for col_name in crosstab.columns:
            ct_table.add_column(str(col_name), str(col_name), col_type="number")
        ct_table.add_column("total", "Total", col_type="number")

        row_totals = crosstab.sum(axis=1)
        col_totals = crosstab.sum(axis=0)
        grand_total = crosstab.values.sum()

        for idx, r_level in enumerate(crosstab.index):
            first = True
            if opts["obs"]:
                row_obs = {"row_val": str(r_level), "stat_type": "Observed"}
                for c_level in crosstab.columns:
                    row_obs[str(c_level)] = int(crosstab.loc[r_level, c_level])
                row_obs["total"] = int(row_totals[r_level])
                ct_table.add_row(row_obs)
                first = False

            if opts["exp"]:
                row_exp = {"row_val": str(r_level) if first else "", "stat_type": "Expected"}
                for j_idx, c_level in enumerate(crosstab.columns):
                    row_exp[str(c_level)] = float(expected[idx, j_idx])
                row_exp["total"] = float(row_totals[r_level])
                ct_table.add_row(row_exp)
                first = False

            if opts["pcRow"]:
                row_pcr = {"row_val": str(r_level) if first else "", "stat_type": "% within row"}
                for c_level in crosstab.columns:
                    pct = (crosstab.loc[r_level, c_level] / max(1, row_totals[r_level])) * 100.0
                    row_pcr[str(c_level)] = f"{pct:.1f}%"
                row_pcr["total"] = "100.0%"
                ct_table.add_row(row_pcr)
                first = False

            if opts["pcCol"]:
                row_pcc = {"row_val": str(r_level) if first else "", "stat_type": "% within column"}
                for c_level in crosstab.columns:
                    pct = (crosstab.loc[r_level, c_level] / max(1, col_totals[c_level])) * 100.0
                    row_pcc[str(c_level)] = f"{pct:.1f}%"
                row_pcc["total"] = f"{(row_totals[r_level] / max(1, grand_total)) * 100.0:.1f}%"
                ct_table.add_row(row_pcc)

        tot_row = {"row_val": "Total", "stat_type": "Observed"}
        for c_level in crosstab.columns:
            tot_row[str(c_level)] = int(col_totals[c_level])
        tot_row["total"] = int(grand_total)
        ct_table.add_row(tot_row)

        test_table = ResultsTable(name="tests", title="χ² Tests")
        test_table.add_column("test", "Test", col_type="text")
        test_table.add_column("val", "Value", col_type="number")
        test_table.add_column("df", "df", col_type="integer")
        test_table.add_column("p", "p", col_type="number", fmt="pvalue")

        if opts["chiSq"]:
            test_table.add_row({"test": "χ²", "val": float(chi2), "df": int(dof), "p": float(p_val)})
        if opts.get("chiSqCorr", False) and crosstab.shape == (2, 2):
            c_yates, p_yates, _, _ = stats.chi2_contingency(crosstab, correction=True)
            test_table.add_row({"test": "χ² continuity correction", "val": float(c_yates), "df": int(dof), "p": float(p_yates)})
        if opts["likeRat"]:
            g_stat, g_p, _, _ = stats.chi2_contingency(crosstab, lambda_="log-likelihood")
            test_table.add_row({"test": "Likelihood ratio", "val": float(g_stat), "df": int(dof), "p": float(g_p)})
        if opts.get("fisher", False) and crosstab.shape == (2, 2):
            or_f, p_f = stats.fisher_exact(crosstab.values)
            test_table.add_row({"test": "Fisher's exact test", "val": float(or_f), "df": "", "p": float(p_f)})
        test_table.add_row({"test": "N", "val": int(grand_total), "df": "", "p": ""})

        self.add_table(ct_table)
        self.add_table(test_table)

        if opts["phiCra"]:
            nom_table = ResultsTable(name="nominal", title="Nominal")
            nom_table.add_column("measure", "Metric", col_type="text")
            nom_table.add_column("val", "Value", col_type="number", fmt="zto")
            phi = np.sqrt(chi2 / grand_total) if grand_total > 0 and crosstab.shape == (2, 2) else np.nan
            k_min = min(crosstab.shape)
            cramer_v = np.sqrt(chi2 / (grand_total * max(1, k_min - 1))) if grand_total > 0 else np.nan
            nom_table.add_row({"measure": "Phi-coefficient", "val": float(phi)})
            nom_table.add_row({"measure": "Cramér's V", "val": float(cramer_v)})
            self.add_table(nom_table)

        if opts.get("odds", False) and crosstab.shape == (2, 2):
            a, b = crosstab.values[0, 0] + 0.5, crosstab.values[0, 1] + 0.5
            c, d = crosstab.values[1, 0] + 0.5, crosstab.values[1, 1] + 0.5
            or_val = (crosstab.values[0, 0] * crosstab.values[1, 1]) / max(1e-9, crosstab.values[0, 1] * crosstab.values[1, 0])
            log_or = np.log((a * d) / (b * c))
            se_lor = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
            or_tbl = ResultsTable(name="odds", title="Comparative Measures (2×2)")
            or_tbl.add_column("metric", "Measure", col_type="text")
            or_tbl.add_column("val", "Value", col_type="number")
            or_tbl.add_column("cil", "95% CI Lower", col_type="number")
            or_tbl.add_column("ciu", "95% CI Upper", col_type="number")
            or_tbl.add_row({
                "metric": "Odds ratio",
                "val": float(or_val),
                "cil": float(np.exp(log_or - 1.96 * se_lor)),
                "ciu": float(np.exp(log_or + 1.96 * se_lor))
            })
            self.add_table(or_tbl)

        if opts.get("barPlot", False):
            fig, ax = plt.subplots(figsize=(6.2, 4.0))
            crosstab.plot(kind="bar", ax=ax, color=JMV_PALETTE[:len(crosstab.columns)], edgecolor="white", width=0.72)
            ax.set_title(f"Clustered Bar Plot — {r_var} by {c_var}", color=JMV_NAVY)
            ax.set_xlabel(r_var)
            ax.set_ylabel("Counts")
            ax.tick_params(axis="x", rotation=0)
            ax.legend(title=c_var, frameon=False)
            _clean_ax(ax)
            self.add_plot(f"Bar Plot — {r_var} × {c_var}", fig_to_base64(fig))

        p_str = "< .001" if p_val < 0.001 else f"= {p_val:.3f}"
        sig = "statistically significant" if p_val < 0.05 else "no statistically significant"
        self.apa_narrative = (
            f"A chi-square test of independence was performed between {r_var} and {c_var}. "
            f"There was {sig} relationship, χ²({dof}, N = {int(grand_total)}) = {chi2:.2f}, p {p_str}."
        )

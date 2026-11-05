"""
pyjamovi Panel Data Econometrics Module (v2.0)
Implements Fixed Effects (Entity, Time, Two-Way), Random Effects, Pooled OLS,
and Automatic Hausman Specification Test (FE vs RE) via linearmodels.
"""

from typing import List, Union
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS

from .base import BaseAnalysis, ResultsTable
from .plots import plot_panel_trends


class PanelAnalysis(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        dep: str,
        covs: Union[str, List[str]],
        model_type: str = "fixed",   # "fixed", "twoway", "random", "pooled"
        cov_type: str = "robust",    # "unadjusted", "robust", "clustered"
        hausman: bool = True,
        plot_trends: bool = True
    ):
        cov_list = [covs] if isinstance(covs, str) else list(covs)
        nice_map = {
            "fixed": "Entity Fixed Effects (Within)",
            "twoway": "Two-Way Fixed Effects (Entity + Time)",
            "random": "Random Effects (GLS)",
            "pooled": "Pooled OLS"
        }
        m_label = nice_map.get(model_type, model_type.title())
        super().__init__(
            name="panel",
            title=f"Panel Data Regression — {m_label}",
            dataset=data,
            options={
                "entity": entity_col,
                "time": time_col,
                "dep": dep,
                "covs": cov_list,
                "modelType": model_type,
                "covType": cov_type,
                "hausman": hausman,
                "plotTrends": plot_trends
            }
        )
        self.entity_col = entity_col
        self.time_col = time_col
        self.dep = dep
        self.covs = cov_list
        self.model_type = model_type
        self.cov_type = cov_type
        self.syntax_python = (
            f"jmv.panel(df, entity_col={entity_col!r}, time_col={time_col!r}, "
            f"dep={dep!r}, covs={cov_list!r}, model_type={model_type!r})"
        )
        self.syntax_r = (
            f"plm::plm({dep} ~ {' + '.join(cov_list)}, data=data, "
            f"index=c({entity_col!r}, {time_col!r}), model={('within' if 'fixed' in model_type or model_type=='twoway' else model_type)!r})"
        )
        self.run()

    def run(self):
        df = self.dataset.copy()
        req_cols = [self.entity_col, self.time_col, self.dep] + self.covs
        clean_df = df[req_cols].dropna().copy()

        # Time index must be numeric or datetime for linearmodels PanelData
        if not pd.api.types.is_numeric_dtype(clean_df[self.time_col]) and not pd.api.types.is_datetime64_any_dtype(clean_df[self.time_col]):
            clean_df["_time_idx"] = pd.Categorical(clean_df[self.time_col]).codes + 1
            time_idx_col = "_time_idx"
        else:
            time_idx_col = self.time_col

        panel_df = clean_df.set_index([self.entity_col, time_idx_col])
        y = panel_df[self.dep].astype(float)
        X_raw = panel_df[self.covs].astype(float)
        X_const = sm.add_constant(X_raw, has_constant="add")

        fit_kwargs = {}
        if self.cov_type == "clustered":
            fit_kwargs = {"cov_type": "clustered", "cluster_entity": True}
        elif self.cov_type == "robust":
            fit_kwargs = {"cov_type": "robust"}
        else:
            fit_kwargs = {"cov_type": "unadjusted"}

        if self.model_type == "fixed":
            model = PanelOLS(y, X_const, entity_effects=True)
        elif self.model_type == "twoway":
            model = PanelOLS(y, X_const, entity_effects=True, time_effects=True)
        elif self.model_type == "pooled":
            model = PooledOLS(y, X_const)
        else:
            model = RandomEffects(y, X_const)

        res = model.fit(**fit_kwargs)

        # 1. Model Summary Table
        sum_table = ResultsTable(name="summary", title="Panel Model Fit Summary")
        sum_table.add_column("metric", "Metric", col_type="text")
        sum_table.add_column("val", "Value", col_type="number")

        n_entities = clean_df[self.entity_col].nunique()
        n_periods = clean_df[self.time_col].nunique()

        sum_table.add_row({"metric": "R² (Overall)", "val": float(res.rsquared_overall)})
        sum_table.add_row({"metric": "R² (Within)", "val": float(res.rsquared_within)})
        sum_table.add_row({"metric": "R² (Between)", "val": float(res.rsquared_between)})
        sum_table.add_row({"metric": "F-statistic", "val": float(res.f_statistic.stat)})
        sum_table.add_row({"metric": "F p-value", "val": float(res.f_statistic.pval)})
        sum_table.add_row({"metric": "Entities (N)", "val": int(n_entities)})
        sum_table.add_row({"metric": "Time Periods (T)", "val": int(n_periods)})
        sum_table.add_row({"metric": "Total Observations", "val": int(res.nobs)})
        self.add_table(sum_table)

        # 2. Parameter Estimates Table
        coef_table = ResultsTable(name="coef", title="Parameter Estimates")
        coef_table.add_column("var", "Variable", col_type="text")
        coef_table.add_column("b", "Estimate", col_type="number")
        coef_table.add_column("se", "Std. Error", col_type="number")
        coef_table.add_column("t", "t-statistic", col_type="number")
        coef_table.add_column("p", "p-value", col_type="number", fmt="pvalue")
        coef_table.add_column("cil", "95% CI Lower", col_type="number")
        coef_table.add_column("ciu", "95% CI Upper", col_type="number")

        ci_df = res.conf_int()
        for param_name in res.params.index:
            lbl = "Intercept" if param_name == "const" else param_name
            coef_table.add_row({
                "var": lbl,
                "b": float(res.params[param_name]),
                "se": float(res.std_errors[param_name]),
                "t": float(res.tstats[param_name]),
                "p": float(res.pvalues[param_name]),
                "cil": float(ci_df.loc[param_name, "lower"]),
                "ciu": float(ci_df.loc[param_name, "upper"])
            })
        coef_table.add_note(f"Standard errors: {self.cov_type}.")
        coef_table.mark_coefficients(estimate="b", se="se", p="p", stat="t", ci=("cil", "ciu"), label="var")
        self.add_table(coef_table)
        self.model_stats = {"N": int(res.nobs), "Groups": int(n_entities), "Periods": int(n_periods),
                            "R2_within": float(res.rsquared_within), "R2_between": float(res.rsquared_between),
                            "R2_overall": float(res.rsquared_overall),
                            "F": (float(res.f_statistic.stat), float(res.f_statistic.pval))}

        # 3. Hausman Specification Test (FE vs RE)
        hausman_note = ""
        if self.options.get("hausman", True) and len(self.covs) >= 1 and n_entities >= 3:
            try:
                fe_u = PanelOLS(y, X_raw, entity_effects=True).fit(cov_type="unadjusted")
                re_u = RandomEffects(y, sm.add_constant(X_raw, has_constant="add")).fit(cov_type="unadjusted")
                b_fe = fe_u.params[self.covs].values
                b_re = re_u.params[self.covs].values
                v_fe = fe_u.cov.loc[self.covs, self.covs].values
                v_re = re_u.cov.loc[self.covs, self.covs].values
                diff_b = b_fe - b_re
                diff_v = v_fe - v_re
                chi2_h = float(diff_b.T @ np.linalg.pinv(diff_v) @ diff_b)
                chi2_h = max(0.0, chi2_h)
                df_h = len(self.covs)
                p_h = float(1.0 - stats.chi2.cdf(chi2_h, df_h))
                rec = "Fixed Effects preferred (p < .05)" if p_h < 0.05 else "Random Effects consistent (p ≥ .05)"

                h_tbl = ResultsTable(name="hausman", title="Hausman Specification Test (Fixed vs. Random Effects)")
                h_tbl.add_column("chi2", "χ²", col_type="number")
                h_tbl.add_column("df", "df", col_type="integer")
                h_tbl.add_column("p", "p", col_type="number", fmt="pvalue")
                h_tbl.add_column("decision", "Diagnostic Decision", col_type="text")
                h_tbl.add_row({"chi2": chi2_h, "df": df_h, "p": p_h, "decision": rec})
                self.add_table(h_tbl)
                hausman_note = f" Hausman test yielded χ²({df_h}) = {chi2_h:.2f}, p = {p_h:.3f} ({rec})."
            except Exception:
                pass

        if self.options.get("plotTrends", True):
            self.add_plot("Panel Entity Trajectories", plot_panel_trends(clean_df, self.entity_col, self.time_col, self.dep))

        p_str = "< .001" if res.f_statistic.pval < 0.001 else f"= {res.f_statistic.pval:.3f}"
        self.apa_narrative = (
            f"Panel regression ({self.model_type}, N = {n_entities} entities × T = {n_periods} periods) "
            f"was significant (F = {res.f_statistic.stat:.2f}, p {p_str}), with Within R² = {res.rsquared_within:.3f} "
            f"and Overall R² = {res.rsquared_overall:.3f}.{hausman_note}"
        )

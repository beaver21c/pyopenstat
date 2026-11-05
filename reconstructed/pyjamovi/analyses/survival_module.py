"""
pyjamovi Multilevel Mixed Models & Survival Analysis Suite (v2.0)
- MixedLinearModel (GAMLj Mixed): Random Intercepts / Multilevel Linear Mixed Model, ICC, Fixed & Random Effects
- SurvivalAnalysis (jsurvival / deathwatch): Kaplan-Meier Survival Table, Log-Rank Test, Cox Proportional Hazards (HR) & Survival Plot
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import restricted_mean_survival_time

from .base import BaseAnalysis, ResultsTable
from .plots import plot_kaplan_meier


class MixedLinearModel(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        dep: str,
        cluster: str,
        covs: Optional[Union[str, List[str]]] = None,
        factors: Optional[Union[str, List[str]]] = None,
        reml: bool = True
    ):
        cov_list = [] if covs is None else ([covs] if isinstance(covs, str) else list(covs))
        fac_list = [] if factors is None else ([factors] if isinstance(factors, str) else list(factors))
        super().__init__(
            name="mixedModel",
            title="Linear Mixed Model (Multilevel)",
            dataset=data,
            options={"dep": dep, "cluster": cluster, "covs": cov_list, "factors": fac_list, "reml": reml}
        )
        self.dep = dep
        self.cluster = cluster
        self.covs = cov_list
        self.factors = fac_list
        self.reml = reml
        self.syntax_python = f"jmv.mixedModel(df, dep={dep!r}, cluster={cluster!r}, covs={cov_list!r}, factors={fac_list!r})"
        self.syntax_r = f"gamlj::gamljMixed(formula={dep} ~ {' + '.join(cov_list + fac_list) or '1'} + (1 | {cluster}), data=data)"
        self.run()

    def run(self):
        preds = self.covs + [f for f in self.factors if f not in self.covs]
        cols = [self.dep, self.cluster] + preds
        clean = self.dataset[cols].dropna().copy()
        clean[self.dep] = pd.to_numeric(clean[self.dep], errors="coerce")
        clean = clean.dropna()

        y = clean[self.dep].astype(float)
        groups = clean[self.cluster].astype(str)

        X_parts = []
        for p in preds:
            if p in self.factors or not pd.api.types.is_numeric_dtype(clean[p]):
                X_parts.append(pd.get_dummies(clean[p].astype(str), prefix=p, drop_first=True, dtype=float))
            else:
                X_parts.append(clean[[p]].astype(float))
        X = sm.add_constant(pd.concat(X_parts, axis=1) if X_parts else pd.DataFrame(index=clean.index), has_constant="add")

        mod = MixedLM(y, X, groups=groups)
        res = mod.fit(reml=self.reml)

        var_re = float(res.cov_re.iloc[0, 0]) if hasattr(res.cov_re, "iloc") else float(np.asarray(res.cov_re).ravel()[0])
        var_resid = float(res.scale)
        icc = var_re / (var_re + var_resid) if (var_re + var_resid) > 0 else 0.0

        # 1. Model Info & ICC
        info_tbl = ResultsTable(name="info", title="Model Info & Fit")
        info_tbl.add_column("info", "Info", col_type="text")
        info_tbl.add_column("val", "Value", col_type="text")
        info_tbl.add_row({"info": "Estimation", "val": "REML" if self.reml else "ML"})
        info_tbl.add_row({"info": "Observations (N)", "val": str(len(clean))})
        info_tbl.add_row({"info": f"Clusters ({self.cluster})", "val": str(groups.nunique())})
        info_tbl.add_row({"info": "Log-Likelihood", "val": f"{res.llf:.3f}"})
        info_tbl.add_row({"info": "Intraclass Correlation (ICC)", "val": f"{icc:.3f}"})
        self.add_table(info_tbl)

        # 2. Fixed Effects Parameter Estimates
        fe_tbl = ResultsTable(name="fixed", title="Fixed Effects Parameter Estimates")
        fe_tbl.add_column("name", "Name", col_type="text")
        fe_tbl.add_column("est", "Estimate", col_type="number")
        fe_tbl.add_column("se", "SE", col_type="number")
        fe_tbl.add_column("cil", "95% CI Lower", col_type="number")
        fe_tbl.add_column("ciu", "95% CI Upper", col_type="number")
        fe_tbl.add_column("z", "z", col_type="number")
        fe_tbl.add_column("p", "p", col_type="number", fmt="pvalue")

        conf = res.conf_int()
        for col_n in X.columns:
            lbl = "(Intercept)" if col_n == "const" else col_n
            fe_tbl.add_row({
                "name": lbl,
                "est": float(res.fe_params[col_n]),
                "se": float(res.bse_fe[col_n]),
                "cil": float(conf.loc[col_n, 0]),
                "ciu": float(conf.loc[col_n, 1]),
                "z": float(res.tvalues[col_n]),
                "p": float(res.pvalues[col_n])
            })
        fe_tbl.mark_coefficients(estimate="est", se="se", p="p", stat="z", ci=("cil", "ciu"), label="name")
        self.add_table(fe_tbl)
        self.model_stats = {"N": int(len(clean)), "Groups": int(groups.nunique()), "LL": float(res.llf),
                            "ICC": float(icc), "Var_between": var_re, "Var_within": var_resid}

        # 3. Random Components Table
        re_tbl = ResultsTable(name="random", title="Random Components")
        re_tbl.add_column("grp", "Groups", col_type="text")
        re_tbl.add_column("name", "Name", col_type="text")
        re_tbl.add_column("sd", "SD", col_type="number")
        re_tbl.add_column("var", "Variance", col_type="number")
        re_tbl.add_column("icc", "ICC", col_type="number", fmt="zto")
        re_tbl.add_row({"grp": self.cluster, "name": "(Intercept)", "sd": float(np.sqrt(max(0.0, var_re))), "var": var_re, "icc": icc})
        re_tbl.add_row({"grp": "Residual", "name": "", "sd": float(np.sqrt(max(0.0, var_resid))), "var": var_resid, "icc": ""})
        self.add_table(re_tbl)

        self.apa_narrative = (
            f"Linear Mixed Model ({'REML' if self.reml else 'ML'}) clustering on '{self.cluster}' "
            f"({groups.nunique()} groups, N = {len(clean)}) estimated ICC = {icc:.3f}."
        )


class SurvivalAnalysis(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        time_col: str,
        event_col: str,
        group: Optional[str] = None,
        covs: Optional[Union[str, List[str]]] = None,
        km_plot: bool = True
    ):
        cov_list = [] if covs is None else ([covs] if isinstance(covs, str) else list(covs))
        super().__init__(
            name="survival",
            title="Survival Analysis (Kaplan-Meier & Cox Proportional Hazards)",
            dataset=data,
            options={"time": time_col, "event": event_col, "group": group, "covs": cov_list, "kmPlot": km_plot}
        )
        self.time_col = time_col
        self.event_col = event_col
        self.group = group
        self.covs = cov_list
        self.syntax_python = f"jmv.survival(df, time_col={time_col!r}, event_col={event_col!r}, group={group!r}, covs={cov_list!r})"
        self.syntax_r = f"jsurvival::survival(data=data, elapsedtime={time_col}, outcome={event_col}, explanatory={group!r})"
        self.run()

    def run(self):
        cols = [self.time_col, self.event_col] + ([self.group] if self.group and self.group in self.dataset.columns else [])
        for c in self.covs:
            if c not in cols and c in self.dataset.columns:
                cols.append(c)

        clean = self.dataset[cols].dropna().copy()
        clean[self.time_col] = pd.to_numeric(clean[self.time_col], errors="coerce")
        # Support 0/1 or categorical event indicator
        ev_s = clean[self.event_col]
        if pd.api.types.is_numeric_dtype(ev_s):
            clean["_event"] = (ev_s.astype(float) > 0).astype(int)
        else:
            lvls = sorted(ev_s.astype(str).unique())
            clean["_event"] = (ev_s.astype(str) == lvls[-1]).astype(int)
        clean = clean.dropna(subset=[self.time_col])
        clean = clean[clean[self.time_col] >= 0]

        km_tbl = ResultsTable(name="km", title="Kaplan-Meier Median Survival Summary")
        km_tbl.add_column("grp", self.group or "Cohort", col_type="text")
        km_tbl.add_column("n", "Total N", col_type="integer")
        km_tbl.add_column("events", "Events", col_type="integer")
        km_tbl.add_column("censored", "Censored", col_type="integer")
        km_tbl.add_column("median", "Median Survival", col_type="number")
        km_tbl.add_column("rmst", "Restricted Mean", col_type="number")

        km_curves = []
        kmf = KaplanMeierFitter()

        if self.group and self.group in clean.columns and clean[self.group].nunique() >= 2:
            for g_val in sorted(clean[self.group].unique(), key=str):
                sub = clean[clean[self.group] == g_val]
                kmf.fit(sub[self.time_col], event_observed=sub["_event"], label=str(g_val))
                ci = kmf.confidence_interval_survival_function_
                km_curves.append({
                    "label": f"{self.group}: {g_val}",
                    "timeline": kmf.survival_function_.index.values,
                    "surv": kmf.survival_function_.iloc[:, 0].values,
                    "ci_low": ci.iloc[:, 0].values,
                    "ci_high": ci.iloc[:, 1].values,
                })
                med = kmf.median_survival_time_
                try:
                    rmst_val = float(restricted_mean_survival_time(kmf))
                except Exception:
                    rmst_val = float(sub[self.time_col].mean())
                km_tbl.add_row({
                    "grp": str(g_val),
                    "n": len(sub),
                    "events": int(sub["_event"].sum()),
                    "censored": int((1 - sub["_event"]).sum()),
                    "median": float(med) if np.isfinite(med) else np.nan,
                    "rmst": rmst_val
                })
            self.add_table(km_tbl)

            lr = multivariate_logrank_test(clean[self.time_col], clean[self.group], clean["_event"])
            lr_tbl = ResultsTable(name="logrank", title="Log-Rank Test for Equality of Survival Curves")
            lr_tbl.add_column("chi2", "χ²", col_type="number")
            lr_tbl.add_column("df", "df", col_type="integer")
            lr_tbl.add_column("p", "p", col_type="number", fmt="pvalue")
            lr_tbl.add_row({"chi2": float(lr.test_statistic), "df": int(lr.degrees_of_freedom), "p": float(lr.p_value)})
            self.add_table(lr_tbl)
            lr_note = f"Log-rank χ²({int(lr.degrees_of_freedom)}) = {lr.test_statistic:.2f}, p = {lr.p_value:.3f}."
        else:
            kmf.fit(clean[self.time_col], event_observed=clean["_event"], label="All Observations")
            ci = kmf.confidence_interval_survival_function_
            km_curves.append({
                "label": "Overall Cohort",
                "timeline": kmf.survival_function_.index.values,
                "surv": kmf.survival_function_.iloc[:, 0].values,
                "ci_low": ci.iloc[:, 0].values,
                "ci_high": ci.iloc[:, 1].values,
            })
            med = kmf.median_survival_time_
            try:
                rmst_overall = float(restricted_mean_survival_time(kmf))
            except Exception:
                rmst_overall = float(clean[self.time_col].mean())
            km_tbl.add_row({
                "grp": "Overall",
                "n": len(clean),
                "events": int(clean["_event"].sum()),
                "censored": int((1 - clean["_event"]).sum()),
                "median": float(med) if np.isfinite(med) else np.nan,
                "rmst": rmst_overall
            })
            self.add_table(km_tbl)
            lr_note = f"Overall cohort N = {len(clean)} ({int(clean['_event'].sum())} events)."

        # Cox Proportional Hazards Regression (when group or covariates provided)
        cox_preds = []
        if self.group and self.group in clean.columns:
            cox_preds.append(self.group)
        for c in self.covs:
            if c not in cox_preds and c in clean.columns:
                cox_preds.append(c)

        cox_note = ""
        if cox_preds:
            try:
                cox_parts = [clean[[self.time_col, "_event"]].copy()]
                for p in cox_preds:
                    if pd.api.types.is_numeric_dtype(clean[p]) and p != self.group:
                        cox_parts.append(clean[[p]].astype(float))
                    else:
                        cox_parts.append(pd.get_dummies(clean[p].astype(str), prefix=p, drop_first=True, dtype=float))
                cox_df = pd.concat(cox_parts, axis=1).dropna()

                cph = CoxPHFitter()
                cph.fit(cox_df, duration_col=self.time_col, event_col="_event")
                summ = cph.summary

                cox_tbl = ResultsTable(name="cox", title=f"Cox Proportional Hazards Regression (Concordance = {cph.concordance_index_:.3f})")
                cox_tbl.add_column("var", "Covariate", col_type="text")
                cox_tbl.add_column("coef", "log(HR)", col_type="number")
                cox_tbl.add_column("hr", "Hazard Ratio exp(β)", col_type="number")
                cox_tbl.add_column("se", "SE", col_type="number")
                cox_tbl.add_column("cil", "95% CI Lower", col_type="number")
                cox_tbl.add_column("ciu", "95% CI Upper", col_type="number")
                cox_tbl.add_column("z", "z", col_type="number")
                cox_tbl.add_column("p", "p", col_type="number", fmt="pvalue")

                for cov_name, row in summ.iterrows():
                    cox_tbl.add_row({
                        "var": str(cov_name),
                        "coef": float(row["coef"]),
                        "hr": float(row["exp(coef)"]),
                        "se": float(row["se(coef)"]),
                        "cil": float(row["exp(coef) lower 95%"]),
                        "ciu": float(row["exp(coef) upper 95%"]),
                        "z": float(row["z"]),
                        "p": float(row["p"])
                    })
                cox_tbl.mark_coefficients(estimate="coef", se="se", p="p", stat="z", exp_estimate="hr", label="var")
                self.add_table(cox_tbl)
                self.model_stats = {"N": int(len(cox_df)), "Events": int(cox_df["_event"].sum()),
                                    "LL": float(cph.log_likelihood_), "Concordance": float(cph.concordance_index_)}
                try:
                    self.model_stats["AIC"] = float(cph.AIC_partial_)
                except Exception:
                    pass
                cox_note = f" Cox PH concordance C-index = {cph.concordance_index_:.3f}."
            except Exception:
                pass

        if self.options.get("kmPlot", True) and km_curves:
            self.add_plot("Kaplan-Meier Survival Plot", plot_kaplan_meier(km_curves, self.time_col))

        self.apa_narrative = f"Survival analysis on {self.time_col} ({self.event_col}): {lr_note}{cox_note}"

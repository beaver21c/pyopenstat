"""
pyjamovi Linear Regression Module (v2.0)
Faithful clean-room Python implementation of jamovi jmv::linReg:
- Supports single or hierarchical block entry (ΔR², ΔF)
- Factors (dummy-coded categorical predictors) + Continuous Covariates
- Standardized Estimates (β), 95% CIs, VIF & Tolerance, Durbin-Watson, Shapiro-Wilk, Cook's Distance, 2x2 Diagnostic Plots
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .base import BaseAnalysis, ResultsTable
from .plots import plot_regression_diagnostics


class LinReg(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        dep: str,
        covs: Optional[Union[str, List[str]]] = None,
        factors: Optional[Union[str, List[str]]] = None,
        blocks: Optional[List[List[str]]] = None,
        r2_adj: bool = True,
        aic_bic: bool = False,
        f_test: bool = True,
        std_est: bool = True,
        ci: bool = False,
        collin: bool = False,
        dw: bool = False,
        norm: bool = False,
        cooks: bool = False,
        plots_diag: bool = False
    ):
        cov_list = [] if covs is None else ([covs] if isinstance(covs, str) else list(covs))
        fac_list = [] if factors is None else ([factors] if isinstance(factors, str) else list(factors))

        if blocks is None:
            all_preds = cov_list + [f for f in fac_list if f not in cov_list]
            resolved_blocks = [all_preds] if all_preds else []
        else:
            resolved_blocks = [list(b) for b in blocks if b]

        super().__init__(
            name="linReg",
            title="Linear Regression",
            dataset=data,
            options={
                "dep": dep,
                "covs": cov_list,
                "factors": fac_list,
                "blocks": resolved_blocks,
                "r2Adj": r2_adj,
                "aicBic": aic_bic,
                "fTest": f_test,
                "stdEst": std_est,
                "ci": ci,
                "collin": collin,
                "dw": dw,
                "norm": norm,
                "cooks": cooks,
                "plotsDiag": plots_diag
            }
        )
        self.dep = dep
        self.covs = cov_list
        self.factors = fac_list
        self.blocks = resolved_blocks
        self.syntax_python = f"jmv.linReg(df, dep={dep!r}, covs={cov_list!r}, factors={fac_list!r}, collin={collin})"
        self.syntax_r = f"jmv::linReg(data=data, dep={dep}, covs=vars({', '.join(cov_list)}), blocks={resolved_blocks!r})"
        self.run()

    def _build_design(self, clean_df: pd.DataFrame, preds: List[str]) -> pd.DataFrame:
        parts = []
        for p in preds:
            if p in self.factors or not pd.api.types.is_numeric_dtype(clean_df[p]):
                dummies = pd.get_dummies(clean_df[p].astype(str), prefix=p, drop_first=True, dtype=float)
                parts.append(dummies)
            else:
                parts.append(clean_df[[p]].astype(float))
        X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=clean_df.index)
        return sm.add_constant(X, has_constant="add")

    def comparison_models(self, table=None):
        """Each block of a hierarchical regression is one model of the comparison table."""
        if table is None and getattr(self, "_block_models", None):
            return [dict(m) for m in self._block_models]
        return super().comparison_models(table)

    def run(self):
        df = self.dataset
        opts = self.options
        dep_var = self.dep

        all_vars = []
        for b in self.blocks:
            for v in b:
                if v not in all_vars:
                    all_vars.append(v)

        if dep_var not in df.columns:
            raise ValueError(f"Dependent variable '{dep_var}' not found")
        for v in all_vars:
            if v not in df.columns:
                raise ValueError(f"Predictor '{v}' not found")
        if not all_vars:
            raise ValueError("At least one covariate or factor is required")

        clean_df = df[[dep_var] + all_vars].dropna().copy()
        clean_df[dep_var] = pd.to_numeric(clean_df[dep_var], errors="coerce")
        clean_df = clean_df.dropna()
        y = clean_df[dep_var].astype(float)

        fitted_models = []
        cum_preds: List[str] = []
        for b_idx, block in enumerate(self.blocks):
            for v in block:
                if v not in cum_preds:
                    cum_preds.append(v)
            X_mat = self._build_design(clean_df, cum_preds)
            m = sm.OLS(y, X_mat).fit()
            fitted_models.append((str(b_idx + 1), list(cum_preds), X_mat, m))

        # 1. Model Fit Measures
        fit_table = ResultsTable(name="fit", title="Model Fit Measures")
        fit_table.add_column("model", "Model", col_type="text")
        fit_table.add_column("r", "R", col_type="number", fmt="zto")
        fit_table.add_column("r2", "R²", col_type="number", fmt="zto")
        if opts["r2Adj"]:
            fit_table.add_column("r2_adj", "Adjusted R²", col_type="number", fmt="zto")
        if opts["aicBic"]:
            fit_table.add_column("aic", "AIC", col_type="number")
            fit_table.add_column("bic", "BIC", col_type="number")
            fit_table.add_column("rmse", "RMSE", col_type="number")
        if opts["fTest"]:
            fit_table.add_column("f", "F", col_type="number")
            fit_table.add_column("df1", "df1", col_type="integer")
            fit_table.add_column("df2", "df2", col_type="integer")
            fit_table.add_column("p", "p", col_type="number", fmt="pvalue")

        for m_id, _, _, m in fitted_models:
            row = {
                "model": m_id,
                "r": float(np.sqrt(max(0.0, m.rsquared))),
                "r2": float(m.rsquared),
                "r2_adj": float(m.rsquared_adj),
                "aic": float(m.aic),
                "bic": float(m.bic),
                "rmse": float(np.sqrt(m.mse_resid)),
                "f": float(m.fvalue),
                "df1": int(m.df_model),
                "df2": int(m.df_resid),
                "p": float(m.f_pvalue)
            }
            fit_table.add_row(row)
        self.add_table(fit_table)

        # 1b. Hierarchical Model Comparisons (if > 1 block)
        if len(fitted_models) > 1:
            comp_tbl = ResultsTable(name="comparison", title="Model Comparisons")
            comp_tbl.add_column("m1", "Model 1", col_type="text")
            comp_tbl.add_column("m2", "Model 2", col_type="text")
            comp_tbl.add_column("dr2", "ΔR²", col_type="number", fmt="zto")
            comp_tbl.add_column("f", "F", col_type="number")
            comp_tbl.add_column("df1", "df1", col_type="integer")
            comp_tbl.add_column("df2", "df2", col_type="integer")
            comp_tbl.add_column("p", "p", col_type="number", fmt="pvalue")
            for k in range(1, len(fitted_models)):
                m_prev = fitted_models[k - 1][3]
                m_curr = fitted_models[k][3]
                f_chg, p_chg, df_diff = m_curr.compare_f_test(m_prev)
                comp_tbl.add_row({
                    "m1": fitted_models[k - 1][0],
                    "m2": fitted_models[k][0],
                    "dr2": float(m_curr.rsquared - m_prev.rsquared),
                    "f": float(f_chg),
                    "df1": int(df_diff),
                    "df2": int(m_curr.df_resid),
                    "p": float(p_chg)
                })
            self.add_table(comp_tbl)

        # Final model coefficients & diagnostics
        _, final_preds, X_final, model = fitted_models[-1]

        coef_table = ResultsTable(name="coef", title=f"Model Coefficients — {dep_var}")
        coef_table.add_column("pred", "Predictor", col_type="text")
        coef_table.add_column("b", "Estimate", col_type="number")
        coef_table.add_column("se", "SE", col_type="number")
        if opts["ci"]:
            coef_table.add_column("cil", "95% CI Lower", col_type="number")
            coef_table.add_column("ciu", "95% CI Upper", col_type="number")
        if opts["stdEst"]:
            coef_table.add_column("beta", "Stand. Estimate", col_type="number", fmt="zto")
        coef_table.add_column("t", "t", col_type="number")
        coef_table.add_column("p", "p", col_type="number", fmt="pvalue")
        if opts["collin"]:
            coef_table.add_column("vif", "VIF", col_type="number")
            coef_table.add_column("tol", "Tolerance", col_type="number", fmt="zto")

        y_std = y.std(ddof=1)
        conf = model.conf_int(alpha=0.05)

        vifs = {}
        if opts["collin"] and X_final.shape[1] > 1:
            for idx, col_n in enumerate(X_final.columns):
                if col_n == "const":
                    continue
                try:
                    v = float(variance_inflation_factor(X_final.values, idx))
                    vifs[col_n] = (v, 1.0 / v if v > 0 else np.nan)
                except Exception:
                    vifs[col_n] = (np.nan, np.nan)

        for col_n in X_final.columns:
            lbl = "Intercept" if col_n == "const" else col_n
            x_std = X_final[col_n].std(ddof=1) if col_n != "const" else 0.0
            beta_val = float(model.params[col_n] * (x_std / y_std)) if (col_n != "const" and y_std > 0) else ""
            row = {
                "pred": lbl,
                "b": float(model.params[col_n]),
                "se": float(model.bse[col_n]),
                "cil": float(conf.loc[col_n, 0]),
                "ciu": float(conf.loc[col_n, 1]),
                "beta": beta_val,
                "t": float(model.tvalues[col_n]),
                "p": float(model.pvalues[col_n]),
                "vif": "" if col_n == "const" else vifs.get(col_n, (np.nan, np.nan))[0],
                "tol": "" if col_n == "const" else vifs.get(col_n, (np.nan, np.nan))[1]
            }
            coef_table.add_row(row)
        coef_table.mark_coefficients(estimate="b", se="se", p="p", std_estimate="beta", stat="t",
                                     ci=("cil", "ciu"), label="pred")
        self.add_table(coef_table)

        # Per-block models for side-by-side (hierarchical) comparison tables
        self._block_models = []
        prev = None
        for m_id, preds_k, X_k, m_k in fitted_models:
            coefs = []
            for col_n in X_k.columns:
                x_sd = X_k[col_n].std(ddof=1) if col_n != "const" else 0.0
                coefs.append({
                    "term": "Intercept" if col_n == "const" else col_n,
                    "est": float(m_k.params[col_n]), "se": float(m_k.bse[col_n]), "p": float(m_k.pvalues[col_n]),
                    "std": float(m_k.params[col_n] * x_sd / y_std) if (col_n != "const" and y_std > 0) else None,
                    "stat": float(m_k.tvalues[col_n]), "exp": None,
                })
            st = {"N": int(m_k.nobs), "R2": float(m_k.rsquared), "R2_adj": float(m_k.rsquared_adj),
                  "F": (float(m_k.fvalue), float(m_k.f_pvalue)), "AIC": float(m_k.aic), "BIC": float(m_k.bic)}
            if prev is not None:
                f_chg, p_chg, _ = m_k.compare_f_test(prev)
                st["R2_change"] = (float(m_k.rsquared - prev.rsquared), float(p_chg))
                st["F_change"] = (float(f_chg), float(p_chg))
            elif len(fitted_models) > 1:     # first block: change from the empty model (SPSS R Square Change)
                st["R2_change"] = (float(m_k.rsquared), float(m_k.f_pvalue))
                st["F_change"] = (float(m_k.fvalue), float(m_k.f_pvalue))
            self._block_models.append({"name": "", "dep": dep_var, "coefs": coefs, "stats": st, "source": self.title})
            prev = m_k
        self.model_stats = dict(self._block_models[-1]["stats"])

        # Assumption Checks
        if opts["dw"]:
            dw_val = float(durbin_watson(model.resid))
            dw_tbl = ResultsTable(name="dw", title="Durbin-Watson Test for Autocorrelation")
            dw_tbl.add_column("auto", "Autocorrelation", col_type="number", fmt="zto")
            dw_tbl.add_column("dw", "DW Statistic", col_type="number")
            auto_r = float(pd.Series(model.resid).autocorr(lag=1)) if len(model.resid) > 2 else 0.0
            dw_tbl.add_row({"auto": auto_r, "dw": dw_val})
            self.add_table(dw_tbl)

        if opts["norm"] and 3 <= len(model.resid) <= 5000:
            w_s, p_s = stats.shapiro(model.resid)
            nt = ResultsTable(name="norm", title="Normality Test (Shapiro-Wilk)")
            nt.add_column("stat", "Statistic", col_type="number")
            nt.add_column("p", "p", col_type="number", fmt="pvalue")
            nt.add_row({"stat": float(w_s), "p": float(p_s)})
            self.add_table(nt)

        infl = model.get_influence()
        cooks_d = infl.cooks_distance[0]
        if opts["cooks"]:
            ct = ResultsTable(name="cooks", title="Cook's Distance Summary")
            ct.add_column("mean", "Mean", col_type="number")
            ct.add_column("med", "Median", col_type="number")
            ct.add_column("sd", "SD", col_type="number")
            ct.add_column("min", "Min", col_type="number")
            ct.add_column("max", "Max", col_type="number")
            ct.add_row({
                "mean": float(np.mean(cooks_d)),
                "med": float(np.median(cooks_d)),
                "sd": float(np.std(cooks_d, ddof=1)),
                "min": float(np.min(cooks_d)),
                "max": float(np.max(cooks_d))
            })
            self.add_table(ct)

        if opts.get("plotsDiag", False):
            self.add_plot("Regression Assumption & Influence Diagnostics", plot_regression_diagnostics(y.values, model.fittedvalues.values, cooks_d))

        p_str = "< .001" if model.f_pvalue < 0.001 else f"= {model.f_pvalue:.3f}"
        sig = "statistically significant" if model.f_pvalue < 0.05 else "not statistically significant"
        self.apa_narrative = (
            f"A multiple linear regression was calculated to predict {dep_var} based on {', '.join(all_vars)}. "
            f"A {sig} regression equation was found (F({int(model.df_model)}, {int(model.df_resid)}) = {model.fvalue:.2f}, "
            f"p {p_str}), with an R² of {model.rsquared:.3f} (Adjusted R² = {model.rsquared_adj:.3f})."
        )

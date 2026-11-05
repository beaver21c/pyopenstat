"""
pyjamovi Advanced Regression & Correlation Suite (v2.0)
- LogisticBinomial (jmv::logRegBin): Binomial Logistic Regression with Odds Ratios, McFadden/Nagelkerke R², Classification Table & ROC Curve
- PartialCorrelation (jmv::partCorr): Partial & Semi-Partial Correlation Matrix controlling for covariates
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import pingouin as pg

from .base import BaseAnalysis, ResultsTable
from .plots import plot_roc_curve


class LogisticBinomial(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        dep: str,
        covs: Optional[Union[str, List[str]]] = None,
        factors: Optional[Union[str, List[str]]] = None,
        odds_ratio: bool = True,
        ci: bool = True,
        class_table: bool = True,
        cutoff: float = 0.5,
        roc_plot: bool = True
    ):
        cov_list = [] if covs is None else ([covs] if isinstance(covs, str) else list(covs))
        fac_list = [] if factors is None else ([factors] if isinstance(factors, str) else list(factors))
        super().__init__(
            name="logRegBin",
            title="Binomial Logistic Regression",
            dataset=data,
            options={
                "dep": dep,
                "covs": cov_list,
                "factors": fac_list,
                "oddsRatio": odds_ratio,
                "ci": ci,
                "classTable": class_table,
                "cutoff": float(cutoff),
                "rocPlot": roc_plot
            }
        )
        self.dep = dep
        self.covs = cov_list
        self.factors = fac_list
        self.cutoff = float(cutoff)
        self.syntax_python = f"jmv.logRegBin(df, dep={dep!r}, covs={cov_list!r}, factors={fac_list!r})"
        self.syntax_r = f"jmv::logRegBin(data=data, dep={dep}, covs=vars({', '.join(cov_list)}), odds={str(odds_ratio).upper()}, rocPlot={str(roc_plot).upper()})"
        self.run()

    def run(self):
        preds = self.covs + [f for f in self.factors if f not in self.covs]
        if not preds:
            raise ValueError("Provide at least one covariate or factor for Binomial Logistic Regression.")

        clean = self.dataset[[self.dep] + preds].dropna().copy()
        levels = sorted(clean[self.dep].unique(), key=str)
        if len(levels) != 2:
            raise ValueError(f"Dependent variable '{self.dep}' must be binary (2 levels), got {len(levels)}: {levels}")

        ref_lvl, target_lvl = levels[0], levels[1]
        y = (clean[self.dep].astype(str) == str(target_lvl)).astype(float)

        X_parts = []
        for p in preds:
            if p in self.factors or not pd.api.types.is_numeric_dtype(clean[p]):
                X_parts.append(pd.get_dummies(clean[p].astype(str), prefix=p, drop_first=True, dtype=float))
            else:
                X_parts.append(clean[[p]].astype(float))
        X = sm.add_constant(pd.concat(X_parts, axis=1), has_constant="add")

        logit_mod = sm.Logit(y, X)
        res = logit_mod.fit(disp=False, maxiter=200)

        ll_full = float(res.llf)
        ll_null = float(res.llnull)
        n = len(y)
        dev = -2.0 * ll_full
        chi2_lr = float(res.llr)
        df_lr = int(res.df_model)
        p_lr = float(res.llr_pvalue)

        r2_mcf = 1.0 - (ll_full / ll_null) if ll_null != 0 else 0.0
        r2_cs = 1.0 - np.exp((2.0 / n) * (ll_null - ll_full))
        r2_nag = r2_cs / (1.0 - np.exp((2.0 / n) * ll_null)) if (1.0 - np.exp((2.0 / n) * ll_null)) != 0 else r2_cs

        # 1. Model Fit Measures
        fit_tbl = ResultsTable(name="fit", title="Model Fit Measures")
        fit_tbl.add_column("model", "Model", col_type="text")
        fit_tbl.add_column("dev", "Deviance", col_type="number")
        fit_tbl.add_column("aic", "AIC", col_type="number")
        fit_tbl.add_column("bic", "BIC", col_type="number")
        fit_tbl.add_column("r2_mcf", "R²McF", col_type="number", fmt="zto")
        fit_tbl.add_column("r2_cs", "R²CS", col_type="number", fmt="zto")
        fit_tbl.add_column("r2_n", "R²N", col_type="number", fmt="zto")
        fit_tbl.add_column("chi2", "χ²", col_type="number")
        fit_tbl.add_column("df", "df", col_type="integer")
        fit_tbl.add_column("p", "p", col_type="number", fmt="pvalue")
        fit_tbl.add_row({
            "model": "1", "dev": dev, "aic": float(res.aic), "bic": float(res.bic),
            "r2_mcf": float(r2_mcf), "r2_cs": float(r2_cs), "r2_n": float(r2_nag),
            "chi2": chi2_lr, "df": df_lr, "p": p_lr
        })
        self.add_table(fit_tbl)

        # 2. Model Coefficients + Odds Ratios
        coef_tbl = ResultsTable(name="coef", title=f"Model Coefficients — {self.dep} ({target_lvl} vs {ref_lvl})")
        coef_tbl.add_column("pred", "Predictor", col_type="text")
        coef_tbl.add_column("b", "Estimate", col_type="number")
        coef_tbl.add_column("se", "SE", col_type="number")
        coef_tbl.add_column("z", "Z", col_type="number")
        coef_tbl.add_column("p", "p", col_type="number", fmt="pvalue")
        if self.options.get("oddsRatio", True):
            coef_tbl.add_column("or", "Odds ratio", col_type="number")
            if self.options.get("ci", True):
                coef_tbl.add_column("or_l", "95% CI Lower", col_type="number")
                coef_tbl.add_column("or_u", "95% CI Upper", col_type="number")

        conf = res.conf_int()
        for col_n in X.columns:
            lbl = "Intercept" if col_n == "const" else col_n
            b_v = float(res.params[col_n])
            or_v = float(np.exp(np.clip(b_v, -50, 50)))
            or_l = float(np.exp(np.clip(conf.loc[col_n, 0], -50, 50)))
            or_u = float(np.exp(np.clip(conf.loc[col_n, 1], -50, 50)))
            coef_tbl.add_row({
                "pred": lbl, "b": b_v, "se": float(res.bse[col_n]),
                "z": float(res.tvalues[col_n]), "p": float(res.pvalues[col_n]),
                "or": or_v, "or_l": or_l, "or_u": or_u
            })
        coef_tbl.add_note(f"Estimates represent the log odds of '{self.dep} = {target_lvl}' vs. '{self.dep} = {ref_lvl}'.")
        coef_tbl.mark_coefficients(estimate="b", se="se", p="p", stat="z", exp_estimate="or", label="pred")
        self.add_table(coef_tbl)
        self.model_stats = {"N": n, "LL": ll_full, "Deviance": dev, "AIC": float(res.aic), "BIC": float(res.bic),
                            "R2_McF": float(r2_mcf), "R2_N": float(r2_nag), "Chi2": (chi2_lr, p_lr), "df": df_lr}

        # 3. Predictive Measures & Confusion Matrix + ROC
        probs = res.predict(X).values
        preds_bin = (probs >= self.cutoff).astype(int)
        tn, fp, fn, tp = confusion_matrix(y.astype(int), preds_bin, labels=[0, 1]).ravel()
        acc = (tp + tn) / max(1, n)
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        auc_val = float(roc_auc_score(y, probs))

        if self.options.get("classTable", True):
            ct = ResultsTable(name="classification", title=f"Classification Table (Cut-off = {self.cutoff:.2f})")
            ct.add_column("obs", "Observed", col_type="text")
            ct.add_column("p0", f"Predicted {ref_lvl}", col_type="integer")
            ct.add_column("p1", f"Predicted {target_lvl}", col_type="integer")
            ct.add_column("pct", "% Correct", col_type="number", fmt="pc")
            ct.add_row({"obs": str(ref_lvl), "p0": int(tn), "p1": int(fp), "pct": spec * 100.0})
            ct.add_row({"obs": str(target_lvl), "p0": int(fn), "p1": int(tp), "pct": sens * 100.0})
            self.add_table(ct)

            pm = ResultsTable(name="pred_measures", title="Predictive Measures")
            pm.add_column("acc", "Accuracy", col_type="number", fmt="zto")
            pm.add_column("spec", "Specificity", col_type="number", fmt="zto")
            pm.add_column("sens", "Sensitivity", col_type="number", fmt="zto")
            pm.add_column("auc", "AUC", col_type="number", fmt="zto")
            pm.add_row({"acc": float(acc), "spec": float(spec), "sens": float(sens), "auc": auc_val})
            self.add_table(pm)

        if self.options.get("rocPlot", True):
            fpr, tpr, _ = roc_curve(y, probs)
            self.add_plot("ROC Curve", plot_roc_curve(fpr, tpr, auc_val))

        p_str = "< .001" if p_lr < 0.001 else f"= {p_lr:.3f}"
        self.apa_narrative = (
            f"Binomial logistic regression predicting {self.dep} ({target_lvl} vs. {ref_lvl}) was statistically "
            f"{'significant' if p_lr < 0.05 else 'non-significant'}, χ²({df_lr}) = {chi2_lr:.2f}, p {p_str}, "
            f"McFadden's R² = {r2_mcf:.3f}, Nagelkerke's R² = {r2_nag:.3f}, AUC = {auc_val:.3f}."
        )


class PartialCorrelation(BaseAnalysis):
    def __init__(self, data: pd.DataFrame, vars: List[str], controls: Union[str, List[str]], method: str = "pearson"):
        v_list = [vars] if isinstance(vars, str) else list(vars)
        c_list = [controls] if isinstance(controls, str) else list(controls)
        super().__init__(
            name="partCorr",
            title="Partial Correlation",
            dataset=data,
            options={"vars": v_list, "controls": c_list, "method": method}
        )
        self.vars = v_list
        self.controls = c_list
        self.method = method
        self.syntax_python = f"jmv.partCorr(df, vars={v_list!r}, controls={c_list!r}, method={method!r})"
        self.syntax_r = f"jmv::partCorr(data=data, vars=vars({', '.join(v_list)}), controls=vars({', '.join(c_list)}))"
        self.run()

    def run(self):
        cols = self.vars + [c for c in self.controls if c not in self.vars]
        clean = self.dataset[cols].apply(pd.to_numeric, errors="coerce").dropna()

        tbl = ResultsTable(name="pcorr", title=f"Partial Correlation (Controlling for {', '.join(self.controls)})")
        tbl.add_column("v1", "Variable 1", col_type="text")
        tbl.add_column("v2", "Variable 2", col_type="text")
        tbl.add_column("n", "N", col_type="integer")
        tbl.add_column("r", f"Partial r ({self.method.title()})", col_type="number", fmt="zto")
        tbl.add_column("p", "p-value", col_type="number", fmt="pvalue")
        tbl.add_column("ci95", "95% CI", col_type="text")

        notes = []
        for i in range(len(self.vars)):
            for j in range(i + 1, len(self.vars)):
                v1, v2 = self.vars[i], self.vars[j]
                res = pg.partial_corr(data=clean, x=v1, y=v2, covar=self.controls, method=self.method)
                r0 = res.iloc[0]
                ci = r0["CI95"] if "CI95" in r0 else r0["CI95%"]
                pv = float(r0["p_val"] if "p_val" in r0 else r0["p-val"])
                ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if isinstance(ci, (list, tuple, np.ndarray)) else str(ci)
                tbl.add_row({
                    "v1": v1, "v2": v2, "n": int(r0["n"]),
                    "r": float(r0["r"]), "p": pv, "ci95": ci_str
                })
                notes.append(f"{v1}–{v2}: partial r = {r0['r']:.3f} (p = {pv:.3f})")

        tbl.add_note(f"Controlling for: {', '.join(self.controls)}")
        self.add_table(tbl)
        self.apa_narrative = f"Partial correlation controlling for {', '.join(self.controls)}: {'; '.join(notes)}."

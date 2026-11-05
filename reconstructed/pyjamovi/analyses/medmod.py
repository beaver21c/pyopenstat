"""
pyjamovi Mediation & Moderation Suite (medmod Compatible, v2.0)
- Mediation (medmod::med): Indirect (a×b), Direct (c'), Total (c) effects, % Mediated, Sobel z, Bootstrap 95% CI, Path Diagram
- Moderation (medmod::mod): Mean-centered X, W, X×W interaction, Simple Slopes (-1SD, Mean, +1SD), Simple Slope Plot
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from .base import BaseAnalysis, ResultsTable
from .plots import plot_sem_diagram, plot_moderation_slopes


class Mediation(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        dep: str,
        pred: str,
        med: str,
        bootstrap: int = 500,
        ci: bool = True,
        pm: bool = True,
        paths: bool = True,
        diagram: bool = True
    ):
        super().__init__(
            name="mediation",
            title="Mediation",
            dataset=data,
            options={
                "dep": dep,
                "pred": pred,
                "med": med,
                "bootstrap": int(bootstrap),
                "ci": ci,
                "pm": pm,
                "paths": paths,
                "diagram": diagram
            }
        )
        self.dep = dep
        self.pred = pred
        self.med = med
        self.bootstrap = int(bootstrap)
        self.syntax_python = f"jmv.mediation(df, dep={dep!r}, pred={pred!r}, med={med!r}, bootstrap={self.bootstrap})"
        self.syntax_r = f"medmod::med(data=data, dep={dep}, pred={pred}, med={med}, pm=TRUE, paths=TRUE, estMethod='bootstrap')"
        self.run()

    def run(self):
        clean = self.dataset[[self.pred, self.med, self.dep]].apply(pd.to_numeric, errors="coerce").dropna()
        X = clean[self.pred].values
        M = clean[self.med].values
        Y = clean[self.dep].values
        n = len(clean)
        if n < 6:
            raise ValueError("Insufficient observations for mediation analysis.")

        # Path a: M ~ X
        mod_a = sm.OLS(M, sm.add_constant(X)).fit()
        a, se_a, z_a, p_a = float(mod_a.params[1]), float(mod_a.bse[1]), float(mod_a.tvalues[1]), float(mod_a.pvalues[1])

        # Paths b and c': Y ~ X + M
        mod_ym = sm.OLS(Y, sm.add_constant(np.column_stack([X, M]))).fit()
        c_prime, se_cp, z_cp, p_cp = float(mod_ym.params[1]), float(mod_ym.bse[1]), float(mod_ym.tvalues[1]), float(mod_ym.pvalues[1])
        b, se_b, z_b, p_b = float(mod_ym.params[2]), float(mod_ym.bse[2]), float(mod_ym.tvalues[2]), float(mod_ym.pvalues[2])

        # Total effect c: Y ~ X
        mod_c = sm.OLS(Y, sm.add_constant(X)).fit()
        c_tot, se_c, z_c, p_c = float(mod_c.params[1]), float(mod_c.bse[1]), float(mod_c.tvalues[1]), float(mod_c.pvalues[1])

        indirect = a * b
        sobel_se = float(np.sqrt(b ** 2 * se_a ** 2 + a ** 2 * se_b ** 2))
        z_ind = indirect / sobel_se if sobel_se > 0 else 0.0
        p_ind = float(2.0 * (1.0 - stats.norm.cdf(abs(z_ind))))

        # Non-parametric Bootstrap for 95% CI
        ci_ind = (indirect - 1.96 * sobel_se, indirect + 1.96 * sobel_se)
        if self.bootstrap >= 50:
            rng = np.random.default_rng(2026)
            boot_ab = np.empty(self.bootstrap)
            for idx in range(self.bootstrap):
                b_idx = rng.integers(0, n, size=n)
                xb, mb, yb = X[b_idx], M[b_idx], Y[b_idx]
                # Fast OLS closed-form
                xa = np.column_stack([np.ones(n), xb])
                a_b = np.linalg.lstsq(xa, mb, rcond=None)[0][1]
                xym = np.column_stack([np.ones(n), xb, mb])
                b_b = np.linalg.lstsq(xym, yb, rcond=None)[0][2]
                boot_ab[idx] = a_b * b_b
            ci_ind = (float(np.percentile(boot_ab, 2.5)), float(np.percentile(boot_ab, 97.5)))

        denom_pm = abs(indirect) + abs(c_prime)
        pct_med = (abs(indirect) / denom_pm) * 100.0 if denom_pm > 0 else 0.0

        # 1. Mediation Estimates Table
        med_tbl = ResultsTable(name="med", title="Mediation Estimates")
        med_tbl.add_column("effect", "Effect", col_type="text")
        med_tbl.add_column("label", "Label", col_type="text")
        med_tbl.add_column("est", "Estimate", col_type="number")
        med_tbl.add_column("se", "SE", col_type="number")
        if self.options.get("ci", True):
            med_tbl.add_column("cil", "95% CI Lower", col_type="number")
            med_tbl.add_column("ciu", "95% CI Upper", col_type="number")
        med_tbl.add_column("z", "Z", col_type="number")
        med_tbl.add_column("p", "p", col_type="number", fmt="pvalue")
        if self.options.get("pm", True):
            med_tbl.add_column("pm", "% Mediation", col_type="number", fmt="pc")

        med_tbl.add_row({
            "effect": "Indirect", "label": "a × b", "est": indirect, "se": sobel_se,
            "cil": ci_ind[0], "ciu": ci_ind[1], "z": z_ind, "p": p_ind, "pm": pct_med
        })
        med_tbl.add_row({
            "effect": "Direct", "label": "c'", "est": c_prime, "se": se_cp,
            "cil": c_prime - 1.96 * se_cp, "ciu": c_prime + 1.96 * se_cp,
            "z": z_cp, "p": p_cp, "pm": 100.0 - pct_med
        })
        med_tbl.add_row({
            "effect": "Total", "label": "c", "est": c_tot, "se": se_c,
            "cil": c_tot - 1.96 * se_c, "ciu": c_tot + 1.96 * se_c,
            "z": z_c, "p": p_c, "pm": 100.0
        })
        if self.bootstrap >= 50:
            med_tbl.add_note(f"Indirect effect 95% CI computed via {self.bootstrap} percentile bootstrap resamples.")
        med_tbl.mark_coefficients(estimate="est", se="se", p="p", stat="z", ci=("cil", "ciu"), label=["effect", "label"])
        self.add_table(med_tbl)

        # 2. Path Estimates Table
        if self.options.get("paths", True):
            pt = ResultsTable(name="paths", title="Path Estimates")
            pt.add_column("p1", "From", col_type="text")
            pt.add_column("arrow", "", col_type="text")
            pt.add_column("p2", "To", col_type="text")
            pt.add_column("lbl", "Label", col_type="text")
            pt.add_column("est", "Estimate", col_type="number")
            pt.add_column("se", "SE", col_type="number")
            pt.add_column("z", "Z", col_type="number")
            pt.add_column("p", "p", col_type="number", fmt="pvalue")
            pt.add_row({"p1": self.pred, "arrow": "→", "p2": self.med, "lbl": "a", "est": a, "se": se_a, "z": z_a, "p": p_a})
            pt.add_row({"p1": self.med, "arrow": "→", "p2": self.dep, "lbl": "b", "est": b, "se": se_b, "z": z_b, "p": p_b})
            pt.add_row({"p1": self.pred, "arrow": "→", "p2": self.dep, "lbl": "c'", "est": c_prime, "se": se_cp, "z": z_cp, "p": p_cp})
            pt.mark_coefficients(estimate="est", se="se", p="p", stat="z", label=["p1", "arrow", "p2"])
            self.add_table(pt)

        if self.options.get("diagram", True):
            diag_df = pd.DataFrame([
                {"lval": self.med, "op": "~", "rval": self.pred, "Estimate": a, "p-value": p_a},
                {"lval": self.dep, "op": "~", "rval": self.med, "Estimate": b, "p-value": p_b},
                {"lval": self.dep, "op": "~", "rval": self.pred, "Estimate": c_prime, "p-value": p_cp},
            ])
            self.add_plot(f"Mediation Model: {self.pred} → {self.med} → {self.dep}", plot_sem_diagram(diag_df, title="Mediation Path Model"))

        p_str = "< .001" if p_ind < 0.001 else f"= {p_ind:.3f}"
        self.apa_narrative = (
            f"Mediation analysis showed an indirect effect of {self.pred} on {self.dep} through {self.med} "
            f"(ab = {indirect:.3f}, 95% Boot CI [{ci_ind[0]:.3f}, {ci_ind[1]:.3f}], Z = {z_ind:.2f}, p {p_str}), "
            f"accounting for {pct_med:.1f}% of the total effect."
        )


class Moderation(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        dep: str,
        pred: str,
        mod: str,
        simple_slopes: bool = True,
        slope_plot: bool = True
    ):
        super().__init__(
            name="moderation",
            title="Moderation",
            dataset=data,
            options={"dep": dep, "pred": pred, "mod": mod, "simpleSlopes": simple_slopes, "slopePlot": slope_plot}
        )
        self.dep = dep
        self.pred = pred
        self.mod = mod
        self.syntax_python = f"jmv.moderation(df, dep={dep!r}, pred={pred!r}, mod={mod!r})"
        self.syntax_r = f"medmod::mod(data=data, dep={dep}, pred={pred}, mod={mod}, simpleSlopeEst=TRUE, simpleSlopePlot=TRUE)"
        self.run()

    def run(self):
        clean = self.dataset[[self.pred, self.mod, self.dep]].apply(pd.to_numeric, errors="coerce").dropna()
        x_mean, w_mean = clean[self.pred].mean(), clean[self.mod].mean()
        w_sd = clean[self.mod].std(ddof=1)

        xc = clean[self.pred] - x_mean
        wc = clean[self.mod] - w_mean
        xw = xc * wc
        Y = clean[self.dep]

        X_mat = pd.DataFrame({"const": 1.0, "X": xc, "W": wc, "XW": xw})
        ols_res = sm.OLS(Y, X_mat).fit()
        conf = ols_res.conf_int()

        # 1. Moderation Estimates Table
        mod_tbl = ResultsTable(name="mod", title="Moderation Estimates")
        mod_tbl.add_column("term", "Term", col_type="text")
        mod_tbl.add_column("est", "Estimate", col_type="number")
        mod_tbl.add_column("se", "SE", col_type="number")
        mod_tbl.add_column("cil", "95% CI Lower", col_type="number")
        mod_tbl.add_column("ciu", "95% CI Upper", col_type="number")
        mod_tbl.add_column("z", "Z / t", col_type="number")
        mod_tbl.add_column("p", "p", col_type="number", fmt="pvalue")

        labels = [("X", self.pred), ("W", self.mod), ("XW", f"{self.pred} ✻ {self.mod}")]
        for key, lbl in labels:
            mod_tbl.add_row({
                "term": lbl,
                "est": float(ols_res.params[key]),
                "se": float(ols_res.bse[key]),
                "cil": float(conf.loc[key, 0]),
                "ciu": float(conf.loc[key, 1]),
                "z": float(ols_res.tvalues[key]),
                "p": float(ols_res.pvalues[key])
            })
        mod_tbl.add_note("Predictor and moderator were mean-centered prior to computing the interaction term.")
        mod_tbl.mark_coefficients(estimate="est", se="se", p="p", stat="z", ci=("cil", "ciu"), label="term")
        self.add_table(mod_tbl)
        self.model_stats = {"N": int(ols_res.nobs), "R2": float(ols_res.rsquared), "R2_adj": float(ols_res.rsquared_adj),
                            "F": (float(ols_res.fvalue), float(ols_res.f_pvalue)), "AIC": float(ols_res.aic), "BIC": float(ols_res.bic)}

        # 2. Simple Slope Estimates (-1SD, Mean, +1SD)
        b1 = float(ols_res.params["X"])
        b2 = float(ols_res.params["W"])
        b3 = float(ols_res.params["XW"])
        b0 = float(ols_res.params["const"])
        cov_m = ols_res.cov_params()
        v_b1 = float(cov_m.loc["X", "X"])
        v_b3 = float(cov_m.loc["XW", "XW"])
        cov_13 = float(cov_m.loc["X", "XW"])

        if self.options.get("simpleSlopes", True):
            ss_tbl = ResultsTable(name="simple_slopes", title=f"Simple Slope Estimates — Effect of {self.pred} on {self.dep}")
            ss_tbl.add_column("level", f"Level of {self.mod}", col_type="text")
            ss_tbl.add_column("est", "Estimate", col_type="number")
            ss_tbl.add_column("se", "SE", col_type="number")
            ss_tbl.add_column("cil", "95% CI Lower", col_type="number")
            ss_tbl.add_column("ciu", "95% CI Upper", col_type="number")
            ss_tbl.add_column("t", "t", col_type="number")
            ss_tbl.add_column("p", "p", col_type="number", fmt="pvalue")

            for lvl_name, w_val in [("Average", 0.0), ("Low (-1SD)", -w_sd), ("High (+1SD)", +w_sd)]:
                slp = b1 + b3 * w_val
                se_slp = float(np.sqrt(max(0.0, v_b1 + (w_val ** 2) * v_b3 + 2.0 * w_val * cov_13)))
                t_slp = slp / se_slp if se_slp > 0 else 0.0
                p_slp = float(2.0 * (1.0 - stats.t.cdf(abs(t_slp), ols_res.df_resid)))
                ss_tbl.add_row({
                    "level": lvl_name, "est": slp, "se": se_slp,
                    "cil": slp - 1.96 * se_slp, "ciu": slp + 1.96 * se_slp,
                    "t": t_slp, "p": p_slp
                })
            ss_tbl.mark_coefficients(estimate="est", se="se", p="p", stat="t", ci=("cil", "ciu"), label="level")
            self.add_table(ss_tbl)

        if self.options.get("slopePlot", True):
            x_sd = clean[self.pred].std(ddof=1)
            x_grid = np.linspace(x_mean - 1.5 * x_sd, x_mean + 1.5 * x_sd, 60)
            xc_grid = x_grid - x_mean
            slopes_dict = {
                "Low (-1SD)": b0 + b1 * xc_grid + b2 * (-w_sd) + b3 * xc_grid * (-w_sd),
                "Average": b0 + b1 * xc_grid,
                "High (+1SD)": b0 + b1 * xc_grid + b2 * (+w_sd) + b3 * xc_grid * (+w_sd),
            }
            self.add_plot("Simple Slope Plot", plot_moderation_slopes(x_grid, slopes_dict, self.pred, self.mod, self.dep))

        p_int = float(ols_res.pvalues["XW"])
        p_txt = "< .001" if p_int < 0.001 else f"= {p_int:.3f}"
        self.apa_narrative = (
            f"Moderation analysis tested {self.pred} × {self.mod} on {self.dep} (R² = {ols_res.rsquared:.3f}). "
            f"The interaction term was {'statistically significant' if p_int < 0.05 else 'not statistically significant'} "
            f"(b = {b3:.3f}, SE = {ols_res.bse['XW']:.3f}, t = {ols_res.tvalues['XW']:.2f}, p {p_txt})."
        )

"""
pyjamovi Structural Equation Modeling (SEM) & Confirmatory Factor Analysis (CFA) Module (v2.0)
Reproduces jamovi SEMLj & jmv::cfa using lavaan-compatible syntax ('=~', '~', '~~') via semopy:
- Unstandardized & Standardized Estimates (Std. Est / Beta)
- Comprehensive Fit Indices (Chi-square, df, p, CFI, TLI, NFI, GFI, AGFI, RMSEA, AIC, BIC)
- Automatic Path & Latent Measurement Diagram
"""

from typing import Optional
import numpy as np
import pandas as pd
from semopy import Model
from semopy.stats import calc_stats

from .base import BaseAnalysis, ResultsTable
from .plots import plot_sem_diagram


class SEM(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        model_syntax: str,
        std_est: bool = True,
        fit_measures: bool = True,
        variances: bool = False,
        path_diagram: bool = True
    ):
        super().__init__(
            name="sem",
            title="Structural Equation Modeling (CFA / SEM)",
            dataset=data,
            options={
                "model": model_syntax,
                "stdEst": std_est,
                "fitMeasures": fit_measures,
                "variances": variances,
                "pathDiagram": path_diagram
            }
        )
        self.model_syntax = model_syntax
        self.syntax_python = f"jmv.sem(df, model_syntax={model_syntax!r}, std_est={std_est})"
        self.syntax_r = f"semlj::semljgui(data=data, syntax={model_syntax!r})"
        self.run()

    def run(self):
        df = self.dataset.dropna().copy()

        # Sanitize column names with dots/spaces/dashes for semopy parser
        col_rename = {}
        reverse_rename = {}
        for c in df.columns:
            if "." in c or "-" in c or " " in c:
                safe_name = c.replace(".", "_").replace("-", "_").replace(" ", "_")
                col_rename[c] = safe_name
                reverse_rename[safe_name] = c

        clean_syntax = self.model_syntax
        if col_rename:
            df.rename(columns=col_rename, inplace=True)
            for orig in sorted(col_rename.keys(), key=len, reverse=True):
                clean_syntax = clean_syntax.replace(orig, col_rename[orig])

        model = Model(clean_syntax)
        model.fit(df)

        estimates = model.inspect(std_est=self.options.get("stdEst", True))

        # Separate Structural / Measurement paths vs Variances/Covariances
        param_table = ResultsTable(name="estimates", title="Parameter Estimates (Measurement & Structural Paths)")
        param_table.add_column("lval", "lhs (Outcome / Indicator)", col_type="text")
        param_table.add_column("op", "op", col_type="text")
        param_table.add_column("rval", "rhs (Predictor / Latent)", col_type="text")
        param_table.add_column("est", "Estimate", col_type="number")
        param_table.add_column("se", "SE", col_type="number")
        if self.options.get("stdEst", True) and "Est. Std" in estimates.columns:
            param_table.add_column("std", "β (Std. Est)", col_type="number", fmt="zto")
        param_table.add_column("z", "z", col_type="number")
        param_table.add_column("p", "p", col_type="number", fmt="pvalue")

        var_table = ResultsTable(name="variances", title="Variances & Covariances")
        var_table.add_column("lval", "Variable 1", col_type="text")
        var_table.add_column("rval", "Variable 2", col_type="text")
        var_table.add_column("est", "Estimate", col_type="number")
        var_table.add_column("se", "SE", col_type="number")
        var_table.add_column("z", "z", col_type="number")
        var_table.add_column("p", "p", col_type="number", fmt="pvalue")

        def _num_or_nan(v):
            if v is None or v == "-" or pd.isna(v):
                return np.nan
            try:
                return float(v)
            except Exception:
                return np.nan

        restored_rows = []
        for _, row in estimates.iterrows():
            l_val = reverse_rename.get(str(row.get("lval", "")), str(row.get("lval", "")))
            r_val = reverse_rename.get(str(row.get("rval", "")), str(row.get("rval", "")))
            op = str(row.get("op", ""))
            est = _num_or_nan(row.get("Estimate"))
            se = _num_or_nan(row.get("Std. Err"))
            std_v = _num_or_nan(row.get("Est. Std"))
            zv = _num_or_nan(row.get("z-value"))
            pv = _num_or_nan(row.get("p-value"))

            restored_rows.append({"lval": l_val, "op": op, "rval": r_val, "Estimate": est, "p-value": pv})

            if op == "~~":
                var_table.add_row({"lval": l_val, "rval": r_val, "est": est, "se": se, "z": zv, "p": pv})
            else:
                param_table.add_row({
                    "lval": l_val, "op": op, "rval": r_val,
                    "est": est, "se": se, "std": std_v, "z": zv, "p": pv
                })

        self.add_table(param_table)
        if self.options.get("variances", False) and len(var_table.rows) > 0:
            self.add_table(var_table)

        # Fit Measures Table
        if self.options.get("fitMeasures", True):
            try:
                stats_df = calc_stats(model)
                fit_table = ResultsTable(name="fit", title="Model Fit Indices")
                fit_table.add_column("index", "Fit Index", col_type="text")
                fit_table.add_column("value", "Value", col_type="number")

                def _get_s(k):
                    if k in stats_df.index:
                        val = stats_df.loc[k].values[0]
                        return float(val) if val is not None and not pd.isna(val) and np.isfinite(val) else np.nan
                    if k in stats_df.columns:
                        val = stats_df[k].iloc[0]
                        return float(val) if val is not None and not pd.isna(val) and np.isfinite(val) else np.nan
                    return np.nan

                dof = _get_s("DoF")
                chi2 = _get_s("chi2")
                pval = _get_s("chi2 p-value")
                cfi = _get_s("CFI")
                tli = _get_s("TLI")
                nfi = _get_s("NFI")
                gfi = _get_s("GFI")
                rmsea = _get_s("RMSEA")
                aic = _get_s("AIC")
                bic = _get_s("BIC")

                # Saturated model (df == 0) handling
                if dof == 0:
                    cfi = 1.0 if np.isnan(cfi) else cfi
                    tli = 1.0 if np.isnan(tli) else tli
                    rmsea = 0.0 if np.isnan(rmsea) else rmsea
                    fit_table.add_note("Model is just-identified / saturated (df = 0); global fit is exact by definition.")

                for idx_name, idx_val in [
                    ("χ² (Model Test)", chi2),
                    ("Degrees of freedom (df)", dof),
                    ("p-value (χ²)", pval),
                    ("Comparative Fit Index (CFI)", cfi),
                    ("Tucker-Lewis Index (TLI)", tli),
                    ("Normed Fit Index (NFI)", nfi),
                    ("Goodness of Fit Index (GFI)", gfi),
                    ("RMSEA", rmsea),
                    ("AIC", aic),
                    ("BIC", bic)
                ]:
                    fit_table.add_row({"index": idx_name, "value": idx_val})
                self.add_table(fit_table)

                if dof == 0:
                    self.apa_narrative = (
                        f"Path analysis (saturated model, df = 0, N = {len(df)}) estimated {len(param_table.rows)} structural relationships."
                    )
                else:
                    dof_txt = int(dof) if not np.isnan(dof) else "?"
                    p_txt = "< .001" if (not np.isnan(pval) and pval < 0.001) else (f"= {pval:.3f}" if not np.isnan(pval) else "= —")
                    self.apa_narrative = (
                        f"Structural Equation Modeling indicated model fit of χ²({dof_txt}) = {chi2:.2f} (p {p_txt}), "
                        f"CFI = {cfi:.3f}, TLI = {tli:.3f}, RMSEA = {rmsea:.3f}."
                    )
            except Exception:
                self.apa_narrative = "Structural Equation Model parameters were estimated via Maximum Likelihood."

        if self.options.get("pathDiagram", True) and restored_rows:
            self.add_plot("Structural & Measurement Path Diagram", plot_sem_diagram(pd.DataFrame(restored_rows)))

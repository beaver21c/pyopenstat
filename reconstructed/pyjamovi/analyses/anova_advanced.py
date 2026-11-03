"""
pyjamovi Advanced ANOVA Suite (v2.0)
- FactorialANOVA (jmv::ANOVA): Multi-way Factorial ANOVA, SS Type I/II/III, η², η²p, ω², Post-Hoc, EMM Interaction Plots
- ANCOVA (jmv::ancova): Analysis of Covariance with Fixed Factors + Continuous Covariates
- RepeatedMeasuresANOVA (jmv::anovaRM): Within-Subjects & Mixed RM-ANOVA, Mauchly Sphericity, Greenhouse-Geisser, Profile Plot
- KruskalWallis (jmv::anovaNP): Non-parametric One-Way ANOVA (χ², ε², Pairwise Comparisons)
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg

from .base import BaseAnalysis, ResultsTable
from .posthoc import compute_posthoc
from .plots import plot_interaction_emm, plot_rm_profile, plot_qq, plot_mean_ci


class FactorialANOVA(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        dep: str,
        factors: Union[str, List[str]],
        ss_type: int = 3,
        eta_sq: bool = True,
        part_eta_sq: bool = True,
        omega_sq: bool = False,
        homo: bool = False,
        norm: bool = False,
        ph_method: str = "tukey",
        emm_plot: bool = True,
        qq_plot: bool = False
    ):
        facs = [factors] if isinstance(factors, str) else list(factors)
        super().__init__(
            name="anovaFactorial",
            title="ANOVA",
            dataset=data,
            options={
                "dep": dep,
                "factors": facs,
                "ssType": int(ss_type),
                "etaSq": eta_sq,
                "partEtaSq": part_eta_sq,
                "omegaSq": omega_sq,
                "homo": homo,
                "norm": norm,
                "phMethod": ph_method,
                "emmPlot": emm_plot,
                "qqPlot": qq_plot
            }
        )
        self.dep = dep
        self.factors = facs
        self.ss_type = int(ss_type) if int(ss_type) in (1, 2, 3) else 3
        self.syntax_python = f"jmv.anova(df, dep={dep!r}, factors={facs!r}, ss_type={self.ss_type}, ph_method={ph_method!r})"
        self.syntax_r = f"jmv::ANOVA(formula={dep} ~ {' * '.join(facs)}, data=data, ss='{self.ss_type}', effectSize=c('eta', 'partEta'))"
        self.run()

    def run(self):
        df = self.dataset
        opts = self.options
        cols = [self.dep] + self.factors
        clean = df[cols].dropna().copy()
        clean[self.dep] = pd.to_numeric(clean[self.dep], errors="coerce")
        clean = clean.dropna()

        # Safe column names for patsy
        cmap = {c: f"v{idx}" for idx, c in enumerate(cols)}
        rev_map = {v: k for k, v in cmap.items()}
        wdf = clean.rename(columns=cmap)
        dv = cmap[self.dep]
        f_safe = [cmap[f] for f in self.factors]
        for fs in f_safe:
            wdf[fs] = wdf[fs].astype(str)

        rhs = " * ".join(f"C({fs}, Sum)" if self.ss_type == 3 else f"C({fs})" for fs in f_safe)
        model = ols(f"{dv} ~ {rhs}", data=wdf).fit()
        aov = sm.stats.anova_lm(model, typ=self.ss_type)

        # Clean row labels back to human factor names
        def _pretty_term(raw_term: str) -> str:
            if raw_term == "Intercept":
                return "Intercept"
            if raw_term == "Residual":
                return "Residuals"
            parts = raw_term.split(":")
            nice_parts = []
            for p in parts:
                for fs in f_safe:
                    if fs in p:
                        nice_parts.append(rev_map[fs])
                        break
            return " ✻ ".join(nice_parts) if nice_parts else raw_term

        ss_resid = float(aov.loc["Residual", "sum_sq"])
        df_resid = float(aov.loc["Residual", "df"])
        ms_resid = ss_resid / max(1.0, df_resid)
        ss_terms = sum(float(aov.loc[idx, "sum_sq"]) for idx in aov.index if idx not in ("Intercept", "Residual"))
        ss_total = ss_terms + ss_resid

        tbl = ResultsTable(name="main", title=f"ANOVA — {self.dep}")
        tbl.add_column("term", "", col_type="text")
        tbl.add_column("ss", "Sum of Squares", col_type="number")
        tbl.add_column("df", "df", col_type="integer")
        tbl.add_column("ms", "Mean Square", col_type="number")
        tbl.add_column("f", "F", col_type="number")
        tbl.add_column("p", "p", col_type="number", fmt="pvalue")
        if opts["etaSq"]:
            tbl.add_column("eta", "η²", col_type="number", fmt="zto")
        if opts["partEtaSq"]:
            tbl.add_column("peta", "η²p", col_type="number", fmt="zto")
        if opts["omegaSq"]:
            tbl.add_column("omega", "ω²", col_type="number", fmt="zto")

        apa_parts = []
        for idx in aov.index:
            if idx == "Intercept":
                continue
            ss = float(aov.loc[idx, "sum_sq"])
            df_t = int(round(float(aov.loc[idx, "df"])))
            ms = ss / max(1, df_t)
            f_v = float(aov.loc[idx, "F"]) if not pd.isna(aov.loc[idx, "F"]) else np.nan
            p_v = float(aov.loc[idx, "PR(>F)"]) if not pd.isna(aov.loc[idx, "PR(>F)"]) else np.nan
            name_t = _pretty_term(str(idx))

            if idx == "Residual":
                tbl.add_row({
                    "term": "Residuals", "ss": ss, "df": df_t, "ms": ms,
                    "f": "", "p": "", "eta": "", "peta": "", "omega": ""
                })
            else:
                eta = ss / ss_total if ss_total > 0 else 0.0
                peta = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else 0.0
                omega = max(0.0, (ss - df_t * ms_resid) / (ss_total + ms_resid)) if (ss_total + ms_resid) > 0 else 0.0
                tbl.add_row({
                    "term": name_t, "ss": ss, "df": df_t, "ms": ms,
                    "f": f_v, "p": p_v, "eta": eta, "peta": peta, "omega": omega
                })
                p_txt = "< .001" if p_v < 0.001 else f"= {p_v:.3f}"
                apa_parts.append(f"{name_t}: F({df_t}, {int(df_resid)}) = {f_v:.2f}, p {p_txt}, η²p = {peta:.3f}")

        tbl.add_note(f"Type {'I' * self.ss_type} Sum of Squares")
        self.add_table(tbl)

        if opts["homo"]:
            cell_key = clean[self.factors].astype(str).agg("_".join, axis=1)
            groups = [clean[cell_key == k][self.dep].values for k in cell_key.unique() if (cell_key == k).sum() >= 2]
            if len(groups) >= 2:
                lf, lp = stats.levene(*groups, center="median")
                lt = ResultsTable(name="homo", title="Homogeneity of Variances Test (Levene's)")
                lt.add_column("f", "F", col_type="number")
                lt.add_column("df1", "df1", col_type="integer")
                lt.add_column("df2", "df2", col_type="integer")
                lt.add_column("p", "p", col_type="number", fmt="pvalue")
                lt.add_row({"f": float(lf), "df1": len(groups) - 1, "df2": int(len(clean) - len(groups)), "p": float(lp)})
                self.add_table(lt)

        if opts["norm"] and 3 <= len(model.resid) <= 5000:
            ws, ps = stats.shapiro(model.resid)
            nt = ResultsTable(name="norm", title="Normality Test (Shapiro-Wilk)")
            nt.add_column("w", "Statistic", col_type="number")
            nt.add_column("p", "p", col_type="number", fmt="pvalue")
            nt.add_row({"w": float(ws), "p": float(ps)})
            self.add_table(nt)

        ph_m = opts.get("phMethod", "tukey")
        if ph_m and str(ph_m).lower() != "none":
            for fac in self.factors:
                if clean[fac].nunique() >= 2:
                    self.add_table(compute_posthoc(clean, dep=self.dep, group=fac, method=str(ph_m)))

        if opts.get("emmPlot", True):
            f1 = self.factors[0]
            f2 = self.factors[1] if len(self.factors) >= 2 else None
            self.add_plot("Estimated Marginal Means", plot_interaction_emm(clean, self.dep, f1, f2))
        if opts.get("qqPlot", False):
            self.add_plot(f"Q-Q Plot of Residuals — {self.dep}", plot_qq(pd.DataFrame({self.dep: model.resid}), self.dep))

        self.apa_narrative = f"Factorial ANOVA on {self.dep} revealed: " + "; ".join(apa_parts) + "."


class ANCOVA(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        dep: str,
        factors: Union[str, List[str]],
        covs: Union[str, List[str]],
        ss_type: int = 3,
        ph_method: str = "tukey",
        emm_plot: bool = True
    ):
        facs = [factors] if isinstance(factors, str) else list(factors)
        cov_list = [covs] if isinstance(covs, str) else list(covs)
        super().__init__(
            name="ancova",
            title="ANCOVA",
            dataset=data,
            options={"dep": dep, "factors": facs, "covs": cov_list, "ssType": ss_type, "phMethod": ph_method, "emmPlot": emm_plot}
        )
        self.dep = dep
        self.factors = facs
        self.covs = cov_list
        self.ss_type = int(ss_type) if int(ss_type) in (1, 2, 3) else 3
        self.syntax_python = f"jmv.ancova(df, dep={dep!r}, factors={facs!r}, covs={cov_list!r})"
        self.syntax_r = f"jmv::ancova(data=data, dep={dep}, factors=vars({', '.join(facs)}), covs=vars({', '.join(cov_list)}))"
        self.run()

    def run(self):
        cols = [self.dep] + self.factors + self.covs
        clean = self.dataset[cols].dropna().copy()
        clean[self.dep] = pd.to_numeric(clean[self.dep], errors="coerce")
        for c in self.covs:
            clean[c] = pd.to_numeric(clean[c], errors="coerce")
        clean = clean.dropna()

        cmap = {c: f"v{idx}" for idx, c in enumerate(cols)}
        rev_map = {v: k for k, v in cmap.items()}
        wdf = clean.rename(columns=cmap)
        dv = cmap[self.dep]
        f_terms = [f"C({cmap[f]})" for f in self.factors]
        c_terms = [cmap[c] for c in self.covs]

        formula = f"{dv} ~ {' * '.join(f_terms)} + {' + '.join(c_terms)}"
        model = ols(formula, data=wdf).fit()
        aov = sm.stats.anova_lm(model, typ=self.ss_type if self.ss_type in (1, 2) else 2)

        ss_resid = float(aov.loc["Residual", "sum_sq"])
        df_resid = int(round(float(aov.loc["Residual", "df"])))

        tbl = ResultsTable(name="ancova", title=f"ANCOVA — {self.dep}")
        tbl.add_column("term", "", col_type="text")
        tbl.add_column("ss", "Sum of Squares", col_type="number")
        tbl.add_column("df", "df", col_type="integer")
        tbl.add_column("ms", "Mean Square", col_type="number")
        tbl.add_column("f", "F", col_type="number")
        tbl.add_column("p", "p", col_type="number", fmt="pvalue")
        tbl.add_column("peta", "η²p", col_type="number", fmt="zto")

        apa_parts = []
        for idx in aov.index:
            if idx == "Intercept":
                continue
            ss = float(aov.loc[idx, "sum_sq"])
            df_t = int(round(float(aov.loc[idx, "df"])))
            ms = ss / max(1, df_t)
            if idx == "Residual":
                tbl.add_row({"term": "Residuals", "ss": ss, "df": df_t, "ms": ms, "f": "", "p": "", "peta": ""})
            else:
                fv = float(aov.loc[idx, "F"])
                pv = float(aov.loc[idx, "PR(>F)"])
                peta = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else 0.0
                # Restore readable name
                parts = []
                for tok in str(idx).split(":"):
                    for k_safe, orig in rev_map.items():
                        if k_safe in tok:
                            parts.append(orig)
                            break
                lbl = " ✻ ".join(parts) if parts else str(idx)
                tbl.add_row({"term": lbl, "ss": ss, "df": df_t, "ms": ms, "f": fv, "p": pv, "peta": peta})
                p_txt = "< .001" if pv < 0.001 else f"= {pv:.3f}"
                apa_parts.append(f"{lbl} (F({df_t}, {df_resid}) = {fv:.2f}, p {p_txt}, η²p = {peta:.3f})")

        self.add_table(tbl)
        ph_m = self.options.get("phMethod", "tukey")
        if ph_m and str(ph_m).lower() != "none":
            self.add_table(compute_posthoc(clean, dep=self.dep, group=self.factors[0], method=str(ph_m)))
        if self.options.get("emmPlot", True):
            f2 = self.factors[1] if len(self.factors) >= 2 else None
            self.add_plot("Estimated Marginal Means (Covariate-Adjusted)", plot_interaction_emm(clean, self.dep, self.factors[0], f2))
        self.apa_narrative = f"ANCOVA controlling for {', '.join(self.covs)} found: {'; '.join(apa_parts)}."


class RepeatedMeasuresANOVA(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        rm_vars: List[str],
        between: Optional[str] = None,
        spher: bool = True,
        posthoc: bool = True,
        profile_plot: bool = True
    ):
        rm_list = list(rm_vars)
        super().__init__(
            name="anovaRM",
            title="Repeated Measures ANOVA",
            dataset=data,
            options={"rmVars": rm_list, "between": between, "spher": spher, "posthoc": posthoc, "profilePlot": profile_plot}
        )
        self.rm_vars = rm_list
        self.between = between
        self.syntax_python = f"jmv.anovaRM(df, rm_vars={rm_list!r}, between={between!r})"
        self.syntax_r = f"jmv::anovaRM(data=data, rmCells={rm_list!r}, bs={between!r})"
        self.run()

    def run(self):
        if len(self.rm_vars) < 2:
            raise ValueError("Repeated Measures ANOVA requires at least 2 repeated-measure columns.")
        cols = self.rm_vars + ([self.between] if self.between and self.between in self.dataset.columns else [])
        clean = self.dataset[cols].dropna().copy()
        for v in self.rm_vars:
            clean[v] = pd.to_numeric(clean[v], errors="coerce")
        clean = clean.dropna().reset_index(drop=True)
        clean["_subj"] = np.arange(len(clean))

        id_vars = ["_subj"] + ([self.between] if self.between and self.between in clean.columns else [])
        long_df = clean.melt(id_vars=id_vars, value_vars=self.rm_vars, var_name="RM_Factor", value_name="Score")

        if self.between and self.between in clean.columns:
            aov = pg.mixed_anova(data=long_df, dv="Score", within="RM_Factor", between=self.between, subject="_subj")
            ws_tbl = ResultsTable(name="rm_main", title="Mixed Repeated Measures ANOVA (Within & Between Effects)")
            ws_tbl.add_column("source", "Source", col_type="text")
            ws_tbl.add_column("ss", "SS", col_type="number")
            ws_tbl.add_column("df1", "df1", col_type="integer")
            ws_tbl.add_column("df2", "df2", col_type="integer")
            ws_tbl.add_column("ms", "MS", col_type="number")
            ws_tbl.add_column("f", "F", col_type="number")
            ws_tbl.add_column("p", "p", col_type="number", fmt="pvalue")
            ws_tbl.add_column("np2", "η²p", col_type="number", fmt="zto")
            for _, r in aov.iterrows():
                p_val_m = r["p_unc"] if "p_unc" in r else r["p-unc"]
                ws_tbl.add_row({
                    "source": str(r["Source"]),
                    "ss": float(r["SS"]),
                    "df1": int(r["DF1"]),
                    "df2": int(r["DF2"]),
                    "ms": float(r["MS"]),
                    "f": float(r["F"]),
                    "p": float(p_val_m),
                    "np2": float(r["np2"])
                })
            self.add_table(ws_tbl)
            f_main = float(aov.iloc[-2]["F"])
            r_m = aov.iloc[-2]
            p_main = float(r_m["p_unc"] if "p_unc" in r_m else r_m["p-unc"])
            np2_main = float(r_m["np2"])
            self.apa_narrative = (
                f"Mixed Repeated Measures ANOVA across {len(self.rm_vars)} timepoints and '{self.between}' yielded "
                f"within-effect F = {f_main:.2f} (p = {p_main:.3f}, η²p = {np2_main:.3f})."
            )
        else:
            aov = pg.rm_anova(data=long_df, dv="Score", within="RM_Factor", subject="_subj", detailed=True)
            ws_tbl = ResultsTable(name="within", title="Within Subjects Effects")
            ws_tbl.add_column("source", "Source", col_type="text")
            ws_tbl.add_column("ss", "Sum of Squares", col_type="number")
            ws_tbl.add_column("df", "df", col_type="integer")
            ws_tbl.add_column("ms", "Mean Square", col_type="number")
            ws_tbl.add_column("f", "F", col_type="number")
            ws_tbl.add_column("p", "p", col_type="number", fmt="pvalue")
            ws_tbl.add_column("p_gg", "p (GG-corr)", col_type="number", fmt="pvalue")
            ws_tbl.add_column("ng2", "η²G", col_type="number", fmt="zto")

            for _, r in aov.iterrows():
                src = str(r["Source"])
                p_u = r["p_unc"] if "p_unc" in r else r.get("p-unc", np.nan)
                p_gg = r["p_GG_corr"] if "p_GG_corr" in r else r.get("p-GG-corr", np.nan)
                ws_tbl.add_row({
                    "source": "RM Factor" if src == "RM_Factor" else "Residual",
                    "ss": float(r["SS"]),
                    "df": int(r["DF"]),
                    "ms": float(r["MS"]),
                    "f": float(r["F"]) if "F" in r and not pd.isna(r["F"]) else "",
                    "p": float(p_u) if not pd.isna(p_u) else "",
                    "p_gg": float(p_gg) if not pd.isna(p_gg) else "",
                    "ng2": float(r["ng2"]) if "ng2" in r and not pd.isna(r["ng2"]) else ""
                })
            self.add_table(ws_tbl)

            if self.options.get("spher", True) and len(self.rm_vars) >= 3:
                spher_bool, w_val, chi2_s, dof_s, p_spher = pg.sphericity(data=long_df, dv="Score", within="RM_Factor", subject="_subj")
                eps_gg = float(aov.loc[0, "eps"]) if "eps" in aov.columns else 1.0
                sp_tbl = ResultsTable(name="sphericity", title="Tests of Sphericity (Mauchly's W)")
                sp_tbl.add_column("term", "Term", col_type="text")
                sp_tbl.add_column("w", "Mauchly's W", col_type="number", fmt="zto")
                sp_tbl.add_column("p", "p", col_type="number", fmt="pvalue")
                sp_tbl.add_column("gg", "Greenhouse-Geisser ε", col_type="number", fmt="zto")
                sp_tbl.add_row({"term": "RM Factor", "w": float(w_val), "p": float(p_spher), "gg": eps_gg})
                self.add_table(sp_tbl)

            r0 = aov.iloc[0]
            p0 = float(r0["p_unc"] if "p_unc" in r0 else r0["p-unc"])
            p_txt = "< .001" if p0 < 0.001 else f"= {p0:.3f}"
            self.apa_narrative = (
                f"Repeated-measures ANOVA across {', '.join(self.rm_vars)} revealed F({int(r0['DF'])}, {int(aov.iloc[1]['DF'])}) = "
                f"{r0['F']:.2f}, p {p_txt}, η²G = {r0.get('ng2', 0.0):.3f}."
            )

        if self.options.get("posthoc", True):
            pw = pg.pairwise_tests(data=long_df, dv="Score", within="RM_Factor", subject="_subj", padjust="holm")
            ph_tbl = ResultsTable(name="posthoc_rm", title="Post-Hoc Comparisons — RM Factor (Holm)")
            ph_tbl.add_column("a", "Level 1", col_type="text")
            ph_tbl.add_column("b", "Level 2", col_type="text")
            ph_tbl.add_column("t", "T", col_type="number")
            ph_tbl.add_column("df", "dof", col_type="number")
            ph_tbl.add_column("p_unc", "p-unc", col_type="number", fmt="pvalue")
            ph_tbl.add_column("p_holm", "p-holm", col_type="number", fmt="pvalue")
            ph_tbl.add_column("hedges", "Hedges' g", col_type="number")
            for _, r in pw.iterrows():
                pu = float(r["p_unc"] if "p_unc" in r else r["p-unc"])
                pc = float(r["p_corr"] if "p_corr" in r else r.get("p-corr", pu))
                ph_tbl.add_row({
                    "a": str(r["A"]), "b": str(r["B"]),
                    "t": float(r["T"]), "df": float(r["dof"]),
                    "p_unc": pu,
                    "p_holm": pc,
                    "hedges": float(r.get("hedges", np.nan))
                })
            self.add_table(ph_tbl)

        if self.options.get("profilePlot", True):
            self.add_plot("Repeated Measures Marginal Means", plot_rm_profile(clean, self.rm_vars, self.between))


class KruskalWallis(BaseAnalysis):
    def __init__(self, data: pd.DataFrame, deps: Union[str, List[str]], group: str, dscf: bool = True, plot: bool = False):
        dep_vars = [deps] if isinstance(deps, str) else list(deps)
        super().__init__(
            name="kruskal",
            title="One-Way ANOVA (Non-parametric — Kruskal-Wallis)",
            dataset=data,
            options={"deps": dep_vars, "group": group, "dscf": dscf, "plot": plot}
        )
        self.deps = dep_vars
        self.group = group
        self.syntax_python = f"jmv.kruskal(df, deps={dep_vars!r}, group={group!r})"
        self.syntax_r = f"jmv::anovaNP(formula={' + '.join(dep_vars)} ~ {group}, data=data, pairs={str(dscf).upper()})"
        self.run()

    def run(self):
        df = self.dataset
        groups = sorted(df[self.group].dropna().unique(), key=str)

        tbl = ResultsTable(name="kruskal", title="Kruskal-Wallis")
        tbl.add_column("var", "Variable", col_type="text")
        tbl.add_column("chi2", "χ²", col_type="number")
        tbl.add_column("df", "df", col_type="integer")
        tbl.add_column("p", "p", col_type="number", fmt="pvalue")
        tbl.add_column("eps2", "ε²", col_type="number", fmt="zto")

        pw_tables = []
        apa = []
        for v in self.deps:
            clean = df[[v, self.group]].dropna().copy()
            clean[v] = pd.to_numeric(clean[v], errors="coerce")
            clean = clean.dropna()
            sub_arrays = [clean[clean[self.group] == g][v].values for g in groups if (clean[self.group] == g).sum() >= 1]
            if len(sub_arrays) < 2:
                continue
            h_stat, p_val = stats.kruskal(*sub_arrays)
            n_tot = len(clean)
            eps2 = float(h_stat / ((n_tot ** 2 - 1.0) / (n_tot + 1.0))) if n_tot > 1 else np.nan
            tbl.add_row({"var": v, "chi2": float(h_stat), "df": len(sub_arrays) - 1, "p": float(p_val), "eps2": eps2})

            p_txt = "< .001" if p_val < 0.001 else f"= {p_val:.3f}"
            apa.append(f"Kruskal-Wallis test for {v} across {self.group} yielded χ²({len(sub_arrays)-1}) = {h_stat:.2f}, p {p_txt}, ε² = {eps2:.3f}.")

            if self.options.get("dscf", True) and len(groups) >= 2:
                pw = pg.pairwise_tests(data=clean, dv=v, between=self.group, parametric=False, padjust="holm")
                pt = ResultsTable(name=f"pw_{v}", title=f"Dwass-Steel-Critchlow-Fligner / Holm Rank Pairwise Comparisons — {v}")
                pt.add_column("g1", "Group 1", col_type="text")
                pt.add_column("g2", "Group 2", col_type="text")
                pt.add_column("u", "U-stat", col_type="number")
                pt.add_column("p_unc", "p-unc", col_type="number", fmt="pvalue")
                pt.add_column("p_adj", "p-adj (Holm)", col_type="number", fmt="pvalue")
                for _, r in pw.iterrows():
                    u_v = float(r["U_val"] if "U_val" in r else r["U-val"])
                    pu = float(r["p_unc"] if "p_unc" in r else r["p-unc"])
                    pc = float(r["p_corr"] if "p_corr" in r else r.get("p-corr", pu))
                    pt.add_row({
                        "g1": str(r["A"]), "g2": str(r["B"]),
                        "u": u_v,
                        "p_unc": pu,
                        "p_adj": pc
                    })
                pw_tables.append(pt)

            if self.options.get("plot", False):
                self.add_plot(f"Medians & Distribution — {v}", plot_mean_ci(clean, v, self.group))

        self.add_table(tbl)
        for pt in pw_tables:
            self.add_table(pt)
        self.apa_narrative = " ".join(apa)

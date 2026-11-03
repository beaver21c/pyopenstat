"""
pyjamovi Factor & Scale Psychometrics Suite (v2.0)
- Reliability (jmv::reliability): Cronbach's α + 95% CI, McDonald's ω, Item-Rest Correlation, α/ω if Item Dropped, Heatmap
- FactorAnalysis (jmv::pca & jmv::efa): PCA & EFA (MinRes / ML / Principal), Varimax / Promax / Oblimin Rotation, KMO & Bartlett's Sphericity, Eigenvalues, Parallel Analysis Scree Plot
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
import pingouin as pg
import factor_analyzer.factor_analyzer as _fa_mod
import factor_analyzer.rotator as _rot_mod
from sklearn.utils.validation import check_array as _sk_check_array


def _compat_check_array(*args, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _sk_check_array(*args, **kwargs)


_fa_mod.check_array = _compat_check_array
if hasattr(_rot_mod, "check_array"):
    _rot_mod.check_array = _compat_check_array

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

from .base import BaseAnalysis, ResultsTable
from .plots import plot_scree, plot_correlation_heatmap


def _compute_mcdonald_omega(df_items: pd.DataFrame) -> float:
    """Compute McDonald's omega total from a 1-factor solution."""
    if df_items.shape[1] < 2 or len(df_items) < 5:
        return np.nan
    try:
        fa = FactorAnalyzer(n_factors=1, rotation=None, method="minres")
        fa.fit(df_items)
        loadings = fa.loadings_.flatten()
        uniquenesses = fa.get_uniquenesses()
        sum_l_sq = float(np.sum(loadings) ** 2)
        denom = sum_l_sq + float(np.sum(uniquenesses))
        return float(sum_l_sq / denom) if denom > 0 else np.nan
    except Exception:
        return np.nan


class Reliability(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        vars: List[str],
        rev_items: Optional[List[str]] = None,
        alpha_scale: bool = True,
        omega_scale: bool = True,
        mean_sd: bool = True,
        item_stats: bool = True,
        cor_plot: bool = True
    ):
        v_list = [vars] if isinstance(vars, str) else list(vars)
        rev_list = list(rev_items) if rev_items else []
        super().__init__(
            name="reliability",
            title="Reliability Analysis",
            dataset=data,
            options={
                "vars": v_list,
                "revItems": rev_list,
                "alphaScale": alpha_scale,
                "omegaScale": omega_scale,
                "meanSd": mean_sd,
                "itemStats": item_stats,
                "corPlot": cor_plot
            }
        )
        self.vars = v_list
        self.rev_items = rev_list
        self.syntax_python = f"jmv.reliability(df, vars={v_list!r}, rev_items={rev_list!r})"
        self.syntax_r = f"jmv::reliability(data=data, vars=vars({', '.join(v_list)}), omegaScale=TRUE, corPlot=TRUE)"
        self.run()

    def run(self):
        if len(self.vars) < 2:
            raise ValueError("Reliability Analysis requires at least 2 scale items.")
        clean = self.dataset[self.vars].apply(pd.to_numeric, errors="coerce").dropna().copy()

        for rv in self.rev_items:
            if rv in clean.columns:
                clean[rv] = (clean[rv].max() + clean[rv].min()) - clean[rv]

        alpha_val, alpha_ci = pg.cronbach_alpha(data=clean)
        omega_val = _compute_mcdonald_omega(clean)
        scale_scores = clean.mean(axis=1)

        scale_tbl = ResultsTable(name="scale", title="Scale Reliability Statistics")
        scale_tbl.add_column("scale", "", col_type="text")
        if self.options.get("meanSd", True):
            scale_tbl.add_column("mean", "Mean", col_type="number")
            scale_tbl.add_column("sd", "SD", col_type="number")
        if self.options.get("alphaScale", True):
            scale_tbl.add_column("alpha", "Cronbach's α", col_type="number", fmt="zto")
            scale_tbl.add_column("ci_l", "95% CI Lower", col_type="number", fmt="zto")
            scale_tbl.add_column("ci_u", "95% CI Upper", col_type="number", fmt="zto")
        if self.options.get("omegaScale", True):
            scale_tbl.add_column("omega", "McDonald's ω", col_type="number", fmt="zto")

        scale_tbl.add_row({
            "scale": "scale",
            "mean": float(scale_scores.mean()),
            "sd": float(scale_scores.std(ddof=1)),
            "alpha": float(alpha_val),
            "ci_l": float(alpha_ci[0]),
            "ci_u": float(alpha_ci[1]),
            "omega": float(omega_val)
        })
        if self.rev_items:
            scale_tbl.add_note(f"Reverse-scaled items: {', '.join(self.rev_items)}")
        self.add_table(scale_tbl)

        if self.options.get("itemStats", True):
            item_tbl = ResultsTable(name="items", title="Item Reliability Statistics")
            item_tbl.add_column("item", "Item", col_type="text")
            item_tbl.add_column("mean", "Mean", col_type="number")
            item_tbl.add_column("sd", "SD", col_type="number")
            item_tbl.add_column("item_rest", "Item-rest correlation", col_type="number", fmt="zto")
            item_tbl.add_column("alpha_drop", "Cronbach's α (if item dropped)", col_type="number", fmt="zto")
            item_tbl.add_column("omega_drop", "McDonald's ω (if item dropped)", col_type="number", fmt="zto")

            for col in self.vars:
                others = [c for c in self.vars if c != col]
                rest_sum = clean[others].sum(axis=1)
                ir_corr = float(clean[col].corr(rest_sum))
                a_drop = float(pg.cronbach_alpha(data=clean[others])[0]) if len(others) >= 2 else np.nan
                w_drop = _compute_mcdonald_omega(clean[others]) if len(others) >= 2 else np.nan
                lbl = f"{col} ᵃ" if col in self.rev_items else col
                item_tbl.add_row({
                    "item": lbl,
                    "mean": float(clean[col].mean()),
                    "sd": float(clean[col].std(ddof=1)),
                    "item_rest": ir_corr,
                    "alpha_drop": a_drop,
                    "omega_drop": w_drop
                })
            self.add_table(item_tbl)

        if self.options.get("corPlot", True):
            self.add_plot("Inter-Item Correlation Heatmap", plot_correlation_heatmap(clean, self.vars))

        self.apa_narrative = (
            f"Internal consistency reliability across {len(self.vars)} items (N = {len(clean)}) yielded "
            f"Cronbach's α = {alpha_val:.3f} (95% CI [{alpha_ci[0]:.3f}, {alpha_ci[1]:.3f}]) and McDonald's ω = {omega_val:.3f}."
        )


class FactorAnalysis(BaseAnalysis):
    def __init__(
        self,
        data: pd.DataFrame,
        vars: List[str],
        mode: str = "efa",             # "pca" or "efa"
        n_factors: Optional[int] = None,
        rotation: str = "varimax",     # "varimax", "promax", "oblimin", "none"
        extraction: str = "minres",    # "minres", "ml", "principal"
        hide_below: float = 0.30,
        kmo_bartlett: bool = True,
        eigen: bool = True,
        scree_plot: bool = True
    ):
        v_list = [vars] if isinstance(vars, str) else list(vars)
        title = "Principal Component Analysis" if mode.lower() == "pca" else "Exploratory Factor Analysis"
        super().__init__(
            name="factorAnalysis",
            title=title,
            dataset=data,
            options={
                "vars": v_list,
                "mode": mode.lower(),
                "nFactors": n_factors,
                "rotation": rotation,
                "extraction": extraction,
                "hideBelow": float(hide_below),
                "kmoBartlett": kmo_bartlett,
                "eigen": eigen,
                "screePlot": scree_plot
            }
        )
        self.vars = v_list
        self.mode = mode.lower()
        self.n_factors = n_factors
        self.rotation = None if rotation.lower() in ("none", "") else rotation.lower()
        self.extraction = "principal" if self.mode == "pca" else extraction.lower()
        self.hide_below = float(hide_below)
        fn_name = "pca" if self.mode == "pca" else "efa"
        self.syntax_python = f"jmv.{fn_name}(df, vars={v_list!r}, n_factors={n_factors!r}, rotation={rotation!r})"
        self.syntax_r = f"jmv::{fn_name}(data=data, vars=vars({', '.join(v_list)}), rotation={(rotation or 'none')!r}, screePlot=TRUE)"
        self.run()

    def run(self):
        if len(self.vars) < 2:
            raise ValueError("PCA / EFA requires at least 2 continuous variables.")
        clean = self.dataset[self.vars].apply(pd.to_numeric, errors="coerce").dropna()
        p = len(self.vars)
        n = len(clean)

        # Correlation eigenvalues + Simulated Parallel Analysis (Horn, 1965)
        corr_mat = clean.corr().values
        ev_raw = np.sort(np.linalg.eigvalsh(corr_mat))[::-1]
        rng = np.random.default_rng(42)
        sim_evs = np.zeros((100, p))
        for b in range(100):
            rand_data = rng.standard_normal((n, p))
            r_c = np.corrcoef(rand_data, rowvar=False)
            sim_evs[b] = np.sort(np.linalg.eigvalsh(r_c))[::-1]
        pa_95 = np.percentile(sim_evs, 95, axis=0)

        if self.n_factors is not None and int(self.n_factors) >= 1:
            nf = min(int(self.n_factors), p - (0 if self.mode == "pca" else 1))
            nf = max(1, nf)
        else:
            # Parallel analysis criterion fallback to Kaiser EV > 1
            nf_pa = int(np.sum(ev_raw > pa_95))
            nf_kaiser = int(np.sum(ev_raw > 1.0))
            nf = max(1, nf_pa if nf_pa >= 1 else nf_kaiser)
            if self.mode == "efa" and nf >= p:
                nf = max(1, p - 1)

        rot = self.rotation if nf > 1 else None
        fa = FactorAnalyzer(n_factors=nf, rotation=rot, method=self.extraction, is_corr_matrix=False)
        fa.fit(clean)

        loadings = fa.loadings_
        uniqueness = fa.get_uniquenesses()
        ss_loadings, prop_var, cum_var = fa.get_factor_variance()

        # 1. Loadings Table
        lbl_prefix = "Component" if self.mode == "pca" else "Factor"
        load_tbl = ResultsTable(name="loadings", title=f"{lbl_prefix} Loadings")
        load_tbl.add_column("var", "Variable", col_type="text")
        for f_idx in range(nf):
            load_tbl.add_column(f"f{f_idx+1}", f"{lbl_prefix} {f_idx+1}", col_type="text")
        load_tbl.add_column("uniq", "Uniqueness", col_type="number", fmt="zto")

        for r_idx, v_name in enumerate(self.vars):
            row = {"var": v_name, "uniq": float(uniqueness[r_idx])}
            for f_idx in range(nf):
                lv = float(loadings[r_idx, f_idx])
                row[f"f{f_idx+1}"] = "" if abs(lv) < self.hide_below else f"{lv:.3f}"
            load_tbl.add_row(row)
        load_tbl.add_note(f"Extraction: '{self.extraction}' with '{rot or 'none'}' rotation. Loadings below {self.hide_below:.2f} hidden.")
        self.add_table(load_tbl)

        # 2. Summary Variance & Eigenvalues Table
        if self.options.get("eigen", True):
            sum_tbl = ResultsTable(name="variance", title=f"{lbl_prefix} Summary & Eigenvalues")
            sum_tbl.add_column("comp", lbl_prefix, col_type="text")
            sum_tbl.add_column("ev", "Initial Eigenvalue", col_type="number")
            sum_tbl.add_column("ssl", "SS Loadings", col_type="number")
            sum_tbl.add_column("pct", "% of Variance", col_type="number", fmt="pc")
            sum_tbl.add_column("cum", "Cumulative %", col_type="number", fmt="pc")
            for f_idx in range(nf):
                sum_tbl.add_row({
                    "comp": f"{f_idx+1}",
                    "ev": float(ev_raw[f_idx]),
                    "ssl": float(ss_loadings[f_idx]),
                    "pct": float(prop_var[f_idx] * 100.0),
                    "cum": float(cum_var[f_idx] * 100.0)
                })
            self.add_table(sum_tbl)

        # 3. KMO & Bartlett's Sphericity
        kmo_overall = np.nan
        bart_chi, bart_p = np.nan, np.nan
        if self.options.get("kmoBartlett", True):
            try:
                bart_chi, bart_p = calculate_bartlett_sphericity(clean)
                kmo_per_item, kmo_overall = calculate_kmo(clean)

                bt = ResultsTable(name="bartlett", title="Bartlett's Test of Sphericity & KMO Measure of Sampling Adequacy")
                bt.add_column("chi2", "Bartlett's χ²", col_type="number")
                bt.add_column("df", "df", col_type="integer")
                bt.add_column("p", "p", col_type="number", fmt="pvalue")
                bt.add_column("kmo", "Overall KMO MSA", col_type="number", fmt="zto")
                bt.add_row({
                    "chi2": float(bart_chi),
                    "df": int(p * (p - 1) // 2),
                    "p": float(bart_p),
                    "kmo": float(kmo_overall)
                })
                self.add_table(bt)
            except Exception:
                pass

        if self.options.get("screePlot", True):
            self.add_plot("Scree Plot with Parallel Analysis", plot_scree(ev_raw, pa_95))

        cum_pct = float(cum_var[-1] * 100.0)
        self.apa_narrative = (
            f"{self.title} extracted {nf} {lbl_prefix.lower()}(s) ({rot or 'unrotated'}), accounting for "
            f"{cum_pct:.1f}% of variance (KMO = {kmo_overall:.3f}, Bartlett's χ² = {bart_chi:.2f})."
        )

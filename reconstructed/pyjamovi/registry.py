"""
pyjamovi Unified Analysis Registry & Dispatcher (v2.0)
Maps all 27 jamovi core & jmv-library modules with consistent parameter schemas for Web UI, Python SDK, CLI, and MCP.
"""

import re
from typing import Dict, Any, List
import pandas as pd

from .analyses import (
    BaseAnalysis,
    Descriptives,
    TTestIS,
    TTestPS,
    TTestOneS,
    AnovaOneW,
    FactorialANOVA,
    ANCOVA,
    RepeatedMeasuresANOVA,
    KruskalWallis,
    CorrMatrix,
    LinReg,
    LogisticBinomial,
    PartialCorrelation,
    ContTables,
    BayesianTTestIS,
    BayesianANOVA,
    BayesianCorrelation,
    Reliability,
    FactorAnalysis,
    NetworkAnalysis,
    GraphNetworkAnalysis,
    TextKeywords,
    TextNetwork,
    TextCluster,
    SummaryTTest,
    SpatialAnalysis,
    SpatialWeightsAnalysis,
    SpatialAutocorrelation,
    SpatialPanelAnalysis,
    GWRAnalysis,
    SpatialMapAnalysis,
    SEM,
    Mediation,
    Moderation,
    PanelAnalysis,
    MixedLinearModel,
    SurvivalAnalysis,
    ScatrPlot,
    GraphBuilder,
    CHART_TYPES,
)


def _as_list(val: Any) -> List[str]:
    if val is None or val == "":
        return []
    if isinstance(val, str):
        return [v.strip() for v in val.split(",") if v.strip()]
    return [str(v) for v in val if v is not None and str(v) != ""]


def _as_single(val: Any) -> Any:
    if isinstance(val, list):
        return val[0] if len(val) > 0 else None
    if val == "":
        return None
    return val


def _require_columns(df: pd.DataFrame, label: str, required: Dict[str, Any], optional: Dict[str, Any]) -> None:
    """Fail with a clear message instead of guessing when a variable is missing or misspelled."""
    absent = [role for role, col in required.items() if not col]
    if absent:
        raise ValueError(f"{label} requires: {', '.join(absent)}. Available columns: {list(df.columns)}")
    not_found = []
    for role, col in list(required.items()) + list(optional.items()):
        for c in (col if isinstance(col, list) else [col]):
            if c and c not in df.columns:
                not_found.append(f"{role}='{c}'")
    if not_found:
        raise ValueError(f"{label}: column(s) not found: {', '.join(not_found)}. Available columns: {list(df.columns)}")


def _opt(p: Dict[str, Any], camel: str, snake: str, default: Any) -> Any:
    """UI params use camelCase, SDK/CLI users may pass snake_case; None/"" mean 'not given'."""
    for k in (camel, snake):
        if k in p and p[k] is not None and p[k] != "":
            return p[k]
    return default


def _flag(p: Dict[str, Any], camel: str, snake: str, default: bool) -> bool:
    v = _opt(p, camel, snake, default)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def _text_prep(p: Dict[str, Any]) -> Dict[str, Any]:
    """Shared Korean text pre-processing options (kiwipiepy tokenization)."""
    return {
        "posMode": str(_opt(p, "posMode", "pos_mode", "nouns_foreign")),
        "minLen": int(_opt(p, "minLen", "min_len", 2)),
        "compound": _flag(p, "compound", "compound", True),
        "defaultStop": _flag(p, "defaultStop", "default_stop", True),
        "stopwords": _opt(p, "stopwords", "stop_words", ""),
        "userWords": _opt(p, "userWords", "user_words", ""),
        "synonyms": _opt(p, "synonyms", "synonym_rules", ""),
    }


ANALYSIS_CATALOG: List[Dict[str, Any]] = [
    {"id": "descriptives", "group": "Exploration", "title": "Descriptives", "module": "jmv", "icon": "analysis-descriptives.svg"},
    {"id": "scatr", "group": "Exploration", "title": "Scatterplot", "module": "scatr", "icon": "analysis-scatr-jmvscatter.svg"},
    {"id": "pareto", "group": "Exploration", "title": "Pareto Chart", "module": "scatr", "icon": "analysis-scatr-jmvbar.svg"},

    {"id": "ttestIS", "group": "T-Tests", "title": "Independent Samples T-Test", "module": "jmv", "icon": "analysis-ttest.svg"},
    {"id": "ttestPS", "group": "T-Tests", "title": "Paired Samples T-Test", "module": "jmv", "icon": "analysis-ttest.svg"},
    {"id": "ttestOneS", "group": "T-Tests", "title": "One Sample T-Test", "module": "jmv", "icon": "analysis-ttest.svg"},
    {"id": "summaryTTest", "group": "T-Tests", "title": "Summary Statistics T-Test (JASP)", "module": "jasp", "icon": "analysis-ttest.svg"},
    {"id": "bayesTTestIS", "group": "T-Tests", "title": "Bayesian Independent T-Test", "module": "jsq", "icon": "analysis-ttest.svg"},

    {"id": "anovaOneW", "group": "ANOVA", "title": "One-Way ANOVA", "module": "jmv", "icon": "analysis-anova.svg"},
    {"id": "anovaFactorial", "group": "ANOVA", "title": "ANOVA (Factorial)", "module": "jmv", "icon": "analysis-anova.svg"},
    {"id": "anovaRM", "group": "ANOVA", "title": "Repeated Measures ANOVA", "module": "jmv", "icon": "analysis-anova.svg"},
    {"id": "ancova", "group": "ANOVA", "title": "ANCOVA", "module": "jmv", "icon": "analysis-anova.svg"},
    {"id": "kruskal", "group": "ANOVA", "title": "One-Way ANOVA (Non-parametric)", "module": "jmv", "icon": "analysis-anova.svg"},
    {"id": "bayesANOVA", "group": "ANOVA", "title": "Bayesian ANOVA", "module": "jsq", "icon": "analysis-anova.svg"},

    {"id": "corrMatrix", "group": "Regression", "title": "Correlation Matrix", "module": "jmv", "icon": "analysis-regression.svg"},
    {"id": "partCorr", "group": "Regression", "title": "Partial Correlation", "module": "jmv", "icon": "analysis-regression.svg"},
    {"id": "bayesCorr", "group": "Regression", "title": "Bayesian Correlation", "module": "jsq", "icon": "analysis-regression.svg"},
    {"id": "linReg", "group": "Regression", "title": "Linear Regression", "module": "jmv", "icon": "analysis-regression.svg"},
    {"id": "logRegBin", "group": "Regression", "title": "2 Outcomes (Binomial Logistic)", "module": "jmv", "icon": "analysis-regression.svg"},
    {"id": "compareModels", "group": "Regression", "title": "Model Comparison Table (side-by-side, esttab)", "module": "openstat", "icon": "analysis-regression.svg"},

    {"id": "contTables", "group": "Frequencies", "title": "Independent Samples (χ² Association)", "module": "jmv", "icon": "analysis-frequencies.svg"},

    {"id": "reliability", "group": "Factor", "title": "Reliability Analysis", "module": "jmv", "icon": "analysis-factor.svg"},
    {"id": "pca", "group": "Factor", "title": "Principal Component Analysis", "module": "jmv", "icon": "analysis-factor.svg"},
    {"id": "efa", "group": "Factor", "title": "Exploratory Factor Analysis", "module": "jmv", "icon": "analysis-factor.svg"},
    {"id": "networkAnalysis", "group": "Factor", "title": "Network Analysis (JASP)", "module": "jasp", "icon": "analysis-factor.svg"},

    {"id": "sem", "group": "SEM", "title": "Structural Equation Modeling (CFA / SEM, Mplus defaults)", "module": "openstat", "icon": "analysis-sem.svg"},
    {"id": "lca", "group": "Mixture", "title": "Latent Class / Latent Profile Analysis (LCA / LPA)", "module": "openstat", "icon": "analysis-mixture.svg"},
    {"id": "lta", "group": "Mixture", "title": "Latent Transition Analysis (LTA)", "module": "openstat", "icon": "analysis-mixture.svg"},
    {"id": "lcga", "group": "Mixture", "title": "Latent Class Growth Analysis (LCGA)", "module": "openstat", "icon": "analysis-mixture.svg"},
    {"id": "mediation", "group": "MedMod", "title": "Mediation", "module": "medmod", "icon": "analysis-medmod.svg"},
    {"id": "moderation", "group": "MedMod", "title": "Moderation", "module": "medmod", "icon": "analysis-medmod.svg"},
    {"id": "panel", "group": "Linear Models", "title": "Panel Econometrics (FE / RE / Hausman)", "module": "plm", "icon": "analysis-linearmodels.svg"},
    {"id": "mixedModel", "group": "Linear Models", "title": "Linear Mixed Model (Multilevel)", "module": "gamlj", "icon": "analysis-linearmodels.svg"},
    {"id": "survival", "group": "Survival", "title": "Kaplan-Meier & Cox PH Survival", "module": "jsurvival", "icon": "analysis-survival.svg"},
    {"id": "spatialWeights", "group": "Spatial", "title": "Spatial Weights Matrix (contiguity / kNN / distance)", "module": "geoda", "icon": "analysis-spatial.svg"},
    {"id": "spatial_moran", "group": "Spatial", "title": "Spatial Autocorrelation (Moran's I / LISA / Gi*)", "module": "geoda", "icon": "analysis-spatial.svg"},
    {"id": "spatial", "group": "Spatial", "title": "Spatial Regression (OLS-LM / SAR / SEM / SDM / SDEM / SLX / SAC)", "module": "geoda", "icon": "analysis-spatial.svg"},
    {"id": "spatialPanel", "group": "Spatial", "title": "Spatial Panel Regression (FE / RE, Stata xsmle)", "module": "geoda", "icon": "analysis-spatial.svg"},
    {"id": "gwr", "group": "Spatial", "title": "Geographically Weighted Regression (GWR)", "module": "geoda", "icon": "analysis-spatial.svg"},
    {"id": "spatialMap", "group": "Spatial", "title": "Thematic Map (Choropleth)", "module": "geoda", "icon": "analysis-spatial.svg"},
    {"id": "graph", "group": "Graphs", "title": "Graph Builder (15 Chart Types)", "module": "openstat", "icon": "analysis-scatr-jmvbar.svg"},

    {"id": "textKeywords", "group": "Text", "title": "Keyword Analysis (Frequency / TF-IDF / Word Cloud)", "module": "openstat", "icon": "analysis-text.svg"},
    {"id": "textNetwork", "group": "Text", "title": "Keyword Co-occurrence Network", "module": "openstat", "icon": "analysis-text.svg"},
    {"id": "textCluster", "group": "Text", "title": "Document Clustering (Embedding / Topics)", "module": "openstat", "icon": "analysis-text.svg"},
    {"id": "graphNetwork", "group": "Network", "title": "Relational Network Analysis (Edge List)", "module": "openstat", "icon": "analysis-network.svg"},
]

BASE_ANALYSIS_CATALOG: List[Dict[str, Any]] = list(ANALYSIS_CATALOG)


def refresh_analysis_catalog() -> List[Dict[str, Any]]:
    """Refreshes `ANALYSIS_CATALOG` in place with any newly added clean-room plugins in `pyjamovi/plugins/`."""
    from .plugins import get_plugin_catalog_entries
    base_ids = {item["id"] for item in BASE_ANALYSIS_CATALOG}
    merged = list(BASE_ANALYSIS_CATALOG)
    for p_entry in get_plugin_catalog_entries():
        if p_entry["id"] not in base_ids:
            merged.append(p_entry)
    ANALYSIS_CATALOG.clear()
    ANALYSIS_CATALOG.extend(merged)
    return list(ANALYSIS_CATALOG)


def get_full_catalog() -> List[Dict[str, Any]]:
    return refresh_analysis_catalog()


refresh_analysis_catalog()


from .analyses.plot_style import use_plot_style


def dispatch_analysis(analysis_type: str, df: pd.DataFrame, params: Dict[str, Any]) -> BaseAnalysis:
    """Execute any registered statistical analysis with optional PlotStyle customization."""
    p = params or {}
    plot_style = p.get("plotStyle") or p.get("plot_style")
    with use_plot_style(plot_style):
        res = _dispatch_inner(analysis_type, df, p)
    if plot_style:
        res.options["plotStyle"] = plot_style
    return res


def _dispatch_inner(analysis_type: str, df: pd.DataFrame, p: Dict[str, Any]) -> BaseAnalysis:
    t = analysis_type.strip()

    # Aliases for CLI / MCP ergonomics
    alias_map = {
        "ttest": "ttestIS",
        "anova": "anovaOneW",
        "corrmatrix": "corrMatrix",
        "corr": "corrMatrix",
        "regression": "linReg",
        "reg": "linReg",
        "crosstab": "contTables",
        "desc": "descriptives",
        "logistic": "logRegBin",
        "bayes_ttest": "bayesTTestIS",
        "bayes_anova": "bayesANOVA",
        "bayes_corr": "bayesCorr",
        "rm_anova": "anovaRM",
        "factorial_anova": "anovaFactorial",
        "mixed": "mixedModel",
        "text_keywords": "textKeywords",
        "keywords": "textKeywords",
        "wordcloud": "textKeywords",
        "text_network": "textNetwork",
        "cooccurrence": "textNetwork",
        "text_cluster": "textCluster",
        "topics": "textCluster",
        "graph_network": "graphNetwork",
        "edge_network": "graphNetwork",
        "sna": "graphNetwork",
        "spatial_weights": "spatialWeights",
        "spmatrix": "spatialWeights",
        "spatialMoran": "spatial_moran",
        "spatial_autocorrelation": "spatial_moran",
        "moran": "spatial_moran",
        "lisa": "spatial_moran",
        "spatial_regression": "spatial",
        "spregress": "spatial",
        "spatialAnalysis": "spatial",
        "spatial_panel": "spatialPanel",
        "xsmle": "spatialPanel",
        "spxtregress": "spatialPanel",
        "geographically_weighted_regression": "gwr",
        "mgwr": "gwr",
        "spatial_map": "spatialMap",
        "choropleth": "spatialMap",
        "latent_class": "lca",
        "latent_profile": "lca",
        "lpa": "lca",
        "latent_transition": "lta",
        "latent_class_growth": "lcga",
        "traj": "lcga",
        "group_based_trajectory": "lcga",
        "cfa": "sem",
        "compare_models": "compareModels",
        "model_comparison": "compareModels",
        "modelcomparison": "compareModels",
        "esttab": "compareModels",
        "compare": "compareModels",
    }
    if t == "lpa" and p.get("vars") and not p.get("continuous"):
        p = dict(p, continuous=p.get("vars"))
    t = alias_map.get(t, t)

    if t == "descriptives":
        return Descriptives(
            df,
            vars=_as_list(p.get("vars") or p.get("dep")),
            split_by=_as_single(p.get("splitBy") or p.get("split_by") or p.get("group")),
            freq=bool(p.get("freq", False)),
            n=bool(p.get("n", True)),
            missing=bool(p.get("missing", True)),
            mean=bool(p.get("mean", True)),
            median=bool(p.get("median", True)),
            mode=bool(p.get("mode", False)),
            sd=bool(p.get("sd", True)),
            variance=bool(p.get("variance", False)),
            iqr=bool(p.get("iqr", False)),
            se=bool(p.get("se", True)),
            min=bool(p.get("min", True)),
            max=bool(p.get("max", True)),
            skew=bool(p.get("skew", False)),
            kurt=bool(p.get("kurt", False)),
            sw=bool(p.get("sw", False)),
            hist=bool(p.get("hist", p.get("plots_hist", False))),
            box=bool(p.get("box", p.get("plots_box", False))),
            violin=bool(p.get("violin", False)),
            bar=bool(p.get("bar", False)),
            qq=bool(p.get("qq", False))
        )

    elif t in ("scatr", "pareto"):
        mode = "pareto" if t == "pareto" else str(p.get("mode", "scatter"))
        return ScatrPlot(
            df,
            x=_as_single(p.get("x") or p.get("vars")),
            y=_as_single(p.get("y") or p.get("dep")),
            group=_as_single(p.get("group")),
            mode=mode,
            reg_line=str(p.get("regLine", p.get("reg_line", "linear"))),
            marginal=str(p.get("marginal", "dens"))
        )

    elif t in ("graph", "chart", "plot") or t in CHART_TYPES:
        default_ct = t if t in CHART_TYPES else "bar"
        ct = str(p.get("chartType") or p.get("chart_type") or default_ct).strip().lower()
        return GraphBuilder(
            df,
            chart_type=ct,
            x=_as_single(p.get("x")),
            y=_as_single(p.get("y") or p.get("dep")),
            vars=_as_list(p.get("vars")),
            group=_as_single(p.get("group") or p.get("splitBy") or p.get("split_by")),
            size_var=_as_single(p.get("sizeVar") or p.get("size_var")),
            agg=str(p.get("agg", "mean")),
            error_bar=str(p.get("errorBar", p.get("error_bar", "ci95"))),
            horizontal=bool(p.get("horizontal", False)),
            stacked_100=bool(p.get("stacked100", p.get("stacked_100", False))),
            show_values=bool(p.get("showValues", p.get("show_values", False))),
            jitter=bool(p.get("jitter", False)),
            fit_line=str(p.get("fitLine", p.get("fit_line", "linear"))),
            bins=int(p.get("bins", 18)),
            show_summary_table=bool(p.get("showSummaryTable", p.get("show_summary_table", True))),
            caption=str(p.get("caption", "")),
        )

    elif t == "ttestIS":
        deps = _as_list(p.get("vars") or p.get("dep") or p.get("deps"))
        return TTestIS(
            df,
            vars=deps,
            group=_as_single(p.get("group")),
            hypothesis=str(p.get("hypothesis", "different")),
            students=bool(p.get("students", True)),
            welchs=bool(p.get("welchs", False)),
            mann=bool(p.get("mann", False)),
            mean_diff=bool(p.get("meanDiff", p.get("mean_diff", True))),
            effect_size=bool(p.get("effectSize", p.get("effect_size", True))),
            ci=bool(p.get("ci", False)),
            ci_width=float(p.get("ciWidth", 95.0)),
            desc=bool(p.get("desc", True)),
            norm=bool(p.get("norm", False)),
            eqv=bool(p.get("eqv", False)),
            plots=bool(p.get("plots", p.get("plots_mean", False))),
            qq=bool(p.get("qq", p.get("plots_qq", False))),
            raincloud=bool(p.get("raincloud", False))
        )

    elif t == "ttestPS":
        pairs = p.get("pairs")
        v_list = _as_list(p.get("vars"))
        if not pairs and len(v_list) >= 2:
            pairs = [(v_list[i], v_list[i + 1]) for i in range(0, len(v_list) - 1, 2)]
        return TTestPS(
            df,
            pairs=pairs,
            var1=_as_single(p.get("var1")),
            var2=_as_single(p.get("var2")),
            students=bool(p.get("students", True)),
            wilcoxon=bool(p.get("wilcoxon", False)),
            hypothesis=str(p.get("hypothesis", "different")),
            mean_diff=bool(p.get("meanDiff", True)),
            effect_size=bool(p.get("effectSize", True)),
            desc=bool(p.get("desc", True)),
            norm=bool(p.get("norm", False)),
            plots=bool(p.get("plots", False))
        )

    elif t == "ttestOneS":
        return TTestOneS(
            df,
            vars=_as_list(p.get("vars") or p.get("dep")),
            test_value=float(p.get("testValue", p.get("test_value", 0.0))),
            students=bool(p.get("students", True)),
            wilcoxon=bool(p.get("wilcoxon", False)),
            hypothesis=str(p.get("hypothesis", "different")),
            mean_diff=bool(p.get("meanDiff", True)),
            effect_size=bool(p.get("effectSize", True)),
            desc=bool(p.get("desc", True)),
            plots=bool(p.get("plots", False))
        )

    elif t in ("summaryTTest", "summary_ttest", "summaryttest"):
        return SummaryTTest(
            m1=float(p.get("m1", 0.0)),
            s1=float(p.get("s1", 1.0)),
            n1=int(p.get("n1", 30)),
            m2=float(p.get("m2", 0.0)),
            s2=float(p.get("s2", 1.0)),
            n2=int(p.get("n2", 30)),
            group1_name=str(p.get("group1_name", p.get("g1_name", "Group 1"))),
            group2_name=str(p.get("group2_name", p.get("g2_name", "Group 2"))),
            hypothesis=str(p.get("hypothesis", "different")),
            prior_width=float(p.get("priorWidth", p.get("prior_width", 0.707)))
        )

    elif t == "bayesTTestIS":
        return BayesianTTestIS(
            df,
            vars=_as_list(p.get("vars") or p.get("dep")),
            group=_as_single(p.get("group")),
            prior_width=float(p.get("priorWidth", p.get("prior_width", 0.707))),
            plot_pp=bool(p.get("plotPP", p.get("plot_pp", True))),
            plot_robust=bool(p.get("plotRobust", p.get("plot_robust", False))),
            desc=bool(p.get("desc", True))
        )

    elif t == "anovaOneW":
        return AnovaOneW(
            df,
            deps=_as_list(p.get("deps") or p.get("dep") or p.get("vars")),
            group=_as_single(p.get("group")),
            welchs=bool(p.get("welchs", True)),
            fishers=bool(p.get("fishers", False)),
            desc=bool(p.get("desc", True)),
            norm=bool(p.get("norm", False)),
            eqv=bool(p.get("eqv", False)),
            ph_method=p.get("phMethod", p.get("ph_method", "tukey")),
            desc_plot=bool(p.get("descPlot", p.get("desc_plot", p.get("plots_mean", False)))),
            qq=bool(p.get("qq", p.get("qq_plot", False)))
        )

    elif t == "anovaFactorial":
        return FactorialANOVA(
            df,
            dep=_as_single(p.get("dep")),
            factors=_as_list(p.get("factors") or p.get("group")),
            ss_type=int(p.get("ssType", p.get("ss_type", 3))),
            eta_sq=bool(p.get("etaSq", True)),
            part_eta_sq=bool(p.get("partEtaSq", True)),
            omega_sq=bool(p.get("omegaSq", False)),
            homo=bool(p.get("homo", False)),
            norm=bool(p.get("norm", False)),
            ph_method=str(p.get("phMethod", p.get("ph_method", "tukey"))),
            emm_plot=bool(p.get("emmPlot", True)),
            qq_plot=bool(p.get("qqPlot", False))
        )

    elif t == "ancova":
        return ANCOVA(
            df,
            dep=_as_single(p.get("dep")),
            factors=_as_list(p.get("factors") or p.get("group")),
            covs=_as_list(p.get("covs")),
            ss_type=int(p.get("ssType", 3)),
            ph_method=str(p.get("phMethod", "tukey")),
            emm_plot=bool(p.get("emmPlot", True))
        )

    elif t == "anovaRM":
        return RepeatedMeasuresANOVA(
            df,
            rm_vars=_as_list(p.get("rmVars") or p.get("rm_vars") or p.get("vars")),
            between=_as_single(p.get("between") or p.get("group")),
            spher=bool(p.get("spher", True)),
            posthoc=bool(p.get("posthoc", True)),
            profile_plot=bool(p.get("profilePlot", True))
        )

    elif t == "kruskal":
        return KruskalWallis(
            df,
            deps=_as_list(p.get("deps") or p.get("dep") or p.get("vars")),
            group=_as_single(p.get("group")),
            dscf=bool(p.get("dscf", True)),
            plot=bool(p.get("plot", False))
        )

    elif t == "bayesANOVA":
        return BayesianANOVA(
            df,
            dep=_as_single(p.get("dep")),
            factors=_as_list(p.get("factors") or p.get("group"))
        )

    elif t == "corrMatrix":
        return CorrMatrix(
            df,
            vars=_as_list(p.get("vars")),
            pearson=bool(p.get("pearson", True)),
            spearman=bool(p.get("spearman", False)),
            kendall=bool(p.get("kendall", False)),
            sig=bool(p.get("sig", True)),
            flag=bool(p.get("flag", True)),
            plots=bool(p.get("plots", p.get("plots_heatmap", False)))
        )

    elif t == "partCorr":
        return PartialCorrelation(
            df,
            vars=_as_list(p.get("vars")),
            controls=_as_list(p.get("controls") or p.get("covs")),
            method=str(p.get("method", "pearson"))
        )

    elif t == "bayesCorr":
        return BayesianCorrelation(
            df,
            vars=_as_list(p.get("vars")),
            plot_heatmap=bool(p.get("plotHeatmap", True))
        )

    elif t == "linReg":
        return LinReg(
            df,
            dep=_as_single(p.get("dep")),
            covs=_as_list(p.get("covs")),
            factors=_as_list(p.get("factors")),
            blocks=p.get("blocks"),
            r2_adj=bool(p.get("r2Adj", True)),
            aic_bic=bool(p.get("aicBic", False)),
            f_test=bool(p.get("fTest", True)),
            std_est=bool(p.get("stdEst", True)),
            ci=bool(p.get("ci", False)),
            collin=bool(p.get("collin", False)),
            dw=bool(p.get("dw", False)),
            norm=bool(p.get("norm", False)),
            cooks=bool(p.get("cooks", False)),
            plots_diag=bool(p.get("plotsDiag", p.get("plots_resid", False)))
        )

    elif t == "logRegBin":
        return LogisticBinomial(
            df,
            dep=_as_single(p.get("dep")),
            covs=_as_list(p.get("covs")),
            factors=_as_list(p.get("factors")),
            odds_ratio=bool(p.get("oddsRatio", True)),
            ci=bool(p.get("ci", True)),
            class_table=bool(p.get("classTable", True)),
            cutoff=float(p.get("cutoff", 0.5)),
            roc_plot=bool(p.get("rocPlot", True))
        )

    elif t == "contTables":
        return ContTables(
            df,
            rows=_as_single(p.get("rows")),
            cols=_as_single(p.get("cols")),
            counts=_as_single(p.get("counts")),
            chi_sq=bool(p.get("chiSq", True)),
            chi_sq_corr=bool(p.get("chiSqCorr", False)),
            like_rat=bool(p.get("likeRat", False)),
            fisher=bool(p.get("fisher", False)),
            phi_cra=bool(p.get("phiCra", True)),
            odds=bool(p.get("odds", False)),
            obs=bool(p.get("obs", True)),
            exp=bool(p.get("exp", False)),
            pc_row=bool(p.get("pcRow", False)),
            pc_col=bool(p.get("pcCol", False)),
            bar_plot=bool(p.get("barPlot", False))
        )

    elif t == "reliability":
        return Reliability(
            df,
            vars=_as_list(p.get("vars")),
            rev_items=_as_list(p.get("revItems") or p.get("rev_items")),
            alpha_scale=bool(p.get("alphaScale", True)),
            omega_scale=bool(p.get("omegaScale", True)),
            mean_sd=bool(p.get("meanSd", True)),
            item_stats=bool(p.get("itemStats", True)),
            cor_plot=bool(p.get("corPlot", True))
        )

    elif t in ("pca", "efa", "factorAnalysis"):
        mode = "pca" if t == "pca" else ("efa" if t == "efa" else str(p.get("mode", "efa")))
        nf = p.get("nFactors", p.get("n_factors"))
        nf_int = int(nf) if nf is not None and str(nf).strip() != "" and int(nf) > 0 else None
        return FactorAnalysis(
            df,
            vars=_as_list(p.get("vars")),
            mode=mode,
            n_factors=nf_int,
            rotation=str(p.get("rotation", "varimax")),
            extraction=str(p.get("extraction", "principal" if mode == "pca" else "minres")),
            hide_below=float(p.get("hideBelow", 0.30)),
            kmo_bartlett=bool(p.get("kmoBartlett", True)),
            eigen=bool(p.get("eigen", True)),
            scree_plot=bool(p.get("screePlot", True))
        )

    elif t in ("networkAnalysis", "network_analysis", "network"):
        return NetworkAnalysis(
            df,
            vars=_as_list(p.get("vars")),
            method=str(p.get("method", "ebicglasso")),
            gamma=float(p.get("gamma", 0.25)),
            plot_network=bool(p.get("plotNetwork", p.get("plot_network", True))),
            plot_centrality=bool(p.get("plotCentrality", p.get("plot_centrality", True)))
        )

    elif t == "sem":
        reserved = ("plotStyle", "plot_style", "model", "model_syntax", "stdEst", "std_est", "fitMeasures", "fit_measures",
                    "variances", "pathDiagram", "path_diagram", "obsVars")
        extra = {k: v for k, v in p.items() if k not in reserved}
        if isinstance(extra.get("indirect"), str):
            ind = [x.strip() for x in re.split(r"[;\n]+", extra["indirect"]) if x.strip()]
            if ind:
                extra["indirect"] = ind
            else:
                extra.pop("indirect")
        if extra.get("group") in ("", None):
            extra.pop("group", None)
            for k in ("invariance", "invarianceTest"):
                extra.pop(k, None)
        if not bool(p.get("stdEst", p.get("std_est", True))):
            extra.pop("standardized", None)
        if extra.get("bootstrap") in (0, "0", "", None):
            extra.pop("bootstrap", None)
            extra.pop("ciType", None)
        return SEM(
            df,
            model_syntax=str(p.get("model") or p.get("model_syntax") or ""),
            std_est=bool(p.get("stdEst", p.get("std_est", True))),
            fit_measures=bool(p.get("fitMeasures", p.get("fit_measures", True))),
            variances=bool(p.get("variances", False)),
            path_diagram=bool(p.get("pathDiagram", p.get("path_diagram", True))),
            **extra,
        )

    elif t == "compareModels":
        from .analyses.model_compare import compare_results, normalize_model_spec
        specs = p.get("models") or []
        if isinstance(specs, str):
            import json as _json
            specs = _json.loads(specs)
        if not isinstance(specs, (list, tuple)) or not specs:
            raise ValueError("모형 비교: models에 비교할 모형 지정 필요 (예: [{'analysis': 'linReg', 'dep': 'y', 'covs': ['x1']}])")
        results = []
        for sp in specs:
            sp = normalize_model_spec(sp)
            a_type = str(sp.pop("analysis"))
            if alias_map.get(a_type, a_type) == "compareModels":
                raise ValueError("모형 비교: 모형 비교표를 다시 비교 대상으로 지정할 수 없음")
            results.append(dispatch_analysis(a_type, df, sp))
        keys = ("models", "names", "stats", "estimate", "labels", "terms", "table", "title", "lang")
        return compare_results(results, names=p.get("names"), stats=p.get("stats", "auto"),
                               estimate=p.get("estimate", "b"), labels=p.get("labels"), terms=p.get("terms"),
                               table=p.get("table"), title=p.get("title"), lang=p.get("lang", "ko"),
                               options={k: p[k] for k in keys if k in p})

    elif t in ("lca", "lta", "lcga"):
        from .analyses.mixture_module import (
            LatentClassAnalysis, LatentTransitionAnalysis, LatentClassGrowthAnalysis,
        )
        q = {k: v for k, v in p.items() if k not in ("plotStyle", "plot_style")}
        for key in ("categorical", "continuous", "covariates", "distalContinuous", "distalCategorical"):
            if key in q and q[key] is not None:
                q[key] = _as_list(q[key])
        if t == "lcga" and "vars" in q and q["vars"] is not None:
            q["vars"] = _as_list(q["vars"])
        if isinstance(q.get("timeScores"), str):
            ts = [x for x in re.split(r"[,\s]+", q["timeScores"]) if x]
            if ts:
                q["timeScores"] = [float(x) for x in ts]
            else:
                q.pop("timeScores")
        if q.get("classRange") in ("", None, []):
            q.pop("classRange", None)
        cls = {"lca": LatentClassAnalysis, "lta": LatentTransitionAnalysis, "lcga": LatentClassGrowthAnalysis}[t]
        return cls(df, **q)

    elif t == "mediation":
        return Mediation(
            df,
            dep=_as_single(p.get("dep")),
            pred=_as_single(p.get("pred")),
            med=_as_single(p.get("med")),
            bootstrap=int(p.get("bootstrap", 500)),
            ci=bool(p.get("ci", True)),
            pm=bool(p.get("pm", True)),
            paths=bool(p.get("paths", True)),
            diagram=bool(p.get("diagram", True))
        )

    elif t == "moderation":
        return Moderation(
            df,
            dep=_as_single(p.get("dep")),
            pred=_as_single(p.get("pred")),
            mod=_as_single(p.get("mod")),
            simple_slopes=bool(p.get("simpleSlopes", True)),
            slope_plot=bool(p.get("slopePlot", True))
        )

    elif t == "panel":
        return PanelAnalysis(
            df,
            entity_col=_as_single(p.get("entity") or p.get("entity_col")),
            time_col=_as_single(p.get("time") or p.get("time_col")),
            dep=_as_single(p.get("dep")),
            covs=_as_list(p.get("covs")),
            model_type=str(p.get("modelType", p.get("model_type", "fixed"))),
            cov_type=str(p.get("covType", p.get("cov_type", "robust"))),
            hausman=bool(p.get("hausman", True)),
            plot_trends=bool(p.get("plotTrends", True))
        )

    elif t == "mixedModel":
        dep_val = _as_single(p.get("dep"))
        cluster_val = _as_single(p.get("cluster") or p.get("random") or p.get("group"))
        covs_val = _as_list(p.get("covs"))
        factors_val = _as_list(p.get("factors"))
        _require_columns(
            df, "Linear mixed model",
            required={"dep": dep_val, "cluster": cluster_val},
            optional={"covs": covs_val, "factors": factors_val},
        )
        return MixedLinearModel(
            df,
            dep=dep_val,
            cluster=cluster_val,
            covs=covs_val,
            factors=factors_val,
            reml=bool(p.get("reml", True))
        )

    elif t == "survival":
        time_c = _as_single(p.get("time") or p.get("time_col"))
        event_c = _as_single(p.get("event") or p.get("event_col"))
        group_c = _as_single(p.get("group"))
        covs_c = _as_list(p.get("covs"))
        _require_columns(
            df, "Survival analysis",
            required={"time": time_c, "event": event_c},
            optional={"group": group_c, "covs": covs_c},
        )
        return SurvivalAnalysis(
            df,
            time_col=time_c,
            event_col=event_c,
            group=group_c,
            covs=covs_c,
            km_plot=bool(p.get("kmPlot", True))
        )

    elif t == "spatial":
        q = dict(p)
        q.pop("plotStyle", None)
        q.pop("plot_style", None)
        if "covs" in q:
            q["covs"] = _as_list(q.get("covs"))
        elif q.get("x") or q.get("indeps"):
            q["covs"] = _as_list(q.get("x") or q.get("indeps"))
        q["dep"] = _as_single(q.get("dep") or q.get("y")) or ""
        return SpatialAnalysis(df, **q)

    elif t in ("spatialWeights", "spatial_moran", "spatialPanel", "gwr", "spatialMap"):
        q = {k: v for k, v in p.items() if k not in ("plotStyle", "plot_style")}
        for key in ("covs", "vars", "durbinVars"):
            if key in q:
                q[key] = _as_list(q[key])
        if t == "spatial_moran" and not q.get("vars") and q.get("dep"):
            q["vars"] = _as_list(q.get("dep"))
        cls = {"spatialWeights": SpatialWeightsAnalysis, "spatial_moran": SpatialAutocorrelation,
               "spatialPanel": SpatialPanelAnalysis, "gwr": GWRAnalysis, "spatialMap": SpatialMapAnalysis}[t]
        return cls(df, **q)

    elif t in ("textKeywords", "textNetwork", "textCluster"):
        text_c = _as_single(p.get("text") or p.get("text_col"))
        group_c = _as_single(p.get("group"))
        label_c = _as_single(p.get("docLabel") or p.get("doc_label"))
        _require_columns(df, "텍스트 분석", required={"text": text_c}, optional={"group": group_c, "docLabel": label_c})
        prep = _text_prep(p)
        seed = int(_opt(p, "seed", "seed", 42))
        if t == "textKeywords":
            return TextKeywords(
                df, text=text_c, group=group_c,
                rank_by=str(_opt(p, "rankBy", "rank_by", "freq")),
                top_n=int(_opt(p, "topN", "top_n", 30)),
                group_top_n=int(_opt(p, "groupTopN", "group_top_n", 10)),
                wordcloud=_flag(p, "wordcloud", "word_cloud", True),
                cloud_weight=str(_opt(p, "cloudWeight", "cloud_weight", "freq")),
                max_words=int(_opt(p, "maxWords", "max_words", 120)),
                cloud_shape=str(_opt(p, "cloudShape", "cloud_shape", "rect")),
                bar_plot=_flag(p, "barPlot", "bar_plot", True),
                group_cloud=_flag(p, "groupCloud", "group_cloud", False),
                seed=seed, **prep,
            )
        if t == "textNetwork":
            return TextNetwork(
                df, text=text_c,
                unit=str(_opt(p, "unit", "unit", "sentence")),
                top_nodes=int(_opt(p, "topNodes", "top_nodes", 60)),
                min_cooc=int(_opt(p, "minCooc", "min_cooc", 2)),
                edge_metric=str(_opt(p, "edgeMetric", "edge_metric", "count")),
                max_edges=int(_opt(p, "maxEdges", "max_edges", 150)),
                community=_flag(p, "community", "community", True),
                resolution=float(_opt(p, "resolution", "resolution", 1.0)),
                layout=str(_opt(p, "layout", "layout", "spring")),
                label_top=int(_opt(p, "labelTop", "label_top", 40)),
                top_pairs=int(_opt(p, "topPairs", "top_pairs", 30)),
                seed=seed, **prep,
            )
        return TextCluster(
            df, text=text_c, doc_label=label_c, group=group_c,
            embedding=str(_opt(p, "embedding", "embedding", "tfidf")),
            model=str(_opt(p, "model", "model_name", "BAAI/bge-m3")),
            method=str(_opt(p, "method", "method", "hdbscan")),
            n_clusters=int(_opt(p, "nClusters", "n_clusters", 0)),
            min_cluster_size=int(_opt(p, "minClusterSize", "min_cluster_size", 0)),
            use_umap=_flag(p, "useUmap", "use_umap", True),
            n_neighbors=int(_opt(p, "nNeighbors", "n_neighbors", 12)),
            assign_noise=_flag(p, "assignNoise", "assign_noise", True),
            noise_assign_min=float(_opt(p, "noiseAssignMin", "noise_assign_min", 0.6)),
            themes=_flag(p, "themes", "themes", True),
            theme_distance=float(_opt(p, "themeDistance", "theme_distance", 0.3)),
            stability=_flag(p, "stability", "stability", False),
            kw_top=int(_opt(p, "kwTop", "kw_top", 8)),
            min_df=int(_opt(p, "minDf", "min_df", 2)),
            cluster_map=_flag(p, "clusterMap", "cluster_map", True),
            cluster_network=_flag(p, "clusterNetwork", "cluster_network", True),
            doc_network=_flag(p, "docNetwork", "doc_network", False),
            doc_sim_min=float(_opt(p, "docSimMin", "doc_sim_min", 0.6)),
            doc_table=_flag(p, "docTable", "doc_table", True),
            seed=seed, **prep,
        )

    elif t == "graphNetwork":
        src_c = _as_single(p.get("source") or p.get("from"))
        tgt_c = _as_single(p.get("target") or p.get("to"))
        w_c = _as_single(p.get("weight"))
        _require_columns(df, "관계망 분석", required={"source": src_c, "target": tgt_c}, optional={"weight": w_c})
        return GraphNetworkAnalysis(
            df, source=src_c, target=tgt_c, weight=w_c,
            directed=_flag(p, "directed", "directed", False),
            min_weight=float(_opt(p, "minWeight", "min_weight", 0.0)),
            community=_flag(p, "community", "community", True),
            resolution=float(_opt(p, "resolution", "resolution", 1.0)),
            size_by=str(_opt(p, "sizeBy", "size_by", "strength")),
            layout=str(_opt(p, "layout", "layout", "spring")),
            label_top=int(_opt(p, "labelTop", "label_top", 30)),
            top_n=int(_opt(p, "topN", "top_n", 30)),
            plot_network=_flag(p, "plotNetwork", "plot_network", True),
            plot_centrality=_flag(p, "plotCentrality", "plot_centrality", True),
            seed=int(_opt(p, "seed", "seed", 42)),
        )

    # Dynamic Clean-Room Plugin Dispatch (for newly added jamovi features)
    from .plugins import discover_plugins
    plugins = discover_plugins()
    if t in plugins:
        return plugins[t]["runner"](df, p)

    raise ValueError(f"Unsupported analysis type: '{analysis_type}'")

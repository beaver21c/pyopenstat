"""
OpenStat Studio (openstat / pyjamovi v3.0)
Standalone MIT-Licensed Pure-Python Statistical Engine, Customizable Plot Studio, Multi-Format Table Exporter & Claude Code Runtime.
Analytical models, menu organization, and .omv format compatibility designed with reference to jamovi (https://www.jamovi.org).
"""

from typing import List, Optional, Union, Dict, Any, Tuple
import pandas as pd

from .omv.types import ColumnType, DataType, MeasureType, INT_MISSING
from .omv.dataset import JamoviDataset, JamoviColumn
from .omv.reader import read_omv
from .omv.writer import write_omv

from .analyses import (
    BaseAnalysis,
    ResultsTable,
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
    SEM,
    Mediation,
    Moderation,
    PanelAnalysis,
    MixedLinearModel,
    SurvivalAnalysis,
    ScatrPlot,
    GraphBuilder,
    CustomFigureResult,
    CHART_TYPES,
    CHART_TYPE_LABELS,
    compute_posthoc,
    compute_games_howell,
    compute_bonferroni,
    plot_descriptives_box,
    plot_descriptives_hist,
    plot_descriptives_bar,
    plot_mean_ci,
    plot_interaction_emm,
    plot_rm_profile,
    plot_qq,
    plot_scatr,
    plot_correlation_heatmap,
    plot_regression_diagnostics,
    plot_regression_residuals,
    plot_roc_curve,
    plot_bayes_posterior,
    plot_bayes_robustness,
    plot_scree,
    plot_sem_diagram,
    plot_moderation_slopes,
    plot_panel_trends,
    plot_kaplan_meier,
)
from .analyses.plot_style import PALETTES, THEMES, use_plot_style, normalize_plot_style
from .registry import ANALYSIS_CATALOG, dispatch_analysis, refresh_analysis_catalog, get_full_catalog
from .renderers.html_renderer import render_jamovi_html_report
from .exporters import export_analyses, SUPPORTED_TABLE_FORMATS
from .report_format import ReportFormat, resolve_report_format
from .analyses.model_compare import ModelComparison
from .syntax_engine import (
    generate_python_script,
    execute_python_syntax,
    save_workspace_sync,
    load_workspace_sync,
    run_script,
)
from .data_ops import (
    compute_variable,
    recode_variable,
    filter_dataset,
    merge_datasets,
    joinby_datasets,
    append_datasets,
    reshape_dataset,
    collapse_dataset,
    egen_variable,
    convert_variables,
    encode_variable,
    decode_variable,
    read_dataset,
    load_tabular_source,
)
from .diagnostics import (
    apply_runtime_compat_shims,
    check_package_freshness,
    run_api_compat_self_check,
    startup_self_check_guard,
    format_doctor_markdown,
)
from .upstream_sync import (
    check_jamovi_upstream,
    scaffold_jamovi_feature,
    rebuild_openstat_package,
)
from .agent import JamoviSession
from .text_import import load_texts, texts_to_dataframe, SUPPORTED_TEXT_EXTENSIONS

# Proactively apply known upstream Python library compatibility shims on import
apply_runtime_compat_shims()

Session = JamoviSession

__version__ = "3.0.0"


def doctor(check_pypi: bool = True) -> Dict[str, Any]:
    """Run full Python package freshness check and API/syntax compatibility self-diagnostics."""
    return startup_self_check_guard(force=True, check_pypi=check_pypi)


def run(analysis_type: str, data: pd.DataFrame, **params) -> BaseAnalysis:
    """Universal Python entry point for researchers & AI agents to run any statistical analysis."""
    return dispatch_analysis(analysis_type, data, params)


def descriptives(data: pd.DataFrame, vars: Union[str, List[str]], split_by: Optional[str] = None, **kwargs) -> Descriptives:
    return Descriptives(data, vars=vars, split_by=split_by, **kwargs)


def scatr(data: pd.DataFrame, x: str, y: Optional[str] = None, group: Optional[str] = None, **kwargs) -> ScatrPlot:
    return ScatrPlot(data, x=x, y=y, group=group, **kwargs)


def graph(
    data: pd.DataFrame,
    chart_type: str = "bar",
    x: Optional[str] = None,
    y: Optional[str] = None,
    vars: Optional[List[str]] = None,
    group: Optional[str] = None,
    **kwargs,
) -> GraphBuilder:
    """Create any of 15 standalone chart types (`bar`, `stacked_bar`, `line`, `area`, `pie`, `donut`, `histogram`, `density`, `box`, `violin`, `raincloud`, `scatter`, `pairplot`, `heatmap`, `errorbar`) as an Analysis Results card."""
    plot_style = kwargs.pop("plotStyle", kwargs.pop("plot_style", None))
    with use_plot_style(plot_style):
        res = GraphBuilder(data, chart_type=chart_type, x=x, y=y, vars=vars, group=group, **kwargs)
    if plot_style:
        res.options["plotStyle"] = plot_style
    return res


def add_figure(
    fig_or_src: Any,
    title: str = "Custom Visualization",
    caption: str = "",
    summary_df: Optional[pd.DataFrame] = None,
    dataset: Optional[pd.DataFrame] = None,
    plot_style: Optional[Dict[str, Any]] = None,
) -> CustomFigureResult:
    """Wrap any custom Matplotlib Figure (or PNG/SVG data URI) into a first-class Analysis Results card."""
    return CustomFigureResult(
        fig_or_src=fig_or_src,
        title=title,
        caption=caption,
        summary_df=summary_df,
        dataset=dataset,
        plot_style=plot_style,
    )


def ttestIS(data: pd.DataFrame, vars: Union[str, List[str]], group: str, hypothesis: str = "different", **kwargs) -> TTestIS:
    return TTestIS(data, vars=vars, group=group, hypothesis=hypothesis, **kwargs)


def ttestPS(data: pd.DataFrame, pairs: Optional[List[Tuple[str, str]]] = None, **kwargs) -> TTestPS:
    return TTestPS(data, pairs=pairs, **kwargs)


def ttestOneS(data: pd.DataFrame, vars: Union[str, List[str]], test_value: float = 0.0, **kwargs) -> TTestOneS:
    return TTestOneS(data, vars=vars, test_value=test_value, **kwargs)


def bayesTTestIS(data: pd.DataFrame, vars: Union[str, List[str]], group: str, **kwargs) -> BayesianTTestIS:
    return BayesianTTestIS(data, vars=vars, group=group, **kwargs)


def anovaOneW(data: pd.DataFrame, deps: Union[str, List[str]], group: str, **kwargs) -> AnovaOneW:
    return AnovaOneW(data, deps=deps, group=group, **kwargs)


def anova(data: pd.DataFrame, dep: str, factors: Union[str, List[str]], **kwargs) -> FactorialANOVA:
    return FactorialANOVA(data, dep=dep, factors=factors, **kwargs)


def ancova(data: pd.DataFrame, dep: str, factors: Union[str, List[str]], covs: Union[str, List[str]], **kwargs) -> ANCOVA:
    return ANCOVA(data, dep=dep, factors=factors, covs=covs, **kwargs)


def anovaRM(data: pd.DataFrame, rm_vars: List[str], between: Optional[str] = None, **kwargs) -> RepeatedMeasuresANOVA:
    return RepeatedMeasuresANOVA(data, rm_vars=rm_vars, between=between, **kwargs)


def kruskal(data: pd.DataFrame, deps: Union[str, List[str]], group: str, **kwargs) -> KruskalWallis:
    return KruskalWallis(data, deps=deps, group=group, **kwargs)


def bayesANOVA(data: pd.DataFrame, dep: str, factors: Union[str, List[str]], **kwargs) -> BayesianANOVA:
    return BayesianANOVA(data, dep=dep, factors=factors, **kwargs)


def corrMatrix(data: pd.DataFrame, vars: Union[str, List[str]], **kwargs) -> CorrMatrix:
    return CorrMatrix(data, vars=vars, **kwargs)


def partCorr(data: pd.DataFrame, vars: List[str], controls: Union[str, List[str]], **kwargs) -> PartialCorrelation:
    return PartialCorrelation(data, vars=vars, controls=controls, **kwargs)


def bayesCorr(data: pd.DataFrame, vars: List[str], **kwargs) -> BayesianCorrelation:
    return BayesianCorrelation(data, vars=vars, **kwargs)


def linReg(data: pd.DataFrame, dep: str, covs: Optional[Union[str, List[str]]] = None, **kwargs) -> LinReg:
    return LinReg(data, dep=dep, covs=covs, **kwargs)


def logRegBin(data: pd.DataFrame, dep: str, covs: Optional[Union[str, List[str]]] = None, **kwargs) -> LogisticBinomial:
    return LogisticBinomial(data, dep=dep, covs=covs, **kwargs)


def contTables(data: pd.DataFrame, rows: str, cols: str, **kwargs) -> ContTables:
    return ContTables(data, rows=rows, cols=cols, **kwargs)


def reliability(data: pd.DataFrame, vars: List[str], **kwargs) -> Reliability:
    return Reliability(data, vars=vars, **kwargs)


def pca(data: pd.DataFrame, vars: List[str], **kwargs) -> FactorAnalysis:
    return FactorAnalysis(data, vars=vars, mode="pca", **kwargs)


def efa(data: pd.DataFrame, vars: List[str], **kwargs) -> FactorAnalysis:
    return FactorAnalysis(data, vars=vars, mode="efa", **kwargs)


def sem(data: pd.DataFrame, model_syntax: str, **kwargs) -> SEM:
    return SEM(data, model_syntax=model_syntax, **kwargs)


def mediation(data: pd.DataFrame, dep: str, pred: str, med: str, **kwargs) -> Mediation:
    return Mediation(data, dep=dep, pred=pred, med=med, **kwargs)


def moderation(data: pd.DataFrame, dep: str, pred: str, mod: str, **kwargs) -> Moderation:
    return Moderation(data, dep=dep, pred=pred, mod=mod, **kwargs)


def panel(data: pd.DataFrame, entity_col: str, time_col: str, dep: str, covs: Union[str, List[str]], model_type: str = "fixed", **kwargs) -> PanelAnalysis:
    return PanelAnalysis(data, entity_col=entity_col, time_col=time_col, dep=dep, covs=covs, model_type=model_type, **kwargs)


def mixedModel(data: pd.DataFrame, dep: str, cluster: str, **kwargs) -> MixedLinearModel:
    return MixedLinearModel(data, dep=dep, cluster=cluster, **kwargs)


def survival(data: pd.DataFrame, time_col: str, event_col: str, group: Optional[str] = None, **kwargs) -> SurvivalAnalysis:
    return SurvivalAnalysis(data, time_col=time_col, event_col=event_col, group=group, **kwargs)


def compare_models(*results: BaseAnalysis, names: Optional[List[str]] = None, stats: Any = "auto", estimate: str = "b",
                   labels: Optional[Dict[str, str]] = None, terms: Optional[List[str]] = None, table: Optional[str] = None,
                   title: Optional[str] = None, lang: str = "ko") -> ModelComparison:
    """Side-by-side comparison table of estimated models (Stata esttab layout).

    results: coefficient-table results (linear/logistic/panel/multilevel/Cox/spatial regression, SEM …);
    a hierarchical linReg with blocks contributes one model per block. For a reproducible script use
    `Session.compare_models`, which records the model specifications.
    """
    from .analyses.model_compare import compare_results
    if len(results) == 1 and isinstance(results[0], (list, tuple)):
        results = tuple(results[0])
    return compare_results(list(results), names=names, stats=stats, estimate=estimate, labels=labels, terms=terms,
                           table=table, title=title, lang=lang)


__all__ = [
    "__version__",
    "compare_models",
    "ModelComparison",
    "ReportFormat",
    "resolve_report_format",
    "Session",
    "JamoviSession",
    "load_texts",
    "texts_to_dataframe",
    "SUPPORTED_TEXT_EXTENSIONS",
    "PALETTES",
    "THEMES",
    "use_plot_style",
    "normalize_plot_style",
    "export_analyses",
    "SUPPORTED_TABLE_FORMATS",
    "generate_python_script",
    "execute_python_syntax",
    "save_workspace_sync",
    "load_workspace_sync",
    "run_script",
    "compute_variable",
    "recode_variable",
    "filter_dataset",
    "merge_datasets",
    "joinby_datasets",
    "append_datasets",
    "reshape_dataset",
    "collapse_dataset",
    "egen_variable",
    "convert_variables",
    "encode_variable",
    "decode_variable",
    "read_dataset",
    "load_tabular_source",
    "doctor",
    "check_package_freshness",
    "run_api_compat_self_check",
    "startup_self_check_guard",
    "format_doctor_markdown",
    "check_jamovi_upstream",
    "scaffold_jamovi_feature",
    "rebuild_openstat_package",
    "ANALYSIS_CATALOG",
    "dispatch_analysis",
    "run",
    "read_omv",
    "write_omv",
    "JamoviDataset",
    "JamoviColumn",
    "ColumnType",
    "DataType",
    "MeasureType",
    "INT_MISSING",
    "descriptives",
    "scatr",
    "graph",
    "add_figure",
    "GraphBuilder",
    "CustomFigureResult",
    "CHART_TYPES",
    "CHART_TYPE_LABELS",
    "ttestIS",
    "ttestPS",
    "ttestOneS",
    "bayesTTestIS",
    "anovaOneW",
    "anova",
    "ancova",
    "anovaRM",
    "kruskal",
    "bayesANOVA",
    "corrMatrix",
    "partCorr",
    "bayesCorr",
    "linReg",
    "logRegBin",
    "contTables",
    "reliability",
    "pca",
    "efa",
    "sem",
    "mediation",
    "moderation",
    "panel",
    "mixedModel",
    "survival",
    "compute_posthoc",
    "compute_games_howell",
    "compute_bonferroni",
    "plot_descriptives_box",
    "plot_descriptives_hist",
    "plot_descriptives_bar",
    "plot_mean_ci",
    "plot_interaction_emm",
    "plot_rm_profile",
    "plot_qq",
    "plot_scatr",
    "plot_correlation_heatmap",
    "plot_regression_diagnostics",
    "plot_regression_residuals",
    "plot_roc_curve",
    "plot_bayes_posterior",
    "plot_bayes_robustness",
    "plot_scree",
    "plot_sem_diagram",
    "plot_moderation_slopes",
    "plot_panel_trends",
    "plot_kaplan_meier",
    "render_jamovi_html_report",
    "refresh_analysis_catalog",
    "get_full_catalog",
    "BaseAnalysis",
    "ResultsTable",
]

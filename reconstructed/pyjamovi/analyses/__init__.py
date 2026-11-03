"""
pyjamovi Complete Statistical Analyses & Visualization Package (v2.0)
"""

from .base import BaseAnalysis, ResultsTable
from .descriptives import Descriptives
from .ttest_is import TTestIS
from .ttest_other import TTestPS, TTestOneS
from .anova_onew import AnovaOneW
from .anova_advanced import FactorialANOVA, ANCOVA, RepeatedMeasuresANOVA, KruskalWallis
from .corrmatrix import CorrMatrix
from .linreg import LinReg
from .regression_advanced import LogisticBinomial, PartialCorrelation
from .conttables import ContTables
from .bayesian import (
    BayesianTTestIS,
    BayesianANOVA,
    BayesianCorrelation,
    jzs_bayes_factor_ttest,
    jzs_bayes_factor_correlation
)
from .factor_suite import Reliability, FactorAnalysis
from .sem_module import SEM
from .medmod import Mediation, Moderation
from .panel_module import PanelAnalysis
from .survival_module import MixedLinearModel, SurvivalAnalysis
from .scatr_module import ScatrPlot
from .graph_builder import (
    GraphBuilder,
    CustomFigureResult,
    CHART_TYPES,
    CHART_TYPE_LABELS,
)
from .posthoc import compute_posthoc, compute_games_howell, compute_bonferroni
from .plots import (
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
    plot_kaplan_meier
)

__all__ = [
    "BaseAnalysis",
    "ResultsTable",
    "Descriptives",
    "TTestIS",
    "TTestPS",
    "TTestOneS",
    "AnovaOneW",
    "FactorialANOVA",
    "ANCOVA",
    "RepeatedMeasuresANOVA",
    "KruskalWallis",
    "CorrMatrix",
    "LinReg",
    "LogisticBinomial",
    "PartialCorrelation",
    "ContTables",
    "BayesianTTestIS",
    "BayesianANOVA",
    "BayesianCorrelation",
    "Reliability",
    "FactorAnalysis",
    "SEM",
    "Mediation",
    "Moderation",
    "PanelAnalysis",
    "MixedLinearModel",
    "SurvivalAnalysis",
    "ScatrPlot",
    "GraphBuilder",
    "CustomFigureResult",
    "CHART_TYPES",
    "CHART_TYPE_LABELS",
    "compute_posthoc",
    "compute_games_howell",
    "compute_bonferroni",
    "jzs_bayes_factor_ttest",
    "jzs_bayes_factor_correlation",
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
    "plot_kaplan_meier"
]

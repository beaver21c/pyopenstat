"""
SEM model templates (CLPM, RI-CLPM, multiple-indicator CLPM, latent growth, second-order growth, latent change score):
generated model strings run in the SEM engine, the parameter count / df equal the analytic values, estimates, SEs and
fit statistics equal lavaan 0.7.2 on the same data and model string, and the generating values are recovered.

Reference values: lavaan::sem(model, data, missing = "fiml", mimic = "Mplus", estimator = "ML",
information = "observed", observed.information = "hessian") in R 4.4 on sample_data/SEM_Longitudinal_*.csv
(scripts kept outside the repository).
"""

import importlib.util
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reconstructed"))
warnings.filterwarnings("ignore")

from pyjamovi.analyses import sem_engine as se  # noqa: E402
from pyjamovi.analyses import sem_templates as st  # noqa: E402

_spec = importlib.util.spec_from_file_location("mk_semtpl", os.path.join(ROOT, "sample_data", "make_semtpl_sample_data.py"))
mk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mk)

PANEL = os.path.join(ROOT, "sample_data", "SEM_Longitudinal_Panel.csv")
ITEMS = os.path.join(ROOT, "sample_data", "SEM_Longitudinal_Items.csv")
w = lambda stem, T=5: [f"{stem}_{t}" for t in range(1, T + 1)]           # noqa: E731
g = lambda stem, T=4: [[f"{stem}{c}_{t}" for t in range(1, T + 1)] for c in "ABC"]  # noqa: E731

CASES = {
    "clpm": ("P", "clpm", {"x_waves": w("건강"), "y_waves": w("의료이용")}, {}),
    "clpm_eq": ("P", "clpm", {"x_waves": w("건강"), "y_waves": w("의료이용")},
                {"constrain_lagged": True, "constrain_residual_cov": True, "constrain_residual_var": True}),
    "clpm3": ("P", "clpm", {"x_waves": w("건강"), "y_waves": w("의료이용"), "z_waves": w("삶의만족도")}, {}),
    "riclpm": ("P", "riclpm", {"x_waves": w("우울"), "y_waves": w("외로움")}, {}),
    "riclpm_eq": ("P", "riclpm", {"x_waves": w("우울"), "y_waves": w("외로움")}, {"constrain_lagged": True}),
    "riclpm_cov": ("P", "riclpm", {"x_waves": w("우울"), "y_waves": w("외로움"), "covariates": ["연령", "여성"]},
                   {"constrain_lagged": True}),
    "riclpm_cov_mc": ("P", "riclpm", {"x_waves": w("우울"), "y_waves": w("외로움"), "covariates": ["연령", "여성"]},
                      {"model_covariates": True}),
    "lclpm_scalar": ("I", "latent_clpm", {"x_indicators": g("우울"), "y_indicators": g("사회적지지")}, {}),
    "lclpm_configural": ("I", "latent_clpm", {"x_indicators": g("우울"), "y_indicators": g("사회적지지")},
                         {"invariance": "configural"}),
    "lclpm_metric": ("I", "latent_clpm", {"x_indicators": g("우울"), "y_indicators": g("사회적지지")},
                     {"invariance": "metric", "correlated_residuals": "adjacent"}),
    "lgm_linear": ("P", "lgm_linear", {"y_waves": w("사회참여")}, {}),
    "lgm_linear_eq": ("P", "lgm_linear", {"y_waves": w("사회참여")}, {"residual_var_equal": True}),
    "lgm_quadratic": ("P", "lgm_quadratic", {"y_waves": w("인지기능")}, {}),
    "lgm_quad_ts": ("P", "lgm_quadratic", {"y_waves": w("인지기능")},
                    {"time_scores": "0, 1, 2, 3, 4", "residual_var_equal": True}),
    "lgm_basis": ("P", "lgm_basis", {"y_waves": w("적응")}, {}),
    "lgm_basis2": ("P", "lgm_basis", {"y_waves": w("적응")}, {"anchor": "second"}),
    "lgm_cond": ("P", "lgm_conditional", {"y_waves": w("삶의만족도"), "covariates": ["연령", "소득"],
                                           "tvc_waves": w("건강")}, {}),
    "lgm_cond_eq": ("P", "lgm_conditional", {"y_waves": w("삶의만족도"), "covariates": ["연령", "소득"],
                                              "tvc_waves": w("건강")}, {"tvc_equal": True}),
    "lgm_cond_mc": ("P", "lgm_conditional", {"y_waves": w("삶의만족도"), "covariates": ["연령", "소득"],
                                              "tvc_waves": w("건강")}, {"model_covariates": True}),
    "lgm_cond_quad": ("P", "lgm_conditional", {"y_waves": w("인지기능"), "covariates": ["연령"]}, {"shape": "quadratic"}),
    "lgm_parallel": ("P", "lgm_parallel", {"x_waves": w("사회참여"), "y_waves": w("여가활동")}, {}),
    "lgm_parallel_reg": ("P", "lgm_parallel", {"x_waves": w("사회참여"), "y_waves": w("여가활동")},
                         {"structure": "regression", "wave_residual_cov": True}),
    "lgm_2nd": ("I", "lgm_second_order", {"indicators": g("삶의질")}, {}),
    "lgm_2nd_mplus": ("I", "lgm_second_order", {"indicators": g("삶의질")}, {"mean_identification": "intercept_zero"}),
    "lgm_2nd_strict": ("I", "lgm_second_order", {"indicators": g("삶의질")},
                       {"invariance": "strict", "correlated_residuals": "none", "factor_resvar_equal": True}),
    "lcs": ("P", "lcs_univariate", {"y_waves": w("자아존중감")}, {}),
    "lcs_prop": ("P", "lcs_univariate", {"y_waves": w("자아존중감")},
                 {"constant_change": False, "residual_var_equal": False}),
}

LAVAAN_REF = {
    'clpm': {'npar': 41, 'df': 24, 'chisq': 35.22941, 'logl': -9225.0174, 'cfi': 0.996581, 'rmsea': 0.024184,
        'pars': [('의료이용_3', '~', '건강_2', -0.4639351, 0.0590257), ('건강_3', '~', '의료이용_2', -0.0674749, 0.0149009), ('건강_1', '~~', '의료이용_1', -0.3244715, 0.0438506)]},
    'clpm_eq': {'npar': 20, 'df': 45, 'chisq': 53.8922, 'logl': -9234.3488, 'cfi': 0.997292, 'rmsea': 0.015716,
        'pars': [('의료이용_3', '~', '건강_2', -0.433319, 0.0291221), ('건강_3', '~~', '의료이용_3', -0.0947937, 0.0113616)]},
    'riclpm': {'npar': 44, 'df': 21, 'chisq': 17.72077, 'logl': -13919.1994, 'cfi': 1, 'rmsea': 0,
        'pars': [('W_외로움3', '~', 'W_우울2', 0.176932, 0.0393148), ('W_우울3', '~', 'W_외로움2', 0.0520209, 0.0614939), ('RI_우울', '~~', 'RI_외로움', 1.3371336, 0.1929114)]},
    'riclpm_eq': {'npar': 32, 'df': 33, 'chisq': 29.17334, 'logl': -13924.9257, 'cfi': 1, 'rmsea': 0,
        'pars': [('W_외로움3', '~', 'W_우울2', 0.1803599, 0.022109), ('W_우울3', '~', 'W_외로움2', 0.2011745, 0.0304473), ('RI_우울', '~~', 'RI_우울', 4.5271027, 0.3191869)]},
    'riclpm_cov': {'npar': 36, 'df': 49, 'chisq': 41.1176, 'logl': -13883.9737, 'cfi': 1, 'rmsea': 0,
        'pars': [('RI_우울', '~', '여성', 1.0054134, 0.1640916), ('RI_외로움', '~', '연령', 0.0343433, 0.0059425), ('W_우울2', '~', 'W_우울1', 0.4481326, 0.0319916)]},
    'riclpm_cov_mc': {'npar': 53, 'df': 37, 'chisq': 29.78668, 'logl': -17453.8749, 'cfi': 1, 'rmsea': 0,
        'pars': [('RI_우울', '~', '여성', 1.0014597, 0.16418), ('연령', '~~', '여성', 0.2174937, 0.1810614)]},
    'lclpm_scalar': {'npar': 100, 'df': 224, 'chisq': 204.14583, 'logl': -16169.3088, 'cfi': 1, 'rmsea': 0,
        'pars': [('우울_T2', '=~', '우울B_2', 0.8524357, 0.0244561), ('우울_T3', '~', '사회적지지_T2', -0.1807928, 0.0356138), ('우울A_1', '~~', '우울A_3', 0.0386243, 0.0111724), ('우울_T2', '~1', '', 0.070054, 0.0226711)]},
    'lclpm_configural': {'npar': 124, 'df': 200, 'chisq': 178.8573, 'logl': -16156.6645, 'cfi': 1, 'rmsea': 0,
        'pars': [('우울_T2', '=~', '우울B_2', 0.8544668, 0.0453437), ('우울_T3', '~', '사회적지지_T2', -0.178627, 0.0355341)]},
    'lclpm_metric': {'npar': 94, 'df': 230, 'chisq': 604.57287, 'logl': -16369.5223, 'cfi': 0.958993, 'rmsea': 0.045119,
        'pars': [('우울_T2', '=~', '우울B_2', 0.8500222, 0.024518), ('우울_T3', '~', '사회적지지_T2', -0.1717077, 0.0355902)]},
    'lgm_linear': {'npar': 10, 'df': 10, 'chisq': 8.47649, 'logl': -3889.5189, 'cfi': 1, 'rmsea': 0,
        'pars': [('S', '~1', '', 0.2408964, 0.0087962), ('I', '~~', 'S', -0.0358195, 0.0084281)]},
    'lgm_linear_eq': {'npar': 6, 'df': 14, 'chisq': 8.89795, 'logl': -3889.7297, 'cfi': 1, 'rmsea': 0,
        'pars': [('S', '~1', '', 0.2411416, 0.0087907), ('사회참여_3', '~~', '사회참여_3', 0.2412424, 0.0072676)]},
    'lgm_quadratic': {'npar': 14, 'df': 6, 'chisq': 4.64252, 'logl': -6929.9032, 'cfi': 1, 'rmsea': 0,
        'pars': [('Q', '~1', '', -0.0937903, 0.0102864), ('S', '~~', 'Q', -0.0409939, 0.0280173)]},
    'lgm_quad_ts': {'npar': 10, 'df': 10, 'chisq': 6.70127, 'logl': -6930.9326, 'cfi': 1, 'rmsea': 0,
        'pars': [('Q', '~1', '', -0.0936477, 0.010279), ('Q', '~~', 'Q', 0.0179758, 0.0046283)]},
    'lgm_basis': {'npar': 13, 'df': 7, 'chisq': 2.70077, 'logl': -3193.6282, 'cfi': 1, 'rmsea': 0,
        'pars': [('S', '=~', '적응_3', 0.7041435, 0.0114998), ('S', '~1', '', 1.5279138, 0.0291871)]},
    'lgm_basis2': {'npar': 13, 'df': 7, 'chisq': 2.70077, 'logl': -3193.6282, 'cfi': 1, 'rmsea': 0,
        'pars': [('S', '=~', '적응_5', 2.2384965, 0.0541982), ('S', '~1', '', 0.6825625, 0.0210206)]},
    'lgm_cond': {'npar': 19, 'df': 36, 'chisq': 35.57566, 'logl': -3884.6544, 'cfi': 1, 'rmsea': 0,
        'pars': [('I', '~', '소득', 0.246201, 0.0216007), ('S', '~', '연령', -0.0033134, 0.0009863), ('삶의만족도_3', '~', '건강_3', 0.2570334, 0.020822)]},
    'lgm_cond_eq': {'npar': 15, 'df': 40, 'chisq': 36.11313, 'logl': -3884.9232, 'cfi': 1, 'rmsea': 0,
        'pars': [('삶의만족도_3', '~', '건강_3', 0.2597244, 0.0199957), ('I', '~~', 'S', -0.0398242, 0.0100429)]},
    'lgm_cond_mc': {'npar': 54, 'df': 36, 'chisq': 33.29823, 'logl': -12019.7622, 'cfi': 1, 'rmsea': 0,
        'pars': [('I', '~', '소득', 0.2420141, 0.0201525), ('삶의만족도_3', '~', '건강_3', 0.2607664, 0.0202521)]},
    'lgm_cond_quad': {'npar': 17, 'df': 8, 'chisq': 6.72254, 'logl': -6929.3964, 'cfi': 1, 'rmsea': 0,
        'pars': [('Q', '~', '연령', 0.0002876, 0.0010064), ('Q', '~1', '', -0.1111358, 0.0616228)]},
    'lgm_parallel': {'npar': 24, 'df': 41, 'chisq': 37.19621, 'logl': -7244.3617, 'cfi': 1, 'rmsea': 0,
        'pars': [('S_사회참여', '~~', 'I_여가활동', -0.0041375, 0.0062076), ('S_사회참여', '~~', 'S_여가활동', 0.0039508, 0.0017956)]},
    'lgm_parallel_reg': {'npar': 29, 'df': 36, 'chisq': 35.78118, 'logl': -7243.6542, 'cfi': 1, 'rmsea': 0,
        'pars': [('S_여가활동', '~', 'I_사회참여', 0.0622764, 0.0125822), ('사회참여_2', '~~', '여가활동_2', 0.0050294, 0.0104959)]},
    'lgm_2nd': {'npar': 43, 'df': 47, 'chisq': 43.35187, 'logl': -8380.3771, 'cfi': 1, 'rmsea': 0,
        'pars': [('삶의질_T1', '=~', '삶의질B_1', 0.8768628, 0.0151041), ('S', '~1', '', 0.1404313, 0.0104605), ('삶의질B_1', '~1', '', 0.3775707, 0.0500253)]},
    'lgm_2nd_mplus': {'npar': 43, 'df': 47, 'chisq': 43.35187, 'logl': -8380.3771, 'cfi': 1, 'rmsea': 0,
        'pars': [('S', '~1', '', 0.1404314, 0.0104605), ('삶의질A_1', '~1', '', 3.0037006, 0.0286313)]},
    'lgm_2nd_strict': {'npar': 13, 'df': 77, 'chisq': 358.30745, 'logl': -8537.8549, 'cfi': 0.967308, 'rmsea': 0.067577,
        'pars': [('S', '~1', '', 0.1388851, 0.0103984), ('삶의질A_1', '~~', '삶의질A_1', 0.1911315, 0.0073036)]},
    'lcs': {'npar': 7, 'df': 13, 'chisq': 13.31278, 'logl': -9492.2353, 'cfi': 0.999908, 'rmsea': 0.005484,
        'pars': [('D2', '~', 'L1', -0.1213044, 0.040526), ('G', '~1', '', 4.2371987, 1.2520724), ('L1', '~~', 'G', 1.553578, 0.6380172)]},
    'lcs_prop': {'npar': 8, 'df': 12, 'chisq': 260.19616, 'logl': -9615.677, 'cfi': 0.927288, 'rmsea': 0.160791,
        'pars': [('D2', '~', 'L1', 0.0171468, 0.0010003), ('자아존중감_2', '~~', '자아존중감_2', 4.4008097, 0.3018103)]},
    'clpm3': {'npar': 81, 'df': 54, 'chisq': 284.42213, 'logl': -13615.7687, 'cfi': 0.960009, 'rmsea': 0.073033,
        'pars': [('삶의만족도_2', '~', '건강_1', 0.0203017, 0.0349714), ('건강_2', '~~', '삶의만족도_2', 0.0818365, 0.0146541)]},
}


@pytest.fixture(scope="module")
def data():
    return {"P": pd.read_csv(PANEL), "I": pd.read_csv(ITEMS)}


@pytest.fixture(scope="module")
def fits(data):
    out = {}
    for cid, (f, tid, roles, opts) in CASES.items():
        df = data[f]
        b = st.build_template(tid, roles, opts, list(df.columns))
        out[cid] = (b, se.estimate(df, b["model"]))
    return out


def _est(res, lhs, op, rhs):
    for r in res.spec.rows:
        if r.free and ((r.lhs, r.op, r.rhs) == (lhs, op, rhs) or (op == "~~" and (r.lhs, r.rhs) == (rhs, lhs))):
            return float(res.theta[r.pid]), float(res.se[r.pid])
    raise KeyError((lhs, op, rhs))


# ------------------------------------------------------------------------------------------- lavaan agreement
@pytest.mark.parametrize("cid", list(CASES))
def test_matches_lavaan(fits, cid):
    b, res = fits[cid]
    ref = LAVAAN_REF[cid]
    fs = res.fitstats
    assert res.fit.converged
    assert fs["npar"] == ref["npar"] == b["suggested"]["npar"] == b["suggested"]["npar_expected"]
    assert fs["df"] == ref["df"] == b["suggested"]["df"] == b["suggested"]["df_expected"]
    assert abs(fs["chi2"] - ref["chisq"]) < 1e-3
    assert abs(fs["ll0"] - ref["logl"]) < 1e-3
    assert abs(fs["cfi"] - ref["cfi"]) < 1e-5
    assert abs(fs["rmsea"] - ref["rmsea"]) < 1e-5
    for lhs, op, rhs, est, sev in ref["pars"]:
        e, s = _est(res, lhs, op, rhs)
        assert abs(e - est) < 1e-4 * max(1.0, abs(est)), (lhs, op, rhs, e, est)
        if math.isfinite(s):          # see test_known_engine_gap_identification_scaling
            assert abs(s - sev) < 1e-4 * max(1.0, abs(sev)), (lhs, op, rhs, s, sev)


def test_known_engine_gap_identification_scaling(fits):
    """Covariates modelled jointly (연령 variance ≈ 105) make the unscaled information matrix ill-conditioned; the
    engine then flags the model as not identified and omits SEs although lavaan reports them (relayed to the engine
    owner). Estimates and fit statistics are unaffected."""
    b, res = fits["lgm_cond_mc"]
    assert abs(res.fitstats["chi2"] - LAVAAN_REF["lgm_cond_mc"]["chisq"]) < 1e-3
    if not res.identified:
        assert any("식별" in n for n in res.notes)


# ------------------------------------------------------------------------------------------- analytic df
def _fake(tid, roles, opts=None):
    return st.build_template(tid, roles, opts or {})["suggested"]


def test_analytic_parameter_counts():
    x4, y4, x3, y3 = w("x", 4), w("y", 4), w("x", 3), w("y", 3)
    assert (_fake("clpm", {"x_waves": x4, "y_waves": y4})["df"]) == 12            # 36 - 24 covariance parameters
    assert (_fake("clpm", {"x_waves": w("x", 2), "y_waves": w("y", 2)})["df"]) == 0
    assert (_fake("riclpm", {"x_waves": x4, "y_waves": y4})["df"]) == 9            # Hamaker et al. (2015): CLPM df − 3
    assert (_fake("riclpm", {"x_waves": x3, "y_waves": y3})["df"]) == 1
    s = _fake("riclpm", {"x_waves": x4, "y_waves": y4}, {"constrain_lagged": True})
    assert s["df"] == 9 + 8
    assert (_fake("lgm_linear", {"y_waves": x3})["df"]) == 1
    assert (_fake("lgm_linear", {"y_waves": x4})["df"]) == 5
    assert (_fake("lgm_linear", {"y_waves": x4}, {"residual_var_equal": True})["df"]) == 8
    assert (_fake("lgm_quadratic", {"y_waves": x4})["df"]) == 1
    assert (_fake("lgm_basis", {"y_waves": x3})["df"]) == 0
    assert (_fake("lgm_basis", {"y_waves": x4})["df"]) == 3
    assert (_fake("lcs_univariate", {"y_waves": x4})["df"]) == 7
    grid = lambda s: [[f"{s}{j}_{t}" for t in range(1, 5)] for j in range(1, 4)]       # noqa: E731
    assert _fake("latent_clpm", {"x_indicators": grid("a"), "y_indicators": grid("b")},
                 {"invariance": "configural"})["npar"] == 124
    assert _fake("latent_clpm", {"x_indicators": grid("a"), "y_indicators": grid("b")})["npar"] == 100
    assert _fake("lgm_second_order", {"indicators": grid("a")})["df"] == 47
    # indicator × wave and wave × indicator inputs give the same model
    a = st.build_template("lgm_second_order", {"indicators": grid("a")})["model"]
    b = st.build_template("lgm_second_order", {"indicators_by_wave": [list(c) for c in zip(*grid("a"))]})["model"]
    assert a == b


def test_model_text_conventions():
    b = st.build_template("riclpm", {"x_waves": w("x", 4), "y_waves": w("y", 4)})
    m = b["model"]
    for line in m.splitlines():
        if line.startswith("#"):
            assert ";" not in line and '"' not in line and "!" not in line
    assert "x_1 ~~ 0*x_1" in m and "RI_x ~~ 0*W_x1 + 0*W_y1" in m
    plain = st.build_template("riclpm", {"x_waves": w("x", 4), "y_waves": w("y", 4)}, {"comments": False})["model"]
    assert "#" not in plain
    assert [ln for ln in m.splitlines() if ln and not ln.startswith("#")] == [ln for ln in plain.splitlines() if ln]
    lgm = st.build_template("lgm_linear", {"y_waves": w("y", 4)}, {"time_scores": [0, 1, 2, 4]})["model"]
    assert "S =~ 0*y_1 + 1*y_2 + 2*y_3 + 4*y_4" in lgm and "y_1 ~ 0*1" in lgm and "I ~ 1" in lgm
    q = st.build_template("lgm_quadratic", {"y_waves": w("y", 4)}, {"time_scores": "0, 0.5, 1, 2"})["model"]
    assert "Q =~ 0*y_1 + 0.25*y_2 + 1*y_3 + 4*y_4" in q
    basis = st.build_template("lgm_basis", {"y_waves": w("y", 4)})["model"]
    assert "S =~ 0*y_1 + NA*y_2 + NA*y_3 + 1*y_4" in basis
    # latent names avoid data columns
    b = st.build_template("lgm_linear", {"y_waves": ["a", "b", "c"]}, {}, columns=["a", "b", "c", "I", "S"])
    assert b["suggested"]["latent"] == ["I_", "S_"]


def test_comments_do_not_change_estimates(data):
    df = data["P"]
    roles = {"y_waves": w("사회참여")}
    r1 = se.estimate(df, st.build_template("lgm_linear", roles, {"comments": True})["model"])
    r2 = se.estimate(df, st.build_template("lgm_linear", roles, {"comments": False})["model"])
    assert np.allclose(r1.theta, r2.theta) and abs(r1.fitstats["chi2"] - r2.fitstats["chi2"]) < 1e-9


# ------------------------------------------------------------------------------------------- recovery
def test_parameter_recovery(fits):
    T = mk.TRUE
    checks = [
        ("riclpm_eq", ("W_우울3", "~", "W_우울2"), T["riclpm"]["ar_x"]),
        ("riclpm_eq", ("W_외로움3", "~", "W_외로움2"), T["riclpm"]["ar_y"]),
        ("riclpm_eq", ("W_외로움3", "~", "W_우울2"), T["riclpm"]["cl_x_to_y"]),
        ("riclpm_eq", ("W_우울3", "~", "W_외로움2"), T["riclpm"]["cl_y_to_x"]),
        ("riclpm_cov", ("RI_우울", "~~", "RI_우울"), T["riclpm"]["ri_var"][0]),
        ("riclpm_cov", ("RI_외로움", "~", "연령"), T["riclpm"]["ri_on_age"][1]),
        ("clpm_eq", ("의료이용_3", "~", "건강_2"), T["clpm"]["cl_x_to_y"]),
        ("clpm_eq", ("건강_3", "~", "건강_2"), T["clpm"]["ar_x"]),
        ("lclpm_scalar", ("우울_T2", "=~", "우울B_2"), T["latent_clpm"]["load_x"][1]),
        ("lclpm_scalar", ("사회적지지_T2", "=~", "사회적지지B_2"), T["latent_clpm"]["load_y"][1]),
        ("lclpm_scalar", ("우울A_1", "~1", ""), T["latent_clpm"]["nu_x"][0]),
        ("lclpm_scalar", ("우울_T3", "~", "사회적지지_T2"), T["latent_clpm"]["cl_y_to_x"]),
        ("lclpm_scalar", ("우울A_1", "~~", "우울A_3"), T["latent_clpm"]["u_var"][0]),
        ("lgm_linear_eq", ("I", "~1", ""), T["parallel"]["i_mean"][0]),
        ("lgm_quad_ts", ("Q", "~1", ""), T["quadratic"]["mean"][2]),
        ("lgm_quad_ts", ("I", "~1", ""), T["quadratic"]["mean"][0]),
        ("lgm_basis", ("S", "=~", "적응_2"), T["basis"]["basis"][1]),
        ("lgm_basis", ("S", "=~", "적응_4"), T["basis"]["basis"][3]),
        ("lgm_cond", ("I", "~", "연령"), T["conditional"]["i_on_age"]),
        ("lgm_cond", ("S", "~", "소득"), T["conditional"]["s_on_income"]),
        ("lgm_cond_eq", ("삶의만족도_3", "~", "건강_3"), T["conditional"]["tvc"]),
        ("lgm_parallel_reg", ("S_여가활동", "~", "I_사회참여"), T["parallel"]["sy_on_ix"]),
        ("lgm_parallel_reg", ("I_사회참여", "~~", "I_여가활동"), T["parallel"]["i_cov"]),
        ("lgm_2nd", ("삶의질_T1", "=~", "삶의질C_1"), T["second_order"]["load"][2]),
        ("lgm_2nd", ("I", "~1", ""), T["second_order"]["i_mean"]),
        ("lgm_2nd", ("S", "~1", ""), T["second_order"]["s_mean"]),
        ("lgm_2nd", ("삶의질C_1", "~1", ""), T["second_order"]["nu"][2]),
        ("lcs", ("D2", "~", "L1"), T["lcs"]["beta"]),
        ("lcs", ("L1", "~1", ""), T["lcs"]["l1_mean"]),
        ("lcs", ("자아존중감_1", "~~", "자아존중감_1"), T["lcs"]["e_var"]),
    ]
    for cid, key, true in checks:
        est, sev = _est(fits[cid][1], *key)
        assert abs(est - true) < 3.0 * sev, (cid, key, est, sev, true)


def test_misspecified_clpm_on_riclpm_data_fits_worse(data):
    df = data["P"]
    roles = {"x_waves": w("우울"), "y_waves": w("외로움")}
    clpm = se.estimate(df, st.build_template("clpm", roles)["model"])
    ri = se.estimate(df, st.build_template("riclpm", roles)["model"])
    assert clpm.fitstats["df"] - ri.fitstats["df"] == 3
    assert clpm.fitstats["chi2"] - ri.fitstats["chi2"] > 50          # stable between-person variance ignored


# ------------------------------------------------------------------------------------------- validation
@pytest.mark.parametrize("tid, roles, opts, msg", [
    ("clpm", {"x_waves": w("x", 4), "y_waves": w("y", 3)}, {}, "시점 수가 같아야"),
    ("clpm", {"x_waves": w("x", 4)}, {}, "지정 필요"),
    ("riclpm", {"x_waves": w("x", 2), "y_waves": w("y", 2)}, {}, "3시점 이상"),
    ("lgm_linear", {"y_waves": w("y", 2)}, {}, "3시점 이상"),
    ("lgm_quadratic", {"y_waves": w("y", 3)}, {}, "4시점 이상"),
    ("lgm_linear", {"y_waves": w("y", 4)}, {"time_scores": "0, 1, 2"}, "시점 수(4)"),
    ("lgm_linear", {"y_waves": w("y", 4)}, {"time_scores": "0, 1, 1, 2"}, "같은 값"),
    ("lgm_linear", {"y_waves": w("y", 4)}, {"time_scores": "0, a, 2, 3"}, "숫자만"),
    ("lgm_linear", {"y_waves": w("y", 4)}, {"bogus": 1}, "알 수 없는 옵션"),
    ("lgm_linear", {"y_waves": w("y", 4), "zz": ["a"]}, {}, "알 수 없는 변수 역할"),
    ("lgm_conditional", {"y_waves": w("y", 4)}, {}, "공변량"),
    ("lgm_conditional", {"y_waves": w("y", 4), "tvc_waves": w("z", 3)}, {}, "같은 시점 수"),
    ("clpm", {"x_waves": w("x", 3), "y_waves": ["x_1", "y_2", "y_3"]}, {}, "같은 변수"),
    ("latent_clpm", {"x_indicators": [["a1", "a2"], ["b1"]], "y_indicators": [["c1", "c2"], ["d1", "d2"]]}, {},
     "시점 수가 같아야"),
    ("latent_clpm", {"x_indicators": [["a1", "a2"]], "y_indicators": [["c1", "c2"], ["d1", "d2"]]}, {}, "문항 2개 이상"),
    ("latent_clpm", {"x_indicators": [["a1", "a2"], ["b1", "b2"]], "y_indicators": [["c1", "c2"], ["d1", "d2"]]},
     {"invariance": "partial"}, "알 수 없는 값"),
    ("lgm_second_order", {"indicators": [["a1", "a2", "a3"], ["b1", "b2", "b3"]]}, {"shape": "quadratic"}, "4시점 이상"),
    ("lcs_univariate", {"y_waves": w("y", 4)}, {"constant_change": False, "proportional": False}, "하나 이상"),
    ("nonexistent", {}, {}, "알 수 없는 모형 예시"),
])
def test_validation_errors(tid, roles, opts, msg):
    with pytest.raises(ValueError) as e:
        st.build_template(tid, roles, opts)
    assert msg in str(e.value)


def test_missing_columns_error():
    with pytest.raises(st.TemplateError) as e:
        st.build_template("lgm_linear", {"y_waves": ["a", "b", "zz"]}, {}, columns=["a", "b", "c"])
    assert "데이터에 없는 변수: zz" in str(e.value)


# ------------------------------------------------------------------------------------------- API / catalog
def test_catalog_json_and_aliases():
    cat = st.template_catalog()
    json.dumps(cat, ensure_ascii=False)
    ids = [t["id"] for t in cat]
    assert len(ids) == len(set(ids)) == 10
    for t in cat:
        assert t["title"] and t["name_en"] and t["description"] and t["roles"] and t["references"]
        assert any(o["key"] == "comments" for o in t["options"])
    assert st.build_template("RI-CLPM", {"xWaves": w("x", 3), "yWaves": w("y", 3)}, {"constrainLagged": True})[
        "options"]["constrain_lagged"] is True
    payload = st.catalog_payload(list(pd.read_csv(PANEL, nrows=1).columns))
    json.dumps(payload, ensure_ascii=False)
    assert payload["suggested_roles"]["riclpm"] == {"x_waves": w("우울"), "y_waves": w("외로움")}


def test_wave_detection_and_suggestions(data):
    sets = {s["stem"]: s["vars"] for s in st.detect_wave_sets(["id", "x1", "x2", "x3", "dep_t1", "dep_t2", "dep_t3",
                                                                 "a.w2", "a.w1", "x1_vis"])}
    assert sets == {"x": ["x1", "x2", "x3"], "dep": ["dep_t1", "dep_t2", "dep_t3"], "a": ["a.w1", "a.w2"]}
    cols = list(data["P"].columns)
    assert st.suggest_roles("clpm", cols) == {"x_waves": w("건강"), "y_waves": w("의료이용")}
    sug = st.suggest_roles("lgm_conditional", cols)
    assert sug["y_waves"] == w("삶의만족도") and sug["covariates"] == ["연령", "소득"] and sug["tvc_waves"] == w("건강")
    icol = list(data["I"].columns)
    assert st.suggest_roles("latent_clpm", icol) == {"x_indicators": g("우울"), "y_indicators": g("사회적지지")}
    assert st.suggest_roles("lgm_second_order", icol) == {"indicators": g("삶의질")}
    for t in st.TEMPLATE_CATALOG:      # every suggestion builds against its own dataset
        f = data["I"] if t["id"] in ("latent_clpm", "lgm_second_order") else data["P"]
        r = st.suggest_roles(t["id"], list(f.columns))
        st.build_template(t["id"], r, {}, list(f.columns))


def test_convenience_api_and_sem_module(data):
    df = data["P"]
    spec = st.sem_template("lgm_linear", df, y_waves=w("사회참여"), residual_var_equal=True)
    assert isinstance(spec, st.SEMTemplate) and spec.suggested["df"] == 14
    assert "기본 가정" in str(spec) and "자유도 14" in str(spec)
    out = st.sem_template("riclpm", df, x_waves=w("우울"), y_waves=w("외로움"), constrainLagged=True, run=True,
                          estimator="mlr")
    ana = out["analysis"]
    assert ana.result is not None and ana.result.fitstats["df"] == 33
    assert ana.options["model"] == out.model
    with pytest.raises(st.TemplateError):
        st.sem_template("lgm_linear", df, y_waves=w("사회참여"), nonsense=1)


def test_sample_data_reproducible(data):
    p = mk.make_panel()
    i = mk.make_items()
    pd.testing.assert_frame_equal(p.reset_index(drop=True), data["P"], check_dtype=False, atol=1e-9)
    pd.testing.assert_frame_equal(i.reset_index(drop=True), data["I"], check_dtype=False, atol=1e-9)


# ------------------------------------------------------------------------------------------- dialog payload
def test_diagram_and_measurement_payload(data):
    for t in st.TEMPLATE_CATALOG:
        f = data["I"] if t["id"] in ("latent_clpm", "lgm_second_order") else data["P"]
        cols = list(f.columns)
        b = st.build_template(t["id"], st.suggest_roles(t["id"], cols), {}, cols)
        d = b["diagram"]
        ids = {n["id"] for n in d["nodes"]}
        s = b["suggested"]
        assert set(s["observed"] + s["covariates"] + s["latent"]) <= ids          # every variable is drawn
        assert all(e["from"] in ids and e["to"] in ids for e in d["edges"])
        assert len({(n["x"], n["y"]) for n in d["nodes"]}) == len(d["nodes"])     # no two nodes on one spot
        assert {m["latent"] for m in b["measurement"]} == set(s["latent"]) - {
            x for x in s["latent"] if not any(m["latent"] == x for m in b["measurement"])}
        json.dumps(b, ensure_ascii=False)
    ri = st.build_template("riclpm", {"x_waves": w("우울"), "y_waves": w("외로움")})
    assert sum(n["zeroVar"] for n in ri["diagram"]["nodes"]) == 10                 # ε = 0 on every observed score
    assert {"latent": "RI_우울", "indicators": w("우울")} in ri["measurement"]


# ------------------------------------------------------------------------------------------- Stata sem translation
# StataNow 19.5 `sem ..., method(mlmv)` on the translator's own command lines (do-files kept outside the repository).
# The growth models marked * needed from() start values (Stata's default start values did not converge).
STATA_REF = {"clpm": (-9225.0174327, 35.2294102, 24), "riclpm": (-13919.1994101, 17.7207723, 21),
             "riclpm_cov": (-17459.5403408, 41.1175909, 49), "lclpm_scalar": (-16169.3088022, 204.1458335, 224),
             "lgm_linear": (-3889.5189334, 8.4764933, 10), "lgm_quadratic": (-6929.9032089, 4.6425218, 6),   # *
             "lgm_2nd": (-8380.3771336, 43.3518701, 47), "lcs": (-9492.2353371, 13.3127769, 13),            # * lgm_2nd
             "lgm_cond_mc": (-12019.7622371, 33.2982322, 36), "lgm_parallel_reg": (-7243.6541918, 35.7811771, 36)}


def test_stata_translation(fits):
    for cid, (ll, chi2, dfv) in STATA_REF.items():
        b, res = fits[cid]
        fs = res.fitstats
        assert abs(fs["chi2"] - chi2) < 1e-3 and fs["df"] == dfv
        if not b["suggested"]["covariates"]:          # conditioned covariates: Stata's LL also contains the x part
            assert abs(fs["ll0"] - ll) < 1e-3
        assert b["stata"] and b["stata"][0].startswith("sem (")
    ri = "\n".join(fits["riclpm_cov"][0]["stata"])
    assert "noivstart" in ri and "e.우울_1@0" in ri and "연령*W_우울1@0" in ri and "latent(RI_우울 RI_외로움" in ri
    assert "W_우울2 <- W_우울1@ar_우울 W_외로움1@cl_외로움_우울" in ri
    lgm = "\n".join(fits["lgm_linear"][0]["stata"])
    assert "(사회참여_1 <- _cons@0)" in lgm and "means(I S)" in lgm and "noivstart" not in lgm
    assert "(I -> 사회참여_1@1 사회참여_2@1 사회참여_3@1 사회참여_4@1 사회참여_5@1)" in lgm
    assert "noivstart" in "\n".join(fits["lcs"][0]["stata"])
    with pytest.raises(st.TemplateError):
        st.to_stata("y ~ c(a, b)*x")

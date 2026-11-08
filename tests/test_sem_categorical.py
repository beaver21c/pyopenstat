"""
SEM with categorical (binary / ordinal) variables — WLSMV / WLSM / WLS / ULSMV / ULS (delta and theta
parameterizations, covariates conditioned on) and MLM for continuous data.

Reference values: lavaan 0.7.2 (R 4.4), `sem(..., ordered = ..., estimator = ..., mimic = "Mplus")` on
sample_data/SEM_Categorical_Care.csv (and SEM_Latent3Factor.csv for MLM), hard-coded below so the tests run without R.
Each tuple is (estimate, SE, std.all). lavaan's `conditional.x = TRUE` default is used for models with covariates.
"""
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "reconstructed"))
warnings.filterwarnings("ignore")

from pyjamovi.analyses import sem_engine as se  # noqa: E402
from pyjamovi.analyses import sem_wls as sw  # noqa: E402
from pyjamovi.analyses.sem_module import SEM  # noqa: E402

CARE = os.path.join(ROOT_DIR, "sample_data", "SEM_Categorical_Care.csv")
LAT3 = os.path.join(ROOT_DIR, "sample_data", "SEM_Latent3Factor.csv")
S = ["만족_친절", "만족_시간", "만족_전문성", "만족_소통"]
T = ["신뢰_정보", "신뢰_약속", "신뢰_안전"]
U = ["이용_방문요양", "이용_주간보호", "이용_방문간호", "이용_식사배달", "이용_이동지원"]
M_A = "S =~ " + " + ".join(S) + "\nT =~ " + " + ".join(T)
M_B = "N =~ " + " + ".join(U)
M_C = "H =~ 주관적건강 + 우울점수 + 일상활동점수 + 삶의만족도"
M_D = M_A + "\n재이용의향 ~ S + T"
M_E = "S =~ " + " + ".join(S) + "\nS ~ 연령 + 여성\n재이용의향 ~ S + 독거"
M_F = "S =~ " + " + ".join(S) + "\n재이용의향 ~ a*S\n추천의향 ~ b*재이용의향 + c*S\nind := a*b\ntot := c + a*b"
M_P = "S =~ " + " + ".join(S) + "\n삶의만족도 ~ S + 연령 + 여성\nS ~ 연령"
M_CFA = "Visual =~ x1_vis + x2_vis + x3_vis\nTextual =~ x4_txt + x5_txt + x6_txt\nSpeed =~ x7_spd + x8_spd + x9_spd"

# id: (model, categorical, estimator, parameterization, conditional_x, data)
CASES = {
    "a": (M_A, S + T, "wlsmv", "delta", True, CARE),          # ordinal CFA (5 / 4 categories)
    "b": (M_B, U, "wlsmv", "delta", True, CARE),              # binary-indicator CFA
    "c": (M_C, ["주관적건강"], "wlsmv", "delta", True, CARE),  # ordinal + continuous indicators (polyserial)
    "d": (M_D, S + T + ["재이용의향"], "wlsmv", "delta", True, CARE),   # binary outcome on latent predictors
    "e": (M_E, S + ["재이용의향"], "wlsmv", "delta", True, CARE),      # covariates (conditional on x)
    "e_nox": (M_E, S + ["재이용의향"], "wlsmv", "delta", False, CARE),  # lavaan conditional.x = FALSE, fixed.x = FALSE
    "f_theta": (M_F, S + ["재이용의향", "추천의향"], "wlsmv", "theta", True, CARE),  # categorical mediator
    "a_wls": (M_A, S + T, "wls", "delta", True, CARE),
    "a_wlsm": (M_A, S + T, "wlsm", "delta", True, CARE),
    "a_ulsmv": (M_A, S + T, "ulsmv", "delta", True, CARE),
    "a_uls": (M_A, S + T, "uls", "delta", True, CARE),
    "a_theta": (M_A, S + T, "wlsmv", "theta", True, CARE),
    "e_theta": (M_E, S + ["재이용의향"], "wlsmv", "theta", True, CARE),
    "e_wls": (M_E, S + ["재이용의향"], "wls", "delta", True, CARE),
    "c_wls": (M_C, ["주관적건강"], "wls", "delta", True, CARE),
    "mlm_a": (M_A, None, "mlm", "delta", True, CARE),         # Likert items treated as continuous
    "mlm_p": (M_P, None, "mlm", "delta", True, CARE),         # MLM with conditioned covariates
    "mlm_cfa": (M_CFA, None, "mlm", "delta", True, LAT3),
}
FIT_MAP = {   # lavaan fitMeasures name -> our fitstats key
    "chisq": "chi2", "chisq.scaled": "chi2_test", "df": "df", "chisq.scaling.factor": "scaling",
    "cfi.scaled": "cfi", "tli.scaled": "tli", "rmsea.scaled": "rmsea", "rmsea.ci.upper.scaled": "rmsea_hi",
    "srmr": "srmr", "wrmr": "wrmr", "baseline.chisq.scaled": "chi2_base", "cfi": "cfi", "tli": "tli", "rmsea": "rmsea",
}

REF = {
    "a": {"N": 766, "par": {
        "S =~ 만족_시간": (0.916946, 0.0558374, 0.65811),
        "T =~ 신뢰_안전": (0.8666935, 0.0554968, 0.6639686),
        "S ~~ T": (0.3490572, 0.0285557, 0.6348336),
        "T ~~ T": (0.5868999, 0.0447933, 1),
        "만족_친절 | t1": (-1.1176709, 0.0572985, -1.1176709),
        "신뢰_안전 | t3": (1.0758724, 0.0562972, 1.0758724),
    }, "fit": {"chisq": 5.5329224, "chisq.scaled": 9.9024264, "df": 13, "chisq.scaling.factor": 0.6169929, "cfi.scaled": 1, "tli.scaled": 1.0025055, "rmsea.scaled": 0, "rmsea.ci.upper.scaled": 0.0277789, "srmr": 0.0154669, "wrmr": 0.3468154, "baseline.chisq.scaled": 2018.1253371}},
    "b": {"N": 800, "par": {
        "N =~ 이용_주간보호": (0.7917133, 0.060818, 0.6416311),
        "N =~ 이용_이동지원": (0.94012, 0.0610831, 0.7619049),
        "N ~~ N": (0.6568027, 0.0550387, 1),
        "이용_방문간호 | t1": (0.6241943, 0.0476553, 0.6241943),
    }, "fit": {"chisq": 3.5563922, "chisq.scaled": 5.9429485, "df": 5, "chisq.scaling.factor": 0.6056829, "cfi.scaled": 0.9991944, "tli.scaled": 0.9983889, "rmsea.scaled": 0.0153633, "rmsea.ci.upper.scaled": 0.053361, "srmr": 0.0234843, "wrmr": 0.4869218, "baseline.chisq.scaled": 1180.53809}},
    "c": {"N": 772, "par": {
        "H =~ 우울점수": (-4.2718484, 0.198967, -0.8170567),
        "H =~ 일상활동점수": (13.7829694, 0.674586, 0.8072896),
        "H =~ 삶의만족도": (1.2726453, 0.0889345, 0.5361783),
        "H ~~ H": (0.6615567, 0.0324531, 1),
        "우울점수 ~~ 우울점수": (6.0114505, 0.5087689, 0.3324183),
        "우울점수 ~1": (12.2871762, 0.1532674, 2.8893817),
        "주관적건강 | t2": (-0.3067057, 0.0459468, -0.3067057),
    }, "fit": {"chisq": 0.2995402, "chisq.scaled": 1.6164962, "df": 2, "chisq.scaling.factor": 0.1859357, "cfi.scaled": 1, "tli.scaled": 1.0013492, "rmsea.scaled": 0, "rmsea.ci.upper.scaled": 0.0669979, "srmr": 0.006712, "wrmr": 0.1368257, "baseline.chisq.scaled": 858.705028}},
    "d": {"N": 766, "par": {
        "재이용의향 ~ S": (0.8833358, 0.1048462, 0.6347932),
        "재이용의향 ~ T": (0.2018098, 0.0998742, 0.1535441),
        "T =~ 신뢰_약속": (0.997628, 0.0591045, 0.759031),
        "S ~~ T": (0.3471707, 0.0279308, 0.6349591),
        "재이용의향 | t1": (-0.3962698, 0.046661, -0.3962698),
    }, "fit": {"chisq": 8.6458791, "chisq.scaled": 15.690462, "df": 18, "chisq.scaling.factor": 0.5966269, "cfi.scaled": 1, "tli.scaled": 1.001446, "rmsea.scaled": 0, "rmsea.ci.upper.scaled": 0.0278826, "srmr": 0.0180401, "wrmr": 0.4001361, "baseline.chisq.scaled": 2512.4673615}},
    "e": {"N": 780, "par": {
        "S ~ 연령": (0.0120714, 0.0041032, 0.1197021),
        "S ~ 여성": (0.1918834, 0.0600041, 0.1325514),
        "재이용의향 ~ S": (1.0255943, 0.0660836, 0.732149),
        "재이용의향 ~ 독거": (-0.0285411, 0.0970008, -0.0136934),
        "S ~~ S": (0.5017693, 0.0390424, 0.9677576),
        "만족_친절 | t2": (0.6505561, 0.4236665, 0.6451856),
        "재이용의향 | t1": (0.8803957, 0.5216215, 0.8728367),
    }, "fit": {"chisq": 8.3324748, "chisq.scaled": 12.5312657, "df": 17, "chisq.scaling.factor": 0.8005159, "cfi.scaled": 1, "tli.scaled": 1.0048833, "rmsea.scaled": 0, "rmsea.ci.upper.scaled": 0.0228296, "srmr": 0.0145854, "wrmr": 0.4454125, "baseline.chisq.scaled": 1370.7294841}},
    "e_nox": {"N": 780, "par": {
        "S ~ 연령": (0.0122352, 0.0039827, 0.1223985),
        "S ~ 여성": (0.1939955, 0.0579471, 0.1352807),
        "재이용의향 ~ S": (1.0295654, 0.064137, 0.7338994),
        "연령 ~~ 연령": (50.8506971, 3.9872392, 1),
        "만족_친절 | t2": (0.6035536, 0.3149949, 0.6035536),
    }, "fit": {"chisq": 7.6785492, "chisq.scaled": 12.0268008, "df": 17, "chisq.scaling.factor": 0.7663168, "cfi.scaled": 1, "tli.scaled": 1.0052908, "rmsea.scaled": 0, "rmsea.ci.upper.scaled": 0.0213865, "srmr": 0.0180735, "wrmr": 0.3880203, "baseline.chisq.scaled": 1407.3120481}},
    "f_theta": {"N": 780, "par": {
        "재이용의향 ~ S": (1.048909, 0.1391163, 0.7288445),
        "추천의향 ~ 재이용의향": (0.6449793, 0.0867763, 0.6250058),
        "추천의향 ~ S": (0.2349589, 0.1089428, 0.1582076),
        "S ~~ S": (1.0299587, 0.1605739, 1),
        "추천의향 | t1": (-0.6332903, 0.0724436, -0.4201731),
    }, "fit": {"chisq": 2.6974947, "chisq.scaled": 5.2259565, "df": 8, "chisq.scaling.factor": 0.545538, "cfi.scaled": 1, "tli.scaled": 1.0024504, "rmsea.scaled": 0, "rmsea.ci.upper.scaled": 0.0306954, "srmr": 0.0134107, "wrmr": 0.2816702, "baseline.chisq.scaled": 2137.6235069}},
    "a_wls": {"N": 766, "par": {
        "S =~ 만족_시간": (0.9126196, 0.0527078, 0.6558225),
        "S ~~ T": (0.3535566, 0.0282641, 0.633735),
        "만족_친절 | t1": (-1.1075147, 0.0569823, -1.1075147),
    }, "fit": {"chisq": 9.9712227, "df": 13, "cfi": 1, "tli": 1.0051859, "rmsea": 0, "srmr": 0.0166989, "wrmr": 0.465581}},
    "a_wlsm": {"N": 766, "par": {
        "S =~ 만족_시간": (0.916946, 0.0558374, 0.65811),
        "S ~~ T": (0.3490572, 0.0285557, 0.6348336),
    }, "fit": {"chisq": 5.5329224, "chisq.scaled": 9.6624116, "df": 13, "chisq.scaling.factor": 0.5726233, "cfi.scaled": 1, "tli.scaled": 1.0019462, "rmsea.scaled": 0, "srmr": 0.0154669}},
    "a_ulsmv": {"N": 766, "par": {
        "S =~ 만족_시간": (0.9179837, 0.0585087, 0.65843),
        "S ~~ T": (0.3468202, 0.0287994, 0.6357462),
        "신뢰_안전 | t3": (1.0758724, 0.0562972, 1.0758724),
    }, "fit": {"chisq": 5.0465409, "chisq.scaled": 9.4616549, "df": 13, "chisq.scaling.factor": 0.5856554, "cfi.scaled": 1, "tli.scaled": 1.0032564, "rmsea.scaled": 0, "rmsea.ci.upper.scaled": 0.0263567, "srmr": 0.0153493, "wrmr": 0.3312211, "baseline.chisq.scaled": 1776.2600227}},
    "a_uls": {"N": 766, "par": {
        "S =~ 만족_시간": (0.9179837, 0.0585087, 0.65843),
        "S ~~ T": (0.3468202, 0.0287994, 0.6357462),
    }, "fit": {"chisq": 5.0465409, "df": 13, "srmr": 0.0153493}},
    "a_theta": {"N": 766, "par": {
        "S =~ 만족_시간": (0.8480288, 0.0980771, 0.65811),
        "S ~~ T": (0.7799238, 0.1092051, 0.6348335),
        "T ~~ T": (1.4207203, 0.2624838, 1),
        "만족_친절 | t1": (-1.6050824, 0.0994736, -1.1176709),
    }, "fit": {"chisq": 5.5329224, "chisq.scaled": 9.9024263, "df": 13, "chisq.scaling.factor": 0.6169929, "cfi.scaled": 1, "tli.scaled": 1.0025055, "rmsea.scaled": 0, "rmsea.ci.upper.scaled": 0.0277789, "srmr": 0.0154669, "wrmr": 0.3468154, "baseline.chisq.scaled": 2018.1253371}},
    "e_theta": {"N": 780, "par": {
        "S ~ 여성": (0.2718457, 0.0867385, 0.1325514),
        "재이용의향 ~ S": (1.0534641, 0.1410416, 0.7321489),
        "만족_친절 | t2": (0.921674, 0.6026695, 0.6451971),
    }, "fit": {"chisq": 8.3324748, "chisq.scaled": 12.5312657, "df": 17, "chisq.scaling.factor": 0.8005159, "cfi.scaled": 1, "tli.scaled": 1.0048834, "rmsea.scaled": 0, "rmsea.ci.upper.scaled": 0.0228296, "srmr": 0.0145854, "wrmr": 0.4454125, "baseline.chisq.scaled": 1370.7294841}},
    "e_wls": {"N": 780, "par": {
        "S ~ 여성": (0.1960901, 0.0594793, 0.1345597),
        "재이용의향 ~ S": (1.0294538, 0.0649104, 0.7395927),
        "만족_친절 | t2": (0.6062241, 0.3216138, 0.6010915),
    }, "fit": {"chisq": 12.8354555, "df": 17, "cfi": 1, "tli": 1.0104251, "rmsea": 0, "srmr": 0.0156217, "wrmr": 0.5528165}},
    "c_wls": {"N": 772, "par": {
        "H =~ 우울점수": (-4.3004942, 0.1939616, -0.8200374),
        "우울점수 ~~ 우울점수": (5.9273528, 0.4953794, 0.3275387),
        "주관적건강 | t2": (-0.3087578, 0.0458604, -0.3087578),
    }, "fit": {"chisq": 1.6853904, "df": 2, "cfi": 1, "tli": 1.0016467, "rmsea": 0, "srmr": 0.0068268, "wrmr": 0.3245565}},
    "mlm_a": {"N": 766, "par": {
        "S =~ 만족_시간": (0.849165, 0.0621532, 0.627916),
        "S ~~ T": (0.3455981, 0.0327405, 0.6313853),
        "만족_친절 ~~ 만족_친절": (0.7550538, 0.0541134, 0.5306565),
    }, "fit": {"chisq": 9.7579828, "chisq.scaled": 9.5798836, "df": 13, "chisq.scaling.factor": 1.018591, "cfi.scaled": 1, "tli.scaled": 1.0044876, "rmsea.scaled": 0, "rmsea.ci.upper.scaled": 0.0267305, "baseline.chisq.scaled": 1252.1396292}},
    "mlm_p": {"N": 780, "par": {
        "삶의만족도 ~ S": (1.2401682, 0.0963293, 0.5335184),
        "삶의만족도 ~ 연령": (-0.0719338, 0.0084301, -0.2666861),
        "S ~ 연령": (0.0137285, 0.0047869, 0.1183098),
        "삶의만족도 ~~ 삶의만족도": (2.5100473, 0.1338452, 0.6775579),
    }, "fit": {"chisq": 12.8918064, "chisq.scaled": 12.5650747, "df": 12, "chisq.scaling.factor": 1.0260032, "cfi.scaled": 0.9992995, "tli.scaled": 0.9988324, "rmsea.scaled": 0.0077699, "rmsea.ci.upper.scaled": 0.0377515, "baseline.chisq.scaled": 826.6183342}},
    "mlm_cfa": {"N": 300, "par": {
        "Visual =~ x2_vis": (0.9940167, 0.075194, 0.8016036),
        "Visual ~~ Textual": (0.3955997, 0.06408, 0.4751992),
        "x9_spd ~~ x9_spd": (0.436948, 0.0520436, 0.4582335),
    }, "fit": {"chisq": 23.0894996, "chisq.scaled": 23.6330712, "df": 24, "chisq.scaling.factor": 0.9769995, "cfi.scaled": 1, "tli.scaled": 1.0003797, "rmsea.scaled": 0, "rmsea.ci.upper.scaled": 0.0459753, "baseline.chisq.scaled": 1485.5375656}},
}
SS_A = [-1.11767091, -0.46084667, 0.39981194, 1.17446486, -1.48184628, -0.76130453, 0.33667851, 0.36953758, 0.30228566, 0.44498755]
SS_A_GAMMA = [2.5083083, 0.426785, 1.0891194, 0.1887419]
SS_C = [-1.13475811, -0.30670574, 0.48603408, 1.47617703, -12.28717617, -68.93445596, -5.54119171, 18.08399358, 192.83860553, 3.72703381, -2.84142393, 9.00431657, 0.85412082, -39.32173431, -3.47294327, 11.81908588]
SS_E_SLOPES = [0.01304793, 0.00913692, 0.00947518, 0.01183977, 0.01422011]
MI_D = [('신뢰_정보', '~~', '신뢰_안전', 2.153143, 0.071689, 0.147777), ('신뢰_약속', '~~', '신뢰_안전', 2.072751, -0.072643, -0.149258), ('S', '=~', '신뢰_약속', 2.069286, 0.164759, 0.118401), ('S', '=~', '신뢰_정보', 1.979563, -0.166987, -0.120002)]


@pytest.fixture(scope="module")
def care():
    return pd.read_csv(CARE)


_CACHE = {}


def _fit(case):
    if case not in _CACHE:
        model, cat, est, par, cx, path = CASES[case]
        d = pd.read_csv(path)
        if est == "mlm":
            _CACHE[case] = se.estimate(d, model, estimator="mlm")
        else:
            _CACHE[case] = se.estimate(d, model, estimator=est, categorical=cat, parameterization=par, conditional_x=cx)
    return _CACHE[case]


def _rows(res):
    der = se.Derived(res)
    std = der.std_values(res.theta, "stdyx")
    out = {}
    for n, r in enumerate(res.spec.rows):
        key = f"{r.lhs} {r.op} {r.rhs}".strip()
        if r.op == "~~" and key not in out:
            out[f"{r.rhs} ~~ {r.lhs}"] = (r, n)
        out[key] = (r, n)
    return out, std


def _close(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_bivariate_normal_cdf_accuracy():
    from scipy import stats
    rng = np.random.default_rng(1)
    for r in (-0.97, -0.6, -0.1, 0.0, 0.2, 0.5, 0.8, 0.93, 0.995):
        h, k = rng.normal(0, 1.5, 40), rng.normal(0, 1.5, 40)
        ref = np.array([stats.multivariate_normal([0, 0], [[1, r], [r, 1]]).cdf([a, b]) for a, b in zip(h, k)])
        assert np.max(np.abs(sw.pbvn(h, k, r) - ref)) < 1e-7
    assert sw.pbvn(np.array([np.inf, -np.inf, 0.3]), np.array([0.4, 1.0, np.inf]), 0.5).tolist() == pytest.approx(
        [stats.norm.cdf(0.4), 0.0, stats.norm.cdf(0.3)])


def test_sample_statistics_and_gamma_match_lavaan(care):
    res = _fit("a")
    ss = res.sem.ss
    # thresholds of 만족_친절 (4) and 만족_시간 (first two), then polychoric r(친절, 시간..) — lavaan wls.obs order
    assert np.max(np.abs(ss.s[:6] - np.array(SS_A[:6]))) < 1e-6
    assert np.max(np.abs(ss.s[28:32] - np.array(SS_A[6:]))) < 1e-6
    G = ss.Gamma
    for (i, j), ref in zip(((0, 0), (5, 3), (30, 30), (40, 29)), SS_A_GAMMA):
        assert abs(G[i, j] - ref) < 1e-5 * max(1, abs(ref))
    # mixed ordinal + continuous: thresholds, means (lavaan stores −mean), variances, polyserial / Pearson covariances
    ssc = _fit("c").sem.ss
    flip = np.array([-1.0 if lab[0] == "mean" else 1.0 for lab in ssc.labels])
    assert np.max(np.abs(ssc.s * flip - np.array(SS_C)) / np.maximum(1, np.abs(SS_C))) < 1e-7
    # probit / linear regressions on the covariates (conditional.x): slopes on 연령
    sse = _fit("e").sem.ss
    assert np.max(np.abs(sse.s[17:22] - np.array(SS_E_SLOPES))) < 1e-7


def _lavaan_to_mplus(case, lk, v, N):
    """lavaan 0.7.2 mimic = "Mplus" -> Mplus 8.8 conventions (the engine default, checked on the Mplus User's Guide
    example outputs in test_sem_mplus_examples.py). WLS family: Mplus divides by N where lavaan divides by N − 1
    (SE × (N−1)/N, unscaled χ² and scaling factor × N/(N−1), RMSEA × √((N−1)/N), WRMR × √(N/(N−1))); scaled χ²,
    baseline χ² and CFI are unchanged. TLI is truncated at 1 (Mplus). SRMR: Mplus 8.1 definition, tested there."""
    wls = CASES[case][2] in sw_estimators
    c = N / (N - 1)
    if lk in ("tli", "tli.scaled"):
        return min(v, 1.0)
    if not wls:
        return v
    if lk == "se":                  # WLS: lavaan E⁻¹/(N−1) for any mimic; Mplus E⁻¹/N (the sandwich with W = Γ⁻¹)
        return v / math.sqrt(c) if CASES[case][2] == "wls" else v / c
    if lk in ("chisq", "chisq.scaling.factor") or (lk == "chisq.scaled" and CASES[case][2] in ("wls", "uls")):
        return v * c
    if lk.startswith("rmsea"):
        return v / math.sqrt(c)
    if lk == "wrmr":
        return v * math.sqrt(c)
    return v


sw_estimators = ("wlsmv", "wlsm", "wls", "ulsmv", "uls")


@pytest.mark.parametrize("case", list(REF))
def test_estimates_se_std_and_fit_match_lavaan(case):
    res = _fit(case)
    ref = REF[case]
    N = ref["N"]
    assert res.sem.N == N
    assert res.fit.converged
    rows, std = _rows(res)
    tol_est = 5e-5 if ("theta" in case or case == "e_nox") else 1e-5
    for key, (est, sev, sall) in ref["par"].items():
        r, n = rows[key]
        val = res.theta[r.pid] if r.free else r.value
        assert _close(val, est, tol_est), (case, key, val, est)
        sev = _lavaan_to_mplus(case, "se", sev, N)
        assert _close(res.se[r.pid], sev, 1e-5), (case, key, res.se[r.pid], sev)
        assert _close(std[n], sall, tol_est), (case, key, std[n], sall)
    fs = res.fitstats
    for lk, ref_v in ref["fit"].items():
        if lk == "wrmr" and CASES[case][2] in ("uls", "ulsmv"):
            assert math.isnan(fs["wrmr"])                   # WRMR is reported for weighted estimators only
            continue
        if lk == "srmr" and CASES[case][2] in sw_estimators:
            continue                                        # Mplus 8.1 SRMR: test_sem_mplus_examples.py
        if CASES[case][2] in ("wls", "uls") and lk == "chisq":
            ours = fs["chi2"]
        else:
            ours = fs[FIT_MAP[lk]]
        ref_v = _lavaan_to_mplus(case, lk, ref_v, N)
        tol = 2e-5 if lk.startswith("baseline") else 1e-5
        assert _close(ours, ref_v, tol), (case, lk, ours, ref_v)


@pytest.mark.parametrize("case", ["a", "e", "f_theta", "a_wls", "a_ulsmv"])
def test_lavaan_conventions_with_mimic_lavaan(case):
    """mimic = "lavaan": lavaan's own N − 1 conventions and categorical SRMR (lavaan mimic = "lavaan" values)."""
    model, cat, est, par, cx, path = CASES[case]
    res = se.estimate(pd.read_csv(path), model, estimator=est, categorical=cat, parameterization=par,
                      conditional_x=cx, mimic="lavaan")
    ref = REF[case]
    N = ref["N"]
    rows, _ = _rows(res)
    for key, (est_v, sev, _s) in ref["par"].items():
        r, _n = rows[key]
        # lavaan mimic = "Mplus" SE = lavaan default SE × √(N/(N−1))... the engine's lavaan mode divides by N − 1
        assert _close(res.se[r.pid], sev / math.sqrt(N / (N - 1)) if est != "wls" else sev, 1e-5), (case, key)
    if "srmr" in ref["fit"]:
        assert _close(res.fitstats["srmr"], ref["fit"]["srmr"], 1e-5)


def test_ordinal_mediator_se_uses_full_delta_derivative():
    """Delta parameterization with a categorical mediator: the residual variance of the mediator's y* depends on the
    other parameters. lavaan 0.7.2's analytic Δ omits that term (its own model function differentiated numerically
    gives the values below); the theta parameterization (no such term) agrees with lavaan's reported SEs."""
    res = se.estimate(pd.read_csv(CARE), M_F, categorical=S + ["재이용의향", "추천의향"])
    rows, _ = _rows(res)
    k = (780 - 1) / 780                 # lavaan mimic = "Mplus" SE -> Mplus 8.8 SE (N = 780)
    ref = {"재이용의향 ~ S": (1.0232190422, 0.0647665724), "추천의향 ~ 재이용의향": (0.6250058806, 0.0689800678),
           "추천의향 ~ S": (0.2221062398, 0.1035341359)}
    for key, (est, sev) in ref.items():
        r, _ = rows[key]
        assert abs(res.theta[r.pid] - est) < 1e-6 and abs(res.se[r.pid] - sev * k) < 1e-6
    der = se.Derived(res)
    val, sev = se.delta_se(der.defined_values, res.theta, res.vcov)
    assert abs(val[0] - 0.6395179185) < 1e-6 and abs(sev[0] - 0.0890107763 * k) < 1e-6
    rt = _fit("f_theta")
    val, sev = se.delta_se(se.Derived(rt).defined_values, rt.theta, rt.vcov)
    assert abs(val[0] - 0.6765245934) < 1e-6 and abs(sev[0] - 0.1190047290 * k) < 1e-6  # lavaan theta: 0.1190047


def test_delta_theta_equivalence_and_rsquare(care):
    rd, rt = _fit("a"), _fit("a_theta")
    assert abs(rd.fitstats["chi2_test"] - rt.fitstats["chi2_test"]) < 1e-6     # reparameterization: same fit
    std_d = se.Derived(rd).std_values(rd.theta, "stdyx")
    std_t = se.Derived(rt).std_values(rt.theta, "stdyx")
    sd = {(r.lhs, r.op, r.rhs): std_d[n] for n, r in enumerate(rd.spec.rows) if r.op in ("=~", "|")}
    st = {(r.lhs, r.op, r.rhs): std_t[n] for n, r in enumerate(rt.spec.rows) if r.op in ("=~", "|")}
    assert max(abs(sd[k] - st[k]) for k in sd) < 1e-5
    # delta: Var(y*) = 1 so R² + residual variance = 1 for every indicator
    A, Sm, M = rd.sem.ram.matrices(rd.theta, 0)
    for g, v, r2 in se.Derived(rd).rsquare_values(rd.theta):
        if v in rd.spec.categorical:
            assert abs(r2 + Sm[rd.spec.idx[v], rd.spec.idx[v]] - 1) < 1e-10


def test_modification_indices_wls_match_lavaan():
    res = se.estimate(pd.read_csv(CARE), M_D, categorical=S + T + ["재이용의향"])
    mis = {(m["lhs"], m["op"], m["rhs"]): m for m in sw.modification_indices_wls(res, 0)}
    assert len(mis) == 35                                     # lavaan modindices: 35 candidates
    for lhs, op, rhs, mi, epc, sepc in MI_D:
        m = mis.get((lhs, op, rhs)) or mis.get((rhs, op, lhs))
        assert abs(m["mi"] - mi) < 1e-4 and abs(m["epc"] - epc) < 1e-5 and abs(m["std_epc"] - sepc) < 1e-5


def test_threshold_syntax_and_constraints(care):
    base = "F =~ 이용_방문요양 + 이용_주간보호 + 이용_방문간호"
    r0 = se.estimate(care, base, categorical=U[:3])
    r1 = se.estimate(care, base + "\n이용_주간보호 | th*t1\n이용_방문간호 | th*t1", categorical=U[:3])
    assert r0.fitstats["df"] == 0 and r1.fitstats["df"] == 1
    th = [r1.theta[x.pid] for x in r1.spec.rows if x.op == "|" and x.lhs in ("이용_주간보호", "이용_방문간호")]
    assert abs(th[0] - th[1]) < 1e-12
    r2 = se.estimate(care, base + "\n이용_방문요양 | 0*t1", categorical=U[:3])
    assert [x.value for x in r2.spec.rows if x.op == "|" and x.lhs == "이용_방문요양"] == [0.0]
    with pytest.raises(se.SEMError):
        se.estimate(care, base + "\n이용_방문요양 | t2", categorical=U[:3])        # binary: only t1
    with pytest.raises(se.SEMError):
        se.estimate(care, "F =~ x1 + x2 + x3\nx1 | t1".replace("x", "만족_친"), categorical=None)


def test_module_tables_labels_and_settings(care):
    d = care.copy()
    lab = {1: "매우 불만", 2: "불만", 3: "보통", 4: "만족", 5: "매우 만족"}
    d["만족_친절"] = pd.Categorical(d["만족_친절"].map(lab), categories=list(lab.values()), ordered=True)
    d["재이용의향"] = d["재이용의향"].map({0: "아니오", 1: "예"})
    m = "S =~ " + " + ".join(S) + "\n재이용의향 ~ a*S\n추천의향 ~ b*재이용의향 + S\nind := a*b"
    cat = S + ["재이용의향", "추천의향"]
    r = SEM(d, m, categorical=", ".join(cat), indirect=["S -> 재이용의향 -> 추천의향"], modIndices=True, sampstat=True)
    rc = SEM(care, m, categorical=cat)
    assert [t.name for t in r.tables] == ["estimates", "fit", "assumptions", "rsquare", "defined", "indirect",
                                          "modindices", "sampstat"]
    e1 = [x["est"] for x in r.get_table("estimates").rows]
    e2 = [x["est"] for x in rc.get_table("estimates").rows]
    assert np.allclose(e1, e2)                                     # labels recoded in their category order
    est = {x["mplus"]: x for x in r.get_table("estimates").rows}
    assert "[만족_친절$4]" in est and "[추천의향$2]" in est and "[재이용의향$1]" in est
    assert not any(x["op"] == "~~" and x["lval"] == x["rval"] and x["lval"] in cat for x in r.get_table("estimates").rows)
    fit = {x["index"]: x["value"] for x in r.get_table("fit").rows}
    assert "척도 보정 계수 (1/a)" in fit and "WRMR" in fit and "AIC" not in fit
    settings = {x["item"]: x["value"] for x in r.get_table("assumptions").rows}
    assert "매우 불만→0" in settings["범주형 변수 (CATEGORICAL)"] and "WLSMV" in settings["추정법"]
    r2 = {x["var"]: x for x in r.get_table("rsquare").rows}
    assert r2["재이용의향"]["type"] == "범주형 (y*)" and abs(r2["만족_시간"]["r2"] + r2["만족_시간"]["resvar"] - 1) < 1e-10
    ind = r.get_table("indirect").rows[0]
    dfn = r.get_table("defined").rows[0]
    assert abs(ind["est"] - dfn["est"]) < 1e-12 and abs(ind["se"] - dfn["se"]) < 1e-8
    assert "categorical=" in r.syntax_python and "ordered = c(" in r.syntax_r and "WLSMV" in r.apa_narrative
    assert set(r.model_stats) >= {"N", "Chi2", "df", "CFI", "TLI", "RMSEA", "SRMR"} and "LL" not in r.model_stats
    ss = r.get_table("sampstat").rows
    assert any(x["kind"].startswith("다분상관") for x in ss) and any(x["kind"] == "문턱값" for x in ss)
    # theta parameterization and ULS through the module
    rt = SEM(care, M_A, categorical=S + T, parameterization="theta")
    assert {x["item"]: x["value"] for x in rt.get_table("assumptions").rows}["모수화"].startswith("theta")
    ru = SEM(care, M_A, categorical=S + T, estimator="uls")
    fu = {x["index"]: x["value"] for x in ru.get_table("fit").rows}
    assert "SRMR" in fu and not any(k.startswith("χ² (모형") for k in fu)


def test_module_mlm(care):
    r = SEM(care, M_A, estimator="mlm")
    fit = {x["index"]: x["value"] for x in r.get_table("fit").rows}
    assert abs(fit["χ² 척도 보정 계수 (MLM)"] - REF["mlm_a"]["fit"]["chisq.scaling.factor"]) < 1e-6
    assert abs(fit["χ² (모형 적합도 검정) — MLM Satorra–Bentler 보정"] - REF["mlm_a"]["fit"]["chisq.scaled"]) < 1e-5
    settings = {x["item"]: x["value"] for x in r.get_table("assumptions").rows}
    assert "MLM" in settings["추정법"] and settings["결측 처리"] == "목록별 삭제"
    with pytest.raises(ValueError):
        SEM(care, M_A, estimator="mlm", missing="fiml")


def test_errors(care):
    m = "S =~ " + " + ".join(S) + "\n재이용의향 ~ S"
    cat = S + ["재이용의향"]
    bad = [dict(estimator="ml"), dict(estimator="mlr"), dict(missing="fiml"), dict(group="여성"), dict(bootstrap=50),
           dict(parameterization="gamma"), dict(estimator="wlsmvx")]
    for kw in bad:
        with pytest.raises(ValueError):
            SEM(care, m, categorical=cat, **kw)
    with pytest.raises(ValueError):                    # WLS family needs categorical variables (Mplus would allow)
        SEM(care, "S =~ " + " + ".join(S), estimator="wlsmv")
    with pytest.raises(ValueError, match="잔차분산"):
        SEM(care, m + "\n재이용의향 ~~ 재이용의향", categorical=cat)
    with pytest.raises(ValueError, match="종속변수"):
        SEM(care, "S =~ 만족_친절 + 만족_시간 + 만족_전문성\nS ~ 여성", categorical=S[:3] + ["여성"])
    with pytest.raises(ValueError, match="최대 10"):
        SEM(care, "S =~ 만족_친절 + 만족_시간 + 연령", categorical=["만족_친절", "만족_시간", "연령"])
    with pytest.raises(ValueError, match="데이터에 없음"):
        SEM(care, m, categorical=cat + ["없는변수"])
    with pytest.raises(ValueError, match="정수가 아닌"):
        SEM(care, "H =~ 주관적건강 + 우울점수 + 일상활동점수", categorical=["주관적건강", "우울점수"])


def test_listwise_note_and_sample_size(care):
    res = se.estimate(care, M_A, categorical=S + T)
    assert res.sem.N == 766 and any("목록별 삭제" in n for n in res.notes)


def test_mlm_matches_stata_sbentler(care):
    """StataNow 19.5 `sem (S -> …) (T -> …), vce(sbentler)` (listwise ML, Satorra–Bentler). Without observed covariates
    Stata, lavaan mimic = "Mplus" and this engine agree; intercept SEs differ by √(N/(N−1)) because the Mplus convention
    rescales the mean block of Γ by (N−1)/N."""
    r = se.estimate(care, M_A, estimator="mlm")
    fs = r.fitstats
    assert abs(fs["chi2"] - 9.7579828225) < 1e-7 and abs(fs["chi2_test"] - 9.5798835819) < 1e-6
    assert abs(fs["scaling"] - 1.0185909608) < 1e-8 and abs(fs["chi2_base"] - 1252.139629204234) < 1e-6
    rows, _ = _rows(r)
    stata = {"S =~ 만족_시간": (.849165, .0621532), "S =~ 만족_소통": (.7155223, .055505), "T =~ 신뢰_안전": (.7838891, .0542216),
             "S ~~ T": (.3455981, .0327405), "S ~~ S": (.6678135, .0657195), "만족_친절 ~~ 만족_친절": (.7550539, .0541134)}
    for key, (est, sev) in stata.items():
        x, _ = rows[key]
        assert abs(r.theta[x.pid] - est) < 1e-6 and abs(r.se[x.pid] - sev) < 1e-6
    x, _ = rows["만족_친절 ~1"]
    assert abs(r.se[x.pid] * math.sqrt(766 / 765) - .0431272) < 1e-6
    rc = se.estimate(pd.read_csv(LAT3), M_CFA, estimator="mlm")
    assert abs(rc.fitstats["chi2_test"] - 23.6330714948) < 1e-6


def test_scale_factor_and_theta_residual_variance_rules(care):
    """Mplus rules: scale factors {y} (lavaan 'y ~*~ y') are parameters only under delta, y* residual variances only
    under theta; both only for categorical variables."""
    base = "S =~ " + " + ".join(S)
    with pytest.raises(se.SEMError, match="척도요인"):
        se.estimate(care, base + "\n만족_시간 ~*~ 만족_시간")                       # no categorical variables
    with pytest.raises(se.SEMError, match="theta 모수화에서는 척도요인"):
        se.estimate(care, base + "\n만족_시간 ~*~ 만족_시간", categorical=S, parameterization="theta")
    with pytest.raises(se.SEMError, match="delta 모수화에서 모수가 아님"):
        se.estimate(care, base + "\n만족_시간 ~~ 만족_시간", categorical=S)
    # a free scale factor in a one-factor CFA is not identified (it trades off with the loading); fixing it at 1
    # explicitly reproduces the default model exactly
    r0 = se.estimate(care, base, categorical=S)
    r1 = se.estimate(care, base + "\n만족_시간 ~*~ 1*만족_시간", categorical=S)
    assert abs(r0.fitstats["chi2_test"] - r1.fitstats["chi2_test"]) < 1e-9
    rt = se.estimate(care, base + "\n만족_시간 ~~ 1*만족_시간", categorical=S, parameterization="theta")
    assert abs(rt.fitstats["chi2_test"] - r0.fitstats["chi2_test"]) < 1e-6


def test_stata_and_r_export_of_categorical_growth_model():
    from pyjamovi import code_export as ce
    U = ["u1", "u2", "u3"]
    model = ("i =~ 1*u1 + 1*u2 + 1*u3\ns =~ 0*u1 + 1*u2 + 2*u3\nu1 | th*t1\nu2 | th*t1\nu3 | th*t1\n"
             "i ~ 0*1\ns ~ 1\nu2 ~*~ u2\nu3 ~*~ u3")
    st = ce._sem_stata({"model": model, "categorical": U})
    gs = [x for x in st if x.startswith("gsem ")][0]
    assert "<- 1" not in gs and "~*~" not in gs           # intercepts / means and scale factors are not gsem paths
    assert any("척도요인" in x for x in st)
    r = "\n".join(ce._sem_r({"model": model, "categorical": U}))
    assert "u2 ~*~ u2" in r and "N/(N−1)" in r

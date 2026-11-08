"""
Multilevel linear mixed models (Stata `mixed`, up to four nested levels): agreement with R lme4 / nlme
(ML and REML; random intercepts, correlated and independent random slopes, noconstant, exchangeable and
identity structures, centering, factor covariates, cross-level interactions, BLUPs and conditional variances),
Stata-metric variance-component standard errors and ICCs (checked against an independent R computation:
marginal likelihood from V blocks + numDeriv Hessian in log-SD / atanh metric), LR test vs linear regression,
backward compatibility of the old two-level options, Stata transpile and code export, option validation, speed.

Reference values were produced with R 4.4 (lme4 2.0-6 with bobyqa rhoend = 1e-10, nlme, numDeriv) and are
hard-coded so the tests run without R.
"""

import importlib.util
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reconstructed"))

from pyjamovi.analyses import multilevel as ml  # noqa: E402
from pyjamovi.analyses.multilevel import GLMM_ENGINE, LMM_ENGINE  # noqa: E402
from pyjamovi.registry import dispatch_analysis  # noqa: E402

SAMPLE = os.path.join(ROOT, "sample_data", "Multilevel_School3L.csv")
BASE = {"dep": "수학성취도", "levels": ["학교ID", "학급ID"], "covs": ["가정SES", "성별", "학습시간"]}


@pytest.fixture(scope="module")
def school():
    return pd.read_csv(SAMPLE)


def run(df, **params):
    return dispatch_analysis("mixedModel", df, params)


def fixed(res):
    rows = res.get_table("fixed").rows
    return np.array([r["est"] for r in rows]), np.array([r["se"] for r in rows])


def vc(res):
    """random-effects table (variance metric) -> {(level, parameter): (est, se, lo, hi)}"""
    out, lev = {}, None
    for r in res.get_table("random").rows:
        if r["grp"]:
            lev = r["grp"].split(":")[0]
        out[(lev, r["name"])] = (r["est"], r["se"], r["cil"], r["ciu"])
    return out


def close(a, b, tol, rel=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.abs(a - b) / (np.abs(b) if rel else 1.0)
    assert np.all(d < tol), (a, b, d.max())


# ---------------------------------------------------------------------------------------------------- data
def test_sample_data_structure_and_generator(school):
    spec = importlib.util.spec_from_file_location("mk_ml", os.path.join(ROOT, "sample_data", "make_multilevel_sample_data.py"))
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    gen = mk.make_school3l()
    assert gen.shape == school.shape == (4930, 13)
    pd.testing.assert_frame_equal(gen.reset_index(drop=True), school, check_dtype=False)
    assert school["학교ID"].nunique() == 50
    assert school.groupby(["학교ID", "학급ID"]).ngroups == 224
    assert school["학급ID"].nunique() == 6            # class ids repeat across schools (nested ids)
    assert set(school["학교유형"]) == {"공립", "사립"}


# ---------------------------------------------------------------------------------------------------- lme4
# y ~ ses + female + hours + (1 | school/class)
M1 = {
    "ml": dict(ll=-17137.90757, aic=34289.81514, bic=34335.16576,
               beta=[52.26631415, 3.269846068, 1.254300648, 0.5983647855],
               se=[0.7726912394, 0.1554828614, 0.2405494861, 0.030331261],
               school=21.96023445, cls=7.48462004, resid=67.11041829),
    "reml": dict(ll=-17141.41048, aic=34296.82097, bic=34342.17159,
                 beta=[52.26603841, 3.268992478, 1.25426281, 0.5983744995],
                 se=[0.7792026183, 0.1555490013, 0.2406266891, 0.0303409773],
                 school=22.46073189, cls=7.486602625, resid=67.1533374),
}


@pytest.mark.parametrize("method", ["ml", "reml"])
def test_three_level_random_intercepts_vs_lme4(school, method):
    ref = M1[method]
    r = run(school, **BASE, method=method)
    b, se = fixed(r)
    close(b, ref["beta"], 1e-5)
    close(se, ref["se"], 1e-5)
    assert abs(r.model_stats["LL"] - ref["ll"]) < 1e-4
    assert abs(r.model_stats["AIC"] - ref["aic"]) < 1e-3 and abs(r.model_stats["BIC"] - ref["bic"]) < 1e-3
    v = vc(r)
    close([v[("학교ID", "var(_cons)")][0], v[("학급ID", "var(_cons)")][0], v[("잔차", "var(Residual)")][0]],
          [ref["school"], ref["cls"], ref["resid"]], 1e-5, rel=True)
    g = {row["level"]: row for row in r.get_table("groups").rows}
    assert g["학교ID"]["groups"] == 50 and g["학급ID"]["groups"] == 224
    assert (g["학급ID"]["min"], g["학급ID"]["max"]) == (14, 30)
    assert r.model_stats["N"] == 4811 and r.model_stats["Groups"] == 224
    assert any("안에서만 고유" in n for n in r.get_table("info").notes)      # nested-id detection noted


# y ~ ses + female + hours + (1 + ses | school) + (1 | school:class)
M2 = {
    "ml": dict(ll=-17111.03506, aic=34240.07013, bic=34298.37807,
               beta=[52.19971591, 3.235370889, 1.271561716, 0.5932531934],
               se=[0.7278955686, 0.2483526538, 0.2390434143, 0.03015524897],
               v_ses=1.840619542, v_cons=18.57675606, c=3.774037879, cls=7.437835335, resid=65.92696666),
    "reml": dict(ll=-17114.23804, aic=34246.47608, bic=34304.78403,
                 beta=[52.20097288, 3.233824927, 1.271361957, 0.5932661198],
                 se=[0.7341162188, 0.2508994296, 0.2391051653, 0.03016359461],
                 v_ses=1.90215687, v_cons=19.02504384, c=3.843100178, cls=7.438861559, resid=65.95389961),
}


@pytest.mark.parametrize("method", ["ml", "reml"])
def test_correlated_random_slope_at_top_level_vs_lme4(school, method):
    ref = M2[method]
    r = run(school, **BASE, method=method, randomSlopes={"학교ID": ["가정SES"]}, covariance={"학교ID": "unstructured"})
    b, se = fixed(r)
    close(b, ref["beta"], 1e-5)
    close(se, ref["se"], 1e-5)
    assert abs(r.model_stats["LL"] - ref["ll"]) < 1e-4
    v = vc(r)
    close([v[("학교ID", "var(가정SES)")][0], v[("학교ID", "var(_cons)")][0], v[("학교ID", "cov(가정SES,_cons)")][0],
           v[("학급ID", "var(_cons)")][0], v[("잔차", "var(Residual)")][0]],
          [ref["v_ses"], ref["v_cons"], ref["c"], ref["cls"], ref["resid"]], 2e-5, rel=True)
    icc = r.get_table("icc")                                  # Stata: conditional ICC (random-effect covariates at 0)
    assert icc.title.startswith("조건부 급내상관")
    close([row["icc"] for row in icc.rows], [x[0] for x in STATA_ICC[{"ml": "m2_mle", "reml": "m2_reml"}[method]]], 1e-5)
    assert r.engine == "statsmodels MixedLM"


def test_ui_parameter_names_give_the_same_model(school):
    a = run(school, **BASE, method="ml", randomSlopes={"학교ID": ["가정SES"]}, covariance={"학교ID": "unstructured"})
    b = run(school, dep="수학성취도", cluster="학급ID", cluster3="학교ID", covs=["가정SES", "성별", "학습시간"],
            method="ml", slopes3=["가정SES"], covStruct3="unstructured", covStruct2="independent", noConstant2=False)
    assert abs(a.model_stats["LL"] - b.model_stats["LL"]) < 1e-9
    assert a.options["levels"] == b.options["levels"] == ["학교ID", "학급ID"]


def test_independent_slopes_at_two_levels_vs_lme4(school):
    # (1 + ses || school) + (1 + hours || school:class), ML
    r = run(school, **BASE, method="ml", randomSlopes={"학교ID": ["가정SES"], "학급ID": ["학습시간"]})
    b, se = fixed(r)
    close(b, [52.26799664, 3.22122602, 1.275489471, 0.5930134308], 1e-5)
    close(se, [0.7221932589, 0.2501559373, 0.2385101646, 0.0323617733], 1e-5)
    assert abs(r.model_stats["LL"] - (-17110.40322)) < 1e-4
    v = vc(r)
    close([v[("학교ID", "var(가정SES)")][0], v[("학교ID", "var(_cons)")][0], v[("학급ID", "var(학습시간)")][0],
           v[("학급ID", "var(_cons)")][0], v[("잔차", "var(Residual)")][0]],
          [1.878402774, 18.57691785, 0.03012533647, 5.527913174, 65.07514656], 1e-4, rel=True)
    assert ("학교ID", "cov(가정SES,_cons)") not in v


def test_unstructured_slopes_at_two_levels_reml_vs_lme4(school):
    r = run(school, **BASE, method="reml", randomSlopes={"학교ID": ["가정SES"], "학급ID": ["학습시간"]},
            covariance="unstructured")
    b, se = fixed(r)
    close(b, [52.14933172, 3.218166401, 1.293424088, 0.5980759856], 1e-5)
    close(se, [0.7750541637, 0.2467365989, 0.2371290641, 0.03910765518], 1e-5)
    assert abs(r.model_stats["LL"] - (-17092.25723)) < 1e-4
    v = vc(r)
    close([v[("학급ID", "var(_cons)")][0], v[("학급ID", "var(학습시간)")][0], v[("학급ID", "cov(학습시간,_cons)")][0],
           v[("학교ID", "var(_cons)")][0], v[("학교ID", "var(가정SES)")][0], v[("학교ID", "cov(가정SES,_cons)")][0]],
          [20.9992725, 0.1353522392, -1.354375784, 19.025997, 1.815448235, 3.588445594], 1e-4, rel=True)


def test_noconstant_level_vs_lme4(school):
    # (0 + ses | school) + (1 | school:class)
    r = run(school, **BASE, method="ml", randomSlopes={"학교ID": ["가정SES"]}, noconstant=["학교ID"])
    b, se = fixed(r)
    close(b, [52.35065563, 3.275865496, 1.290222561, 0.590357916], 1e-5)
    close(se, [0.4891844474, 0.2702827767, 0.2401291515, 0.03029459451], 1e-5)
    assert abs(r.model_stats["LL"] - (-17180.17572)) < 1e-4
    v = vc(r)
    assert ("학교ID", "var(_cons)") not in v
    assert r.get_table("icc") is None            # Stata: estat icc not allowed when a level has no random intercept
    close([v[("학교ID", "var(가정SES)")][0], v[("학급ID", "var(_cons)")][0]], [2.400667695, 25.75693144], 1e-5, rel=True)


def test_centering_factors_and_cross_level_interaction_vs_lme4(school):
    # ses_cwc = ses - class mean, ses_m = class mean; factor(type) factor(region) ses_cwc:factor(type)
    r = run(school, dep="수학성취도", levels=["학교ID", "학급ID"], covs=["가정SES", "성별", "학습시간"],
            factors=["학교유형", "지역"], interactions=["가정SES:학교유형"], centerGroup=["가정SES"],
            groupMeans=["가정SES"], method="ml", randomSlopes={"학교ID": ["가정SES"]}, covariance={"학교ID": "unstructured"})
    names = [row["name"] for row in r.get_table("fixed").rows]
    assert names == ["Intercept", "가정SES_cwc", "가정SES_학급ID평균", "성별", "학습시간", "학교유형_사립", "지역_읍면",
                     "지역_중소도시", "가정SES_cwc:학교유형_사립"]
    b, se = fixed(r)
    close(b, [51.93007296, 2.798097217, 6.156492663, 1.287138264, 0.5949146684, 1.966697796, -2.863837928,
              -0.651939256, 0.7554549457], 1e-5)
    close(se, [1.2175869, 0.3107574334, 0.718660769, 0.2389277493, 0.03014325257, 1.2896938, 1.607483719,
               1.265513062, 0.4849596658], 1e-5)
    assert abs(r.model_stats["LL"] - (-17099.44527)) < 1e-4
    v = vc(r)
    close([v[("학교ID", "var(가정SES_cwc)")][0], v[("학교ID", "var(_cons)")][0], v[("학교ID", "cov(가정SES_cwc,_cons)")][0]],
          [1.642076476, 15.67496675, 2.93514855], 1e-4, rel=True)
    assert any("가정SES_cwc" in n for n in r.get_table("info").notes)


def test_grand_mean_centering_and_numeric_interaction_reml_vs_lme4(school):
    r = run(school, **BASE, method="reml", centerGrand=["학습시간"], interactions=["가정SES:학습시간"])
    assert [row["name"] for row in r.get_table("fixed").rows][-1] == "가정SES:학습시간_cgm"
    b, se = fixed(r)
    close(b, [58.29939907, 3.262395032, 1.239659596, 0.5863969578, 0.0617793679], 1e-5)
    close(se, [0.7177228122, 0.15550913, 0.2406121534, 0.03080397184, 0.02787185956], 1e-5)
    assert abs(r.model_stats["LL"] - (-17141.61661)) < 1e-4


def test_exchangeable_and_identity_structures_vs_nlme(school):
    # nlme::lme(random = list(school = pdCompSymm(~ ses), class = ~ 1), method = "ML")
    r = run(school, **BASE, method="ml", randomSlopes={"학교ID": ["가정SES"]}, covariance={"학교ID": "exchangeable"})
    assert abs(r.model_stats["LL"] - (-17132.3751149362)) < 1e-6          # nlme stops at a looser tolerance
    b, se = fixed(r)
    close(b, [52.2237718036531, 3.2370292055218, 1.27442268937869, 0.59256932443841], 1e-4)
    close(se, [0.580952770143455, 0.44738563991685, 0.239435374239006, 0.0302225119987983], 1e-4)
    v = vc(r)
    close([v[("학교ID", "var(가정SES _cons)")][0], v[("학교ID", "cov(가정SES _cons)")][0], v[("학급ID", "var(_cons)")][0]],
          [8.73187731127735, 3.46924019601035, 8.05362223291564], 1e-4, rel=True)
    # pdIdent(~ ses), REML
    r = run(school, **BASE, method="reml", randomSlopes={"학교ID": ["가정SES"]}, covariance={"학교ID": "identity"})
    assert abs(r.model_stats["LL"] - (-17138.2082524184)) < 1e-6
    b, se = fixed(r)
    close(b, [52.2468796776725, 3.22857004343828, 1.27123965862078, 0.593297227240567], 1e-4)
    close(se, [0.592994286732811, 0.463244087798328, 0.239474088368579, 0.0302319400871988], 1e-4)
    v = vc(r)
    close([v[("학교ID", "var(가정SES _cons)")][0], v[("학급ID", "var(_cons)")][0]], [9.44914315207454, 7.93554446036131], 1e-4, rel=True)


# ---------------------------------------------------------------------------------------------------- Stata 19.5
# Reference values from StataNow 19.5 MP (`mixed ..., ltolerance(1e-13) nrtolerance(1e-11)`; `melogit ..., intmethod(laplace)`)
# run on the same data (ASCII variable names; typ = 학교유형 사립, reg 1/2/3 = 대도시/읍면/중소도시). Fixed effects in Stata order
# with Stata SEs; `phi` = e(b) entries of the variance part (ln SD, atanh correlation, ln σ_e; melogit: variances) with SEs.
STATA = {
    'm1_mle': dict(ll=-17137.90759, chi2_c=1379.971285, fixed=[('가정SES', 3.269845543, 0.1554828602), ('성별', 1.254300668, 0.2405494722), ('학습시간', 0.5983647716, 0.0303312592), ('Intercept', 52.26631435, 0.7726932952)],
        phi=[(1.544620146, 0.1111028836), (1.006426757, 0.07656543233), (2.103169575, 0.01044006423)]),
    'm1_reml': dict(ll=-17141.4105, chi2_c=1382.55319, fixed=[('가정SES', 3.268991891, 0.155549), ('성별', 1.254262834, 0.2406266733), ('학습시간', 0.5983744853, 0.03034097526), ('Intercept', 52.26603861, 0.7792050778)],
        phi=[(1.55588841, 0.1119837615), (1.006559315, 0.07657986901), (2.103489229, 0.01044332746)]),
    'm2_mle': dict(ll=-17111.03509, chi2_c=1433.716298, fixed=[('가정SES', 3.235370239, 0.2483530062), ('성별', 1.271561704, 0.2390433969), ('학습시간', 0.5932531815, 0.03015524681), ('Intercept', 52.19971643, 0.7278976505)],
        phi=[(0.3050534595, 0.1604721781), (1.460959475, 0.1150644152), (0.7673995909, 0.2411827224), (1.003291811, 0.07667526523), (2.094273684, 0.01050022366)]),
    'm2_reml': dict(ll=-17114.23806, chi2_c=1436.89807, fixed=[('가정SES', 3.233824181, 0.2509005826), ('성별', 1.271361949, 0.239105145), ('학습시간', 0.5932661001, 0.0301635922), ('Intercept', 52.20097344, 0.7341182504)],
        phi=[(0.3215017326, 0.1602782018), (1.472881862, 0.1158968935), (0.7562186605, 0.2389171276), (1.003360645, 0.07668775378), (2.094477881, 0.01050232965)]),
    'm3_mle': dict(ll=-17110.40325, chi2_c=1434.979976, fixed=[('가정SES', 3.221225213, 0.2501567756), ('성별', 1.275489466, 0.2385101411), ('학습시간', 0.5930134238, 0.03236176833), ('Intercept', 52.26799669, 0.7221967977)],
        phi=[(0.3152164349, 0.1593671281), (1.460966682, 0.116256951), (-1.751194911, 0.163119017), (0.8549069057, 0.113546164), (2.08777123, 0.01062971437)]),
    'm4_reml': dict(ll=-17092.25725, chi2_c=1480.859694, fixed=[('가정SES', 3.218165524, 0.2467382601), ('성별', 1.293424014, 0.2371290397), ('학습시간', 0.5980759727, 0.03910766856), ('Intercept', 52.14933267, 0.7750565665)],
        phi=[(0.2981773113, 0.163810504), (1.472907965, 0.1160612679), (0.7098352669, 0.237325689), (-0.9999363856, 0.1106943808), (1.522244139, 0.1010017998), (-1.107984627, 0.1373586098), (2.076419112, 0.01074828383)]),
    'm6_mle': dict(ll=-17180.17574, chi2_c=1295.434979, fixed=[('가정SES', 3.275864558, 0.2702834506), ('성별', 1.290222534, 0.240129126), ('학습시간', 0.5903579048, 0.03029459132), ('Intercept', 52.35065607, 0.4891849967)],
        phi=[(0.4378772181, 0.1445285972), (1.624354301, 0.05611034699), (2.094330695, 0.01049691319)]),
    'exch_mle': dict(ll=-17132.37514, chi2_c=1391.036195, fixed=[('가정SES', 3.237023181, 0.4473911488), ('성별', 1.274422974, 0.2394351845), ('학습시간', 0.5925692814, 0.03022248791), ('Intercept', 52.22377714, 0.5809585534)],
        phi=[(1.083504297, 0.09865992909), (0.420416497, 0.1708965517), (1.043086725, 0.07987274687), (2.093944585, 0.01049610046)]),
    'ident_reml': dict(ll=-17138.20827, chi2_c=1388.957648, fixed=[('가정SES', 3.228571062, 0.4632496655), ('성별', 1.271238736, 0.2394741437), ('학습시간', 0.5932973594, 0.03023194771), ('Intercept', 52.24687742, 0.592997296)],
        phi=[(1.122975664, 0.09233853905), (1.035653315, 0.07907225942), (2.094009343, 0.01049591612)]),
    'm7_mle': dict(ll=-17099.4453, chi2_c=1080.401796, fixed=[('가정SES_cwc', 2.798097129, 0.3107577943), ('가정SES_학급ID평균', 6.156494399, 0.7186622001), ('성별', 1.287138197, 0.2389277329), ('학습시간', 0.5949146555, 0.03014325051), ('학교유형_사립', 1.966696669, 1.289698973), ('지역_읍면', -2.863836052, 1.607489715), ('지역_중소도시', -0.6519400113, 1.265517814), ('가정SES_cwc:학교유형_사립', 0.7554550375, 0.4849602443), ('Intercept', 51.9300732, 1.217591304)],
        phi=[(0.2479828749, 0.1715888714), (1.376037059, 0.1160298529), (0.6602589346, 0.2426574198), (0.9603919719, 0.07819933943), (2.093965123, 0.01049707615)]),
    'm8_reml': dict(ll=-17141.61663, chi2_c=1377.233701, fixed=[('가정SES', 3.262394467, 0.1555091284), ('성별', 1.239659627, 0.240612138), ('학습시간_cgm', 0.5863969422, 0.03080396982), ('가정SES:학습시간_cgm', 0.0617793693, 0.02787185784), ('Intercept', 58.29939914, 0.7177252171)],
        phi=[(1.553946455, 0.1120813134), (1.008803924, 0.07647790591), (2.103025591, 0.01044452288)]),
    'four_mle': dict(ll=-2712.711412, chi2_c=1029.878522, fixed=[('x', 0.9974047719, 0.1496667506), ('w', 0.3362755765, 0.02828629417), ('Intercept', 10.59964675, 0.6517058242)],
        phi=[(-0.9401394386, 0.288180293), (0.4224875648, 0.3473104359), (0.02768092309, 0.4476170212), (0.569274512, 0.166181804), (-0.0242871617, 0.1172600256), (0.6553202841, 0.02084620079)]),
    'four_reml': dict(ll=-2715.806865, chi2_c=1034.026613, fixed=[('x', 0.9971201107, 0.1597054733), ('w', 0.3361823645, 0.02830033323), ('Intercept', 10.60713507, 0.6938884406)],
        phi=[(-0.8649649968, 0.3023771185), (0.5111140063, 0.3539142405), (0.02209851475, 0.4633085514), (0.5701599016, 0.1662623563), (-0.0252290297, 0.1173402472), (0.6557965068, 0.02085689103)]),
    'lg3_lap': dict(ll=-2089.587179, chi2_c=295.241104, fixed=[('가정SES', -0.9315545987, 0.05631643308), ('학습시간', -0.08281615024, 0.01067248441), ('지역_읍면', 0.7921429009, 0.3071682516), ('지역_중소도시', 0.269692823, 0.2506333648), ('Intercept', -0.9479542137, 0.2104247686)],
        phi=[(0.4783313889, 0.1223800372), (0.2002895783, 0.06292279942)]),
}
STATA_ICC = {
    'm1_mle': [[0.2274381218, 0.03938975115, 0.1594657759, 0.3135747742], [0.3049546693, 0.03604118533, 0.2391972029, 0.3797658958]],
    'm1_reml': [[0.2313153036, 0.0401678545, 0.1619669827, 0.3190516887], [0.3084168656, 0.03671859688, 0.2414152312, 0.3845864382]],
    'm2_mle': [[0.2020508395, 0.0374543711, 0.1383761107, 0.2853232395], [0.2829484283, 0.03436815115, 0.2206487773, 0.35482964]],
    'm2_reml': [[0.2058602821, 0.03824427952, 0.1408113103, 0.290788902], [0.2863520778, 0.03505181139, 0.2228127723, 0.359626449]],
    'four_reml': [[0.2629492695, 0.1426872226, 0.07772350972, 0.6016383064], [0.558858547, 0.08970100346, 0.3830566948, 0.7210459427], [0.6488104441, 0.07098281288, 0.5008088654, 0.7728357977]],
    'lg3_lap': [[0.120532368, 0.02732171727, 0.07638121736, 0.1850901839], [0.1710023513, 0.02791314028, 0.1229847868, 0.2327913304]],
}

STATA_SPECS = {
    "m1_mle": dict(BASE, method="ml"), "m1_reml": dict(BASE, method="reml"),
    "m2_mle": dict(BASE, method="ml", randomSlopes={"학교ID": ["가정SES"]}, covariance={"학교ID": "unstructured"}),
    "m2_reml": dict(BASE, method="reml", randomSlopes={"학교ID": ["가정SES"]}, covariance={"학교ID": "unstructured"}),
    "m3_mle": dict(BASE, method="ml", randomSlopes={"학교ID": ["가정SES"], "학급ID": ["학습시간"]}),
    "m4_reml": dict(BASE, method="reml", randomSlopes={"학교ID": ["가정SES"], "학급ID": ["학습시간"]}, covariance="unstructured"),
    "m6_mle": dict(BASE, method="ml", randomSlopes={"학교ID": ["가정SES"]}, noconstant=["학교ID"]),
    "exch_mle": dict(BASE, method="ml", randomSlopes={"학교ID": ["가정SES"]}, covariance={"학교ID": "exchangeable"}),
    "ident_reml": dict(BASE, method="reml", randomSlopes={"학교ID": ["가정SES"]}, covariance={"학교ID": "identity"}),
    "m7_mle": dict(BASE, factors=["학교유형", "지역"], interactions=["가정SES:학교유형"], centerGroup=["가정SES"],
                   groupMeans=["가정SES"], method="ml", randomSlopes={"학교ID": ["가정SES"]}, covariance={"학교ID": "unstructured"}),
    "m8_reml": dict(BASE, method="reml", centerGrand=["학습시간"], interactions=["가정SES:학습시간"]),
    "four_mle": dict(dep="y", levels=["a", "b", "c"], covs=["x", "w"], method="ml", randomSlopes={"a": ["x"]},
                     covariance={"a": "unstructured"}),
    "four_reml": dict(dep="y", levels=["a", "b", "c"], covs=["x", "w"], method="reml", randomSlopes={"a": ["x"]},
                      covariance={"a": "unstructured"}),
}
OWN_ENGINE = {"m4_reml", "exch_mle", "ident_reml"}        # structures statsmodels MixedLM does not fit


@pytest.mark.parametrize("tag", list(STATA_SPECS))
def test_mixed_vs_stata(school, tag):
    ref = STATA[tag]
    data = four_level_data() if tag.startswith("four") else school
    r = run(data, **STATA_SPECS[tag])
    assert r.engine == (LMM_ENGINE if tag in OWN_ENGINE else "statsmodels MixedLM")
    if tag in OWN_ENGINE:
        assert any("lme4 방식 프로파일 우도" in n for n in r.get_table("info").notes)
    fx = {row["name"]: row for row in r.get_table("fixed").rows}
    for name, b, se in ref["fixed"]:
        assert abs(fx[name]["est"] - b) < 5e-6 * max(1.0, abs(b)), (name, fx[name]["est"], b)
        assert abs(fx[name]["se"] / se - 1) < 2e-5, (name, fx[name]["se"], se)
    phi, C = r._fit["phi"], r._fit["phi_cov"]
    assert len(phi) == len(ref["phi"])
    for i, (b, se) in enumerate(ref["phi"]):
        assert abs(phi[i] - b) < 3e-5, (i, phi[i], b)
        assert abs(math.sqrt(C[i, i]) / se - 1) < 5e-5, (i, math.sqrt(C[i, i]), se)
    assert abs(r.model_stats["LL"] - ref["ll"]) < 5e-5            # Stata's LL is lower by about 4e-9 per observation
    assert abs(r.get_table("lrtest").rows[0]["chi2"] - ref["chi2_c"]) < 1e-3
    if tag in STATA_ICC:
        for row, (icc, se, lo, hi) in zip(r.get_table("icc").rows, STATA_ICC[tag]):
            assert abs(row["icc"] - icc) < 1e-5 and abs(row["se"] / se - 1) < 1e-4
            assert abs(row["cil"] - lo) < 1e-5 and abs(row["ciu"] - hi) < 1e-5


@pytest.mark.parametrize("kw,ll,chi2_c,fixed,var", [
    # melogit low ses || school:, intmethod(laplace)
    (dict(dep="기초학력미달", levels=["학교ID"], covs=["가정SES"], family="binomial"), -2134.83185412855, 347.757888848756,
     [("가정SES", -1.01213815935949, 0.0525998326354852), ("Intercept", -1.42419565114006, 0.114839000942463)],
     [(0.567097132554193, 0.130382344023103)]),
    # mepoisson absent ses female || school: || class:, intmethod(laplace)   (started from the mvaghermite estimates;
    # from Stata's default start the laplace fit stopped at LL = -8392.41 without convergence)
    (dict(dep="결석일수", levels=["학교ID", "학급ID"], covs=["가정SES", "성별"], family="poisson"), -8369.57387100259,
     1526.57835160788, [("가정SES", -0.269840660078933, 0.0120913367951287), ("성별", -0.292336910285726, 0.0196671488692987),
                        ("Intercept", 0.896948917846482, 0.0531538623471725)],
     [(0.116373943307587, 0.027362891711269), (0.0648252103325002, 0.00940475513284382)]),
    # mepoisson absent ses female, exposure(expo) || school:, intmethod(laplace)
    (dict(dep="결석일수", levels=["학교ID"], covs=["가정SES", "성별"], family="poisson", exposure="노출"), -9605.3832936924,
     1188.06907062095, [("가정SES", -0.270149779378737, 0.0117312685322582), ("성별", -0.300797531948532, 0.0193840968614706),
                        ("Intercept", 0.238429451406812, 0.051658776483852)], [(0.124607984122835, 0.026071654140486)]),
])
def test_glmm_laplace_vs_stata(school, kw, ll, chi2_c, fixed, var):
    d = school.dropna(subset=["학습시간"]).copy()
    d["노출"] = 1 + d["학생ID"] % 3
    r = run(d, **kw)
    assert abs(r.model_stats["LL"] - ll) < 1e-5
    assert abs(r.get_table("lrtest").rows[0]["chi2"] - chi2_c) < 1e-3
    fx = {row["name"]: row for row in r.get_table("fixed").rows}
    for name, b, se in fixed:
        assert abs(fx[name]["est"] - b) < 1e-4, (name, fx[name]["est"], b)
        assert abs(fx[name]["se"] / se - 1) < 1e-3, (name, fx[name]["se"], se)
    rows = [row for row in r.get_table("random").rows]
    for row, (v, se) in zip(rows, var):
        assert abs(row["est"] / v - 1) < 2e-4 and abs(row["se"] / se - 1) < 2e-3
    assert r.engine == GLMM_ENGINE


def test_melogit_laplace_vs_stata(school):
    """melogit low ses hours i.reg || school: || class:, intmethod(laplace)  (own Laplace engine)."""
    ref = STATA["lg3_lap"]
    d = school.dropna(subset=["학습시간"])
    r = run(d, dep="기초학력미달", levels=["학교ID", "학급ID"], covs=["가정SES", "학습시간"], factors=["지역"],
            family="binomial", expB=True)
    assert abs(r.model_stats["LL"] - ref["ll"]) < 1e-5
    fx = {row["name"]: row for row in r.get_table("fixed").rows}
    for name, b, se in ref["fixed"]:
        assert abs(fx[name]["est"] - b) < 1e-4, (name, fx[name]["est"], b)        # flat Laplace likelihood
        # SEs: exact Hessian of the Laplace log likelihood; Stata's laplace SEs differ by up to about 3.5 %
        assert abs(fx[name]["se"] / se - 1) < 0.04, (name, fx[name]["se"], se)
        assert abs(fx[name]["exp"] - math.exp(fx[name]["est"])) < 1e-12
    v = vc(r)
    close([v[("학교ID", "var(_cons)")][0], v[("학급ID", "var(_cons)")][0]], [x[0] for x in ref["phi"]], 1e-4, rel=True)
    assert abs(r.get_table("lrtest").rows[0]["chi2"] - ref["chi2_c"]) < 1e-3
    for row, (icc, se, lo, hi) in zip(r.get_table("icc").rows, STATA_ICC["lg3_lap"]):
        assert abs(row["icc"] - icc) < 1e-5
    assert r.title.startswith("다층 로지스틱")


@pytest.mark.parametrize("levels,extra,ll,beta,se,v", [
    # glmer(absent ~ ses + female + (1 | school/class), family = poisson, nAGQ = 1)
    (["학교ID", "학급ID"], {}, -8369.5738727086, [0.89692824778771, -0.269834542225321, -0.292278434965035],
     [0.0531358993965817, 0.0120441041013774, 0.019588737499637], [0.11637194456921, 0.0648252088727682]),
    # glmer(absent ~ ses + female + offset(log(expo)) + (1 | school), family = poisson), expo = 1 + (학생ID mod 3)
    (["학교ID"], {"exposure": "노출"}, -9605.38329378616, [0.238430616343296, -0.270149885059745, -0.300797514564995],
     [0.0516568525565416, 0.0117192717453185, 0.0193638015470769], [0.124622377005931]),
])
def test_mepoisson_laplace_vs_lme4(school, levels, extra, ll, beta, se, v):
    d = school.dropna(subset=["학습시간"]).copy()
    d["노출"] = 1 + d["학생ID"] % 3
    r = run(d, dep="결석일수", levels=levels, covs=["가정SES", "성별"], family="poisson", expB=True, **extra)
    assert abs(r.model_stats["LL"] - ll) < 2e-5
    b, s_ = fixed(r)
    close(b, beta, 1e-4)
    close(s_, se, 0.01, rel=True)                 # lme4: finite-difference Hessian of its deviance
    t = vc(r)
    close([t[(g, "var(_cons)")][0] for g in levels], v, 1e-3, rel=True)
    assert r.get_table("icc") is None                                    # no ICC after mepoisson (Stata)
    assert any("라플라스" in row["val"] for row in r.get_table("info").rows)


# ---------------------------------------------------------------------------------------------------- Stata metric
# Independent R computation: ML / REML log likelihood from per-school V blocks (beta profiled by GLS), numDeriv
# Hessian in (ln sd_school, ln sd_class, ln sigma) and (ln sd_cons, ln sd_ses, atanh rho, ln sd_class, ln sigma).
SEPHI = {
    "m1_ml": dict(phi=[1.54461664651629, 1.00642512687099, 2.10316964852103],
                  se=[0.111102615323704, 0.0765654346175457, 0.0104400652443138],
                  icc=[0.227436926245003, 0.304953356205865], se_icc=[0.0393895107409645, 0.0360409880579643]),
    "m1_reml": dict(phi=[1.55588426751056, 1.0065575532506, 2.10348931120202],
                    se=[0.111983443366369, 0.0765798745539955, 0.010443328492454],
                    icc=[0.231313866781135, 0.308415317034028], se_icc=[0.0401675672095497, 0.0367183593488011]),
    "m2_ml": dict(phi=[1.4609555624118, 0.305051111220575, 0.767400538503934, 1.0032899284487, 2.09427378178439],
                  se=[0.115064326916731, 0.16047210218318, 0.241182138401285, 0.0766752754520593, 0.0105002249454985],
                  c01=0.00365482745975959, c02=0.00434777158027938, c12=0.000348578270522611),
    "m2_reml": dict(phi=[1.47287810398389, 0.321494218532228, 0.756220402432914, 1.00335891044027, 2.09447800359778],
                    se=[0.115896822499017, 0.160278256663655, 0.238916797346567, 0.0766877590994199, 0.0105023312676866]),
}
Z95 = 1.959963984540054


@pytest.mark.parametrize("method", ["ml", "reml"])
def test_variance_component_se_and_ci_in_stata_metric(school, method):
    ref = SEPHI[f"m1_{method}"]
    r = run(school, **BASE, method=method)
    v = vc(r)
    for key, ph, s in zip([("학교ID", "var(_cons)"), ("학급ID", "var(_cons)"), ("잔차", "var(Residual)")], ref["phi"], ref["se"]):
        est, se, lo, hi = v[key]
        var = math.exp(2 * ph)
        assert abs(est / var - 1) < 1e-5
        assert abs(se / (2 * var * s) - 1) < 2e-4                                    # SE(var) = 2 var SE(ln sd)
        assert abs(lo / math.exp(2 * (ph - Z95 * s)) - 1) < 2e-4 and abs(hi / math.exp(2 * (ph + Z95 * s)) - 1) < 2e-4
    rs = run(school, **BASE, method=method, reStat="sd")
    est, se, lo, hi = vc(rs)[("학교ID", "sd(_cons)")]
    assert abs(se / (math.exp(ref["phi"][0]) * ref["se"][0]) - 1) < 2e-4
    assert abs(lo / math.exp(ref["phi"][0] - Z95 * ref["se"][0]) - 1) < 2e-4
    icc = r.get_table("icc").rows
    assert [row["level"] for row in icc] == ["학교ID", "학급ID|학교ID"]
    close([row["icc"] for row in icc], ref["icc"], 1e-5, rel=True)
    close([row["se"] for row in icc], ref["se_icc"], 2e-4, rel=True)
    for row, p, s in zip(icc, ref["icc"], ref["se_icc"]):                               # logit-scale interval
        lg, slg = math.log(p / (1 - p)), s / (p * (1 - p))
        assert abs(row["cil"] - 1 / (1 + math.exp(-(lg - Z95 * slg)))) < 1e-4
        assert abs(row["ciu"] - 1 / (1 + math.exp(-(lg + Z95 * slg)))) < 1e-4
    assert abs(r.model_stats["ICC"] - ref["icc"][1]) < 1e-5


@pytest.mark.parametrize("method", ["ml", "reml"])
def test_slope_variance_se_and_correlation_ci(school, method):
    ref = SEPHI[f"m2_{method}"]
    r = run(school, **BASE, method=method, randomSlopes={"학교ID": ["가정SES"]}, covariance={"학교ID": "unstructured"},
            reStat="both")
    rows, lev = {}, None
    for row in r.get_table("random").rows:
        lev = row["grp"].split(":")[0] if row["grp"] else lev
        rows[(lev, row["name"])] = row
    ph, s = ref["phi"], ref["se"]
    assert abs(rows[("학교ID", "_cons")]["sd_se"] / (math.exp(ph[0]) * s[0]) - 1) < 2e-4
    assert abs(rows[("학교ID", "가정SES")]["sd_se"] / (math.exp(ph[1]) * s[1]) - 1) < 2e-4
    assert abs(rows[("학급ID", "_cons")]["sd_se"] / (math.exp(ph[3]) * s[3]) - 1) < 2e-4
    assert abs(rows[("잔차", "Residual")]["sd_se"] / (math.exp(ph[4]) * s[4]) - 1) < 2e-4
    corr = rows[("학교ID", "가정SES,_cons")]
    rho = math.tanh(ph[2])
    assert abs(corr["sd_est"] - rho) < 1e-5
    assert abs(corr["sd_se"] / ((1 - rho * rho) * s[2]) - 1) < 2e-4
    assert abs(corr["sd_cil"] - math.tanh(ph[2] - Z95 * s[2])) < 1e-4 and abs(corr["sd_ciu"] - math.tanh(ph[2] + Z95 * s[2])) < 1e-4
    if method == "ml":   # covariance SE by the delta method from the full R covariance matrix of φ
        C = np.diag(np.square(s[:3]))
        C[0, 1] = C[1, 0] = ref["c01"]
        C[0, 2] = C[2, 0] = ref["c02"]
        C[1, 2] = C[2, 1] = ref["c12"]
        sd0, sd1 = math.exp(ph[0]), math.exp(ph[1])
        cov = rho * sd0 * sd1
        g = np.array([cov, cov, (1 - rho * rho) * sd0 * sd1])
        se_cov = math.sqrt(g @ C @ g)
        assert abs(corr["var_est"] - cov) < 1e-4
        assert abs(corr["var_se"] / se_cov - 1) < 5e-4
        assert abs(corr["var_cil"] - (corr["var_est"] - Z95 * corr["var_se"])) < 1e-9   # covariance: symmetric Wald


@pytest.mark.parametrize("method,ols_ll", [("ml", -17827.8932149857), ("reml", -17832.6870792558)])
def test_lr_test_against_linear_regression(school, method, ols_ll):
    r = run(school, **BASE, method=method)
    row = r.get_table("lrtest").rows[0]
    assert row["stat"] == "chi2(2)" and row["df"] == 2
    assert abs(row["chi2"] - 2 * (r.model_stats["LL"] - ols_ll)) < 1e-4
    r1 = run(school, dep="수학성취도", cluster="학교ID", covs=["가정SES"], method=method)
    row = r1.get_table("lrtest").rows[0]
    assert row["stat"] == "chibar2(01)"
    assert abs(row["p"] - 0.5 * __import__("scipy").stats.chi2.sf(row["chi2"], 1)) < 1e-15


def test_two_level_random_intercept_vs_lme4(school):
    # (1 | school) REML and ML on the same complete cases as the three-level models
    d = school.dropna(subset=["학습시간"])
    r = run(d, dep="수학성취도", cluster="학교ID", covs=["가정SES"], method="reml")
    assert abs(r.model_stats["LL"] - (-17423.658539)) < 1e-5
    r = run(d, dep="수학성취도", cluster="학교ID", covs=["가정SES"], method="ml")
    assert abs(r.model_stats["LL"] - (-17423.276576)) < 1e-5


# ---------------------------------------------------------------------------------------------------- BLUPs
def test_blups_and_conditional_variances_vs_lme4(school):
    r = run(school, **BASE, method="ml", randomSlopes={"학교ID": ["가정SES"]}, covariance={"학교ID": "unstructured"},
            blups=True, caterpillar=True, residPlot=True)
    t = r.get_table("blups")
    rows = {(row["level"], row["group"], row["term"]): row for row in t.rows}
    close([rows[("학교ID", str(g), "_cons")]["blup"] for g in range(1, 6)],
          [8.40392909499867, -6.7771127318701, -4.50094851155437, -0.945102905663328, -2.82861857303159], 1e-5)
    close([rows[("학교ID", str(g), "가정SES")]["blup"] for g in range(1, 6)],
          [2.79778468806248, -0.439636924902602, -1.11139363784439, -0.114280174873712, -0.0441362769477709], 1e-5)
    close([rows[("학교ID", "1", "_cons")]["se"] ** 2, rows[("학교ID", "1", "가정SES")]["se"] ** 2],
          [1.48975176848655, 0.363392137746391], 1e-5, rel=True)
    close([rows[("학급ID", f"1/{c}", "_cons")]["blup"] for c in range(1, 6)],
          [-0.0154932756559633, -1.21188437791717, 3.3973885201703, 1.09593481007012, -1.85424016318229], 1e-5)
    close([rows[("학급ID", f"1/{c}", "_cons")]["se"] ** 2 for c in range(1, 6)],
          [2.69535912343859, 2.68671519967941, 3.22867009526693, 2.61297589435936, 2.88151518967969], 1e-5, rel=True)
    assert len(t.rows) == 50 * 2 + 224
    assert len(r.plots) == 3                                  # two caterpillar plots + residual plot


# ---------------------------------------------------------------------------------------------------- 4 levels
def four_level_data():
    """Synthetic 4-level data (8 regions > districts > schools > students), same generator as the R check."""
    rng = np.random.default_rng(7)
    rows = []
    for a in range(1, 9):
        ua = rng.normal(0, 2.0)
        sa = rng.normal(0, 0.5)
        for b in range(1, rng.integers(3, 6) + 1):
            ub = rng.normal(0, 1.5)
            for c in range(1, rng.integers(2, 5) + 1):
                uc = rng.normal(0, 1.0)
                m = rng.integers(8, 20)
                x = rng.normal(0, 1, m)
                w = rng.normal(5, 2, m)
                y = 10 + 1.0 * x + 0.3 * w + ua + sa * x + ub + uc + rng.normal(0, 2.0, m)
                for i in range(m):
                    rows.append(dict(a=a, b=b, c=c, x=round(x[i], 4), w=round(w[i], 4), y=round(y[i], 4)))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("method,ll,beta,se,vcs", [
    ("ml", -2712.711408, [10.59964659, 0.9974048142, 0.3362755794], [0.6517031173, 0.1496662904, 0.02828629683],
     [0.1525464542, 2.327895779, 0.01649120165, 3.122218994, 0.9525861716, 3.708549279]),
    ("reml", -2715.80686, [10.60713491, 0.9971201537, 0.3361823672], [0.6938852611, 0.1597049803, 0.02830033591],
     [0.17729556, 2.779349341, 0.01551017967, 3.127752007, 0.9507933247, 3.712083159]),
])
def test_four_level_model_vs_lme4(method, ll, beta, se, vcs):
    # (1 + x | a) + (1 | a:b) + (1 | a:b:c)
    d = four_level_data()
    assert len(d) == 1252
    r = run(d, dep="y", levels=["a", "b", "c"], covs=["x", "w"], method=method, randomSlopes={"a": ["x"]},
            covariance={"a": "unstructured"})
    b, s = fixed(r)
    close(b, beta, 1e-5)
    close(s, se, 1e-5)
    assert abs(r.model_stats["LL"] - ll) < 1e-4
    v = vc(r)
    close([v[("a", "var(x)")][0], v[("a", "var(_cons)")][0], v[("a", "cov(x,_cons)")][0], v[("b", "var(_cons)")][0],
           v[("c", "var(_cons)")][0], v[("잔차", "var(Residual)")][0]], vcs, 1e-4, rel=True)
    assert [row["groups"] for row in r.get_table("groups").rows] == [8, 32, 95]


def test_four_level_blups_with_slopes_at_two_levels_vs_lme4():
    # (1 + x | a) + (1 + w | a:b) + (1 | a:b:c), ML: conditional modes and variances through three tree levels
    d = four_level_data()
    r = run(d, dep="y", levels=["a", "b", "c"], covs=["x", "w"], method="ml", randomSlopes={"a": ["x"], "b": ["w"]},
            covariance={"a": "unstructured", "b": "unstructured"}, blups=True)
    assert abs(r.model_stats["LL"] - (-2712.50322682351)) < 1e-4
    rows = {(row["level"], row["group"], row["term"]): row for row in r.get_table("blups").rows}
    close([rows[("c", g, "_cons")]["blup"] for g in ("1/1/1", "1/1/2", "1/1/3", "1/1/4", "1/2/1", "1/2/2")],
          [-0.477974663517542, 0.298347394841573, -0.032111528954366, -0.0877070469471704, -0.464191123637912,
           0.155042011444206], 1e-5)
    close([rows[("c", g, "_cons")]["se"] ** 2 for g in ("1/1/1", "1/1/2", "1/1/3", "1/1/4", "1/2/1", "1/2/2")],
          [0.41948126930888, 0.37440876605966, 0.447260440196941, 0.449852299200231, 0.523999398035231,
           0.519681888065831], 1e-5, rel=True)
    close([rows[("b", "1/1", "_cons")]["blup"], rows[("b", "1/1", "w")]["blup"], rows[("a", "1", "x")]["blup"]],
          [-0.8942760255507, 0.00309126702886191, 0.235296093735481], 1e-5)
    close([rows[("b", "1/1", "w")]["se"] ** 2, rows[("a", "1", "_cons")]["se"] ** 2],
          [0.00216830112916895, 0.652250196550488], 1e-5, rel=True)


# ---------------------------------------------------------------------------------------------------- compat
def test_old_two_level_options_and_report_format():
    gr = pd.read_csv(os.path.join(ROOT, "sample_data", "Grunfeld_Panel.csv"))
    r = dispatch_analysis("mixedModel", gr, {"dep": "invest", "cluster": "firm", "covs": ["value", "capital"]})
    assert isinstance(r, ml.MultilevelLinearModel)
    assert r.options["method"] == "reml"                       # calls without the option keep the old REML default
    assert r.get_table("random") is not None and r.get_table("info") is not None
    assert r.get_table("fixed").kind == "coefficients"
    b, se = fixed(r)
    close(b, [-54.03182044, 0.1093519794, 0.3081997216], 1e-5)
    close(se, [26.62354925, 0.009986680188, 0.01636768874], 1e-5)
    assert abs(r.model_stats["LL"] - (-1197.308054)) < 1e-5
    v = vc(r)
    close([v[("firm", "var(_cons)")][0], v[("잔차", "var(Residual)")][0]], [6740.897234, 2527.551604], 1e-5, rel=True)
    assert {"N", "Groups", "LL", "AIC", "BIC", "ICC", "Var_between", "Var_within", "Chi2"} <= set(r.model_stats)
    assert abs(r.model_stats["ICC"] - 6740.897234 / (6740.897234 + 2527.551604)) < 1e-6
    r_ml = dispatch_analysis("mixedModel", gr, {"dep": "invest", "cluster": "firm", "covs": ["value"], "reml": False})
    assert r_ml.options["method"] == "ml"
    # statsmodels MixedLM (two-level cross-check)
    import statsmodels.api as sm
    X = sm.add_constant(gr[["value", "capital"]])
    smr = sm.MixedLM(gr["invest"], X, groups=gr["firm"]).fit(reml=True)
    close(b, smr.fe_params.to_numpy(), 1e-4, rel=True)
    assert abs(r.model_stats["LL"] - smr.llf) < 1e-3
    from pyjamovi.analyses.model_compare import ModelComparison  # noqa: F401  (comparison uses the fixed table)
    models = r.comparison_models()
    assert models[0]["coefs"][0]["term"] == "Intercept" and models[0]["stats"]["N"] == 220


def test_session_run_and_aliases(school):
    import pyjamovi as ost
    sess = ost.Session(school, auto_sync_workspace=False)
    r = sess.run("mixed", dep="수학성취도", levels=["학교ID", "학급ID"], covs=["가정SES"], method="ml")
    assert isinstance(r, ml.MultilevelLinearModel) and r.options["levels"] == ["학교ID", "학급ID"]
    assert r.syntax_python.startswith("res = sess.run('mixedModel', dep='수학성취도', levels=['학교ID', '학급ID']")
    assert "lme4::lmer" in r.syntax_r and "(1 | `학교ID`:`학급ID`)" in r.syntax_r


# ---------------------------------------------------------------------------------------------------- validation
def test_option_validation_messages(school):
    with pytest.raises(ValueError, match="가능: independent, unstructured, identity, exchangeable"):
        run(school, **BASE, randomSlopes={"학교ID": ["가정SES"]}, covariance={"학교ID": "toeplitz"})
    with pytest.raises(ValueError, match="가능: reml, ml"):
        run(school, **BASE, method="gls")
    with pytest.raises(ValueError, match="자료에 없는 변수"):
        run(school, dep="수학성취도", levels=["학교ID", "반"], covs=["가정SES"])
    with pytest.raises(ValueError, match="숫자형"):
        run(school, **BASE, randomSlopes={"학교ID": ["학교유형"]})
    with pytest.raises(ValueError, match="집단 변수가 아님"):
        run(school, **BASE, randomSlopes={"지역": ["가정SES"]})
    with pytest.raises(ValueError, match="절편을 제외"):
        run(school, **BASE, noconstant=["학교ID"])
    with pytest.raises(ValueError, match="3수준 집단 변수"):
        run(school, dep="수학성취도", cluster4="지역", cluster="학급ID", covs=["가정SES"])
    with pytest.raises(ValueError, match="최대 3개"):
        run(school, dep="수학성취도", levels=["지역", "학교ID", "학급ID", "학생ID"], covs=["가정SES"])
    with pytest.raises(ValueError, match="reStat"):
        run(school, **BASE, reStat="cor")
    with pytest.raises(ValueError, match="한 값뿐"):
        run(school, **BASE, family="binomial")                # 수학성취도 > 0 everywhere: a single outcome value
    with pytest.raises(ValueError, match="가능: gaussian, binomial, poisson"):
        run(school, **BASE, family="gamma")
    with pytest.raises(ValueError, match="offset"):
        run(school, **BASE, exposure="학급규모")
    r = run(school, dep="수학성취도", cluster="학급ID", cluster3="학교ID", covs=["가정SES"], slopes4=["가정SES"])
    assert any("slopes4" in n for n in r.get_table("info").notes)            # stale UI option: noted, not applied


# ---------------------------------------------------------------------------------------------------- Stata / R code
def test_stata_transpile():
    from pyjamovi.stata_transpiler import transpile_stata_line
    line = transpile_stata_line("mixed y x1 x2 || school: x1, cov(un) || class:, reml")
    assert line == ("res = sess.run('mixedModel', dep='y', levels=['school', 'class'], covs=['x1', 'x2'], "
                    "randomSlopes={'school': ['x1']}, covariance={'school': 'unstructured'}, method='reml')")
    line = transpile_stata_line("mixed gsp private emp hwy water other unemp || region: || state:")
    assert line.endswith("method='ml')") and "levels=['region', 'state']" in line          # Stata default ML
    line = transpile_stata_line("mixed y c.ses##i.sector female || school: ses female, covariance(exchangeable) nocons "
                                "|| class:, stddev level(90)")
    assert "interactions=['ses*sector']" in line and "factors=['sector']" in line and "noconstant=['school']" in line
    assert "reStat='sd'" in line and "ciWidth=90.0" in line and "covariance={'school': 'exchangeable'}" in line
    assert "level" not in transpile_stata_line("xtmixed y x || id:, mle").split("levels")[0]
    for bad in ("mixed y x || id: x, vce(robust)", "mixed y x if z>1 || id:", "mixed y x [pw=w] || id:",
                "mixed y x || _all: R.g", "mixed y x || id: x, dfmethod(kroger) reml", "mixed y x, noconstant || id:",
                "mixed y x || a: || b: || c: || d:", "estat wcorrelation"):
        assert transpile_stata_line(bad).startswith("raise NotImplementedError"), bad
    assert transpile_stata_line("estat icc").startswith("# [Stata] estat icc")
    line = transpile_stata_line("melogit low ses i.reg || school: || class:, intmethod(laplace) or")
    assert line == ("res = sess.run('mixedModel', dep='low', levels=['school', 'class'], covs=['ses'], factors=['reg'], "
                    "family='binomial', expB=True)")
    line = transpile_stata_line("mepoisson absent ses || school:, irr exposure(expo)")
    assert "family='poisson'" in line and "exposure='expo'" in line and "# 라플라스 근사" in line   # Stata default: mvaghermite
    for bad in ("melogit y x || id:, intpoints(12)", "melogit y x || id:, reml", "mepoisson y x || id:, intmethod(ghermite)"):
        assert transpile_stata_line(bad).startswith("raise NotImplementedError"), bad


def test_stata_and_r_code_export():
    p = {"dep": "수학성취도", "cluster": "학급ID", "cluster3": "학교ID", "covs": ["가정SES", "성별"], "factors": ["학교유형"],
         "interactions": ["가정SES:학교유형"], "method": "ml", "slopes3": ["가정SES"], "covStruct3": "unstructured",
         "centerGroup": ["가정SES"], "groupMeans": ["가정SES"]}
    st = ml.stata_code("mixedModel", p)
    assert "egen double __m_가정SES = mean(가정SES) if __touse, by(학교ID 학급ID)" in st
    assert ("mixed 수학성취도 가정SES_cwc 가정SES_학급ID평균 성별 i.학교유형 c.가정SES_cwc#i.학교유형 if __touse "
            "|| 학교ID: 가정SES_cwc, covariance(unstructured) || 학급ID:, mle") in st
    rr = ml.r_code("mixedModel", p)
    assert any(x.startswith("fit <- lme4::lmer(`수학성취도` ~ `가정SES_cwc` + `가정SES_학급ID평균` + `성별` + factor(`학교유형`) + "
                            "`가정SES_cwc`:factor(`학교유형`) + (1 + `가정SES_cwc` | `학교ID`) + (1 | `학교ID`:`학급ID`)") for x in rr)
    old = {"dep": "invest", "cluster": "firm", "covs": ["value"], "reml": True}
    assert ml.stata_code("mixedModel", old) == ["mixed invest value || firm:, reml", "estat icc", "estat ic"]
    assert ml.r_code("mixedModel", old)[1] == "fit <- lme4::lmer(invest ~ value + (1 | firm), data = data, REML = TRUE)"
    ind = ml.r_code("mixedModel", {"dep": "y", "levels": ["s", "c"], "covs": ["x"], "randomSlopes": {"c": ["x"]}, "method": "ml"})
    assert "(1 + x || s:c)" in ind[1]
    ex = ml.r_code("mixedModel", {"dep": "y", "levels": ["s"], "covs": ["x"], "randomSlopes": {"s": ["x"]},
                                  "covariance": "exchangeable"})
    assert "nlme::lme(y ~ x, random = list(s = nlme::pdCompSymm(~ x))" in ex[1]
    gl = {"dep": "기초학력미달", "levels": ["학교ID", "학급ID"], "covs": ["가정SES"], "family": "binomial", "expB": True}
    assert ml.stata_code("mixedModel", gl)[0] == "melogit 기초학력미달 가정SES || 학교ID: || 학급ID:, intmethod(laplace) or"
    assert ml.r_code("mixedModel", gl)[1].startswith("fit <- lme4::glmer(`기초학력미달` ~ `가정SES` + (1 | `학교ID`) + "
                                                      "(1 | `학교ID`:`학급ID`), data = data, family = binomial, nAGQ = 1)")
    from pyjamovi.analyses import extensions
    assert "mixedModel" in extensions.ids() and extensions.aliases()["xtmixed"] == "mixedModel"
    assert extensions.stata_code("mixedModel", old) == ml.stata_code("mixedModel", old)


# ---------------------------------------------------------------------------------------------------- speed / Produc
def test_large_three_level_model_with_slopes_is_fast_and_matches_lme4():
    """20,000 students, 2,000 classes in 100 schools, correlated random slopes at both levels (REML)."""
    rng = np.random.default_rng(1)
    S, C, M = 100, 20, 10
    sch = np.repeat(np.arange(S), C * M)
    cls = np.tile(np.repeat(np.arange(C), M), S)
    x = rng.normal(size=S * C * M)
    w = rng.normal(size=S * C * M)
    z = np.repeat(rng.normal(size=S), C * M)
    u0, u1 = rng.normal(0, 2, S)[sch], rng.normal(0, .5, S)[sch]
    v0, v1 = rng.normal(0, 1, S * C)[sch * C + cls], rng.normal(0, .3, S * C)[sch * C + cls]
    y = 1 + x + .5 * w + .3 * z + u0 + u1 * x + v0 + v1 * x + rng.normal(0, 2, S * C * M)
    d = pd.DataFrame(dict(y=y, x=x, w=w, z=z, s=sch, c=cls))
    t0 = time.time()
    r = run(d, dep="y", cluster="c", cluster3="s", covs=["x", "w", "z"], method="reml", slopes3=["x"],
            covStruct3="unstructured", slopes2=["x"], covStruct2="unstructured")
    assert time.time() - t0 < 20.0
    assert abs(r.model_stats["LL"] - (-43976.8427868621)) < 1e-4
    b, se = fixed(r)
    close(b, [0.960738319662947, 0.967623583820685, 0.506583853944575, 0.469220033210459], 1e-5)
    close(se, [0.201197304127288, 0.0506386745231933, 0.0149150980357457, 0.189020382765972], 1e-5)
    v = vc(r)
    close([v[("s", "var(x)")][0], v[("s", "var(_cons)")][0], v[("s", "cov(x,_cons)")][0], v[("c", "var(x)")][0],
           v[("c", "var(_cons)")][0], v[("c", "cov(x,_cons)")][0]],
          [0.229480710201101, 3.84103911611875, 0.00946160732326451, 0.0860288641914721, 0.987091049248841,
           -0.00448777426866038], 1e-4)


PRODUC = os.environ.get("OPENSTAT_PRODUC_CSV", "")


@pytest.mark.skipif(not os.path.exists(PRODUC), reason="plm::Produc (log variables) not available; set OPENSTAT_PRODUC_CSV")
@pytest.mark.parametrize("method,ll,beta,se,v", [
    ("ml", 1430.501593, [2.128823948, 0.2671484914, 0.7540720185, 0.07097658173, 0.07611880792, -0.09999559844, -0.005898290776],
     [0.1543853753, 0.02125908258, 0.02618682481, 0.02304099916, 0.01392483289, 0.01693660909, 0.00090313163],
     [0.001450609736, 0.006275699486, 0.001346108148]),
    ("reml", 1404.710042, [2.126995777, 0.266030901, 0.7555059029, 0.07188553231, 0.07615527913, -0.1005396814, -0.005881511991],
     [0.1574864077, 0.02154708543, 0.02645560561, 0.02334775868, 0.01399522784, 0.01701732183, 0.0009092504308],
     [0.001896355265, 0.006443874012, 0.001354299907]),
])
def test_produc_year_in_state_in_region_vs_lme4(method, ll, beta, se, v):
    """mixed gsp private emp hwy water other unemp || region: || state:  (log variables of plm::Produc)."""
    P = pd.read_csv(PRODUC)
    r = run(P, dep="gsp", levels=["region", "state"], covs=["private", "emp", "hwy", "water", "other", "unemp"], method=method)
    b, s = fixed(r)
    close(b, beta, 1e-5)
    close(s, se, 1e-5)
    assert abs(r.model_stats["LL"] - ll) < 1e-4
    t = vc(r)
    # flat likelihood in the region variance (9 regions): LL agrees to 1e-7, the variance to about 1e-4
    close([t[("region", "var(_cons)")][0], t[("state", "var(_cons)")][0], t[("잔차", "var(Residual)")][0]], v, 2e-4, rel=True)
    if method == "ml":   # Stata 19.5: webuse productivity, mixed ... || region: || state:, mle (e(b) variance metric)
        stata_phi = [(-3.26788293156773, 0.447898518963545), (-2.53553505364654, 0.118483057680315),
                     (-3.30526898313365, 0.0255920557591044)]
        phi, C = r._fit["phi"], r._fit["phi_cov"]
        for i, (b, se_) in enumerate(stata_phi):
            assert abs(phi[i] - b) < 2e-5 and abs(math.sqrt(C[i, i]) / se_ - 1) < 1e-4
        icc = r.get_table("icc").rows
        close([icc[0]["icc"], icc[1]["icc"]], [0.159893018751417, 0.85162648333694], 1e-5)

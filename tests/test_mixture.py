"""
Latent class / profile (LCA, LPA), latent transition (LTA) and latent class growth (LCGA) analyses:
recovery of known structures, agreement with scikit-learn, standard errors, three-step auxiliary models,
TECH11 / TECH14 machinery and option validation.
"""

import importlib.util
import itertools
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reconstructed"))

from pyjamovi.analyses import mixture_models as mm  # noqa: E402
from pyjamovi.analyses.mixture_module import (  # noqa: E402
    LatentClassAnalysis, LatentClassGrowthAnalysis, LatentTransitionAnalysis,
)

_spec = importlib.util.spec_from_file_location("mk_mixture", os.path.join(ROOT, "sample_data", "make_mixture_sample_data.py"))
mk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mk)

CATS = mk.LCA_BINARY + [mk.LCA_ORDINAL]


@pytest.fixture(scope="module")
def lca_data():
    return mk.make_lca()


def _table(an, name):
    for t in an.tables:
        if t.name == name:
            return t.to_dataframe()
    raise KeyError(name)


def _best_perm(est, truth):
    K = truth.shape[0]
    return min(itertools.permutations(range(K)), key=lambda pm: np.abs(est[list(pm)] - truth).sum())


def _ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(a, b)


def test_lca_enumeration_and_recovery(lca_data):
    df, tr = lca_data
    a = LatentClassAnalysis(df, categorical=CATS, classes=3, classRange="1-4")
    enum = _table(a, "enumeration")
    assert int(enum.loc[enum["BIC"].idxmin(), "계층 수"]) == 3
    assert (enum["최우 LL 반복"].iloc[1:] == "예").all()
    mlc = a.posteriors["MLC"].to_numpy()
    assert _ari(tr["cls"], mlc - 1) > 0.55
    # item probabilities P(=1) of the binary items, matched to the generating classes
    probs = _table(a, "catProbs")
    rows = [probs[(probs["지표"] == it) & (probs["범주"] == "1")].iloc[0] for it in mk.LCA_BINARY]
    est = np.array([[r.iloc[2 + 2 * k] for r in rows] for k in range(3)])
    se = np.array([[r.iloc[3 + 2 * k] for r in rows] for k in range(3)])
    pm = list(_best_perm(est, tr["pbin"]))
    assert (np.abs(est[pm] - tr["pbin"]) < 3.5 * se[pm]).all()  # within sampling error (MLR SE)
    assert a.posteriors.shape == (len(df), 4)
    assert np.allclose(a.posteriors[["P_1", "P_2", "P_3"]].sum(1), 1)


def test_lpa_matches_sklearn_gaussian_mixture(lca_data):
    from sklearn.mixture import GaussianMixture
    df, _ = lca_data
    Y = df[mk.LCA_CONT].to_numpy(float)
    Y = Y[np.isfinite(Y).all(1)]
    fit = mm.fit_model(mm.LatentClassModel(mm.Measurement(None, [], Y, "free", "zero"), 3))
    g = GaussianMixture(3, covariance_type="diag", weights_init=fit.par["p"], means_init=fit.par["meas"]["mu"],
                        precisions_init=1 / fit.par["meas"]["var"], tol=1e-10, max_iter=5000, reg_covar=0).fit(Y)
    assert abs(g.score(Y) * len(Y) - fit.ll) < 1e-3
    g2 = GaussianMixture(3, covariance_type="diag", n_init=5, random_state=0, tol=1e-10, max_iter=5000, reg_covar=0).fit(Y)
    assert fit.ll >= g2.score(Y) * len(Y) - 1e-2
    # Mplus default (equal variances) is the generating model and has the lower BIC
    eq = mm.fit_model(mm.LatentClassModel(mm.Measurement(None, [], Y, "equal", "zero"), 3))
    assert eq.bic < fit.bic


def test_mlr_close_to_ml_on_well_specified_model(lca_data):
    df, _ = lca_data
    Y = df[mk.LCA_CONT].to_numpy(float)
    fit = mm.fit_model(mm.LatentClassModel(mm.Measurement(None, [], Y, "equal", "zero"), 3))
    fn = lambda th: fit.model.unpack(th)["meas"]["mu"].ravel()
    se_mlr = mm.delta(fn, mm.inference(fit.model, fit.par, "mlr"))[1]
    se_ml = mm.delta(fn, mm.inference(fit.model, fit.par, "ml"))[1]
    ratio = se_mlr / se_ml
    assert 0.9 < np.median(ratio) < 1.1 and ratio.min() > 0.75 and ratio.max() < 1.3


def test_analytic_scores_match_numeric(lca_data):
    df, _ = lca_data
    U = np.column_stack([df[c].map(lambda v: -1 if v != v else int(v)).to_numpy() for c in CATS])
    meas = mm.Measurement(U, [2] * 6 + [3], df[mk.LCA_CONT].to_numpy(float), "free", "zero")
    fit = mm.fit_model(mm.LatentClassModel(meas, 3), starts=4, final=2)
    inf = mm.inference(fit.model, fit.par)
    S_num = mm.numeric_case_scores(fit.model.case_ll, inf.theta)
    assert np.abs(inf.scores - S_num).max() < 1e-5


def test_three_step_auxiliary_recovery(lca_data):
    df, tr = lca_data
    a = LatentClassAnalysis(df, continuous=mk.LCA_CONT, classes=3, covariates=["연령", "독거"],
                            distalContinuous=["월의료비"], distalCategorical=["시설입소의향"])
    r3 = _table(a, "r3step")
    slope = r3[(r3["비교"] == "1계층 대 3계층") & (r3["변수"] == "연령")]["추정치"].iloc[0]
    assert abs(slope - tr["gamma"][0, 1] / 8.0) < 0.03  # generating slope is per 8 years of age
    bch = _table(a, "bchDistal")
    assert np.abs(bch["평균"].to_numpy() - tr["dist_mu"]).max() < 2.5
    tests = _table(a, "bchTests")
    assert tests.loc[tests["비교"] == "전체", "p"].iloc[0] < 0.001
    dc = _table(a, "dcatDistal")
    est = np.array([dc.iloc[:, 2 + 2 * k].to_numpy() for k in range(3)])
    assert np.abs(est - tr["dist_cat"]).max() < 0.1


def test_lta_recovery_long_format():
    df, tr = mk.make_lta()
    a = LatentTransitionAnalysis(df, categorical=mk.LTA_ITEMS, idVar="ID", timeVar="조사차수", classes=3)
    trans = _table(a, "transitions")
    est = trans.iloc[:, [2, 4, 6]].to_numpy().reshape(2, 3, 3)
    assert np.abs(est - tr["tau"]).max() < 0.12
    probs = _table(a, "catProbs")
    ones = probs[probs["범주"] == "1"]
    P = np.array([ones.iloc[:, 2 + 2 * k].to_numpy() for k in range(3)])
    assert np.abs(P - tr["P"]).max() < 0.08
    prev = _table(a, "prevalence")
    assert abs(prev.iloc[0, 1] / 100 - tr["pi"][0]) < 0.05
    assert a.posteriors["MLC"].notna().all()
    assert _ari(tr["cls"][df["ID"].to_numpy() - 1, df["조사차수"].to_numpy() - 1], a.posteriors["MLC"].to_numpy()) > 0.5


def test_lcga_enumeration_growth_and_covariate():
    df, tr = mk.make_lcga()
    a = LatentClassGrowthAnalysis(df, dep="삶의만족도", idVar="ID", timeVar="연도", classes=3, classRange=[1, 4],
                                  covariates=["소득분위"])
    enum = _table(a, "enumeration")
    assert int(enum.loc[enum["BIC"].idxmin(), "계층 수"]) == 3
    gr = _table(a, "growth")
    beta = gr["추정치"].to_numpy().reshape(3, 2)
    pm = _best_perm(beta, tr["beta"])
    assert np.abs(beta[list(pm)][:, 0] - tr["beta"][:, 0]).max() < 0.25
    assert np.abs(beta[list(pm)][:, 1] - tr["beta"][:, 1]).max() < 0.1
    res = _table(a, "residuals")
    assert np.abs(np.sqrt(res["추정치"].to_numpy()) - tr["rsd"]).max() < 0.12
    r3 = _table(a, "r3step")
    slope = r3[(r3["비교"] == "1계층 대 3계층") & (r3["변수"] == "소득분위")]["추정치"].iloc[0]
    assert abs(slope - tr["eta_slope"][0]) < 0.2
    assert a.posteriors.shape[0] == len(df)
    has_y = df.groupby("ID")["삶의만족도"].transform(lambda s: s.notna().any())
    assert (a.posteriors["MLC"].notna() == has_y).all()  # subjects without any observed outcome are excluded


def test_vlmr_weights_reduce_to_chi_square_for_regular_nested_models():
    # f: N(mu, sigma^2), g: N(0, sigma^2), data from g -> 2LR ~ chi2(1): weights {1, 0, 0}
    rng = np.random.default_rng(3)
    y = rng.standard_normal(4000) * 1.5
    s0 = float(np.mean(y ** 2))
    mu1, s1 = float(y.mean()), float(y.var())
    S1 = np.column_stack([(y - mu1) / s1, 0.5 * ((y - mu1) ** 2 / s1 - 1)])
    H1 = np.array([[-len(y) / s1, 0], [0, -len(y) / 2]])
    S0 = (0.5 * (y ** 2 / s0 - 1))[:, None]
    H0 = np.array([[-len(y) / 2]])
    inf1 = mm.Inference(np.zeros(2), np.zeros((2, 2)), np.ones(2, bool), S1, H1, False, 0)
    inf0 = mm.Inference(np.zeros(1), np.zeros((1, 1)), np.ones(1, bool), S0, H0, False, 0)
    ll1 = float(np.sum(-0.5 * (np.log(2 * np.pi * s1) + (y - mu1) ** 2 / s1)))
    ll0 = float(np.sum(-0.5 * (np.log(2 * np.pi * s0) + y ** 2 / s0)))
    for method in ("vuong", "mplus"):
        r = mm.vlmr_test(inf1, inf0, ll1, ll0, len(y), method)
        lam = np.sort(r["weights"])[::-1]
        assert abs(lam[0] - 1) < 0.1 and np.abs(lam[1:]).max() < 0.1
        from scipy.stats import chi2
        assert abs(r["p"] - chi2.sf(r["lr"], 1)) < 0.03


def test_imhof_matches_monte_carlo():
    rng = np.random.default_rng(5)
    lam = np.array([2.0, 0.5, -0.3, 0.1])
    Q = (lam[None] * rng.standard_normal((300000, 4)) ** 2).sum(1)
    for x in (0.5, 3.0, 8.0):
        assert abs(mm.imhof_upper(x, lam) - (Q > x).mean()) < 0.005


def test_tech11_tech14_and_enumeration_columns(lca_data):
    df, _ = lca_data
    sub = df.iloc[:300]
    a = LatentClassAnalysis(sub, categorical=mk.LCA_BINARY, classes=2, classRange=[1, 2], tech11=True, tech14=True, blrtDraws=3)
    lrt = _table(a, "lrtTests")
    assert lrt.shape[0] == 1
    vals = lrt.iloc[0]
    for col in lrt.columns:
        if col == "p":
            assert ((vals[col] >= 0) & (vals[col] <= 1)).all()
    enum = _table(a, "enumeration")
    assert {"VLMR p", "조정 LMR p", "BLRT p"} <= set(enum.columns)
    assert 0.0 <= enum["BLRT p"].iloc[1] <= 1.0


def test_option_validation(lca_data):
    df, _ = lca_data
    bad = [dict(categorical=CATS, estimator="wls"), dict(continuous=mk.LCA_CONT, variances="free", covariances="equal"),
           dict(categorical=["없는변수"]), dict(), dict(categorical=["연령"]), dict(categorical=CATS, continuous=CATS[:1]),
           dict(categorical=CATS, starts=3, finalStarts=5), dict(categorical=CATS, classes=1, covariates=["연령"])]
    for kw in bad:
        with pytest.raises(ValueError):
            LatentClassAnalysis(df, **kw)
    lcga, _ = mk.make_lcga()
    with pytest.raises(ValueError):
        LatentClassGrowthAnalysis(lcga, dep="삶의만족도", idVar="ID", timeVar="연도", growth="exp")
    with pytest.raises(ValueError):
        LatentClassGrowthAnalysis(lcga, dep="삶의만족도", idVar="ID", timeVar="연도", covariates=["연도"])
    lta, _ = mk.make_lta()
    with pytest.raises(ValueError):
        LatentTransitionAnalysis(lta, categorical=mk.LTA_ITEMS, idVar="ID", classes=3)


def test_stepmix_crosscheck_when_available(lca_data):
    stepmix = pytest.importorskip("stepmix")
    from stepmix.stepmix import StepMix
    df, _ = lca_data
    X = df[mk.LCA_BINARY].to_numpy(float)
    U = np.where(np.isfinite(X), X, -1).astype(int)
    ours = mm.fit_model(mm.LatentClassModel(mm.Measurement(U, [2] * 6, None), 3))
    sm = StepMix(n_components=3, measurement="binary_nan", n_init=10, random_state=1, verbose=0, progress_bar=0, max_iter=5000, abs_tol=1e-8)
    sm.fit(X)
    assert abs(sm.score(X) * len(X) - ours.ll) < 0.05

"""
SEM engine (Mplus default conventions) — validation against semopy, statsmodels and analytic identities.
"""
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
from pyjamovi.analyses.sem_module import SEM  # noqa: E402

DATA = os.path.join(ROOT_DIR, "sample_data", "SEM_Latent3Factor.csv")
CFA = "Visual =~ x1_vis + x2_vis + x3_vis\nTextual =~ x4_txt + x5_txt + x6_txt\nSpeed =~ x7_spd + x8_spd + x9_spd"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


def _semopy_compare(df, model):
    semopy = pytest.importorskip("semopy")
    r = se.estimate(df, model, information="expected")
    m = semopy.Model(model)
    m.fit(df, obj="MLW")
    ins = m.inspect(information="expected")
    mine = {(x.lhs, x.op, x.rhs): (r.theta[x.pid], r.se[x.pid]) for x in r.spec.rows if x.free}
    d_est, d_se, matched = 0.0, 0.0, 0
    for _, row in ins.iterrows():
        l, op, rv = row["lval"], row["op"], row["rval"]
        key = (rv, "=~", l) if op == "~" and rv in r.spec.latent and l not in r.spec.latent else (l, op, rv)
        if key not in mine:
            key = (rv, op, l)
        if key in mine:
            matched += 1
            d_est = max(d_est, abs(float(row["Estimate"]) - mine[key][0]))
            d_se = max(d_se, abs(float(row["Std. Err"]) - mine[key][1]))
    st = semopy.calc_stats(m).T
    return r, d_est, d_se, matched, float(st.loc["chi2"].iloc[0]), int(st.loc["DoF"].iloc[0])


def test_complete_data_ml_matches_semopy(df):
    for model in (CFA, CFA + "\nGPA ~ Visual + Textual + Speed"):
        r, d_est, d_se, matched, chi2, dof = _semopy_compare(df, model)
        assert matched >= 20
        assert d_est < 1e-3 and d_se < 1e-3
        assert abs(r.fitstats["chi2"] - chi2) < 1e-3 and r.fitstats["df"] == dof
    # Mplus parameter count: 9 loadings-6 free + 9 residuals + 3 variances + 3 covariances + 9 intercepts
    r = se.estimate(df, CFA)
    assert r.fitstats["npar"] == 30 and r.fitstats["df"] == 24
    assert 0 <= r.fitstats["srmr"] < 0.05 and r.fitstats["cfi"] > 0.99


def test_fiml_saturated_identity_and_casewise_loglik(df):
    from scipy import stats
    rng = np.random.default_rng(3)
    d = df.copy()
    pm = 1 / (1 + np.exp(-(d["x4_txt"] - d["x4_txt"].mean()))) * 0.5
    d.loc[rng.random(len(d)) < pm, "x1_vis"] = np.nan          # MAR on x4
    d.loc[rng.random(len(d)) < 0.2, "x3_vis"] = np.nan
    r = se.estimate(d, "V =~ x1_vis + x2_vis + x3_vis")       # just identified
    mu, Sig = r.sem.ram.moments(r.theta, jac=False)[0][:2]
    h1 = r.fitstats["h1"]
    assert r.fitstats["df"] == 0 and abs(r.fitstats["ll0"] - r.fitstats["ll1"]) < 1e-6
    assert np.abs(Sig - h1["Sigs"][0]).max() < 1e-7 and np.abs(mu - h1["mus"][0]).max() < 1e-7
    d.loc[rng.random(len(d)) < 0.15, "x8_spd"] = np.nan
    r = se.estimate(d, CFA)
    mu, Sig = r.sem.ram.moments(r.theta, jac=False)[0][:2]
    X = d[r.spec.obs].to_numpy(float)
    brute = sum(stats.multivariate_normal(mu[o], Sig[np.ix_(o, o)]).logpdf(x[o]) for x in X for o in [~np.isnan(x)])
    assert abs(brute - r.fit.ll_joint) < 1e-6
    assert r.sem.N == 300 and r.fitstats["df"] == 24


def test_mlr_regression_equals_hc0_and_conditional_loglik(df):
    sm = pytest.importorskip("statsmodels.api")
    d = df.copy()
    rng = np.random.default_rng(1)
    d["y"] = 0.5 * d.x1_vis - 0.3 * d.x4_txt + rng.standard_t(3, len(d)) * (0.5 + 0.3 * np.abs(d.x1_vis - 5))
    r = se.estimate(d, "y ~ x1_vis + x4_txt", estimator="mlr")
    ols = sm.OLS(d.y, sm.add_constant(d[["x1_vis", "x4_txt"]])).fit(cov_type="HC0")
    b = {x.rhs: (r.theta[x.pid], r.se[x.pid]) for x in r.spec.rows if x.op == "~"}
    for v in ("x1_vis", "x4_txt"):
        assert abs(b[v][0] - ols.params[v]) < 1e-6
        assert abs(b[v][1] - ols.bse[v]) < 1e-6
    fs = r.fitstats
    assert abs(fs["ll0"] - ols.llf) < 1e-6              # Mplus: likelihood conditional on x
    assert fs["npar"] == 4 and fs["df"] == 0            # x means/variances are not parameters
    assert abs(fs["c0"] - fs["c1"]) < 1e-6              # saturated: H0 and H1 scaling factors coincide


def test_multigroup_configural_metric_scalar(df):
    m = "Visual =~ x1_vis + x2_vis + x3_vis\nTextual =~ x4_txt + x5_txt + x6_txt"
    rc = se.estimate(df, m, group="School_Type", invariance="configural")
    parts = [se.estimate(df[df.School_Type == g], m) for g in sorted(df.School_Type.unique())]
    assert abs(rc.fitstats["chi2"] - sum(p.fitstats["chi2"] for p in parts)) < 1e-5
    assert abs(rc.fitstats["ll0"] - sum(p.fitstats["ll0"] for p in parts)) < 1e-5
    rm = se.estimate(df, m, group="School_Type", invariance="metric")
    rs = se.estimate(df, m, group="School_Type")        # Mplus default: loadings + intercepts equal
    assert (rc.fitstats["df"], rm.fitstats["df"], rs.fitstats["df"]) == (16, 20, 24)
    means = [x for x in rs.spec.rows if x.op == "~1" and x.lhs in rs.spec.latent]
    assert {x.group for x in means} == {1} and all(x.free for x in means)
    rows = [x for x in rs.spec.rows if x.op == "=~" and x.rhs == "x2_vis"]
    assert len({x.pid for x in rows}) == 1


def test_modification_index_close_to_lr():
    rng = np.random.default_rng(11)
    n = 1500
    F1 = rng.normal(size=n)
    F2 = 0.4 * F1 + rng.normal(size=n) * 0.9
    Y = pd.DataFrame({"a1": F1 + rng.normal(size=n) * .7, "a2": .9 * F1 + rng.normal(size=n) * .7,
                      "a3": .8 * F1 + .15 * F2 + rng.normal(size=n) * .7, "a4": .8 * F1 + rng.normal(size=n) * .7,
                      "b1": F2 + rng.normal(size=n) * .7, "b2": .9 * F2 + rng.normal(size=n) * .7,
                      "b3": .8 * F2 + rng.normal(size=n) * .7, "b4": .7 * F2 + rng.normal(size=n) * .7})
    base = "F1 =~ a1 + a2 + a3 + a4\nF2 =~ b1 + b2 + b3 + b4"
    r0 = se.estimate(Y, base)
    mi = {(m["lhs"], m["op"], m["rhs"]): m for m in se.modification_indices(r0, 0)}
    cand = mi[("F2", "=~", "a3")]
    r1 = se.estimate(Y, base + "\nF2 =~ a3")
    lr = r0.fitstats["chi2"] - r1.fitstats["chi2"]
    assert lr > 5 and abs(cand["mi"] - lr) / lr < 0.15
    freed = [r1.theta[x.pid] for x in r1.spec.rows if x.lhs == "F2" and x.rhs == "a3"][0]
    assert abs(cand["epc"] - freed) < 0.05


def test_indirect_delta_bootstrap_and_defined(df):
    m = ("Visual =~ x1_vis + x2_vis + x3_vis\nTextual =~ x4_txt + x5_txt + x6_txt\n"
         "Textual ~ a*Visual\nGPA ~ b*Textual + c*Visual\nind := a*b")
    res = SEM(df, m, indirect=["GPA <- Visual"])
    ind = {r["type"]: r for r in res.get_table("indirect").rows}
    dfn = res.get_table("defined").rows[0]
    assert abs(ind["특정 간접효과"]["est"] - dfn["est"]) < 1e-10 and abs(ind["특정 간접효과"]["se"] - dfn["se"]) < 1e-6
    assert abs(ind["총효과"]["est"] - ind["총 간접효과"]["est"] - ind["직접효과"]["est"]) < 1e-10
    rb = SEM(df, m, indirect=[("Visual", "Textual", "GPA")], bootstrap=300, ciType="bc", seed=5)
    bse = rb.get_table("indirect").rows[0]["se"]
    assert 0.75 < bse / dfn["se"] < 1.33
    lo, hi = rb.get_table("indirect").rows[0]["ci_lo"], rb.get_table("indirect").rows[0]["ci_hi"]
    assert lo < dfn["est"] < hi
    with pytest.raises(ValueError):
        SEM(df, m, estimator="mlr", bootstrap=50)       # Mplus: no bootstrap with MLR


def test_sem_module_tables_and_errors(df):
    r = SEM(df, CFA + "\nGPA ~ Visual + Textual + Speed")
    assert [t.name for t in r.tables][:2] == ["estimates", "fit"]
    fit = {x["index"]: x["value"] for x in r.get_table("fit").rows}
    assert "Model Fit Indices" in r.get_table("fit").title
    assert fit["Comparative Fit Index (CFI)"] > 0.95 and fit["RMSEA"] < 0.08 and "SRMR" in fit
    assert len(r.plots) == 1 and r.get_table("rsquare") is not None
    k = df.rename(columns={"x1_vis": "시각 1", "x2_vis": "시각-2", "x3_vis": "시각.3"})
    rk = SEM(k, "시각 =~ 시각 1 + 시각-2 + 시각.3")
    assert [x["rval"] for x in rk.tables[0].rows][:3] == ["시각 1", "시각-2", "시각.3"]
    ri = SEM(df, "Visual =~ x1_vis + x2_vis + x3_vis", group="School_Type", invarianceTest=True, estimator="mlr")
    inv = ri.get_table("invariance").rows
    assert len(inv) == 3 and inv[1]["ddf"] == 2 and inv[2]["ddf"] == 2
    d = df.copy()
    d["cat"] = np.where(d.x1_vis > 5, "a", "b")
    for bad in (dict(model_syntax="V =~ cat + x2_vis + x3_vis"), dict(model_syntax="V =~ x1_vis + nope"),
                dict(model_syntax=CFA, estimator="wlsmv")):
        with pytest.raises(ValueError):
            SEM(d, **bad)

"""Tests for analyses/limdep_models.py (binaryReg, ordinalReg, multinomReg, tobitReg, truncReg, heckman, heckprobit).

Reference values (STATA dict at the end of this file): StataNow 19.5 on the synthetic sample datasets
sample_data/LimDep_*.csv — logit/probit/cloglog, margins, estat classification, estat gof, lroc, ologit/oprobit,
oparallel (Brant), mlogit, tobit, truncreg, heckman ML / twostep, heckprobit. Stata stops at its own convergence
tolerance (about 1e-6 relative), hence rtol 1e-5.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reconstructed"))

from pyjamovi.analyses import limdep_models as L  # noqa: E402

DATA = os.path.join(ROOT, "sample_data")
REG = {"1": "농어촌", "2": "대도시", "3": "중소도시"}       # Stata encode order of 지역유형 (= our sorted order)
SERV = ["미이용", "시설입소", "재가방문", "주간보호"]           # Stata encode order of 이용서비스


def close(ours, ref, rtol=1e-5, atol=1e-7):
    ours, ref = float(ours), float(ref)
    assert abs(ours - ref) <= atol + rtol * abs(ref), (ours, ref)


def our_term(term):
    m = re.match(r"^(\d+)\.region$", term)
    if m:
        return f"지역유형_{REG[m.group(1)]}"
    if term.startswith("/var("):
        return term[1:]
    return term


def coef_frame(res, name="coef"):
    c = pd.DataFrame(res.get_table(name).rows)
    eq = c["eq"].astype(str) if "eq" in c else ""
    c["key"] = eq + "|" + c["term"].astype(str)
    return c.set_index("key")


def check_coefs(res, tag, eq=lambda e: "", rtol=1e-5):
    c = coef_frame(res)
    ref = STATA[tag]
    assert ref["b"], tag
    for nm, b in ref["b"].items():
        e, _, term = nm.rpartition(":")
        key = eq(e) + "|" + our_term(term)
        close(c.loc[key, "b"], b, rtol, 1e-5)        # Stata's own convergence tolerance on estimates
        close(c.loc[key, "se"], ref["se"][nm], rtol)
    return c


def check_margins(res, tname, tag, outcome=lambda oc: "", rtol=1e-5):
    m = pd.DataFrame(res.get_table(tname).rows)
    oc = m["outcome"].astype(str) if "outcome" in m else ""
    m["key"] = oc + "|" + m["term"].astype(str)
    m = m.set_index("key")
    for nm, v in STATA[tag]["m"].items():
        term, _, o = nm.partition(":")
        key = (outcome(o) if o else "") + "|" + our_term(term)
        close(m.loc[key, "b"], v, rtol, 1e-8)
        close(m.loc[key, "se"], STATA[tag]["mse"][nm], rtol, 1e-8)


def fitval(res, start, table="fit"):
    t = pd.DataFrame(res.get_table(table).rows)
    return float(t[t["stat"].str.startswith(start)]["value"].iloc[0])


@pytest.fixture(scope="module")
def emp():
    return pd.read_csv(os.path.join(DATA, "LimDep_Employment_Wage.csv"))


@pytest.fixture(scope="module")
def sat():
    return pd.read_csv(os.path.join(DATA, "LimDep_Satisfaction_Service.csv"))


@pytest.fixture(scope="module")
def med():
    return pd.read_csv(os.path.join(DATA, "LimDep_MedicalCost.csv"))


X_EMP = dict(covs=["연령", "교육연수", "미취학자녀수", "배우자소득"], factors=["지역유형"])
Z_SAT = dict(covs=["연령", "여성", "독거", "일상생활장애", "월소득"], factors=["지역유형"])
W_MED = dict(covs=["연령", "여성", "만성질환수", "민간보험", "월소득"], factors=["지역유형"])
HECK = dict(dep="시간당임금", covs=["교육연수", "경력"], factors=["지역유형"], select_dep="취업여부",
            select_covs=X_EMP["covs"], select_factors=["지역유형"])
EQ = lambda e: {"/": "", "": ""}.get(e, e)


# ------------------------------------------------------------------------------------------------ binary
def test_probit_matches_stata(emp):
    r = L.BinaryReg(emp, dep="취업여부", **X_EMP, link="probit", ame=True, mem=True, class_table=True, hosmer=True, roc=True)
    check_coefs(r, "probit")
    e = STATA["probit"]["e"]
    close(r.model_stats["LL"], e["ll"])
    close(r.model_stats["Chi2"][0], e["chi2"])
    close(r.model_stats["R2_McF"], e["r2_p"])
    check_margins(r, "ame", "probit_ame")
    check_margins(r, "mem", "probit_mem")
    cs = pd.DataFrame(r.get_table("class_stats").rows).set_index("stat")["value"]
    close(cs["민감도 Pr(+|D)"], STATA["probit_class"]["r"]["P_p1"])
    close(cs["특이도 Pr(−|~D)"], STATA["probit_class"]["r"]["P_n0"])
    close(cs["정분류율 (Correctly classified)"], STATA["probit_class"]["r"]["P_corr"])
    close(fitval(r, "Hosmer", "hosmer"), STATA["probit_gof"]["r"]["chi2"])
    assert fitval(r, "집단 수", "hosmer") == STATA["probit_gof"]["r"]["m"]
    close(fitval(r, "ROC", "roc"), STATA["probit_lroc"]["r"]["area"], 1e-6)


@pytest.mark.parametrize("cov, tag", [("robust", "probit_rob"), ("cluster", "probit_cl")])
def test_probit_robust_cluster(emp, cov, tag):
    r = L.BinaryReg(emp, dep="취업여부", **X_EMP, link="probit", cov_type=cov, cluster="시군구" if cov == "cluster" else None)
    check_coefs(r, tag)
    close(r.model_stats["Chi2"][0], STATA[tag]["e"]["chi2"])          # Wald chi2 with robust / cluster VCE


def test_logit_or_cluster_gof(emp):
    r = L.BinaryReg(emp, dep="취업여부", **X_EMP, link="logit", odds_ratio=True, ame=True, hosmer=True, class_table=True, roc=True)
    c = check_coefs(r, "logit")
    close(c.loc["|연령", "eb"], np.exp(STATA["logit"]["b"]["취업여부:연령"]))
    close(c.loc["|연령", "eb_se"], np.exp(STATA["logit"]["b"]["취업여부:연령"]) * STATA["logit"]["se"]["취업여부:연령"])
    check_margins(r, "ame", "logit_ame")
    close(fitval(r, "Hosmer", "hosmer"), STATA["logit_gof"]["r"]["chi2"])
    close(fitval(r, "ROC", "roc"), STATA["logit_lroc"]["r"]["area"], 1e-6)
    r = L.BinaryReg(emp, dep="취업여부", **X_EMP, link="logit", cov_type="cluster", cluster="시군구")
    check_coefs(r, "logit_cl")


def test_cloglog_oim(emp):
    r = L.BinaryReg(emp, dep="취업여부", **X_EMP, link="cloglog", ame=True)
    check_coefs(r, "cloglog")                                          # OIM (IRLS / expected information would differ)
    close(r.model_stats["LL"], STATA["cloglog"]["e"]["ll"])
    check_margins(r, "ame", "cloglog_ame")


def test_logit_satisfaction_data(sat):
    check_coefs(L.BinaryReg(sat, dep="정책인지", **Z_SAT), "logit2")


# ------------------------------------------------------------------------------------------------ ordered
def test_ologit_matches_stata(sat):
    r = L.OrdinalReg(sat, dep="서비스만족도", **Z_SAT, pred_probs=True, ame=True, brant=True)
    check_coefs(r, "ologit")
    e = STATA["ologit"]["e"]
    close(r.model_stats["LL"], e["ll"])
    close(r.model_stats["Chi2"][0], e["chi2"])
    close(r.model_stats["R2_McF"], e["r2_p"])
    pp = pd.DataFrame(r.get_table("pred_probs").rows)
    for k in range(5):
        nm = f"{k + 1}{'bn' if k == 0 else ''}._predict"
        close(pp["b"][k], STATA["ologit_pp"]["m"][nm])
        close(pp["se"][k], STATA["ologit_pp"]["mse"][nm])
    check_margins(r, "ame", "ologit_ame", outcome=lambda o: re.match(r"(\d+)", o).group(1))
    b = pd.DataFrame(r.get_table("brant").rows)
    close(b["chi2"][0], STATA["sat_brant"]["r"]["chi2_b"])            # Stata oparallel, brant
    assert b["df"][0] == 21


def test_ologit_robust_oprobit(sat):
    check_coefs(L.OrdinalReg(sat, dep="서비스만족도", **Z_SAT, cov_type="robust"), "ologit_rob")
    r = L.OrdinalReg(sat, dep="서비스만족도", **Z_SAT, link="probit", ame=True)
    check_coefs(r, "oprobit")
    check_margins(r, "ame", "oprobit_ame", outcome=lambda o: re.match(r"(\d+)", o).group(1))


# ------------------------------------------------------------------------------------------------ multinomial
def test_mlogit_matches_stata(sat):
    r = L.MultinomReg(sat, dep="이용서비스", **Z_SAT, ame=True)
    assert "Exer" not in r.title
    c = check_coefs(r, "mlogit", eq=lambda e: e)
    assert "|(기준 범주 base outcome)" not in c.index and "미이용|(기준 범주 base outcome)" in c.index
    close(r.model_stats["LL"], STATA["mlogit"]["e"]["ll"])
    close(r.model_stats["Chi2"][0], STATA["mlogit"]["e"]["chi2"])
    check_margins(r, "ame", "mlogit_ame", outcome=lambda o: SERV[int(re.match(r"(\d+)", o).group(1)) - 1])
    r = L.MultinomReg(sat, dep="이용서비스", **Z_SAT, base_outcome="재가방문", rrr=True)
    c = check_coefs(r, "mlogit_b3", eq=lambda e: e)
    close(c.loc["미이용|연령", "eb"], np.exp(STATA["mlogit_b3"]["b"]["미이용:연령"]))
    r = L.MultinomReg(sat, dep="이용서비스", **Z_SAT, cov_type="robust")
    check_coefs(r, "mlogit_rob", eq=lambda e: e)


# ------------------------------------------------------------------------------------------------ tobit / truncreg
def test_tobit_matches_stata(med):
    r = L.TobitReg(med, dep="본인부담의료비", **W_MED, ll=0, me_conditional=True, me_prob=True, me_censored=True)
    c = check_coefs(r, "tobit")
    e = STATA["tobit"]["e"]
    close(r.model_stats["LL"], e["ll"])
    close(r.model_stats["Chi2"][0], e["chi2"])
    close(r.model_stats["R2_McF"], e["r2_p"])
    assert fitval(r, "좌측 검열") == e["N_lc"] and fitval(r, "미검열") == e["N_unc"]
    assert f"자유도 {int(e['df_r'])} t 분포" in r.get_table("coef").notes[0]       # Stata e(df_r)
    close(c.loc["|/sigma", "b"] ** 2, c.loc["|var(e.본인부담의료비)", "b"])
    check_margins(r, "me_cond", "tobit_me_e")
    check_margins(r, "me_pr", "tobit_me_pr")
    check_margins(r, "me_ystar", "tobit_me_ystar")
    assert r.get_table("me_cond").columns[3]["title"] == "z"          # Stata margins after tobit: z


def test_tobit_upper_robust_min(med):
    r = L.TobitReg(med, dep="본인부담의료비", **W_MED, ll=0, ul=300, me_censored=True, me_conditional=True)
    check_coefs(r, "tobit2")
    assert fitval(r, "우측 검열") == STATA["tobit2"]["e"]["N_rc"]
    check_margins(r, "me_ystar", "tobit2_me_ystar")
    check_margins(r, "me_cond", "tobit2_me_e")
    r = L.TobitReg(med, dep="본인부담의료비", **W_MED, ll=0, cov_type="robust")
    check_coefs(r, "tobit_rob")
    close(r.model_stats["F"][0], STATA["tobit_rob"]["e"]["F"])
    check_coefs(L.TobitReg(med, dep="본인부담의료비", **W_MED, ll="min"), "tobit_llmin")


def test_truncreg_matches_stata(med):
    r = L.TruncReg(med, dep="본인부담의료비", **W_MED, ll=0)
    c = check_coefs(r, "trunc")
    close(r.model_stats["LL"], STATA["trunc"]["e"]["ll"])
    close(r.model_stats["Chi2"][0], STATA["trunc"]["e"]["chi2"])
    close(c.loc["|/sigma", "b"], STATA["trunc"]["e"]["sigma"])
    assert r.model_stats["N"] == STATA["trunc"]["e"]["N"]
    assert fitval(r, "절단으로") == STATA["trunc"]["e"]["N_bf"] - STATA["trunc"]["e"]["N"]   # y <= ll truncated
    check_coefs(L.TruncReg(med, dep="본인부담의료비", **W_MED, ll=0, ul=300), "trunc2")


# ------------------------------------------------------------------------------------------------ selection
def test_heckman_ml_matches_stata(emp):
    r = L.HeckmanSel(emp, **HECK)
    c = check_coefs(r, "heck_ml", eq=EQ)
    e = STATA["heck_ml"]["e"]
    for term, key in (("rho", "rho"), ("sigma", "sigma"), ("lambda", "lambda")):
        close(c.loc["|" + term, "b"], e[key])
    close(c.loc["|lambda", "se"], e["selambda"])
    close(r.model_stats["LL"], e["ll"])
    close(r.model_stats["Chi2"][0], e["chi2"])
    close(fitval(r, "LR", "indep"), e["chi2_c"], 1e-5, 1e-6)
    r = L.HeckmanSel(emp, **dict(HECK, select_dep=None))                # select(varlist): y observed = selected
    check_coefs(r, "heck_ml_nodep", eq=EQ)


def test_heckman_twostep_and_cluster(emp):
    r = L.HeckmanSel(emp, **HECK, method="twostep")
    c = check_coefs(r, "heck_2s", eq=EQ)
    e = STATA["heck_2s"]["e"]
    close(c.loc["/mills|lambda", "b"], e["lambda"])
    close(c.loc["/mills|lambda", "se"], e["selambda"])
    close(c.loc["|rho", "b"], e["rho"])
    close(c.loc["|sigma", "b"], e["sigma"])
    close(r.model_stats["Chi2"][0], e["chi2"])
    r = L.HeckmanSel(emp, **HECK, cov_type="cluster", cluster="시군구")
    check_coefs(r, "heck_cl", eq=EQ)
    close(fitval(r, "Wald", "indep"), STATA["heck_cl"]["e"]["chi2_c"], 1e-4)
    with pytest.raises(ValueError):
        L.HeckmanSel(emp, **HECK, method="twostep", cov_type="robust")


def test_heckprobit_matches_stata(emp):
    P = dict(dep="고임금여부", covs=["교육연수", "경력"], select_dep="취업여부", select_covs=X_EMP["covs"],
             select_factors=["지역유형"])
    r = L.HeckProbit(emp, **P)
    c = check_coefs(r, "heckprob", eq=EQ)
    e = STATA["heckprob"]["e"]
    close(c.loc["|rho", "b"], e["rho"])
    close(r.model_stats["LL"], e["ll"])
    close(fitval(r, "LR", "indep"), e["chi2_c"], 1e-5, 1e-6)
    check_coefs(L.HeckProbit(emp, **P, cov_type="robust"), "heckprob_rob", eq=EQ)


def test_bivariate_normal_cdf():
    from scipy import integrate, special
    rng = np.random.default_rng(0)
    for r in (-0.97, -0.5, 0.0, 0.2, 0.6, 0.94, 0.999):
        a, b = rng.normal(0, 1.5, 5), rng.normal(0, 1.5, 5)
        ours = L._bvn_cdf(a, b, r)
        for ai, bi, oi in zip(a, b, ours):
            ref = integrate.quad(lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi) * special.ndtr((bi - r * x) / np.sqrt(1 - r * r)),
                                 -np.inf, ai, epsabs=1e-14)[0]
            assert abs(oi - ref) < 1e-12


# ------------------------------------------------------------------------------------------------ options, errors
def test_option_validation(emp, med):
    with pytest.raises(ValueError, match="알 수 없는 값"):
        L.BinaryReg(emp, dep="취업여부", **X_EMP, link="gompit")
    with pytest.raises(ValueError, match="알 수 없는 값"):
        L.BinaryReg(emp, dep="취업여부", **X_EMP, cov_type="bootstrap")
    with pytest.raises(ValueError, match="자료에 없는 변수"):
        L.BinaryReg(emp, dep="취업여부", covs=["연령", "학력"])
    with pytest.raises(ValueError, match="clusterVar"):
        L.BinaryReg(emp, dep="취업여부", **X_EMP, cov_type="cluster")
    with pytest.raises(ValueError, match="probit"):
        L.BinaryReg(emp, dep="취업여부", **X_EMP, link="probit", odds_ratio=True)
    with pytest.raises(ValueError, match="검열 하한 ll"):
        L.TobitReg(med, dep="본인부담의료비", **W_MED)
    with pytest.raises(ValueError, match="baseOutcome"):
        L.MultinomReg(pd.read_csv(os.path.join(DATA, "LimDep_Satisfaction_Service.csv")), dep="이용서비스",
                      covs=["연령"], base_outcome="없는범주")


def test_perfect_prediction_and_separation_notes():
    rng = np.random.default_rng(3)
    n = 300
    d = pd.DataFrame({"x": rng.normal(size=n), "g": rng.integers(0, 3, n)})
    d["y"] = (d.x + rng.normal(size=n) > 0).astype(int)
    d.loc[d.g == 2, "y"] = 1
    r = L.BinaryReg(d, dep="y", covs=["x"], factors=["g"])
    assert r.model_stats["N"] == int((d.g != 2).sum())                 # Stata: g==2 obs not used, dummy dropped
    assert any("완전 예측" in nt for nt in r.get_table("coef").notes)


def test_dispatch_aliases_and_model_compare(emp):
    import pyjamovi as ost
    from pyjamovi.analyses import extensions
    assert {"binaryReg", "ordinalReg", "multinomReg", "tobitReg", "truncReg", "heckman", "heckprobit"} <= extensions.ids()
    a = ost.run("binaryReg", emp, dep="취업여부", covs=X_EMP["covs"], factors=["지역유형"], link="probit", covType="robust")
    b = ost.run("binaryReg", emp, dep="취업여부", covs=X_EMP["covs"], factors=["지역유형"], link="logit")
    assert a.options["covType"] == "robust" and a.get_table("coef").kind == "coefficients"
    from pyjamovi.analyses.model_compare import ModelComparison
    mc = ModelComparison(a.comparison_models() + b.comparison_models())
    assert any(r.get("term") == "연령" for r in mc.tables[0].rows)
    t = ost.run("tobit", pd.read_csv(os.path.join(DATA, "LimDep_MedicalCost.csv")), dep="본인부담의료비", covs=["연령"], ll=0)
    assert t.name == "tobitReg"


# ------------------------------------------------------------------------------------------------ Stata input / code export
@pytest.mark.parametrize("line, expect", [
    ("probit lfp age educ i.city, vce(robust)",
     "sess.run('binaryReg', dep='lfp', covs=['age', 'educ'], factors=['city'], link='probit', covType='robust')"),
    ("logit y x i.f, or vce(cluster g) level(90)",
     "sess.run('binaryReg', dep='y', covs=['x'], factors=['f'], oddsRatio=True, covType='cluster', clusterVar='g', ciWidth=90.0)"),
    ("cloglog y x, eform", "sess.run('binaryReg', dep='y', covs=['x'], link='cloglog', oddsRatio=True)"),
    ("ologit y x i.f, or", "sess.run('ordinalReg', dep='y', covs=['x'], factors=['f'], oddsRatio=True)"),
    ("oprobit y x", "sess.run('ordinalReg', dep='y', covs=['x'], link='probit')"),
    ("mlogit y x, baseoutcome(2) rrr", "sess.run('multinomReg', dep='y', covs=['x'], baseOutcome='2', rrr=True)"),
    ("tobit y x, ll(0) ul(300)", "sess.run('tobitReg', dep='y', covs=['x'], ll=0.0, ul=300.0)"),
    ("tobit y x, ll", "sess.run('tobitReg', dep='y', covs=['x'], ll='min')"),
    ("truncreg y x, ll(0)", "sess.run('truncReg', dep='y', covs=['x'], ll=0.0)"),
    ("heckman w x, select(s = z x) twostep",
     "sess.run('heckman', dep='w', covs=['x'], selectDep='s', selectCovs=['z', 'x'], method='twostep')"),
    ("heckprobit h x, select(z i.r)", "sess.run('heckprobit', dep='h', covs=['x'], selectCovs=['z'], selectFactors=['r'])"),
])
def test_transpile_models(line, expect):
    assert L.transpile(line) == expect


@pytest.mark.parametrize("line", ["probit y x if a > 1", "probit y c.x#c.z", "probit y x, offset(z)", "tobit y x",
                                  "ologit y ib2.f", "mlogit y x, b(#2)", "heckman w x, select(z) twostep vce(robust)",
                                  "margins x", "margins, at(x=1)", "tobit y x, ll(lo)"])
def test_transpile_unsupported_raise(line):
    out = L.transpile(line)
    assert out.startswith("raise NotImplementedError(")


def test_transpile_leaves_plain_logit_and_others():
    assert L.transpile("logit y x") is None and L.transpile("logit y x, or") is None
    assert L.transpile("regress y x") is None and L.transpile("estat gof") is None
    assert "stata_followup" in L.transpile("margins, dydx(*)")


def test_followup_updates_result_in_place(emp, med):
    from pyjamovi.agent import JamoviSession
    sess = JamoviSession(emp.copy(), auto_sync_workspace=False)
    res = sess.run("binaryReg", dep="취업여부", covs=X_EMP["covs"], factors=["지역유형"], link="probit")
    out = L.stata_followup(sess, "margins, dydx(*)")
    assert out is res and res.get_table("ame") is not None and len(sess.history) == 1
    L.stata_followup(sess, "estat gof, group(10)")
    L.stata_followup(sess, "estat classification, cutoff(0.4)")
    assert res.options["hosmer"] and res.options["ame"] and res.options["cutoff"] == 0.4
    sess2 = JamoviSession(med.copy(), auto_sync_workspace=False)
    t = sess2.run("tobitReg", dep="본인부담의료비", covs=W_MED["covs"], ll=0)
    L.stata_followup(sess2, "margins, dydx(*) predict(ystar(0,.))")
    assert t.get_table("me_ystar") is not None
    with pytest.raises(NotImplementedError):
        L.stata_followup(sess2, "margins, dydx(*) predict(e(5,.))")


def test_code_export(emp):
    st = L.stata_code("heckman", {"dep": "w", "covs": ["x"], "factors": ["f"], "selectDep": "s", "selectCovs": ["z"],
                                  "method": "twostep"})
    assert st == ["heckman w x i.f, select(s = z) twostep"]
    st = L.stata_code("binaryReg", {"dep": "y", "covs": ["x"], "link": "probit", "ame": True, "covType": "cluster",
                                    "clusterVar": "g", "hosmer": True})
    assert st[0] == "probit y x, vce(cluster g)" and "margins, dydx(*)" in st and "estat gof, group(10)" in st
    assert L.stata_code("tobitReg", {"dep": "y", "covs": ["x"], "ll": 0, "meCensored": True}) == \
        ["tobit y x, ll(0)", "margins, dydx(*) predict(ystar(0,.))"]
    r = L.r_code("multinomReg", {"dep": "y", "covs": ["x"], "baseOutcome": "2"})
    assert any("nnet::multinom" in ln for ln in r) and any('ref = "2"' in ln for ln in r)
    assert any("sampleSelection::selection" in ln for ln in L.r_code("heckprobit", {"dep": "h", "covs": ["x"], "selectCovs": ["z"]}))
    assert L.stata_code("linReg", {}) is None and L.r_code("linReg", {}) is None
    res = L.BinaryReg(emp, dep="취업여부", **X_EMP, link="logit")
    assert res.syntax_python.startswith("sess.run('binaryReg'") and "glm(" in res.syntax_r


# Stata 19.5 reference values (ld_sample.do on sample_data/LimDep_*.csv); generated, do not edit by hand
STATA = {'cloglog': {'b': {'취업여부:2.region': -0.351178152277,
                   '취업여부:3.region': -0.293785790597,
                   '취업여부:_cons': -0.955040273595,
                   '취업여부:교육연수': 0.111675430985,
                   '취업여부:미취학자녀수': -0.423775348799,
                   '취업여부:배우자소득': -0.0786178578309,
                   '취업여부:연령': 0.0166887086971},
             'e': {'N': 1600.0, 'chi2': 262.283165552, 'll': -789.90742505},
             'se': {'취업여부:2.region': 0.0893112527409,
                    '취업여부:3.region': 0.0943045132128,
                    '취업여부:_cons': 0.257405620594,
                    '취업여부:교육연수': 0.0143929572602,
                    '취업여부:미취학자녀수': 0.0402740431253,
                    '취업여부:배우자소득': 0.0131970835413,
                    '취업여부:연령': 0.00329325126743}},
 'cloglog_ame': {'m': {'2.region': -0.0996568714844,
                       '3.region': -0.0822098257286,
                       '교육연수': 0.0328668416529,
                       '미취학자녀수': -0.124719977908,
                       '배우자소득': -0.0231377722174,
                       '연령': 0.00491160088929},
                 'mse': {'2.region': 0.0243807861443,
                         '3.region': 0.0257599026657,
                         '교육연수': 0.00400172978627,
                         '미취학자녀수': 0.0105237639765,
                         '배우자소득': 0.0037607609966,
                         '연령': 0.000945458803053}},
 'heck_2s': {'b': {'/mills:lambda': 0.163865411314,
                   '시간당임금:2.region': 0.28161847192,
                   '시간당임금:3.region': 0.140677958521,
                   '시간당임금:_cons': 0.236319637748,
                   '시간당임금:경력': 0.015499373968,
                   '시간당임금:교육연수': 0.063040984967,
                   '취업여부:2.region': -0.410043105325,
                   '취업여부:3.region': -0.335788650972,
                   '취업여부:_cons': -0.687725260044,
                   '취업여부:교육연수': 0.123068863199,
                   '취업여부:미취학자녀수': -0.44100895037,
                   '취업여부:배우자소득': -0.0830823652354,
                   '취업여부:연령': 0.0178696244158},
             'e': {'N': 1600.0,
                   'N_selected': 1180.0,
                   'N_unc': 1180.0,
                   'chi2': 245.209995116,
                   'lambda': 0.163865411314,
                   'rho': 0.347833320776,
                   'selambda': 0.072465574536,
                   'sigma': 0.471103259885},
             'se': {'/mills:lambda': 0.072465574536,
                    '시간당임금:2.region': 0.0367934572399,
                    '시간당임금:3.region': 0.0380182736445,
                    '시간당임금:_cons': 0.123115578883,
                    '시간당임금:경력': 0.00140408357959,
                    '시간당임금:교육연수': 0.0070733731031,
                    '취업여부:2.region': 0.101070840041,
                    '취업여부:3.region': 0.107200618979,
                    '취업여부:_cons': 0.268098138985,
                    '취업여부:교육연수': 0.0153612478524,
                    '취업여부:미취학자녀수': 0.0397924217781,
                    '취업여부:배우자소득': 0.012999696652,
                    '취업여부:연령': 0.00358533981034}},
 'heck_cl': {'b': {'/athrho': 0.557933146862,
                   '/lnsigma': -0.725501218636,
                   '시간당임금:2.region': 0.268858441858,
                   '시간당임금:3.region': 0.130615232138,
                   '시간당임금:_cons': 0.141842403355,
                   '시간당임금:경력': 0.0160322241249,
                   '시간당임금:교육연수': 0.0676925517117,
                   '취업여부:2.region': -0.403423450905,
                   '취업여부:3.region': -0.327748195801,
                   '취업여부:_cons': -0.643240602109,
                   '취업여부:교육연수': 0.118016503471,
                   '취업여부:미취학자녀수': -0.434012577607,
                   '취업여부:배우자소득': -0.0816658245705,
                   '취업여부:연령': 0.0179762066916},
             'e': {'N': 1600.0,
                   'N_selected': 1180.0,
                   'N_unc': 1180.0,
                   'chi2': 311.465423326,
                   'chi2_c': 2.86121049066,
                   'lambda': 0.245159540863,
                   'll': -1543.47780715,
                   'rho': 0.506442303191,
                   'selambda': 0.133191365123,
                   'sigma': 0.484081877281},
             'se': {'/athrho': 0.329842990485,
                    '/lnsigma': 0.0616330237745,
                    '시간당임금:2.region': 0.0301498614995,
                    '시간당임금:3.region': 0.0321362054822,
                    '시간당임금:_cons': 0.175943496587,
                    '시간당임금:경력': 0.0015830559211,
                    '시간당임금:교육연수': 0.00879324705437,
                    '취업여부:2.region': 0.0930065815154,
                    '취업여부:3.region': 0.109752769743,
                    '취업여부:_cons': 0.230489725478,
                    '취업여부:교육연수': 0.0149132742487,
                    '취업여부:미취학자녀수': 0.0554950318915,
                    '취업여부:배우자소득': 0.0141999929332,
                    '취업여부:연령': 0.00287583834468}},
 'heck_ml': {'b': {'/athrho': 0.557933146862,
                   '/lnsigma': -0.725501218636,
                   '시간당임금:2.region': 0.268858441858,
                   '시간당임금:3.region': 0.130615232138,
                   '시간당임금:_cons': 0.141842403355,
                   '시간당임금:경력': 0.0160322241249,
                   '시간당임금:교육연수': 0.0676925517117,
                   '취업여부:2.region': -0.403423450905,
                   '취업여부:3.region': -0.327748195801,
                   '취업여부:_cons': -0.643240602109,
                   '취업여부:교육연수': 0.118016503471,
                   '취업여부:미취학자녀수': -0.434012577607,
                   '취업여부:배우자소득': -0.0816658245705,
                   '취업여부:연령': 0.0179762066916},
             'e': {'N': 1600.0,
                   'N_selected': 1180.0,
                   'N_unc': 1180.0,
                   'chi2': 243.559871605,
                   'chi2_c': 7.77656598768,
                   'lambda': 0.245159540863,
                   'll': -1543.47780715,
                   'rho': 0.506442303191,
                   'selambda': 0.0782655218379,
                   'sigma': 0.484081877281},
             'se': {'/athrho': 0.195374103971,
                    '/lnsigma': 0.0378523844338,
                    '시간당임금:2.region': 0.037821807013,
                    '시간당임금:3.region': 0.0389175716353,
                    '시간당임금:_cons': 0.130992362032,
                    '시간당임금:경력': 0.00143515940889,
                    '시간당임금:교육연수': 0.00748113333962,
                    '취업여부:2.region': 0.10035972191,
                    '취업여부:3.region': 0.106362346422,
                    '취업여부:_cons': 0.261645116551,
                    '취업여부:교육연수': 0.0151835417116,
                    '취업여부:미취학자녀수': 0.040241746231,
                    '취업여부:배우자소득': 0.0126523944835,
                    '취업여부:연령': 0.00355446857462}},
 'heck_ml_nodep': {'b': {'/athrho': 0.557933146862,
                         '/lnsigma': -0.725501218636,
                         'select:2.region': -0.403423450905,
                         'select:3.region': -0.327748195801,
                         'select:_cons': -0.643240602109,
                         'select:교육연수': 0.118016503471,
                         'select:미취학자녀수': -0.434012577607,
                         'select:배우자소득': -0.0816658245705,
                         'select:연령': 0.0179762066916,
                         '시간당임금:2.region': 0.268858441858,
                         '시간당임금:3.region': 0.130615232138,
                         '시간당임금:_cons': 0.141842403355,
                         '시간당임금:경력': 0.0160322241249,
                         '시간당임금:교육연수': 0.0676925517117},
                   'e': {'N': 1600.0,
                         'N_selected': 1180.0,
                         'N_unc': 1180.0,
                         'chi2': 243.559871605,
                         'chi2_c': 7.77656598768,
                         'lambda': 0.245159540863,
                         'll': -1543.47780715,
                         'rho': 0.506442303191,
                         'selambda': 0.0782655218379,
                         'sigma': 0.484081877281},
                   'se': {'/athrho': 0.195374103971,
                          '/lnsigma': 0.0378523844338,
                          'select:2.region': 0.10035972191,
                          'select:3.region': 0.106362346422,
                          'select:_cons': 0.261645116551,
                          'select:교육연수': 0.0151835417116,
                          'select:미취학자녀수': 0.040241746231,
                          'select:배우자소득': 0.0126523944835,
                          'select:연령': 0.00355446857462,
                          '시간당임금:2.region': 0.037821807013,
                          '시간당임금:3.region': 0.0389175716353,
                          '시간당임금:_cons': 0.130992362032,
                          '시간당임금:경력': 0.00143515940889,
                          '시간당임금:교육연수': 0.00748113333962}},
 'heckprob': {'b': {'/athrho': 0.858672675289,
                    '고임금여부:_cons': -2.78823773908,
                    '고임금여부:경력': 0.0338691465387,
                    '고임금여부:교육연수': 0.169397876509,
                    '취업여부:2.region': -0.517239960915,
                    '취업여부:3.region': -0.34645188876,
                    '취업여부:_cons': -0.641726854774,
                    '취업여부:교육연수': 0.120764538119,
                    '취업여부:미취학자녀수': -0.427709425216,
                    '취업여부:배우자소득': -0.0763548407611,
                    '취업여부:연령': 0.0178589235408},
              'e': {'N': 1600.0,
                    'N_selected': 1180.0,
                    'N_unc': 1180.0,
                    'chi2': 153.038542512,
                    'chi2_c': 17.5595859195,
                    'll': -1484.85514707,
                    'rho': 0.695573169288},
              'se': {'/athrho': 0.21103909087,
                     '고임금여부:_cons': 0.249992434708,
                     '고임금여부:경력': 0.00361214284879,
                     '고임금여부:교육연수': 0.0161982599381,
                     '취업여부:2.region': 0.0957275332262,
                     '취업여부:3.region': 0.100847538914,
                     '취업여부:_cons': 0.26149340808,
                     '취업여부:교육연수': 0.0150706809131,
                     '취업여부:미취학자녀수': 0.0399155251176,
                     '취업여부:배우자소득': 0.01259333696,
                     '취업여부:연령': 0.00354858881858}},
 'heckprob_rob': {'b': {'/athrho': 0.858672675289,
                        '고임금여부:_cons': -2.78823773908,
                        '고임금여부:경력': 0.0338691465387,
                        '고임금여부:교육연수': 0.169397876509,
                        '취업여부:2.region': -0.517239960915,
                        '취업여부:3.region': -0.34645188876,
                        '취업여부:_cons': -0.641726854774,
                        '취업여부:교육연수': 0.120764538119,
                        '취업여부:미취학자녀수': -0.427709425216,
                        '취업여부:배우자소득': -0.0763548407611,
                        '취업여부:연령': 0.0178589235408},
                  'e': {'N': 1600.0,
                        'N_selected': 1180.0,
                        'N_unc': 1180.0,
                        'chi2': 148.893280399,
                        'chi2_c': 15.344078181,
                        'll': -1484.85514707,
                        'rho': 0.695573169288},
                  'se': {'/athrho': 0.219208424984,
                         '고임금여부:_cons': 0.251011268238,
                         '고임금여부:경력': 0.00366177471668,
                         '고임금여부:교육연수': 0.0163513723174,
                         '취업여부:2.region': 0.0985829181813,
                         '취업여부:3.region': 0.103499745853,
                         '취업여부:_cons': 0.2524859397,
                         '취업여부:교육연수': 0.0146907148446,
                         '취업여부:미취학자녀수': 0.0418014454665,
                         '취업여부:배우자소득': 0.0131377363246,
                         '취업여부:연령': 0.00357412123544}},
 'logit': {'b': {'취업여부:2.region': -0.727811384851,
                 '취업여부:3.region': -0.590765316661,
                 '취업여부:_cons': -1.19055323691,
                 '취업여부:교육연수': 0.211890242708,
                 '취업여부:미취학자녀수': -0.74807504444,
                 '취업여부:배우자소득': -0.143067912455,
                 '취업여부:연령': 0.0303831403675},
           'e': {'N': 1600.0, 'chi2': 265.899042719, 'll': -788.099486466, 'r2_p': 0.144345762527},
           'se': {'취업여부:2.region': 0.178149407106,
                  '취업여부:3.region': 0.189243556663,
                  '취업여부:_cons': 0.455664248878,
                  '취업여부:교육연수': 0.0266645672791,
                  '취업여부:미취학자녀수': 0.068631912195,
                  '취업여부:배우자소득': 0.0222866741585,
                  '취업여부:연령': 0.00620920290115}},
 'logit2': {'b': {'정책인지:2.region': 0.339452862876,
                  '정책인지:3.region': 0.272098596786,
                  '정책인지:_cons': 1.48480710491,
                  '정책인지:독거': -0.426092729086,
                  '정책인지:여성': 0.375898735882,
                  '정책인지:연령': -0.0300891770375,
                  '정책인지:월소득': 0.27468583049,
                  '정책인지:일상생활장애': -0.0442876614177},
            'e': {'N': 1200.0, 'chi2': 66.4502329109, 'll': -789.151932276, 'r2_p': 0.0404013177492},
            'se': {'정책인지:2.region': 0.155608301466,
                   '정책인지:3.region': 0.156020582038,
                   '정책인지:_cons': 0.594111367755,
                   '정책인지:독거': 0.125612701344,
                   '정책인지:여성': 0.121784722876,
                   '정책인지:연령': 0.00745394684824,
                   '정책인지:월소득': 0.0666544852787,
                   '정책인지:일상생활장애': 0.0505441936416}},
 'logit_ame': {'m': {'2.region': -0.109792556443,
                     '3.region': -0.0866201449364,
                     '교육연수': 0.0341132117557,
                     '미취학자녀수': -0.120436137474,
                     '배우자소득': -0.0230331794927,
                     '연령': 0.00489152538556},
               'mse': {'2.region': 0.0249115508499,
                       '3.region': 0.0266148558179,
                       '교육연수': 0.00402502025774,
                       '미취학자녀수': 0.00961653377402,
                       '배우자소득': 0.00343768919755,
                       '연령': 0.000976672279353}},
 'logit_cl': {'b': {'취업여부:2.region': -0.727811384851,
                    '취업여부:3.region': -0.590765316661,
                    '취업여부:_cons': -1.19055323691,
                    '취업여부:교육연수': 0.211890242708,
                    '취업여부:미취학자녀수': -0.74807504444,
                    '취업여부:배우자소득': -0.143067912455,
                    '취업여부:연령': 0.0303831403675},
              'e': {'N': 1600.0, 'chi2': 164.488721496, 'll': -788.099486466, 'r2_p': 0.144345762527},
              'se': {'취업여부:2.region': 0.154983980537,
                     '취업여부:3.region': 0.184718056932,
                     '취업여부:_cons': 0.398617410358,
                     '취업여부:교육연수': 0.0249222118813,
                     '취업여부:미취학자녀수': 0.0805454170481,
                     '취업여부:배우자소득': 0.0232160630581,
                     '취업여부:연령': 0.00480022967811}},
 'logit_class': {'r': {'P_0n': 62.3036649215,
                       'P_0p': 21.3626685593,
                       'P_1n': 37.6963350785,
                       'P_1p': 78.6373314407,
                       'P_corr': 76.6875,
                       'P_n0': 28.3333333333,
                       'P_n1': 6.10169491525,
                       'P_p0': 71.6666666667,
                       'P_p1': 93.8983050847}},
 'logit_gof': {'r': {'N': 1600.0, 'chi2': 7.30402012151, 'df': 8.0, 'm': 10.0, 'p': 0.504214465361}},
 'logit_lroc': {'r': {'N': 1600.0, 'area': 0.754221142574}},
 'mlogit': {'b': {'시설입소:2.region': 0.194272332528,
                  '시설입소:3.region': -0.205861058074,
                  '시설입소:_cons': -7.17976371655,
                  '시설입소:독거': 1.01501851435,
                  '시설입소:여성': -0.129557455522,
                  '시설입소:연령': 0.0435672207669,
                  '시설입소:월소득': 0.0814185662596,
                  '시설입소:일상생활장애': 0.753437295624,
                  '재가방문:2.region': 0.46860860764,
                  '재가방문:3.region': 0.354491342163,
                  '재가방문:_cons': -4.13357140098,
                  '재가방문:독거': 0.603208804963,
                  '재가방문:여성': 0.217385132141,
                  '재가방문:연령': 0.024573913227,
                  '재가방문:월소득': 0.0648333858589,
                  '재가방문:일상생활장애': 0.534665324556,
                  '주간보호:2.region': 0.254915583564,
                  '주간보호:3.region': 0.212392349603,
                  '주간보호:_cons': -2.41690737603,
                  '주간보호:독거': 0.0800633469608,
                  '주간보호:여성': -0.144183422902,
                  '주간보호:연령': 0.000880796580345,
                  '주간보호:월소득': 0.190525611703,
                  '주간보호:일상생활장애': 0.420748032272},
            'e': {'N': 1200.0, 'chi2': 226.328026167, 'll': -1290.33838267, 'r2_p': 0.0806297256247},
            'se': {'시설입소:2.region': 0.290581100271,
                   '시설입소:3.region': 0.310025742226,
                   '시설입소:_cons': 1.27768283792,
                   '시설입소:독거': 0.237707496817,
                   '시설입소:여성': 0.235153892954,
                   '시설입소:연령': 0.0152870176388,
                   '시설입소:월소득': 0.131732614098,
                   '시설입소:일상생활장애': 0.0933127162801,
                   '재가방문:2.region': 0.190535593055,
                   '재가방문:3.region': 0.191327115648,
                   '재가방문:_cons': 0.728664511997,
                   '재가방문:독거': 0.149264513004,
                   '재가방문:여성': 0.147224562206,
                   '재가방문:연령': 0.00887102365429,
                   '재가방문:월소득': 0.0811774855078,
                   '재가방문:일상생활장애': 0.0641409482211,
                   '주간보호:2.region': 0.234774219015,
                   '주간보호:3.region': 0.233759908908,
                   '주간보호:_cons': 0.895458002237,
                   '주간보호:독거': 0.194232883166,
                   '주간보호:여성': 0.18155608876,
                   '주간보호:연령': 0.011146630362,
                   '주간보호:월소득': 0.0925812676313,
                   '주간보호:일상생활장애': 0.0798565976354}},
 'mlogit_ame': {'m': {'2.region:1bn._predict': -0.0780339534979,
                      '2.region:2._predict': -0.00264813486992,
                      '2.region:3._predict': 0.0708326625268,
                      '2.region:4._predict': 0.00984942584101,
                      '3.region:1bn._predict': -0.0503944593845,
                      '3.region:2._predict': -0.0250808559171,
                      '3.region:3._predict': 0.0617567334541,
                      '3.region:4._predict': 0.0137185818474,
                      '독거:1bn._predict': -0.107366476027,
                      '독거:2._predict': 0.0508970764294,
                      '독거:3._predict': 0.0810861344174,
                      '독거:4._predict': -0.0246167348201,
                      '여성:1bn._predict': -0.0131570474488,
                      '여성:2._predict': -0.0134724117222,
                      '여성:3._predict': 0.0499380291252,
                      '여성:4._predict': -0.0233085699542,
                      '연령:1bn._predict': -0.00428138897409,
                      '연령:2._predict': 0.00224975019165,
                      '연령:3._predict': 0.00333017812896,
                      '연령:4._predict': -0.00129853934653,
                      '월소득:1bn._predict': -0.0226732820985,
                      '월소득:2._predict': 0.00163455406366,
                      '월소득:3._predict': 0.0026138113787,
                      '월소득:4._predict': 0.0184249166562,
                      '일상생활장애:1bn._predict': -0.113907510702,
                      '일상생활장애:2._predict': 0.031414103697,
                      '일상생활장애:3._predict': 0.0628002434606,
                      '일상생활장애:4._predict': 0.0196931635441},
                'mse': {'2.region:1bn._predict': 0.0344828465759,
                        '2.region:2._predict': 0.0206156257522,
                        '2.region:3._predict': 0.0316604520076,
                        '2.region:4._predict': 0.0247954185276,
                        '3.region:1bn._predict': 0.0344959800362,
                        '3.region:2._predict': 0.0199885262866,
                        '3.region:3._predict': 0.0316910894437,
                        '3.region:4._predict': 0.0250011576998,
                        '독거:1bn._predict': 0.0273569029665,
                        '독거:2._predict': 0.015291943189,
                        '독거:3._predict': 0.0250482592544,
                        '독거:4._predict': 0.0205248535434,
                        '여성:1bn._predict': 0.0270823808474,
                        '여성:2._predict': 0.0151069648235,
                        '여성:3._predict': 0.0253203736145,
                        '여성:4._predict': 0.0195608726844,
                        '연령:1bn._predict': 0.00161466495217,
                        '연령:2._predict': 0.0010036772299,
                        '연령:3._predict': 0.00154375934739,
                        '연령:4._predict': 0.00120105331907,
                        '월소득:1bn._predict': 0.014729553473,
                        '월소득:2._predict': 0.00846650129826,
                        '월소득:3._predict': 0.0139340340888,
                        '월소득:4._predict': 0.00987566248276,
                        '일상생활장애:1bn._predict': 0.0105656532889,
                        '일상생활장애:2._predict': 0.00560256789346,
                        '일상생활장애:3._predict': 0.00953698007476,
                        '일상생활장애:4._predict': 0.00767205009741}},
 'mlogit_b3': {'b': {'미이용:2.region': -0.46860860764,
                     '미이용:3.region': -0.354491342163,
                     '미이용:_cons': 4.13357140098,
                     '미이용:독거': -0.603208804963,
                     '미이용:여성': -0.217385132141,
                     '미이용:연령': -0.024573913227,
                     '미이용:월소득': -0.0648333858589,
                     '미이용:일상생활장애': -0.534665324556,
                     '시설입소:2.region': -0.274336275112,
                     '시설입소:3.region': -0.560352400238,
                     '시설입소:_cons': -3.04619231558,
                     '시설입소:독거': 0.411809709386,
                     '시설입소:여성': -0.346942587663,
                     '시설입소:연령': 0.0189933075398,
                     '시설입소:월소득': 0.0165851804007,
                     '시설입소:일상생활장애': 0.218771971068,
                     '주간보호:2.region': -0.213693024077,
                     '주간보호:3.region': -0.142098992561,
                     '주간보호:_cons': 1.71666402494,
                     '주간보호:독거': -0.523145458002,
                     '주간보호:여성': -0.361568555043,
                     '주간보호:연령': -0.0236931166467,
                     '주간보호:월소득': 0.125692225844,
                     '주간보호:일상생활장애': -0.113917292284},
               'e': {'N': 1200.0, 'chi2': 226.328026167, 'll': -1290.33838267, 'r2_p': 0.0806297256247},
               'se': {'미이용:2.region': 0.190535593055,
                      '미이용:3.region': 0.191327115648,
                      '미이용:_cons': 0.728664511997,
                      '미이용:독거': 0.149264513004,
                      '미이용:여성': 0.147224562206,
                      '미이용:연령': 0.00887102365429,
                      '미이용:월소득': 0.0811774855078,
                      '미이용:일상생활장애': 0.0641409482211,
                      '시설입소:2.region': 0.29756529891,
                      '시설입소:3.region': 0.318146699722,
                      '시설입소:_cons': 1.3069813377,
                      '시설입소:독거': 0.237911675968,
                      '시설입소:여성': 0.237675256725,
                      '시설입소:연령': 0.0156312859072,
                      '시설입소:월소득': 0.1330502761,
                      '시설입소:일상생활장애': 0.0885779010199,
                      '주간보호:2.region': 0.25771377449,
                      '주간보호:3.region': 0.258882723281,
                      '주간보호:_cons': 0.981521439154,
                      '주간보호:독거': 0.204891701931,
                      '주간보호:여성': 0.196154627761,
                      '주간보호:연령': 0.0120978673494,
                      '주간보호:월소득': 0.100887379212,
                      '주간보호:일상생활장애': 0.0803149933757}},
 'mlogit_rob': {'b': {'시설입소:2.region': 0.194272332528,
                      '시설입소:3.region': -0.205861058074,
                      '시설입소:_cons': -7.17976371655,
                      '시설입소:독거': 1.01501851435,
                      '시설입소:여성': -0.129557455522,
                      '시설입소:연령': 0.0435672207669,
                      '시설입소:월소득': 0.0814185662596,
                      '시설입소:일상생활장애': 0.753437295624,
                      '재가방문:2.region': 0.46860860764,
                      '재가방문:3.region': 0.354491342163,
                      '재가방문:_cons': -4.13357140098,
                      '재가방문:독거': 0.603208804963,
                      '재가방문:여성': 0.217385132141,
                      '재가방문:연령': 0.024573913227,
                      '재가방문:월소득': 0.0648333858589,
                      '재가방문:일상생활장애': 0.534665324556,
                      '주간보호:2.region': 0.254915583564,
                      '주간보호:3.region': 0.212392349603,
                      '주간보호:_cons': -2.41690737603,
                      '주간보호:독거': 0.0800633469608,
                      '주간보호:여성': -0.144183422902,
                      '주간보호:연령': 0.000880796580345,
                      '주간보호:월소득': 0.190525611703,
                      '주간보호:일상생활장애': 0.420748032272},
                'e': {'N': 1200.0, 'chi2': 173.35505323, 'll': -1290.33838267, 'r2_p': 0.0806297256247},
                'se': {'시설입소:2.region': 0.290874288854,
                       '시설입소:3.region': 0.307805708587,
                       '시설입소:_cons': 1.31714755737,
                       '시설입소:독거': 0.240206161805,
                       '시설입소:여성': 0.233047067308,
                       '시설입소:연령': 0.0152417522301,
                       '시설입소:월소득': 0.123368890697,
                       '시설입소:일상생활장애': 0.0941370372987,
                       '재가방문:2.region': 0.191502744518,
                       '재가방문:3.region': 0.19405681529,
                       '재가방문:_cons': 0.7445806711,
                       '재가방문:독거': 0.14843008638,
                       '재가방문:여성': 0.14724534762,
                       '재가방문:연령': 0.00882835832405,
                       '재가방문:월소득': 0.0847490950089,
                       '재가방문:일상생활장애': 0.0639452883372,
                       '주간보호:2.region': 0.236880566219,
                       '주간보호:3.region': 0.234630358315,
                       '주간보호:_cons': 0.868868395099,
                       '주간보호:독거': 0.194882783596,
                       '주간보호:여성': 0.18191736206,
                       '주간보호:연령': 0.0113668674833,
                       '주간보호:월소득': 0.0878009162327,
                       '주간보호:일상생활장애': 0.0822285782508}},
 'ologit': {'b': {'/cut1': 0.517613822486,
                  '/cut2': 1.75930874545,
                  '/cut3': 3.11057773671,
                  '/cut4': 4.34174070195,
                  '서비스만족도:2.region': -0.32383707698,
                  '서비스만족도:3.region': -0.277861044851,
                  '서비스만족도:독거': -0.389412335009,
                  '서비스만족도:여성': 0.312388828625,
                  '서비스만족도:연령': 0.0326048022734,
                  '서비스만족도:월소득': 0.273968603276,
                  '서비스만족도:일상생활장애': -0.213133740319},
            'e': {'N': 1200.0, 'chi2': 84.8441265757, 'll': -1832.2637509, 'r2_p': 0.0226288922479},
            'se': {'/cut1': 0.520683858788,
                   '/cut2': 0.520976430924,
                   '/cut3': 0.526884006251,
                   '/cut4': 0.53400647915,
                   '서비스만족도:2.region': 0.132765826862,
                   '서비스만족도:3.region': 0.133789294235,
                   '서비스만족도:독거': 0.107741350351,
                   '서비스만족도:여성': 0.104653021011,
                   '서비스만족도:연령': 0.00649700188458,
                   '서비스만족도:월소득': 0.0573772862451,
                   '서비스만족도:일상생활장애': 0.0438324608546}},
 'ologit_ame': {'m': {'2.region:1bn._predict': 0.033660345048,
                      '2.region:2._predict': 0.0330583920833,
                      '2.region:3._predict': 0.00563693787106,
                      '2.region:4._predict': -0.0309295807602,
                      '2.region:5._predict': -0.0414260942422,
                      '3.region:1bn._predict': 0.028392085192,
                      '3.region:2._predict': 0.0284182505961,
                      '3.region:3._predict': 0.00561633110532,
                      '3.region:4._predict': -0.0263571046626,
                      '3.region:5._predict': -0.0360695622309,
                      '독거:1bn._predict': 0.0424555412168,
                      '독거:2._predict': 0.0392550259742,
                      '독거:3._predict': 0.00370264298345,
                      '독거:4._predict': -0.0375355992187,
                      '독거:5._predict': -0.0478776109558,
                      '여성:1bn._predict': -0.0340580808491,
                      '여성:2._predict': -0.0314906090005,
                      '여성:3._predict': -0.00297028162806,
                      '여성:4._predict': 0.0301112749071,
                      '여성:5._predict': 0.0384076965706,
                      '연령:1bn._predict': -0.0035547269625,
                      '연령:2._predict': -0.00328675351308,
                      '연령:3._predict': -0.000310015712166,
                      '연령:4._predict': 0.00314278896869,
                      '연령:5._predict': 0.00400870721905,
                      '월소득:1bn._predict': -0.0298693294557,
                      '월소득:2._predict': -0.027617627052,
                      '월소득:3._predict': -0.00260497122306,
                      '월소득:4._predict': 0.0264079351539,
                      '월소득:5._predict': 0.0336839925768,
                      '일상생활장애:1bn._predict': 0.0232368301755,
                      '일상생활장애:2._predict': 0.0214851193967,
                      '일상생활장애:3._predict': 0.00202653608318,
                      '일상생활장애:4._predict': -0.020544040179,
                      '일상생활장애:5._predict': -0.0262044454764},
                'mse': {'2.region:1bn._predict': 0.0136248134492,
                        '2.region:2._predict': 0.013674713043,
                        '2.region:3._predict': 0.00387634772698,
                        '2.region:4._predict': 0.0126714876267,
                        '2.region:5._predict': 0.0174557829609,
                        '3.region:1bn._predict': 0.0135327690596,
                        '3.region:2._predict': 0.0137716108012,
                        '3.region:3._predict': 0.00386946129236,
                        '3.region:4._predict': 0.0127015883387,
                        '3.region:5._predict': 0.0177300059772,
                        '독거:1bn._predict': 0.0119747593975,
                        '독거:2._predict': 0.0108173432098,
                        '독거:3._predict': 0.00273697198605,
                        '독거:4._predict': 0.0103508004457,
                        '독거:5._predict': 0.0134443643768,
                        '여성:1bn._predict': 0.01156327595,
                        '여성:2._predict': 0.0105394231063,
                        '여성:3._predict': 0.00225392505813,
                        '여성:4._predict': 0.01007974013,
                        '여성:5._predict': 0.0129761197225,
                        '연령:1bn._predict': 0.00073873368336,
                        '연령:2._predict': 0.000651330731643,
                        '연령:3._predict': 0.000218738507399,
                        '연령:4._predict': 0.000622129704759,
                        '연령:5._predict': 0.000818416717967,
                        '월소득:1bn._predict': 0.0065234588613,
                        '월소득:2._predict': 0.00579950654514,
                        '월소득:3._predict': 0.00182092957891,
                        '월소득:4._predict': 0.00553017442604,
                        '월소득:5._predict': 0.00716660290824,
                        '일상생활장애:1bn._predict': 0.00496074570703,
                        '일상생활장애:2._predict': 0.0043903208785,
                        '일상생활장애:3._predict': 0.00144553803823,
                        '일상생활장애:4._predict': 0.00420697570318,
                        '일상생활장애:5._predict': 0.00552973571298}},
 'ologit_pp': {'m': {'1bn._predict': 0.128302163107,
                     '2._predict': 0.199175864833,
                     '3._predict': 0.309210060881,
                     '4._predict': 0.213872270707,
                     '5._predict': 0.149439640471},
               'mse': {'1bn._predict': 0.00951644913379,
                       '2._predict': 0.0113903926782,
                       '3._predict': 0.0132974054594,
                       '4._predict': 0.0117032678211,
                       '5._predict': 0.0100899890858}},
 'ologit_rob': {'b': {'/cut1': 0.517613822486,
                      '/cut2': 1.75930874545,
                      '/cut3': 3.11057773671,
                      '/cut4': 4.34174070195,
                      '서비스만족도:2.region': -0.32383707698,
                      '서비스만족도:3.region': -0.277861044851,
                      '서비스만족도:독거': -0.389412335009,
                      '서비스만족도:여성': 0.312388828625,
                      '서비스만족도:연령': 0.0326048022734,
                      '서비스만족도:월소득': 0.273968603276,
                      '서비스만족도:일상생활장애': -0.213133740319},
                'e': {'N': 1200.0, 'chi2': 75.7768472957, 'll': -1832.2637509, 'r2_p': 0.0226288922479},
                'se': {'/cut1': 0.539264601367,
                       '/cut2': 0.540157793561,
                       '/cut3': 0.548088600679,
                       '/cut4': 0.559846946232,
                       '서비스만족도:2.region': 0.131254671635,
                       '서비스만족도:3.region': 0.134799111148,
                       '서비스만족도:독거': 0.108172874811,
                       '서비스만족도:여성': 0.104835718571,
                       '서비스만족도:연령': 0.00671197129738,
                       '서비스만족도:월소득': 0.0574310468796,
                       '서비스만족도:일상생활장애': 0.0450443602809}},
 'oprobit': {'b': {'/cut1': 0.185987198748,
                   '/cut2': 0.896642142567,
                   '/cut3': 1.72414481746,
                   '/cut4': 2.43874780567,
                   '서비스만족도:2.region': -0.19515482213,
                   '서비스만족도:3.region': -0.183119268383,
                   '서비스만족도:독거': -0.201627593668,
                   '서비스만족도:여성': 0.16180751769,
                   '서비스만족도:연령': 0.0177682340596,
                   '서비스만족도:월소득': 0.157785468872,
                   '서비스만족도:일상생활장애': -0.118767885404},
             'e': {'N': 1200.0, 'chi2': 78.4083885501, 'll': -1835.48161991, 'r2_p': 0.0209124078171},
             'se': {'/cut1': 0.302922381195,
                    '/cut2': 0.303292407207,
                    '/cut3': 0.30534780345,
                    '/cut4': 0.307605106318,
                    '서비스만족도:2.region': 0.0782038143545,
                    '서비스만족도:3.region': 0.0784210216421,
                    '서비스만족도:독거': 0.0630558620297,
                    '서비스만족도:여성': 0.0613710528576,
                    '서비스만족도:연령': 0.00377474811062,
                    '서비스만족도:월소득': 0.0337077989353,
                    '서비스만족도:일상생활장애': 0.0254876559237}},
 'oprobit_ame': {'m': {'2.region:1bn._predict': 0.0375496089517,
                       '2.region:2._predict': 0.029224142157,
                       '2.region:3._predict': 0.00505806683556,
                       '2.region:4._predict': -0.02615180412,
                       '2.region:5._predict': -0.0456800138244,
                       '3.region:1bn._predict': 0.0349889800154,
                       '3.region:2._predict': 0.0274947226368,
                       '3.region:3._predict': 0.00504941934495,
                       '3.region:4._predict': -0.0244207540899,
                       '3.region:5._predict': -0.0431123679073,
                       '독거:1bn._predict': 0.0407942168829,
                       '독거:2._predict': 0.0294101699698,
                       '독거:3._predict': 0.00275984622978,
                       '독거:4._predict': -0.027759518345,
                       '독거:5._predict': -0.0452047147375,
                       '여성:1bn._predict': -0.0327376369963,
                       '여성:2._predict': -0.0236018617844,
                       '여성:3._predict': -0.00221479540337,
                       '여성:4._predict': 0.0222772026089,
                       '여성:5._predict': 0.0362770915751,
                       '연령:1bn._predict': -0.00359495037691,
                       '연령:2._predict': -0.00259174240118,
                       '연령:3._predict': -0.000243208743839,
                       '연령:4._predict': 0.00244628034468,
                       '연령:5._predict': 0.00398362117724,
                       '월소득:1bn._predict': -0.0319238776848,
                       '월소득:2._predict': -0.0230151904007,
                       '월소득:3._predict': -0.00215974224291,
                       '월소득:4._predict': 0.021723458273,
                       '월소득:5._predict': 0.0353753520554,
                       '일상생활장애:1bn._predict': 0.0240296617529,
                       '일상생활장애:2._predict': 0.017323936834,
                       '일상생활장애:3._predict': 0.00162567580552,
                       '일상생활장애:4._predict': -0.0163516274419,
                       '일상생활장애:5._predict': -0.0266276469505},
                 'mse': {'2.region:1bn._predict': 0.0148453115839,
                         '2.region:2._predict': 0.0119127254169,
                         '2.region:3._predict': 0.00326488893374,
                         '2.region:4._predict': 0.0104552272154,
                         '2.region:5._predict': 0.0187317668433,
                         '3.region:1bn._predict': 0.0147967477666,
                         '3.region:2._predict': 0.0119542921158,
                         '3.region:3._predict': 0.00325839266782,
                         '3.region:4._predict': 0.0104375129096,
                         '3.region:5._predict': 0.0188437304089,
                         '독거:1bn._predict': 0.0128670461861,
                         '독거:2._predict': 0.00923267871667,
                         '독거:3._predict': 0.00200955628651,
                         '독거:4._predict': 0.00871782177322,
                         '독거:5._predict': 0.0142284454853,
                         '여성:1bn._predict': 0.0124899837836,
                         '여성:2._predict': 0.00898231795934,
                         '여성:3._predict': 0.00167377400461,
                         '여성:4._predict': 0.00848005580319,
                         '여성:5._predict': 0.0138063905654,
                         '연령:1bn._predict': 0.000779855400536,
                         '연령:2._predict': 0.000556174592311,
                         '연령:3._predict': 0.000166828438026,
                         '연령:4._predict': 0.000524976762722,
                         '연령:5._predict': 0.000856657126729,
                         '월소득:1bn._predict': 0.00699026262678,
                         '월소득:2._predict': 0.00496890213315,
                         '월소득:3._predict': 0.00146926610598,
                         '월소득:4._predict': 0.00471154396867,
                         '월소득:5._predict': 0.0076145659662,
                         '일상생활장애:1bn._predict': 0.00526285443923,
                         '일상생활장애:2._predict': 0.0037481155306,
                         '일상생활장애:3._predict': 0.00112107993812,
                         '일상생활장애:4._predict': 0.00354824814527,
                         '일상생활장애:5._predict': 0.00578686963378}},
 'probit': {'b': {'취업여부:2.region': -0.410043105385,
                  '취업여부:3.region': -0.335788651036,
                  '취업여부:_cons': -0.687725260114,
                  '취업여부:교육연수': 0.123068863209,
                  '취업여부:미취학자녀수': -0.441008950393,
                  '취업여부:배우자소득': -0.0830823652403,
                  '취업여부:연령': 0.0178696244174},
            'e': {'N': 1600.0, 'chi2': 265.791327748, 'll': -788.153343952, 'r2_p': 0.144287288455},
            'se': {'취업여부:2.region': 0.101071168778,
                   '취업여부:3.region': 0.107200950394,
                   '취업여부:_cons': 0.268098639912,
                   '취업여부:교육연수': 0.0153612858292,
                   '취업여부:미취학자녀수': 0.0397924937367,
                   '취업여부:배우자소득': 0.0129997185295,
                   '취업여부:연령': 0.00358534706863}},
 'probit_ame': {'m': {'2.region': -0.107394259903,
                      '3.region': -0.0859112155062,
                      '교육연수': 0.0340869783398,
                      '미취학자녀수': -0.122148382197,
                      '배우자소득': -0.0230117245787,
                      '연령': 0.00494943631212},
                'mse': {'2.region': 0.0249023872277,
                        '3.region': 0.0265001085442,
                        '교육연수': 0.00403807460705,
                        '미취학자녀수': 0.00983909867673,
                        '배우자소득': 0.00348301187385,
                        '연령': 0.000973164225863}},
 'probit_cl': {'b': {'취업여부:2.region': -0.410043105385,
                     '취업여부:3.region': -0.335788651036,
                     '취업여부:_cons': -0.687725260114,
                     '취업여부:교육연수': 0.123068863209,
                     '취업여부:미취학자녀수': -0.441008950393,
                     '취업여부:배우자소득': -0.0830823652403,
                     '취업여부:연령': 0.0178696244174},
               'e': {'N': 1600.0, 'chi2': 167.220549466, 'll': -788.153343952, 'r2_p': 0.144287288455},
               'se': {'취업여부:2.region': 0.0879177847313,
                      '취업여부:3.region': 0.105490232723,
                      '취업여부:_cons': 0.24031855762,
                      '취업여부:교육연수': 0.0144348876827,
                      '취업여부:미취학자녀수': 0.0467033461431,
                      '취업여부:배우자소득': 0.0133756236263,
                      '취업여부:연령': 0.00287197822112}},
 'probit_class': {'r': {'P_0n': 63.4408602151,
                        'P_0p': 21.3578500707,
                        'P_1n': 36.5591397849,
                        'P_1p': 78.6421499293,
                        'P_corr': 76.875,
                        'P_n0': 28.0952380952,
                        'P_n1': 5.76271186441,
                        'P_p0': 71.9047619048,
                        'P_p1': 94.2372881356}},
 'probit_gof': {'r': {'N': 1600.0, 'chi2': 8.06305626072, 'df': 8.0, 'm': 10.0, 'p': 0.427334970654}},
 'probit_lroc': {'r': {'N': 1600.0, 'area': 0.754110166185}},
 'probit_mem': {'m': {'2.region': -0.115908583465,
                      '3.region': -0.0920784387886,
                      '교육연수': 0.0375871910253,
                      '미취학자녀수': -0.134691157699,
                      '배우자소득': -0.0253746776537,
                      '연령': 0.00545766791872},
                'mse': {'2.region': 0.0264839880417,
                        '3.region': 0.0282416930473,
                        '교육연수': 0.00465777103249,
                        '미취학자녀수': 0.0121207578239,
                        '배우자소득': 0.00397101728462,
                        '연령': 0.0010901339724}},
 'probit_rob': {'b': {'취업여부:2.region': -0.410043105385,
                      '취업여부:3.region': -0.335788651036,
                      '취업여부:_cons': -0.687725260114,
                      '취업여부:교육연수': 0.123068863209,
                      '취업여부:미취학자녀수': -0.441008950393,
                      '취업여부:배우자소득': -0.0830823652403,
                      '취업여부:연령': 0.0178696244174},
                'e': {'N': 1600.0, 'chi2': 221.356753659, 'll': -788.153343952, 'r2_p': 0.144287288455},
                'se': {'취업여부:2.region': 0.101572717208,
                       '취업여부:3.region': 0.108213002412,
                       '취업여부:_cons': 0.2595313026,
                       '취업여부:교육연수': 0.015113052506,
                       '취업여부:미취학자녀수': 0.0403517702673,
                       '취업여부:배우자소득': 0.0131824897991,
                       '취업여부:연령': 0.00361538284707}},
 'sat_brant': {'r': {'chi2_b': 24.456997685657242, 'df': 21}},
 'tobit': {'b': {'/var(e.본인부담의료비)': 4715.86834633,
                 '본인부담의료비:2.region': 11.0588505284,
                 '본인부담의료비:3.region': 3.82927587418,
                 '본인부담의료비:_cons': -67.0227233511,
                 '본인부담의료비:만성질환수': 27.6601368948,
                 '본인부담의료비:민간보험': 20.7997742283,
                 '본인부담의료비:여성': 7.93683324702,
                 '본인부담의료비:연령': 1.20069876078,
                 '본인부담의료비:월소득': 2.61684893593},
           'e': {'N': 1000.0,
                 'N_lc': 247.0,
                 'N_rc': 0.0,
                 'N_unc': 753.0,
                 'chi2': 389.794738522,
                 'df_r': 993.0,
                 'll': -4463.76409324,
                 'r2_p': 0.0418354866156},
           'se': {'/var(e.본인부담의료비)': 254.722074783,
                  '본인부담의료비:2.region': 6.06660044001,
                  '본인부담의료비:3.region': 6.35841367103,
                  '본인부담의료비:_cons': 10.6417081746,
                  '본인부담의료비:만성질환수': 2.07847965669,
                  '본인부담의료비:민간보험': 4.66145845589,
                  '본인부담의료비:여성': 4.5393316084,
                  '본인부담의료비:연령': 0.137113052235,
                  '본인부담의료비:월소득': 1.42807665322}},
 'tobit2': {'b': {'/var(e.본인부담의료비)': 4764.9512745,
                  '본인부담의료비:2.region': 11.2263761932,
                  '본인부담의료비:3.region': 3.82045624769,
                  '본인부담의료비:_cons': -67.4991753961,
                  '본인부담의료비:만성질환수': 27.9230208009,
                  '본인부담의료비:민간보험': 20.9921766295,
                  '본인부담의료비:여성': 7.8192021551,
                  '본인부담의료비:연령': 1.20318409824,
                  '본인부담의료비:월소득': 2.59778509587},
            'e': {'N': 1000.0,
                  'N_lc': 247.0,
                  'N_rc': 4.0,
                  'N_unc': 749.0,
                  'chi2': 390.717372048,
                  'df_r': 993.0,
                  'll': -4450.00984373,
                  'r2_p': 0.0420545075751},
            'se': {'/var(e.본인부담의료비)': 258.541201101,
                   '본인부담의료비:2.region': 6.09943465344,
                   '본인부담의료비:3.region': 6.39272549051,
                   '본인부담의료비:_cons': 10.7014359315,
                   '본인부담의료비:만성질환수': 2.09335851683,
                   '본인부담의료비:민간보험': 4.68835280419,
                   '본인부담의료비:여성': 4.56461488099,
                   '본인부담의료비:연령': 0.137867463051,
                   '본인부담의료비:월소득': 1.435708142}},
 'tobit2_me_e': {'m': {'2.region': 6.3719548807,
                       '3.region': 2.1231429943,
                       '만성질환수': 15.912490412,
                       '민간보험': 11.9628106044,
                       '여성': 4.45592832559,
                       '연령': 0.685658459506,
                       '월소득': 1.48039965752},
                 'mse': {'2.region': 3.42320028203,
                         '3.region': 3.54341297808,
                         '만성질환수': 1.16211775837,
                         '민간보험': 2.67135186134,
                         '여성': 2.60087510008,
                         '연령': 0.0788911233783,
                         '월소득': 0.817778532911}},
 'tobit2_me_ystar': {'m': {'2.region': 8.40976068374,
                           '3.region': 2.80843900323,
                           '만성질환수': 20.9860844614,
                           '민간보험': 15.7770749418,
                           '여성': 5.87667208422,
                           '연령': 0.904276198777,
                           '월소득': 1.9524154576},
                     'mse': {'2.region': 4.52191208689,
                             '3.region': 4.6885510669,
                             '만성질환수': 1.50834027614,
                             '민간보험': 3.5094476487,
                             '여성': 3.42818258154,
                             '연령': 0.101683123189,
                             '월소득': 1.07814556099}},
 'tobit_llmin': {'b': {'/var(e.본인부담의료비)': 4715.86834633,
                       '본인부담의료비:2.region': 11.0588505284,
                       '본인부담의료비:3.region': 3.82927587418,
                       '본인부담의료비:_cons': -67.0227233511,
                       '본인부담의료비:만성질환수': 27.6601368948,
                       '본인부담의료비:민간보험': 20.7997742283,
                       '본인부담의료비:여성': 7.93683324702,
                       '본인부담의료비:연령': 1.20069876078,
                       '본인부담의료비:월소득': 2.61684893593},
                 'e': {'N': 1000.0,
                       'N_lc': 247.0,
                       'N_rc': 0.0,
                       'N_unc': 753.0,
                       'chi2': 389.794738522,
                       'df_r': 993.0,
                       'll': -4463.76409324,
                       'r2_p': 0.0418354866156},
                 'se': {'/var(e.본인부담의료비)': 254.722074783,
                        '본인부담의료비:2.region': 6.06660044001,
                        '본인부담의료비:3.region': 6.35841367103,
                        '본인부담의료비:_cons': 10.6417081746,
                        '본인부담의료비:만성질환수': 2.07847965669,
                        '본인부담의료비:민간보험': 4.66145845589,
                        '본인부담의료비:여성': 4.5393316084,
                        '본인부담의료비:연령': 0.137113052235,
                        '본인부담의료비:월소득': 1.42807665322}},
 'tobit_me_e': {'m': {'2.region': 6.46319201868,
                      '3.region': 2.18749647638,
                      '만성질환수': 16.2408603569,
                      '민간보험': 12.2127460895,
                      '여성': 4.66017217961,
                      '연령': 0.704999435787,
                      '월소득': 1.53650281288},
                'mse': {'2.region': 3.50163046655,
                        '3.region': 3.62128492365,
                        '만성질환수': 1.21735441137,
                        '민간보험': 2.74098871329,
                        '여성': 2.66549829559,
                        '연령': 0.0810975989036,
                        '월소득': 0.8384133113}},
 'tobit_me_pr': {'m': {'2.region': 0.0428579873204,
                       '3.region': 0.0152606297316,
                       '만성질환수': 0.10653793879,
                       '민간보험': 0.0801140313227,
                       '여성': 0.0305701254435,
                       '연령': 0.00462470491625,
                       '월소득': 0.01007925929},
                 'mse': {'2.region': 0.0239081346079,
                         '3.region': 0.0254415786875,
                         '만성질환수': 0.00834126658469,
                         '민간보험': 0.0178880474917,
                         '여성': 0.0174683381287,
                         '연령': 0.000512516624405,
                         '월소득': 0.00550088174986}},
 'tobit_me_ystar': {'m': {'2.region': 8.33103070208,
                          '3.region': 2.83083589291,
                          '만성질환수': 20.9089385806,
                          '민간보험': 15.7230314327,
                          '여성': 5.99963620996,
                          '연령': 0.907636022861,
                          '월소득': 1.97813676354},
                    'mse': {'2.region': 4.52246842109,
                            '3.region': 4.6893069163,
                            '만성질환수': 1.51904224109,
                            '민간보험': 3.51166549715,
                            '여성': 3.42896283626,
                            '연령': 0.101747468947,
                            '월소득': 1.07870107587}},
 'tobit_rob': {'b': {'/var(e.본인부담의료비)': 4715.86834633,
                     '본인부담의료비:2.region': 11.0588505284,
                     '본인부담의료비:3.region': 3.82927587418,
                     '본인부담의료비:_cons': -67.0227233511,
                     '본인부담의료비:만성질환수': 27.6601368948,
                     '본인부담의료비:민간보험': 20.7997742283,
                     '본인부담의료비:여성': 7.93683324702,
                     '본인부담의료비:연령': 1.20069876078,
                     '본인부담의료비:월소득': 2.61684893593},
               'e': {'F': 59.4104135739,
                     'N': 1000.0,
                     'N_lc': 247.0,
                     'N_rc': 0.0,
                     'N_unc': 753.0,
                     'df_r': 993.0,
                     'll': -4463.76409324,
                     'r2_p': 0.0418354866156},
               'se': {'/var(e.본인부담의료비)': 252.326932498,
                      '본인부담의료비:2.region': 5.85987452084,
                      '본인부담의료비:3.region': 6.09005128058,
                      '본인부담의료비:_cons': 10.4727450347,
                      '본인부담의료비:만성질환수': 2.04890889463,
                      '본인부담의료비:민간보험': 4.65911932984,
                      '본인부담의료비:여성': 4.5764138022,
                      '본인부담의료비:연령': 0.135516708835,
                      '본인부담의료비:월소득': 1.34610180656}},
 'trunc': {'b': {'/sigma': 67.3708917492,
                 'eq1:2.region': 20.5916817573,
                 'eq1:3.region': 13.4488189094,
                 'eq1:_cons': -75.9397391445,
                 'eq1:만성질환수': 26.0838778428,
                 'eq1:민간보험': 26.6268934912,
                 'eq1:여성': 7.60816714355,
                 'eq1:연령': 1.27582264471,
                 'eq1:월소득': 2.05121794139},
           'e': {'N': 753.0, 'N_bf': 1000.0, 'chi2': 197.795884626, 'll': -3984.40752246, 'sigma': 67.3708917492},
           'se': {'/sigma': 2.81384382844,
                  'eq1:2.region': 8.71116377411,
                  'eq1:3.region': 9.08209065234,
                  'eq1:_cons': 18.0146706003,
                  'eq1:만성질환수': 2.65849115543,
                  'eq1:민간보험': 6.41308998148,
                  'eq1:여성': 6.22775769085,
                  'eq1:연령': 0.198650183708,
                  'eq1:월소득': 1.94165388373}},
 'trunc2': {'b': {'/sigma': 67.6914898076,
                  'eq1:2.region': 19.5386851008,
                  'eq1:3.region': 13.7332123767,
                  'eq1:_cons': -77.3007729738,
                  'eq1:만성질환수': 25.723985013,
                  'eq1:민간보험': 26.2925147171,
                  'eq1:여성': 8.84093124988,
                  'eq1:연령': 1.28447861194,
                  'eq1:월소득': 2.35052015674},
            'e': {'N': 749.0, 'N_bf': 1000.0, 'chi2': 154.827197903, 'll': -3954.19521115, 'sigma': 67.6914898076},
            'se': {'/sigma': 3.10497731529,
                   'eq1:2.region': 8.8741885115,
                   'eq1:3.region': 9.24560679886,
                   'eq1:_cons': 19.0923943396,
                   'eq1:만성질환수': 2.91323400165,
                   'eq1:민간보험': 6.64035383571,
                   'eq1:여성': 6.39430785964,
                   'eq1:연령': 0.204808716333,
                   'eq1:월소득': 1.98165590044}}}


def test_transpile_accepts_korean_variable_names():
    """Stata commands typed with Korean (Unicode) variable names translate instead of stopping."""
    from pyjamovi.analyses import limdep_models as ld
    line = ld.EXTENSION["transpile"]("heckman 시간당임금 교육연수 경력, select(취업여부 = 연령 미취학자녀수)")
    assert line and not line.startswith("raise") and "'시간당임금'" in line and "'취업여부'" in line
    line = ld.EXTENSION["transpile"]("probit 취업여부 연령 i.지역유형")
    assert line and not line.startswith("raise") and "'지역유형'" in line
    assert ld.EXTENSION["transpile"]("probit 1abc x").startswith("raise NotImplementedError")

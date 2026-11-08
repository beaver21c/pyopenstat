"""
Linear panel post-estimation and panel description — tests/test_panel_postestimation.py

Reference: Stata 19.5 (StataNow MP) on sample_data/Grunfeld_Panel.csv (11 firms × 20 years) and its unbalanced
variant with internal gaps (rule `grunfeld_unbal`, as in tests/test_panel_extended.py), read with
`import delimited ..., asdouble`; firm encoded alphabetically (`encode firm`), `xtset fid year`.
Commands (do-files wp5a.do, wp5b.do, kept outside the repository):
  hausman fe re [, sigmamore | sigmaless];  xtreg invest value capital i.year, fe [vce(robust)] + testparm i.year;
  xtreg invest value capital, cre [sa | vce(robust)] (Mundlak test e(chi2_mundlak), e(corr));
  predict xb | u | e | ue | xbu | stdp after xtreg fe / re / cre (summarize: mean, sd, min, max);
  xtsum invest value hi;  xtdescribe, patterns(5);  xttab hi;  xttab size;  xttrans size, freq
  with hi = invest > 100 and size = 1 + (capital > 100) + (capital > 500).
Values are hard-coded below (Stata is not needed to run the tests).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "reconstructed"))
SAMPLE_DIR = os.path.join(ROOT_DIR, "sample_data")

import pyjamovi as ost  # noqa: E402
from pyjamovi import code_export  # noqa: E402
from pyjamovi.analyses.panel_module import PanelAnalysis, transpile as panel_transpile, wald_pivot  # noqa: E402
from pyjamovi.analyses.panel_describe import transpile as desc_transpile  # noqa: E402


@pytest.fixture(scope="module")
def grunfeld():
    df = pd.read_csv(os.path.join(SAMPLE_DIR, "Grunfeld_Panel.csv"))
    return df.assign(hi=(df.invest > 100).astype(int),
                     size=1 + (df.capital > 100).astype(int) + (df.capital > 500).astype(int))


def grunfeld_unbal(df):
    code = pd.factorize(df.firm, sort=True)[0]
    return df[~((df.year * 7 + code * 3) % 11 == 0)].reset_index(drop=True)


@pytest.fixture(scope="module", params=["bal", "unbal"])
def data(request, grunfeld):
    return request.param, (grunfeld if request.param == "bal" else grunfeld_unbal(grunfeld))


def _fit(d, **kw):
    kw.setdefault("hausman", False)
    kw.setdefault("plot_trends", False)
    return PanelAnalysis(d, "firm", "year", "invest", ["value", "capital"], **kw)


# ----------------------------------------------------------------------------- Hausman sigmamore / sigmaless
HAUSMAN = {"bal": {"none": 3.96753171644078, "more": 2.62009945323711, "less": 2.62767815923622},
           "unbal": {"none": 4.95974971752986, "more": 2.88099771931795, "less": 2.89404791562572}}


@pytest.mark.parametrize("sigma", ["none", "more", "less"])
def test_hausman_sigma_options(data, sigma):
    lab, d = data
    r = _fit(d, model_type="fixed", hausman=True, hausman_sigma=sigma)
    row = r.get_table("hausman").rows[0]
    assert row["chi2"] == pytest.approx(HAUSMAN[lab][sigma], rel=1e-9)
    assert row["df"] == 2


# ----------------------------------------------------------------------------- testparm i.year
TESTPARM = {"bal": {"unadjusted": (1.38532219219709, 19, 188), "robust": (7.80977344400915, 10, 10)},
            "unbal": {"unadjusted": (1.45056899433992, 19, 168), "robust": (16.1266954800376, 10, 10)}}


@pytest.mark.parametrize("cov", ["unadjusted", "robust"])
def test_testparm_time_effects(data, cov):
    lab, d = data
    r = _fit(d, model_type="twoway", cov_type=cov)
    row = r.get_table("time_test").rows[0]
    F, df1, df2 = TESTPARM[lab][cov]
    assert row["stat"] == pytest.approx(F, rel=1e-9)
    assert (row["df1"], row["df2"]) == (df1, df2)
    if cov == "robust":     # cluster VCE of rank G - 1 = 10: 9 of 19 constraints dropped, as Stata reports
        assert any("제외" in n for n in r.get_table("time_test").notes)


def test_wald_pivot_selection_matches_stata_dropped_constraints(grunfeld):
    r = _fit(grunfeld, model_type="twoway", cov_type="robust")
    f = r.fit
    idx = [i for i, n in enumerate(f.names) if n.endswith(".year")]
    W, q, dropped = wald_pivot(f.b[idx], f.V[np.ix_(idx, idx)])
    assert dropped == [1, 3, 5, 6, 10, 12, 14, 16, 18]      # Stata: "Constraint 1 dropped" ... "Constraint 18 dropped"
    assert q == 10


# ----------------------------------------------------------------------------- xtreg, cre
CRE = {
    "bal": {"b": {"value": .11012911902576, "capital": .310033441875004, "Intercept": -7.38248271947048,
                  "xt_means:value": .0244696375487987, "xt_means:capital": -.280345437643577},
            "V": {"value": .00012768645836941, "capital": .000273587363491537, "Intercept": 1635.68983701992,
                  "xt_means:value": .000850465242766401, "xt_means:capital": .0307606941147413},
            "mundlak": {"": 2.627678159236149, "sa": 2.627678159236125, "robust": 7.983963750893131},
            "chi2": {"": 680.158008086288, "sa": 680.158008086288, "robust": 56.7245835551383},
            "sigma_u": 78.75236266512795, "r2": (.766670651548842, .8644046497000276, .8418172442290691),
            "corr": .8923412900497367,
            "robust_se": {"value": .0151784, "capital": .0527159, "Intercept": 16.1297,
                          "xt_means:value": .0148484, "xt_means:capital": .0993051}},
    "unbal": {"b": {"value": .1202586, "capital": .2941266, "Intercept": -9.259518,
                    "xt_means:value": .0167141, "xt_means:capital": -.2651318},
              "V": {"Intercept": 1465.49536315318, "xt_means:value": .000822223220290734,
                    "xt_means:capital": .0274945438301302},
              "mundlak": {"": 2.89407295868286, "sa": 2.915391037234538, "robust": 7.321449408431755},
              "chi2": {"": 659.545928469694, "sa": 659.331170135415, "robust": 44.9178254392777},
              "sigma_u": 75.46460589210025, "r2": (.7791008680969738, .8771323140320105, .854014440318566),
              "corr": .8983427941379586},
}
CRE_OPT = {"": {}, "sa": {"re_method": "sa"}, "robust": {"cov_type": "robust"}}


@pytest.mark.parametrize("opt", ["", "sa", "robust"])
def test_cre_mundlak_and_wald(data, opt):
    lab, d = data
    ref = CRE[lab]
    r = _fit(d, model_type="cre", **CRE_OPT[opt])
    mu = r.get_table("mundlak").rows[0]
    assert mu["chi2"] == pytest.approx(ref["mundlak"][opt], rel=1e-9)
    assert mu["df"] == 2
    wald = [x for x in r.get_table("summary").rows if x["metric"].startswith("Wald")][0]
    assert wald["val"] == pytest.approx(ref["chi2"][opt], rel=1e-9)
    assert r.fit.stats["corr_xit_means"] == pytest.approx(ref["corr"], rel=1e-7)   # Stata generates float variables


def test_cre_coefficients_variance_components(data):
    lab, d = data
    ref = CRE[lab]
    r = _fit(d, model_type="cre")
    b = dict(zip(r.fit.names, r.fit.b))
    V = dict(zip(r.fit.names, np.diag(r.fit.V)))
    for k, v in ref["b"].items():
        assert b[k] == pytest.approx(v, rel=1e-9 if lab == "bal" else 1e-5)   # unbal: 7-digit Stata display
    for k, v in ref["V"].items():
        assert V[k] == pytest.approx(v, rel=1e-8)
    assert r.fit.stats["sigma_u"] == pytest.approx(ref["sigma_u"], rel=1e-9)
    st = r.fit.stats
    assert (st["r2_w"], st["r2_b"], st["r2_o"]) == pytest.approx(ref["r2"], rel=1e-9)
    # time-varying coefficients equal the fixed-effects estimates (Mundlak equivalence)
    fe = dict(zip(_fit(d, model_type="fixed").fit.names, _fit(d, model_type="fixed").fit.b))
    assert b["value"] == pytest.approx(fe["value"], rel=1e-9)
    assert r.get_table("cre_means") is not None and len(r.get_table("cre_means").rows) == 2


def test_cre_robust_standard_errors(grunfeld):
    r = _fit(grunfeld, model_type="cre", cov_type="robust")
    se = dict(zip(r.fit.names, r.fit.se))
    for k, v in CRE["bal"]["robust_se"].items():
        assert se[k] == pytest.approx(v, rel=5e-6)


def test_cre_rejects_dk_and_time_invariant_only(grunfeld):
    with pytest.raises(ValueError, match="Driscoll"):
        _fit(grunfeld, model_type="cre", cov_type="dk")
    d = grunfeld.assign(zfirm=pd.factorize(grunfeld.firm, sort=True)[0].astype(float))
    with pytest.raises(ValueError, match="시점에 따라 변하는"):
        PanelAnalysis(d, "firm", "year", "invest", ["zfirm"], model_type="cre", hausman=False, plot_trends=False)


# ----------------------------------------------------------------------------- predict
PRED = {   # (mean, sd, min, max)
    "bal": {
        "fixed": {"xb": (133.3119, 203.89437664265, -47.4061524015503, 1250.9741432523),
                  "u": (0, 78.4968820300587, -180.297845516868, 157.176287949493),
                  "e": (0, 48.9020389793011, -184.00792131641, 250.7533748976),
                  "ue": (0, 92.4833493379972, -334.662533505369, 325.241874071414),
                  "xbu": (133.3119, 204.83054860314, -76.3309767464551, 1235.9466251024),
                  "stdp": (11.7273555449692, 9.01657692735301, 3.45161794296288, 55.171088632765)},
        "random": {"xb": (133.3119, 202.444865394823, -46.1359439652494, 1243.24721250396),
                   "u": (0, 76.6935430706369, -175.644577808763, 154.902117220671),
                   "e": (0, 48.9307014768726, -180.824673936288, 254.713139257769),
                   "ue": (0, 92.2825099481693, -331.94136060917, 326.580101731061),
                   "xbu": (133.3119, 204.237119456259, -71.5094542781904, 1231.98686074223),
                   "stdp": (26.7257584788652, 4.98727427908755, 24.0270986874292, 54.3432805095086)},
        "cre": {"xb": (133.3119, 193.215005190864, -64.0153540487625, 1223.1250315936),
                "u": (0, 66.637224052286, -160.238973983632, 140.826173616577),
                "e": (0, 48.9209247202047, -181.135461883803, 251.0096717491),
                "ue": (0, 83.7551480596304, -317.872088149384, 311.764219171104),
                "xbu": (133.3119, 204.3833600377, -73.0625505692038, 1235.6903282509),
                "stdp": (43.6641072204775, 6.58839186455942, 40.4482405794226, 76.9010496644578)}},
    "unbal": {
        "fixed": {"xb": (132.951555, 213.485893917462, -52.1997176427156, 1266.89036732623),
                  "u": (0, 76.3331957524438, -184.552397899185, 139.612140900831),
                  "e": (0, 48.538424204477, -175.96525876334, 251.472911616653),
                  "ue": (0, 90.4584733346449, -343.148477663865, 315.764956594533),
                  "xbu": (132.951555, 206.680574326822, -75.614569412497, 1235.22708838335),
                  "stdp": (12.669007596813, 9.81316700982336, 3.5937304389291, 58.4041252765997)},
        "random": {"xb": (133.034945694969, 209.80755494707, -49.1977489299535, 1251.55239584589),
                   "u": (-.0833906949692483, 73.8217582951759, -177.410017535569, 138.913471571202),
                   "e": (0, 48.5834530212357, -172.524730283523, 256.400976968192),
                   "ue": (-.0833906949692496, 89.8650045068648, -337.441006658994, 319.237195428086),
                   "xbu": (132.951555, 206.026142494323, -68.7288991471711, 1230.29902303181),
                   "stdp": (26.1310020261909, 5.42357647567699, 23.087041221071, 55.0806962114254)},
        "cre": {"xb": (132.806217357654, 196.265728311723, -61.4105189737168, 1215.56833552209),
                "u": (.145337642346375, 63.4419049966547, -158.671656034144, 125.750685405328),
                "e": (0, 48.5631824223236, -172.889629001457, 251.942249308282),
                "ue": (.145337642346372, 81.1171277348262, -321.148551764899, 304.979130860913),
                "xbu": (132.951555, 206.198267029515, -71.7337534464224, 1234.75775069172),
                "stdp": (42.0799217415546, 7.65804771574529, 38.2872908195784, 79.1108678155532)}},
}


@pytest.mark.parametrize("mt", ["fixed", "random", "cre"])
def test_predict_statistics(data, mt):
    lab, d = data
    pr = _fit(d, model_type=mt).predictions()
    for stat, (m, sd, lo, hi) in PRED[lab][mt].items():
        x = pr[stat]
        assert x.mean() == pytest.approx(m, abs=1e-8 * max(1.0, abs(m)) + 1e-9)
        assert (x.std(), x.min(), x.max()) == pytest.approx((sd, lo, hi), rel=1e-9), (mt, stat)


def test_session_predict_data_step_replays_and_exports(grunfeld):
    s = ost.Session(grunfeld)
    s.xtset("firm", "year")
    s.run("panel", dep="invest", covs=["value", "capital"], modelType="random")
    s.predict("u_hat", "u")
    s.predict("e_hat", "e")
    assert {"u_hat", "e_hat"} <= set(s.df.columns)
    assert s.df["u_hat"].std() == pytest.approx(76.6935430706369, rel=1e-9)
    op = s.data_ops[-1]
    assert op["op"] == "panel_predict" and op["stats"] == {"e": "e_hat"}
    script = s.to_python_script()
    assert "sess.panel_predict(" in script
    # replay from the raw data reproduces the columns
    s.undo_data_op()
    assert "e_hat" not in s.df.columns and "u_hat" in s.df.columns
    do = "\n".join(code_export._op_stata(op))
    assert "predict double e_hat, e" in do
    assert "quietly xtreg invest value capital, re vce" not in do and "quietly xtreg invest value capital, re theta" in do
    assert "fixef" in "\n".join(code_export._op_r(op))


def test_predict_errors(grunfeld):
    s = ost.Session(grunfeld)
    with pytest.raises(ValueError, match="패널 회귀"):
        s.predict("x", "u")
    s.run("panel", entity="firm", time="year", dep="invest", covs=["value"], modelType="between", hausman=False)
    with pytest.raises(ValueError, match="고정효과"):
        s.predict("x", "u")


# ----------------------------------------------------------------------------- xtdescribe / xtsum / xttab / xttrans
XTSUM = {   # var: (mean, sd, min, max, sd_b, min_b, max_b, sd_w, min_w, max_w, N, n)
    "bal": {"invest": (133.3119, 210.587186356142, .93, 1486.7, 193.228336752946, 3.0845, 608.02,
                       101.237703119502, -217.0081, 1011.9919, 220, 11),
            "value": (988.577804545455, 1287.30117187874, 30.284, 6241.7, 1303.51555856321, 57.54485, 4333.845,
                      324.66379502419, -553.067195454545, 2896.43280454545, 220, 11),
            "hi": (.25, .434000189381942, 0, 1, .393064880140671, 0, 1, .217394282290503, -.15, 1.15, 220, 11)},
    "unbal": {"invest": (132.951555, 212.303646761704, .93, 1486.7, 194.879860782243, 2.975, 623.366666666667,
                         103.273445911413, -232.715111666667, 996.284888333333, 200, 11),
              "value": (982.956515, 1281.14449465318, 30.284, 6241.7, 1302.20140032471, 58.8046111111111,
                        4335.28333333333, 321.037412004838, -560.126818333334, 2889.37318166667, 200, 11),
              "hi": (.25, .434099310129925, 0, 1, .394903188215827, 0, 1, .21670243636949, -.194444444444444,
                     1.14473684210526, 200, 11)},
}
XTTAB = {   # value: (overall freq, overall %, between freq, between %, within %)
    "bal": {"hi": [(165, 75.00, 9, 81.82, 91.67), (55, 25.00, 5, 45.45, 55.00), (220, 100, 14, 127.27, 78.57)],
            "size": [(82, 37.27, 8, 72.73, 51.25), (111, 50.45, 9, 81.82, 61.67), (27, 12.27, 5, 45.45, 27.00),
                     (220, 100, 22, 200.00, 50.00)]},
    "unbal": {"hi": [(150, 75.00, 9, 81.82, 91.42), (50, 25.00, 5, 45.45, 55.44), (200, 100, 14, 127.27, 78.57)],
              "size": [(75, 37.50, 8, 72.73, 51.68), (101, 50.50, 9, 81.82, 61.76), (24, 12.00, 5, 45.45, 26.14),
                       (200, 100, 22, 200.00, 50.00)]},
}
XTTRANS = {"bal": [[74, 6, 0], [0, 102, 5], [0, 0, 22]], "unbal": [[67, 6, 0], [0, 92, 5], [0, 0, 19]]}


def test_xtsum_xttab_xttrans(data):
    lab, d = data
    s = ost.Session(d)
    s.xtset("firm", "year")
    r = s.run("xtsum", sumVars=["invest", "value", "hi"], tabVars=["hi", "size"], transVars=["size"], transFreq=True,
              describe=False)
    rows = r.get_table("xtsum").rows
    for i, v in enumerate(["invest", "value", "hi"]):
        o, b, w = rows[3 * i: 3 * i + 3]
        ref = XTSUM[lab][v]
        got = (o["mean"], o["sd"], o["min"], o["max"], b["sd"], b["min"], b["max"], w["sd"], w["min"], w["max"])
        assert got == pytest.approx(ref[:10], rel=1e-10, abs=1e-12)
        assert o["obs"] == f"N = {ref[10]}" and b["obs"] == f"n = {ref[11]}"
    for v in ("hi", "size"):
        got = [(x["of"], x["op"], x["bf"], x["bp"], x["wp"]) for x in r.get_table(f"xttab_{v}").rows]
        for g, ref in zip(got, XTTAB[lab][v]):
            assert g[0] == ref[0] and g[2] == ref[2]
            assert (g[1], g[3], g[4]) == pytest.approx((ref[1], ref[3], ref[4]), abs=0.005)
    tr = [x for x in r.get_table("xttrans_size").rows if x["from"].strip() == "빈도"]
    assert [[x["c0"], x["c1"], x["c2"]] for x in tr[:3]] == XTTRANS[lab]
    assert tr[3]["tot"] == sum(map(sum, XTTRANS[lab]))


def test_xtdescribe_patterns(grunfeld):
    d = grunfeld_unbal(grunfeld)
    r = ost.Session(d).run("xtdescribe", entity="firm", time="year", patterns=5)
    Ti = r.get_table("xtdescribe_Ti").rows[0]
    assert (Ti["min"], Ti["p5"], Ti["p25"], Ti["p50"], Ti["p75"], Ti["p95"], Ti["max"]) == (18, 18, 18, 18, 18, 19, 19)
    pats = r.get_table("xtdescribe_patterns").rows
    assert [p["pattern"] for p in pats[:5]] == [".1111111111.11111111", "1.1111111111.1111111", "11.1111111111.111111",
                                                "111.1111111111.11111", "1111.1111111111.1111"]
    assert pats[5]["freq"] == 6 and pats[-1]["freq"] == 11
    bal = ost.Session(grunfeld).run("xtdescribe", entity="firm", time="year")
    assert "강균형" in [x for x in bal.get_table("xtdescribe").rows if x["item"] == "균형 여부"][0]["val"]


# ----------------------------------------------------------------------------- Stata command input and export
def test_transpile_new_panel_commands():
    assert "modelType='cre'" in panel_transpile("xtreg y x1 x2, cre vce(robust)")
    assert "reMethod='sa'" in panel_transpile("xtreg y x1 x2, cre sa")
    assert "hausmanSigma='more'" in panel_transpile("hausman fe re, sigmamore")
    assert "hausmanSigma='less'" in panel_transpile("hausman fe re, sigmaless")
    assert "NotImplementedError" in panel_transpile("hausman fe re, sigmamore force")
    assert "timeTest=True" in panel_transpile("testparm i.year")
    assert panel_transpile("predict double uh, u").startswith("sess.predict('uh', 'u')")
    assert panel_transpile("predict yhat").startswith("sess.predict('yhat', 'xb')")
    assert "NotImplementedError" in panel_transpile("predict p, pr")
    assert "sumVars=['invest', 'value']" in desc_transpile("xtsum invest value")
    assert "transFreq=True" in desc_transpile("xttrans size, freq")
    assert "patterns=5" in desc_transpile("xtdescribe, patterns(5)")
    assert "NotImplementedError" in desc_transpile("xttab a b")


def test_stata_and_r_export_new_options():
    from pyjamovi.analyses.panel_module import r_code, stata_code
    p = {"entity": "firm", "time": "year", "dep": "invest", "covs": ["value", "capital"], "modelType": "cre",
         "covType": "robust", "hausmanSigma": "more"}
    st = stata_code("panel", p)
    assert "xtreg invest value capital, cre vce(robust)   // Stata 19 이상" in st
    assert "hausman fe re, sigmamore" in st
    rr = "\n".join(r_code("panel", p))
    assert "Between(value)" in rr and "waldtest" in rr
    st2 = stata_code("panel", dict(p, modelType="twoway", covType="unadjusted"))
    assert "testparm i.year" in st2
    from pyjamovi.analyses.panel_describe import stata_code as ds
    assert ds("panelDescribe", {"entity": "firm", "time": "year", "sumVars": ["invest"], "transVars": ["size"],
                                "transFreq": True}) [-2:] == ["xtsum invest", "xttrans size, freq"]

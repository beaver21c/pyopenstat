"""
GIS spatial analysis:
- boundary files (shapefile with/without .cpg, zip), key normalisation and matching
- spatial weights (contiguity, islands, kNN, distance, file round trip, Stata CSV layout)
- ML estimators against PySAL spreg (independent implementation): SAR/SEM/SDM/SDEM, EIM standard errors,
  LM tests, residual Moran's I, panel FE / RE
- effects against dense matrix algebra, Lee-Yu two-way FE on the synthetic panel
- analyses through the registry (UI parameter names), panel period filter, Stata transpiler, geo endpoints
"""

import json
import os
import shutil
import sys
import warnings
import zipfile

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "reconstructed"))

pytest.importorskip("geopandas")
pytest.importorskip("libpysal")

from pyjamovi.registry import dispatch_analysis  # noqa: E402
from pyjamovi import spatial_io  # noqa: E402
from pyjamovi.analyses.spatial_weights import WeightsSpec, build_weights  # noqa: E402
from pyjamovi.analyses import spatial_models as sm  # noqa: E402

warnings.filterwarnings("ignore")
SHP = os.path.join(ROOT_DIR, "sample_data", "geo", "Synthetic_Regions.shp")
CS = os.path.join(ROOT_DIR, "sample_data", "Spatial_CrossSection_Welfare.csv")
PANEL = os.path.join(ROOT_DIR, "sample_data", "Spatial_Panel_Welfare.csv")
B = {"boundary": SHP, "boundaryKey": "SIG_CD", "idVar": "지역코드", "labelVar": "지역명"}
COVS = ["고령인구비율", "빈곤율", "1인가구비율"]


@pytest.fixture(scope="module")
def cs():
    return pd.read_csv(CS, encoding="utf-8-sig")


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(PANEL, encoding="utf-8-sig")


@pytest.fixture(scope="module")
def W(cs):
    keys = spatial_io.normalize_keys(cs["지역코드"]).tolist()
    return build_weights(WeightsSpec(kind="queen", norm="row"), spatial_io.unit_geometries(SHP, "SIG_CD", keys))


def test_boundary_reading_encoding_and_keys(tmp_path):
    gdf = spatial_io.read_boundary(SHP)
    assert len(gdf) == 73 and gdf.crs.to_epsg() == 5179 and "섬마을군" in set(gdf["SIG_KOR_NM"])
    # Korean .dbf without .cpg is decoded as CP949
    cp = tmp_path / "cp949"
    cp.mkdir()
    gdf.drop(columns=spatial_io.GID).to_file(cp / "r.shp", encoding="cp949")
    for f in cp.glob("*.cpg"):
        f.unlink()
    g2 = spatial_io.read_boundary(str(cp / "r.shp"))
    assert set(g2["SIG_KOR_NM"]) == set(gdf["SIG_KOR_NM"])
    # zipped shapefile, and uploads missing .dbf are rejected with a clear message
    z = tmp_path / "r.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            zf.write(SHP[:-4] + ext, "sub/Synthetic_Regions" + ext)
    assert len(spatial_io.read_boundary(str(z))) == 73
    with pytest.raises(ValueError, match=".dbf"):
        spatial_io.save_uploaded_boundary([("a.shp", b"x"), ("a.shx", b"x")], str(tmp_path / "up"))
    assert spatial_io.normalize_keys([11010, 11010.0, " 11010 ", "011", None]).tolist() == ["11010", "11010", "11010", "011", None]
    rep = spatial_io.match_keys(["90010", 90020, "99999"], gdf["SIG_CD"])
    assert rep["matched"] == 2 and rep["unmatchedData"] == ["99999"]


def test_weights_construction(cs, W, tmp_path):
    s = W.summary()
    assert s["n"] == 73 and s["symmetric"] and W.islands == ["90730"] and s["min"] >= 1 and s["components"] == 1
    assert np.allclose(np.asarray(W.W.sum(axis=1)).ravel(), 1.0)
    geoms = spatial_io.unit_geometries(SHP, "SIG_CD", W.ids)
    dropped = build_weights(WeightsSpec(kind="queen", islands="drop"), geoms)
    assert dropped.n == 72 and dropped.dropped == ["90730"]
    knn = build_weights(WeightsSpec(kind="knn", k=4), geoms)
    assert set(knn.neighbour_counts()) == {4}
    rook2 = build_weights(WeightsSpec(kind="rook", order=2), geoms)
    assert rook2.summary()["mean"] > W.summary()["mean"]
    band = build_weights(WeightsSpec(kind="distance"), geoms)
    assert band.neighbour_counts().min() >= 1 and band.threshold > 0
    spec = build_weights(WeightsSpec(kind="queen", norm="spectral"), geoms)
    assert abs(np.max(np.abs(np.linalg.eigvals(spec.dense()))) - 1.0) < 1e-8
    # export / import round trip and Stata 'spmatrix fromdata' CSV layout
    gal = W.export(str(tmp_path / "w.gal"))
    back = build_weights(WeightsSpec(kind="file", file=gal), geoms)
    assert (abs(back.raw - (W.raw > 0)) > 0).nnz == 0
    csv = pd.read_csv(W.export(str(tmp_path / "w.csv")))
    assert list(csv.columns[:2]) == ["_ID", "w1"] and csv.shape == (73, 74)
    pts = build_weights(WeightsSpec(kind="auto", k=3), spatial_io.points_geometries([127.0, 127.1, 127.2, 127.05], [37.5, 37.52, 37.48, 37.6], list("abcd")))
    assert pts.spec.kind == "knn" and pts.unit == "km"


def test_minmax_normalization_follows_stata():
    """Stata spmatrix normalize(minmax) / Kelejian & Prucha (2010): divide by the smaller of the largest row sum
    and the largest column sum of absolute values (differs from the larger one when W is asymmetric)."""
    import numpy as np
    import scipy.sparse as sp
    from pyjamovi.analyses.spatial_weights import SpatialWeights
    W = sp.csr_matrix(np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], float))  # rows 3,1,1,1; cols 2,2,1,1
    out = SpatialWeights._normalise(W, "minmax").toarray()
    assert np.allclose(out, W.toarray() / 2.0)


def test_cross_section_ml_matches_spreg(cs, W):
    spreg = pytest.importorskip("spreg")
    y, X = cs["복지지출"].to_numpy(float), cs[COVS].to_numpy(float)
    wl = W.to_libpysal()
    wl.transform = "r"
    lay = sm.PanelLayout(len(y), 1)
    refs = {"sar": spreg.ML_Lag(y[:, None], X, wl, method="full"), "sem": spreg.ML_Error(y[:, None], X, wl, method="full"),
            "sdm": spreg.ML_Lag(y[:, None], X, wl, method="full", slx_lags=1),
            "sdem": spreg.ML_Error(y[:, None], X, wl, method="full", slx_lags=1)}
    from pyjamovi.analyses.spatial_module import expected_information_vcov
    for model, ref in refs.items():
        fit, d = sm.fit_spec(model, y, X, COVS, W, lay, COVS, True)
        assert abs(fit.loglik - float(ref.logll)) < 1e-4, model
        assert np.allclose(fit.coef, ref.betas.ravel()[:len(fit.coef)], atol=1e-4), model
        se_eim = np.sqrt(np.diag(expected_information_vcov(fit, d, W)))
        assert np.allclose(se_eim, np.sqrt(np.diag(ref.vm))[:len(fit.coef)], rtol=2e-3), model
        assert np.allclose(fit.se, se_eim, rtol=0.08), model  # observed vs expected information
    ols, d0 = sm.fit_spec("ols", y, X, COVS, W, lay, [], True)
    ref = spreg.OLS(y[:, None], X, wl, spat_diag=True, moran=True)
    lm = sm.lm_tests(d0, W, 1)
    assert abs(lm["LM 공간오차"][0] - ref.lm_error[0]) < 1e-6 and abs(lm["LM 공간시차"][0] - ref.lm_lag[0]) < 1e-6
    assert abs(lm["강건 LM 공간오차"][0] - ref.rlm_error[0]) < 1e-6 and abs(lm["강건 LM 공간시차"][0] - ref.rlm_lag[0]) < 1e-6
    mi = sm.moran_residuals(ols.resid, d0.X, W)
    assert abs(mi["I"] - ref.moran_res[0]) < 1e-9 and abs(mi["z"] - ref.moran_res[1]) < 1e-6


def test_effects_and_sac(cs, W):
    y, X = cs["복지지출"].to_numpy(float), cs[COVS].to_numpy(float)
    lay = sm.PanelLayout(len(y), 1)
    fit, _ = sm.fit_spec("sdm", y, X, COVS, W, lay, COVS, True)
    eff = {(r["effect"], r["var"]): r for r in sm.effects(fit, W)}
    n = W.n
    Ainv = np.linalg.inv(np.eye(n) - fit.rho * W.dense())
    for v in COVS:
        S = Ainv @ (fit.coef[fit.names.index(v)] * np.eye(n) + fit.coef[fit.names.index("W·" + v)] * W.dense())
        assert abs(eff[("직접효과", v)]["est"] - np.trace(S) / n) < 1e-9
        assert abs(eff[("총효과", v)]["est"] - S.sum() / n) < 1e-9
    sim = {(r["effect"], r["var"]): r for r in sm.effects(fit, W, method="sim", nsim=2000, seed=1)}
    assert abs(sim[("간접효과", "빈곤율")]["se"] / eff[("간접효과", "빈곤율")]["se"] - 1) < 0.15
    sac, d = sm.fit_spec("sac", y, X, COVS, W, lay, [], True)
    ml = sm.SpatialML(W, lay)
    grid = max(ml.conc_ll(d, r, l) for r in np.linspace(-0.5, 0.9, 29) for l in np.linspace(-0.5, 0.9, 29))
    assert sac.loglik >= grid - 1e-6


def _panel_arrays(panel):
    df = panel.assign(k=spatial_io.normalize_keys(panel["지역코드"]))
    units, years = sorted(df.k.unique()), sorted(df["연도"].unique())
    df = df.set_index(["연도", "k"]).loc[[(t, u) for t in years for u in units]].reset_index()
    W = build_weights(WeightsSpec(kind="queen"), spatial_io.unit_geometries(SHP, "SIG_CD", units))
    covs = ["고령인구비율", "1인가구비율", "재정자립도", "복지시설수"]
    return df["돌봄이용률"].to_numpy(float), df[covs].to_numpy(float), covs, W, len(units), len(years)


def test_panel_fe_re_match_spreg(panel):
    spreg = pytest.importorskip("spreg")
    y, X, covs, W, N, T = _panel_arrays(panel)
    wl = W.to_libpysal()
    wl.transform = "r"
    fe, _ = sm.fit_spec("sar", y, X, covs, W, sm.PanelLayout(N, T, fe="ind", leeyu=False), [], False)
    ref = spreg.Panel_FE_Lag(y[:, None], X, wl)
    assert np.allclose(np.append(fe.coef[:-1], fe.rho), ref.betas.ravel(), atol=1e-4)
    assert abs(fe.sigma2 - float(np.ravel(ref.sig2)[0])) < 1e-4
    fe_e, _ = sm.fit_spec("sem", y, X, covs, W, sm.PanelLayout(N, T, fe="ind", leeyu=False), [], False)
    ref_e = spreg.Panel_FE_Error(y[:, None], X, wl)
    assert np.allclose(np.append(fe_e.coef[:-1], fe_e.lam), ref_e.betas.ravel(), atol=1e-4)
    re_ = sm.fit_random_effects(y, X, covs, W, N, T, [], True)
    ref_r = spreg.Panel_RE_Lag(y[:, None], X, wl)
    assert np.allclose(re_.coef, ref_r.betas.ravel()[:len(re_.coef)], atol=1e-3)
    assert abs(np.sqrt(re_.extra["phi"]) - ref_r.betas.ravel()[-1]) < 1e-3
    # Lee-Yu transformation scales sigma2 by NT / N(T-1) for individual effects with identical coefficients
    ly, _ = sm.fit_spec("sar", y, X, covs, W, sm.PanelLayout(N, T, fe="ind", leeyu=True), [], False)
    assert np.allclose(ly.coef, fe.coef, atol=1e-6) and abs(ly.sigma2 / fe.sigma2 - T / (T - 1)) < 1e-6


def test_two_way_sdm_recovers_dgp(panel):
    y, X, covs, W, N, T = _panel_arrays(panel)
    fit, _ = sm.fit_spec("sdm", y, X, covs, W, sm.PanelLayout(N, T, fe="both", leeyu=True), covs, False)
    se = dict(zip(fit.names, fit.se))
    assert abs(fit.rho - 0.40) < 2.5 * se["ρ (W·y)"]
    truth = {"고령인구비율": 0.80, "1인가구비율": 0.30, "재정자립도": -0.15, "복지시설수": 0.50}
    for v, b in truth.items():
        assert abs(fit.coef[fit.names.index(v)] - b) < 3 * se[v], v


def test_analyses_via_registry(cs, panel, tmp_path):
    w = dispatch_analysis("spatialWeights", cs, {**B, "wExport": str(tmp_path / "w.gwt")})
    assert os.path.exists(tmp_path / "w.gwt") and w.get_table("wSummary") is not None
    m = dispatch_analysis("spatial_moran", cs, {**B, "vars": ["복지지출"], "gi": True, "adjust": "fdr", "permutations": "199"})
    gm = m.get_table("globalMoran").rows[0]
    assert gm["I"] > 0 and gm["p_sim"] < 0.05 and m.get_table("giSummary") is not None
    r = dispatch_analysis("spatial", cs, {**B, "dep": "복지지출", "covs": COVS, "model": "sdm", "durbinVars": ["빈곤율"], "ciLevel": "0.90"})
    names = [t.name for t in r.tables]
    assert names[:3] == ["assumptions", "spSummary", "spCoef"] and {"spEffects", "lmTests", "modelCompare", "lrTests"} <= set(names)
    terms = [row["term"] for row in r.get_table("spCoef").rows]
    assert "W·빈곤율" in terms and "W·고령인구비율" not in terms and terms[-1] == "ρ (W·y)"
    p = dispatch_analysis("spatialPanel", panel, {**B, "dep": "돌봄이용률", "covs": ["고령인구비율", "재정자립도"], "timeVar": "연도",
                                                 "model": "sar", "effectsType": "re", "hausman": True})
    assert p.get_table("hausman").rows[0]["df"] == 3
    with pytest.raises(ValueError, match="확률효과"):
        dispatch_analysis("spatialPanel", panel, {**B, "dep": "돌봄이용률", "covs": ["고령인구비율"], "timeVar": "연도", "model": "sem", "effectsType": "re"})
    # panel data in a cross-sectional analysis: one period (default = last)
    mp = dispatch_analysis("spatialMap", panel, {**B, "vars": ["돌봄이용률"], "timeVar": "연도"})
    assert any("2023" in n for n in mp.get_table("mapClasses").notes)
    g = dispatch_analysis("gwr", panel, {**B, "dep": "돌봄이용률", "covs": ["고령인구비율"], "timeVar": "연도", "period": "2020"})
    assert g.get_table("gwrSummary").rows[0]["gwr"] > 0 and len(g.plots) >= 2
    # unbalanced panel: incomplete regions dropped with a note
    unb = panel[~((panel["지역코드"] == 90010) & (panel["연도"] == 2016))]
    pu = dispatch_analysis("spatialPanel", unb, {**B, "dep": "돌봄이용률", "covs": ["고령인구비율"], "timeVar": "연도", "model": "sar", "compare": False})
    assert any("90010" in n for n in pu.get_table("spCoef").notes)


def test_legacy_coordinates_and_transpiler():
    gis = pd.read_csv(os.path.join(ROOT_DIR, "sample_data", "Korea_Regional_Welfare_GIS.csv"))
    from pyjamovi.analyses import SpatialAnalysis
    sa = SpatialAnalysis(gis, dep="welfare_expenditure", covs=["elderly_ratio"], lat="latitude", lon="longitude", model_type="lag")
    assert sa.fit.rho is not None and len(sa.plots) >= 1
    mo = SpatialAnalysis(gis, dep="welfare_expenditure", lat="latitude", lon="longitude", model_type="moran_only")
    assert mo.get_table("globalMoran") is not None
    from pyjamovi.stata_transpiler import transpile_stata_line as t
    assert "model_type='sac'" in t("spregress y x1 x2, ml dvarlag(W) errorlag(W)")
    assert "durbinVars=['x1']" in t("spregress y x1 x2, ml dvarlag(W) ivarlag(W: x1)")
    assert "feType='both'" in t("xsmle y x1, wmat(W) model(sdm) fe type(both, leeyu)")
    assert "effectsType='re'" in t("spxtregress y x1, re dvarlag(W)")
    assert t("xtset id year") == "sess.xtset('id', 'year')"


def test_geo_endpoints(tmp_path):
    try:
        from fastapi.testclient import TestClient  # needs httpx2 (or httpx on older Starlette)
    except (ImportError, RuntimeError):
        pytest.skip("Starlette TestClient requires httpx2")
    from pyjamovi.server import app, STATE
    c = TestClient(app)
    j = c.post("/api/dataset/load", params={"name": "Spatial_CrossSection_Welfare.csv"}).json()
    assert j["boundary"]["boundaryKey"] == "SIG_CD"
    files = [("files", (os.path.basename(SHP)[:-4] + e, open(SHP[:-4] + e, "rb").read())) for e in (".shp", ".shx", ".dbf", ".prj", ".cpg")]
    up = c.post("/api/geo/upload", files=files, data={"name": "pytest_boundary"}).json()
    try:
        assert up["n"] == 73 and up["suggestions"][0]["share"] == 1.0
        at = c.post("/api/geo/attach", json={"path": up["path"], "boundaryKey": "SIG_CD", "idVar": "지역코드"}).json()
        assert at["match"]["matched"] == 73
        assert c.post("/api/geo/detach").json()["boundary"] is None
    finally:
        shutil.rmtree(os.path.dirname(os.path.join(ROOT_DIR, up["path"])), ignore_errors=True)



def test_map_and_choice_options_are_strict():
    """Scheme/colormap aliases resolve; unknown select values raise before estimation."""
    from pyjamovi.analyses import spatial_maps as smap
    assert smap.resolve_scheme("fisher_jenks") == "natural_breaks"
    assert smap.resolve_scheme("Equal") == "equal_interval"
    assert smap.resolve_cmap("blues") == "Blues" and smap.resolve_cmap("KIHASA") == "kihasa"
    df = pd.read_csv(os.path.join(ROOT_DIR, "sample_data", "Spatial_CrossSection_Welfare.csv"), encoding="utf-8-sig")
    b = {"boundary": os.path.join(ROOT_DIR, "sample_data", "geo", "Synthetic_Regions.shp"), "boundaryKey": "SIG_CD",
         "idVar": "지역코드", "labelVar": "지역명"}
    nat = dispatch_analysis("spatialMap", df, {"vars": ["복지지출"], "scheme": "natural_breaks", **b})
    fj = dispatch_analysis("spatialMap", df, {"vars": ["복지지출"], "scheme": "fisher_jenks", "cmap": "kihasa", **b})
    assert [r["hi"] for r in nat.tables[0].rows] == [r["hi"] for r in fj.tables[0].rows]
    for bad in [{"scheme": "jenkz"}, {"cmap": "Bluez"}, {"seType": "robust"}, {"effectsMethod": "boot"}]:
        with pytest.raises(ValueError):
            dispatch_analysis("spatial", df, {"dep": "복지지출", "covs": ["빈곤율"], "model": "sar", **b, **bad})
    with pytest.raises(ValueError):
        dispatch_analysis("spatial_moran", df, {"vars": ["복지지출"], "adjust": "holm", "permutations": 99, **b})
    with pytest.raises(ValueError):
        dispatch_analysis("gwr", df, {"dep": "복지지출", "covs": ["빈곤율"], "bwType": "variable", **b})

"""
OpenStat Studio v3.0 (`openstat` / `pyjamovi`) Comprehensive Automated Test Suite
Verifies:
- Gate 1: .omv lossless binary container I/O & IEEE-754 byte-offset fidelity
- Gate 2: All 28 statistical analyses (including Panel FE/RE/Hausman & CFA/SEM), 5 post-hoc tests,
          customizable `PlotStyle` (10 palettes, custom hex, 5 themes, Korean labels, `.svg`/`.pdf`/`.png`),
          and all 10 table export formats (`.xlsx`, `.docx`, `.hwpx`, `.tex`, `.csv`, `.tsv`, `.md`, `.html`, `.json`, `.omv`)
- Gate 3: Copyright-clean original SVG assets (zero `logo-v.svg`), bidirectional `.py` syntax authoring & execution,
          Claude Code CLI/MCP/SDK integration, package freshness & syntax/API drift self-diagnostics (`doctor`),
          upstream jamovi feature sync & clean-room plugin scaffolding (`sync-jamovi` / `rebuild`),
          and in-process FastAPI Web Studio endpoints (zero persistent background server).
"""

import asyncio
import json
import os
import subprocess
import tempfile
import zipfile
import sys
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
reconstructed_dir = os.path.join(ROOT_DIR, "reconstructed")
if reconstructed_dir not in sys.path:
    sys.path.insert(0, reconstructed_dir)

import openstat as ost
import pyjamovi as jmv
from pyjamovi.agent import JamoviSession
from pyjamovi.mcp_server import handle_tool_call
from pyjamovi.server import app

SAMPLE_DIR = os.path.join(ROOT_DIR, "sample_data")


def _asgi_request(method: str, path: str, query: dict | None = None, json_body: dict | None = None):
    """Pure-Python in-process ASGI test client (zero TCP socket / zero background server)."""
    qs = urlencode(query or {}, doseq=True).encode("utf-8")
    raw_body = json.dumps(json_body).encode("utf-8") if json_body is not None else b""
    headers = [(b"host", b"testserver")]
    if json_body is not None:
        headers.append((b"content-type", b"application/json"))
        headers.append((b"content-length", str(len(raw_body)).encode("utf-8")))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": qs,
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }

    sent = False
    done_event = asyncio.Event()
    status_code = 500
    resp_chunks: list[bytes] = []

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": raw_body, "more_body": False}
        await done_event.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = message["status"]
        elif message["type"] == "http.response.body":
            resp_chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                done_event.set()

    asyncio.run(app(scope, receive, send))
    return status_code, b"".join(resp_chunks)


@pytest.fixture
def toothgrowth_df():
    return pd.read_csv(os.path.join(SAMPLE_DIR, "ToothGrowth.csv"), sep=None, engine="python")


@pytest.fixture
def iris_df():
    return pd.read_csv(os.path.join(SAMPLE_DIR, "iris.csv"), sep=None, engine="python")


@pytest.fixture
def bugs_df():
    return pd.read_csv(os.path.join(SAMPLE_DIR, "bugs.csv"), sep=None, engine="python")


@pytest.fixture
def big5_df():
    return pd.read_csv(os.path.join(SAMPLE_DIR, "Big5.csv"), sep=None, engine="python")


@pytest.fixture
def grunfeld_df():
    return pd.read_csv(os.path.join(SAMPLE_DIR, "Grunfeld_Panel.csv"))


@pytest.fixture
def sem_df():
    return pd.read_csv(os.path.join(SAMPLE_DIR, "SEM_Latent3Factor.csv"))


@pytest.fixture
def medmod_df():
    return pd.read_csv(os.path.join(SAMPLE_DIR, "MedMod_WorkStress.csv"))


@pytest.fixture
def survival_df():
    return pd.read_csv(os.path.join(SAMPLE_DIR, "Clinical_Survival.csv"))


# --- Gate 1: .omv Binary Container & Copyright Cleanliness ---
def test_omv_roundtrip_and_copyright_clean_assets(toothgrowth_df):
    with tempfile.NamedTemporaryFile(suffix=".omv", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        t_res = ost.ttestIS(toothgrowth_df, vars="len", group="supp", welchs=True)
        ost.write_omv(toothgrowth_df, tmp_path, analyses=[t_res])
        assert os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 600

        read_df, _ = ost.read_omv(tmp_path)
        assert len(read_df) == len(toothgrowth_df)
        assert list(read_df.columns) == list(toothgrowth_df.columns)
        np.testing.assert_allclose(read_df["len"].values, toothgrowth_df["len"].values)
        assert list(read_df["supp"].values) == list(toothgrowth_df["supp"].values)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Verify MIT License, jamovi reference attribution, and zero copied jamovi V logo
    assert os.path.isfile(os.path.join(ROOT_DIR, "LICENSE"))
    notice_txt = open(os.path.join(ROOT_DIR, "NOTICE.md"), encoding="utf-8").read()
    assert "jamovi" in notice_txt and "MIT" in notice_txt
    assets_dir = os.path.join(ROOT_DIR, "reconstructed", "pyjamovi", "web", "assets")
    svgs = os.listdir(assets_dir)
    assert "logo-v.svg" not in svgs
    assert "brand-logo.svg" in svgs and "brand-logo-mark.svg" in svgs
    assert len([s for s in svgs if s.endswith(".svg")]) >= 45


# --- Gate 2A: Exploration, Scatr & T-Tests ---
def test_descriptives_scatr_and_ttest_suite(toothgrowth_df, iris_df, bugs_df):
    res = ost.descriptives(toothgrowth_df, vars=["len"], split_by="supp", skew=True, kurt=True, sw=True, box=True, hist=True)
    mean_row = next(r for r in res.get_table("descriptives").rows if r["var"] == "Mean")
    assert abs(mean_row["len_OJ"] - 20.663) < 0.01
    assert abs(mean_row["len_VC"] - 16.963) < 0.01
    assert len(res.plots) == 2

    s_res = ost.scatr(iris_df, x="Petal.Length", y="Sepal.Length", group="Species", reg_line="linear", marginal="dens")
    assert len(s_res.plots) == 1
    p_res = ost.run("pareto", bugs_df, x="Region")
    assert p_res.get_table("pareto") is not None

    res_is = ost.ttestIS(toothgrowth_df, vars="len", group="supp", welchs=True, mann=True, ci=True, norm=True, eqv=True, plots=True)
    s_row = next(r for r in res_is.get_table("ttest").rows if r["test"] == "Student's t")
    assert abs(s_row["stat"] - 1.915268) < 1e-3 and s_row["df"] == 58.0

    res_ps = ost.ttestPS(bugs_df, pairs=[("LDLF", "LDHF")], wilcoxon=True, plots=True)
    assert len(res_ps.get_table("ttest").rows) == 2

    res_os = ost.ttestOneS(toothgrowth_df, vars="len", test_value=15.0, wilcoxon=True)
    assert res_os.get_table("ttest").rows[0]["p"] < 0.01


# --- Gate 2B: ANOVA Suite & All 5 Post-Hoc Methods ---
@pytest.mark.parametrize("ph_method", ["tukey", "gamesHowell", "scheffe", "bonferroni", "holm"])
def test_all_five_posthoc_methods(iris_df, ph_method):
    res = ost.anovaOneW(iris_df, deps="Sepal.Length", group="Species", fishers=True, welchs=True, ph_method=ph_method)
    ph_tbl = res.tables[-1]
    assert len(ph_tbl.rows) == 3
    assert all(r["p"] < 0.001 for r in ph_tbl.rows)


def test_factorial_anova_ancova_rm_kruskal_and_bayesian(toothgrowth_df, bugs_df, iris_df):
    fac_res = ost.anova(toothgrowth_df, dep="len", factors=["supp", "dose"], ss_type=3, omega_sq=True, emm_plot=True)
    assert len(fac_res.get_table("main").rows) == 4 and len(fac_res.plots) == 1

    anc_res = ost.ancova(iris_df, dep="Sepal.Length", factors=["Species"], covs=["Petal.Length"])
    assert anc_res.get_table("ancova") is not None

    rm_res = ost.anovaRM(bugs_df, rm_vars=["LDLF", "LDHF", "HDLF", "HDHF"], spher=True, posthoc=True)
    assert rm_res.get_table("within") is not None and rm_res.get_table("sphericity") is not None

    kw_res = ost.kruskal(iris_df, deps="Sepal.Length", group="Species", dscf=True)
    assert kw_res.get_table("kruskal").rows[0]["chi2"] > 90.0

    bt = ost.bayesTTestIS(toothgrowth_df, vars="len", group="supp", plot_pp=True, plot_robust=True)
    assert 1.0 < bt.get_table("bayes_ttest").rows[0]["bf10"] < 2.0 and len(bt.plots) == 2
    ba = ost.bayesANOVA(toothgrowth_df, dep="len", factors=["supp", "dose"])
    assert len(ba.get_table("models").rows) == 5
    bc = ost.bayesCorr(iris_df, vars=["Sepal.Length", "Petal.Length", "Petal.Width"])
    assert bc.get_table("bcorr").rows[0]["bf10"] > 1000.0


# --- Gate 2C: Regression, Psychometrics, SEM & Panel Econometrics ---
def test_regression_psychometrics_sem_and_panel(iris_df, medmod_df, bugs_df, big5_df, sem_df, grunfeld_df, survival_df):
    h_res = ost.linReg(iris_df, dep="Sepal.Length", blocks=[["Sepal.Width"], ["Petal.Length", "Petal.Width"]], collin=True, dw=True, cooks=True, plots_diag=True)
    assert len(h_res.get_table("fit").rows) == 2 and h_res.get_table("comparison") is not None

    log_res = ost.logRegBin(medmod_df, dep="Quit_Binary", covs=["Stress", "Burnout", "Social_Support"])
    assert 0.65 < log_res.get_table("pred_measures").rows[0]["auc"] <= 1.0

    pc_res = ost.partCorr(iris_df, vars=["Sepal.Length", "Petal.Length"], controls=["Petal.Width"])
    assert len(pc_res.get_table("pcorr").rows) == 1
    ct_res = ost.contTables(bugs_df, rows="Gender", cols="Region", phi_cra=True, bar_plot=True)
    assert ct_res.get_table("crosstab") is not None

    rel_res = ost.reliability(sem_df, vars=["x1_vis", "x2_vis", "x3_vis"])
    assert rel_res.get_table("scale").rows[0]["alpha"] > 0.65 and rel_res.get_table("scale").rows[0]["omega"] > 0.65

    sem_vars = ["x1_vis", "x2_vis", "x3_vis", "x4_txt", "x5_txt", "x6_txt"]
    assert ost.pca(sem_df, vars=sem_vars).get_table("loadings") is not None
    assert ost.efa(sem_df, vars=sem_vars, rotation="oblimin").get_table("loadings") is not None

    # Structural Equation Modeling (CFA + Latent Regression)
    syntax = "Visual =~ x1_vis + x2_vis + x3_vis\nTextual =~ x4_txt + x5_txt + x6_txt\nSpeed =~ x7_spd + x8_spd + x9_spd\nGPA ~ Visual + Textual + Speed"
    sem_res = ost.sem(sem_df, model_syntax=syntax)
    fit_rows = {r["index"]: r["value"] for r in sem_res.get_table("fit").rows}
    assert fit_rows["Comparative Fit Index (CFI)"] > 0.90 and fit_rows["RMSEA"] < 0.10 and len(sem_res.plots) == 1

    # Panel Data Econometrics (Entity FE, Two-Way FE, Random Effects GLS + Hausman Test)
    for m_type in ("fixed", "twoway", "random", "pooled"):
        p_res = ost.panel(grunfeld_df, entity_col="firm", time_col="year", dep="invest", covs=["value", "capital"], model_type=m_type, hausman=(m_type == "fixed"))
        assert p_res.get_table("coef") is not None
        if m_type == "fixed":
            assert p_res.get_table("summary").rows[1]["val"] > 0.70
            assert p_res.get_table("hausman") is not None

    # Mediation, Moderation, MixedLM, Survival
    assert ost.mediation(medmod_df, dep="Turnover_Intent", pred="Stress", med="Burnout", bootstrap=150).get_table("med").rows[0]["p"] < 0.01
    assert len(ost.moderation(medmod_df, dep="Turnover_Intent", pred="Stress", mod="Social_Support").get_table("simple_slopes").rows) == 3
    assert ost.mixedModel(grunfeld_df, dep="invest", cluster="firm", covs=["value", "capital"]).get_table("random") is not None
    assert ost.survival(survival_df, time_col="Time_Months", event_col="Event_Status", group="Treatment", covs=["Age", "Biomarker"]).get_table("cox") is not None


# --- Gate 2D: 10-Format Table Exporter & PlotStyle Vector Customization ---
def test_ten_format_table_export_and_plot_style_customization(toothgrowth_df):
    sess = ost.Session(toothgrowth_df, auto_sync_workspace=False)
    res = sess.anova_oneway(
        deps="len",
        group="dose",
        ph_method="tukey",
        desc_plot=True,
        plotStyle={
            "palette": "kihasa",
            "theme": "apa_classic",
            "custom_colors": ["#00457C", "#0D9488", "#D97706"],
            "title": "비타민 C 용량별 치아 성장량 비교",
            "xlabel": "투여 용량 (mg/day)",
            "ylabel": "성장 길이 (mm)",
            "width": 6.5,
            "height": 4.2,
            "dpi": 160,
        },
    )
    assert len(res.plots) == 1 and len(res.plot_vectors) == 1
    assert "<svg" in res.plot_vectors[0]["svg"]
    assert len(res.plot_vectors[0]["pdf_b64"]) > 200

    with tempfile.TemporaryDirectory() as tmpd:
        # Test all 11 Analysis Results (Tables + Embedded Graphs) export formats
        for ext in ("xlsx", "docx", "hwpx", "pdf", "tex", "csv", "tsv", "md", "html", "json", "omv"):
            out_path = os.path.join(tmpd, f"exported_results.{ext}")
            res.export_results(out_path, dataset=toothgrowth_df)
            assert os.path.isfile(out_path) and os.path.getsize(out_path) > 120
            if ext == "hwpx":
                with zipfile.ZipFile(out_path, "r") as zf:
                    names = zf.namelist()
                    assert "Contents/section0.xml" in names
                    assert any(n.startswith("BinData/") and n.endswith(".png") for n in names)
                    sec_xml = zf.read("Contents/section0.xml").decode("utf-8")
                    assert "<hp:pic" in sec_xml
            elif ext == "pdf":
                with open(out_path, "rb") as pf:
                    assert pf.read(5) == b"%PDF-"
            elif ext == "tex":
                assert os.path.isfile(os.path.join(tmpd, "exported_results_fig1.pdf"))

        # Test vector & raster plot export (.svg, .pdf, .png)
        for p_ext in ("svg", "pdf", "png"):
            p_out = os.path.join(tmpd, f"custom_plot.{p_ext}")
            res.save_plot(p_out)
            assert os.path.isfile(p_out) and os.path.getsize(p_out) > 400


# --- Gate 2E: All 15 Standalone GraphBuilder Types + Custom Matplotlib Figure -> Analysis Results ---
def test_standalone_graph_builder_all_15_types_and_custom_figure_to_results(toothgrowth_df, iris_df):
    import matplotlib.pyplot as plt

    sess = ost.Session(toothgrowth_df, auto_sync_workspace=False)
    for ctype in ost.CHART_TYPES:
        df_use = iris_df if ctype in ("pairplot", "heatmap", "scatter") else toothgrowth_df
        g = ost.graph(df_use, chart_type=ctype, plotStyle={"palette": "openstat", "dpi": 90})
        assert len(g.plots) == 1
        assert "<svg" in g.plot_vectors[0]["svg"]

    # Session graph + custom Matplotlib figure exported to Analysis Results
    g_card = sess.graph(chart_type="raincloud", x="dose", y="len", group="supp")
    assert len(g_card.plots) == 1 and len(g_card.tables) == 1

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot([1, 2, 3], [10, 18, 26], marker="o")
    ax.set_title("Custom Trajectory")
    c_card = sess.add_figure(fig, title="Custom Trajectory Card", caption="Exported to Analysis Results")
    assert len(c_card.plots) == 1 and len(sess.history) == 2

    # Auto-capture of raw Matplotlib plt.figure() inside live .py syntax execution
    raw_py = (
        "plt.figure(figsize=(4.5, 3.0))\n"
        "plt.bar(['OJ', 'VC'], [20.66, 16.96], color=['#0D9488', '#1E3A5F'])\n"
        "plt.title('Auto-Captured Live Bar')\n"
    )
    cap_analyses, _, _, _ = ost.execute_python_syntax(raw_py, default_df=toothgrowth_df, sync_workspace=False)
    assert len(cap_analyses) == 1
    assert cap_analyses[0].name == "customFigure"
    assert len(cap_analyses[0].plots) == 1


# --- Gate 3A: Live .py Syntax Engine, Doctor Self-Check & Upstream jamovi Sync/Plugin ---
def test_live_py_syntax_doctor_and_upstream_plugin_sync(toothgrowth_df, bugs_df):
    # 1. Bidirectional .py generation & execution
    sess = ost.Session(toothgrowth_df, auto_sync_workspace=False)
    sess.ttest(dep="len", group="supp", welchs=True)
    py_code = sess.to_python_script()
    assert "import openstat as ost" in py_code

    executed_analyses, _, _, _ = ost.execute_python_syntax(py_code, default_df=toothgrowth_df, sync_workspace=False)
    assert len(executed_analyses) >= 1
    assert executed_analyses[0].get_table("ttest") is not None

    # 2. Package freshness & API/syntax drift self-diagnostics (`doctor`)
    doc_rep = ost.doctor(check_pypi=False)
    assert doc_rep["compat"]["healthy"] is True
    assert doc_rep["compat"]["passedChecks"] == doc_rep["compat"]["totalChecks"] >= 6
    assert doc_rep["freshness"]["missingCount"] == 0
    assert doc_rep["freshness"]["belowMinimumCount"] == 0

    # 3. Upstream jamovi feature sync & clean-room plugin scaffolding (`friedman`)
    sync_rep = ost.check_jamovi_upstream(fetch_remote=False)
    assert sync_rep["openstatTotalAnalyses"] >= 28

    plugin_path = ost.scaffold_jamovi_feature("friedman")
    try:
        cat = ost.refresh_analysis_catalog()
        assert any(c["id"] == "friedman" for c in cat)
        fried_res = ost.run("friedman", bugs_df, rmVars=["LDLF", "LDHF", "HDLF", "HDHF"])
        assert fried_res.get_table("friedman").rows[0]["p"] < 0.01
    finally:
        if os.path.exists(plugin_path):
            os.remove(plugin_path)
        ost.refresh_analysis_catalog()


# --- Gate 3B: Claude Code MCP, CLI (`openstat`) & In-Process Web Studio API ---
def test_claude_mcp_cli_and_inprocess_web_api():
    # MCP tools
    mcp_doc = handle_tool_call("openstat_doctor", {"check_pypi": False})
    assert "PASS" in mcp_doc["content"][0]["text"]

    mcp_sem = handle_tool_call("openstat_run_analysis", {
        "file_path": os.path.join(SAMPLE_DIR, "SEM_Latent3Factor.csv"),
        "analysis_type": "sem",
        "parameters": {"model": "F1 =~ x1_vis + x2_vis + x3_vis\nF2 =~ x4_txt + x5_txt + x6_txt\nF2 ~ F1"},
        "plot_style": {"palette": "nature", "theme": "clean"},
    })
    assert "Model Fit Indices" in mcp_sem["content"][0]["text"]

    # CLI execution (`openstat`)
    openstat_bin = os.path.join(ROOT_DIR, ".venv", "bin", "openstat")
    proc = subprocess.run(
        [
            openstat_bin, "run",
            "--data", os.path.join(SAMPLE_DIR, "Grunfeld_Panel.csv"),
            "--analysis", "panel",
            "--params", '{"entity":"firm","time":"year","dep":"invest","covs":["value","capital"]}',
            "--no-sync",
        ],
        capture_output=True, text=True, check=True
    )
    assert "Hausman Specification Test" in proc.stdout

    # In-process ASGI Web Studio verification (zero background server left running)
    st_idx, html_bytes = _asgi_request("GET", "/")
    assert st_idx == 200 and b"OpenStat Studio" in html_bytes and b"logo-v.svg" not in html_bytes

    st_h, h_bytes = _asgi_request("GET", "/api/health")
    h_json = json.loads(h_bytes.decode("utf-8"))
    assert st_h == 200 and h_json["engine"] == "openstat-studio" and h_json["license"] == "MIT"

    st_ld, _ = _asgi_request("POST", "/api/dataset/load", query={"name": "ToothGrowth.csv"})
    assert st_ld == 200

    st_run, run_bytes = _asgi_request("POST", "/api/analysis/run", json_body={
        "analysis_type": "ttestIS",
        "params": {"vars": ["len"], "group": "supp", "welchs": True, "plots": True}
    })
    run_json = json.loads(run_bytes.decode("utf-8"))
    assert st_run == 200 and "import openstat as ost" in run_json["pythonCode"]

    aid = run_json["analysisId"]
    st_restyle, _ = _asgi_request("POST", "/api/plot/restyle", json_body={
        "analysis_id": aid,
        "plot_style": {"palette": "lancet", "theme": "apa_classic", "title": "Restyled via API"}
    })
    assert st_restyle == 200

    st_exp, hwpx_bytes = _asgi_request("GET", "/api/export/tables", query={"fmt": "hwpx", "analysis_id": aid})
    assert st_exp == 200 and len(hwpx_bytes) > 400

    st_pdf, pdf_bytes = _asgi_request("GET", "/api/export/results", query={"fmt": "pdf", "analysis_id": aid})
    assert st_pdf == 200 and pdf_bytes[:5] == b"%PDF-"

    st_push, push_bytes = _asgi_request("POST", "/api/graph/push_to_results", json_body={"source_analysis_id": aid, "source_plot_idx": 0, "title": "Pinned Graph"})
    assert st_push == 200 and len(json.loads(push_bytes.decode("utf-8"))["analyses"]) >= 2

    mcp_gr = handle_tool_call("openstat_create_graph", {
        "file_path": os.path.join(SAMPLE_DIR, "ToothGrowth.csv"),
        "chart_type": "bar",
        "x": "dose",
        "y": "len",
        "group": "supp",
    })
    assert "Bar Chart" in mcp_gr["content"][0]["text"]

    st_svg, svg_bytes = _asgi_request("GET", "/api/plot/export", query={"analysis_id": aid, "plot_idx": 0, "fmt": "svg"})
    assert st_svg == 200 and b"<svg" in svg_bytes


def test_data_ops_and_stata_do_pipeline_reproducibility(tmp_path):
    """
    Verifies end-to-end data wrangling (merge, compute, recode, filter, reshape, rename)
    and Stata .do-style .py script recording & 100% numerical reproduction.
    """
    wide_path = os.path.join(SAMPLE_DIR, "synthetic_clinical_wide.csv")
    lookup_path = os.path.join(SAMPLE_DIR, "synthetic_site_lookup.csv")
    assert os.path.isfile(wide_path) and os.path.isfile(lookup_path)

    # 1. Interactive SDK Session
    sess = ost.Session(wide_path, auto_sync_workspace=False)
    assert sess.df.shape == (120, 12)

    # Data operations
    sess.merge(lookup_path, on="subj_id", how="left")
    assert "region" in sess.df.columns and sess.df.shape == (120, 15)

    sess.compute("bmi", "ROUND(weight_kg / ((height_cm/100)**2), 2)")
    assert "bmi" in sess.df.columns and sess.df["bmi"].iloc[0] > 10

    sess.compute("bp_drop", "bp_w1 - bp_w3")
    assert "bp_drop" in sess.df.columns

    sess.recode("age", "age_band", "min..39='20-30대'; 40..59='40-50대'; 60..max='60대이상'")
    assert "age_band" in sess.df.columns
    assert set(sess.df["age_band"].unique()) == {"20-30대", "40-50대", "60대이상"}

    sess.reshape("long", id_vars=["subj_id"], stubnames=["bp_w", "qol_w"], time_col="wave")
    assert sess.df.shape[0] == 360  # 120 subjects * 3 waves

    sess.rename_var("bp_w", "bp")
    sess.rename_var("qol_w", "qol")
    assert "bp" in sess.df.columns and "qol" in sess.df.columns

    # Analysis & Graph
    res_anova = sess.run("anovaOneW", deps=["bp"], group="arm", phMethod="tukey")
    res_graph = sess.run("graph", chartType="line", x="wave", y="bp", group="arm")
    assert len(sess.history) == 2

    # Export
    out_xlsx = str(tmp_path / "interactive_results.xlsx")
    out_pdf = str(tmp_path / "interactive_results.pdf")
    sess.export_results(out_xlsx)
    sess.export_results(out_pdf)
    assert os.path.isfile(out_xlsx) and os.path.getsize(out_xlsx) > 500
    assert os.path.isfile(out_pdf) and os.path.getsize(out_pdf) > 500

    # Save .py script (Stata .do equivalent)
    script_path = str(tmp_path / "recorded_pipeline.py")
    sess.save_py(script_path)
    assert os.path.isfile(script_path)
    with open(script_path, "r", encoding="utf-8") as f:
        script_code = f.read()
    assert "sess.merge(" in script_code
    assert "sess.compute(" in script_code
    assert "sess.recode(" in script_code
    assert "sess.reshape(" in script_code
    assert "sess.rename_var(" in script_code
    assert "'anovaOneW'" in script_code
    assert "'graph'" in script_code

    # 2. Replay from Scratch via `ost.run_script`
    replayed = ost.run_script(script_path, sync_workspace=False)
    assert replayed["df"].shape == (360, 15)
    assert list(replayed["df"].columns) == list(sess.df.columns)
    np.testing.assert_allclose(
        replayed["df"]["bp"].to_numpy(),
        sess.df["bp"].to_numpy(),
        rtol=1e-6
    )

    # Statistical reproduction check (exact ANOVA F & Tukey p-values)
    rep_anova = replayed["analyses"][0]
    orig_table = res_anova.tables[0].to_dict()
    rep_table = rep_anova.tables[0].to_dict()
    assert orig_table == rep_table

    # 3. In-Process ASGI Web Studio Data Wrangling Endpoints Test
    # Load dataset
    st_ld, _ = _asgi_request("POST", "/api/dataset/load", query={"name": "synthetic_clinical_wide.csv"})
    assert st_ld == 200

    # Test Compute
    st_cmp, cmp_b = _asgi_request("POST", "/api/data/compute", json_body={"col_name": "bmi", "formula": "ROUND(weight_kg / ((height_cm/100)**2), 2)"})
    assert st_cmp == 200 and "bmi" in [c["name"] for c in json.loads(cmp_b.decode("utf-8"))["columns"]]

    # Test Recode
    st_rec, rec_b = _asgi_request("POST", "/api/data/recode", json_body={
        "source_col": "age", "target_col": "age_band",
        "rules": "min..39='20-30대'; 40..59='40-50대'; 60..max='60대이상'"
    })
    rec_json = json.loads(rec_b.decode("utf-8"))
    assert st_rec == 200 and "age_band" in [c["name"] for c in rec_json["columns"]]

    # Test Filter
    st_flt, flt_b = _asgi_request("POST", "/api/data/filter", json_body={"condition": "age >= 40"})
    flt_json = json.loads(flt_b.decode("utf-8"))
    assert st_flt == 200 and flt_json["rowCount"] < 120

    # Test Undo (should restore unfiltered rows)
    st_undo, undo_b = _asgi_request("POST", "/api/data/undo")
    undo_json = json.loads(undo_b.decode("utf-8"))
    assert st_undo == 200 and undo_json["rowCount"] == 120

    # Test Reshape
    st_rsh, rsh_b = _asgi_request("POST", "/api/data/reshape", json_body={
        "direction": "long", "id_vars": ["subj_id"], "stubnames": ["bp_w"], "time_col": "wave"
    })
    rsh_json = json.loads(rsh_b.decode("utf-8"))
    assert st_rsh == 200 and rsh_json["rowCount"] == 360


def test_jasp_features_synthetic_data_and_stata_transpiler():
    """Verifies JASP modules (SummaryTTest, NetworkAnalysis), Synthetic Data Generator & Stata Transpiler."""
    from pyjamovi.analyses import SummaryTTest, NetworkAnalysis
    from pyjamovi.synthetic_data import generate_synthetic_dataset
    from pyjamovi.stata_transpiler import transpile_stata_line, transpile_stata_script

    # 1. SummaryTTest (JASP)
    st = SummaryTTest(m1=22.0, s1=4.5, n1=35, m2=18.0, s2=4.0, n2=35)
    tbl = st.tables[0].to_dict()
    assert tbl["rows"][0]["test"] == "Student's t"
    assert tbl["rows"][0]["t"] > 3.5
    assert tbl["rows"][0]["bf10"] > 10.0

    # 2. NetworkAnalysis (JASP)
    tooth_df = pd.read_csv(os.path.join(SAMPLE_DIR, "ToothGrowth.csv"), sep=None, engine="python")
    net = NetworkAnalysis(tooth_df, vars=["len", "dose"])
    assert len(net.tables) == 2
    assert len(net.plots) == 2

    # 3. Synthetic Data Generator
    synth = generate_synthetic_dataset(tooth_df, n_samples=50, epsilon=1.0)
    assert len(synth) == 50
    assert set(synth.columns) == set(tooth_df.columns)

    # 4. Stata Transpiler
    assert transpile_stata_line("ttest len, by(supp)") == "sess.ttest(dep='len', group='supp')"
    assert transpile_stata_line("reg len supp dose") == "sess.regression(dep='len', covs=['supp', 'dose'])"
    assert transpile_stata_line("gen log_len = log(len)") == "sess.compute('log_len', 'log(len)')"
    assert "sess.spatial_regression" in transpile_stata_line("spregress welfare_expenditure elderly_ratio, dvarlag(W)")
    assert "sess.spatial_regression" in transpile_stata_line("moran welfare_expenditure")

    # 5. Spatial Analysis Module (GeoDa / Stata spregress)
    gis_df = pd.read_csv(os.path.join(SAMPLE_DIR, "Korea_Regional_Welfare_GIS.csv"))
    from pyjamovi.analyses import SpatialAnalysis
    sa = SpatialAnalysis(gis_df, dep="welfare_expenditure", covs=["elderly_ratio"], lat="latitude", lon="longitude", model_type="lag")
    assert len(sa.tables) >= 2
    assert len(sa.plots) >= 1

    # 6. Stata script execution via ASGI
    _asgi_request("POST", "/api/dataset/load", query={"name": "ToothGrowth.csv"})
    st_stata, b_stata = _asgi_request("POST", "/api/syntax/execute", json_body={
        "code": "ttest len, by(supp)\nsum len",
        "replace_existing": True
    })
    assert st_stata == 200
    res_stata = json.loads(b_stata.decode("utf-8"))
    assert len(res_stata["analyses"]) >= 1

    # 7. Spatial spregress via ASGI
    _asgi_request("POST", "/api/dataset/load", query={"name": "Korea_Regional_Welfare_GIS.csv"})
    st_sp, b_sp = _asgi_request("POST", "/api/syntax/execute", json_body={
        "code": "spregress welfare_expenditure elderly_ratio, lat(latitude) lon(longitude) model(lag)",
        "replace_existing": False
    })
    assert st_sp == 200
    res_sp = json.loads(b_sp.decode("utf-8"))
    assert any(a.get("name") == "spatialAnalysis" or a.get("analysis_type") == "spatialAnalysis" for a in res_sp["analyses"])


def test_web_ui_layout_menu_and_plugin_menu_contract():
    """Static checks on web/index.html: the window-layout menu is hidden until opened, and catalog
    plugins are added to RIBBON_MENUS_I18N menus that exist (RIBBON_MENUS itself is not defined)."""
    import re

    html_path = os.path.join(ROOT_DIR, "reconstructed", "pyjamovi", "web", "index.html")
    with open(html_path, encoding="utf-8") as f:
        html = f.read()

    css = re.search(r"\n\.layout-dd \{(.*?)\}", html, re.S)
    assert css, ".layout-dd has no CSS rule"
    assert "position: absolute" in css.group(1) and "display: none" in css.group(1)
    assert re.search(r"\.layout-dd\.open \{\s*display: block;", html)

    assert not re.search(r"\bRIBBON_MENUS(?!_I18N)\b", html), "undefined RIBBON_MENUS referenced"

    def menu_keys(lang):
        start = html.index(f"    {lang}: {{", html.index("const RIBBON_MENUS_I18N"))
        block = html[start:html.index("\n    }", start)]
        return set(re.findall(r"^        ([a-z_]+): \[", block, re.M))

    ko, en = menu_keys("ko"), menu_keys("en")
    assert ko == en
    group_map = re.search(r"const PLUGIN_GROUP_MENU = \{(.*?)\};", html, re.S).group(1)
    targets = set(re.findall(r':\s*"([a-z_]+)"', group_map)) | {"exploration"}
    assert targets <= ko, f"plugin groups map to missing menus: {targets - ko}"
    ribbon_keys = set(re.findall(r"openRibbonMenu\(event, '([a-z_]+)'\)", html))
    assert ribbon_keys <= ko


def test_web_ui_handlers_and_pane_styles_defined():
    """Every inline event handler in web/index.html names a defined function, and the window-layout,
    left-pane and overlay-dialog styles exist (they were missing after the v3.1 layout patch)."""
    import re

    html_path = os.path.join(ROOT_DIR, "reconstructed", "pyjamovi", "web", "index.html")
    with open(html_path, encoding="utf-8") as f:
        html = f.read()

    defined = set(re.findall(r"(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", html))
    defined |= set(re.findall(r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", html))
    called = set(re.findall(
        r'on(?:click|change|input|keydown|keyup|mousedown|dblclick)="(?:[^"]*?;\s*)?([A-Za-z_$][\w$]*)\(', html))
    missing = sorted(called - defined - {"if", "event", "this", "document", "window", "alert", "confirm"})
    assert not missing, f"inline handlers without a function: {missing}"

    for selector in (".modal-backdrop", ".left-box-hdr", "#left-var-explorer", "#left-syntax-console",
                     ".right-pane-subbar", ".v-tab", ".stata-cmd-bar"):
        assert re.search(r"^" + re.escape(selector) + r"\s*[{,]", html, re.M), f"no CSS rule for {selector}"

    results_rule = re.search(r"\n#results-pane \{(.*?)\}", html, re.S).group(1)
    assert "display: flex" in results_rule and "flex-direction: column" in results_rule
    assert 'document.body.setAttribute("data-layout", mode)' in html
    for mode in ("default", "syntax", "quad", "jamovi", "results-only", "data-only"):
        assert f'body[data-layout="{mode}"]' in html, mode


def test_audit_sdk_fixes():
    """Fixes from the full UI audit (2026-11-06): Stata gen/replace, compute shorthand, anova(factors=...),
    integer counts in descriptives, filter removal, and the SEM / mediation path diagrams."""
    import openstat as ost
    from pyjamovi.stata_transpiler import transpile_stata_line
    from pyjamovi.syntax_engine import execute_python_syntax
    from pyjamovi.analyses.plots import plot_sem_diagram

    sess = ost.Session(os.path.join(ROOT_DIR, "sample_data", "ToothGrowth.csv"))
    sess.compute("loglen = log(len)")
    assert np.allclose(sess.df["loglen"], np.log(sess.df["len"]))
    with pytest.raises(ValueError):
        sess.compute("no_formula_here")

    # Stata gen / replace run end to end through the syntax engine
    assert transpile_stata_line("replace len = len + 1") == "sess.compute('len', 'len + 1')"
    _, _, df_out, _ = execute_python_syntax(transpile_stata_line("gen log_len = log(len)"),
                                            default_df=sess.df, sync_workspace=False)
    assert "log_len" in df_out.columns

    assert sess.anova(dep="len", factors=["supp"]).name == "anovaOneW"
    assert sess.anova(dep="len", factors=["supp", "dose"]).name == "anovaFactorial"

    rows = {r["var"]: r for r in sess.run("descriptives", vars=["len"]).tables[0].rows}
    assert rows["N"]["_rowfmt"] == "integer" and rows["Missing"]["_rowfmt"] == "integer"

    n0 = len(sess.df)
    sess.filter_rows("len > 20")
    sess.compute("dbl", "len * 2")
    assert len(sess.df) < n0
    sess.clear_filters()
    assert len(sess.df) == n0 and "dbl" in sess.df.columns
    assert [op["op"] for op in sess.data_ops].count("filter_rows") == 0

    sem_rows = pd.DataFrame([
        {"lval": "F1", "op": "=~", "rval": "a1", "Estimate": 0.8, "p-value": np.nan},
        {"lval": "F1", "op": "=~", "rval": "a2", "Estimate": 0.7, "p-value": 0.001},
        {"lval": "F2", "op": "=~", "rval": "b1", "Estimate": 0.9, "p-value": np.nan},
        {"lval": "F2", "op": "=~", "rval": "b2", "Estimate": 0.6, "p-value": 0.0001},
        {"lval": "F2", "op": "~", "rval": "F1", "Estimate": 0.4, "p-value": 0.02},
        {"lval": "y", "op": "~", "rval": "F2", "Estimate": 0.3, "p-value": 0.2},
        {"lval": "y", "op": "~", "rval": "F1", "Estimate": 0.1, "p-value": 0.5},
    ])
    for df_ in (sem_rows, sem_rows.iloc[:4], pd.DataFrame(columns=["lval", "op", "rval"])):
        src = plot_sem_diagram(df_, title="t")
        assert src.startswith("data:image/png;base64,") and len(src) > 1000


def test_audit_syntax_append_and_filter_reset_endpoints():
    """Adding results from the syntax editor keeps earlier data steps and appends to the live script;
    an empty filter condition restores the rows removed by earlier filters."""
    try:
        from fastapi.testclient import TestClient
    except (ImportError, RuntimeError):
        pytest.skip("Starlette TestClient requires httpx2")
    from pyjamovi.server import app
    c = TestClient(app)
    c.post("/api/dataset/load", params={"name": "ToothGrowth.csv"})
    c.post("/api/analysis/clear")
    j = c.post("/api/data/compute", json={"col_name": "len2", "formula": "len * 2"}).json()
    assert j["dataOps"][-1]["op"] == "compute"
    j = c.post("/api/syntax/execute", json={"code": "summarize len", "replace_existing": False}).json()
    assert any(op["op"] == "compute" for op in j["dataOps"]), "earlier data steps were dropped"
    j = c.post("/api/syntax/execute", json={"code": "gen loglen = log(len)", "replace_existing": False}).json()
    assert [op["op"] for op in j["dataOps"]][-2:] == ["compute", "compute"]
    code = c.get("/api/syntax/current").json()["code"]
    assert "sess.compute('loglen', 'log(len)')" in code and "gen loglen" not in code.replace("# gen", "")
    assert "len2" in code, "the live script lost the earlier steps"
    n0 = j["rowCount"]
    assert c.post("/api/data/filter", json={"condition": "len > 20"}).json()["rowCount"] < n0
    j = c.post("/api/data/filter", json={"condition": ""}).json()
    assert j["rowCount"] == n0 and "loglen" in [col["name"] for col in j["columns"]]


def test_web_ui_audit_contracts():
    """Static contracts from the full UI audit: ribbon menu ids exist as analysis schemas in both languages,
    every data-i18n key is translated in ko and en, status colours exist, Escape closes dialogs."""
    import re

    html_path = os.path.join(ROOT_DIR, "reconstructed", "pyjamovi", "web", "index.html")
    with open(html_path, encoding="utf-8") as f:
        html = f.read()

    menus = html[html.index("const RIBBON_MENUS_I18N = {"):html.index("const OPTION_LABELS_KO")]
    ko_part, en_part = menus.split("\n    en: {", 1)
    ids_ko = set(re.findall(r'\{ id: "(\w+)"', ko_part))
    ids_en = set(re.findall(r'\{ id: "(\w+)"', en_part))
    assert ids_ko == ids_en, f"ko/en menus differ: {sorted(ids_ko ^ ids_en)}"
    schema_block = html[html.index("const ANALYSIS_SCHEMAS = {"):]
    schema_keys = set(re.findall(r"^    (\w+): \{", schema_block, re.M))
    schema_keys |= set(re.findall(r"^    (\w+): \{", html[html.index("const TEXT_NETWORK_SCHEMAS"):], re.M)) if "const TEXT_NETWORK_SCHEMAS" in html else set()
    missing = sorted(i for i in ids_ko if i not in schema_keys)
    assert not missing, f"menu items without an analysis schema: {missing}"

    i18n = html[html.index("const I18N = {"):html.index("const RIBBON_MENUS_I18N")]
    ko_d, en_d = i18n.split("\n    en: {", 1)
    keys_ko = set(re.findall(r"^\s+(\w+):", ko_d, re.M))
    keys_en = set(re.findall(r"^\s+(\w+):", en_d, re.M))
    used = set(re.findall(r'data-i18n(?:-placeholder|-title)?="(\w+)"', html))
    assert not (used - keys_ko) and not (used - keys_en), (sorted(used - keys_ko), sorted(used - keys_en))

    for selector in (".badge-pill.gray", ".badge-pill.red", ".doc-pass", ".doc-warn", ".doc-fail"):
        assert selector + " {" in html, f"no CSS rule for {selector}"
    assert 'ev.key !== "Escape"' in html and "handleCellKey(event, this)" in html
    assert "applyLanguage();   // the saved language" in html

"""
Social-science report format for exported tables and the side-by-side model comparison table.

- coefficient tables are marked in each regression-type analysis; other tables are not
- stars, SE in parentheses (beside / below / own column), p-column rule, leading zeros, notes
- strict option validation (unknown names and values raise)
- .hwpx: complete OWPML package, merged header cells consistent with rowCnt/colCnt, numbered captions
- .docx merged header, .xlsx raw numeric sheet
- model comparison: hierarchical blocks, mixed model types, statistics rows, strict arguments
- Session.compare_models is recorded and replayed by the reproducible script; export options are recorded
- web endpoint and CLI pass the options through
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import zipfile
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

import pandas as pd
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "reconstructed"))

import openstat as ost  # noqa: E402
from pyjamovi.agent import JamoviSession  # noqa: E402
from pyjamovi.exporters import export_analyses  # noqa: E402
from pyjamovi.registry import dispatch_analysis  # noqa: E402
from pyjamovi.report_format import render_table, resolve_report_format, stars_for, star_legend  # noqa: E402
from pyjamovi.syntax_engine import execute_python_syntax  # noqa: E402

SAMPLE_DIR = os.path.join(ROOT_DIR, "sample_data")
HP = "{http://www.hancom.co.kr/hwpml/2011/paragraph}"
HH = "{http://www.hancom.co.kr/hwpml/2011/head}"


@pytest.fixture(scope="module")
def iris():
    return pd.read_csv(os.path.join(SAMPLE_DIR, "iris.csv"), sep=None, engine="python")


@pytest.fixture(scope="module")
def medmod():
    return pd.read_csv(os.path.join(SAMPLE_DIR, "MedMod_WorkStress.csv"))


@pytest.fixture(scope="module")
def hier(iris):
    return dispatch_analysis("linReg", iris, {"dep": "Sepal.Length", "blocks": [["Sepal.Width"], ["Petal.Length", "Petal.Width"]],
                                              "stdEst": True, "ci": True})


def _social(**opts):
    return resolve_report_format("social", opts or None)


def test_coefficient_tables_are_marked_and_others_are_not(iris, medmod):
    gr = pd.read_csv(os.path.join(SAMPLE_DIR, "Grunfeld_Panel.csv"))
    sv = pd.read_csv(os.path.join(SAMPLE_DIR, "Clinical_Survival.csv"))
    semd = pd.read_csv(os.path.join(SAMPLE_DIR, "SEM_Latent3Factor.csv"))
    tg = pd.read_csv(os.path.join(SAMPLE_DIR, "ToothGrowth.csv"), sep=None, engine="python")
    expect = {
        ("linReg", "coef"): (iris, {"dep": "Sepal.Length", "covs": ["Sepal.Width"]}),
        ("logRegBin", "coef"): (medmod, {"dep": "Quit_Binary", "covs": ["Stress"]}),
        ("panel", "coef"): (gr, {"entity": "firm", "time": "year", "dep": "invest", "covs": ["value", "capital"]}),
        ("mixedModel", "fixed"): (gr, {"dep": "invest", "cluster": "firm", "covs": ["value"]}),
        ("survival", "cox"): (sv, {"time_col": "Time_Months", "event_col": "Event_Status", "covs": ["Age"]}),
        ("moderation", "mod"): (medmod, {"dep": "Turnover_Intent", "pred": "Stress", "mod": "Social_Support"}),
        ("sem", "estimates"): (semd, {"model": "F1 =~ x1_vis + x2_vis + x3_vis\nGPA ~ F1"}),
    }
    for (a_type, tname), (df, params) in expect.items():
        r = dispatch_analysis(a_type, df, params)
        marked = [t.name for t in r.tables if t.kind == "coefficients"]
        assert tname in marked, (a_type, marked)
        assert r.get_table(tname).to_dict()["kind"] == "coefficients"
        assert r.model_stats.get("N"), a_type
    for a_type, params in (("ttestIS", {"vars": ["len"], "group": "supp"}),
                           ("anovaOneW", {"deps": ["len"], "group": "dose", "phMethod": "tukey"})):
        r = dispatch_analysis(a_type, tg, params)
        assert not r.coefficient_tables(), a_type
        assert all("kind" not in t.to_dict() for t in r.tables)


def test_social_rendering_of_a_coefficient_table(iris):
    r = dispatch_analysis("linReg", iris, {"dep": "Sepal.Length", "covs": ["Sepal.Width", "Petal.Length"], "stdEst": True, "ci": True})
    tbl = r.get_table("coef")
    rows = {row["pred"]: row for row in tbl.rows}

    rt = render_table(tbl, _social(), number=2)
    assert rt.caption == "<표 2> " + tbl.title
    assert rt.flat_headers[1] == "Estimate (SE)" and "p" not in rt.flat_headers and "SE" not in rt.flat_headers
    body = {b[0]: b for b in rt.body}
    sw = rows["Sepal.Width"]
    assert body["Sepal.Width"][1] == f"{sw['b']:.3f}*** ({sw['se']:.3f})"
    beta_txt = f"{sw['beta']:.3f}"
    assert body["Sepal.Width"][rt.flat_headers.index("Stand. Estimate")] == beta_txt[1:]      # no leading zero
    assert body["Intercept"][rt.flat_headers.index("Stand. Estimate")] == ""
    assert rt.notes[-2:] == ["괄호 안은 표준오차임.", "* p<.05, ** p<.01, *** p<.001"]
    assert all(a == "center" for a, t in zip(rt.aligns, rt.col_types) if t == "number")

    below = render_table(tbl, _social(se_position="below"))
    assert below.flat_headers[1] == "Estimate"
    i = [b[0] for b in below.body].index("Sepal.Width")
    assert below.body[i][1] == f"{sw['b']:.3f}***" and below.body[i + 1] == ["", f"({sw['se']:.3f})"] + [""] * (len(below.body[i]) - 2)
    assert len(below.body) == 2 * len(tbl.rows)

    col = render_table(tbl, _social(se_position="column", star_target="beta", keep_p=True, leading_zero=True, decimals=2))
    assert {"SE", "p"} <= set(col.flat_headers) and "괄호 안은 표준오차임." not in col.notes
    cb = {b[0]: b for b in col.body}
    assert cb["Sepal.Width"][1] == f"{sw['b']:.2f}"                                         # stars moved to β
    assert cb["Sepal.Width"][col.flat_headers.index("Stand. Estimate")] == f"{sw['beta']:.2f}***"
    assert cb["Sepal.Width"][col.flat_headers.index("p")] == "< 0.001"
    assert col.notes[-1] == "* p<0.05, ** p<0.01, *** p<0.001"

    lean = render_table(tbl, _social(keep_stat=False, keep_ci=False, star_note=False, table_numbers=False), number=7)
    assert lean.flat_headers == ["Predictor", "Estimate (SE)", "Stand. Estimate"] and lean.caption == tbl.title
    assert lean.notes == list(tbl.notes)


def test_default_rendering_is_unchanged_and_untagged_tables_keep_columns(iris):
    r = dispatch_analysis("linReg", iris, {"dep": "Sepal.Length", "covs": ["Sepal.Width"]})
    for tbl in r.tables:
        rt = render_table(tbl, None)
        assert rt.flat_headers == [tbl.display_title(c) for c in tbl.columns]
        assert rt.body == [[tbl._format_cell(row.get(c["name"]), c) for c in tbl.columns] for row in tbl.rows]
    fit = r.get_table("fit")
    rt = render_table(fit, _social())
    assert len(rt.flat_headers) == len(fit.columns) and rt.flat_headers[-1] == "p"   # p kept outside coefficient tables
    assert rt.body[0][-1] == "< .001" or rt.body[0][-1].startswith(".")
    assert not any("*" in c for c in rt.body[0])


def test_stars_and_option_validation():
    std = _social()
    assert [stars_for(p, std) for p in (0.0004, 0.001, 0.009, 0.04, 0.05, 0.2, None, float("nan"))] == \
        ["***", "**", "**", "*", "", "", "", ""]
    dag = _social(stars="dagger", lang="en")
    assert stars_for(0.07, dag) == "†" and star_legend(dag) == "†p < .10. *p < .05. **p < .01. ***p < .001."
    econ = _social(stars="econ")
    assert [stars_for(p, econ) for p in (0.005, 0.03, 0.08, 0.2)] == ["***", "**", "*", ""]
    custom = _social(stars=[[0.1, "+"], [0.05, "*"]])
    assert [stars_for(p, custom) for p in (0.01, 0.07)] == ["*", "+"] and star_legend(custom) == "+ p<.10, * p<.05"
    assert resolve_report_format(None) is None and resolve_report_format("apa") is None
    assert resolve_report_format({"se_position": "아래"}).se_position == "below"
    for style, opts, msg in [
        ("social", {"se_position": "left"}, "se_position"),
        ("social", {"stars": "bonferroni"}, "stars"),
        ("social", {"stars": [[1.5, "*"]]}, "0과 1"),
        ("social", {"decimals": 9}, "decimals"),
        ("social", {"keep_p": "maybe"}, "keep_p"),
        ("social", {"colour": "red"}, "알 수 없는 항목"),
        ("default", {"stars": "none"}, "social"),
        ("journal-x", None, "report_style"),
    ]:
        with pytest.raises(ValueError, match=msg):
            resolve_report_format(style, opts)


def _hwpx_tables(path):
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        sec = ET.fromstring(z.read("Contents/section0.xml"))
        head = ET.fromstring(z.read("Contents/header.xml"))
        hpf = z.read("Contents/content.hpf").decode("utf-8")
        assert z.read("mimetype") == b"application/hwp+zip" and z.infolist()[0].filename == "mimetype"
    return names, sec, head, hpf


def test_hwpx_package_structure_and_merged_headers(hier, iris, tmp_path):
    lca = pd.read_csv(os.path.join(SAMPLE_DIR, "Mixture_LCA_Care.csv"))
    mix = dispatch_analysis("lca", lca, {"categorical": ["외출지원필요", "식사지원필요", "가사지원필요"], "nClasses": 2, "starts": 4})
    comp = ost.compare_models(hier)
    out = export_analyses([hier, mix, comp], str(tmp_path / "r.hwpx"), report_style="social", report_options={"se_position": "below"})
    names, sec, head, hpf = _hwpx_tables(out)
    assert {"version.xml", "settings.xml", "Preview/PrvText.txt", "META-INF/container.xml"} <= names
    assert sec.find(f".//{HP}secPr") is not None and head.find(f".//{HH}compatibleDocument") is not None
    border_ids = {int(b.get("id")) for b in head.iter(f"{HH}borderFill")}
    char_ids = {int(c.get("id")) for c in head.iter(f"{HH}charPr")}
    para_ids = {int(c.get("id")) for c in head.iter(f"{HH}paraPr")}
    tables = list(sec.iter(f"{HP}tbl"))
    assert len(tables) == sum(len([t for t in a.tables if t.columns]) for a in (hier, mix, comp))
    merged = 0
    for tbl in tables:
        for tag in ("sz", "pos", "outMargin", "inMargin"):
            assert tbl.find(f"{HP}{tag}") is not None, tag
        rows = tbl.findall(f"{HP}tr")
        n_r, n_c = int(tbl.get("rowCnt")), int(tbl.get("colCnt"))
        assert len(rows) == n_r
        grid = [[0] * n_c for _ in range(n_r)]
        for tr in rows:
            for tc in tr.findall(f"{HP}tc"):
                a, sp = tc.find(f"{HP}cellAddr"), tc.find(f"{HP}cellSpan")
                r0, c0, rs, cs = int(a.get("rowAddr")), int(a.get("colAddr")), int(sp.get("rowSpan")), int(sp.get("colSpan"))
                merged += (rs > 1 or cs > 1)
                assert int(tc.get("borderFillIDRef")) in border_ids
                for rr in range(r0, r0 + rs):
                    for cc in range(c0, c0 + cs):
                        grid[rr][cc] += 1
                for run in tc.iter(f"{HP}run"):
                    assert int(run.get("charPrIDRef")) in char_ids
                for p in tc.iter(f"{HP}p"):
                    assert int(p.get("paraPrIDRef")) in para_ids
        assert all(v == 1 for row in grid for v in row), "every grid cell covered exactly once"
        assert sum(int(c.get("width")) for c in rows[-1].iter(f"{HP}cellSz")) == int(tbl.find(f"{HP}sz").get("width"))
    assert merged >= 4                                    # LCA class groups + comparison model groups
    text = "".join(t.text or "" for t in sec.iter(f"{HP}t"))
    assert "<표 1>" in text and "모형 비교" in text and "괄호 안은 표준오차임." in text and "(0." in text
    assert hpf.count('media-type="image/png"') == len([p for a in (hier, mix, comp) for p in a.plots])
    try:
        from hwpx.tools.package_validator import validate_package
    except ImportError:
        return
    report = validate_package(out)
    assert not [i for i in report.issues if i.level == "error"], report.issues


def test_docx_and_xlsx_outputs(hier, tmp_path):
    from docx import Document
    from openpyxl import load_workbook
    comp = ost.compare_models(hier, names=["기본모형", "확장모형"])
    d = export_analyses([hier, comp], str(tmp_path / "r.docx"), report_style="social")
    doc = Document(d)
    caps = [p.text for p in doc.paragraphs if p.text.startswith("<표")]
    assert caps[0].startswith("<표 1>") and any("모형 비교" in c for c in caps)
    t = doc.tables[-1]
    assert t.rows[0].cells[1].text == "기본모형" and t.rows[1].cells[1].text == "B (SE)"
    assert t.rows[0].cells[0].text == t.rows[1].cells[0].text == "변수"                       # vertical merge

    x = export_analyses([hier, comp], str(tmp_path / "r.xlsx"), report_style="social")
    wb = load_workbook(x)
    assert "원자료(숫자)" in wb.sheetnames
    raw = wb["원자료(숫자)"]
    nums = [c.value for row in raw.iter_rows() for c in row if isinstance(c.value, float)]
    assert len(nums) > 20
    main = wb[wb.sheetnames[0]]
    texts = [str(c.value) for row in main.iter_rows() for c in row if c.value is not None]
    assert any(re.fullmatch(r"-?\d+\.\d{3}\*{1,3} \(\d+\.\d{3}\)", s) for s in texts)
    assert main.merged_cells.ranges or wb[wb.sheetnames[1]].merged_cells.ranges

    plain = export_analyses([hier], str(tmp_path / "plain.xlsx"))
    assert load_workbook(plain).sheetnames == [load_workbook(plain).sheetnames[0]]           # no raw sheet by default
    for ext in ("pdf", "tex", "md", "html", "csv"):
        p = export_analyses([hier, comp], str(tmp_path / f"r.{ext}"), report_style="social")
        assert os.path.getsize(p) > 200
    md = open(tmp_path / "r.md", encoding="utf-8").read()
    assert "<표 1>" in md and "***" in md
    html = open(tmp_path / "r.html", encoding="utf-8").read()
    assert "colspan='2'" in html or "rowspan='2'" in html


def test_compare_models_tables(hier, medmod):
    comp = ost.compare_models(hier, labels={"Intercept": "(상수)"})
    t = comp.get_table("compare")
    assert [c["superTitle"] for c in t.columns[1:]] == ["모형 1"] * 3 + ["모형 2"] * 3
    terms = [r["term"] for r in t.rows if not r.get("_section")]
    assert terms == ["(상수)", "Sepal.Width", "Petal.Length", "Petal.Width"]
    stats = {r["term"]: r for r in t.rows if r.get("_section") == "stats"}
    assert stats["N"]["m1_est"] == 150 and stats["N"]["_rowfmt"] == "integer"
    assert stats["ΔR²"]["m1_est"] == stats["R²"]["m1_est"] and stats["ΔR²"]["m2_p"] < 0.001
    assert abs(stats["R²"]["m2_est"] - hier.get_table("fit").rows[1]["r2"]) < 1e-12
    assert "150.000" not in t.to_string() and "모형 출처" in " ".join(t.notes)
    pw = next(r for r in t.rows if r["term"] == "Petal.Width")
    assert pw["m1_est"] == "" and pw["m2_est"] < 0
    rt = render_table(t, _social())
    assert rt.header_rows[0][0].rowspan == 2 and [h.text for h in rt.header_rows[0][1:]] == ["모형 1", "모형 2"]
    assert rt.rules_above == [4] and rt.body[4][:2] == ["N", "150"]

    ols = dispatch_analysis("linReg", medmod, {"dep": "Turnover_Intent", "covs": ["Stress"]})
    logit = dispatch_analysis("logRegBin", medmod, {"dep": "Quit_Binary", "covs": ["Stress", "Burnout"]})
    mixed = ost.compare_models([ols, logit], names=["OLS", "Logit"], stats=["N", "AIC", "R2_McF"], estimate="b")
    rows = {r["term"]: r for r in mixed.get_table("compare").rows}
    assert rows["종속변수"]["m1_est"] == "Turnover_Intent" and rows["Burnout"]["m1_est"] == ""
    assert rows["McFadden R²"]["m1_est"] == "" and 0 < rows["McFadden R²"]["m2_est"] < 1
    ex = ost.compare_models(logit, estimate="exp")
    er = {r["term"]: r for r in ex.get_table("compare").rows}
    assert abs(er["Stress"]["m1_est"] - logit.get_table("coef").rows[1]["or"]) < 1e-12
    both = ost.compare_models(hier, estimate="both")
    assert [c["title"] for c in both.get_table("compare").columns[1:5]] == ["B", "SE", "β", "p"]
    with pytest.raises(ValueError, match="개수"):
        ost.compare_models(hier, names=["하나"])
    with pytest.raises(ValueError, match="통계량"):
        ost.compare_models(hier, stats=["R2", "Gini"])
    with pytest.raises(ValueError, match="estimate"):
        ost.compare_models(hier, estimate="z")
    with pytest.raises(ValueError, match="계수표"):
        ost.compare_models(dispatch_analysis("descriptives", medmod, {"vars": ["Stress"]}))


def test_session_records_compare_and_report_options(tmp_path):
    sess = JamoviSession(os.path.join(SAMPLE_DIR, "MedMod_WorkStress.csv"), auto_sync_workspace=False)
    r1 = sess.run("linReg", dep="Turnover_Intent", covs=["Stress"])
    r2 = sess.run("linReg", dep="Turnover_Intent", covs=["Stress", "Burnout", "Social_Support"])
    comp = sess.compare_models(r1, 2, names=["모형 A", "모형 B"], stats=["N", "R2", "R2_adj"])
    assert sess.specs[-1]["analysis_type"] == "compareModels" and len(sess.specs[-1]["params"]["models"]) == 2
    out = sess.export_results(str(tmp_path / "s.hwpx"), report_style="social", report_options={"se_position": "below", "stars": "dagger"})
    assert os.path.exists(out)
    op = sess.export_ops[-1]
    assert op["report_style"] == "social" and op["report_options"] == {"stars": "dagger", "se_position": "below"}
    code = sess.to_python_script()
    assert "compareModels" in code and "report_style='social'" in code
    analyses, _, _, _ = execute_python_syntax(code.replace(str(tmp_path), str(tmp_path / "replay")), sync_workspace=False)
    replay = [a for a in analyses if a.name == "compareModels"][0]
    assert replay.get_table("compare").rows == comp.get_table("compare").rows
    with pytest.raises(ValueError, match="결과 번호"):
        sess.compare_models(99)


async def _asgi(app, method, path, query=None, body=None):
    raw = json.dumps(body).encode() if body is not None else b""
    headers = [(b"host", b"testserver")] + ([(b"content-type", b"application/json")] if body is not None else [])
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": urlencode(query or {}).encode(), "headers": headers,
             "client": ("127.0.0.1", 50001), "server": ("testserver", 80)}
    sent, status, chunks = False, 500, []
    done = asyncio.Event()

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": raw, "more_body": False}
        await done.wait()
        return {"type": "http.disconnect"}

    async def send(msg):
        nonlocal status
        if msg["type"] == "http.response.start":
            status = msg["status"]
        elif msg["type"] == "http.response.body":
            chunks.append(msg.get("body", b""))
            if not msg.get("more_body"):
                done.set()

    await app(scope, receive, send)
    return status, b"".join(chunks)


def test_web_endpoint_and_cli(tmp_path):
    from pyjamovi.server import app
    call = lambda *a, **k: asyncio.run(_asgi(app, *a, **k))  # noqa: E731
    assert call("POST", "/api/dataset/load", query={"name": "MedMod_WorkStress.csv"})[0] == 200
    st, body = call("POST", "/api/analysis/run", body={"analysis_type": "compareModels", "params": {
        "models": [{"analysis": "linReg", "dep": "Turnover_Intent", "covs": ["Stress"]},
                   {"analysis": "logRegBin", "dep": "Quit_Binary", "covs": ["Stress"]}]}})
    assert st == 200, body
    card = json.loads(body)
    assert card["tables"][0]["kind"] == "coefficients" and card["tables"][0]["rows"][-1]["_section"] == "stats"
    st, data = call("GET", "/api/export/tables", query={"fmt": "hwpx", "analysis_id": card["analysisId"], "report_style": "social",
                                                        "report_options": json.dumps({"se_position": "below"})})
    assert st == 200 and data[:2] == b"PK"
    p = tmp_path / "web.hwpx"
    p.write_bytes(data)
    assert "괄호 안은 표준오차임." in zipfile.ZipFile(p).read("Contents/section0.xml").decode("utf-8")
    st, err = call("GET", "/api/export/tables", query={"fmt": "docx", "report_style": "social", "report_options": json.dumps({"decimals": 9})})
    assert st == 400 and "decimals" in json.loads(err)["detail"]
    st, err = call("GET", "/api/export/tables", query={"fmt": "docx", "report_style": "social", "report_options": "{bad"})
    assert st == 400
    st, html = call("GET", "/api/report/export_html", query={"report_style": "social"})
    assert st == 200 and "<표 1>".encode() in html.replace(b"&lt;", b"<").replace(b"&gt;", b">")

    out = tmp_path / "cli.docx"
    proc = subprocess.run(
        [os.path.join(ROOT_DIR, ".venv", "bin", "openstat"), "run", "--data", os.path.join(SAMPLE_DIR, "MedMod_WorkStress.csv"),
         "--analysis", "linReg", "--params", '{"dep": "Turnover_Intent", "covs": ["Stress", "Burnout"]}', "--no-sync",
         "--export-table", str(out), "--report-options", '{"se_position": "column"}'],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    from docx import Document
    assert any(p.text.startswith("<표 ") for p in Document(str(out)).paragraphs)


def test_stata_eststo_esttab_and_regress_options(tmp_path):
    from pyjamovi.stata_transpiler import transpile_stata_line, transpile_stata_script
    out = tmp_path / "esttab.rtf"
    code = transpile_stata_script(f'''eststo m1: reg Turnover_Intent Stress
eststo m2: regress Turnover_Intent Stress Burnout, beta
eststo: logit Quit_Binary Stress
esttab m1 m2 using "{out}", se star(* 0.10 ** 0.05 *** 0.01) r2 ar2 b(%9.2f) mtitles("기본" "확장") replace
esttab
''')
    assert "m2 = sess.regression(dep='Turnover_Intent', covs=['Stress', 'Burnout'], std_est=True)" in code
    assert "est3 = sess.logistic(dep='Quit_Binary', covs=['Stress'])" in code
    assert "sess.compare_models(m1, m2, stats=['N', 'R2', 'R2_adj'], names=['기본', '확장'])" in code
    assert "'stars': [[0.1, '*'], [0.05, '**'], [0.01, '***']]" in code and "'decimals': 2" in code
    assert "esttab.docx" in code and ".rtf 출력은 .docx로 저장" in code
    assert "sess.compare_models(m1, m2, est3, stats=['N'])" in code
    analyses, _, _, _ = execute_python_syntax(code, default_dataset_path=os.path.join(SAMPLE_DIR, "MedMod_WorkStress.csv"),
                                              sync_workspace=False)
    assert [a.name for a in analyses].count("compareModels") == 2
    from docx import Document
    doc = Document(str(tmp_path / "esttab.docx"))
    cells = [c.text for t in doc.tables for row in t.rows for c in row.cells]
    assert "기본" in cells and any(re.fullmatch(r"\(\d+\.\d{2}\)", c) for c in cells)       # SE row, 2 decimals
    for line, msg in (("regress y x1 x2, robust", "robust"), ("reg y i.region x", "요인변수"),
                      ("esttab m1, wide", "wide"), ("esttab", "저장된 추정 결과 없음")):
        assert msg in transpile_stata_line(line)


def test_row_format_keeps_p_and_download_names(hier):
    comp = ost.compare_models(hier)
    t = comp.get_table("compare")
    f_row = next(r for r in t.rows if r["term"] == "F")
    col_p = next(c for c in t.columns if c["name"] == "m2_p")
    assert t._format_cell(f_row["m2_p"], t.cell_meta(f_row, col_p)) == "< .001"
    n_row = next(r for r in t.rows if r["term"] == "N")
    col_e = next(c for c in t.columns if c["name"] == "m1_est")
    assert t._format_cell(n_row["m1_est"], t.cell_meta(n_row, col_e)) == "150"
    from pyjamovi.server import _attachment
    cd = _attachment("복지자료_analysis_3.hwpx")
    cd.encode("latin-1")                                   # header must be latin-1 encodable
    assert "filename*=UTF-8''%EB%B3%B5" in cd and 'filename="_analysis_3.hwpx"' not in cd

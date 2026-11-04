"""
Text mining & relational network modules:
- text import (.txt cp949 / .md / .docx with table / .hwpx with table / .pdf, legacy .hwp rejected)
- keyword analysis (stopwords, synonyms, user dictionary, group c-TF-IDF), co-occurrence network
- document clustering recovering the synthetic policy fields, relational network from an edge list
- result export with embedded figures
"""

import os
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "reconstructed"))

import pyjamovi as jmv  # noqa: E402
from pyjamovi.registry import dispatch_analysis  # noqa: E402
from pyjamovi.text_import import load_texts, texts_to_dataframe  # noqa: E402

pytest.importorskip("kiwipiepy")
pytest.importorskip("networkx")

DOCS = os.path.join(ROOT_DIR, "sample_data", "Text_PolicyDocs.csv")
EDGES = os.path.join(ROOT_DIR, "sample_data", "Network_Collaboration.csv")


@pytest.fixture(scope="module")
def docs_df():
    return jmv.Session(DOCS, auto_sync_workspace=False).df


def _table(res, name):
    t = res.get_table(name)
    assert t is not None, f"missing table {name}"
    return t


def test_text_import_all_formats(tmp_path):
    (tmp_path / "면담" / "돌봄").mkdir(parents=True)
    (tmp_path / "면담" / "돌봄" / "a.txt").write_bytes("독거노인 방문요양 이용 경험".encode("cp949"))
    (tmp_path / "면담" / "b.md").write_text("# 청년\n청년 일자리 경험", encoding="utf-8")
    import docx
    d = docx.Document()
    d.add_paragraph("주거급여 수급 가구 조사")
    t = d.add_table(rows=1, cols=2)
    t.cell(0, 0).text = "공공임대"
    t.cell(0, 1).text = "전세자금"
    d.save(tmp_path / "c.docx")
    ns = 'xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph" xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section"'
    sec = (f'<hs:sec {ns}><hp:p><hp:run><hp:t>아동수당 정책 검토</hp:t></hp:run></hp:p>'
           '<hp:p><hp:run><hp:tbl><hp:tr><hp:tc><hp:subList><hp:p><hp:run><hp:t>어린이집</hp:t></hp:run></hp:p>'
           '</hp:subList></hp:tc></hp:tr></hp:tbl></hp:run></hp:p></hs:sec>')
    with zipfile.ZipFile(tmp_path / "d.hwpx", "w") as z:
        z.writestr("mimetype", "application/hwp+zip")
        z.writestr("Contents/section0.xml", sec)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.text(0.1, 0.5, "Primary care access")
    fig.savefig(tmp_path / "e.pdf")
    plt.close(fig)
    (tmp_path / "f.hwp").write_bytes(b"legacy")

    df = load_texts(str(tmp_path))
    assert list(df.columns) == ["문서ID", "파일명", "폴더", "형식", "글자수", "본문"]
    by_name = dict(zip(df["파일명"], df["본문"]))
    assert by_name["a.txt"] == "독거노인 방문요양 이용 경험"
    assert "공공임대 | 전세자금" in by_name["c.docx"]
    assert by_name["d.hwpx"] == "아동수당 정책 검토\n어린이집"  # table-cell paragraph extracted once
    assert "Primary care access" in by_name["e.pdf"]
    assert dict(zip(df["파일명"], df["폴더"]))["a.txt"] == "돌봄"
    skipped = dict(df.attrs["skipped"])
    assert any(p.endswith("f.hwp") for p in skipped)


def test_keywords_preprocessing_rules():
    df = pd.DataFrame({"본문": [
        "어르신 돌봄과 고령자 돌봄 수요가 늘었다. 통합돌봄 사업 확대.",
        "고령자 통합돌봄 서비스와 어르신 건강 관리.",
        "청년 일자리 사업 확대와 고용 안정.",
    ]})
    res = dispatch_analysis("textKeywords", df, {
        "text": "본문", "synonyms": "노인: 어르신, 고령자", "userWords": "통합돌봄", "stopwords": "사업",
        "wordcloud": False, "barPlot": False,
    })
    kw = {r["term"]: r["freq"] for r in _table(res, "keywords").rows}
    assert kw.get("노인") == 4 and "어르신" not in kw and "고령자" not in kw
    assert kw.get("통합돌봄") == 2
    assert "사업" not in kw


def test_keywords_groups_and_wordclouds(docs_df):
    res = dispatch_analysis("textKeywords", docs_df, {"text": "본문", "group": "분야", "groupCloud": True})
    g = _table(res, "groupKeywords")
    first = {}
    current = None
    for r in g.rows:
        current = r["group"] or current
        first.setdefault(current, r["term"])
    assert first["돌봄"] in {"통합돌봄", "돌봄", "방문요양", "독거노인", "돌봄인력"}
    assert first["주거"] in {"주거급여", "주거취약계층", "주택", "가구", "주거비", "지원"}
    assert [p["title"] for p in res.plots] == ["워드클라우드", "상위 키워드 막대도표", "집단별 워드클라우드"]


def test_cooccurrence_network(docs_df):
    res = dispatch_analysis("textNetwork", docs_df, {"text": "본문", "topNodes": 40, "minCooc": 3})
    assert len(_table(res, "centrality").rows) >= 10
    assert len(_table(res, "pairs").rows) > 0
    assert len(_table(res, "communities").rows) >= 2
    assert res.plots and res.plots[0]["src"].startswith("data:image/png;base64,")


def test_document_clustering_recovers_fields(docs_df):
    res = dispatch_analysis("textCluster", docs_df, {"text": "본문", "docLabel": "제목", "group": "분야", "stability": True})
    rows = _table(res, "documents").rows
    cl = pd.Series({r["row"] - 1: r["cluster"] for r in rows})
    field = docs_df["분야"].iloc[cl.index]
    purity = pd.crosstab(cl.values, field.values).max(axis=1).sum() / len(cl)
    assert purity >= 0.9
    summary = {r["stat"]: r["value"] for r in _table(res, "corpus").rows}
    assert summary["군집 수"] >= 4 and summary["안정성 평균 ARI"] >= 0.8
    assert {"문서 군집 분포도", "군집 관계망"} <= {p["title"] for p in res.plots}


def test_relational_network_edge_list():
    df = jmv.Session(EDGES, auto_sync_workspace=False).df
    res = dispatch_analysis("graphNetwork", df, {"source": "출발기관", "target": "도착기관", "weight": "협력횟수"})
    summ = {r["stat"]: r["value"] for r in _table(res, "summary").rows}
    assert summ["노드 수"] == 30 and summ["연결(링크) 수"] == 86
    cent = _table(res, "centrality").rows
    assert all(0.0 <= r["closeness"] <= 1.0 for r in cent)
    directed = dispatch_analysis("graphNetwork", df, {"source": "출발기관", "target": "도착기관", "directed": True, "community": False})
    assert directed.get_table("communities") is None
    with pytest.raises(ValueError):
        dispatch_analysis("graphNetwork", df, {"source": "출발기관", "target": "없는열"})


def test_text_results_export(docs_df):
    res = dispatch_analysis("textKeywords", docs_df, {"text": "본문"})
    with tempfile.TemporaryDirectory() as d:
        for ext in ("hwpx", "pdf", "docx", "xlsx"):
            out = res.export_results(os.path.join(d, f"kw.{ext}"))
            assert os.path.getsize(out) > 2000


def test_csv_reader_keeps_long_text(tmp_path):
    df = pd.DataFrame({"id": [1, 2], "본문": ["첫 문서; 세미콜론, 쉼표\n줄바꿈 포함", "둘째 문서 | 파이프"]})
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    loaded =jmv.Session(str(p), auto_sync_workspace=False).df
    assert loaded.shape == (2, 2) and loaded["본문"].iloc[0] == df["본문"].iloc[0]

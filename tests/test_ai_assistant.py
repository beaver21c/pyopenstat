"""AI assistant (Gemini): program knowledge, pre-run check, request loop and web API — all without network.

The language model is replaced by a scripted client. The live check against the Gemini API is
tests/ai_eval/run_ai_eval.py (needs a key).
"""
import json
import os
import re

import pandas as pd
import pytest

from pyjamovi.ai import assistant as A
from pyjamovi.ai import gemini_client as G
from pyjamovi.ai import guard
from pyjamovi.ai import knowledge as K
from pyjamovi.ai.note import build_note
from pyjamovi.syntax_engine import execute_python_syntax

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOTH = pd.read_csv(os.path.join(ROOT, "sample_data", "ToothGrowth.csv"), sep=";")
COLS = list(TOOTH.columns)


class FakeClient:
    """Replies in order; records what it was sent."""
    def __init__(self, replies, route=None, model="fake-model"):
        self.replies, self.route_ids, self.used_model = list(replies), route or [], model
        self.sent, self.usage, self.fallbacks = [], {"calls": 0}, []

    def chat(self, messages, **kw):
        self.sent.append(messages)
        self.usage["calls"] += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def chat_json(self, messages, schema, **kw):
        self.usage["calls"] += 1
        return {"analyses": self.route_ids}


def run_code(code):
    try:
        an, out, df, name, meta = execute_python_syntax(code, default_df=TOOTH, sync_workspace=False, return_meta=True)
        return {"ok": True, "analyses": an, "stdout": out, "raw": (an, out, df, name, meta)}
    except Exception as e:                                   # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def block(code):
    return f"분석 계획\n- 계획\n\n```python\n{code}\n```"


# ------------------------------------------------------------------------------------------------ knowledge
def test_every_catalog_analysis_is_described():
    from pyjamovi.registry import get_full_catalog
    ids = [c["id"] for c in get_full_catalog()]
    assert len(ids) >= 60
    index = K.index_text()
    for i in ids:
        assert f"- {i} |" in index, i
        assert K.card(i).startswith(f"### {i} "), i
    assert "{{" not in K.system_prompt() and "anovaOneW" in K.system_prompt()


def test_schema_snapshot_matches_the_forms_in_index_html():
    """Option names of the web forms must all be in ai/analysis_schemas.json (rebuild: node scripts/build_ai_knowledge.js)."""
    html = open(os.path.join(ROOT, "reconstructed", "pyjamovi", "web", "index.html"), encoding="utf-8").read()
    region = html[html.index("const TEXT_PREP_SECTION"):html.index("const REPORT_STYLE_KEY")]
    in_forms = set(re.findall(r'\bkey\s*:\s*"([A-Za-z_][A-Za-z0-9_]*)"', region))
    known = set()
    for aid in K._snapshot()["schemas"]:
        known |= {s["key"] for s in K.slots(aid)} | {it["key"] for it in K.option_items(aid)}
    missing = sorted(in_forms - known)
    assert len(in_forms) > 300
    assert not missing, f"form options missing from ai/analysis_schemas.json: {missing[:20]}"


def test_card_lists_real_option_names_and_choices():
    c = K.card("anovaOneW")
    assert "phMethod" in c and "'gamesHowell'" in c and "deps=" in c and "group=" in c
    assert "gamesHowell" in K.choices("anovaOneW")["phMethod"]
    assert "blocks" in K.allowed_keys("linReg")            # read by the dispatcher, not shown on the form
    assert K.aliases()["xtreg"] == "panel"


def test_dataset_profile_has_no_rows():
    df = pd.DataFrame({"이름": [f"사람{i}" for i in range(50)], "본문": ["가나다라마바사 " * 20] * 50,
                       "점수": range(50), "성별": ["남", "여"] * 25})
    p = K.dataset_profile(df, "x.csv")
    assert "사람3" not in p and "가나다라" not in p
    assert "'남'" in p and "평균 24.5" in p and "50행" in p


# ------------------------------------------------------------------------------------------------ guard
@pytest.mark.parametrize("code, fragment", [
    ("sess.run('anova', deps=['len'], group='dose')", "'anovaOneW'"),
    ("sess.run('anovaOneW', deps=['len'], group='dose', postHoc='tukey')", "phMethod"),
    ("sess.run('anovaOneW', deps=['len'], group='dose', phMethod='bonf')", "선택지"),
    ("sess.run('anovaOneW', deps=['length'], group='dose')", "변수 'length' 없음"),
    ("sess.run('anovaOneW', deps=['len'])", "필수 변수 역할 group"),
    ("sess.ttest(dep='len', group='supp')", "sess.run"),
    ("import os\nsess.run('descriptives', vars=['len'])", "import"),
    ("sess.run('descriptives', vars=[c for c in cols])", "인자"),
    ("sess.compute('x', expr='len*2')", "사용법"),
    ("x = open('/etc/passwd')", "sess"),
])
def test_guard_rejects_and_explains(code, fragment):
    r = guard.check(code, COLS)
    assert not r["ok"]
    assert any(fragment in p for p in r["problems"]), r["problems"]


def test_guard_accepts_valid_code_and_new_variables():
    code = ("sess.compute('loglen', 'log(len)')\n"
            "m1 = sess.run('linReg', dep='loglen', covs=['dose'])\n"
            "m2 = sess.run('linReg', dep='loglen', covs=['dose'], factors=['supp'], collin=True)\n"
            "sess.compare_models(m1, m2, names=['모형 1', '모형 2'])\n"
            "sess.run('anovaOneW', deps=['len'], group='dose', phMethod='gamesHowell', eqv=True)")
    r = guard.check(code, COLS)
    assert r["ok"], r["problems"]
    assert r["analyses"] == ["linReg", "compareModels", "anovaOneW"] and r["changes_data"]
    sem = 'model = """\nF =~ len + dose\n"""\nsess.run("sem", model=model, stdEst=True)'
    r2 = guard.check(sem, COLS)
    assert r2["ok"] and r2["calls"][0].kwargs["model"].strip() == "F =~ len + dose", r2["problems"]
    assert not guard.check("cols = [c for c in range(3)]\nsess.run('descriptives', vars=cols)", COLS)["ok"]
    assert run_code(code)["ok"]


# ------------------------------------------------------------------------------------------------ loop
def test_wrong_option_is_returned_to_the_model_then_runs():
    fc = FakeClient([block("sess.run('anovaOneW', deps=['len'], group='dose', posthoc='gamesHowell')"),
                     block("sess.run('anovaOneW', deps=['len'], group='dose', phMethod='gamesHowell')"),
                     "결과\n- F(2, 37.743) = 68.401, p < .001\n\n유의 사항\n- Welch 검정 적용"], route=["anovaOneW"])
    stages = []
    out = A.Assistant(fc).ask("용량별 차이", TOOTH, run_code, name="ToothGrowth.csv", progress=lambda s, d="": stages.append(s))
    assert out["executed"] and out["analyses"] == ["anovaOneW"] and not out["failure"]
    assert [s["ok"] for s in out["steps"]] == [False, True] and out["steps"][0]["stage"] == "check"
    talk = fc.sent[-1]                                       # the conversation list grows in place
    assert any(m["role"] == "user" and "[실행 전 점검]" in m["content"] and "phMethod" in m["content"] for m in talk)
    assert any(m["role"] == "user" and "[실행 결과]" in m["content"] and "68.401" in m["content"] for m in talk)
    first = talk[1]["content"]
    assert "[옵션 카드]" in first and "### anovaOneW" in first and "- len:" in first
    assert stages[:3] == ["route", "code", "check"] and "fix" in stages and stages[-1] == "report"
    assert not any("확인되지 않는" in f for f in out["footer"])


def test_invented_number_is_flagged():
    fc = FakeClient([block("sess.run('anovaOneW', deps=['len'], group='dose')"),
                     "결과\n- F = 68.401, 효과크기 0.777 로 나타남"], route=["anovaOneW"])
    out = A.Assistant(fc).ask("용량별 차이", TOOTH, run_code)
    assert out["executed"]
    assert any("0.777" in f for f in out["footer"]) and not any("68.401" in f for f in out["footer"])


def test_failure_reports_the_reason_and_runs_nothing():
    bad = block("sess.run('anovaOneW', deps=['len'], group='없는변수')")
    fc = FakeClient([bad] * 6, route=[])
    out = A.Assistant(fc, max_attempts=3).ask("차이 검정", TOOTH, run_code)
    assert not out["executed"] and out["exec"] is None
    assert "없는변수" in out["failure"] and out["failure"].startswith("실행 전 점검")
    assert len(out["steps"]) == 3


def test_answer_without_code_is_not_a_failure():
    fc = FakeClient(["확인 필요\n- 성별 변수가 자료에 없음"], route=[])
    out = A.Assistant(fc).ask("성별 차이", TOOTH, run_code)
    assert not out["executed"] and out["failure"] == "" and "성별" in out["final"]


def test_note_card_shows_answer_and_failure_reason():
    n = build_note({"question": "질문", "answer": "결과\n- **값** 1.23\n\n유의 사항\n- 주의", "executed": False,
                    "failure": "실행 오류: ValueError", "model": "m", "footer": ["점검"]})
    rows = n.tables[0].rows
    assert rows[0]["text"] == "질문" and rows[1] == {"part": "결과", "text": "값 1.23"}
    assert rows[-1]["part"] == "실행 실패 원인" and "ValueError" in rows[-1]["text"]
    from pyjamovi.registry import dispatch_analysis
    assert dispatch_analysis("aiNote", TOOTH, n.options).tables[0].rows == rows


# ------------------------------------------------------------------------------------------------ client
def test_retry_seconds_and_message_merge():
    assert G.retry_seconds("Please retry in 3h31m45.2s.") == pytest.approx(3 * 3600 + 31 * 60 + 45.2)
    assert G.retry_seconds("Please retry in 12.5s.") == 12.5 and G.retry_seconds("nothing") is None
    sys_, contents = G.to_gemini([{"role": "system", "content": "s"}, {"role": "user", "content": "a"},
                                  {"role": "user", "content": "b"}, {"role": "assistant", "content": "c"}])
    assert sys_ == "s" and [c["role"] for c in contents] == ["user", "model"] and "a\n\nb" == contents[0]["parts"][0]["text"]


def test_used_up_quota_moves_to_the_next_model_without_waiting(monkeypatch):
    calls = []

    def fake_request(url, key, body=None, timeout=120):
        if url.endswith("/models?pageSize=200"):
            return {"models": [{"name": "models/gemini-9-flash", "supportedGenerationMethods": ["generateContent"]},
                               {"name": "models/gemini-9-flash-lite", "supportedGenerationMethods": ["generateContent"]}]}
        model = url.split("/models/")[1].split(":")[0]
        calls.append(model)
        if model == "gemini-9-flash":
            e = G.GeminiError(429, "quota")
            e.retry_after = 9000.0
            raise e
        return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}], "usageMetadata": {}}

    monkeypatch.setattr(G, "_request", fake_request)
    monkeypatch.setattr(G.time, "sleep", lambda s: pytest.fail("must not wait for a daily quota"))
    G._BLOCKED.clear()
    c = G.GeminiClient("k")
    assert c.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert calls == ["gemini-9-flash", "gemini-9-flash-lite"] and c.used_model == "gemini-9-flash-lite"
    assert c.fallbacks == [("gemini-9-flash", 429)]
    assert c.chat([{"role": "user", "content": "hi"}]) == "ok" and calls[-1] == "gemini-9-flash-lite" and len(calls) == 3
    G._BLOCKED.clear()


# ------------------------------------------------------------------------------------------------ web API
@pytest.fixture
def web():
    try:
        from fastapi.testclient import TestClient
        from pyjamovi.server import STATE, app
        c = TestClient(app)
    except Exception:                                        # noqa: BLE001
        pytest.skip("Starlette TestClient requires httpx")
    backend = app.state.assistant
    saved = (backend.key, backend.model, backend.client_factory, list(backend.history))
    c.post("/api/dataset/load", params={"name": "ToothGrowth.csv"})
    c.post("/api/analysis/clear")
    yield c, backend, STATE
    backend.key, backend.model, backend.client_factory, backend.history = saved


def test_api_run_adds_cards_and_records_the_request(web):
    c, backend, STATE = web
    backend.key = "k"
    backend.client_factory = lambda key, model: FakeClient(
        [block("sess.run('ttestIS', vars=['len'], group='supp', welchs=True)"), "결과\n- 차이는 유의하지 않음\n\n유의 사항\n- 없음"],
        route=["ttestIS"])
    d = c.post("/api/assistant/run", json={"question": "보충제별 차이"}).json()
    assert d["status"] == "success" and d["assistant"]["executed"] and d["assistant"]["model"] == "fake-model"
    assert [a["analysis_type"] for a in d["analyses"]] == ["ttestIS", "aiNote"]
    assert "자연어 요청: 보충제별 차이" in d["commandLog"] and "sess.run('ttestIS'" in d["commandLog"]
    assert c.get("/api/assistant/progress").json()["busy"] is False
    assert "aiNote" not in d["pythonCode"] or "sess.run(\n    'aiNote'" not in d["pythonCode"]


def test_api_failures_say_why(web):
    c, backend, STATE = web
    backend.key = ""
    d = c.post("/api/assistant/run", json={"question": "q"}).json()
    assert d["status"] == "error" and d["reason"] == "key"
    backend.key = "k"
    backend.client_factory = lambda key, model: FakeClient([G.GeminiError(429, "quota")])
    n_before = len(STATE.analyses_store)
    d = c.post("/api/assistant/run", json={"question": "q"}).json()
    assert d["status"] == "error" and d["reason"] == "quota" and "한도" in d["message"]
    assert len(STATE.analyses_store) == n_before
    backend.client_factory = lambda key, model: FakeClient([block("sess.run('ttestIS', vars=['len'], group='nope')")] * 6)
    d = c.post("/api/assistant/run", json={"question": "q"}).json()
    assert d["status"] == "success" and not d["assistant"]["executed"] and "nope" in d["assistant"]["failure"]
    assert d["analyses"][-1]["analysis_type"] == "aiNote"
    assert any(r.get("part") == "실행 실패 원인" for r in d["analyses"][-1]["tables"][0]["rows"])


def test_live_schemas_replace_the_snapshot():
    try:
        n = K.set_live_schemas({"ttestIS": {"title": "t", "slots": [{"key": "vars", "multi": True, "req": True}],
                                            "sections": [{"items": [{"type": "check", "key": "brandNew", "def": False}]}]}})
        assert n == 1 and "brandNew" in K.allowed_keys("ttestIS") and "brandNew" in K.card("ttestIS")
    finally:
        K.set_live_schemas({})
    assert "brandNew" not in K.allowed_keys("ttestIS")

# -*- coding: utf-8 -*-
"""Natural-language analysis: question -> program commands -> checked run -> report.

One request goes through these steps:
  1. routing        the model picks the analyses the question needs from the list of all analyses (ids only)
  2. code           with the option cards of those analyses and the variable descriptions it writes sess.<...> calls
  3. pre-run check  pyjamovi.ai.guard: ids, option names, choices, variables, signatures (problems go back to the model)
  4. run            on a copy of the session (the caller's `run` function); errors go back to the model
  5. report         the model reads the result tables and writes the findings; numbers that do not appear in the
                    result tables are flagged
The caller decides what to do with the run that succeeded (the web server adds it to the result sheet and the
command record). The model never receives rows of the dataset.
"""
import re
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import guard
from . import knowledge as K

MAX_ATTEMPTS = 4          # code blocks the model may submit for one question
MAX_CARDS = 4
RESULT_CHARS = 9000
_CODE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
_DECIMAL = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)?\.\d+")
_COMMON = {0.05, 0.01, 0.001, 0.1, 0.5, 0.95, 0.99, 0.9, 0.8, 0.2, 0.3}

ROUTER_SYSTEM = """통계 프로그램 OpenStat Studio 의 요청 분류기임. 이용자의 요청을 처리하는 데 쓸 분석을 아래 「분석 목록」에서 고름.
- analyses 에 분석 id 를 0~4개 적음. 요청한 분석과, 그 분석에 흔히 함께 쓰는 분석(요청에 드러난 경우)만 고름
- 자료 처리(변수 계산·재코딩·필터 등)만 필요하거나 설명만 필요한 질문이면 빈 목록
- 이용자가 Stata·R·SPSS 명령 이름으로 말하면 「관련 명령」과 이름으로 대응하는 분석을 고름
- 모형 비교표는 compareModels, 종단 구조방정식 견본(교차지연·잠재성장·잠재변화)은 semTemplate, 그림만 요청하면 graph

# 분석 목록
형식: `id | 이름 | 변수 역할(* 필수, [] 변수 여러 개) | 관련 명령`

{index}
"""


def extract_code(text: str) -> Optional[str]:
    blocks = [b.strip("\n") for b in _CODE.findall(text or "") if b.strip()]
    return "\n".join(blocks) if blocks else None


def strip_code(text: str) -> str:
    return _CODE.sub("", text or "").strip()


# ----------------------------------------------------------------------------------------------- results -> text
def summarize_results(analyses: List[Any], max_chars: int = RESULT_CHARS, max_rows: int = 40) -> str:
    """Result tables as the screen shows them (same rounding), for the model to read."""
    if not analyses:
        return "(결과표 없음)"
    budget = max(1500, max_chars // max(1, len(analyses)))
    parts = []
    for a in analyses:
        out = [f"■ {getattr(a, 'title', '')} (분석 id: {getattr(a, 'name', '')})"]
        for t in getattr(a, "tables", []) or []:
            heads = [t.display_title(c) or c.get("name", "") for c in t.columns]
            out.append(f"[표] {t.title}")
            out.append(" | ".join(str(h) for h in heads))
            for r in t.rows[:max_rows]:
                out.append(" | ".join(t._format_cell(r.get(c["name"]), t.cell_meta(r, c)) for c in t.columns))
            if len(t.rows) > max_rows:
                out.append(f"… (이하 {len(t.rows) - max_rows}행 생략)")
            for n in t.notes:
                out.append(f"주: {n}")
        apa = getattr(a, "apa_narrative", "") or ""
        if apa:
            out.append("요약 문장: " + " ".join(str(apa).split())[:600])
        plots = [p.get("title", "그림") for p in getattr(a, "plots", []) or []]
        if plots:
            out.append("그림(화면에 표시됨): " + ", ".join(plots[:8]))
        txt = "\n".join(out)
        parts.append(txt if len(txt) <= budget else txt[:budget] + "\n… (이 분석의 나머지 결과 생략)")
    return "\n\n".join(parts)


def _result_numbers(analyses: List[Any]) -> List[float]:
    vals: List[float] = []
    for a in analyses:
        for t in getattr(a, "tables", []) or []:
            for r in t.rows:
                for v in r.values():
                    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and np.isfinite(v):
                        vals.append(float(v))
    return vals


def unverified_numbers(final: str, analyses: List[Any], shown: str) -> List[str]:
    """Decimal numbers of the report that match no value of the result tables (after rounding)."""
    vals = _result_numbers(analyses)
    arr = np.array(vals) if vals else np.array([])
    flagged = []
    for tok in dict.fromkeys(_DECIMAL.findall(final or "")):
        body = tok.replace(",", "")
        digits = len(body.split(".")[1])
        x = float(body)
        if abs(x) in _COMMON or tok in shown or body in shown or body.lstrip("-0") in shown:
            continue
        tol = 0.51 * 10 ** (-digits)
        if arr.size and (np.any(np.abs(arr - x) <= tol) or np.any(np.abs(np.abs(arr) - abs(x)) <= tol)
                         or np.any(np.abs(arr * 100 - x) <= tol)):
            continue
        flagged.append(tok)
    return flagged


# ----------------------------------------------------------------------------------------------- assistant
class Assistant:
    """`client` needs chat(messages) -> str and chat_json(messages, schema) -> dict (see gemini_client.GeminiClient)."""

    def __init__(self, client, max_attempts: int = MAX_ATTEMPTS):
        self.client, self.max_attempts = client, max_attempts

    # -- step 1
    def route(self, question: str, profile: str, history_text: str = "") -> List[str]:
        ids = K.analysis_ids() + list(K.EXTRA_IDS)
        schema = {"type": "OBJECT", "properties": {"analyses": {"type": "ARRAY", "items": {"type": "STRING", "enum": ids}}},
                  "required": ["analyses"]}
        user = "\n\n".join(x for x in ("[자료]\n" + profile, history_text, "[요청]\n" + question) if x)
        try:
            d = self.client.chat_json([{"role": "system", "content": ROUTER_SYSTEM.format(index=K.index_text())},
                                       {"role": "user", "content": user}], schema)
        except Exception:                                 # noqa: BLE001 — routing is a help, not a requirement
            return []
        picked = [i for i in (d.get("analyses") or []) if i in ids]
        return list(dict.fromkeys(picked))[:MAX_CARDS]

    # -- steps 2-5
    def ask(self, question: str, df, run: Callable[[str], Dict[str, Any]], *, name: str = "", meta: Optional[Dict[str, Any]] = None,
            history: Optional[List[Dict[str, str]]] = None, existing: Optional[List[str]] = None, lang: str = "ko",
            progress: Optional[Callable[[str, str], None]] = None) -> Dict[str, Any]:
        """`progress(stage, detail)` is called as the request moves on: route, code, check, run, report, fix."""
        t0 = time.time()
        meta = meta or {}
        say = progress or (lambda stage, detail="": None)
        say("route", "")
        profile = K.dataset_profile(df, name, meta.get("value_labels"), meta.get("variable_labels"), meta.get("measure_levels"))
        hist = ""
        if history:
            hist = "[이전 요청] (이어지는 요청이면 앞의 변수·방법을 이어받음)\n" + "\n".join(
                f"- 요청: {h.get('question', '')}\n  실행한 코드: {(h.get('code') or '(없음)').strip()}" for h in history[-3:])
        routed = self.route(question, profile, hist)
        shown = list(routed)
        blocks = ["[자료]\n" + profile]
        if existing:
            blocks.append("[결과 시트에 이미 있는 분석]\n" + "\n".join(f"- {t}" for t in existing[-8:]))
        if hist:
            blocks.append(hist)
        if shown:
            blocks.append("[옵션 카드]\n" + "\n\n".join(K.card(i) for i in shown))
        blocks.append(f"[요청]\n{question}\n\n(요청이 가리키는 변수가 [자료]에 없으면 뜻이 다른 변수로 대신하지 않고, 코드 없이 「확인 필요」로 되물음.)"
                      + ("\n(Answer in English.)" if lang == "en" else ""))
        messages = [{"role": "system", "content": K.system_prompt()}, {"role": "user", "content": "\n\n".join(blocks)}]

        steps: List[Dict[str, Any]] = []
        good: Optional[Dict[str, Any]] = None          # the last run that succeeded
        final, attempts = "", 0
        columns = [str(c) for c in df.columns]
        say("code", ", ".join(routed))
        while True:
            reply = self.client.chat(messages)
            messages.append({"role": "assistant", "content": reply})
            code = extract_code(reply)
            if not code:
                final = reply.strip()
                break
            if attempts >= self.max_attempts:
                final = strip_code(reply)
                break
            attempts += 1
            say("check", f"{attempts}")
            chk = guard.check(code, columns)
            if not chk["ok"]:
                steps.append({"code": code, "stage": "check", "ok": False, "error": " / ".join(chk["problems"])[:1500]})
                need = [i for i in chk["analyses"] if i not in shown][:MAX_CARDS]
                shown += need
                fb = "[실행 전 점검] 실행하지 않음. 아래를 고쳐 코드 블록을 다시 작성 바람.\n" + "\n".join(f"- {p}" for p in chk["problems"][:12])
                if need:
                    fb += "\n\n[옵션 카드]\n" + "\n\n".join(K.card(i) for i in need)
                messages.append({"role": "user", "content": fb})
                say("fix", chk["problems"][0][:160])
                continue
            need = [i for i in chk["analyses"] if i not in shown][:MAX_CARDS]
            say("run", ", ".join(chk["analyses"]))
            res = run(code)
            if not res.get("ok"):
                err = str(res.get("error") or "실행 오류")[:1500]
                steps.append({"code": code, "stage": "run", "ok": False, "error": err})
                shown += need
                fb = f"[실행 오류]\n{err}\n\n원인을 고친 코드 블록을 다시 작성 바람. 같은 코드를 다시 내지 않음."
                if need:
                    fb += "\n\n[옵션 카드]\n" + "\n\n".join(K.card(i) for i in need)
                messages.append({"role": "user", "content": fb})
                say("fix", err.splitlines()[0][:160])
                continue
            say("report", "")
            summary = summarize_results(res.get("analyses") or [])
            good = {"code": code, "exec": res, "summary": summary, "changes_data": chk["changes_data"], "analyses": chk["analyses"]}
            steps.append({"code": code, "stage": "run", "ok": True, "error": ""})
            note = ""
            if chk["changes_data"]:
                note = "\n(자료 처리 명령이 실행되어 열려 있는 자료가 바뀜. 「유의 사항」에 밝힘)"
            out = (res.get("stdout") or "").strip()
            messages.append({"role": "user", "content": "[실행 결과]\n" + summary + (f"\n\n[출력]\n{out[:1500]}" if out else "") + note
                             + "\n\n이 결과만 근거로 「결과」와 「유의 사항」을 작성 바람. 결과표에 없는 수치는 적지 않음."})

        footer: List[str] = []
        if good:
            if extract_code(final) or not final:
                final = strip_code(final) or "해석 문장을 만들지 못함. 위 결과표 참조."
            bad = unverified_numbers(final, good["exec"].get("analyses") or [], good["summary"])
            if bad:
                footer.append("결과표에서 확인되지 않는 수치: " + ", ".join(bad[:12]) + " (결과표의 값을 우선함)")
            failed = [s for s in steps if not s["ok"]]
            if failed:
                footer.append(f"코드 수정 {len(failed)}회 후 실행됨")
        elif steps:
            last = steps[-1]
            footer.append("실행된 분석 없음: " + (last.get("error") or "")[:300])
            if not final or extract_code(final):
                final = "요청을 실행하지 못함. 마지막 오류: " + (last.get("error") or "")[:400]
        failure = ""
        if not good and steps:
            last = steps[-1]
            failure = ("실행 전 점검에서 걸림: " if last["stage"] == "check" else "실행 오류: ") + (last.get("error") or "")[:400]
        usage = dict(getattr(self.client, "usage", {}) or {})
        return {"ok": True, "question": question, "final": final, "footer": footer, "code": good["code"] if good else "",
                "exec": good["exec"] if good else None, "executed": bool(good), "changes_data": bool(good and good["changes_data"]),
                "analyses": good["analyses"] if good else [], "routed": routed, "steps": steps, "failure": failure,
                "fallbacks": list(getattr(self.client, "fallbacks", []) or []),
                "n_calls": usage.get("calls"), "usage": usage, "model": getattr(self.client, "used_model", None),
                "elapsed_sec": round(time.time() - t0, 1)}

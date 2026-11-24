# -*- coding: utf-8 -*-
"""Result card that holds the assistant's answer to a natural-language request (analysis type ``aiNote``).

The card is built from its stored params (question, answer, notes), so it is never recomputed by the language
model when the sheet is redrawn. The analyses the request ran are separate cards placed above it.
"""
from typing import Any, Dict

from ..analyses.base import BaseAnalysis, ResultsTable

NOTE_TYPE = "aiNote"
_HEADS = ("결과", "유의 사항", "분석 계획", "확인 필요", "안내", "Results", "Notes", "Plan", "Clarification", "Guidance")


def _clean(line: str) -> str:
    return line.replace("**", "").replace("$", "").replace("\\text{", "").replace("`", "").strip()


def build_note(params: Dict[str, Any]) -> BaseAnalysis:
    q = str(params.get("question") or "")
    lang_en = params.get("lang") == "en"
    a = BaseAnalysis(NOTE_TYPE, ("AI assistant · " if lang_en else "AI 분석 지원 · ") + (q if len(q) <= 64 else q[:62] + "…"), None, dict(params))
    t = ResultsTable("answer", "Answer" if lang_en else "응답")
    t.add_column("part", "Part" if lang_en else "구분", "text")
    t.add_column("text", "Content" if lang_en else "내용", "text")
    t.add_row({"part": "Request" if lang_en else "요청", "text": q})
    part = ""
    for raw in str(params.get("answer") or "").splitlines():
        line = _clean(raw)
        if not line:
            continue
        head = line.strip("#:： ").strip()
        if head in _HEADS or (line.startswith("#") and len(head) <= 20):
            part = head
            continue
        t.add_row({"part": part, "text": line.lstrip("-·•* ").strip()})
        part = ""
    if params.get("failure"):
        t.add_row({"part": "Not run — reason" if lang_en else "실행 실패 원인", "text": str(params["failure"])})
    for f in params.get("footer") or []:
        t.add_note(("[check] " if lang_en else "[자동 점검] ") + str(f))
    model = params.get("model") or ("unknown model" if lang_en else "모형 미상")
    if lang_en:
        t.add_note(f"Written by Gemini API ({model}) from the result tables above; values in the tables take precedence."
                   if params.get("executed") else f"Answer of Gemini API ({model}); no analysis was run.")
    else:
        t.add_note(f"Gemini API({model})가 위 결과표를 읽고 작성한 글임. 수치는 결과표의 값을 우선함" if params.get("executed")
                   else f"Gemini API({model})의 응답이며 실행된 분석은 없음")
    a.add_table(t)
    code = str(params.get("code") or "").strip()
    a.syntax_python = (f"# {q}\n{code}" if code else f"# {q}")
    a.apa_narrative = ""
    return a

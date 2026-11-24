#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live check of the AI assistant against the Gemini API (not part of pytest: it needs a key and network).

Each request of questions.json runs through the assistant on a sample dataset; the commands the model wrote are
compared with the expected analysis ids, options and data methods. Nothing is compared with fixed numbers:
the question here is whether the model drives the program correctly.

    export GEMINI_API_KEY=...                       # never written to a file
    .venv/bin/python tests/ai_eval/run_ai_eval.py --name lite --model gemini-3.1-flash-lite
    .venv/bin/python tests/ai_eval/run_ai_eval.py --name lite --only T01,M04 --no-knowledge   # baseline without the option cards

Results: tests/ai_eval/results/<name>.json and <name>.md
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "reconstructed"))
os.environ.setdefault("MPLBACKEND", "Agg")

import openstat as ost  # noqa: E402
from pyjamovi.ai import Assistant, GeminiClient, guard  # noqa: E402
from pyjamovi.ai import knowledge as K  # noqa: E402
from pyjamovi.syntax_engine import execute_python_syntax  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def score(item, out):
    """-> (verdict, detail). verdict: pass | wrong | not_run | error"""
    if item.get("expect") == "no_run":
        return ("pass", "") if not out["executed"] and not out["steps"] else ("wrong", "코드를 실행함(되묻기·설명이 기대됨)")
    if not out["executed"]:
        return "not_run", out.get("failure") or "코드 없음"
    calls, _ = guard.parse(out["code"])
    used = out["analyses"]
    miss = []
    for want in item.get("analyses", []):
        opts = want if isinstance(want, list) else [want]
        if not any(o in used for o in opts):
            miss.append("분석 " + "/".join(opts))
    for m in item.get("methods", []):
        if not any(c.method == m or (m == "filter_rows" and c.method == "filter") for c in calls):
            miss.append("메서드 " + m)
    for aid, exp in (item.get("options") or {}).items():
        kws = [c.kwargs for c in calls if c.method == "run" and c.args and c.args[0] == aid]
        if not kws:
            continue
        for k, v in exp.items():
            def ok(kw):
                got = kw.get(k)
                if item.get("either_roles") and k in ("rows", "cols"):
                    return {kw.get("rows"), kw.get("cols")} == {exp.get("rows"), exp.get("cols")}
                if isinstance(v, list):
                    return isinstance(got, (list, tuple)) and set(v) <= set(got) or got == v[0]
                if isinstance(got, (list, tuple)) and not isinstance(v, list):
                    return v in got
                return got == v
            if not any(ok(kw) for kw in kws):
                miss.append(f"옵션 {aid}.{k}={v!r} (작성: {kws[0].get(k)!r})")
    return ("pass", "") if not miss else ("wrong", "; ".join(miss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run")
    ap.add_argument("--model", default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between requests (rate limits)")
    ap.add_argument("--no-knowledge", action="store_true", help="baseline: no analysis list, no option cards, no pre-run check")
    a = ap.parse_args()
    items = json.load(open(os.path.join(HERE, "questions.json"), encoding="utf-8"))
    if a.only:
        keep = set(a.only.split(","))
        items = [i for i in items if i["id"] in keep]
    if a.no_knowledge:                                   # what the model does when it is told nothing about the program
        K.index_text = lambda: "(목록 없음)"
        K.card = lambda aid: ""
        guard.check = lambda code, columns: {"ok": True, "problems": [], "analyses": [c.args[0] for c in guard.parse(code)[0] if c.method == "run" and c.args],
                                             "changes_data": False, "calls": []}
    client = GeminiClient(model=a.model)
    rows, t0 = [], time.time()
    for it in items:
        sess = ost.Session(os.path.join(ROOT, "sample_data", it["data"]), auto_sync_workspace=False)

        def run(code, sess=sess):
            try:
                an, out, df, name, meta = execute_python_syntax(code, default_df=sess.df, sync_workspace=False, return_meta=True)
                return {"ok": True, "analyses": an, "stdout": out}
            except Exception as e:                       # noqa: BLE001
                return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        try:
            out = Assistant(client).ask(it["q"], sess.df, run, name=it["data"])
            verdict, detail = score(it, out)
        except Exception as e:                           # noqa: BLE001
            out, verdict, detail = {"steps": [], "code": "", "final": "", "footer": []}, "error", f"{type(e).__name__}: {e}"[:300]
        fixes = sum(1 for s in out["steps"] if not s["ok"])
        rows.append({"id": it["id"], "data": it["data"], "q": it["q"], "verdict": verdict, "detail": detail, "code": out.get("code", ""),
                     "rewrites": fixes, "first_errors": [s["error"][:300] for s in out["steps"] if not s["ok"]][:2],
                     "routed": out.get("routed"), "model": out.get("model"), "sec": out.get("elapsed_sec"),
                     "footer": out.get("footer"), "final": (out.get("final") or "")[:700]})
        print(f"{it['id']} {verdict:8s} rewrites={fixes} {out.get('elapsed_sec')}s {detail[:150]}", flush=True)
        if a.sleep:
            time.sleep(a.sleep)
    n = len(rows)
    ok = sum(r["verdict"] == "pass" for r in rows)
    first = sum(r["verdict"] == "pass" and r["rewrites"] == 0 for r in rows)
    meta = {"name": a.name, "model": a.model or "auto", "knowledge": not a.no_knowledge, "n": n, "pass": ok, "pass_first_try": first,
            "usage": client.usage, "elapsed_sec": round(time.time() - t0, 1)}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump({"meta": meta, "rows": rows}, open(os.path.join(HERE, "results", a.name + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    md = [f"# AI 분석 지원 실측 — {a.name}", "",
          f"- 모형: {meta['model']} · 프로그램 지식 제공: {'예' if meta['knowledge'] else '아니오(기준선)'} · 문항 {n}개",
          f"- 통과 {ok}/{n} ({ok / max(n, 1):.0%}) · 고쳐 쓰기 없이 통과 {first}/{n} · 모형 호출 {client.usage.get('calls')}회 · {meta['elapsed_sec']}초", "",
          "| 문항 | 자료 | 판정 | 고쳐 쓰기 | 초 | 비고 |", "|---|---|---|---:|---:|---|"]
    md += [f"| {r['id']} | {r['data']} | {r['verdict']} | {r['rewrites']} | {r['sec']} | {r['detail'][:120].replace('|', '/')} |" for r in rows]
    md += ["", "## 통과하지 못한 문항", ""]
    for r in rows:
        if r["verdict"] != "pass":
            md += [f"### {r['id']} · {r['verdict']}", f"- 요청: {r['q']}", f"- 사유: {r['detail']}", "", "```python", r["code"] or "(코드 없음)", "```", ""]
    open(os.path.join(HERE, "results", a.name + ".md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

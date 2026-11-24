# -*- coding: utf-8 -*-
"""What the AI assistant is told about OpenStat Studio — built from the program itself, not written by hand.

- ``analysis_schemas.json``: the analysis forms of the web UI (role slots, option names, defaults, choices),
  exported by ``scripts/build_ai_knowledge.js``. An open browser window also posts its live copy
  (``set_live_schemas``), so a form edited after the snapshot is still described correctly.
- ``registry.py``: option names the dispatcher reads for the built-in analyses (accepted, not advertised).
- The dataset in the session: variable names, types, summary statistics and category labels (no rows).

The language model sees a one-line index of every analysis in each request and the full option card only of the
analyses a question needs; ``pyjamovi.ai.guard`` checks its code against the same cards before anything runs.
"""
import ast
import difflib
import json
import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_FILE = os.path.join(HERE, "analysis_schemas.json")
PROMPT_FILE = os.path.join(HERE, "system_prompt.md")

PREFER = {"num": "수치", "cat": "범주", "bin": "이항(0/1)", "any": "모든 유형", "text": "문자", "id": "ID"}
COMMON_KEYS = {"plotStyle"}
# Analyses without a UI form: described by hand (kept short; checked by tests against the dispatcher)
EXTRA_CARDS = {
    "compareModels": {
        "title": "모형 비교표 (Stata esttab 형식, 계수표가 있는 분석 결과를 나란히)",
        "text": ("sess.run 대신 다음 형식을 씀: 각 모형을 변수에 담은 뒤 sess.compare_models(m1, m2, names=['모형 1', '모형 2'], "
                 "stats=['N', 'R2', 'R2_adj', 'F']).\n"
                 "예: m1 = sess.run('linReg', dep='y', covs=['x1'])\n    m2 = sess.run('linReg', dep='y', covs=['x1', 'x2'])\n"
                 "    sess.compare_models(m1, m2, names=['모형 1', '모형 2'])"),
    },
}

EXTRA_IDS = ("semTemplate",)          # not analyses of the catalog: session methods the router can still ask a card for
_LIVE: Dict[str, Any] = {}


# ----------------------------------------------------------------------------------------------- schemas
@lru_cache(maxsize=1)
def _snapshot() -> Dict[str, Any]:
    with open(SCHEMA_FILE, encoding="utf-8") as f:
        return json.load(f)


def set_live_schemas(schemas: Dict[str, Any]) -> int:
    """Forms posted by an open browser window replace the snapshot entries of the same analysis."""
    ok = {k: v for k, v in (schemas or {}).items() if isinstance(v, dict) and isinstance(v.get("sections", []), list)}
    _LIVE.clear()
    _LIVE.update(ok)
    return len(ok)


def schemas() -> Dict[str, Any]:
    return {**_snapshot()["schemas"], **_LIVE}


def catalog() -> List[Dict[str, Any]]:
    from ..registry import get_full_catalog
    return [c for c in get_full_catalog() if c["id"] in schemas() or c["id"] in EXTRA_CARDS]


def analysis_ids() -> List[str]:
    return [c["id"] for c in catalog()]


def aliases() -> Dict[str, str]:
    """Other names the dispatcher accepts -> catalog id (Stata command names included)."""
    from ..analyses import extensions
    out = dict(_registry()["aliases"])
    out.update(extensions.aliases())
    return {k: v for k, v in out.items() if v in schemas() or v in EXTRA_CARDS}


def _items(schema: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for sec in schema.get("sections") or []:
        for it in sec.get("items") or []:
            if isinstance(it, dict) and it.get("key"):
                yield {**it, "_secShowIf": sec.get("showIf")}


def slots(aid: str) -> List[Dict[str, Any]]:
    return list((schemas().get(aid) or {}).get("slots") or [])


def option_items(aid: str) -> List[Dict[str, Any]]:
    return list(_items(schemas().get(aid) or {}))


def default_of(item: Dict[str, Any]) -> Any:
    if "def" in item:
        return item["def"]
    t = item.get("type")
    if t == "select":
        opts = item.get("options") or []
        return opts[0][0] if opts else None
    return {"check": False, "subset": [], "text": "", "textarea": ""}.get(t)


def choices(aid: str) -> Dict[str, List[Any]]:
    return {it["key"]: [o[0] for o in it.get("options") or []] for it in option_items(aid) if it.get("type") == "select"}


def allowed_keys(aid: str) -> Optional[set]:
    """Option names analysis `aid` reads; None = no form to check against (anything is passed through)."""
    if aid not in schemas():
        return None
    keys = {s["key"] for s in slots(aid)} | {it["key"] for it in option_items(aid)}
    return keys | _registry()["keys"].get(aid, set()) | COMMON_KEYS


# ----------------------------------------------------------------------------------------------- registry
@lru_cache(maxsize=1)
def _registry() -> Dict[str, Any]:
    """Option names read by each branch of registry._dispatch_inner, and its alias table."""
    path = os.path.join(os.path.dirname(HERE), "registry.py")
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_dispatch_inner"), None)
    keys: Dict[str, set] = {}
    alias: Dict[str, str] = {}
    if fn is None:
        return {"keys": keys, "aliases": alias}

    def names_in_test(test) -> List[str]:
        out = []
        for n in ast.walk(test):
            if isinstance(n, ast.Compare) and isinstance(n.left, ast.Name) and n.left.id == "t":
                for c in n.comparators:
                    out += [x.value for x in ast.walk(c) if isinstance(x, ast.Constant) and isinstance(x.value, str)]
        return out

    def keys_in(body) -> set:
        found = set()
        for stmt in body:
            for n in ast.walk(stmt):
                if not isinstance(n, ast.Call):
                    continue
                f_ = n.func
                if isinstance(f_, ast.Attribute) and f_.attr == "get" and isinstance(f_.value, ast.Name) and f_.value.id == "p":
                    if n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                        found.add(n.args[0].value)
                elif isinstance(f_, ast.Name) and f_.id in ("_opt", "_flag"):
                    found |= {a.value for a in n.args[1:3] if isinstance(a, ast.Constant) and isinstance(a.value, str)}
        return found

    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "alias_map" for t in node.targets):
            try:
                alias.update(ast.literal_eval(node.value))
            except (ValueError, SyntaxError):
                pass
        if isinstance(node, ast.If):
            ids = names_in_test(node.test)
            if ids:
                ks = keys_in(node.body)
                for i in ids:
                    keys.setdefault(i, set()).update(ks)
    return {"keys": keys, "aliases": alias}


# ----------------------------------------------------------------------------------------------- text for the model
def _clip(s: Any, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _lit(v: Any) -> str:
    return repr(v)


def _show_if(cond: Any) -> str:
    if not isinstance(cond, dict) or not cond:
        return ""
    parts = []
    for k, v in cond.items():
        if v == "__filled__":
            parts.append(f"{k} 지정 시")
        elif isinstance(v, list):
            parts.append(f"{k} ∈ {{{', '.join(map(str, v))}}}")
        else:
            parts.append(f"{k}={v}")
    return " [적용 조건: " + ", ".join(parts) + "]"


def title_of(aid: str) -> str:
    snap = _snapshot()
    ko = snap.get("menu_ko", {}).get(aid)
    sc = schemas().get(aid) or {}
    base = sc.get("title") or EXTRA_CARDS.get(aid, {}).get("title") or aid
    cat = next((c["title"] for c in catalog() if c["id"] == aid), "")
    out = [ko] if ko else []
    for t in (cat, base):                 # Korean menu name + the catalog title (it carries the Stata command names)
        if t and len(out) < 2 and all(t not in o and o not in t for o in out):
            out.append(t)
    return " / ".join(out) or aid


def example_call(aid: str) -> str:
    args = []
    for s in slots(aid):
        if s.get("req"):
            args.append(f"{s['key']}=" + ("['변수', …]" if s.get("multi") else "'변수'"))
    return f"sess.run({aid!r}" + ("".join(", " + a for a in args)) + ")"


def index_text() -> str:
    """One line per analysis: id | names | role slots (* = required, [] = list of variables)."""
    by_target: Dict[str, List[str]] = {}
    for a, t in aliases().items():
        by_target.setdefault(t, []).append(a)
    lines, group = [], None
    for c in catalog():
        if c.get("group") != group:
            group = c.get("group")
            lines.append(f"[{group}]")
        aid = c["id"]
        sl = ", ".join(s["key"] + ("[]" if s.get("multi") else "") + ("*" if s.get("req") else "") for s in slots(aid))
        al = [a for a in by_target.get(aid, []) if a.isascii() and a.lower() != aid.lower()][:6]
        lines.append(f"- {aid} | {_clip(title_of(aid), 110)}" + (f" | 변수 역할: {sl}" if sl else "")
                     + (f" | 관련 명령: {', '.join(al)}" if al else ""))
    lines.append("[sess.run 이 아닌 형식]")
    lines.append("- semTemplate | 종단 구조방정식 견본: 교차지연(CLPM·RI-CLPM), 잠재성장(LGM), 잠재변화(LCS) | sess.sem_template('<견본 id>', …)")
    return "\n".join(lines)


def _sem_template_card() -> str:
    from ..analyses.sem_templates import template_catalog
    out = ["### semTemplate — 종단 구조방정식 견본 (교차지연·잠재성장·잠재변화)",
           "sess.run 대신 sess.sem_template('<견본 id>', 역할=[변수, …], 옵션=값) 형식으로 씀. 시점별 변수는 1시점부터 순서대로 목록에 넣음.",
           "견본 (id — 이름: 역할 [* 필수] / 옵션 = 기본값):"]
    for t in template_catalog():
        roles = ", ".join(r["key"] + ("*" if r.get("req") else "") for r in t.get("roles") or [])
        opts = ", ".join(f"{o['key']}={o.get('default')!r}" for o in t.get("options") or [])
        out.append(f"- {t['id']} — {t.get('title', '')} ({t.get('short', '')}): {roles}" + (f" / {opts}" if opts else ""))
    out.append("호출 예: sess.sem_template('riclpm', x_waves=['x1', 'x2', 'x3'], y_waves=['y1', 'y2', 'y3'])")
    return "\n".join(out)


def card(aid: str) -> str:
    """Full option reference of one analysis."""
    if aid == "semTemplate":
        return _sem_template_card()
    if aid in EXTRA_CARDS and aid not in schemas():
        e = EXTRA_CARDS[aid]
        return f"### {aid} — {e['title']}\n{e['text']}"
    sc = schemas().get(aid)
    if not sc:
        return ""
    out = [f"### {aid} — {title_of(aid)}"]
    if sc.get("intro"):
        out.append("설명: " + _clip(sc["intro"], 600))
    sl = slots(aid)
    if sl:
        out.append("변수 역할 (값은 자료의 변수 이름):")
        for s in sl:
            form = "['변수', …]" if s.get("multi") else "'변수'"
            tags = ("필수" if s.get("req") else "선택") + ", " + PREFER.get(s.get("prefer"), str(s.get("prefer") or "모든 유형"))
            out.append(f"- {s['key']}={form} ({tags}) {_clip(s.get('label', ''), 90)}"
                       + (f" — {_clip(s['hint'], 120)}" if s.get("hint") else "") + _show_if(s.get("showIf")))
    else:
        out.append("변수 역할: 없음(자료의 변수를 지정하지 않는 분석)")
    items = option_items(aid)
    if items:
        out.append("옵션 (이름 = 기본값 — 뜻):")
    for it in items:
        t, d = it.get("type"), default_of(it)
        line = f"- {it['key']} = {_lit(d)} — {_clip(it.get('label', ''), 90)}"
        if t == "select":
            line += " | 선택지: " + "; ".join(f"{_lit(o[0])} {_clip(o[1], 60)}" for o in it.get("options") or [])
        elif t == "number":
            rng = [f"{k} {it[k]}" for k in ("min", "max") if k in it]
            line += f" (숫자{', ' + ', '.join(rng) if rng else ''})"
        elif t == "check":
            line += " (True/False)"
        elif t == "subset":
            line += f" (변수 목록, {it.get('from', '')}에 넣은 변수 중에서)"
        elif t in ("text", "textarea"):
            line += " (문자열" + (f", 예: {_clip(it['placeholder'], 70)}" if it.get("placeholder") else "") + ")"
        if it.get("help"):
            line += " · " + _clip(it["help"], 140)
        line += _show_if(it.get("showIf")) + _show_if(it.get("_secShowIf"))
        out.append(line)
    out.append("호출 예: " + example_call(aid))
    return "\n".join(out)


def suggest(name: str, pool: Iterable[str], n: int = 4) -> List[str]:
    pool = list(pool)
    low = {p.lower(): p for p in pool}
    hits = difflib.get_close_matches(str(name).lower(), list(low), n=n, cutoff=0.55)
    return [low[h] for h in hits]


# ----------------------------------------------------------------------------------------------- dataset
def _num(v: float) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    a = abs(v)
    if a != 0 and (a >= 1e6 or a < 1e-3):
        return f"{v:.3g}"
    return f"{v:.3f}".rstrip("0").rstrip(".") if a < 1000 else f"{v:,.1f}".rstrip("0").rstrip(".")


def dataset_profile(df: pd.DataFrame, name: str = "", value_labels: Optional[Dict[str, Dict[Any, str]]] = None,
                    variable_labels: Optional[Dict[str, str]] = None, measure_levels: Optional[Dict[str, str]] = None,
                    max_cols: int = 120, max_levels: int = 12) -> str:
    """Variable descriptions only: name, type, missing count, summary statistics, category labels. No rows."""
    value_labels, variable_labels, measure_levels = value_labels or {}, variable_labels or {}, measure_levels or {}
    out = [f"자료: {name or '(이름 없음)'} · {len(df):,}행 × {df.shape[1]}변수"]
    for i, col in enumerate(df.columns):
        if i >= max_cols:
            rest = [str(c) for c in df.columns[max_cols:]]
            out.append(f"- (그 밖의 변수 {len(rest)}개, 이름만) " + ", ".join(rest[:300]))
            break
        s = df[col]
        miss = int(s.isna().sum())
        nun = int(s.nunique(dropna=True))
        lab = variable_labels.get(col)
        head = f"- {col}" + (f" 「{_clip(lab, 60)}」" if lab else "")
        level = measure_levels.get(col)
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            x = pd.to_numeric(s, errors="coerce")
            kind = level or ("수치(이산)" if nun <= max_levels else "수치(연속)")
            desc = f"{kind}, 최솟값 {_num(x.min())}, 평균 {_num(x.mean())}, 표준편차 {_num(x.std())}, 최댓값 {_num(x.max())}"
            if nun <= max_levels:
                vals = sorted(x.dropna().unique().tolist())
                vl = value_labels.get(col) or {}
                shown = [f"{_num(float(v))}" + (f"={_clip(vl.get(v, vl.get(int(v) if float(v).is_integer() else v, '')), 20)}"
                                                if vl else "") for v in vals]
                desc += f", 값 {nun}개: {', '.join(shown)}"
        elif pd.api.types.is_datetime64_any_dtype(s):
            desc = f"날짜, {s.min()} ~ {s.max()}"
        else:
            st = s.dropna().astype(str)
            avg = float(st.str.len().mean()) if len(st) else 0.0
            if avg > 40:
                desc = f"문자(긴 글, 평균 {avg:.0f}자, 값은 제시하지 않음)"
            elif nun <= max_levels:
                desc = f"{level or '범주(문자)'}, 범주 {nun}개: " + ", ".join(repr(_clip(v, 24)) for v in st.value_counts().index.tolist())
            else:
                desc = f"{level or '문자'}, 서로 다른 값 {nun:,}개(값은 제시하지 않음)"
        out.append(f"{head}: {desc}" + (f", 결측 {miss:,}" if miss else ""))
    return "\n".join(out)


def system_prompt() -> str:
    with open(PROMPT_FILE, encoding="utf-8") as f:
        return f.read().replace("{{ANALYSIS_INDEX}}", index_text())

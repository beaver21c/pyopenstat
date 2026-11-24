# -*- coding: utf-8 -*-
"""Check of model-written code before it runs.

The assistant may only write a flat list of ``sess.<method>(...)`` calls with literal arguments. Every call is
checked against what the program really has — analysis ids, option names, option choices, variable names and
method signatures — so a misspelled or invented option is returned to the model with the valid names instead of
being silently ignored by the dispatcher. Code that passes contains no imports, loops, attribute access or
function calls other than the listed session methods, so running it cannot do anything a menu click could not.
"""
import ast
import inspect
from typing import Any, Dict, List, Optional, Set, Tuple

from . import knowledge as K

DATA_METHODS = ("compute", "recode", "filter_rows", "filter", "standardize", "egen", "collapse", "reshape", "merge",
                "joinby", "append", "convert_type", "destring", "tostring", "encode", "decode", "set_value_labels",
                "set_measure", "rename_var", "drop_vars", "clear_filters")
SETUP_METHODS = ("xtset", "tsset", "attach_boundary")
RESULT_METHODS = ("run", "compare_models", "sem_template", "predict")
ALLOWED = DATA_METHODS + SETUP_METHODS + RESULT_METHODS
# methods whose string arguments may name a variable that does not exist yet
_CREATES = ("compute", "recode", "egen", "encode", "decode", "rename_var", "reshape", "collapse", "merge", "joinby",
            "append", "convert_type", "destring", "tostring", "predict", "standardize")
_PANEL_KEYS = ("entity", "time")
_DEP_KEYS = ("vars", "dep", "deps")


class Call:
    def __init__(self, method: str, args: List[Any], kwargs: Dict[str, Any], line: int, target: Optional[str]):
        self.method, self.args, self.kwargs, self.line, self.target = method, args, kwargs, line, target


class _Ref:
    """A variable assigned earlier in the code (a result passed to compare_models)."""
    def __init__(self, name: str):
        self.name = name


def _strings(v: Any) -> List[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple, set)):
        return [s for x in v for s in _strings(x)]
    if isinstance(v, dict):
        return [s for kv in v.items() for x in kv for s in _strings(x)]
    return []


def parse(code: str) -> Tuple[List[Call], List[str]]:
    """Code -> calls, or the reasons it is not a flat list of sess.<method>(literal, ...) calls."""
    problems: List[str] = []
    calls: List[Call] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [], [f"파이썬 문법 오류 ({e.lineno}행): {e.msg}"]
    assigned: Set[str] = set()
    consts: Dict[str, Any] = {}                 # name = "literal" (e.g. a SEM model string written before the call)
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, expr = node.targets[0].id, node.value
            if not isinstance(expr, ast.Call) and target != "sess":
                try:
                    consts[target] = ast.literal_eval(expr)
                    continue
                except (ValueError, SyntaxError, TypeError):
                    pass
        elif isinstance(node, ast.Expr):
            expr = node.value
        else:
            problems.append(f"{node.lineno}행: sess.<메서드>(…) 호출과 `이름 = sess.run(…)` 대입만 쓸 수 있음 "
                            f"(import, for, if, def, print 등은 쓰지 않음)")
            continue
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            continue                                              # a stray docstring
        f = expr.func if isinstance(expr, ast.Call) else None
        if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "sess"):
            problems.append(f"{node.lineno}행: `sess.<메서드>(…)` 형식이 아님. 모든 명령은 sess 의 메서드 호출로 씀")
            continue
        if f.attr not in ALLOWED:
            hint = K.suggest(f.attr, ALLOWED)
            problems.append(f"{node.lineno}행: sess.{f.attr}() 는 쓸 수 없음. 분석은 sess.run('<분석 id>', …) 하나로 실행함"
                            + (f". 비슷한 이름: {', '.join(hint)}" if hint else ""))
            continue

        def value(n):
            if isinstance(n, ast.Name) and n.id in assigned:
                return _Ref(n.id)
            if isinstance(n, ast.Name) and n.id in consts:
                return consts[n.id]
            if isinstance(n, ast.Name) and n.id in ("true", "false", "null"):
                return {"true": True, "false": False, "null": None}[n.id]
            return ast.literal_eval(n)

        try:
            args = [value(a) for a in expr.args]
            kwargs = {}
            for kw in expr.keywords:
                if kw.arg is None:
                    raise ValueError("**kwargs")
                kwargs[kw.arg] = value(kw.value)
        except (ValueError, SyntaxError, TypeError):
            problems.append(f"{node.lineno}행: 인자는 값(문자열·숫자·True/False·None·목록·사전)이나 앞에서 대입한 결과 이름만 쓸 수 있음. "
                            f"변수·식·함수 호출을 인자에 넣지 않음")
            continue
        calls.append(Call(f.attr, args, kwargs, node.lineno, target))
        if target:
            assigned.add(target)
            consts.pop(target, None)
    if not calls and not problems:
        problems.append("실행할 명령이 없음")
    return calls, problems


def _check_run(c: Call, columns: Set[str], declared: Set[str]) -> Tuple[List[str], Optional[str]]:
    where = f"{c.line}행 sess.run"
    if len(c.args) != 1 or not isinstance(c.args[0], str):
        return [f"{where}: 첫 인자는 분석 id 문자열 하나, 나머지는 이름=값 형식으로 씀"], None
    raw = c.args[0]
    ids = K.analysis_ids()
    if raw not in ids:
        canon = K.aliases().get(raw)
        if canon:
            return [f"{where}: '{raw}' 대신 분석 id '{canon}' 를 씀 (옵션은 '{canon}' 카드의 이름으로 지정)"], canon
        hint = K.suggest(raw, ids)
        return [f"{where}: 분석 id '{raw}' 없음" + (f". 비슷한 id: {', '.join(hint)}" if hint else "")
                + ". 분석 목록의 id 를 그대로 씀"], None
    aid, out = raw, []
    allowed = K.allowed_keys(aid)
    if allowed is None:
        return out, aid
    slot_list = K.slots(aid)
    slot_keys = {s["key"] for s in slot_list}
    card_keys = sorted(slot_keys | {it["key"] for it in K.option_items(aid)})
    for k in c.kwargs:
        if k not in allowed:
            hint = K.suggest(k, card_keys)
            out.append(f"{where}('{aid}'): 옵션 '{k}' 없음" + (f" → 비슷한 이름: {', '.join(hint)}" if hint else "")
                       + f". 쓸 수 있는 이름: {', '.join(card_keys)}")
    ch = K.choices(aid)
    for it in K.option_items(aid):
        k = it["key"]
        if k not in c.kwargs:
            continue
        v = c.kwargs[k]
        if it.get("type") == "select" and ch.get(k) and v not in ch[k] and str(v) not in [str(x) for x in ch[k]]:
            out.append(f"{where}('{aid}'): {k}={v!r} 는 선택지에 없음 → {', '.join(repr(x) for x in ch[k])} 중 하나")
        elif it.get("type") == "check" and not isinstance(v, (bool, int)):
            out.append(f"{where}('{aid}'): {k} 는 True/False 로 지정")
        elif it.get("type") == "number" and v is not None and not isinstance(v, (int, float)):
            out.append(f"{where}('{aid}'): {k} 는 숫자로 지정")
        elif it.get("type") == "subset":
            out += _missing_cols(f"{where}('{aid}') {k}", v, columns)
    for s in slot_list:
        k = s["key"]
        if k in c.kwargs and c.kwargs[k] not in (None, "", []):
            out += _missing_cols(f"{where}('{aid}') {k}", c.kwargs[k], columns)
            if not s.get("multi") and isinstance(c.kwargs[k], (list, tuple)) and len(c.kwargs[k]) > 1:
                out.append(f"{where}('{aid}'): {k} 에는 변수 하나만 지정 ({k}='변수')")
        elif s.get("req") and not s.get("showIf") and not (k in _PANEL_KEYS and k in declared):
            alt = k in _DEP_KEYS and any(a in c.kwargs and a in allowed for a in _DEP_KEYS)   # dep / deps / vars are interchangeable
            if not alt:
                out.append(f"{where}('{aid}'): 필수 변수 역할 {k} 지정 필요 ({s.get('label', '')})")
    return out, aid


def _missing_cols(where: str, value: Any, columns: Set[str]) -> List[str]:
    names = [value] if isinstance(value, str) else list(value) if isinstance(value, (list, tuple)) else []
    out = []
    for n in names:
        if isinstance(n, (list, tuple)):
            out += _missing_cols(where, n, columns)
        elif isinstance(n, str) and n not in columns:
            hint = K.suggest(n, columns)
            out.append(f"{where}: 자료에 변수 '{n}' 없음" + (f" → 비슷한 변수: {', '.join(hint)}" if hint else "")
                       + ". 변수 이름은 [자료]에 적힌 그대로 씀")
    return out


def _check_method(c: Call) -> List[str]:
    from ..agent import JamoviSession
    fn = getattr(JamoviSession, c.method)
    try:
        inspect.signature(fn).bind(None, *c.args, **c.kwargs)
    except TypeError as e:
        sig = str(inspect.signature(fn)).replace("(self, ", "(").replace("(self)", "()")
        sig = sig.split(" -> ")[0]
        return [f"{c.line}행 sess.{c.method}: 인자가 맞지 않음 ({e}). 사용법: sess.{c.method}{sig[:260]}"]
    return []


def check(code: str, columns: List[str]) -> Dict[str, Any]:
    """Returns {"ok", "problems", "analyses" (ids used), "changes_data", "calls"}."""
    calls, problems = parse(code)
    cols: Set[str] = set(map(str, columns))
    declared: Set[str] = set()
    used: List[str] = []
    changes = False
    for c in calls:
        if c.method == "run":
            p, aid = _check_run(c, cols, declared)
            problems += p
            if aid and aid not in used:
                used.append(aid)
        else:
            problems += _check_method(c)
            if c.method in DATA_METHODS:
                changes = True
            if c.method == "xtset":
                declared.update(_PANEL_KEYS[: 1 + (len(c.args) + len(c.kwargs) > 1)])
            if c.method == "tsset":
                declared.add("time")
            if c.method == "compare_models" and "compareModels" not in used:
                used.append("compareModels")
            if c.method in _CREATES:
                cols.update(_strings(c.args) + _strings(c.kwargs))        # names a data step may create
    return {"ok": not problems, "problems": problems, "analyses": used, "changes_data": changes, "calls": calls}

"""Analysis extension modules: one place that collects catalog entries, dispatch, code conversion and Stata input rules.

Each extension module defines a module-level ``EXTENSION`` dict:

    EXTENSION = {
        "catalog":   [{"id": "countReg", "group": "Regression", "title": "...", "module": "openstat", "icon": "..."}],
        "aliases":   {"nbreg": "countReg"},                 # extra names accepted by sess.run / CLI / MCP
        "dispatch":  fn(t, df, params) -> BaseAnalysis | None,   # t is the canonical id after alias mapping
        "stata":     fn(t, params) -> list[str] | None,          # Stata (.do) lines for the code export
        "r":         fn(t, params) -> list[str] | None,          # R lines for the code export
        "transpile": fn(stata_line) -> str | None,               # Stata command typed by the user -> Python line
    }

Ids listed in an extension catalog take precedence over the built-in dispatch branches with the same id, so an
extension can replace an older implementation (for example ``panel`` or ``mixedModel``) without editing registry.py.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional

# module names under pyjamovi.analyses, loaded in this order
EXTENSION_MODULES = (
    "panel_module",
    "dynamic_panel",
    "iv_models",
    "time_series",
    "count_models",
    "limdep_models",
    "multilevel",
    "sem_templates",
)

_CACHE: Optional[List[Dict[str, Any]]] = None
_ERRORS: Dict[str, str] = {}


def _load() -> List[Dict[str, Any]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    out = []
    for name in EXTENSION_MODULES:
        try:
            mod = importlib.import_module(f"{__package__}.{name}")
        except ModuleNotFoundError as e:
            if e.name and e.name.endswith(name):
                continue            # module not present in this build
            _ERRORS[name] = f"{type(e).__name__}: {e}"
            continue
        except Exception as e:      # a broken extension must not take the whole program down
            _ERRORS[name] = f"{type(e).__name__}: {e}"
            continue
        ext = getattr(mod, "EXTENSION", None)
        if isinstance(ext, dict):
            out.append(dict(ext, _module=name))
    _CACHE = out
    return out


def reload() -> None:
    global _CACHE
    _CACHE = None
    _ERRORS.clear()


def load_errors() -> Dict[str, str]:
    _load()
    return dict(_ERRORS)


def catalog() -> List[Dict[str, Any]]:
    seen, out = set(), []
    for ext in _load():
        for item in ext.get("catalog") or []:
            if item.get("id") and item["id"] not in seen:
                seen.add(item["id"])
                out.append(dict(item))
    return out


def ids() -> set:
    return {item["id"] for item in catalog()}


def aliases() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ext in _load():
        out.update(ext.get("aliases") or {})
    return out


def _first(key: str, *args) -> Any:
    for ext in _load():
        fn: Optional[Callable] = ext.get(key)
        if fn is None:
            continue
        res = fn(*args)
        if res is not None:
            return res
    return None


def dispatch(t: str, df, params: Dict[str, Any], alias: Optional[str] = None):
    """Run extension analysis `t`. `alias` = the name the user typed when it differs from `t`; it is passed as
    params["_alias"] only to extensions that declare "uses_alias": True (alias -> option preset)."""
    for ext in _load():
        fn: Optional[Callable] = ext.get("dispatch")
        if fn is None:
            continue
        q = dict(params, _alias=alias) if (alias and ext.get("uses_alias")) else params
        res = fn(t, df, q)
        if res is not None:
            return res
    return None


def stata_code(t: str, params: Dict[str, Any]) -> Optional[List[str]]:
    return _first("stata", t, params)


def r_code(t: str, params: Dict[str, Any]) -> Optional[List[str]]:
    return _first("r", t, params)


def transpile(line: str) -> Optional[str]:
    return _first("transpile", line)

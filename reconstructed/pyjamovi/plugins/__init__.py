"""
OpenStat Studio Dynamic Clean-Room Plugin Loader (`reconstructed/pyjamovi/plugins/`)
License: MIT
Any `.py` module placed in this folder exposing `PLUGIN_SPEC` and `run_plugin(df, params)`
is automatically registered into `ANALYSIS_CATALOG`, Python SDK, CLI, MCP, and the Web UI upon load or rebuild.
"""

import importlib.util
import os
from typing import Dict, Any, List

PLUGINS_DIR = os.path.dirname(os.path.abspath(__file__))


def discover_plugins() -> Dict[str, Any]:
    """Discovers and loads all clean-room statistical plugins in `pyjamovi/plugins/*.py`."""
    loaded: Dict[str, Any] = {}
    if not os.path.isdir(PLUGINS_DIR):
        return loaded

    for fname in sorted(os.listdir(PLUGINS_DIR)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        fpath = os.path.join(PLUGINS_DIR, fname)
        mod_name = f"pyjamovi.plugins.{fname[:-3]}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, fpath)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                p_spec = getattr(mod, "PLUGIN_SPEC", None)
                runner = getattr(mod, "run_plugin", None)
                if isinstance(p_spec, dict) and "id" in p_spec and callable(runner):
                    loaded[p_spec["id"]] = {
                        "spec": p_spec,
                        "runner": runner,
                        "path": fpath,
                    }
        except Exception:
            continue
    return loaded


def get_plugin_catalog_entries() -> List[Dict[str, Any]]:
    plugins = discover_plugins()
    entries = []
    for pid, info in plugins.items():
        s = info["spec"]
        entries.append({
            "id": pid,
            "group": s.get("group", "Plugins"),
            "title": s.get("title", pid),
            "module": s.get("module", "openstat-plugin"),
            "icon": s.get("icon", "brand-modules.svg"),
            "ui_schema": s.get("ui_schema"),
            "is_plugin": True,
        })
    return entries

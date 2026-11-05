"""
OpenStat Studio — Bidirectional Python (.py) Syntax Generator, Live Script Runner & Claude Code Workspace Sync
License: MIT
Enables:
1. Real-time generation of clean, executable .py scripts from UI interactions or Claude Code tool calls.
2. Direct execution of user-edited or Claude-authored .py syntax to produce APA tables and customizable plots.
3. Persistent workspace synchronization (workspace/live_session.py & workspace/live_session.json)
   so Claude Code can operate offline/serverless and still sync seamlessly with the UI when opened.
"""

import contextlib
import io
import json
import os
import pprint
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from .analyses.base import BaseAnalysis
from .registry import dispatch_analysis


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_DIR = os.path.join(ROOT_DIR, "workspace")
LIVE_PY_PATH = os.path.join(WORKSPACE_DIR, "live_session.py")
LIVE_JSON_PATH = os.path.join(WORKSPACE_DIR, "live_session.json")


def _clean_params_for_code(params: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for k, v in (params or {}).items():
        if v is None or v == "" or v == []:
            continue
        cleaned[k] = v
    return cleaned


def _portable_path(p: Any) -> Any:
    """Converts absolute paths inside the project root into project-relative paths for portable scripts."""
    if isinstance(p, (list, tuple)):
        return [_portable_path(x) for x in p]
    if not isinstance(p, str) or not os.path.isabs(p):
        return p
    try:
        rel = os.path.relpath(p, ROOT_DIR)
        if not rel.startswith(".."):
            return rel.replace(os.sep, "/")
    except ValueError:
        pass
    return p


_STATA_EQUIV = {
    "compute": "generate / replace",
    "recode": "recode ..., generate()",
    "filter_rows": "keep if",
    "merge": "merge / append using",
    "reshape": "reshape long / wide",
    "rename_var": "rename",
    "drop_vars": "drop",
    "set_value": "replace ... in #",
    "add_rows": "set obs",
    "drop_rows": "drop in",
    "export_results": "esttab / putdocx / putexcel",
    "save_plots": "graph export",
    "save_data": "save",
}

_DATA_OP_ARG_ORDER = {
    "compute": ["col_name", "formula"],
    "recode": ["source_col", "target_col", "rules", "else_value"],
    "filter_rows": ["condition"],
    "merge": ["other_source", "on", "kind", "keep", "keepusing", "indicator", "update", "replace", "assert_results", "how", "suffixes"],
    "joinby": ["other_source", "on", "unmatched", "indicator", "update", "replace"],
    "append": ["other_source", "generate", "force"],
    "reshape": ["direction", "id_vars", "value_vars", "stubnames", "time_col", "value_col", "sep", "strip_prefix"],
    "collapse": ["stats", "by"],
    "egen": ["target", "func", "source", "by"],
    "convert_type": ["columns", "to", "ignore", "force", "percent", "generate", "date_format", "text_format"],
    "encode": ["source", "target", "order"],
    "decode": ["source", "target"],
    "set_value_labels": ["column", "labels"],
    "set_measure": ["columns", "level"],
    "standardize": ["columns", "method"],
    "rename_var": ["old_name", "new_name"],
    "drop_vars": ["cols"],
    "set_value": ["row_idx", "col_name", "value"],
    "add_rows": ["n"],
    "drop_rows": ["rows"],
    "export_results": ["output_path", "fmt", "report_style", "report_options"],
    "save_plots": ["out_dir", "fmt"],
    "save_data": ["output_path"],
}

_DATA_OP_DEFAULTS = {
    "recode": {"else_value": "copy"},
    "reshape": {"time_col": "time", "value_col": "value", "sep": "", "strip_prefix": True},
    "save_plots": {"fmt": "svg"},
}


def _format_op_call(op: Dict[str, Any]) -> str:
    name = op.get("op")
    order = _DATA_OP_ARG_ORDER.get(name, [k for k in op if k != "op"])
    defaults = _DATA_OP_DEFAULTS.get(name, {})
    parts = []
    for k in order:
        if k not in op:
            continue
        v = op[k]
        if (v is None and not (name == "set_value_labels" and k == "labels")) or v == [] or (k in defaults and v == defaults[k]):
            continue
        if k in ("other_source", "output_path", "out_dir"):
            v = _portable_path(v)
        parts.append(f"{k}={v!r}")
    one_line = f"sess.{name}(" + ", ".join(parts) + ")"
    if len(one_line) <= 100:
        return one_line
    return f"sess.{name}(\n" + "".join(f"    {p},\n" for p in parts) + ")"


def generate_python_script(
    specs: List[Dict[str, Any]],
    dataset_path: str = "sample_data/ToothGrowth.csv",
    export_hint: bool = True,
    data_ops: Optional[List[Dict[str, Any]]] = None,
    export_ops: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Generates a Stata `.do`-style, PEP8-formatted, runnable standalone `.py` command script:
    [STEP 1] load raw data -> [STEP 2] data management (merge/compute/recode/reshape/filter)
    -> [STEP 3] analyses, post-hoc tests & graphs -> [STEP 4] exports.
    Re-executing the script reproduces the entire pipeline from the raw file.
    """
    data_ops = list(data_ops or [])
    export_ops = list(export_ops or [])
    lines = [
        '"""',
        "OpenStat Studio — Reproducible Analysis Command Script (.py, Stata .do-file equivalent)",
        "Engine: openstat / pyjamovi v3.0 (100% Pure Python, MIT License)",
        "Analytical models, menu taxonomy & .omv format modeled with reference to jamovi (https://www.jamovi.org).",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Pipeline: {len(data_ops)} data step(s) | {len(specs or [])} analysis/graph step(s) | {len(export_ops)} export step(s)",
        "Replay  : `openstat exec-py --script <this_file.py>`  or  `python <this_file.py>`",
        '"""',
        "",
        "# ==========================================================================",
        "# Universal OpenStat Environment Loader (Standalone, Agent & Cloud/Colab)",
        "# ==========================================================================",
        "import os, sys",
        "",
        "try:",
        "    import openstat as ost",
        "except ImportError:",
        "    # Dynamic multi-tier search across local, parent and reconstructed/ paths",
        "    _curr = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()",
        "    _search_dirs = [",
        "        os.path.join(_curr, 'reconstructed'),",
        "        os.path.join(_curr, '..', 'reconstructed'),",
        "        os.path.join(_curr, '..', '..', 'reconstructed'),",
        "        os.path.join(os.getcwd(), 'reconstructed'),",
        "        os.path.join(os.getcwd(), '..', 'reconstructed'),",
        "    ]",
        "    for _d in _search_dirs:",
        "        _d = os.path.abspath(_d)",
        "        if os.path.isdir(_d) and _d not in sys.path:",
        "            sys.path.insert(0, _d)",
        "            break",
        "    try:",
        "        import openstat as ost",
        "    except ImportError:",
        "        try:",
        "            import pyjamovi as ost",
        "        except ImportError:",
        "            import subprocess",
        "            print('[OpenStat] Package not found in current path; bootstrapping via pip...')",
        "            subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', os.path.join(_curr, '..'), '--quiet'], check=False)",
        "            import openstat as ost",
        "",
        "def _resolve_data_file(p: str) -> str:",
        "    if os.path.isabs(p) and os.path.exists(p): return p",
        "    if os.path.exists(p): return p",
        "    if '__file__' in locals():",
        "        sdir = os.path.dirname(os.path.abspath(__file__))",
        "        for cand in [os.path.join(sdir, p), os.path.join(sdir, '..', p), os.path.join(sdir, '..', '..', p)]:",
        "            if os.path.exists(cand): return os.path.abspath(cand)",
        "    return p",
        "",
        "# ==========================================================================",
        "# [STEP 1] Load raw dataset (.csv, .xlsx, .omv)            — Stata: use \"...\", clear",
        "# ==========================================================================",
        f"sess = ost.Session(_resolve_data_file({_portable_path(dataset_path)!r}))",
        "",
    ]

    if data_ops:
        lines.extend([
            "# ==========================================================================",
            f"# [STEP 2] Data management ({len(data_ops)} step(s), executed in recorded order)",
            "# ==========================================================================",
        ])
        for i, op in enumerate(data_ops, start=1):
            lines.append(f"# (2.{i}) {op.get('op')}  — Stata: {_STATA_EQUIV.get(op.get('op'), '')}")
            lines.append(_format_op_call(op))
        lines.append("")
    lines.append("df = sess.df  # active (transformed) dataset")
    lines.append("print(f'Active dataset: {df.shape[0]} rows x {df.shape[1]} columns')")
    lines.append("")

    lines.extend([
        "# ==========================================================================",
        "# [STEP 3] Statistical analyses, post-hoc tests & graphs (-> Results Sheet)",
        "# ==========================================================================",
    ])
    if not specs:
        lines.extend([
            "res = sess.run(",
            "    'descriptives',",
            "    vars=list(df.columns[:2]),",
            "    violin=True,",
            "    plotStyle={'palette': 'openstat', 'theme': 'clean', 'dpi': 150},",
            ")",
            "print(res.to_markdown())",
            "",
        ])
    else:
        for idx, sp in enumerate(specs, start=1):
            a_type = sp.get("analysis_type") or sp.get("analysisType") or sp.get("name") or "descriptives"
            title = sp.get("title") or a_type
            p = _clean_params_for_code(sp.get("params") or sp.get("options") or {})

            lines.append(f"# (3.{idx}) {title}")
            if a_type == "customFigure":
                lines.append("# Custom Matplotlib figure exported to Analysis Results:")
                lines.append("# sess.add_figure(fig, title=" + repr(p.get("title", title)) + ")")
                lines.append("")
                continue
            lines.append(f"res_{idx} = sess.run(")
            lines.append(f"    {a_type!r},")
            for pk, pv in p.items():
                formatted_v = pprint.pformat(pv, width=88, sort_dicts=False)
                lines.append(f"    {pk}={formatted_v},")
            lines.append(")")
            lines.append(f"print(res_{idx}.to_markdown())")
            lines.append("")

    if export_ops:
        lines.extend([
            "# ==========================================================================",
            f"# [STEP 4] Export results, graphs & data ({len(export_ops)} step(s))",
            "# ==========================================================================",
        ])
        for i, op in enumerate(export_ops, start=1):
            lines.append(f"# (4.{i}) {op.get('op')}  — Stata: {_STATA_EQUIV.get(op.get('op'), '')}")
            lines.append(_format_op_call(op))
        lines.append("")
    elif export_hint:
        lines.extend([
            "# [STEP 4] Optional: Export full Analysis Results (tables + embedded graphs) & vector figures",
            "# sess.export_results('reports/analysis_results.pdf')  # .pdf, .hwpx, .docx, .xlsx, .tex, .csv, .md, .html, .omv",
            "# sess.export_results('reports/analysis_results.hwpx', report_style='social')  # 별표·괄호 표준오차·<표 n> 번호",
            "# sess.save_plots('reports/plots', fmt='svg')          # .svg, .pdf, .png",
            "# sess.save_data('reports/analysis_data.csv')          # transformed dataset (.csv, .xlsx, .omv)",
            "",
        ])

    return "\n".join(lines)


class _RecordingProxy:
    """Wraps openstat / pyjamovi module and Session during live .py script execution to capture all BaseAnalysis results."""
    def __init__(
        self,
        base_module,
        default_df: Optional[pd.DataFrame],
        default_path: Optional[str],
        collected: List[BaseAnalysis],
        collected_specs: List[Dict[str, Any]],
    ):
        self._base = base_module
        self._default_df = default_df
        self._default_path = default_path
        self._collected = collected
        self._collected_specs = collected_specs
        self.last_df = default_df
        self.last_dataset_name = os.path.basename(default_path) if default_path else "Dataset.csv"
        self.sessions: List[Any] = []

    def Session(self, data_source=None, auto_sync_workspace: bool = False):
        from .agent import JamoviSession
        if data_source is None and self._default_df is not None:
            sess = JamoviSession(self._default_df.copy(), auto_sync_workspace=False)
            sess.source_path = self._default_path
        elif isinstance(data_source, str) and not os.path.exists(data_source):
            cand1 = os.path.join(ROOT_DIR, data_source)
            cand2 = os.path.join(ROOT_DIR, "sample_data", os.path.basename(data_source))
            target = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else data_source)
            sess = JamoviSession(target, auto_sync_workspace=False)
        else:
            sess = JamoviSession(data_source, auto_sync_workspace=False)
        self.sessions.append(sess)

        self.last_df = sess.df
        if sess.source_path:
            self.last_dataset_name = os.path.basename(sess.source_path)

        orig_run = sess.run
        orig_add_fig = sess.add_figure

        def _hooked_run(analysis_type: str, **params):
            res = orig_run(analysis_type, **params)
            if isinstance(res, BaseAnalysis) and res not in self._collected:
                self._collected.append(res)
                self._collected_specs.append({
                    "analysis_type": analysis_type,
                    "title": res.title,
                    "params": params
                })
            return res

        def _hooked_add_figure(fig_or_src, title: str = "Custom Visualization", caption: str = "", summary_df=None, attach_to_last: bool = False, plot_style=None):
            res = orig_add_fig(
                fig_or_src,
                title=title,
                caption=caption,
                summary_df=summary_df,
                attach_to_last=attach_to_last,
                plot_style=plot_style,
            )
            if isinstance(res, BaseAnalysis) and res not in self._collected:
                self._collected.append(res)
                self._collected_specs.append({
                    "analysis_type": res.name,
                    "title": res.title,
                    "params": res.options,
                })
            return res

        sess.run = _hooked_run
        sess.add_figure = _hooked_add_figure
        return sess

    def __getattr__(self, name: str):
        attr = getattr(self._base, name)
        if callable(attr):
            def _wrapped(*args, **kwargs):
                out = attr(*args, **kwargs)
                if isinstance(out, BaseAnalysis) and out not in self._collected:
                    self._collected.append(out)
                    self._collected_specs.append({
                        "analysis_type": out.name,
                        "title": out.title,
                        "params": out.options
                    })
                return out
            return _wrapped
        return attr


def execute_python_syntax(
    code: str,
    default_df: Optional[pd.DataFrame] = None,
    default_dataset_path: Optional[str] = None,
    sync_workspace: bool = True,
    return_meta: bool = False,
    session_attrs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[BaseAnalysis], str, Optional[pd.DataFrame], str]:
    """
    Executes user- or Claude-written Python (.py) statistical syntax and captures:
    - All BaseAnalysis objects produced (via ost.*, jmv.*, sess.*, or standalone variables)
    - Printed stdout log
    - Active DataFrame & dataset name
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pyjamovi as _jmv_mod
    from .analyses.graph_builder import CustomFigureResult

    plt.close("all")

    collected: List[BaseAnalysis] = []
    collected_specs: List[Dict[str, Any]] = []
    proxy = _RecordingProxy(_jmv_mod, default_df, default_dataset_path, collected, collected_specs)

    active_sess = proxy.Session(default_df if default_df is not None else default_dataset_path) if (default_df is not None or default_dataset_path) else None
    if active_sess is not None and session_attrs:
        for _k, _v in session_attrs.items():
            if _v is not None:
                setattr(active_sess, _k, _v)

    def _push_figure(fig=None, title: str = "Custom Visualization", caption: str = "", summary_df=None, attach_to_last: bool = False):
        target_fig = fig if fig is not None else plt.gcf()
        if active_sess is not None:
            return active_sess.add_figure(target_fig, title=title, caption=caption, summary_df=summary_df, attach_to_last=attach_to_last)
        card = CustomFigureResult(target_fig, title=title, caption=caption, summary_df=summary_df)
        collected.append(card)
        collected_specs.append({"analysis_type": "customFigure", "title": card.title, "params": card.options})
        return card

    import builtins as _builtins
    _real_import = _builtins.__import__

    def _recording_import(name, globals=None, locals=None, fromlist=(), level=0):
        # `import openstat as ost` / `from pyjamovi import Session` inside a script must resolve to the recording proxy
        if level == 0 and name in ("openstat", "pyjamovi"):
            return proxy
        return _real_import(name, globals, locals, fromlist, level)

    sandbox_builtins = dict(vars(_builtins))
    sandbox_builtins["__import__"] = _recording_import

    exec_globals: Dict[str, Any] = {
        "__name__": "__openstat_syntax__",
        "__builtins__": sandbox_builtins,
        "ost": proxy,
        "openstat": proxy,
        "jmv": proxy,
        "pyjamovi": proxy,
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "push_figure": _push_figure,
        "df": default_df.copy() if default_df is not None else None,
        "sess": active_sess,
        "session": active_sess,
        "Session": proxy.Session,
        "JamoviSession": proxy.Session,
        "true": True,
        "false": False,
        "null": None,
    }

    stdout_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf):
        exec(compile(code, "<openstat_live_script.py>", "exec"), exec_globals)

    for val in list(exec_globals.values()):
        if isinstance(val, BaseAnalysis) and val not in collected:
            collected.append(val)
            collected_specs.append({
                "analysis_type": val.name,
                "title": val.title,
                "params": val.options
            })
        elif hasattr(val, "history") and isinstance(val.history, list):
            for a in val.history:
                if isinstance(a, BaseAnalysis) and a not in collected:
                    collected.append(a)
                    collected_specs.append({
                        "analysis_type": a.name,
                        "title": a.title,
                        "params": a.options
                    })

    # Auto-capture any unclosed user-drawn Matplotlib figures directly into Analysis Results
    open_fignums = list(plt.get_fignums())
    for idx_f, fnum in enumerate(open_fignums, start=1):
        fig = plt.figure(fnum)
        if not fig.axes:
            plt.close(fig)
            continue
        ax_title = ""
        for ax in fig.axes:
            t_cand = ax.get_title()
            if t_cand:
                ax_title = t_cand
                break
        fig_title = ax_title or (fig._suptitle.get_text() if getattr(fig, "_suptitle", None) else "") or f"Custom Python Figure {idx_f}"
        card = CustomFigureResult(
            fig,
            title=fig_title,
            caption=f"Auto-captured Matplotlib figure #{idx_f} from live Python syntax.",
            dataset=proxy.last_df,
        )
        collected.append(card)
        collected_specs.append({
            "analysis_type": "customFigure",
            "title": card.title,
            "params": card.options,
        })

    final_df = proxy.last_df if proxy.last_df is not None else exec_globals.get("df")
    data_ops_out: List[Dict[str, Any]] = []
    export_ops_out: List[Dict[str, Any]] = []
    source_path_out: Optional[str] = default_dataset_path
    final_sess = exec_globals.get("sess")
    if not (final_sess is not None and any(final_sess is s for s in proxy.sessions)):
        final_sess = proxy.sessions[-1] if proxy.sessions else None
    if final_sess is None:
        from .agent import JamoviSession as _JS
        cands = [v for v in exec_globals.values() if isinstance(v, _JS)]
        final_sess = cands[-1] if cands else None
    if final_sess is not None and getattr(final_sess, "df", None) is not None:
        final_df = final_sess.df
        data_ops_out = list(getattr(final_sess, "data_ops", []) or [])
        export_ops_out = list(getattr(final_sess, "export_ops", []) or [])
        if getattr(final_sess, "source_path", None):
            source_path_out = final_sess.source_path
            proxy.last_dataset_name = os.path.basename(final_sess.source_path)

    if sync_workspace and (collected or data_ops_out):
        save_workspace_sync(
            dataset_name=proxy.last_dataset_name,
            specs=collected_specs,
            analyses=collected,
            py_code=code,
            data_ops=data_ops_out,
            export_ops=export_ops_out,
        )
    if return_meta:
        meta = {
            "data_ops": data_ops_out,
            "export_ops": export_ops_out,
            "specs": collected_specs,
            "source_path": source_path_out,
        }
        return collected, stdout_buf.getvalue(), final_df, proxy.last_dataset_name, meta
    return collected, stdout_buf.getvalue(), final_df, proxy.last_dataset_name


def run_script(script_path: str, sync_workspace: bool = False) -> Dict[str, Any]:
    """
    Replays a recorded `.py` command script (Stata `do file.do` equivalent) from scratch and returns:
    `{'analyses', 'stdout', 'df', 'dataset', 'data_ops', 'export_ops', 'specs'}`.
    """
    with open(script_path, "r", encoding="utf-8") as f:
        code = f.read()
    analyses, out, df, ds, meta = execute_python_syntax(code, sync_workspace=sync_workspace, return_meta=True)
    return {"analyses": analyses, "stdout": out, "df": df, "dataset": ds, **meta}


def save_workspace_sync(
    dataset_name: str,
    specs: List[Dict[str, Any]],
    analyses: List[BaseAnalysis],
    py_code: Optional[str] = None,
    data_ops: Optional[List[Dict[str, Any]]] = None,
    export_ops: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, str]:
    """Writes live_session.py and live_session.json to workspace/ so Claude Code and UI stay in sync."""
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    script_text = py_code if py_code else generate_python_script(
        specs, dataset_path=f"sample_data/{dataset_name}", data_ops=data_ops, export_ops=export_ops
    )

    with open(LIVE_PY_PATH, "w", encoding="utf-8") as f:
        f.write(script_text)

    def _jsonable(obj):
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return json.loads(json.dumps(obj, default=str))

    state_doc = {
        "updatedAt": datetime.now().isoformat(),
        "dataset": dataset_name,
        "pythonScriptPath": LIVE_PY_PATH,
        "pythonCode": script_text,
        "analysesCount": len(analyses),
        "dataOps": _jsonable(data_ops or []),
        "exportOps": _jsonable(export_ops or []),
        "specs": _jsonable(specs),
        "summaries": [
            {
                "name": a.name,
                "title": a.title,
                "apa": a.apa_narrative,
                "tables": [t.title for t in a.tables],
                "plots": [p["title"] for p in a.plots]
            }
            for a in analyses
        ]
    }
    with open(LIVE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(state_doc, f, indent=2, ensure_ascii=False)

    return LIVE_PY_PATH, LIVE_JSON_PATH


def load_workspace_sync() -> Optional[Dict[str, Any]]:
    """Reads workspace/live_session.json if present."""
    if not os.path.isfile(LIVE_JSON_PATH):
        return None
    try:
        with open(LIVE_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

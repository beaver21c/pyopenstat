"""
OpenStat Studio Model Context Protocol (MCP) Server (`openstat-mcp` / `pyjamovi-mcp` v3.0)
License: MIT
Exposes all 28 statistical analyses, real-time `.py` syntax authoring & execution,
customizable `PlotStyle` graph design, and 10-format table exports (`.xlsx`, `.docx`, `.hwpx`, `.tex`, `.csv`, `.md`, `.html`, `.omv`)
directly to Claude Code, Claude Desktop, Codex, and Antigravity.
"""

import sys
import json
import os
from typing import Dict, Any, List
import pandas as pd

from .agent import JamoviSession
from .registry import ANALYSIS_CATALOG, dispatch_analysis
from .analyses.plot_style import PALETTES, THEMES
from .analyses.graph_builder import CHART_TYPES
from .exporters import SUPPORTED_TABLE_FORMATS, export_analyses
from .syntax_engine import execute_python_syntax, load_workspace_sync
from .omv.reader import read_omv
from .omv.writer import write_omv

ALL_ANALYSIS_IDS = sorted(list({item["id"] for item in ANALYSIS_CATALOG} | set(CHART_TYPES) | {
    "ttest", "anova", "corrmatrix", "regression", "crosstab", "logistic"
}))

PLOT_STYLE_SCHEMA = {
    "type": "object",
    "description": "Graph design options: palette ('openstat','apa_bw','nature','lancet','okabe_ito','kihasa','viridis','warm','pastel','classic_blue'), custom_colors (['#1E3A5F','#0D9488']), theme ('clean','apa_classic','whitegrid','minimal','dark'), title, xlabel, ylabel, width, height, dpi (72-600), font_scale, show_grid, legend_pos.",
    "properties": {
        "palette": {"type": "string", "enum": list(PALETTES.keys())},
        "custom_colors": {"type": "array", "items": {"type": "string"}},
        "theme": {"type": "string", "enum": list(THEMES)},
        "title": {"type": "string"},
        "xlabel": {"type": "string"},
        "ylabel": {"type": "string"},
        "width": {"type": "number"},
        "height": {"type": "number"},
        "dpi": {"type": "integer"},
        "font_scale": {"type": "number"},
        "show_grid": {"type": "boolean"},
        "legend_pos": {"type": "string"},
    }
}

RUN_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "Path to dataset file (.csv, .omv, .xlsx)"},
        "analysis_type": {
            "type": "string",
            "enum": ALL_ANALYSIS_IDS,
            "description": "Statistical analysis ID (e.g. ttestIS, anovaOneW, anovaFactorial, anovaRM, bayesTTestIS, bayesANOVA, linReg, logRegBin, reliability, efa, sem, mediation, moderation, panel, mixedModel, survival, scatr)"
        },
        "parameters": {
            "type": "object",
            "description": "Analysis options dictionary (e.g. {'dep': 'len', 'group': 'dose', 'phMethod': 'gamesHowell'} or {'model': 'F1 =~ x1 + x2 + x3'} or {'entity': 'firm', 'time': 'year', 'dep': 'invest', 'covs': ['value', 'capital']})"
        },
        "plot_style": PLOT_STYLE_SCHEMA,
        "out_py_path": {"type": "string", "description": "Optional path to save reproducible standalone Python (.py) script"},
        "export_table_paths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional list of output file paths to export analyzed tables (.xlsx, .docx, .hwpx, .tex, .csv, .tsv, .md, .html, .json, .omv)"
        },
        "export_plot_path": {"type": "string", "description": "Optional path to export figure (.svg, .pdf, or .png)"},
        "save_omv_path": {"type": "string", "description": "Optional output .omv path"},
        "save_html_path": {"type": "string", "description": "Optional output standalone .html report path"},
        "push_to_live_ui": {"type": "boolean", "description": "If true and UI is open, pushes result straight into OpenStat Studio Web UI"}
    },
    "required": ["file_path", "analysis_type"]
}

TOOLS = [
    {
        "name": "openstat_list_analyses",
        "description": "List all 28 statistical analyses in OpenStat Studio (T-Tests, ANOVA, Post-Hoc, Bayesian, Regression, Logistic, Factor/PCA/Reliability, CFA/SEM, Mediation/Moderation, Panel FE/RE/Hausman, Mixed Models, Survival, Plots), plus available PlotStyle palettes and 10 table export formats.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "openstat_inspect_dataset",
        "description": "Inspect a dataset (.omv, .csv, .xlsx). Returns variable names, measurement scales (Continuous/Ordinal/Nominal), summary stats, missing counts, and recommended OpenStat analyses.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to dataset (.omv, .csv, or .xlsx)"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "openstat_wrangle_data",
        "description": "Perform Stata-equivalent data cleaning and transformation: compute (new variable formula), recode (value mapping/intervals), filter (query rows), merge (Stata 1:1/m:1/1:m/m:m with _merge), joinby, append, reshape (wide <-> long), collapse, egen, convert_type (destring/tostring/date), encode/decode, value labels, measure levels, or standardize (z-score/minmax/robust). Automatically saves changes and logs executable syntax.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Dataset path (.csv, .xlsx, .omv, .dta, .sav, .parquet)"},
                "operation": {
                    "type": "string",
                    "enum": ["compute", "recode", "filter", "merge", "joinby", "append", "reshape", "collapse", "egen",
                             "convert_type", "encode", "decode", "set_value_labels", "set_measure", "standardize"],
                    "description": "Data wrangling operation"
                },
                "params": {
                    "type": "object",
                    "description": "Parameters for the operation: compute {'target': 'bmi', 'expr': 'weight / (height/100)**2'}; recode {'source': 'age', 'target': 'age_grp', 'rules': [{'type': 'range', 'from': 0, 'to': 19, 'value': 1, 'label': '청소년'}]}; filter {'query': 'age >= 20 and sex == 1'}; merge {'right_file': '...', 'on': ['id'], 'kind': '1:1|m:1|1:m|m:m (omit = infer from key uniqueness)', 'keep': ['master','using','match'], 'keepusing': [...], 'update': false, 'replace': false} (Stata merge; adds _merge 1-5); joinby {'right_file': '...', 'on': ['id'], 'unmatched': 'none|both|master|using'}; append {'right_file': '...' or [...], 'generate': 'src', 'force': false}; reshape {'direction': 'long', 'id_vars': ['id'], 'stubnames': ['bp_w'], 'time_col': 'wave'} or {'direction': 'wide', 'id_vars': ['id'], 'value_vars': ['y'], 'time_col': 't'}; collapse {'stats': {'mean': ['y']}, 'by': ['id']}; egen {'target': 'm', 'func': 'mean', 'source': 'y', 'by': ['id']}; convert_type {'columns': ['x'], 'to': 'numeric|integer|text|date', 'ignore': ',', 'force': false}; encode/decode {'source': 's', 'target': 's_code'}; set_value_labels {'column': 'sex', 'labels': {'1': '남', '2': '여'}}; set_measure {'columns': ['x'], 'level': 'Ordinal'}; standardize {'columns': ['x1', 'x2'], 'method': 'zscore'}"
                },
                "save_path": {"type": "string", "description": "Optional output path to save wrangled dataset (.csv, .xlsx, .omv)"}
            },
            "required": ["file_path", "operation", "params"]
        }
    },
    {
        "name": "openstat_run_analysis",
        "description": "Execute any statistical analysis in Python, automatically writing real-time `.py` syntax (`workspace/live_session.py`), returning APA tables & narrative, customizing plots (`plot_style`), and optionally exporting Analysis Results (tables + graphs) to `.pdf`, `.hwpx`, `.docx`, `.xlsx`, `.tex`, `.csv`, `.md`, `.html`, or `.omv`.",
        "inputSchema": RUN_INPUT_SCHEMA
    },
    {
        "name": "openstat_create_graph",
        "description": "Draw any of 15 standalone publication chart types (`bar`, `stacked_bar`, `line`, `area`, `pie`, `donut`, `histogram`, `density`, `box`, `violin`, `raincloud`, `scatter`, `pairplot`, `heatmap`, `errorbar`), push directly to the Analysis Results Sheet, and optionally export to `.svg`/`.pdf`/`.png` or `.pdf`/`.hwpx`/`.docx`/`.xlsx` reports.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to dataset file (.csv, .omv, .xlsx)"},
                "chart_type": {"type": "string", "enum": list(CHART_TYPES), "default": "bar"},
                "x": {"type": "string", "description": "X-axis / category variable"},
                "y": {"type": "string", "description": "Y-axis / numeric variable"},
                "vars": {"type": "array", "items": {"type": "string"}, "description": "Multi-variable list for pairplot / heatmap"},
                "group": {"type": "string", "description": "Color / split grouping variable"},
                "size_var": {"type": "string", "description": "Bubble size variable (scatter)"},
                "agg": {"type": "string", "enum": ["mean", "median", "sum", "count", "pct"], "default": "mean"},
                "error_bar": {"type": "string", "enum": ["ci95", "se", "sd", "none"], "default": "ci95"},
                "horizontal": {"type": "boolean", "default": False},
                "stacked_100": {"type": "boolean", "default": False},
                "show_values": {"type": "boolean", "default": False},
                "jitter": {"type": "boolean", "default": False},
                "fit_line": {"type": "string", "enum": ["linear", "quadratic", "smooth", "none"], "default": "linear"},
                "bins": {"type": "integer", "default": 18},
                "show_summary_table": {"type": "boolean", "default": True},
                "caption": {"type": "string"},
                "plot_style": PLOT_STYLE_SCHEMA,
                "export_plot_path": {"type": "string", "description": "Optional path to save chart (.svg, .pdf, or .png)"},
                "export_results_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional paths to export Analysis Results report (.pdf, .hwpx, .docx, .xlsx, .html, .tex, .md)"
                },
                "push_to_live_ui": {"type": "boolean", "default": False}
            },
            "required": ["file_path", "chart_type"]
        }
    },
    {
        "name": "openstat_export_results",
        "description": "Export all active Analysis Results (statistical APA tables + standalone/analysis graphs) in the current workspace session to `.pdf`, `.hwpx`, `.docx`, `.xlsx`, `.tex`, `.html`, `.md`, `.csv`, `.tsv`, `.json`, or `.omv`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "output_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Target output file paths (e.g. ['reports/results.pdf', 'reports/results.hwpx', 'reports/results.docx'])"
                }
            },
            "required": ["output_paths"]
        }
    },
    {
        "name": "openstat_exec_python_syntax",
        "description": "Directly execute Python (`.py`) statistical syntax authored or edited by Claude Code (`import openstat as ost; sess = ost.Session(...); sess.panel(...)`), sync with workspace, and optionally export analyzed tables and `.svg`/`.pdf`/`.png` plots.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python statistical code string using `ost` / `sess`"},
                "script_path": {"type": "string", "description": "Alternative path to an existing `.py` script file to execute"},
                "default_data_path": {"type": "string", "description": "Optional dataset path pre-bound as `sess` and `df`"},
                "save_py_path": {"type": "string", "description": "Optional path to save the executed `.py` code"},
                "export_table_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional output paths for analyzed tables (e.g. ['reports/tables.xlsx', 'reports/tables.hwpx', 'reports/tables.docx'])"
                },
                "export_plot_dir": {"type": "string", "description": "Optional directory to export all generated figures"},
                "plot_format": {"type": "string", "enum": ["svg", "pdf", "png"], "default": "svg"}
            }
        }
    },
    {
        "name": "openstat_convert_dataset",
        "description": "Convert datasets losslessly between .omv and .csv/.xlsx formats.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string"},
                "output_path": {"type": "string"}
            },
            "required": ["input_path", "output_path"]
        }
    },
    {
        "name": "openstat_doctor",
        "description": "Check installed Python package freshness against PyPI and run syntax/API drift self-diagnostics across all statistical and export subsystems.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "check_pypi": {"type": "boolean", "default": True, "description": "Query PyPI for latest upstream versions"}
            }
        }
    },
    {
        "name": "openstat_sync_jamovi",
        "description": "Check upstream jamovi for newly added analyses, optionally scaffold a clean-room MIT Python plugin (`scaffold_id`), and optionally rebuild OpenStat Studio (`rebuild`).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "fetch_remote": {"type": "boolean", "default": True},
                "scaffold_id": {"type": "string", "description": "Optional new analysis ID to scaffold into plugins/ (e.g. 'friedman')"},
                "scaffold_title": {"type": "string"},
                "rebuild": {"type": "boolean", "default": False}
            }
        }
    },
    # Backward-compatible aliases so existing jamovi_* prompts keep working seamlessly
    {
        "name": "jamovi_list_analyses",
        "description": "Alias for openstat_list_analyses.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "jamovi_inspect_dataset",
        "description": "Alias for openstat_inspect_dataset.",
        "inputSchema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"]
        }
    },
    {
        "name": "jamovi_run_analysis",
        "description": "Alias for openstat_run_analysis.",
        "inputSchema": RUN_INPUT_SCHEMA
    },
    {
        "name": "jamovi_convert_dataset",
        "description": "Alias for openstat_convert_dataset.",
        "inputSchema": {
            "type": "object",
            "properties": {"input_path": {"type": "string"}, "output_path": {"type": "string"}},
            "required": ["input_path", "output_path"]
        }
    },
]


def handle_tool_call(name: str, arguments: Dict[str, Any]) -> Any:
    if name in ("openstat_list_analyses", "jamovi_list_analyses"):
        lines = [f"### Available OpenStat Studio Statistical & Graph Modules (v3.0, MIT License — {len(ANALYSIS_CATALOG)} Modules)\n"]
        for c in ANALYSIS_CATALOG:
            lines.append(f"- **`{c['id']}`** ({c['group']}): {c['title']}")
        lines.append(f"\n**Standalone Graph Builder Types (15):** `{', '.join(CHART_TYPES)}`")
        lines.append(f"**Results Export Formats (11):** `{', '.join('.' + f for f in SUPPORTED_TABLE_FORMATS)}`")
        lines.append(f"**Plot Palettes:** `{', '.join(PALETTES.keys())}`")
        lines.append(f"**Plot Themes:** `{', '.join(THEMES)}`")
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    elif name in ("openstat_inspect_dataset", "jamovi_inspect_dataset"):
        sess = JamoviSession(arguments["file_path"], auto_sync_workspace=False)
        return {"content": [{"type": "text", "text": sess.inspect_for_llm()}]}

    elif name == "openstat_wrangle_data":
        file_path = arguments["file_path"]
        op = arguments["operation"]
        params = arguments.get("params", {})
        save_path = arguments.get("save_path")

        sess = JamoviSession(file_path, auto_sync_workspace=True)
        if op == "compute":
            sess.compute(params["target"], params["expr"])
        elif op == "recode":
            sess.recode(params["source"], params["target"], params["rules"], params.get("default_val"))
        elif op == "filter":
            sess.filter(params["query"])
        elif op == "merge":
            kw = {k: params[k] for k in ("kind", "keep", "keepusing", "indicator", "update", "replace", "assert_results", "how")
                  if k in params}
            sess.merge(params.get("right_file") or params["other_source"], on=params.get("on") or params.get("left_on"), **kw)
        elif op == "joinby":
            sess.joinby(params.get("right_file") or params["other_source"], on=params.get("on"),
                        unmatched=params.get("unmatched", "none"))
        elif op == "append":
            sess.append(params.get("right_file") or params["other_source"], generate=params.get("generate"),
                        force=bool(params.get("force", False)))
        elif op == "reshape":
            direction = params.get("direction", "long")
            alias = {"var_name": "time_col", "val_name": "value_col", "i": "id_vars", "j": "time_col", "stubs": "stubnames"}
            clean_params = {alias.get(k, k): v for k, v in params.items() if k != "direction"}
            sess.reshape(direction, **clean_params)
        elif op == "collapse":
            sess.collapse(params["stats"], by=params.get("by"))
        elif op == "egen":
            sess.egen(params["target"], params["func"], params.get("source"), by=params.get("by"))
        elif op == "convert_type":
            sess.convert_type(params["columns"], params["to"], ignore=params.get("ignore", ""), force=bool(params.get("force", False)),
                              percent=bool(params.get("percent", False)), generate=params.get("generate"),
                              date_format=params.get("date_format"))
        elif op == "encode":
            sess.encode(params["source"], params["target"], order=params.get("order"))
        elif op == "decode":
            sess.decode(params["source"], params["target"])
        elif op == "set_value_labels":
            sess.set_value_labels(params["column"], params.get("labels"))
        elif op == "set_measure":
            sess.set_measure(params["columns"], params["level"])
        elif op == "standardize":
            sess.standardize(params["columns"], params.get("method", "zscore"))
        else:
            raise ValueError(f"Unknown data wrangling operation: {op}")

        out_msg = f"[Success] Operation '{op}' applied successfully. Resulting dataset shape: {sess.df.shape[0]} rows × {sess.df.shape[1]} columns."
        if save_path:
            sess.save_dataset(save_path)
            out_msg += f" Saved wrangled data to: {save_path}"

        return {"content": [{"type": "text", "text": out_msg}]}

    elif name in ("openstat_run_analysis", "jamovi_run_analysis"):
        path = arguments["file_path"]
        a_type = arguments["analysis_type"]
        params = dict(arguments.get("parameters") or {})
        plot_style = arguments.get("plot_style")
        if plot_style:
            params["plotStyle"] = plot_style

        out_py = arguments.get("out_py_path")
        export_tables_list: List[str] = arguments.get("export_table_paths") or []
        export_plot = arguments.get("export_plot_path")
        save_omv = arguments.get("save_omv_path")
        save_html = arguments.get("save_html_path")
        push_ui = bool(arguments.get("push_to_live_ui", False))

        sess = JamoviSession(path, auto_sync_workspace=True)
        res = sess.run(a_type, **params)
        output_text = res.to_markdown()
        output_text += f"\n\n### Reproducible Python (`.py`) Script\n```python\n{sess.to_python_script()}\n```"

        if out_py:
            sess.save_py(out_py)
            output_text += f"\n- Saved `.py` syntax: `{out_py}`"
        for t_path in export_tables_list:
            sess.export_tables(t_path)
            output_text += f"\n- Exported analyzed table: `{t_path}`"
        if export_plot and res.plots:
            os.makedirs(os.path.dirname(os.path.abspath(export_plot)), exist_ok=True)
            res.save_plot(export_plot, plot_index=0)
            output_text += f"\n- Saved plot figure: `{export_plot}`"
        if save_omv:
            sess.save_omv(save_omv)
            output_text += f"\n- Saved `.omv` archive: `{save_omv}`"
        if save_html:
            sess.save_html(save_html)
            output_text += f"\n- Saved HTML report: `{save_html}`"
        if push_ui:
            try:
                sess.push_to_ui(a_type, params)
                output_text += "\n- Pushed live to OpenStat Studio UI."
            except Exception as e:
                output_text += f"\n- *(OpenStat UI currently offline: {e})*"

        return {"content": [{"type": "text", "text": output_text}]}

    elif name == "openstat_create_graph":
        path = arguments["file_path"]
        g_params: Dict[str, Any] = {
            "chartType": arguments.get("chart_type", "bar"),
            "x": arguments.get("x"),
            "y": arguments.get("y"),
            "vars": arguments.get("vars") or [],
            "group": arguments.get("group"),
            "sizeVar": arguments.get("size_var"),
            "agg": arguments.get("agg", "mean"),
            "errorBar": arguments.get("error_bar", "ci95"),
            "horizontal": bool(arguments.get("horizontal", False)),
            "stacked100": bool(arguments.get("stacked_100", False)),
            "showValues": bool(arguments.get("show_values", False)),
            "jitter": bool(arguments.get("jitter", False)),
            "fitLine": arguments.get("fit_line", "linear"),
            "bins": int(arguments.get("bins", 18)),
            "showSummaryTable": bool(arguments.get("show_summary_table", True)),
            "caption": arguments.get("caption", ""),
        }
        if arguments.get("plot_style"):
            g_params["plotStyle"] = arguments["plot_style"]

        sess = JamoviSession(path, auto_sync_workspace=True)
        res = sess.run("graph", **g_params)
        output_text = res.to_markdown()
        output_text += f"\n\n### Reproducible Python (`.py`) Script\n```python\n{sess.to_python_script()}\n```"

        export_plot = arguments.get("export_plot_path")
        if export_plot and res.plots:
            os.makedirs(os.path.dirname(os.path.abspath(export_plot)), exist_ok=True)
            res.save_plot(export_plot, plot_index=0)
            output_text += f"\n- Saved standalone graph: `{export_plot}`"

        for r_path in (arguments.get("export_results_paths") or []):
            sess.export_results(r_path)
            output_text += f"\n- Exported Analysis Results (table + graph): `{r_path}`"

        if bool(arguments.get("push_to_live_ui", False)):
            try:
                sess.push_to_ui("graph", g_params)
                output_text += "\n- Pushed graph live to OpenStat Studio UI."
            except Exception as e:
                output_text += f"\n- *(OpenStat UI currently offline: {e})*"

        return {"content": [{"type": "text", "text": output_text}]}

    elif name == "openstat_export_results":
        doc = load_workspace_sync()
        if not doc or not doc.get("specs"):
            raise ValueError("No active workspace session found in workspace/live_session.json.")
        ds_name = doc.get("dataset", "ToothGrowth.csv")
        sess = JamoviSession(ds_name, auto_sync_workspace=False)
        analyses_out = []
        for sp in doc.get("specs", []):
            a_type = sp.get("analysis_type") or sp.get("name")
            if not a_type or a_type == "customFigure":
                continue
            analyses_out.append(dispatch_analysis(a_type, sess.df, sp.get("params") or {}))
        saved_lines = []
        for out_p in (arguments.get("output_paths") or []):
            export_analyses(analyses_out, output_path=out_p, dataset=sess.df)
            saved_lines.append(f"- Exported full Analysis Results -> `{out_p}`")
        return {"content": [{"type": "text", "text": "### Exported Analysis Results\n" + "\n".join(saved_lines)}]}

    elif name == "openstat_exec_python_syntax":
        code = arguments.get("code")
        script_path = arguments.get("script_path")
        if not code and script_path:
            with open(script_path, "r", encoding="utf-8") as f:
                code = f.read()
        if not code:
            raise ValueError("Provide either `code` or `script_path`.")

        default_data = arguments.get("default_data_path")
        default_df = JamoviSession(default_data, auto_sync_workspace=False).df if default_data else None

        analyses, stdout_log, final_df, _ = execute_python_syntax(
            code, default_df=default_df, default_dataset_path=default_data, sync_workspace=True
        )

        save_py = arguments.get("save_py_path")
        if save_py:
            os.makedirs(os.path.dirname(os.path.abspath(save_py)), exist_ok=True)
            with open(save_py, "w", encoding="utf-8") as f:
                f.write(code)

        exported_notes = []
        for t_path in (arguments.get("export_table_paths") or []):
            export_analyses(analyses, output_path=t_path, dataset=final_df)
            exported_notes.append(f"- Exported analyzed tables -> `{t_path}`")

        plot_dir = arguments.get("export_plot_dir")
        p_fmt = arguments.get("plot_format", "svg")
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
            for idx, a in enumerate(analyses, start=1):
                for p_i in range(len(a.plots)):
                    p_out = os.path.join(plot_dir, f"analysis_{idx}_plot_{p_i + 1}.{p_fmt}")
                    a.save_plot(p_out, plot_index=p_i)
                    exported_notes.append(f"- Exported plot -> `{p_out}`")

        parts = []
        if stdout_log.strip():
            parts.append(f"### Script Output\n```\n{stdout_log.rstrip()}\n```")
        for a in analyses:
            parts.append(a.to_markdown())
        if save_py:
            exported_notes.insert(0, f"- Saved Python script -> `{save_py}`")
        if exported_notes:
            parts.append("### Saved Artifacts\n" + "\n".join(exported_notes))

        return {"content": [{"type": "text", "text": "\n\n".join(parts) or "Executed Python script (0 analyses emitted)."}]}

    elif name in ("openstat_convert_dataset", "jamovi_convert_dataset"):
        in_p = arguments["input_path"]
        out_p = arguments["output_path"]
        in_ext = os.path.splitext(in_p)[1].lower()
        out_ext = os.path.splitext(out_p)[1].lower()

        if in_ext == ".omv":
            df, _ = read_omv(in_p)
        elif in_ext == ".csv":
            try:
                df = pd.read_csv(in_p, sep=None, engine="python")
            except Exception:
                df = pd.read_csv(in_p)
        elif in_ext in (".xlsx", ".xls"):
            df = pd.read_excel(in_p)
        else:
            return {"isError": True, "content": [{"type": "text", "text": f"Input format {in_ext} not supported."}]}

        os.makedirs(os.path.dirname(os.path.abspath(out_p)), exist_ok=True)
        if out_ext == ".omv":
            write_omv(df, out_p)
        elif out_ext == ".csv":
            df.to_csv(out_p, index=False, encoding="utf-8-sig")
        elif out_ext == ".xlsx":
            df.to_excel(out_p, index=False)
        else:
            return {"isError": True, "content": [{"type": "text", "text": f"Output format {out_ext} not supported."}]}

        return {"content": [{"type": "text", "text": f"Successfully converted {in_p} -> {out_p}"}]}

    elif name == "openstat_doctor":
        from .diagnostics import startup_self_check_guard, format_doctor_markdown
        check_pypi = bool(arguments.get("check_pypi", True))
        rep = startup_self_check_guard(force=True, check_pypi=check_pypi)
        return {"content": [{"type": "text", "text": format_doctor_markdown(rep)}]}

    elif name == "openstat_sync_jamovi":
        from .upstream_sync import check_jamovi_upstream, scaffold_jamovi_feature, rebuild_openstat_package
        lines = []
        sc_id = arguments.get("scaffold_id")
        if sc_id:
            p = scaffold_jamovi_feature(sc_id, title=arguments.get("scaffold_title"))
            lines.append(f"- Scaffolded clean-room MIT plugin: `{p}`")
        if arguments.get("rebuild"):
            reb = rebuild_openstat_package(run_tests=False)
            lines.append(f"- Rebuilt OpenStat Studio (`passed={reb['passed']}`, total analyses={reb['totalAnalyses']})")
        sync_rep = check_jamovi_upstream(fetch_remote=bool(arguments.get("fetch_remote", True)))
        lines.append(f"### Upstream jamovi Feature Sync (`{sync_rep['sourceMode']}`)")
        lines.append(f"- Reference Release: `{sync_rep['latestJamoviRelease']}`")
        lines.append(f"- Active OpenStat Analyses: **{sync_rep['openstatTotalAnalyses']}** (`jmv` Core Covered: {sync_rep['jmvCoreCoveredCount']})")
        if sync_rep["availableExtensionsToScaffold"]:
            lines.append("- Available Upstream Extensions to Scaffold: " + ", ".join(f"`{x['suggestedId']}` ({x['title']})" for x in sync_rep["availableExtensionsToScaffold"]))
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    raise ValueError(f"Unknown tool: {name}")


def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            req = json.loads(line)
            method = req.get("method")
            msg_id = req.get("id")

            if method == "initialize":
                res = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "openstat-studio-mcp", "version": "3.0.0"}
                    }
                }
            elif method == "tools/list":
                res = {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
            elif method == "tools/call":
                params = req.get("params", {})
                call_res = handle_tool_call(params.get("name"), params.get("arguments", {}))
                res = {"jsonrpc": "2.0", "id": msg_id, "result": call_res}
            elif method and method.startswith("notifications/"):
                # MCP client notifications (e.g., notifications/initialized) do not expect a response
                continue
            else:
                res = {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}

            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err_res = {
                "jsonrpc": "2.0",
                "id": req.get("id") if "req" in locals() else None,
                "error": {"code": -32603, "message": str(e)}
            }
            sys.stdout.write(json.dumps(err_res) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()

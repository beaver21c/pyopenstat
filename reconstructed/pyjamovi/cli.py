"""
OpenStat Studio CLI (`openstat` / `pyjamovi` v3.0)
License: MIT
Allows Claude Code, Codex, Antigravity, and researchers to run all 28 statistical analyses,
author/execute `.py` scripts, customize plot designs (`PlotStyle`), and export analyzed tables to 10 formats.
"""

import argparse
import sys
import os
import json
import pandas as pd

from .agent import JamoviSession
from .registry import ANALYSIS_CATALOG
from .analyses.plot_style import PALETTES, THEMES
from .analyses.graph_builder import CHART_TYPES, CHART_TYPE_LABELS
from .exporters import SUPPORTED_TABLE_FORMATS, export_analyses
from .syntax_engine import execute_python_syntax
from .omv.reader import read_omv
from .omv.writer import write_omv


def _report_kwargs(args) -> dict:
    """report_style / report_options of an export command (options given as JSON)."""
    style = getattr(args, "report_style", None)
    raw = getattr(args, "report_options", None)
    opts = None
    if raw:
        try:
            opts = json.loads(raw)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--report-options: JSON 형식 오류 ({e})")
        if not isinstance(opts, dict):
            raise SystemExit("--report-options: JSON 객체({...}) 필요")
        style = style or "social"
    return {"report_style": style, "report_options": opts}


def main():
    parser = argparse.ArgumentParser(
        description="OpenStat Studio v3.0 (`openstat` / `pyjamovi`): Pure-Python Statistical Engine, .py Syntax Runner, Customizable Plots & Multi-Format Exporter"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Command: catalog / list
    subparsers.add_parser("catalog", help="List all 28 statistical analyses, plot palettes & export formats")
    subparsers.add_parser("list", help="Alias for catalog")

    # Command: palettes
    subparsers.add_parser("palettes", help="List available PlotStyle color palettes and visual themes")

    # Command: info
    p_info = subparsers.add_parser("info", help="Inspect a dataset (.omv, .csv, .xlsx)")
    p_info.add_argument("--data", required=True, help="Path to dataset file")

    # Command: run
    p_run = subparsers.add_parser("run", help="Run any statistical analysis, generate .py syntax, and export tables/plots")
    p_run.add_argument("--data", required=True, help="Path to dataset file (.csv, .omv, .xlsx)")
    p_run.add_argument("--analysis", required=True, help="Analysis ID (e.g., ttestIS, anovaOneW, anovaFactorial, anovaRM, bayesTTestIS, sem, panel, mediation, moderation, logRegBin, reliability, efa, survival)")
    p_run.add_argument("--params", help="JSON dictionary of analysis parameters")
    p_run.add_argument("--vars", help="Target variable(s), comma-separated")
    p_run.add_argument("--dep", help="Dependent variable")
    p_run.add_argument("--group", help="Grouping / factor variable")
    p_run.add_argument("--factors", help="Factors, comma-separated")
    p_run.add_argument("--covs", help="Covariates, comma-separated")
    p_run.add_argument("--rows", help="Row variable for crosstab")
    p_run.add_argument("--cols", help="Column variable for crosstab")
    p_run.add_argument("--model", help="lavaan model syntax for SEM/CFA")
    p_run.add_argument("--welch", action="store_true", help="Use Welch's test")
    # Plot customization flags
    p_run.add_argument("--plot-style", help="JSON dictionary of PlotStyle options (palette, theme, custom_colors, title, xlabel, ylabel, width, height, dpi, font_scale, show_grid, legend_pos)")
    p_run.add_argument("--palette", help=f"Plot color palette ({', '.join(PALETTES.keys())})")
    p_run.add_argument("--theme", help=f"Plot visual theme ({', '.join(THEMES)})")
    p_run.add_argument("--plot-title", help="Custom plot title")
    p_run.add_argument("--xlabel", help="Custom X-axis label")
    p_run.add_argument("--ylabel", help="Custom Y-axis label")
    p_run.add_argument("--dpi", type=int, help="Plot resolution DPI (72-600)")
    # Output & Export flags
    p_run.add_argument("--format", choices=["markdown", "text", "apa", "json", "py"], default="markdown")
    p_run.add_argument("--out-py", help="Save reproducible standalone Python (.py) script to path")
    p_run.add_argument("--export-table", "--export-results", dest="export_table", action="append", default=[], help="Export Analysis Results (tables + embedded graphs) to path (.pdf, .hwpx, .docx, .xlsx, .tex, .csv, .tsv, .md, .html, .json, .omv). Repeatable.")
    p_run.add_argument("--export-plot", help="Save primary figure to path (.svg, .pdf, or .png)")
    p_run.add_argument("--out-omv", help="Save dataset and result to .omv file")
    p_run.add_argument("--out-html", help="Save report (including plots) to standalone .html file")
    p_run.add_argument("--push-ui", action="store_true", help="Also push analysis live to OpenStat Studio Web UI if running")
    p_run.add_argument("--no-sync", action="store_true", help="Skip updating workspace/live_session.py")

    # Command: graph (Standalone 15-type Graph Builder -> Analysis Results + Multi-Format Export)
    p_gr = subparsers.add_parser("graph", help="Draw any of 15 standalone chart types into Analysis Results and export to .svg/.pdf/.png or .pdf/.hwpx/.docx/.xlsx reports")
    p_gr.add_argument("--data", required=True, help="Path to dataset file (.csv, .omv, .xlsx)")
    p_gr.add_argument("--type", dest="chart_type", choices=list(CHART_TYPES), default="bar", help=f"Chart type ({', '.join(CHART_TYPES)})")
    p_gr.add_argument("--x", help="X-axis / category variable")
    p_gr.add_argument("--y", help="Y-axis / value variable")
    p_gr.add_argument("--vars", help="Multiple variables (comma-separated, for pairplot / heatmap)")
    p_gr.add_argument("--group", help="Color / split grouping variable")
    p_gr.add_argument("--size-var", help="Bubble size variable (for scatter)")
    p_gr.add_argument("--agg", default="mean", choices=["mean", "median", "sum", "count", "pct"], help="Aggregation function")
    p_gr.add_argument("--error-bar", default="ci95", choices=["ci95", "se", "sd", "none"], help="Error bar / ribbon interval")
    p_gr.add_argument("--horizontal", action="store_true", help="Horizontal orientation")
    p_gr.add_argument("--stacked-100", action="store_true", help="100%% proportional stacked bar")
    p_gr.add_argument("--show-values", action="store_true", help="Annotate bar data values")
    p_gr.add_argument("--jitter", action="store_true", help="Overlay jittered data points on box plot")
    p_gr.add_argument("--fit-line", default="linear", choices=["linear", "quadratic", "smooth", "none"], help="Scatter trend fit line")
    p_gr.add_argument("--bins", type=int, default=18, help="Histogram bins")
    p_gr.add_argument("--caption", default="", help="Custom figure caption / APA note")
    p_gr.add_argument("--no-summary-table", action="store_true", help="Omit companion summary data table")
    p_gr.add_argument("--plot-style", help="JSON dictionary of PlotStyle options")
    p_gr.add_argument("--palette", help=f"Plot color palette ({', '.join(PALETTES.keys())})")
    p_gr.add_argument("--theme", help=f"Plot visual theme ({', '.join(THEMES)})")
    p_gr.add_argument("--plot-title", help="Custom plot title")
    p_gr.add_argument("--xlabel", help="Custom X-axis label")
    p_gr.add_argument("--ylabel", help="Custom Y-axis label")
    p_gr.add_argument("--dpi", type=int, help="Plot resolution DPI (72-600)")
    p_gr.add_argument("--export-plot", help="Save rendered graph directly to .svg, .pdf, or .png")
    p_gr.add_argument("--export-results", "--export-table", dest="export_results", action="append", default=[], help="Export Analysis Results (summary table + embedded graph) to .pdf, .hwpx, .docx, .xlsx, .html, .tex, .md, .omv")
    p_gr.add_argument("--out-py", help="Save reproducible standalone Python (.py) script")
    p_gr.add_argument("--format", choices=["markdown", "json", "apa", "py"], default="markdown")
    p_gr.add_argument("--push-ui", action="store_true", help="Push graph live to OpenStat Studio Web UI if running")

    # Command: exec-py
    p_exec = subparsers.add_parser("exec-py", help="Execute a Python (.py) statistical syntax file or inline code directly")
    p_exec.add_argument("--script", help="Path to .py statistical syntax file")
    p_exec.add_argument("--code", help="Inline Python statistical code string")
    p_exec.add_argument("--data", help="Optional default dataset path injected as `sess` / `df`")
    p_exec.add_argument("--export-table", "--export-results", dest="export_table", action="append", default=[], help="Export resulting tables & graphs to path (.pdf, .hwpx, .docx, .xlsx, .tex, .csv, .md, .html, .omv)")
    p_exec.add_argument("--export-plot-dir", help="Directory to export all generated figures")
    p_exec.add_argument("--plot-format", choices=["svg", "pdf", "png"], default="svg", help="Vector/raster format for --export-plot-dir")
    p_exec.add_argument("--format", choices=["markdown", "json", "apa"], default="markdown")

    # Command: convert
    for _sp in (p_run, p_gr, p_exec):
        _sp.add_argument("--report-style", choices=["default", "social"], default=None,
                         help="Table format of exported results: default (APA, every statistic in its own column) or "
                              "social (significance stars, SE in parentheses, numbered <표 n> captions; coefficient tables only)")
        _sp.add_argument("--report-options", default=None,
                         help='JSON options of the social format, e.g. \'{"se_position": "below", "stars": "dagger", "keep_p": true}\'')

    p_conv = subparsers.add_parser("convert", help="Convert datasets between .csv/.xlsx and .omv")
    p_conv.add_argument("--input", required=True, help="Input file path")
    p_conv.add_argument("--output", required=True, help="Output file path (.omv, .csv, or .xlsx)")

    # Command: doctor (Package freshness + API/syntax drift self-check)
    p_doc = subparsers.add_parser("doctor", help="Check Python package freshness (PyPI) and run API/syntax compatibility self-diagnostics")
    p_doc.add_argument("--offline", action="store_true", help="Skip live PyPI version lookup")
    p_doc.add_argument("--json", action="store_true", help="Output raw JSON report")

    # Command: sync-jamovi (Upstream jamovi feature diff + clean-room plugin scaffolder)
    p_sync = subparsers.add_parser("sync-jamovi", help="Check upstream jamovi for new statistical features and scaffold clean-room plugins")
    p_sync.add_argument("--offline", action="store_true", help="Use local reference manifest without querying GitHub API")
    p_sync.add_argument("--scaffold", help="Analysis ID to scaffold in reconstructed/pyjamovi/plugins/ (e.g. friedman, mcnemar)")
    p_sync.add_argument("--title", help="Custom display title when scaffolding a new analysis")
    p_sync.add_argument("--group", default="Plugins", help="Ribbon menu group for scaffolded plugin")
    p_sync.add_argument("--rebuild", action="store_true", help="Automatically rebuild package after scaffolding")

    # Command: rebuild (One-command clean rebuild + self-check + pytest)
    p_reb = subparsers.add_parser("rebuild", help="Rebuild OpenStat Studio assets, plugin catalog, wheel & run self-diagnostics")
    p_reb.add_argument("--skip-tests", action="store_true", help="Skip pytest suite during rebuild")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "doctor":
        from .diagnostics import startup_self_check_guard, format_doctor_markdown
        rep = startup_self_check_guard(force=True, check_pypi=not args.offline)
        if args.json:
            print(json.dumps(rep, indent=2, ensure_ascii=False))
        else:
            print(format_doctor_markdown(rep))
        return

    if args.command == "sync-jamovi":
        from .upstream_sync import check_jamovi_upstream, scaffold_jamovi_feature, rebuild_openstat_package
        if args.scaffold:
            created = scaffold_jamovi_feature(args.scaffold, title=args.title, group=args.group)
            print(f"[Scaffolded Clean-Room MIT Plugin] -> {created}")
            if args.rebuild:
                reb = rebuild_openstat_package(run_tests=False)
                print(f"[Rebuilt Catalog] {reb['totalAnalyses']} analyses active (healthy={reb['passed']})")
            return

        rep = check_jamovi_upstream(fetch_remote=not args.offline)
        print(f"=== OpenStat Studio <-> Upstream jamovi Feature Sync ({rep['sourceMode']}) ===")
        print(f"Upstream jamovi Reference : {rep['latestJamoviRelease']}")
        print(f"OpenStat Active Analyses  : {rep['openstatTotalAnalyses']} (jmv Core Covered: {rep['jmvCoreCoveredCount']})")
        print(f"Extended Modules Included : {', '.join(rep['openstatExtendedModules'])}")
        if rep["availableExtensionsToScaffold"]:
            print("\nAvailable Upstream Analyses Ready to Scaffold (`openstat sync-jamovi --scaffold <id> --rebuild`):")
            for item in rep["availableExtensionsToScaffold"]:
                print(f"  - {item['suggestedId']:14s} ({item['group']:11s}) : {item['title']}")
        else:
            print("\nAll tracked upstream jamovi analyses are implemented!")
        return

    if args.command == "rebuild":
        from .upstream_sync import rebuild_openstat_package
        manifest = rebuild_openstat_package(run_tests=not args.skip_tests)
        print(f"=== OpenStat Studio Rebuild ({'PASS' if manifest['passed'] else 'FAIL'}) in {manifest['elapsedSeconds']}s ===")
        for s in manifest["steps"]:
            status = "PASS" if s["passed"] else "FAIL"
            print(f"  [{status}] {s['step']} — {s['output']}")
        if not manifest["passed"]:
            sys.exit(1)
        return

    if args.command in ("catalog", "list"):
        print(f"=== OpenStat Studio v3.0 — {len(ANALYSIS_CATALOG)} Statistical & Visualization Modules ===")
        for item in ANALYSIS_CATALOG:
            print(f"[{item['group']:13s}] {item['id']:16s} — {item['title']}")
        print(f"\nStandalone Chart Types (15) : {', '.join(CHART_TYPES)}")
        print(f"Results Export Formats (11) : {', '.join('.' + f for f in SUPPORTED_TABLE_FORMATS)}")
        print(f"Plot Palettes               : {', '.join(PALETTES.keys())}")
        print(f"Plot Themes                 : {', '.join(THEMES)}")
        return

    if args.command == "palettes":
        print("=== PlotStyle Color Palettes ===")
        for k, cols in PALETTES.items():
            print(f"  {k:14s}: {', '.join(cols)}")
        print(f"\n=== PlotStyle Themes ===\n  {', '.join(THEMES)}")
        return

    if args.command == "info":
        sess = JamoviSession(args.data, auto_sync_workspace=False)
        print(sess.inspect_for_llm())
        return

    if args.command == "convert":
        in_ext = os.path.splitext(args.input)[1].lower()
        out_ext = os.path.splitext(args.output)[1].lower()
        if in_ext == ".omv":
            df, _ = read_omv(args.input)
        elif in_ext == ".csv":
            try:
                df = pd.read_csv(args.input, sep=None, engine="python")
            except Exception:
                df = pd.read_csv(args.input)
        elif in_ext in (".xlsx", ".xls"):
            df = pd.read_excel(args.input)
        else:
            print(f"Error: Unsupported input format {in_ext}")
            sys.exit(1)

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        if out_ext == ".omv":
            write_omv(df, args.output)
        elif out_ext == ".csv":
            df.to_csv(args.output, index=False, encoding="utf-8-sig")
        elif out_ext == ".xlsx":
            df.to_excel(args.output, index=False)
        else:
            print(f"Error: Unsupported output format {out_ext}")
            sys.exit(1)
        print(f"Successfully converted {args.input} -> {args.output}")
        return

    if args.command == "exec-py":
        if not args.script and not args.code:
            print("Error: Provide either --script <file.py> or --code '...'")
            sys.exit(1)
        code = args.code
        if args.script:
            with open(args.script, "r", encoding="utf-8") as f:
                code = f.read()
        default_df = None
        if args.data:
            s_tmp = JamoviSession(args.data, auto_sync_workspace=False)
            default_df = s_tmp.df
        analyses, stdout_log, final_df, _ = execute_python_syntax(
            code, default_df=default_df, default_dataset_path=args.data, sync_workspace=True
        )
        if args.format == "json":
            print(json.dumps({
                "stdout": stdout_log,
                "analyses": [a.to_payload(i + 1) for i, a in enumerate(analyses)]
            }, indent=2, ensure_ascii=False))
        elif args.format == "apa":
            for a in analyses:
                if a.apa_narrative:
                    print(a.apa_narrative)
        else:
            if stdout_log.strip():
                print(stdout_log.rstrip())
            else:
                for a in analyses:
                    print(a.to_markdown())

        for t_path in (args.export_table or []):
            export_analyses(analyses, output_path=t_path, dataset=final_df, **_report_kwargs(args))
            print(f"[Exported analyzed tables -> {t_path}]")

        if args.export_plot_dir:
            os.makedirs(args.export_plot_dir, exist_ok=True)
            for idx, a in enumerate(analyses, start=1):
                for p_i in range(len(a.plots)):
                    p_out = os.path.join(args.export_plot_dir, f"analysis_{idx}_plot_{p_i + 1}.{args.plot_format}")
                    a.save_plot(p_out, plot_index=p_i)
                    print(f"[Exported plot -> {p_out}]")
        return

    if args.command == "graph":
        sess = JamoviSession(args.data, auto_sync_workspace=True)
        g_params: Dict[str, Any] = {
            "chartType": args.chart_type,
            "agg": args.agg,
            "errorBar": args.error_bar,
            "horizontal": bool(args.horizontal),
            "stacked100": bool(args.stacked_100),
            "showValues": bool(args.show_values),
            "jitter": bool(args.jitter),
            "fitLine": args.fit_line,
            "bins": args.bins,
            "showSummaryTable": not bool(args.no_summary_table),
            "caption": args.caption or "",
        }
        if args.x:
            g_params["x"] = args.x
        if args.y:
            g_params["y"] = args.y
        if args.vars:
            g_params["vars"] = [v.strip() for v in args.vars.split(",") if v.strip()]
        if args.group:
            g_params["group"] = args.group
        if args.size_var:
            g_params["sizeVar"] = args.size_var

        p_style = json.loads(args.plot_style) if args.plot_style else {}
        if args.palette:
            p_style["palette"] = args.palette
        if args.theme:
            p_style["theme"] = args.theme
        if args.plot_title:
            p_style["title"] = args.plot_title
        if args.xlabel:
            p_style["xlabel"] = args.xlabel
        if args.ylabel:
            p_style["ylabel"] = args.ylabel
        if args.dpi:
            p_style["dpi"] = args.dpi
        if p_style:
            g_params["plotStyle"] = p_style

        res = sess.run("graph", **g_params)

        if args.push_ui:
            try:
                sess.push_to_ui("graph", g_params)
            except Exception as e:
                sys.stderr.write(f"[Notice] OpenStat UI not active: {e}\n")

        if args.format == "markdown":
            print(res.to_markdown())
        elif args.format == "apa":
            print(res.apa_narrative)
        elif args.format == "py":
            print(sess.to_python_script())
        elif args.format == "json":
            print(json.dumps(res.to_payload(), indent=2, ensure_ascii=False))

        if args.out_py:
            sess.save_py(args.out_py)
            print(f"[Saved Python syntax script: {args.out_py}]")

        if args.export_plot and res.plots:
            os.makedirs(os.path.dirname(os.path.abspath(args.export_plot)), exist_ok=True)
            res.save_plot(args.export_plot, plot_index=0)
            print(f"[Saved standalone graph: {args.export_plot}]")

        for r_out in (args.export_results or []):
            sess.export_results(r_out, **_report_kwargs(args))
            print(f"[Exported Analysis Results (table + graph): {r_out}]")
        return

    if args.command == "run":
        sess = JamoviSession(args.data, auto_sync_workspace=not args.no_sync)
        params = json.loads(args.params) if args.params else {}
        if args.vars:
            params.setdefault("vars", [v.strip() for v in args.vars.split(",")])
        if args.dep:
            params.setdefault("dep", args.dep)
        if args.group:
            params.setdefault("group", args.group)
        if args.factors:
            params.setdefault("factors", [v.strip() for v in args.factors.split(",")])
        if args.covs:
            params.setdefault("covs", [v.strip() for v in args.covs.split(",")])
        if args.rows:
            params.setdefault("rows", args.rows)
        if args.cols:
            params.setdefault("cols", args.cols)
        if args.model:
            params.setdefault("model", args.model)
        if args.welch:
            params["welchs"] = True

        p_style = json.loads(args.plot_style) if args.plot_style else dict(params.get("plotStyle") or {})
        if args.palette:
            p_style["palette"] = args.palette
        if args.theme:
            p_style["theme"] = args.theme
        if args.plot_title:
            p_style["title"] = args.plot_title
        if args.xlabel:
            p_style["xlabel"] = args.xlabel
        if args.ylabel:
            p_style["ylabel"] = args.ylabel
        if args.dpi:
            p_style["dpi"] = args.dpi
        if p_style:
            params["plotStyle"] = p_style

        res = sess.run(args.analysis, **params)

        if args.push_ui:
            try:
                sess.push_to_ui(args.analysis, params)
            except Exception as e:
                sys.stderr.write(f"[Notice] OpenStat UI not active: {e}\n")

        if args.format == "markdown":
            print(res.to_markdown())
        elif args.format == "text":
            print(res.to_string())
        elif args.format == "apa":
            print(res.apa_narrative)
        elif args.format == "py":
            print(sess.to_python_script())
        elif args.format == "json":
            print(json.dumps(res.to_payload(), indent=2, ensure_ascii=False))

        if args.out_py:
            sess.save_py(args.out_py)
            print(f"\n[Saved Python syntax script: {args.out_py}]")

        for t_out in (args.export_table or []):
            sess.export_tables(t_out, **_report_kwargs(args))
            print(f"[Exported analyzed tables: {t_out}]")

        if args.export_plot and res.plots:
            os.makedirs(os.path.dirname(os.path.abspath(args.export_plot)), exist_ok=True)
            res.save_plot(args.export_plot, plot_index=0)
            print(f"[Saved customized plot: {args.export_plot}]")

        if args.out_omv:
            sess.save_omv(args.out_omv)
            print(f"[Saved .omv archive: {args.out_omv}]")

        if args.out_html:
            sess.save_html(args.out_html)
            print(f"[Saved standalone HTML report: {args.out_html}]")


if __name__ == "__main__":
    main()

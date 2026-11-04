# CLAUDE.md — OpenStat Studio v3.0 Direct Manipulation Guide for Claude Code

This repository contains **OpenStat Studio v3.0** (`openstat` / `pyjamovi`, MIT License), a standalone open-source pure-Python statistical engine, **15-type standalone Graph Builder**, live `.py` syntax studio, customizable publication plot designer (`PlotStyle`), Stata-equivalent data wrangling engine (`compute`, `recode`, `filter`, `merge`, `reshape`), and **11-format Analysis Results exporter (tables + embedded graphs)** designed with reference to [jamovi](https://www.jamovi.org).

> **Background Server Policy**: Do **not** leave background web servers running persistently unless the user explicitly asks to open the browser UI (`START_OPENSTAT.sh` / `START_OPENSTAT.bat`). Execute analyses directly via Python SDK (`import openstat as ost`), CLI (`openstat`), or MCP (`openstat-mcp`).

---

## 1. Quick Setup & One-Click MCP Registration

If you need to register OpenStat Studio tools with Claude Code in any environment:
- **Windows**: Run `SETUP_CLAUDE_CODE.bat`
- **macOS / Linux**: Run `./SETUP_CLAUDE_CODE.sh` (or `python scripts/setup_claude_code.py`)

This automatically configures `.mcp.json` for the current workspace and registers the `openstat` toolset into your Claude Code session.

---

## 2. Quick Commands (`openstat` CLI)

Run using the local virtualenv (`.venv/bin/openstat` on Unix, `.venv\Scripts\openstat` on Windows, or `openstat` if installed):

```bash
# 1. Check Python package freshness (PyPI) & syntax/API compatibility self-diagnostics
openstat doctor

# 2. Check upstream jamovi for newly added features & rebuild OpenStat Studio
openstat sync-jamovi
openstat rebuild

# 3. List all 28+ statistical analyses, graph types, and plot palettes/themes
openstat catalog
openstat palettes

# 4. Run an analysis, save live .py syntax, customize PlotStyle, and export tables & vector plots
openstat run \
  --data sample_data/ToothGrowth.csv --analysis anovaOneW \
  --params '{"deps":["len"],"group":"dose","phMethod":"gamesHowell","descPlot":true}' \
  --palette kihasa --theme apa_classic --plot-title "용량별 치아 성장 비교" \
  --out-py workspace/anova_analysis.py \
  --export-table reports/anova_tables.hwpx \
  --export-plot reports/anova_plot.svg

# 5. Draw any of 15 standalone chart types into Analysis Results & export to .svg / .pdf / .hwpx
openstat graph \
  --data sample_data/ToothGrowth.csv --type raincloud \
  --x dose --y len --group supp \
  --palette kihasa --theme apa_classic \
  --export-plot reports/raincloud.svg \
  --export-results reports/graph_report.pdf \
  --export-results reports/graph_report.hwpx

# 6. Execute a standalone .py syntax script directly and export all tables + figures
openstat exec-py --script workspace/live_session.py \
  --export-results reports/session_results.pdf \
  --export-results reports/session_results.xlsx
```

---

## 3. Live Python (`.py`) Authoring & Real-Time Session API (`ost.Session`)

Every `ost.Session` automatically runs `startup_self_check_guard()` (applying compatibility shims and checking for upstream package API changes) and syncs real-time syntax + results to `workspace/live_session.py` and `workspace/live_session.json`.

```python
import openstat as ost

# Load CSV, Excel (.xlsx), Parquet, or jamovi (.omv)
sess = ost.Session("sample_data/Grunfeld_Panel.csv")

# 1. Stata-Equivalent Data Wrangling (Logged to live syntax & reproducible)
sess.compute("inv_cap_ratio", "invest / capital")
sess.recode("year", "decade", [
    {"type": "range", "from": 1935, "to": 1939, "value": 1, "label": "1930s"},
    {"type": "range", "from": 1940, "to": 1949, "value": 2, "label": "1940s"},
    {"type": "range", "from": 1950, "to": 1959, "value": 3, "label": "1950s"},
])
sess.filter("firm in [1, 2, 3, 4, 5]")
sess.standardize(columns=["value", "capital"], method="zscore")

# 2. Panel Data Econometrics (Entity FE, Two-Way FE, Random Effects GLS, Pooled OLS + Hausman Test)
res_panel = sess.panel(
    entity="firm",
    time="year",
    dep="invest",
    covs=["value", "capital"],
    model_type="fixed",
    cov_type="clustered",
    hausman=True,
    plotStyle={
        "palette": "kihasa",
        "theme": "whitegrid",
        "title": "패널 고정효과 모형 추세선 (Grunfeld)",
        "xlabel": "연도 (Year)",
        "ylabel": "설비투자 (Investment)",
        "dpi": 200,
    },
)

# 3. Structural Equation Modeling (CFA & Latent Path SEM via lavaan syntax)
sem_sess = ost.Session("sample_data/SEM_Latent3Factor.csv")
res_sem = sem_sess.sem(
    model="""
    Visual  =~ x1_vis + x2_vis + x3_vis
    Textual =~ x4_txt + x5_txt + x6_txt
    Speed   =~ x7_spd + x8_spd + x9_spd
    GPA ~ Visual + Textual + Speed
    """,
    std_est=True,
    fit_measures=True,
    path_diagram=True,
    plotStyle={"palette": "nature", "theme": "clean", "title": "3요인 구조방정식 경로모형"},
)

# 4. Standalone Graph Builder (15 Chart Types -> Analysis Results) & Custom Matplotlib Figures
sess.graph(
    chart_type="raincloud",  # bar, stacked_bar, line, area, pie, donut, histogram, density, box, violin, raincloud, scatter, pairplot, heatmap, errorbar
    x="firm",
    y="invest",
    plotStyle={"palette": "nature", "theme": "clean", "title": "기업별 설비투자 분포 (Raincloud)"},
)

# Any custom Matplotlib figure can also be exported/attached directly into Analysis Results:
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(sess.df.groupby("year")["invest"].mean(), marker="o", color="#0D9488")
ax.set_title("연도별 평균 설비투자 추이")
sess.add_figure(fig, title="연도별 평균 설비투자 추이", attach_to_last=False)

# 5. Export FULL Analysis Results (APA tables + ALL embedded graphs) to ANY of 11 formats:
#    .pdf, .hwpx (한글 표+그림), .docx, .xlsx, .tex (LaTeX+PDF figures), .html, .md, .csv, .tsv, .json, .omv
sess.export_results("reports/full_results.pdf")
sess.export_results("reports/full_results.hwpx")
sess.export_results("reports/full_results.docx")
sess.export_results("reports/full_results.xlsx")
res_sem.export_results("reports/sem_results.tex")

# 6. Restyle plots dynamically & export vector SVG / PDF / high-DPI PNG
res_panel = sess.restyle_plots(
    palette="okabe_ito",
    theme="apa_classic",
    custom_colors=["#00457C", "#0D9488", "#D97706"],
)
res_panel.save_plot("reports/panel_trajectory.svg")
res_panel.save_plot("reports/panel_trajectory.pdf")

# 7. Save reproducible standalone Python (.py) script
sess.save_py("reports/reproducible_study.py")

# 8. Text mining & networks (.txt .md .docx .pdf .hwpx -> one row per document)
docs = ost.Session(ost.load_texts("interviews/"))          # columns: 문서ID, 파일명, 폴더, 형식, 글자수, 본문
docs.text_keywords(text="본문", group="폴더", stopwords="연구, 분석", synonyms="노인: 어르신, 고령자")
docs.text_network(text="본문", unit="sentence", topNodes=60)
docs.text_cluster(text="본문", docLabel="파일명", embedding="tfidf")   # embedding="sbert" + requirements-embedding.txt for BAAI/bge-m3
edges = ost.Session("sample_data/Network_Collaboration.csv")
edges.graph_network(source="출발기관", target="도착기관", weight="협력횟수")

# 9. GIS spatial analysis (boundary .shp/.zip/.geojson/.gpkg or lon/lat columns)
geo = ost.Session("sample_data/Spatial_Panel_Welfare.csv")
geo.attach_boundary("sample_data/geo/Synthetic_Regions.shp", key="SIG_CD", data_key="지역코드")
geo.spatial_weights(wType="queen", wNorm="row", wIslands="nearest", wExport="workspace/W_queen.gal")
geo.spatial_autocorrelation(vars=["돌봄이용률"], timeVar="연도", period="2023", gi=True)
geo.spatial_regression(dep="돌봄이용률", covs=["고령인구비율", "재정자립도"], timeVar="연도", model="sdm")   # OLS-LM, SAR, SEM, SDM, SDEM, SLX, SAC
geo.spatial_panel(dep="돌봄이용률", covs=["고령인구비율", "1인가구비율"], timeVar="연도",
                  model="sdm", effectsType="fe", feType="both", leeyu=True)                     # Stata: xsmle ..., model(sdm) fe type(both, leeyu)
geo.gwr(dep="돌봄이용률", covs=["고령인구비율"], timeVar="연도", kernel="bisquare", bwType="adaptive")
```

---

## 4. MCP Tools Registered in `.mcp.json` (`openstat-mcp`)

Claude Code automatically connects to `.mcp.json` and gains native tool calls:
- `openstat_list_analyses`: Lists all 28+ statistical modules, 15 standalone chart types, 10 palettes, 5 themes, and 11 results export formats.
- `openstat_inspect_dataset`: Profiles column types, missing counts, summary stats, and recommended analyses for `.csv`/`.xlsx`/`.omv`.
- `openstat_wrangle_data`: Performs Stata-equivalent data cleaning (`compute`, `recode`, `filter`, `merge`, `reshape`, `standardize`) with live `.py` syntax logging.
- `openstat_run_analysis`: Runs any analysis with `plot_style`, `out_py_path`, `export_table_paths`, and `export_plot_path`.
- `openstat_create_graph`: Draws any of 15 standalone chart types into Analysis Results and exports `.svg`/`.pdf`/`.png` or `.pdf`/`.hwpx`/`.docx`/`.xlsx` reports.
- `openstat_export_results`: Exports all active workspace Analysis Results (tables + embedded graphs) to 11 formats.
- `openstat_exec_python_syntax`: Executes raw Python (`.py`) code directly (auto-capturing any `plt.figure()` charts into Analysis Results) and exports reports.
- `openstat_doctor`: Runs live PyPI package freshness checks + 6-subsystem syntax/API drift self-diagnostics.
- `openstat_sync_jamovi`: Checks upstream jamovi for new features, scaffolds clean-room plugins (`reconstructed/pyjamovi/plugins/`), or rebuilds the package.
- `openstat_convert_dataset`: Converts datasets seamlessly between `.omv`, `.csv`, and `.xlsx`.

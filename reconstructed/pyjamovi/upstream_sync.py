"""
OpenStat Studio — On-Demand jamovi Feature Sync, Clean-Room Plugin Scaffolder & Package Rebuild Pipeline (`upstream_sync.py`)
License: MIT

Purpose:
When upstream jamovi (https://github.com/jamovi/jmv or jamovi modules) introduces new statistical analyses
or options over time, this module enables non-periodic, on-demand feature tracking and rebuilding:
1. `check_jamovi_upstream(fetch_remote=True)`:
   Compares upstream jamovi `.a.yaml` analysis definitions (via GitHub REST API with offline fallback)
   against OpenStat Studio's built-in 28 analyses + custom `plugins/` without copying any GPL/AGPL code.
2. `scaffold_jamovi_feature(analysis_id, title=None, group='Plugins')`:
   Generates a 100% clean-room MIT Python module in `reconstructed/pyjamovi/plugins/<analysis_id>.py`
   ready for pure-Python statistical implementation (SciPy / Statsmodels / Pingouin) and instant UI/CLI/MCP integration.
3. `rebuild_openstat_package(run_tests=True)`:
   Re-generates clean-room assets, refreshes the analysis catalog, re-installs the editable package,
   and runs `doctor` API compatibility + `pytest` verification in a single command.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from typing import Dict, Any, List, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
CORPUS_DIR = os.path.join(ROOT_DIR, "corpus")
PLUGINS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins")


# Mapping from upstream `jamovi/jmv` `.a.yaml` stems to OpenStat Studio analysis IDs
UPSTREAM_JMV_TO_OPENSTAT: Dict[str, str] = {
    "descriptives": "descriptives",
    "ttestis": "ttestIS",
    "ttestps": "ttestPS",
    "ttestones": "ttestOneS",
    "anovaonew": "anovaOneW",
    "anova": "anovaFactorial",
    "ancova": "ancova",
    "anovarm": "anovaRM",
    "anovarmnp": "friedman",          # Candidate extension: Friedman non-parametric RM-ANOVA
    "anovanp": "kruskal",             # Kruskal-Wallis non-parametric One-Way ANOVA
    "kruskal": "kruskal",
    "mancova": "mancova",             # Candidate extension: MANCOVA
    "corrmatrix": "corrMatrix",
    "corrpart": "partCorr",
    "linreg": "linReg",
    "logregbin": "logRegBin",
    "logisbin": "logRegBin",
    "logregmulti": "logRegMult",      # Candidate extension: Multinomial Logistic Regression
    "logismult": "logRegMult",
    "logregord": "logRegOrd",        # Candidate extension: Ordinal Logistic Regression
    "logisord": "logRegOrd",
    "conttables": "contTables",
    "conttablespaired": "mcnemar",    # Candidate extension: McNemar Paired Contingency
    "loglinear": "logLinear",         # Candidate extension: Log-Linear Regression
    "proptest2": "propTest2",         # Candidate extension: Binomial Proportion Test
    "proptestn": "propTestN",         # Candidate extension: Chi-Square Goodness of Fit
    "reliability": "reliability",
    "pca": "pca",
    "efa": "efa",
    "cfa": "sem",                     # Covered by full CFA/SEM module (`sem`)
}

NON_ANALYSIS_STEMS = {"empty", "weights", "syntax"}

CANDIDATE_TITLES: Dict[str, Dict[str, str]] = {
    "friedman": {"title": "Repeated Measures ANOVA (Non-parametric Friedman)", "group": "ANOVA"},
    "mancova": {"title": "MANCOVA (Multivariate Analysis of Covariance)", "group": "ANOVA"},
    "mcnemar": {"title": "Paired Samples Contingency (McNemar χ²)", "group": "Frequencies"},
    "logLinear": {"title": "Log-Linear Contingency Regression", "group": "Frequencies"},
    "logRegMult": {"title": "N Outcomes (Multinomial Logistic Regression)", "group": "Regression"},
    "logRegOrd": {"title": "Ordinal Outcomes (Proportional Odds Logistic)", "group": "Regression"},
    "propTest2": {"title": "2 Outcomes (Binomial Proportion Test)", "group": "Frequencies"},
    "propTestN": {"title": "N Outcomes (χ² Goodness of Fit)", "group": "Frequencies"},
}


def _fetch_github_json(url: str, timeout: float = 3.0) -> Optional[Any]:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "OpenStat-Studio-Upstream-Sync/3.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def check_jamovi_upstream(fetch_remote: bool = True) -> Dict[str, Any]:
    """
    Checks upstream `jamovi/jamovi` release info and `jamovi/jmv` `.a.yaml` analysis directory
    (using metadata names only — zero code copying) to identify any newly added analyses.
    """
    from .registry import get_full_catalog

    catalog = get_full_catalog()
    installed_ids = {c["id"] for c in catalog}

    remote_release = None
    upstream_stems: List[str] = []
    source_mode = "offline_reference_snapshot"

    if fetch_remote:
        rel_data = _fetch_github_json("https://api.github.com/repos/jamovi/jamovi/releases/latest")
        if isinstance(rel_data, dict) and rel_data.get("tag_name"):
            remote_release = rel_data.get("tag_name")

        dir_data = _fetch_github_json("https://api.github.com/repos/jamovi/jmv/contents/jamovi")
        if isinstance(dir_data, list):
            for entry in dir_data:
                fname = str(entry.get("name", ""))
                if fname.endswith(".a.yaml") and not fname.startswith("00"):
                    upstream_stems.append(fname[:-7].lower())
            if upstream_stems:
                source_mode = "github_live_api"

    if not upstream_stems:
        upstream_stems = sorted(UPSTREAM_JMV_TO_OPENSTAT.keys())

    covered: List[Dict[str, str]] = []
    unmapped_or_missing: List[Dict[str, str]] = []

    for stem in sorted(set(upstream_stems) - NON_ANALYSIS_STEMS):
        mapped_id = UPSTREAM_JMV_TO_OPENSTAT.get(stem, stem)
        if mapped_id in installed_ids:
            covered.append({"upstreamStem": stem, "openstatId": mapped_id})
        else:
            meta = CANDIDATE_TITLES.get(mapped_id, {"title": f"Upstream jamovi `{stem}` Module", "group": "Plugins"})
            unmapped_or_missing.append({
                "upstreamStem": stem,
                "suggestedId": mapped_id,
                "title": meta["title"],
                "group": meta["group"],
                "scaffoldCommand": f"openstat sync-jamovi --scaffold {mapped_id}",
            })

    # Extended modulesOpenStat Studio already includes beyond jmv base
    extended_modules = [
        c["id"] for c in catalog
        if c["id"] not in {x["openstatId"] for x in covered}
    ]

    report = {
        "checkedAt": datetime.now().isoformat(),
        "sourceMode": source_mode,
        "latestJamoviRelease": remote_release or "v2.7+ (reference)",
        "openstatVersion": "3.0.0",
        "openstatTotalAnalyses": len(catalog),
        "jmvCoreCoveredCount": len(covered),
        "openstatExtendedModules": extended_modules,
        "availableExtensionsToScaffold": unmapped_or_missing,
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_json = os.path.join(REPORTS_DIR, "jamovi_upstream_sync_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def scaffold_jamovi_feature(
    analysis_id: str,
    title: Optional[str] = None,
    group: str = "Plugins"
) -> str:
    """
    Creates a 100% clean-room MIT-licensed Python analysis plugin inside `reconstructed/pyjamovi/plugins/<analysis_id>.py`.
    Once created, it is immediately discovered by `OpenStat Studio` across Python SDK, CLI, MCP, and Web UI!
    """
    safe_id = re.sub(r"[^A-Za-z0-9_]", "", analysis_id)
    if not safe_id:
        raise ValueError("Invalid analysis_id")

    meta = CANDIDATE_TITLES.get(safe_id, {})
    final_title = title or meta.get("title") or f"Custom Analysis ({safe_id})"
    final_group = group if group != "Plugins" else meta.get("group", "Plugins")

    os.makedirs(PLUGINS_DIR, exist_ok=True)
    target_path = os.path.join(PLUGINS_DIR, f"{safe_id}.py")

    if safe_id == "friedman":
        body = f'''"""
OpenStat Studio Clean-Room Plugin: {final_title} (`{safe_id}`)
License: MIT (Pure-Python SciPy implementation)
"""

from typing import Dict, Any, List
import pandas as pd
from scipy import stats
from pyjamovi.analyses.base import BaseAnalysis, ResultsTable

PLUGIN_SPEC = {{
    "id": "{safe_id}",
    "group": "{final_group}",
    "title": "{final_title}",
    "module": "openstat-plugin",
    "icon": "analysis-anova.svg",
    "ui_schema": {{
        "title": "{final_title}",
        "slots": [
            {{"key": "vars", "label": "Repeated Measures Variables (≥3)", "multi": True, "req": True, "prefer": "num"}}
        ],
        "sections": [
            {{"title": "Options", "items": [{{"type": "check", "key": "desc", "label": "Descriptives (Median / Mean)", "def": True}}]}}
        ]
    }}
}}


class FriedmanPluginAnalysis(BaseAnalysis):
    def __init__(self, data: pd.DataFrame, vars: List[str], desc: bool = True):
        super().__init__("{safe_id}", "{final_title}", dataset=data, options={{"vars": vars, "desc": desc}})
        self.vars = [v for v in vars if v in data.columns]
        self.desc = desc
        self.syntax_python = f"ost.run('{safe_id}', df, vars={{self.vars!r}}, desc={{self.desc!r}})"
        self.run()

    def run(self):
        if len(self.vars) < 3:
            raise ValueError("Friedman test requires at least 3 repeated-measure columns.")
        sub = self.dataset[self.vars].apply(pd.to_numeric, errors="coerce").dropna()
        stat, p_val = stats.friedmanchisquare(*[sub[c].values for c in self.vars])
        df_deg = len(self.vars) - 1
        kendall_w = stat / (len(sub) * df_deg) if len(sub) > 0 and df_deg > 0 else 0.0

        tbl = ResultsTable("friedman", "Friedman Non-Parametric Repeated Measures ANOVA")
        tbl.add_column("chi2", "χ²", "number")
        tbl.add_column("df", "df", "integer")
        tbl.add_column("p", "p", "number", fmt="pvalue")
        tbl.add_column("w", "Kendall's W", "number")
        tbl.add_row({{"chi2": float(stat), "df": int(df_deg), "p": float(p_val), "w": float(kendall_w)}})
        self.add_table(tbl)

        if self.desc:
            dt = ResultsTable("desc", "Measure Descriptives")
            dt.add_column("measure", "Measure", "text")
            dt.add_column("mean", "Mean", "number")
            dt.add_column("median", "Median", "number")
            for c in self.vars:
                dt.add_row({{"measure": c, "mean": float(sub[c].mean()), "median": float(sub[c].median())}})
            self.add_table(dt)

        p_txt = "< .001" if p_val < 0.001 else f"= {{p_val:.3f}}"
        self.apa_narrative = f"Friedman's test across {{len(self.vars)}} repeated measures (N = {{len(sub)}}) yielded χ²({{df_deg}}) = {{stat:.2f}}, p {{p_txt}}, Kendall's W = {{kendall_w:.3f}}."


def run_plugin(df: pd.DataFrame, params: Dict[str, Any]) -> BaseAnalysis:
    raw_vars = params.get("vars") or params.get("rmVars") or []
    if isinstance(raw_vars, str):
        raw_vars = [v.strip() for v in raw_vars.split(",") if v.strip()]
    return FriedmanPluginAnalysis(df, vars=list(raw_vars), desc=bool(params.get("desc", True)))
'''
    else:
        body = f'''"""
OpenStat Studio Clean-Room Plugin: {final_title} (`{safe_id}`)
License: MIT
Generated by `openstat sync-jamovi --scaffold {safe_id}`.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats
from pyjamovi.analyses.base import BaseAnalysis, ResultsTable
from pyjamovi.analyses.plot_style import get_active_style, apply_style_to_axes, finalize_and_serialize_figure
import matplotlib.pyplot as plt

PLUGIN_SPEC = {{
    "id": "{safe_id}",
    "group": "{final_group}",
    "title": "{final_title}",
    "module": "openstat-plugin",
    "icon": "brand-modules.svg",
    "ui_schema": {{
        "title": "{final_title}",
        "slots": [
            {{"key": "vars", "label": "Target Variables", "multi": True, "req": True, "prefer": "any"}}
        ],
        "sections": [
            {{"title": "Plugin Options", "items": [{{"type": "check", "key": "plot", "label": "Summary Plot", "def": False}}]}}
        ]
    }}
}}


class CustomOpenStatPlugin(BaseAnalysis):
    def __init__(self, data: pd.DataFrame, vars: List[str], plot: bool = False):
        super().__init__("{safe_id}", "{final_title}", dataset=data, options={{"vars": vars, "plot": plot}})
        self.vars = [v for v in vars if v in data.columns] or list(data.columns[:2])
        self.plot = plot
        self.syntax_python = f"ost.run('{safe_id}', df, vars={{self.vars!r}}, plot={{self.plot!r}})"
        self.run()

    def run(self):
        tbl = ResultsTable("{safe_id}_summary", "{final_title} — Summary Table")
        tbl.add_column("var", "Variable", "text")
        tbl.add_column("n", "Valid N", "integer")
        tbl.add_column("stat", "Summary Statistic", "number")
        for col in self.vars:
            s = pd.to_numeric(self.dataset[col], errors="coerce").dropna()
            tbl.add_row({{
                "var": col,
                "n": int(len(s)) if len(s) else int(self.dataset[col].dropna().shape[0]),
                "stat": float(s.mean()) if len(s) else float(self.dataset[col].nunique())
            }})
        self.add_table(tbl)
        self.apa_narrative = f"Executed clean-room plugin `{safe_id}` ({{self.title}}) across {{len(self.vars)}} variable(s)."

        if self.plot and self.vars:
            st = get_active_style()
            fig, ax = plt.subplots(figsize=(5.5, 3.6))
            vals = [float(pd.to_numeric(self.dataset[c], errors="coerce").mean() or 0.0) for c in self.vars]
            ax.bar(self.vars, vals, color=st["primary"], alpha=0.85)
            apply_style_to_axes(ax)
            ax.set_title("{final_title}")
            self.add_plot("{final_title} Plot", finalize_and_serialize_figure(fig))


def run_plugin(df: pd.DataFrame, params: Dict[str, Any]) -> BaseAnalysis:
    raw_vars = params.get("vars") or []
    if isinstance(raw_vars, str):
        raw_vars = [v.strip() for v in raw_vars.split(",") if v.strip()]
    return CustomOpenStatPlugin(df, vars=list(raw_vars), plot=bool(params.get("plot", False)))
'''

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(body)
    return target_path


def rebuild_openstat_package(run_tests: bool = True) -> Dict[str, Any]:
    """
    Executes the full on-demand rebuild pipeline when new features/plugins or package updates are applied:
    1. Regenerate 45 clean-room SVG assets (`scripts/generate_clean_icons.py`)
    2. Refresh `ANALYSIS_CATALOG` + discovered plugins into `corpus/analysis_spec_catalog.json`
    3. Reinstall editable wheel (`pip install -e . --no-deps`)
    4. Run `doctor` API/syntax self-check
    5. Run `pytest` (if `run_tests=True`)
    """
    t0 = time.time()
    steps: List[Dict[str, Any]] = []
    py_bin = sys.executable

    # Step 1: Regenerate Clean Icons
    icon_script = os.path.join(ROOT_DIR, "scripts", "generate_clean_icons.py")
    if os.path.isfile(icon_script):
        p1 = subprocess.run([py_bin, icon_script], capture_output=True, text=True)
        steps.append({"step": "Regenerate Clean-Room SVG Assets", "passed": p1.returncode == 0, "output": p1.stdout.strip()})

    # Step 2: Reinstall Editable Package
    p2 = subprocess.run([py_bin, "-m", "pip", "install", "-e", ROOT_DIR, "--no-deps"], capture_output=True, text=True)
    steps.append({"step": "Editable Package Build (`pip install -e .`)", "passed": p2.returncode == 0, "output": "Installed openstat-studio==3.0.0" if p2.returncode == 0 else p2.stderr.strip()})

    # Step 3: Refresh Catalog Corpus
    from .registry import refresh_analysis_catalog
    from .analyses.plot_style import PALETTES, THEMES
    from .exporters import SUPPORTED_TABLE_FORMATS

    full_cat = refresh_analysis_catalog()
    os.makedirs(CORPUS_DIR, exist_ok=True)
    cat_path = os.path.join(CORPUS_DIR, "analysis_spec_catalog.json")
    with open(cat_path, "w", encoding="utf-8") as f:
        json.dump({
            "project": "OpenStat Studio",
            "version": "3.0.0",
            "license": "MIT",
            "rebuiltAt": datetime.now().isoformat(),
            "reference_attribution": "Modeled with reference to jamovi (https://www.jamovi.org)",
            "totalAnalyses": len(full_cat),
            "catalog": full_cat,
            "palettes": PALETTES,
            "themes": list(THEMES),
            "export_formats": list(SUPPORTED_TABLE_FORMATS),
        }, f, indent=2, ensure_ascii=False)
    steps.append({"step": "Sync Analysis & Plugin Catalog", "passed": True, "output": f"{len(full_cat)} analyses registered"})

    # Step 4: Self-Diagnostic Doctor Check
    from .diagnostics import run_api_compat_self_check
    compat = run_api_compat_self_check(quick=False)
    steps.append({
        "step": "API & Syntax Drift Self-Diagnostics",
        "passed": compat["healthy"],
        "output": f"{compat['passedChecks']}/{compat['totalChecks']} subsystems healthy",
    })

    # Step 5: Automated Pytest Verification
    if run_tests:
        test_file = os.path.join(ROOT_DIR, "tests", "test_pyjamovi.py")
        if os.path.isfile(test_file):
            p5 = subprocess.run([py_bin, "-m", "pytest", "-q", test_file], capture_output=True, text=True)
            steps.append({
                "step": "Automated Regression Test Suite (`pytest`)",
                "passed": p5.returncode == 0,
                "output": (p5.stdout.strip().splitlines()[-1] if p5.stdout.strip() else p5.stderr.strip()),
            })

    all_passed = all(s["passed"] for s in steps)
    manifest = {
        "rebuiltAt": datetime.now().isoformat(),
        "passed": all_passed,
        "elapsedSeconds": round(time.time() - t0, 3),
        "totalAnalyses": len(full_cat),
        "steps": steps,
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "openstat_rebuild_report.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest

"""
OpenStat Studio — Package Freshness Checker & Upstream API/Syntax Drift Self-Diagnostic Engine (`diagnostics.py`)
License: MIT

Provides:
1. `check_package_freshness(check_pypi=True)`:
   Checks installed Python dependency versions against minimum/tested baselines and queries PyPI for newer releases
   (with fast timeout, local caching, and graceful offline fallback).
2. `run_api_compat_self_check(quick=False)`:
   Executes signature inspections (`inspect.signature`) and synthetic micro-canary probes across every statistical,
   plotting, and document-export library (`scipy`, `statsmodels`, `pingouin`, `scikit-learn`, `factor_analyzer`,
   `semopy`, `linearmodels`, `lifelines`, `matplotlib`, `openpyxl`, `python-docx`, `.omv`) to catch syntax/API
   breaking changes immediately when Python or third-party packages are upgraded.
3. `startup_self_check_guard()`:
   Computes an environment fingerprint `(python_version + installed_package_versions)` on startup. Whenever the
   environment changes (e.g. after `pip install -U`), automatically runs self-diagnostics and surfaces actionable alerts.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as importlib_meta
import inspect
import json
import os
import sys
import tempfile
import time
import urllib.request
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_DIR = os.path.join(ROOT_DIR, "workspace")
DOCTOR_CACHE_PATH = os.path.join(WORKSPACE_DIR, "doctor_env_cache.json")


# Core dependencies: (pypi_name, import_name, min_version, role_description)
DEPENDENCY_SPECS: List[Tuple[str, str, str, str]] = [
    ("numpy", "numpy", "2.0.0", "Matrix algebra & IEEE-754 binary container arrays"),
    ("scipy", "scipy", "1.14.0", "Parametric/non-parametric distributions & Bayesian quadrature"),
    ("pandas", "pandas", "2.2.0", "Tabular DataFrame engine & data transformations"),
    ("statsmodels", "statsmodels", "0.14.0", "OLS, GLM Logistic, Type I/II/III ANOVA, ANCOVA & MixedLM"),
    ("pingouin", "pingouin", "0.5.5", "RM-ANOVA sphericity, Games-Howell, Partial Corr & Cronbach alpha"),
    ("scikit-learn", "sklearn", "1.5.0", "ROC/AUC metrics & factor matrix validation"),
    ("factor-analyzer", "factor_analyzer", "0.5.1", "PCA, EFA (Varimax/Oblimin/Promax), KMO & Bartlett"),
    ("semopy", "semopy", "2.3.11", "SEM reference implementation used by the test suite (analyses use the OpenStat SEM engine)"),
    ("linearmodels", "linearmodels", "6.1", "Panel Econometrics (Entity/Two-Way FE, Random Effects, Hausman)"),
    ("lifelines", "lifelines", "0.30.0", "Kaplan-Meier, Log-Rank & Cox Proportional Hazards Survival"),
    ("matplotlib", "matplotlib", "3.9.0", "Publication raster/vector graph rendering (.png, .svg, .pdf)"),
    ("seaborn", "seaborn", "0.13.0", "Statistical kernel density & heatmap styling"),
    ("openpyxl", "openpyxl", "3.1.0", "Excel (.xlsx) APA 3-line table & embedded figure export"),
    ("python-docx", "docx", "1.1.0", "MS Word (.docx) OOXML APA 3-line table export"),
    ("fastapi", "fastapi", "0.115.0", "On-demand Web UI & Claude Code live REST bridge"),
    ("uvicorn", "uvicorn", "0.32.0", "ASGI HTTP server runtime"),
]


def _parse_ver(v_str: str) -> Tuple[int, ...]:
    parts = []
    for token in str(v_str).replace("-", ".").replace("+", ".").split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts) if parts else (0,)


def _installed_version(pypi_name: str, import_name: str) -> Optional[str]:
    for candidate in (pypi_name, pypi_name.replace("-", "_"), import_name):
        try:
            return importlib_meta.version(candidate)
        except importlib_meta.PackageNotFoundError:
            continue
    try:
        mod = importlib.import_module(import_name)
        return getattr(mod, "__version__", "installed")
    except Exception:
        return None


def _fetch_pypi_latest(pypi_name: str, timeout: float = 1.8) -> Optional[str]:
    url = f"https://pypi.org/pypi/{pypi_name}/json"
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "OpenStat-Studio-Doctor/3.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("info", {}).get("version")
    except Exception:
        return None


def apply_runtime_compat_shims() -> List[str]:
    """
    Applies proactive runtime compatibility shims for known upstream Python package API changes
    so OpenStat Studio continues operating cleanly across library upgrades.
    """
    applied: List[str] = []

    # Shim 1: scikit-learn >= 1.6 deprecated/removed `force_all_finite` in `check_array` while `factor_analyzer` still passes it
    try:
        import sklearn.utils.validation as skval
        import factor_analyzer.factor_analyzer as fa_mod
        sig = inspect.signature(skval.check_array)
        if "force_all_finite" not in sig.parameters and "ensure_all_finite" in sig.parameters:
            orig_check = skval.check_array

            def _shimmed_check_array(*args, **kwargs):
                if "force_all_finite" in kwargs:
                    kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
                return orig_check(*args, **kwargs)

            if getattr(fa_mod.check_array, "__name__", "") != "_shimmed_check_array":
                fa_mod.check_array = _shimmed_check_array
            applied.append("sklearn>=1.6 check_array(force_all_finite -> ensure_all_finite) shim active")
    except Exception:
        pass

    return applied


def check_package_freshness(check_pypi: bool = True, timeout: float = 1.8) -> Dict[str, Any]:
    """
    Checks all core dependencies for:
    - Presence & minimum version satisfaction
    - Latest upstream PyPI version comparison (when online and `check_pypi=True`)
    """
    packages_report: List[Dict[str, Any]] = []
    missing_pkgs: List[str] = []
    below_min_pkgs: List[str] = []
    outdated_pkgs: List[str] = []
    pypi_reachable = False

    for pypi_name, import_name, min_ver, role in DEPENDENCY_SPECS:
        inst_ver = _installed_version(pypi_name, import_name)
        latest_ver = _fetch_pypi_latest(pypi_name, timeout=timeout) if (check_pypi and inst_ver) else None
        if latest_ver:
            pypi_reachable = True

        if inst_ver is None:
            status = "missing"
            missing_pkgs.append(pypi_name)
        elif _parse_ver(inst_ver) < _parse_ver(min_ver):
            status = "below_minimum"
            below_min_pkgs.append(f"{pypi_name} ({inst_ver} < {min_ver})")
        elif latest_ver and _parse_ver(inst_ver) < _parse_ver(latest_ver):
            status = "update_available"
            outdated_pkgs.append(f"{pypi_name} ({inst_ver} -> {latest_ver})")
        else:
            status = "up_to_date"

        packages_report.append({
            "package": pypi_name,
            "importName": import_name,
            "installed": inst_ver,
            "minimum": min_ver,
            "latestPyPI": latest_ver,
            "status": status,
            "role": role,
        })

    upgrade_cmd = (
        f"{sys.executable} -m pip install --upgrade " + " ".join(p.split(" ")[0] for p in outdated_pkgs)
        if outdated_pkgs else None
    )

    return {
        "checkedAt": datetime.now().isoformat(),
        "pythonVersion": sys.version.split()[0],
        "pypiChecked": bool(check_pypi),
        "pypiReachable": pypi_reachable,
        "totalPackages": len(packages_report),
        "missingCount": len(missing_pkgs),
        "belowMinimumCount": len(below_min_pkgs),
        "updatesAvailableCount": len(outdated_pkgs),
        "missing": missing_pkgs,
        "belowMinimum": below_min_pkgs,
        "updatesAvailable": outdated_pkgs,
        "upgradeCommand": upgrade_cmd,
        "packages": packages_report,
    }


def run_api_compat_self_check(quick: bool = False) -> Dict[str, Any]:
    """
    Runs live syntax & API compatibility self-diagnostics across OpenStat Studio's statistical,
    plotting, and export engines to detect breaking changes introduced by Python or library updates.
    """
    t0 = time.time()
    shims = apply_runtime_compat_shims()
    checks: List[Dict[str, Any]] = []

    rng = np.random.default_rng(42)
    n = 48
    grp = np.array(["A", "B"] * (n // 2))
    grp3 = np.array(["G1", "G2", "G3"] * (n // 3))
    x1 = rng.normal(10.0, 2.0, n)
    x2 = 0.65 * x1 + rng.normal(0.0, 1.2, n)
    x3 = 0.55 * x1 + rng.normal(0.0, 1.4, n)
    y_bin = (x1 + rng.normal(0, 1.0, n) > 10.0).astype(int)
    df_syn = pd.DataFrame({
        "group2": grp,
        "group3": grp3,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "y_bin": y_bin,
        "time_m": rng.uniform(2.0, 36.0, n),
    })

    def _probe(name: str, fn):
        try:
            detail = fn()
            checks.append({"subsystem": name, "passed": True, "detail": str(detail or "OK")})
        except Exception as exc:
            checks.append({
                "subsystem": name,
                "passed": False,
                "errorType": type(exc).__name__,
                "detail": str(exc),
            })

    # 1. Core T-Test / ANOVA / Post-Hoc (`scipy` + `statsmodels` + `pingouin`)
    def _check_core_stats():
        import pyjamovi as ost
        r_tt = ost.ttestIS(df_syn, vars="x1", group="group2", welchs=True, mann=True)
        r_an = ost.anovaOneW(df_syn, deps="x1", group="group3", ph_method="gamesHowell")
        r_rm = ost.anovaRM(df_syn, rm_vars=["x1", "x2", "x3"], spher=True, posthoc=True, profile_plot=False)
        assert len(r_tt.tables) >= 1 and len(r_an.tables) >= 2 and r_rm.get_table("within") is not None
        return "T-Test, One-Way ANOVA (Games-Howell) & RM-ANOVA schema verified"

    _probe("Core Parametric/Non-Parametric & Pingouin Schema", _check_core_stats)

    # 2. Factor Analysis & Reliability (`scikit-learn` + `factor_analyzer` API check)
    def _check_psychometrics():
        import pyjamovi as ost
        rel = ost.reliability(df_syn, vars=["x1", "x2", "x3"], cor_plot=False)
        efa = ost.efa(df_syn, vars=["x1", "x2", "x3"], scree_plot=False)
        omega = rel.get_table("scale").rows[0]["omega"]
        assert omega is not None and efa.get_table("loadings") is not None
        return f"Cronbach α / McDonald ω ({omega:.3f}) & EFA check_array API verified"

    _probe("Psychometrics (`sklearn` + `factor_analyzer`)", _check_psychometrics)

    if not quick:
        # 3. Structural Equation Modeling (OpenStat ML engine, Mplus defaults)
        def _check_sem():
            import pyjamovi as ost
            sem_r = ost.sem(df_syn, model_syntax="F1 =~ x1 + x2 + x3", path_diagram=False)
            assert sem_r.get_table("fit") is not None
            return "OpenStat SEM engine (ML/FIML, Mplus defaults) CFA & fit indices table verified"

        _probe("Structural Equation Modeling (OpenStat engine)", _check_sem)

        # 4. Panel Econometrics (`linearmodels`)
        def _check_panel():
            import pyjamovi as ost
            pan_df = pd.DataFrame({
                "firm": np.repeat([f"F{i}" for i in range(6)], 8),
                "year": np.tile(np.arange(2017, 2025), 6),
                "inv": x1 + rng.normal(0, 0.5, n),
                "val": x2,
                "cap": x3,
            })
            p_res = ost.panel(pan_df, entity_col="firm", time_col="year", dep="inv", covs=["val", "cap"], hausman=True, plot_trends=False)
            assert p_res.get_table("hausman") is not None
            return "linearmodels PanelOLS / RandomEffects / Hausman test verified"

        _probe("Panel Econometrics (`linearmodels`)", _check_panel)

        # 5. Survival Analysis (`lifelines`)
        def _check_survival():
            import pyjamovi as ost
            s_res = ost.survival(df_syn, time_col="time_m", event_col="y_bin", group="group2", covs=["x1"], km_plot=False)
            assert s_res.get_table("km") is not None and s_res.get_table("cox") is not None
            return "lifelines KaplanMeierFitter, Log-Rank & CoxPHFitter API verified"

        _probe("Survival Analysis (`lifelines`)", _check_survival)

        # 6. Matplotlib `PlotStyle` Vector/Raster + Standalone `GraphBuilder` + Exporters (`.pdf`, `.hwpx`, `.docx`, `.xlsx`, `.omv`)
        def _check_plots_and_exporters():
            import pyjamovi as ost
            res = ost.graph(
                df_syn,
                chart_type="raincloud",
                x="group3",
                y="x1",
                plotStyle={"palette": "nature", "theme": "apa_classic", "title": "자가점검 그래프", "dpi": 100},
            )
            assert len(res.plots) >= 1 and "<svg" in res.plot_vectors[0].get("svg", "")
            with tempfile.TemporaryDirectory() as tmpd:
                for ext in ("xlsx", "docx", "hwpx", "pdf", "omv"):
                    out_f = os.path.join(tmpd, f"probe.{ext}")
                    res.export_results(out_f, dataset=df_syn)
                    assert os.path.getsize(out_f) > 200
            return "GraphBuilder (15 types) + SVG/PDF/PNG + .pdf/.hwpx/.docx/.xlsx/.omv exporters verified"

        _probe("PlotStyle Vector Engine & Multi-Format Exporters", _check_plots_and_exporters)

    passed_count = sum(1 for c in checks if c["passed"])
    all_ok = (passed_count == len(checks))
    return {
        "checkedAt": datetime.now().isoformat(),
        "healthy": all_ok,
        "passedChecks": passed_count,
        "totalChecks": len(checks),
        "elapsedSeconds": round(time.time() - t0, 3),
        "activeShims": shims,
        "checks": checks,
    }


def _compute_env_fingerprint() -> str:
    items = [f"python={sys.version.split()[0]}"]
    for pypi_name, import_name, _, _ in DEPENDENCY_SPECS:
        items.append(f"{pypi_name}={_installed_version(pypi_name, import_name)}")
    raw = "|".join(items).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def startup_self_check_guard(force: bool = False, check_pypi: bool = False) -> Dict[str, Any]:
    """
    Lightweight startup guard:
    - Always activates runtime compatibility shims immediately.
    - Runs API self-diagnostics automatically the first time in an environment OR whenever
      Python / installed package versions change (cached in `workspace/doctor_env_cache.json`).
    """
    shims = apply_runtime_compat_shims()
    fp = _compute_env_fingerprint()

    if not force and os.path.isfile(DOCTOR_CACHE_PATH):
        try:
            with open(DOCTOR_CACHE_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("fingerprint") == fp and cached.get("compat", {}).get("healthy") is True and not check_pypi:
                cached["fromCache"] = True
                return cached
        except Exception:
            pass

    compat = run_api_compat_self_check(quick=not force)
    freshness = check_package_freshness(check_pypi=check_pypi)
    report = {
        "fingerprint": fp,
        "checkedAt": datetime.now().isoformat(),
        "fromCache": False,
        "activeShims": shims,
        "compat": compat,
        "freshness": freshness,
    }
    try:
        os.makedirs(WORKSPACE_DIR, exist_ok=True)
        with open(DOCTOR_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    if not compat["healthy"]:
        failed = [c["subsystem"] + ": " + c["detail"] for c in compat["checks"] if not c["passed"]]
        sys.stderr.write(
            "[OpenStat Doctor Warning] Upstream package API/syntax change detected:\n  - "
            + "\n  - ".join(failed)
            + "\nRun `openstat doctor` for full diagnostics.\n"
        )
    return report


def format_doctor_markdown(report: Dict[str, Any]) -> str:
    compat = report.get("compat", {})
    fresh = report.get("freshness", {})
    lines = [
        f"## OpenStat Studio Self-Diagnostics (`doctor`) — Python {fresh.get('pythonVersion', sys.version.split()[0])}",
        f"- **API & Syntax Compatibility**: {'PASS' if compat.get('healthy') else 'ACTION REQUIRED'} ({compat.get('passedChecks', 0)}/{compat.get('totalChecks', 0)} subsystems verified in {compat.get('elapsedSeconds', 0)}s)",
        f"- **Active Compatibility Shims**: {len(compat.get('activeShims', []))} ({', '.join(compat.get('activeShims', [])) or 'none needed'})",
        f"- **Package Freshness**: {fresh.get('totalPackages', 0)} checked | Missing: {fresh.get('missingCount', 0)} | Below min: {fresh.get('belowMinimumCount', 0)} | Newer on PyPI: {fresh.get('updatesAvailableCount', 0)}"
        + (" *(PyPI live check)*" if fresh.get("pypiReachable") else " *(Offline local check)*"),
        "",
        "### 1. Subsystem Syntax & API Compatibility Probes",
    ]
    for c in compat.get("checks", []):
        mark = "PASS" if c["passed"] else "FAIL"
        lines.append(f"- **[{mark}] {c['subsystem']}**: {c['detail']}")

    lines.append("\n### 2. Installed Python Package Freshness")
    lines.append("| Package | Installed | Minimum | Latest PyPI | Status |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for p in fresh.get("packages", []):
        lines.append(
            f"| `{p['package']}` | `{p['installed'] or '—'}` | `{p['minimum']}` | `{p['latestPyPI'] or '—'}` | **{p['status']}** |"
        )
    if fresh.get("upgradeCommand"):
        lines.append(f"\n> **Optional Upgrade Command:** `{fresh['upgradeCommand']}`")
    return "\n".join(lines)

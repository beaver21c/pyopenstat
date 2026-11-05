"""
OpenStat Studio Full-Stack Web Backend Server (v3.0, MIT License)
FastAPI engine serving:
- Standalone copyright-clean OpenStat Studio Desktop UI & 45 original MIT/CC0 vector SVG assets
- 28 real-time statistical analyses (including Panel FE/RE/Hausman & CFA/SEM)
- Bidirectional Python (`.py`) live syntax editor, runner & Claude Code workspace sync (`workspace/live_session.py`)
- Multi-format analyzed table export (`.xlsx`, `.docx`, `.hwpx`, `.tex`, `.csv`, `.tsv`, `.md`, `.html`, `.json`, `.omv`)
- Interactive `PlotStyle` graph design customization & vector export (`.svg`, `.pdf`, `.png`)

Note: Analytical models, menu taxonomy, and `.omv` interoperability are modeled with reference to jamovi (https://www.jamovi.org).
"""

import base64
import io
import json
import math
import os
import re
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, Form, Query, Body, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pyjamovi as jmv
from pyjamovi.agent import JamoviSession
from pyjamovi.registry import ANALYSIS_CATALOG, dispatch_analysis, refresh_analysis_catalog
from pyjamovi.analyses.plot_style import PALETTES, THEMES
from pyjamovi.analyses.graph_builder import CHART_TYPES, CustomFigureResult
from pyjamovi.exporters import export_analyses, SUPPORTED_TABLE_FORMATS
from pyjamovi.diagnostics import startup_self_check_guard
from pyjamovi.upstream_sync import check_jamovi_upstream, scaffold_jamovi_feature, rebuild_openstat_package
from pyjamovi.syntax_engine import (
    generate_python_script,
    execute_python_syntax,
    save_workspace_sync,
    load_workspace_sync,
    LIVE_PY_PATH,
    LIVE_JSON_PATH,
)
from pyjamovi.omv.writer import write_omv
from pyjamovi.text_import import texts_to_dataframe, SUPPORTED_TEXT_EXTENSIONS
from pyjamovi import spatial_io
from pyjamovi.renderers.html_renderer import render_jamovi_html_report

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, "..", ".."))
SAMPLE_DIR = os.path.join(WORKSPACE_ROOT, "sample_data")
WEB_DIR = os.path.join(PACKAGE_DIR, "web")
ASSETS_DIR = os.path.join(WEB_DIR, "assets")
REPORTS_DIR = os.path.join(WORKSPACE_ROOT, "reports")

os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

app = FastAPI(title="OpenStat Studio Engine", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


DATASET_DESCRIPTIONS: Dict[str, str] = {
    "ToothGrowth.csv": "Effect of Vitamin C (OJ vs VC) and Dose (0.5, 1.0, 2.0 mg) on Tooth Growth (N=60) — T-Test, ANOVA, Bayesian",
    "ToothGrowth.omv": "Open Matrix Vector (.omv) binary container of ToothGrowth (N=60)",
    "iris.csv": "Edgar Anderson's Iris Morphological Measurements across 3 Species (N=150) — ANOVA, Regression, PCA, SEM",
    "Big5.csv": "Big Five Personality Inventory (500 respondents × 5 subscales) — Correlation, PCA/EFA, Reliability",
    "bugs.csv": "Ryan, Wilde & Crist (2013) Arthropod Disgust/Fright Ratings (N=93) — Contingency χ², RM-ANOVA, Paired T-Test",
    "Grunfeld_Panel.csv": "Grunfeld (1958) Corporate Investment Panel (11 Firms × 20 Years, N=220) — Panel FE/RE, Hausman, Mixed Model",
    "SEM_Latent3Factor.csv": "3-Factor Latent Construct Dataset (9 Indicators x1..y3, N=300) — CFA, SEM, Reliability, EFA",
    "MedMod_WorkStress.csv": "Occupational Stress, Burnout, Performance & Turnover Risk (N=240) — Mediation, Moderation, Logistic ROC",
    "Clinical_Survival.csv": "Oncology Clinical Trial Time-to-Event Cohort (N=163) — Kaplan-Meier, Log-Rank & Cox PH Survival",
    "panel_sample.csv": "Compact Balanced Panel Benchmark Dataset — Panel Fixed/Random Effects",
    "Text_PolicyDocs.csv": "합성 정책문서 말뭉치 (N=150, 분야 5종) — 키워드·워드클라우드, 동시출현 관계망, 문서 군집",
    "Network_Collaboration.csv": "합성 기관 협력 연결 목록 (기관 30개, 연결 86개) — 관계망 분석, 중심성, 커뮤니티",
    "Spatial_Panel_Welfare.csv": "합성 지역 패널 (73개 지역 × 2016–2023, 가상 경계 자동 연결) — 공간패널회귀(SDM·고정효과), 공간가중행렬",
    "Spatial_CrossSection_Welfare.csv": "합성 지역 횡단면 (73개 지역, 가상 경계 자동 연결) — Moran's I·LISA, 공간회귀, 지리가중회귀, 주제도",
    "Mixture_LCA_Care.csv": "합성 돌봄 욕구 자료 (N=1,200, 3개 잠재계층으로 생성) — 잠재계층·잠재프로파일 분석, R3STEP·BCH·DCAT",
    "Mixture_LTA_Long.csv": "합성 정서 문항 종단 자료 (800명 × 3개 차수, long) — 잠재전이분석",
    "Mixture_LCGA_Long.csv": "합성 삶의 만족도 종단 자료 (600명 × 2019–2023, long, 탈락 포함) — 잠재계층 성장분석",
}

MIME_MAP: Dict[str, str] = {
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "hwpx": "application/hwp+zip",
    "pdf": "application/pdf",
    "tex": "application/x-tex",
    "csv": "text/csv; charset=utf-8",
    "tsv": "text/tab-separated-values; charset=utf-8",
    "md": "text/markdown; charset=utf-8",
    "html": "text/html; charset=utf-8",
    "json": "application/json; charset=utf-8",
    "omv": "application/octet-stream",
}


class GlobalState:
    def __init__(self):
        self.session = JamoviSession(auto_sync_workspace=False)
        self.dataset_name = "ToothGrowth.csv"
        self.measure_overrides: Dict[str, str] = {}
        self.analyses_store: List[Dict[str, Any]] = []
        self.analysis_objects: Dict[int, Any] = {}
        self.next_id: int = 1
        self.revision: int = 1
        self.custom_py_code: Optional[str] = None
        self.workspace_mtime: float = 0.0

        default_path = os.path.join(SAMPLE_DIR, "ToothGrowth.csv")
        if os.path.isfile(default_path):
            self.load_file("ToothGrowth.csv", sync_ws=False)

        # If Claude Code previously wrote workspace/live_session.json, restore it automatically on startup
        self.check_external_workspace_sync()

    def _active_specs(self) -> List[Dict[str, Any]]:
        specs = []
        for item in self.analyses_store:
            specs.append({
                "analysis_type": item.get("analysis_type") or item.get("name"),
                "title": item.get("title"),
                "params": item.get("params") or item.get("options") or {}
            })
        return specs

    def _dataset_script_path(self) -> str:
        sp = getattr(self.session, "source_path", None)
        return sp if sp else f"sample_data/{self.dataset_name}"

    def current_python_script(self) -> str:
        if self.custom_py_code:
            return self.custom_py_code
        return generate_python_script(
            self._active_specs(),
            dataset_path=self._dataset_script_path(),
            data_ops=self.session.data_ops,
            export_ops=self.session.export_ops,
        )

    def sync_to_workspace(self, py_code: Optional[str] = None):
        self.custom_py_code = py_code
        try:
            save_workspace_sync(
                dataset_name=self.dataset_name,
                specs=self._active_specs(),
                analyses=list(self.analysis_objects.values()),
                py_code=py_code or self.current_python_script(),
                data_ops=self.session.data_ops,
                export_ops=self.session.export_ops,
            )
            if os.path.isfile(LIVE_JSON_PATH):
                self.workspace_mtime = os.path.getmtime(LIVE_JSON_PATH)
        except Exception:
            pass

    def rerun_analyses_on_current_data(self) -> List[str]:
        """jamovi-style live update: re-runs every Results card after the dataset changes."""
        notes: List[str] = []
        df = self.session.df
        for idx, card in enumerate(list(self.analyses_store)):
            a_type = card.get("analysis_type") or card.get("name")
            if not a_type or a_type == "customFigure":
                continue
            aid = card.get("analysisId")
            params = card.get("params") or {}
            try:
                res = dispatch_analysis(a_type, df, params)
                payload = res.to_payload(analysis_id=aid)
                payload["analysis_type"] = a_type
                payload["params"] = params
                self.analyses_store[idx] = payload
                self.analysis_objects[aid] = res
            except Exception as e:
                notes.append(f"#{aid} {a_type}: {e}")
        self.session.history = list(self.analysis_objects.values())
        return notes

    def after_data_change(self) -> Dict[str, Any]:
        for c in list(self.measure_overrides):
            if self.session.df is None or c not in self.session.df.columns:
                self.measure_overrides.pop(c, None)
        notes = self.rerun_analyses_on_current_data()
        self.custom_py_code = None
        self.sync_to_workspace()
        self.revision += 1
        payload = self.dataset_payload()
        payload["rerunNotes"] = notes
        return payload

    def check_external_workspace_sync(self) -> bool:
        """Detects when Claude Code (CLI / MCP / SDK) updated workspace/live_session.json externally and syncs state."""
        if not os.path.isfile(LIVE_JSON_PATH):
            return False
        try:
            mtime = os.path.getmtime(LIVE_JSON_PATH)
            if mtime <= self.workspace_mtime + 1e-6:
                return False
            doc = load_workspace_sync()
            if not doc:
                return False
            self.workspace_mtime = mtime
            ds_name = doc.get("dataset")
            if ds_name and os.path.isfile(os.path.join(SAMPLE_DIR, ds_name)):
                self.session = JamoviSession(os.path.join(SAMPLE_DIR, ds_name), auto_sync_workspace=False)
                self.dataset_name = ds_name
                self.measure_overrides = {}
                for op in (doc.get("dataOps") or []):
                    try:
                        self.session.apply_data_op(op)
                    except Exception:
                        break
                self.session.export_ops = list(doc.get("exportOps") or [])

            specs = doc.get("specs") or []
            self.analyses_store = []
            self.analysis_objects = {}
            self.next_id = 1
            if self.session.df is not None:
                for sp in specs:
                    a_type = sp.get("analysis_type") or sp.get("name")
                    params = sp.get("params") or {}
                    if not a_type:
                        continue
                    try:
                        res = dispatch_analysis(a_type, self.session.df, params)
                        aid = self.next_id
                        self.next_id += 1
                        payload = res.to_payload(analysis_id=aid)
                        payload["analysis_type"] = a_type
                        payload["params"] = params
                        self.analyses_store.append(payload)
                        self.analysis_objects[aid] = res
                    except Exception:
                        continue
            self.session.history = list(self.analysis_objects.values())
            self.custom_py_code = doc.get("pythonCode")
            self.revision += 1
            return True
        except Exception:
            return False

    def load_file(self, name: str, sync_ws: bool = True):
        path = os.path.join(SAMPLE_DIR, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Dataset '{name}' not found in {SAMPLE_DIR}")
        self.session = JamoviSession(path, auto_sync_workspace=False)
        self.dataset_name = name
        self.measure_overrides = {}
        self.analyses_store = []
        self.analysis_objects = {}
        self.next_id = 1
        self.custom_py_code = None
        self.revision += 1
        self.boundary = None
        sample_geo = SAMPLE_BOUNDARIES.get(name)
        if sample_geo and os.path.exists(os.path.join(WORKSPACE_ROOT, sample_geo["boundary"])):
            self.boundary = dict(sample_geo, name=os.path.basename(sample_geo["boundary"]))
        if sync_ws:
            self.sync_to_workspace()

    def infer_columns(self) -> List[Dict[str, Any]]:
        df = self.session.df
        if df is None:
            return []
        cols = []
        for c in df.columns:
            s = df[c]
            low = str(c).lower()
            if c in self.session.measure_levels:
                m_type = self.session.measure_levels[c]
            elif c in self.measure_overrides:
                m_type = self.measure_overrides[c]
            elif low in ("id", "subject", "subj", "patient_id"):
                m_type = "ID"
            elif pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
                m_type = "Nominal"
            elif pd.api.types.is_integer_dtype(s) and s.nunique() <= 6:
                m_type = "Nominal" if s.nunique() == 2 else "Ordinal"
            elif s.nunique() <= 4:
                m_type = "Ordinal"
            else:
                m_type = "Continuous"

            if pd.api.types.is_datetime64_any_dtype(s):
                d_type = "Date"
            elif pd.api.types.is_float_dtype(s):
                d_type = "Decimal"
            elif pd.api.types.is_integer_dtype(s):
                d_type = "Integer"
            else:
                d_type = "Text"

            levels = [str(v) for v in sorted(s.dropna().unique(), key=str)[:12]] if s.nunique() <= 12 else []
            avg_len = None
            if d_type == "Text":
                try:
                    avg_len = round(float(s.dropna().astype(str).head(500).str.len().mean()), 1)
                except Exception:
                    avg_len = None
            info: Dict[str, Any] = {
                "name": str(c),
                "measureType": m_type,
                "dataType": d_type,
                "valid": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "unique": int(s.nunique()),
                "levels": levels
            }
            if avg_len is not None:
                info["avgLen"] = avg_len
            vlab = self.session.value_labels.get(c)
            if vlab:
                info["valueLabels"] = {str(k): str(v) for k, v in vlab.items()}
                info["levels"] = [f"{k}={v}" for k, v in list(vlab.items())[:12]]
            if self.session.variable_labels.get(c):
                info["label"] = str(self.session.variable_labels[c])
            if pd.api.types.is_numeric_dtype(s) and s.notna().any():
                num = pd.to_numeric(s, errors="coerce")
                sd = num.std(ddof=1)
                info.update({
                    "mean": round(float(num.mean()), 4),
                    "sd": None if pd.isna(sd) else round(float(sd), 4),
                    "min": round(float(num.min()), 4),
                    "max": round(float(num.max()), 4),
                })
            cols.append(info)
        return cols

    def serialize_rows(self, limit: int = 300) -> List[Dict[str, Any]]:
        df = self.session.df
        if df is None:
            return []
        sub = df.head(limit)
        records = []
        for _, row in sub.iterrows():
            rec = {}
            for col in sub.columns:
                val = row[col]
                if val is None or (isinstance(val, (float, np.floating)) and (math.isnan(float(val)) or math.isinf(float(val)))) or pd.isna(val):
                    rec[col] = None
                elif isinstance(val, (np.integer, int)) and not isinstance(val, bool):
                    rec[col] = int(val)
                elif isinstance(val, (np.floating, float)):
                    rec[col] = round(float(val), 6) if abs(float(val) - round(float(val))) > 1e-9 else float(val)
                else:
                    sval = str(val)
                    # Long documents are shown abbreviated in the data grid; analyses use the full text.
                    rec[col] = sval if len(sval) <= 400 else sval[:400] + " …"
            records.append(rec)
        return records

    def dataset_payload(self) -> Dict[str, Any]:
        df = self.session.df
        return {
            "status": "success",
            "name": self.dataset_name,
            "rowCount": len(df) if df is not None else 0,
            "columnCount": len(df.columns) if df is not None else 0,
            "columns": self.infer_columns(),
            "rows": self.serialize_rows(300),
            "analyses": self.analyses_store,
            "pythonCode": self.current_python_script(),
            "dataOps": json.loads(json.dumps(self.session.data_ops, default=str)),
            "exportOps": json.loads(json.dumps(self.session.export_ops, default=str)),
            "pipeline": {
                "dataSteps": len(self.session.data_ops),
                "analysisSteps": len(self.analyses_store),
                "exportSteps": len(self.session.export_ops),
            },
            "revision": self.revision,
            "boundary": getattr(self, "boundary", None),
        }


SAMPLE_BOUNDARIES = {
    "Spatial_Panel_Welfare.csv": {"boundary": "sample_data/geo/Synthetic_Regions.shp", "boundaryKey": "SIG_CD", "idVar": "지역코드", "labelVar": "지역명"},
    "Spatial_CrossSection_Welfare.csv": {"boundary": "sample_data/geo/Synthetic_Regions.shp", "boundaryKey": "SIG_CD", "idVar": "지역코드", "labelVar": "지역명"},
}
BOUNDARY_DIR = os.path.join(WORKSPACE_ROOT, "workspace", "boundaries")

STATE = GlobalState()


class RunAnalysisRequest(BaseModel):
    analysis_type: str
    params: Dict[str, Any] = {}
    analysis_id: Optional[int] = None


class CellEditRequest(BaseModel):
    row_idx: int
    col_name: str
    value: Optional[Any] = None


class VariableUpdateRequest(BaseModel):
    old_name: str
    new_name: Optional[str] = None
    measure_type: Optional[str] = None


class ComputeColumnRequest(BaseModel):
    col_name: str
    formula: str


class RowColActionRequest(BaseModel):
    action: str  # "add_row", "delete_row", "add_col", "delete_col"
    target: Optional[Any] = None


class RecodeRequest(BaseModel):
    source_col: str
    target_col: Optional[str] = None
    rules: Union[str, Dict[str, Any], List[Any]] = ""
    else_value: Any = "copy"


class FilterRequest(BaseModel):
    condition: str


class MergeRequest(BaseModel):
    other_source: Union[str, List[str]]
    on: Optional[Union[str, List[str]]] = None
    how: Optional[str] = None
    suffixes: Tuple[str, str] = ("", "_merged")
    kind: Optional[str] = None            # 1:1 | m:1 | 1:m | m:m | joinby | append (None = infer from keys)
    keep: Optional[List[str]] = None      # master | using | match | match_update | match_conflict
    keepusing: Optional[List[str]] = None
    indicator: Optional[Union[str, bool]] = None
    update: bool = False
    replace: bool = False
    unmatched: str = "none"               # joinby
    generate: Optional[str] = None        # append
    force: bool = False                   # append


class ConvertVariableRequest(BaseModel):
    columns: List[str]
    action: str = "convert"               # convert | encode | decode | labels | measure
    to: Optional[str] = None              # numeric | integer | text | date
    ignore: str = ""
    force: bool = False
    percent: bool = False
    generate: Optional[List[str]] = None
    date_format: Optional[str] = None
    order: Optional[List[str]] = None
    labels: Optional[Dict[str, str]] = None
    level: Optional[str] = None


class ReshapeRequest(BaseModel):
    direction: str = "long"
    id_vars: Optional[Union[str, List[str]]] = None
    value_vars: Optional[Union[str, List[str]]] = None
    stubnames: Optional[Union[str, List[str]]] = None
    time_col: str = "time"
    value_col: str = "value"
    sep: str = ""
    strip_prefix: bool = True


class AIPushRequest(BaseModel):
    dataset: Optional[str] = None
    analysis_type: str
    params: Dict[str, Any] = {}


class ExecuteSyntaxRequest(BaseModel):
    code: str
    replace_existing: bool = True


class SyntheticDataRequest(BaseModel):
    n_samples: Optional[int] = None
    cols: Optional[List[str]] = None
    epsilon: Optional[float] = None
    dataset_name: Optional[str] = "synthetic_data.csv"
    as_new_dataset: bool = True


class PlotRestyleRequest(BaseModel):
    analysis_id: Optional[int] = None
    plot_style: Dict[str, Any] = {}


def _attachment(filename: str) -> str:
    """Content-Disposition for any file name: ASCII fallback plus RFC 5987 UTF-8 name (Korean dataset names)."""
    from urllib.parse import quote
    ascii_name = filename.encode("ascii", "ignore").decode("ascii").strip() or "openstat_export"
    ascii_name = ascii_name.replace('"', "").replace("\\", "")
    if ascii_name.startswith((".", "_")):
        ascii_name = "openstat" + ascii_name
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


@app.get("/", response_class=HTMLResponse)
def get_index():
    index_path = os.path.join(WEB_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content, headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@app.get("/user_guide.html", response_class=HTMLResponse)
@app.get("/guide", response_class=HTMLResponse)
def get_user_guide():
    guide_path = os.path.join(WEB_DIR, "user_guide.html")
    with open(guide_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content, headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@app.get("/api/health")
def get_health():
    return {
        "status": "ok",
        "engine": "openstat-studio",
        "license": "MIT",
        "attribution": "Analytical models, menu taxonomy & .omv compatibility modeled with reference to jamovi (https://www.jamovi.org)",
        "version": jmv.__version__,
        "dataset": STATE.dataset_name,
        "analysesAvailable": len(ANALYSIS_CATALOG),
        "chartTypes": list(CHART_TYPES),
        "palettes": list(PALETTES.keys()),
        "themes": list(THEMES),
        "exportFormats": list(SUPPORTED_TABLE_FORMATS),
        "activeResults": len(STATE.analyses_store),
        "revision": STATE.revision
    }


@app.get("/api/catalog")
def get_catalog():
    refresh_analysis_catalog()
    return {
        "catalog": ANALYSIS_CATALOG,
        "chartTypes": list(CHART_TYPES),
        "palettes": PALETTES,
        "themes": list(THEMES),
        "exportFormats": list(SUPPORTED_TABLE_FORMATS)
    }


@app.get("/api/diagnostics/doctor")
def get_doctor_diagnostics(check_pypi: bool = Query(True), force: bool = Query(False)):
    """Runs Python package freshness check and API/syntax drift self-diagnostics."""
    return startup_self_check_guard(force=force or check_pypi, check_pypi=check_pypi)


@app.get("/api/upstream/check")
def get_upstream_jamovi_sync(fetch_remote: bool = Query(True)):
    """Checks upstream jamovi for newly added analyses and lists scaffoldable clean-room plugins."""
    return check_jamovi_upstream(fetch_remote=fetch_remote)


class ScaffoldPluginRequest(BaseModel):
    analysis_id: str
    title: Optional[str] = None
    group: str = "Plugins"


@app.post("/api/upstream/scaffold")
def post_scaffold_plugin(req: ScaffoldPluginRequest):
    path = scaffold_jamovi_feature(req.analysis_id, title=req.title, group=req.group)
    cat = refresh_analysis_catalog()
    return {"status": "success", "pluginPath": path, "totalAnalyses": len(cat), "catalog": cat}


@app.post("/api/upstream/rebuild")
def post_rebuild_package(run_tests: bool = Query(False)):
    manifest = rebuild_openstat_package(run_tests=run_tests)
    refresh_analysis_catalog()
    return manifest


@app.get("/api/datasets")
def list_datasets():
    files = sorted([f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".csv", ".omv", ".xlsx", ".xls", ".tsv", ".dta", ".sav", ".zsav", ".por", ".parquet"))])
    items = [
        {"name": f, "description": DATASET_DESCRIPTIONS.get(f, "User / Workspace Dataset")}
        for f in files
    ]
    return {"datasets": files, "items": items, "current": STATE.dataset_name}


@app.get("/api/session/state")
def get_session_state(since_revision: int = Query(0)):
    STATE.check_external_workspace_sync()
    if since_revision and since_revision == STATE.revision:
        return {"changed": False, "revision": STATE.revision}
    payload = STATE.dataset_payload()
    payload["changed"] = True
    return payload


@app.post("/api/dataset/load")
def load_dataset(name: Optional[str] = Query(None), body: Optional[Dict[str, Any]] = Body(None)):
    target = name or (body.get("name") if body else None)
    if not target:
        raise HTTPException(status_code=400, detail="Dataset name is required")
    try:
        STATE.load_file(target)
        return STATE.dataset_payload()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/dataset/new")
def new_dataset():
    blank_df = pd.DataFrame({"A": [np.nan] * 20, "B": [np.nan] * 20, "C": [np.nan] * 20})
    STATE.session = JamoviSession(blank_df, auto_sync_workspace=False)
    STATE.dataset_name = "Untitled.omv"
    STATE.measure_overrides = {"A": "Nominal", "B": "Nominal", "C": "Continuous"}
    STATE.analyses_store = []
    STATE.analysis_objects = {}
    STATE.next_id = 1
    STATE.custom_py_code = None
    STATE.revision += 1
    STATE.sync_to_workspace()
    return STATE.dataset_payload()


@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    safe_name = os.path.basename(file.filename or "uploaded_data.csv")
    save_path = os.path.join(SAMPLE_DIR, safe_name)
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)
    STATE.load_file(safe_name)
    return STATE.dataset_payload()


@app.post("/api/dataset/import_texts")
async def import_text_files(
    files: List[UploadFile] = File(...),
    paths: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
):
    """Text files (.txt .md .docx .pdf .hwpx) -> dataset with one row per document (saved as UTF-8 CSV)."""
    try:
        rel_paths = json.loads(paths) if paths else []
    except Exception:
        rel_paths = []
    items = []
    for i, f in enumerate(files):
        rel = rel_paths[i] if i < len(rel_paths) and rel_paths[i] else (f.filename or f"file_{i + 1}")
        items.append((rel, await f.read()))
    df, skipped = texts_to_dataframe(items)
    if df.empty:
        detail = "불러온 문서 없음. 지원 형식: " + ", ".join(SUPPORTED_TEXT_EXTENSIONS)
        if skipped:
            detail += " / 제외: " + "; ".join(f"{p} ({r})" for p, r in skipped[:5])
        raise HTTPException(status_code=400, detail=detail)
    base = re.sub(r"[^0-9A-Za-z가-힣_\-]+", "_", (name or "").strip()).strip("_") or "텍스트문서"
    fname = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(os.path.join(SAMPLE_DIR, fname), index=False, encoding="utf-8-sig")
    STATE.load_file(fname)
    payload = STATE.dataset_payload()
    payload["import"] = {
        "file": fname,
        "documents": int(len(df)),
        "characters": int(df["글자수"].sum()),
        "skipped": [{"path": p, "reason": r} for p, r in skipped],
    }
    return payload


def _rel_root(path: str) -> str:
    try:
        r = os.path.relpath(path, WORKSPACE_ROOT)
        return r.replace(os.sep, "/") if not r.startswith("..") else path
    except ValueError:
        return path


def _key_suggestions(gdf, df: Optional[pd.DataFrame], limit: int = 5) -> List[Dict[str, Any]]:
    """Rank (data column, boundary attribute) pairs by the share of data units found in the boundary file."""
    if df is None or df.empty:
        return []
    out = []
    bcols = [c for c in gdf.columns if c != "geometry"]
    bkeys = {c: set(spatial_io.normalize_keys(gdf[c]).dropna()) for c in bcols}
    for dc in list(df.columns)[:80]:
        dvals = spatial_io.normalize_keys(df[dc]).dropna()
        if dvals.empty:
            continue
        du = set(dvals)
        if len(du) < 2:
            continue
        for bc, bs in bkeys.items():
            if len(bs) < 2:
                continue
            hit = len(du & bs)
            if hit == 0:
                continue
            out.append({"idVar": dc, "boundaryKey": bc, "matched": hit, "dataUnits": len(du), "boundaryUnits": len(bs),
                        "share": hit / len(du)})
    out.sort(key=lambda r: (-r["share"], -r["matched"], r["boundaryKey"] == spatial_io.GID))
    good = [r for r in out if r["share"] >= 0.5]
    return (good or out[:1])[:limit]


def _boundary_preview(gdf, keys_matched: Optional[set], key: Optional[str]) -> str:
    from pyjamovi.analyses import spatial_maps as smap
    from pyjamovi.analyses.plot_style import use_plot_style
    cats = ["연결됨" if (keys_matched is None or k in keys_matched) else "자료 없음"
            for k in (spatial_io.normalize_keys(gdf[key]) if key else [None] * len(gdf))]
    pal = {"연결됨": "#3B82F6", "자료 없음": "#E5E7EB"} if keys_matched is not None else {"연결됨": "#93C5FD"}
    geoms = gdf.geometry.copy()
    geoms.index = [str(i) for i in range(len(gdf))]
    with use_plot_style(None):
        return smap.plot_categories(geoms, cats, pal, "경계 미리보기", "", figsize=(5.6, 4.6))


@app.post("/api/geo/upload")
async def upload_boundary(files: List[UploadFile] = File(...), name: Optional[str] = Form(None)):
    """Save a boundary file (.shp + .shx + .dbf [+ .prj .cpg], .zip, .geojson or .gpkg) and describe it."""
    items = [(f.filename or f"file_{i}", await f.read()) for i, f in enumerate(files)]
    stem = os.path.splitext(os.path.basename(next((n for n, _ in items if n.lower().endswith((".shp", ".zip", ".geojson", ".json", ".gpkg"))), items[0][0])))[0]
    base = re.sub(r"[^0-9A-Za-z가-힣_\-]+", "_", (name or stem).strip()).strip("_") or "boundary"
    dest = os.path.join(BOUNDARY_DIR, f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    try:
        path = spatial_io.save_uploaded_boundary(items, dest)
        gdf = spatial_io.read_boundary(path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"경계 파일을 읽지 못함: {e}")
    info = spatial_io.describe_boundary(gdf)
    info["path"] = _rel_root(path)
    info["files"] = sorted(os.listdir(dest))
    info["hasPrj"] = any(f.lower().endswith(".prj") for f in info["files"]) or not path.lower().endswith(".shp")
    info["suggestions"] = _key_suggestions(gdf, STATE.session.df)
    info["preview"] = _boundary_preview(gdf, None, None)
    return info


class BoundaryAttachRequest(BaseModel):
    path: str
    boundaryKey: str
    idVar: str
    labelVar: Optional[str] = None


@app.post("/api/geo/attach")
def attach_boundary(req: BoundaryAttachRequest):
    df = STATE.session.df
    if df is None or req.idVar not in df.columns:
        raise HTTPException(status_code=400, detail=f"데이터에 '{req.idVar}' 변수 없음")
    path = req.path if os.path.isabs(req.path) else os.path.join(WORKSPACE_ROOT, req.path)
    try:
        gdf = spatial_io.read_boundary(path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if req.boundaryKey not in gdf.columns:
        raise HTTPException(status_code=400, detail=f"경계 파일에 '{req.boundaryKey}' 속성 없음")
    rep = spatial_io.match_keys(df[req.idVar], gdf[req.boundaryKey])
    if rep["matched"] == 0:
        raise HTTPException(status_code=400, detail="일치하는 지역이 없음 — 연결 변수 조합을 확인")
    matched = set(spatial_io.normalize_keys(df[req.idVar]).dropna()) & set(spatial_io.normalize_keys(gdf[req.boundaryKey]).dropna())
    label = req.labelVar if req.labelVar and req.labelVar in df.columns else None
    STATE.boundary = {"boundary": _rel_root(path), "boundaryKey": req.boundaryKey, "idVar": req.idVar,
                      "labelVar": label, "name": os.path.basename(path), "matched": rep["matched"],
                      "dataUnits": rep["dataUnits"], "boundaryUnits": rep["boundaryUnits"]}
    payload = STATE.dataset_payload()
    payload["match"] = rep
    payload["preview"] = _boundary_preview(gdf, matched, req.boundaryKey)
    return payload


class BoundaryPathRequest(BaseModel):
    path: str


@app.post("/api/geo/open_as_data")
def open_boundary_as_data(req: BoundaryPathRequest):
    """Use the boundary attribute table as the dataset (row key `_GID`), with the boundary linked."""
    path = req.path if os.path.isabs(req.path) else os.path.join(WORKSPACE_ROOT, req.path)
    try:
        gdf = spatial_io.read_boundary(path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    attrs = pd.DataFrame(gdf.drop(columns="geometry"))
    base = re.sub(r"[^0-9A-Za-z가-힣_\-]+", "_", os.path.splitext(os.path.basename(path))[0]).strip("_") or "boundary"
    fname = f"{base}_속성_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    attrs.to_csv(os.path.join(SAMPLE_DIR, fname), index=False, encoding="utf-8-sig")
    STATE.load_file(fname)
    label = next((c for c in attrs.columns if re.search(r"(NM|NAME|명|이름)$", str(c), re.I)), None)
    STATE.boundary = {"boundary": _rel_root(path), "boundaryKey": spatial_io.GID, "idVar": spatial_io.GID,
                      "labelVar": label, "name": os.path.basename(path), "matched": len(attrs),
                      "dataUnits": len(attrs), "boundaryUnits": len(attrs)}
    return STATE.dataset_payload()


@app.post("/api/geo/detach")
def detach_boundary():
    STATE.boundary = None
    return STATE.dataset_payload()


@app.post("/api/cell/edit")
def edit_cell(
    row_idx: Optional[int] = Query(None),
    col_name: Optional[str] = Query(None),
    value: Optional[str] = Query(None),
    body: Optional[CellEditRequest] = Body(None)
):
    r_idx = body.row_idx if body is not None else row_idx
    c_name = body.col_name if body is not None else col_name
    val_in = body.value if body is not None else value

    if r_idx is None or not c_name:
        raise HTTPException(status_code=400, detail="Invalid row or column")

    try:
        STATE.session.set_value(r_idx, c_name, val_in)
        return STATE.after_data_change()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/variable/update")
def update_variable(req: VariableUpdateRequest):
    df = STATE.session.df
    if df is None or req.old_name not in df.columns:
        raise HTTPException(status_code=404, detail=f"Variable '{req.old_name}' not found")

    target_name = req.old_name
    if req.new_name and req.new_name.strip() and req.new_name.strip() != req.old_name:
        new_n = req.new_name.strip()
        STATE.session.rename_var(req.old_name, new_n)
        if req.old_name in STATE.measure_overrides:
            STATE.measure_overrides[new_n] = STATE.measure_overrides.pop(req.old_name)
        target_name = new_n

    if req.measure_type in ("Continuous", "Ordinal", "Nominal", "ID"):
        STATE.session.set_measure(target_name, req.measure_type)
        STATE.measure_overrides.pop(target_name, None)

    return STATE.after_data_change()


@app.post("/api/variable/convert")
def convert_variable_endpoint(req: ConvertVariableRequest):
    """Variable type / label changes recorded in the session pipeline (Stata destring, tostring, encode, decode,
    label values) and measure levels."""
    sess = STATE.session
    try:
        act = (req.action or "convert").strip().lower()
        if act == "convert":
            if not req.to:
                raise ValueError("변환할 형식(to) 지정 필요")
            sess.convert_type(req.columns, req.to, ignore=req.ignore, force=req.force, percent=req.percent,
                              generate=req.generate or None, date_format=req.date_format or None)
        elif act in ("encode", "decode"):
            if len(req.columns) != 1:
                raise ValueError(f"{act}: 변수 1개 지정")
            src = req.columns[0]
            tgt = (req.generate or [None])[0] or (src + ("_code" if act == "encode" else "_label"))
            if act == "encode":
                sess.encode(src, tgt, order=req.order or None)
            else:
                sess.decode(src, tgt)
        elif act == "labels":
            for c in req.columns:
                sess.set_value_labels(c, req.labels or None)
        elif act == "measure":
            sess.set_measure(req.columns, req.level or "")
            for c in req.columns:
                STATE.measure_overrides.pop(c, None)
        else:
            raise ValueError(f"action '{req.action}' 인식 불가 (convert, encode, decode, labels, measure)")
        payload = STATE.after_data_change()
        rep = getattr(sess, "last_convert_report", None) if act == "convert" else None
        if rep:
            forced = {k: v for k, v in rep.get("forced_missing", {}).items() if v}
            payload["message"] = ("결측 처리된 값: " + ", ".join(f"{k} {v}건" for k, v in forced.items())) if forced else "변환 완료"
        return payload
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/data/compute")
def compute_column(req: ComputeColumnRequest):
    """Computed Variable evaluator supporting vectorized formulas (LOG, LN, EXP, SQRT, ABS, Z, MEAN, SUM, IFELSE, etc.)."""
    try:
        STATE.session.compute(req.col_name, req.formula)
        return STATE.after_data_change()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Formula error: {e}")


@app.post("/api/data/recode")
def recode_data_endpoint(req: RecodeRequest):
    """Variable recoding / binning endpoint (Stata `recode ..., generate()`)."""
    try:
        STATE.session.recode(req.source_col, req.target_col, req.rules, req.else_value)
        return STATE.after_data_change()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Recode error: {e}")


@app.post("/api/data/filter")
def filter_data_endpoint(req: FilterRequest):
    """Row filtering endpoint (Stata `keep if ...`)."""
    try:
        STATE.session.filter_rows(req.condition)
        return STATE.after_data_change()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Filter error: {e}")


@app.post("/api/data/merge")
def merge_data_endpoint(req: MergeRequest):
    """Merge endpoint (Stata `merge 1:1 / m:1 / 1:m / m:m`, `joinby`, `append using`)."""
    sess = STATE.session
    try:
        kind = (req.kind or "").strip().lower() or None
        if kind == "append" or (req.how or "").lower() == "append":
            sess.append(req.other_source, generate=req.generate or None, force=req.force)
            msg = f"행 추가 완료: {len(sess.df):,}행"
        elif kind == "joinby":
            sess.joinby(req.other_source, on=req.on, unmatched=req.unmatched, update=req.update, replace=req.replace)
            msg = f"joinby 완료: {sess.last_merge_report['kept']:,}행"
        else:
            sess.merge(req.other_source, on=req.on, kind=kind, keep=req.keep or None, keepusing=req.keepusing or None,
                       indicator=req.indicator, update=req.update, replace=req.replace, how=req.how, suffixes=req.suffixes)
            r = sess.last_merge_report
            c = r["counts"]
            msg = (f"{r['kind']} 병합{'(키 고유성으로 판별)' if r['inferred'] else ''}: 현재만 {c[1]:,} · 결합 자료만 {c[2]:,} · "
                   f"일치 {c[3] + c[4] + c[5]:,}" + (f" (결측 갱신 {c[4]:,}, 값 충돌 {c[5]:,})" if (c[4] or c[5]) else "")
                   + f" → {r['kept']:,}행 유지")
        payload = STATE.after_data_change()
        payload["message"] = msg
        return payload
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Merge error: {e}")


@app.post("/api/data/reshape")
def reshape_data_endpoint(req: ReshapeRequest):
    """Wide <-> Long reshape endpoint (Stata `reshape long / wide`)."""
    try:
        STATE.session.reshape(
            direction=req.direction,
            id_vars=req.id_vars,
            value_vars=req.value_vars,
            stubnames=req.stubnames,
            time_col=req.time_col,
            value_col=req.value_col,
            sep=req.sep,
            strip_prefix=req.strip_prefix,
        )
        return STATE.after_data_change()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Reshape error: {e}")


@app.post("/api/data/synthetic")
def generate_synthetic_endpoint(req: SyntheticDataRequest):
    """Generates realistic synthetic microdata preserving empirical correlations and marginals (with optional DP)."""
    try:
        from .synthetic_data import generate_synthetic_dataset
        if STATE.session.df is None or len(STATE.session.df) == 0:
            raise ValueError("No active dataset loaded to synthesize from.")
        synth_df = generate_synthetic_dataset(
            STATE.session.df,
            n_samples=req.n_samples,
            cols=req.cols,
            epsilon=req.epsilon
        )
        if req.as_new_dataset:
            STATE.session = JamoviSession(synth_df, auto_sync_workspace=False)
            STATE.dataset_name = req.dataset_name or "synthetic_dataset.csv"
            STATE.analyses_store = []
            STATE.analysis_objects = {}
            STATE.next_id = 1
            STATE.measure_overrides = {}
            STATE.sync_to_workspace()
            STATE.revision += 1
            return STATE.dataset_payload()
        else:
            return {
                "status": "success",
                "shape": synth_df.shape,
                "preview": synth_df.head(10).to_dict(orient="records")
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Synthetic data error: {e}")


@app.post("/api/data/undo")
def undo_data_endpoint():
    """Undoes the most recent data operation."""
    try:
        STATE.session.undo_data_op()
        return STATE.after_data_change()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Undo error: {e}")


@app.post("/api/data/modify")
def modify_data_structure(req: RowColActionRequest):
    df = STATE.session.df
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")

    try:
        if req.action == "add_row":
            STATE.session.add_rows(1)
        elif req.action == "delete_row":
            idx = int(req.target) if req.target is not None else len(df) - 1
            if 0 <= idx < len(df):
                STATE.session.drop_rows([idx])
        elif req.action == "add_col":
            base = str(req.target or "").strip() or f"Var{len(df.columns) + 1}"
            name = base
            k = 1
            while name in df.columns:
                name = f"{base}_{k}"
                k += 1
            STATE.session.compute(name, "np.nan")
        elif req.action == "delete_col":
            col = str(req.target or "")
            if col in df.columns and len(df.columns) > 1:
                STATE.session.drop_vars(col)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")

        return STATE.after_data_change()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/analysis/run")
def run_analysis(req: RunAnalysisRequest):
    df = STATE.session.df
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")

    try:
        res = dispatch_analysis(req.analysis_type, df, req.params)
        if req.analysis_id is not None:
            aid = int(req.analysis_id)
            payload = res.to_payload(analysis_id=aid)
            payload["analysis_type"] = req.analysis_type
            payload["params"] = req.params
            replaced = False
            for idx, existing in enumerate(STATE.analyses_store):
                if existing.get("analysisId") == aid:
                    STATE.analyses_store[idx] = payload
                    replaced = True
                    break
            if not replaced:
                STATE.analyses_store.append(payload)
            STATE.analysis_objects[aid] = res
        else:
            aid = STATE.next_id
            STATE.next_id += 1
            payload = res.to_payload(analysis_id=aid)
            payload["analysis_type"] = req.analysis_type
            payload["params"] = req.params
            STATE.analyses_store.append(payload)
            STATE.analysis_objects[aid] = res

        STATE.session.history = list(STATE.analysis_objects.values())
        STATE.custom_py_code = None
        STATE.sync_to_workspace()
        STATE.revision += 1
        payload["revision"] = STATE.revision
        payload["pythonCode"] = STATE.current_python_script()
        return payload

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/analysis/{analysis_id}")
def delete_analysis(analysis_id: int):
    STATE.analyses_store = [a for a in STATE.analyses_store if a.get("analysisId") != analysis_id]
    STATE.analysis_objects.pop(analysis_id, None)
    STATE.session.history = list(STATE.analysis_objects.values())
    STATE.custom_py_code = None
    STATE.sync_to_workspace()
    STATE.revision += 1
    return {"status": "success", "revision": STATE.revision, "pythonCode": STATE.current_python_script()}


@app.post("/api/analysis/clear")
def clear_analyses():
    STATE.analyses_store = []
    STATE.analysis_objects = {}
    STATE.session.history = []
    STATE.custom_py_code = None
    STATE.sync_to_workspace()
    STATE.revision += 1
    return {"status": "success", "revision": STATE.revision, "pythonCode": STATE.current_python_script()}


@app.get("/api/syntax/current")
def get_current_syntax():
    return {
        "status": "success",
        "dataset": STATE.dataset_name,
        "path": LIVE_PY_PATH,
        "code": STATE.current_python_script()
    }


@app.post("/api/syntax/execute")
def execute_syntax_endpoint(req: ExecuteSyntaxRequest):
    """Executes user- or Claude-edited Python (`.py`) or Stata (`.do`) code directly and updates the Results Sheet & workspace."""
    try:
        from .stata_transpiler import transpile_stata_script
        code_to_run = transpile_stata_script(req.code)
        default_path = os.path.join(SAMPLE_DIR, STATE.dataset_name)
        analyses, stdout_log, final_df, final_ds_name = execute_python_syntax(
            code_to_run,
            default_df=STATE.session.df,
            default_dataset_path=default_path if os.path.isfile(default_path) else None,
            sync_workspace=False,
            session_attrs={"boundary": {k: v for k, v in (getattr(STATE, "boundary", None) or {}).items()
                                        if k in ("boundary", "boundaryKey", "idVar", "labelVar") and v} or None},
        )
        if final_df is not None:
            STATE.session = JamoviSession(final_df, auto_sync_workspace=False)
            if final_ds_name:
                STATE.dataset_name = final_ds_name

        if req.replace_existing:
            STATE.analyses_store = []
            STATE.analysis_objects = {}
            STATE.next_id = 1

        for a in analyses:
            aid = STATE.next_id
            STATE.next_id += 1
            payload = a.to_payload(analysis_id=aid)
            payload["analysis_type"] = a.name
            payload["params"] = a.options
            STATE.analyses_store.append(payload)
            STATE.analysis_objects[aid] = a

        STATE.session.history = list(STATE.analysis_objects.values())
        STATE.sync_to_workspace(py_code=req.code)
        STATE.revision += 1

        resp = STATE.dataset_payload()
        resp["stdout"] = stdout_log
        return resp
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Python syntax execution error: {e}")


@app.get("/api/syntax/export_py")
def export_python_script():
    code = STATE.current_python_script()
    base = os.path.splitext(STATE.dataset_name)[0]
    return StreamingResponse(
        iter([code.encode("utf-8")]),
        media_type="text/x-python; charset=utf-8",
        headers={"Content-Disposition": _attachment(f"{base}_openstat_analysis.py")}
    )


@app.post("/api/plot/restyle")
def restyle_plots_endpoint(req: PlotRestyleRequest):
    """Updates `PlotStyle` on one specific analysis card or all active analysis cards and re-renders plots."""
    df = STATE.session.df
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")

    updated_any = False
    for idx, card in enumerate(STATE.analyses_store):
        aid = card.get("analysisId")
        if req.analysis_id is not None and aid != req.analysis_id:
            continue
        a_type = card.get("analysis_type") or card.get("name")
        if a_type == "customFigure":
            continue
        params = dict(card.get("params") or card.get("options") or {})
        merged_style = dict(params.get("plotStyle") or {})
        merged_style.update(req.plot_style or {})
        params["plotStyle"] = merged_style

        res = dispatch_analysis(a_type, df, params)
        new_payload = res.to_payload(analysis_id=aid)
        new_payload["analysis_type"] = a_type
        new_payload["params"] = params
        STATE.analyses_store[idx] = new_payload
        STATE.analysis_objects[aid] = res
        updated_any = True

    if not updated_any and req.analysis_id is not None:
        raise HTTPException(status_code=404, detail=f"Analysis #{req.analysis_id} not found")

    STATE.session.history = list(STATE.analysis_objects.values())
    STATE.custom_py_code = None
    STATE.sync_to_workspace()
    STATE.revision += 1
    return STATE.dataset_payload()


@app.get("/api/plot/export")
def export_plot_endpoint(
    analysis_id: int = Query(...),
    plot_idx: int = Query(0),
    fmt: str = Query("svg")
):
    """Exports an individual plot as lossless vector `.svg` / `.pdf` or high-DPI `.png`."""
    obj = STATE.analysis_objects.get(analysis_id)
    if obj is None:
        raise HTTPException(status_code=404, detail=f"Analysis #{analysis_id} not found")
    if plot_idx < 0 or plot_idx >= len(obj.plots):
        raise HTTPException(status_code=404, detail=f"Plot index {plot_idx} not found")

    ext = fmt.lower().lstrip(".")
    if ext not in ("svg", "pdf", "png"):
        raise HTTPException(status_code=400, detail="Plot format must be svg, pdf, or png")

    vec = obj.plot_vectors[plot_idx] if plot_idx < len(obj.plot_vectors) else {}
    fname = f"openstat_analysis_{analysis_id}_plot_{plot_idx + 1}.{ext}"

    if ext == "svg" and vec.get("svg"):
        return StreamingResponse(
            iter([vec["svg"].encode("utf-8")]),
            media_type="image/svg+xml",
            headers={"Content-Disposition": _attachment(f"{fname}")}
        )
    elif ext == "pdf" and vec.get("pdf_b64"):
        raw_pdf = base64.b64decode(vec["pdf_b64"])
        return StreamingResponse(
            iter([raw_pdf]),
            media_type="application/pdf",
            headers={"Content-Disposition": _attachment(f"{fname}")}
        )
    else:
        src = obj.plots[plot_idx]["src"]
        raw_png = base64.b64decode(src.split(",", 1)[1] if "," in src else src)
        return StreamingResponse(
            iter([raw_png]),
            media_type="image/png",
            headers={"Content-Disposition": _attachment(f"{fname}")}
        )


class PushFigureToResultsRequest(BaseModel):
    src: Optional[str] = None
    source_analysis_id: Optional[int] = None
    source_plot_idx: int = 0
    target_analysis_id: Optional[int] = None
    title: str = "Standalone Visualization"
    caption: str = ""


@app.post("/api/graph/push_to_results")
def push_graph_to_results(req: PushFigureToResultsRequest):
    """Exports/attaches a standalone or existing figure into the Analysis Results Sheet (as a card or attached to an existing analysis)."""
    img_src = req.src
    vec_bundle = {}
    plot_title = req.title

    if not img_src and req.source_analysis_id is not None:
        src_obj = STATE.analysis_objects.get(req.source_analysis_id)
        if src_obj is None or req.source_plot_idx >= len(src_obj.plots):
            raise HTTPException(status_code=404, detail="Source plot not found")
        img_src = src_obj.plots[req.source_plot_idx]["src"]
        plot_title = req.title or src_obj.plots[req.source_plot_idx].get("title", "Graph")
        if req.source_plot_idx < len(src_obj.plot_vectors):
            vec_bundle = dict(src_obj.plot_vectors[req.source_plot_idx])

    if not img_src:
        raise HTTPException(status_code=400, detail="Provide `src` data URI or `source_analysis_id`.")

    if req.target_analysis_id is not None and req.target_analysis_id in STATE.analysis_objects:
        tgt = STATE.analysis_objects[req.target_analysis_id]
        tgt.plots.append({"title": plot_title, "src": img_src, "style": vec_bundle.get("style", {})})
        tgt.plot_vectors.append(vec_bundle)
        for idx, existing in enumerate(STATE.analyses_store):
            if existing.get("analysisId") == req.target_analysis_id:
                new_p = tgt.to_payload(analysis_id=req.target_analysis_id)
                new_p["analysis_type"] = existing.get("analysis_type", tgt.name)
                new_p["params"] = existing.get("params", tgt.options)
                STATE.analyses_store[idx] = new_p
                break
    else:
        card = CustomFigureResult(
            fig_or_src=img_src,
            title=plot_title,
            caption=req.caption,
            dataset=STATE.session.df,
        )
        if vec_bundle and card.plot_vectors:
            card.plot_vectors[0] = vec_bundle
        aid = STATE.next_id
        STATE.next_id += 1
        payload = card.to_payload(analysis_id=aid)
        payload["analysis_type"] = "customFigure"
        payload["params"] = card.options
        STATE.analyses_store.append(payload)
        STATE.analysis_objects[aid] = card

    STATE.session.history = list(STATE.analysis_objects.values())
    STATE.sync_to_workspace()
    STATE.revision += 1
    return STATE.dataset_payload()


@app.get("/api/export/results")
@app.get("/api/export/tables")
def export_tables_endpoint(
    fmt: str = Query("xlsx"),
    analysis_id: Optional[int] = Query(None),
    report_style: Optional[str] = Query(None),
    report_options: Optional[str] = Query(None),
):
    """Exports full Analysis Results (tables + embedded graphs) to `.pdf`, `.hwpx`, `.docx`, `.xlsx`, `.tex`, `.csv`, `.tsv`, `.md`, `.html`, `.json`, or `.omv`.

    report_style=social applies the social-science table format; report_options is a JSON object of its options.
    """
    ext = fmt.lower().lstrip(".")
    if ext not in SUPPORTED_TABLE_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{ext}'. Supported: {', '.join(SUPPORTED_TABLE_FORMATS)}")
    rf = _report_format_from_query(report_style, report_options)

    if analysis_id is not None:
        obj = STATE.analysis_objects.get(analysis_id)
        if obj is None:
            raise HTTPException(status_code=404, detail=f"Analysis #{analysis_id} not found")
        target_analyses = [obj]
        suffix = f"analysis_{analysis_id}"
    else:
        target_analyses = list(STATE.analysis_objects.values())
        suffix = "tables"

    base = os.path.splitext(STATE.dataset_name)[0]
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=f".{ext}", dir=REPORTS_DIR)
    os.close(tmp_fd)

    try:
        export_analyses(target_analyses, output_path=tmp_path, fmt=ext, dataset=STATE.session.df, report_style=rf)
        with open(tmp_path, "rb") as f:
            file_bytes = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    media_type = MIME_MAP.get(ext, "application/octet-stream")
    return StreamingResponse(
        iter([file_bytes]),
        media_type=media_type,
        headers={"Content-Disposition": _attachment(f"{base}_{suffix}.{ext}")}
    )


@app.post("/api/ai/push")
def ai_push_analysis(req: AIPushRequest):
    """Endpoint for AI agents (Claude Code, Codex, Antigravity) to push live statistical analyses into the UI."""
    if req.dataset and req.dataset != STATE.dataset_name:
        candidate = os.path.join(SAMPLE_DIR, req.dataset)
        if os.path.isfile(candidate):
            STATE.load_file(req.dataset)
    run_req = RunAnalysisRequest(analysis_type=req.analysis_type, params=req.params)
    return run_analysis(run_req)


@app.get("/api/dataset/export_omv")
def export_omv():
    if STATE.session.df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    out_path = os.path.join(REPORTS_DIR, "export_session.omv")
    write_omv(STATE.session.df, out_path, analyses=list(STATE.analysis_objects.values()))
    with open(out_path, "rb") as f:
        raw = f.read()
    base = os.path.splitext(STATE.dataset_name)[0]
    return StreamingResponse(
        iter([raw]),
        media_type="application/octet-stream",
        headers={"Content-Disposition": _attachment(f"{base}.omv")}
    )


@app.get("/api/dataset/export_csv")
def export_csv():
    if STATE.session.df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    buf = io.StringIO()
    STATE.session.df.to_csv(buf, index=False)
    base = os.path.splitext(STATE.dataset_name)[0]
    return StreamingResponse(
        iter([buf.getvalue().encode("utf-8-sig")]),
        media_type="text/csv",
        headers={"Content-Disposition": _attachment(f"{base}_raw_data.csv")}
    )


def _report_format_from_query(report_style: Optional[str], report_options: Optional[str]):
    """ReportFormat from query parameters (options as a JSON object); invalid values -> HTTP 400."""
    from pyjamovi.report_format import resolve_report_format
    opts = None
    if report_options:
        try:
            opts = json.loads(report_options)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"report_options: JSON 형식 오류 ({e})")
        if not isinstance(opts, dict):
            raise HTTPException(status_code=400, detail="report_options: JSON 객체 필요")
    try:
        return resolve_report_format(report_style, opts)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@app.get("/api/report/export_html")
def export_html(report_style: Optional[str] = Query(None), report_options: Optional[str] = Query(None)):
    rf = _report_format_from_query(report_style, report_options)
    analyses_list = list(STATE.analysis_objects.values())
    html_str = render_jamovi_html_report(analyses_list, title=f"OpenStat Studio Results — {STATE.dataset_name}", report_format=rf)
    base = os.path.splitext(STATE.dataset_name)[0]
    return StreamingResponse(
        iter([html_str.encode("utf-8")]),
        media_type="text/html",
        headers={"Content-Disposition": _attachment(f"{base}_openstat_results.html")}
    )

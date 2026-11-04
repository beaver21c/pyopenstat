"""
OpenStat GIS spatial analysis (GeoDa / PySAL / Stata Sp / xsmle equivalent scope).

Analyses
- SpatialWeightsAnalysis  공간가중행렬 생성·진단 (contiguity, kNN, distance band, inverse distance, external file; export .gal/.gwt/.csv)
- SpatialAutocorrelation  전역 Moran's I(정규 근사 + 순열), LISA(조건부 순열, 다중검정 보정), Getis–Ord Gi*
- SpatialAnalysis         횡단면 공간회귀: OLS 진단(LM 검정) · SLX · SAR · SEM · SDM · SDEM · SAC, 직접·간접·총효과, 모형 비교(LR)
- SpatialPanelAnalysis    공간패널회귀: 개체·시점·양방향 고정효과(Lee–Yu 보정), 확률효과(SAR·SDM), Hausman, 패널 LM 검정
- GWRAnalysis             지리가중회귀(mgwr): 커널·대역폭 선택, 국지 계수 지도, 공간 변이 검정
- SpatialMapAnalysis      주제도(단계구분도)

Spatial information comes from a boundary file (.shp/.zip/.geojson/.gpkg) matched to the data by a key column,
or from x/y coordinate columns (lon/lat are projected to UTM for distances).
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseAnalysis, ResultsTable, resolve_choice
from .spatial_weights import ISLAND_LABELS, NORM_LABELS, WEIGHT_KIND_LABELS, SpatialWeights, WeightsSpec, build_weights
from . import spatial_models as sm
from . import spatial_maps as smap

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


# ---------------------------------------------------------------- parameter helpers
def _p(p: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in p and p[k] not in (None, "", []):
            return p[k]
    return default


def _flag(v, default=False) -> bool:
    if v is None or v == "":
        return default
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def _as_list(v) -> List[str]:
    if v is None or v == "":
        return []
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return [str(x) for x in v if str(x).strip()]


def _resolve_path(path: str) -> str:
    if not path:
        return ""
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    for base in (os.getcwd(), PROJECT_ROOT):
        cand = os.path.join(base, path)
        if os.path.exists(cand):
            return cand
    return os.path.join(PROJECT_ROOT, path)


def _rel(path: str) -> str:
    try:
        r = os.path.relpath(path, PROJECT_ROOT)
        return r if not r.startswith("..") else path
    except ValueError:
        return path


def _numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _sort_keys(keys: Sequence[str]) -> List[str]:
    keys = list(keys)
    if all(re.fullmatch(r"-?\d+", k or "") for k in keys):
        return sorted(keys, key=lambda k: int(k))
    return sorted(keys)


def _pstars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ---------------------------------------------------------------- spatial context
class SpatialContext:
    """Where the geometry comes from and how data rows map onto spatial units."""

    def __init__(self, df: pd.DataFrame, p: Dict[str, Any]):
        self.df = df
        self.p = p
        self.boundary = _resolve_path(str(_p(p, "boundary", "boundary_path", "shp", default="") or ""))
        self.boundary_key = str(_p(p, "boundaryKey", "boundary_key", default="") or "")
        self.id_col = str(_p(p, "idVar", "id", "region_id", "unit", default="") or "")
        self.x_col = str(_p(p, "coordX", "lon", "longitude", "x", default="") or "")
        self.y_col = str(_p(p, "coordY", "lat", "latitude", "y", default="") or "")
        self.label_col = str(_p(p, "labelVar", "label", default="") or "")
        if self.id_col and self.id_col not in df.columns:
            raise ValueError(f"지역 식별 변수 '{self.id_col}'가 데이터에 없음")
        if self.boundary:
            from ..spatial_io import read_boundary
            gdf = read_boundary(self.boundary)
            if not self.boundary_key:
                raise ValueError("경계 파일의 연결 속성(boundaryKey)을 지정해야 함")
            if not self.id_col:
                if self.boundary_key in df.columns:
                    self.id_col = self.boundary_key
                else:
                    raise ValueError("경계 파일과 연결할 지역 식별 변수(idVar)를 지정해야 함")
            self.source = "boundary"
            self.boundary_n = len(gdf)
        else:
            if not (self.x_col and self.y_col):
                self.x_col = self.x_col or next((c for c in df.columns if re.search(r"^(lon|long|longitude|경도|x|x_?coord|utm_?x)$", str(c), re.I)), "")
                self.y_col = self.y_col or next((c for c in df.columns if re.search(r"^(lat|latitude|위도|y|y_?coord|utm_?y)$", str(c), re.I)), "")
            if not (self.x_col and self.y_col):
                raise ValueError("공간 정보 없음: 데이터 탭의 「지도 경계 연결」로 경계 파일(.shp)을 연결하거나 x·y(경도·위도) 좌표 변수를 지정")
            for c in (self.x_col, self.y_col):
                if c not in df.columns:
                    raise ValueError(f"좌표 변수 '{c}'가 데이터에 없음")
            self.source = "coords"
        if self.label_col and self.label_col not in df.columns:
            self.label_col = ""

    def describe(self) -> str:
        if self.source == "boundary":
            return f"경계 파일 {os.path.basename(self.boundary)} (연결: 데이터 '{self.id_col}' = 경계 '{self.boundary_key}')"
        return f"좌표 변수 x='{self.x_col}', y='{self.y_col}'"

    def keys(self, frame: pd.DataFrame) -> pd.Series:
        from ..spatial_io import normalize_keys
        if self.id_col:
            return normalize_keys(frame[self.id_col]).set_axis(frame.index)
        return pd.Series([str(i + 1) for i in range(len(frame))], index=frame.index)

    def required_columns(self) -> List[str]:
        cols = [c for c in (self.id_col,) if c]
        if self.source == "coords":
            cols += [self.x_col, self.y_col]
        return cols

    def boundary_keys(self) -> set:
        from ..spatial_io import normalize_keys, read_boundary
        gdf = read_boundary(self.boundary)
        if self.boundary_key not in gdf.columns:
            raise ValueError(f"경계 파일에 '{self.boundary_key}' 속성 없음")
        return set(normalize_keys(gdf[self.boundary_key]).dropna())

    def geometries(self, units: List[str], frame: pd.DataFrame, keys: pd.Series):
        from ..spatial_io import points_geometries, unit_geometries
        if self.source == "boundary":
            return unit_geometries(self.boundary, self.boundary_key, units)
        first = frame.assign(_k=keys.values).drop_duplicates("_k").set_index("_k").loc[units]
        return points_geometries(first[self.x_col].astype(float).values, first[self.y_col].astype(float).values, units)

    def unit_labels(self, units: List[str], frame: pd.DataFrame, keys: pd.Series) -> List[str]:
        col = self.label_col or self.id_col
        if not col:
            return list(units)
        first = frame.assign(_k=keys.values).drop_duplicates("_k").set_index("_k")
        return [str(first.loc[u, col]) for u in units]


def _weights_from_params(p: Dict[str, Any], geoms) -> SpatialWeights:
    spec = WeightsSpec.from_params(p, default_kind="auto")
    if spec.kind == "file":
        spec.file = _resolve_path(spec.file)
    return build_weights(spec, geoms)


def _prepare_units(ctx: SpatialContext, frame: pd.DataFrame, notes: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    keys = ctx.keys(frame)
    frame = frame[keys.notna()]
    keys = keys[keys.notna()]
    if ctx.source == "boundary":
        bkeys = ctx.boundary_keys()
        unmatched = sorted(set(keys) - bkeys)
        if unmatched:
            notes.append(f"경계 파일에서 찾지 못한 지역 {len(unmatched)}곳 제외: {', '.join(unmatched[:8])}{' …' if len(unmatched) > 8 else ''}")
            m = keys.isin(bkeys)
            frame, keys = frame[m], keys[m]
        if frame.empty:
            raise ValueError("데이터의 지역 식별자가 경계 파일의 연결 속성과 하나도 일치하지 않음 — 연결 변수를 확인")
    return frame, keys


def _assumption_table(rows: List[Tuple[str, str]]) -> ResultsTable:
    t = ResultsTable(name="assumptions", title="분석 가정·설정")
    t.add_column("item", "항목", col_type="text")
    t.add_column("value", "설정", col_type="text")
    for a, b in rows:
        t.add_row({"item": a, "value": b})
    return t


def _w_rows(W: SpatialWeights, ctx: SpatialContext) -> List[Tuple[str, str]]:
    s = W.summary()
    rows = [("공간 정보", ctx.describe()),
            ("공간가중행렬", W.spec.describe()),
            ("정규화", NORM_LABELS[W.spec.norm]),
            ("이웃 수", f"평균 {s['mean']:.2f} (최소 {s['min']}, 최대 {s['max']})")]
    if W.spec.kind in ("distance", "idw") and W.threshold is not None:
        rows.append(("거리 임계값", f"{W.threshold:,.2f} {W.unit}" if W.threshold > 0 else "없음(모든 지역 쌍)"))
    if W.islands:
        rows.append(("이웃 없는 지역", f"{len(W.islands)}곳 — {ISLAND_LABELS[W.island_action]}"))
    if s["components"] > 1:
        rows.append(("연결 묶음", f"{s['components']}개(서로 연결되지 않은 지역 묶음 존재)"))
    return rows


def _filter_period(df: pd.DataFrame, p: Dict[str, Any], notes: List[str]) -> pd.DataFrame:
    """Panel data in a cross-sectional analysis: keep one period (`period`, default = the last one)."""
    tv = str(_p(p, "timeVar", "time", default="") or "")
    if not tv:
        return df
    if tv not in df.columns:
        raise ValueError(f"시점 변수 '{tv}'가 데이터에 없음")
    vals = pd.Series(df[tv].dropna().unique())
    if vals.empty:
        return df
    try:
        order = sorted(vals.tolist())
    except TypeError:
        order = sorted(vals.astype(str).tolist())
    want = str(_p(p, "period", default="") or "").strip()
    if want:
        match = [v for v in order if str(v).strip() == want or (str(v).strip().rstrip("0").rstrip(".") == want.rstrip("0").rstrip("."))]
        if not match:
            raise ValueError(f"시점 '{want}'가 '{tv}'에 없음 (가능: {', '.join(map(str, order[:12]))}{' …' if len(order) > 12 else ''})")
        sel = match[0]
    else:
        sel = order[-1]
    notes.append(f"패널 자료에서 {tv} = {sel} 시점만 사용" + ("" if want else "(마지막 시점)"))
    return df[df[tv] == sel]


# ---------------------------------------------------------------- Moran / LISA / Gi*
def global_moran(y: np.ndarray, W: SpatialWeights, permutations: int = 999, seed: int = 42) -> Dict[str, float]:
    Wd = W.W
    n = len(y)
    z = y - y.mean()
    S0 = float(Wd.sum())
    m2 = float(z @ z)
    I = (n / S0) * float(z @ (Wd @ z)) / m2
    EI = -1.0 / (n - 1)
    S1 = 0.5 * float(((Wd + Wd.T).multiply(Wd + Wd.T)).sum())
    S2 = float(((np.asarray(Wd.sum(axis=1)).ravel() + np.asarray(Wd.sum(axis=0)).ravel()) ** 2).sum())
    b2 = n * float((z ** 4).sum()) / m2 ** 2
    A = n * ((n * n - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
    B = b2 * ((n * n - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)
    VI = (A - B) / ((n - 1) * (n - 2) * (n - 3) * S0 ** 2) - EI ** 2
    zr = (I - EI) / np.sqrt(VI) if VI > 0 else np.nan
    out = {"I": I, "E": EI, "V": VI, "z": zr, "p": float(2 * stats.norm.sf(abs(zr))) if np.isfinite(zr) else np.nan}
    if permutations:
        rng = np.random.default_rng(seed)
        sims = np.empty(permutations)
        for s in range(permutations):
            zp = rng.permutation(z)
            sims[s] = (n / S0) * float(zp @ (Wd @ zp)) / m2
        larger = int((sims >= I).sum())
        if permutations - larger < larger:
            larger = permutations - larger
        out["p_sim"] = (larger + 1.0) / (permutations + 1.0)
    return out


def local_moran(y: np.ndarray, W: SpatialWeights, permutations: int = 999, seed: int = 42) -> Dict[str, np.ndarray]:
    """Local Moran's I with conditional permutation pseudo p-values (folded, as in GeoDa / esda)."""
    Wd = W.W.tocsr()
    n = len(y)
    z = y - y.mean()
    m2 = float(z @ z) / n
    lag = Wd @ z
    Ii = z * lag / m2
    p = np.full(n, np.nan)
    if permutations:
        rng = np.random.default_rng(seed)
        P = int(permutations)
        pool = np.tile(np.arange(n - 1), (P, 1))
        rows = np.arange(P)
        for i in range(n):
            lo, hi = Wd.indptr[i], Wd.indptr[i + 1]
            k = hi - lo
            if k == 0:
                continue
            w = Wd.data[lo:hi]
            for s in range(k):
                j = rng.integers(s, n - 1, size=P)
                tmp = pool[rows, s].copy()
                pool[rows, s] = pool[rows, j]
                pool[rows, j] = tmp
            idx = pool[:, :k]
            idx = idx + (idx >= i)
            sim = z[i] * (z[idx] @ w) / m2
            larger = int((sim >= Ii[i]).sum())
            if P - larger < larger:
                larger = P - larger
            p[i] = (larger + 1.0) / (P + 1.0)
    quad = np.where(z >= 0, np.where(lag >= 0, "High-High", "High-Low"), np.where(lag >= 0, "Low-High", "Low-Low"))
    no_nb = np.diff(Wd.indptr) == 0
    return {"I": Ii, "p": p, "quad": quad, "z": z / np.sqrt(m2), "lag": lag / np.sqrt(m2), "no_nb": no_nb}


def adjust_p(p: np.ndarray, method: str) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if method == "bonferroni":
        return np.minimum(p * np.isfinite(p).sum(), 1.0)
    if method == "fdr":
        out = np.full_like(p, np.nan)
        ok = np.isfinite(p)
        pv = p[ok]
        m = len(pv)
        order = np.argsort(pv)
        ranked = pv[order] * m / (np.arange(m) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        res = np.empty(m)
        res[order] = np.minimum(ranked, 1.0)
        out[ok] = res
        return out
    return p


def getis_ord_star(y: np.ndarray, W: SpatialWeights) -> np.ndarray:
    """Gi* z-scores with binary weights including the unit itself (Ord & Getis 1995)."""
    n = len(y)
    B = (W.raw != 0).astype(float).tolil()
    B.setdiag(1.0)
    B = B.tocsr()
    wsum = np.asarray(B.sum(axis=1)).ravel()
    w2 = np.asarray(B.multiply(B).sum(axis=1)).ravel()
    xbar = y.mean()
    s = np.sqrt((y ** 2).mean() - xbar ** 2)
    num = B @ y - xbar * wsum
    den = s * np.sqrt((n * w2 - wsum ** 2) / (n - 1))
    return np.divide(num, den, out=np.zeros(n), where=den > 0)


def gi_category(zs: np.ndarray) -> List[str]:
    out = []
    for z in zs:
        a = abs(z)
        lvl = "99%" if a >= 2.576 else "95%" if a >= 1.96 else "90%" if a >= 1.645 else None
        out.append("유의하지 않음" if lvl is None else (f"열점 {lvl}" if z > 0 else f"냉점 {lvl}"))
    return out


# ---------------------------------------------------------------- 1. weights
class SpatialWeightsAnalysis(BaseAnalysis):
    def __init__(self, data: pd.DataFrame, **params):
        super().__init__(name="spatialWeights", title="공간가중행렬 생성·진단", dataset=data, options=dict(params))
        self.syntax_python = "sess.spatial_weights(" + ", ".join(f"{k}={v!r}" for k, v in params.items() if v not in (None, "", [])) + ")"
        self.run()

    def run(self):
        p = self.options
        notes: List[str] = []
        ctx = SpatialContext(self.dataset, p)
        frame = self.dataset.dropna(subset=ctx.required_columns()) if ctx.required_columns() else self.dataset
        frame, keys = _prepare_units(ctx, frame, notes)
        units = list(pd.unique(keys))
        if ctx.source == "boundary" and len(units) < len(keys):
            notes.append(f"지역 식별자가 중복된 행이 있어 고유 지역 {len(units)}곳 기준으로 행렬 작성(패널 자료 등)")
        units = _sort_keys(units)
        geoms = ctx.geometries(units, frame, keys)
        W = _weights_from_params(p, geoms)
        labels = ctx.unit_labels(W.ids, frame, keys)
        s = W.summary()

        self.add_table(_assumption_table(_w_rows(W, ctx) + [("Stata 대응 명령(참고)", W.stata_hint())]))

        t = ResultsTable(name="wSummary", title="공간가중행렬 요약")
        t.add_column("stat", "지표", col_type="text")
        t.add_column("value", "값", col_type="number", fmt="auto")
        for k_, v_ in (("공간 단위 수", s["n"]), ("연결(0이 아닌 가중치) 수", s["links"]), ("0이 아닌 비율(%)", s["pct"]),
                       ("평균 이웃 수", s["mean"]), ("이웃 수 중앙값", s["median"]), ("최소 이웃 수", s["min"]),
                       ("최대 이웃 수", s["max"]), ("이웃 없는 지역 수(원래)", len(s["islands"])), ("연결 묶음 수", s["components"])):
            t.add_row({"stat": k_, "value": float(v_)})
        t.add_note(("원 행렬 대칭" if s["symmetric"] else "원 행렬 비대칭(k-최근접 등)") + ". " + W.describe())
        if W.dropped:
            t.add_note(f"분석에서 제외한 지역: {', '.join(W.dropped)}")
        for n_ in notes:
            t.add_note(n_)
        self.add_table(t)

        cnt = W.neighbour_counts()
        h = ResultsTable(name="wHistogram", title="이웃 수 분포")
        h.add_column("k", "이웃 수", col_type="integer")
        h.add_column("n", "지역 수", col_type="integer")
        h.add_column("pct", "비율", col_type="number", fmt="pc")
        vals, freq = np.unique(cnt, return_counts=True)
        for v_, f_ in zip(vals, freq):
            h.add_row({"k": int(v_), "n": int(f_), "pct": 100.0 * f_ / len(cnt)})
        self.add_table(h)

        nb = ResultsTable(name="wNeighbors", title="지역별 이웃 목록")
        nb.add_column("key", "지역", col_type="text")
        nb.add_column("name", "지역명", col_type="text")
        nb.add_column("k", "이웃 수", col_type="integer")
        nb.add_column("neighbors", "이웃 지역", col_type="text")
        R = W.raw.tocsr()
        lab = dict(zip(W.ids, labels))
        limit = int(_p(p, "listLimit", default=400))
        for i, key in enumerate(W.ids[:limit]):
            cols = R.indices[R.indptr[i]:R.indptr[i + 1]]
            nb.add_row({"key": key, "name": lab[key], "k": int(len(cols)), "neighbors": ", ".join(lab[W.ids[j]] for j in cols)})
        if len(W.ids) > limit:
            nb.add_note(f"상위 {limit}곳만 표시(전체 {len(W.ids)}곳).")
        self.add_table(nb)

        export = str(_p(p, "wExport", "export", default="") or "")
        if export:
            path = W.export(_resolve_path(export) if not os.path.isabs(export) else export)
            t.add_note(f"가중행렬 저장: {_rel(path)}" + (" (Stata: import delimited → spset _ID → spmatrix fromdata W = w*, normalize(none))" if path.lower().endswith(".csv") else ""))
        if _flag(_p(p, "plotLinks"), True):
            self.add_plot("이웃 연결 지도", smap.plot_links(geoms.loc[W.ids], W, f"공간가중행렬 연결 — {W.spec.describe()}",
                                                     labels=labels if _flag(_p(p, "showLabels"), False) else None))
        if _flag(_p(p, "plotHist"), True):
            self.add_plot("이웃 수 분포", smap.plot_neighbour_hist(cnt, "지역별 이웃 수 분포"))
        self.apa_narrative = (f"공간가중행렬은 {W.spec.describe()} 기준({NORM_LABELS[W.spec.norm]})으로 {s['n']}개 지역에 대해 작성하였다. "
                              f"지역당 평균 이웃 수는 {s['mean']:.2f}개(범위 {s['min']}–{s['max']})이다.")
        self.W = W


# ---------------------------------------------------------------- 2. autocorrelation
class SpatialAutocorrelation(BaseAnalysis):
    def __init__(self, data: pd.DataFrame, **params):
        super().__init__(name="spatialMoran", title="공간자기상관 분석 (Moran's I · LISA · Gi*)", dataset=data, options=dict(params))
        self.syntax_python = "sess.spatial_autocorrelation(" + ", ".join(f"{k}={v!r}" for k, v in params.items() if v not in (None, "", [])) + ")"
        self.run()

    def run(self):
        p = self.options
        _check_map_style(p)
        vars_ = _as_list(_p(p, "vars", "dep", "variables"))
        if not vars_:
            raise ValueError("분석할 변수(vars)를 1개 이상 지정")
        for v in vars_:
            if v not in self.dataset.columns:
                raise ValueError(f"변수 '{v}'가 데이터에 없음")
        notes: List[str] = []
        ctx = SpatialContext(self.dataset, p)
        frame = _numeric(_filter_period(self.dataset, p, notes), vars_).dropna(subset=vars_ + ctx.required_columns())
        frame, keys = _prepare_units(ctx, frame, notes)
        if keys.duplicated().any():
            raise ValueError("지역 식별자 중복: 지역당 1행 자료에서 실행 — 연도별 반복 자료는 시점 변수 지정")
        units = list(keys)
        geoms = ctx.geometries(units, frame, keys)
        W = _weights_from_params(p, geoms)
        pos = {k: i for i, k in enumerate(units)}
        sel = [pos[k] for k in W.ids]
        frame = frame.iloc[sel]
        labels = ctx.unit_labels(W.ids, frame, keys.iloc[sel])
        perms = int(_p(p, "permutations", default=999))
        alpha = float(_p(p, "alpha", default=0.05))
        adjust = resolve_choice("다중검정 보정", _p(p, "adjust", default="none"), ["none", "fdr", "bonferroni"],
                                {"bh": "fdr", "fdr_bh": "fdr", "benjamini": "fdr", "bonf": "bonferroni"})
        seed = int(_p(p, "seed", default=42))
        show_labels = _flag(_p(p, "showLabels"), False)
        g = geoms.loc[W.ids]

        rows = _w_rows(W, ctx) + [("순열 횟수", f"{perms:,}회 (seed {seed})"), ("국지 지표 유의수준", f"{alpha:g}" + {"none": "", "fdr": " (FDR 보정)", "bonferroni": " (Bonferroni 보정)"}.get(adjust, ""))]
        self.add_table(_assumption_table(rows))

        tg = ResultsTable(name="globalMoran", title="전역 공간자기상관 (Moran's I)")
        tg.add_column("var", "변수", col_type="text")
        tg.add_column("I", "Moran's I", col_type="number")
        tg.add_column("E", "기댓값 E[I]", col_type="number")
        tg.add_column("z", "z", col_type="number")
        tg.add_column("p", "p(정규 근사)", col_type="number", fmt="pvalue")
        tg.add_column("p_sim", "p(순열)", col_type="number", fmt="pvalue")
        tg.add_column("pattern", "공간 패턴", col_type="text")
        tl = ResultsTable(name="lisaSummary", title="국지 공간자기상관(LISA) 군집 분포")
        tl.add_column("var", "변수", col_type="text")
        for key in ("High-High", "Low-Low", "High-Low", "Low-High", "유의하지 않음"):
            tl.add_column(key, smap.LISA_LABELS[key], col_type="integer")
        tsig = ResultsTable(name="lisaRegions", title="유의한 국지 군집 지역")
        tsig.add_column("var", "변수", col_type="text")
        tsig.add_column("region", "지역", col_type="text")
        tsig.add_column("value", "값", col_type="number")
        tsig.add_column("Ii", "국지 Moran's I", col_type="number")
        tsig.add_column("p", "p", col_type="number", fmt="pvalue")
        tsig.add_column("cluster", "군집 유형", col_type="text")
        tgi = ResultsTable(name="giSummary", title="열점·냉점 분석 (Getis–Ord Gi*)")
        tgi.add_column("var", "변수", col_type="text")
        for key in smap.GI_COLORS:
            tgi.add_column(key, key, col_type="integer")

        do_lisa = _flag(_p(p, "lisa"), True)
        do_gi = _flag(_p(p, "gi"), False)
        narrative = []
        for v in vars_:
            y = frame[v].astype(float).values
            gm = global_moran(y, W, perms, seed)
            zp = gm["z"]
            pattern = "양(+)의 공간자기상관(유사한 값의 군집)" if gm["p"] < alpha and gm["I"] > gm["E"] else (
                "음(−)의 공간자기상관(상이한 값의 인접)" if gm["p"] < alpha else "유의한 공간자기상관 없음")
            tg.add_row({"var": v, "I": gm["I"], "E": gm["E"], "z": zp, "p": gm["p"], "p_sim": gm.get("p_sim", np.nan), "pattern": pattern})
            narrative.append(f"{v}의 Moran's I는 {gm['I']:.3f}(z = {zp:.2f}, p {'< .001' if gm['p'] < 0.001 else '= ' + format(gm['p'], '.3f')})")
            if do_lisa:
                lm = local_moran(y, W, perms, seed)
                padj = adjust_p(lm["p"], adjust)
                cat = np.where((padj <= alpha) & ~lm["no_nb"], lm["quad"], "유의하지 않음")
                row = {"var": v}
                for key in ("High-High", "Low-Low", "High-Low", "Low-High", "유의하지 않음"):
                    row[key] = int((cat == key).sum())
                tl.add_row(row)
                for i in np.argsort(padj):
                    if cat[i] == "유의하지 않음":
                        continue
                    tsig.add_row({"var": v, "region": labels[i], "value": float(y[i]), "Ii": float(lm["I"][i]), "p": float(padj[i]),
                                  "cluster": smap.LISA_LABELS[str(cat[i])]})
                if _flag(_p(p, "plotScatter", "plotMoran"), True):
                    self.add_plot(f"Moran 산점도 — {v}", smap.plot_moran_scatter(
                        lm["z"], lm["lag"], gm["I"], f"Moran 산점도 — {v} (I = {gm['I']:.3f})", f"표준화 {v}", f"공간시차 W·{v}", cats=list(cat)))
                if _flag(_p(p, "plotLisa", "plotMap"), True):
                    pal = {k: c for k, c in smap.LISA_COLORS.items() if k != "이웃 없음"}
                    self.add_plot(f"LISA 군집 지도 — {v}", smap.plot_categories(
                        g, list(cat), pal, f"LISA 군집 지도 — {v} (p ≤ {alpha:g})", "군집 유형", smap.LISA_LABELS,
                        labels=labels if show_labels else None))
            if do_gi:
                gz = getis_ord_star(y, W)
                gcat = gi_category(gz)
                row = {"var": v}
                for key in smap.GI_COLORS:
                    row[key] = int(sum(1 for c in gcat if c == key))
                tgi.add_row(row)
                if _flag(_p(p, "plotGi"), True):
                    self.add_plot(f"열점·냉점 지도 — {v}", smap.plot_categories(
                        g, gcat, smap.GI_COLORS, f"Getis–Ord Gi* 열점·냉점 — {v}", "신뢰수준", labels=labels if show_labels else None))
            if _flag(_p(p, "plotValue"), False):
                self.add_plot(f"주제도 — {v}", smap.plot_choropleth(
                    g, y, f"{v}", smap.resolve_scheme(_p(p, "scheme", default="quantiles")), int(_p(p, "classes", "k_classes", default=5)),
                    smap.resolve_cmap(_p(p, "cmap", default="YlOrRd")), v, labels=labels if show_labels else None))
        tg.add_note(f"정규 근사 p는 무작위화 가정(Cliff & Ord), 순열 p는 {perms:,}회 무작위 재배치 결과. " + W.describe())
        for n_ in notes:
            tg.add_note(n_)
        self.add_table(tg)
        if do_lisa:
            tl.add_note(f"조건부 순열 {perms:,}회, 유의수준 {alpha:g}" + {"none": "(다중검정 보정 없음)", "fdr": "(Benjamini–Hochberg FDR 보정)", "bonferroni": "(Bonferroni 보정)"}.get(adjust, "") + ".")
            self.add_table(tl)
            if not tsig.rows:
                tsig.add_note("유의한 국지 군집 지역 없음.")
            self.add_table(tsig)
        if do_gi:
            tgi.add_note("Gi*: 자기 지역을 포함한 이진 가중치, z 값 기준 신뢰수준(90%: 1.645, 95%: 1.96, 99%: 2.576). 다중검정 보정 없음.")
            self.add_table(tgi)
        self.apa_narrative = "; ".join(narrative) + "."


# ---------------------------------------------------------------- shared regression output
def _coef_table(fit: sm.FitResult, title: str, level: float = 0.95) -> ResultsTable:
    t = ResultsTable(name="spCoef", title=title)
    t.add_column("term", "변수", col_type="text")
    t.add_column("est", "계수", col_type="number")
    t.add_column("se", "표준오차", col_type="number")
    t.add_column("z", "z", col_type="number")
    t.add_column("p", "p", col_type="number", fmt="pvalue")
    t.add_column("lo", f"{int(level * 100)}% 하한", col_type="number")
    t.add_column("hi", f"{int(level * 100)}% 상한", col_type="number")
    zc = stats.norm.ppf(0.5 + level / 2)
    se = fit.se
    for nm, b, s in zip(fit.names, fit.coef, se):
        z = b / s if s > 0 else np.nan
        t.add_row({"term": nm, "est": float(b), "se": float(s), "z": float(z),
                   "p": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan, "lo": float(b - zc * s), "hi": float(b + zc * s)})
    return t


def _ols_table(fit: sm.FitResult, title: str) -> ResultsTable:
    t = ResultsTable(name="olsCoef", title=title)
    t.add_column("term", "변수", col_type="text")
    t.add_column("est", "계수", col_type="number")
    t.add_column("se", "표준오차", col_type="number")
    t.add_column("t", "t", col_type="number")
    t.add_column("p", "p", col_type="number", fmt="pvalue")
    dof = fit.extra.get("dof", fit.n_obs - len(fit.coef))
    for nm, b, s in zip(fit.names, fit.coef, fit.se):
        tv = b / s if s > 0 else np.nan
        t.add_row({"term": nm, "est": float(b), "se": float(s), "t": float(tv),
                   "p": float(2 * stats.t.sf(abs(tv), dof)) if np.isfinite(tv) else np.nan})
    t.add_note(f"R² = {fit.extra.get('r2', np.nan):.3f}, 오차 자유도 = {dof}.")
    return t


def _effects_table(rows: List[Dict], method: str, nsim: int, level: float) -> ResultsTable:
    t = ResultsTable(name="spEffects", title="직접·간접·총효과 (LeSage–Pace)")
    t.add_column("effect", "효과", col_type="text")
    t.add_column("var", "변수", col_type="text")
    t.add_column("est", "추정값", col_type="number")
    t.add_column("se", "표준오차", col_type="number")
    t.add_column("z", "z", col_type="number")
    t.add_column("p", "p", col_type="number", fmt="pvalue")
    t.add_column("lo", f"{int(level * 100)}% 하한", col_type="number")
    t.add_column("hi", f"{int(level * 100)}% 상한", col_type="number")
    last = None
    for r in rows:
        t.add_row({**r, "effect": r["effect"] if r["effect"] != last else ""})
        last = r["effect"]
    t.add_note("직접효과: 자기 지역 설명변수 변화가 자기 지역 종속변수에 미치는 평균 효과(공간 되먹임 포함). "
               "간접효과(파급효과): 다른 지역 설명변수 변화의 평균 영향. 총효과 = 직접효과 + 간접효과.")
    t.add_note("표준오차: " + ("델타 방법(Stata estat impact 방식)." if method != "sim" else f"모수 분포 모의추출 {nsim:,}회(xsmle nsim 방식)."))
    return t


def _lm_table(lm: Dict[str, Tuple[float, float]], title: str) -> ResultsTable:
    t = ResultsTable(name="lmTests", title=title)
    t.add_column("test", "검정", col_type="text")
    t.add_column("stat", "통계량", col_type="number")
    t.add_column("df", "자유도", col_type="integer")
    t.add_column("p", "p", col_type="number", fmt="pvalue")
    for k, (v, pv) in lm.items():
        t.add_row({"test": k, "stat": float(v), "df": 2 if "SARMA" in k else 1, "p": float(pv)})
    return t


def _lm_guidance(lm: Dict[str, Tuple[float, float]], alpha: float = 0.05) -> str:
    lag = lm["LM 공간시차"][1] < alpha
    err = lm["LM 공간오차"][1] < alpha
    rlag = lm["강건 LM 공간시차"][1] < alpha
    rerr = lm["강건 LM 공간오차"][1] < alpha
    if not lag and not err:
        return "모형 선택 참고: LM 공간시차·공간오차 검정 모두 유의하지 않음(OLS 잔차의 공간 의존성 미확인)."
    if lag and not err:
        return "모형 선택 참고: LM 공간시차만 유의 — 공간시차모형(SAR) 우선 검토."
    if err and not lag:
        return "모형 선택 참고: LM 공간오차만 유의 — 공간오차모형(SEM) 우선 검토."
    if rlag and not rerr:
        return "모형 선택 참고: 두 LM 검정 모두 유의, 강건 LM은 공간시차만 유의 — SAR 우선 검토."
    if rerr and not rlag:
        return "모형 선택 참고: 두 LM 검정 모두 유의, 강건 LM은 공간오차만 유의 — SEM 우선 검토."
    if not rlag and not rerr:
        return "모형 선택 참고: 두 LM 검정은 유의하나 강건 LM은 둘 다 유의하지 않음 — 한쪽으로 구분되지 않으므로 SDM·SAC 등 일반 모형과 LR 검정으로 축약 여부 검토(Elhorst 2010)."
    return "모형 선택 참고: 공간시차·공간오차 검정(강건 포함)이 함께 유의 — SDM·SAC 등 일반 모형과 LR 검정으로 축약 여부 검토(Elhorst 2010)."


def _compare(models: Sequence[str], fitter, alpha: float = 0.05) -> Tuple[ResultsTable, ResultsTable, Dict[str, sm.FitResult]]:
    fits: Dict[str, sm.FitResult] = {}
    for m in models:
        try:
            fits[m] = fitter(m)
        except Exception:
            continue
    tc = ResultsTable(name="modelCompare", title="모형 비교")
    tc.add_column("model", "모형", col_type="text")
    tc.add_column("ll", "로그우도", col_type="number")
    tc.add_column("k", "모수 수", col_type="integer")
    tc.add_column("aic", "AIC", col_type="number")
    tc.add_column("bic", "BIC", col_type="number")
    tc.add_column("rho", "ρ", col_type="number")
    tc.add_column("lam", "λ", col_type="number")
    best = min(fits, key=lambda m: fits[m].aic) if fits else None
    for m, f in fits.items():
        tc.add_row({"model": sm.MODEL_LABELS[m] + (" ◀ AIC 최소" if m == best else ""), "ll": f.loglik, "k": f.n_params,
                    "aic": f.aic, "bic": f.bic, "rho": f.rho if f.rho is not None else "", "lam": f.lam if f.lam is not None else ""})
    tc.add_note("같은 표본·가중행렬·효과 구조에서 추정. AIC·BIC는 작을수록 적합도가 높음.")
    tr = ResultsTable(name="lrTests", title="우도비(LR) 검정: 제약모형 대 일반모형")
    tr.add_column("restricted", "제약모형", col_type="text")
    tr.add_column("general", "일반모형", col_type="text")
    tr.add_column("h0", "귀무가설", col_type="text")
    tr.add_column("lr", "LR", col_type="number")
    tr.add_column("df", "자유도", col_type="integer")
    tr.add_column("p", "p", col_type="number", fmt="pvalue")
    tr.add_column("decision", f"판정(α={alpha:g})", col_type="text")
    for a, b, h in sm.LR_PAIRS:
        if a in fits and b in fits:
            lr, df, pv = sm.lr_test(fits[a], fits[b])
            tr.add_row({"restricted": a.upper(), "general": b.upper(), "h0": h, "lr": lr, "df": df, "p": pv,
                        "decision": "귀무가설 기각(일반모형 지지)" if pv < alpha else "기각 못함(제약모형 유지 가능)"})
    tr.add_note("SDM 대 SAR·SEM 검정이 모두 기각되면 SDM이 지지됨(Elhorst 2010; LeSage & Pace 2009).")
    return tc, tr, fits


def _fit_summary_rows(fit: sm.FitResult, extra: List[Tuple[str, Any]] = ()) -> ResultsTable:
    t = ResultsTable(name="spSummary", title="모형 적합도")
    t.add_column("stat", "지표", col_type="text")
    t.add_column("value", "값", col_type="number", fmt="auto")
    for k, v in [("관측 수", fit.n_obs), ("로그우도", fit.loglik), ("AIC", fit.aic), ("BIC", fit.bic),
                 ("유사 R²(관측·예측 상관 제곱)", fit.pseudo_r2), ("오차분산 σ²", fit.sigma2)] + list(extra):
        t.add_row({"stat": k, "value": float(v)})
    return t


def _narrative_effects(eff: List[Dict], dep: str) -> str:
    parts = []
    for r in eff:
        if r["effect"] != "간접효과" or not (np.isfinite(r["p"]) and r["p"] < 0.05):
            continue
        parts.append(f"{r['var']}의 간접효과 {r['est']:.3f}(p {'< .001' if r['p'] < 0.001 else '= ' + format(r['p'], '.3f')})")
    return ("유의한 공간 파급효과: " + ", ".join(parts) + ".") if parts else "유의한(p < .05) 간접효과는 확인되지 않았다."


def _check_map_style(p: Dict[str, Any]) -> None:
    """Validate map options before estimation so a typo does not fail after a long fit."""
    smap.resolve_scheme(_p(p, "scheme", default="quantiles"))
    smap.resolve_cmap(_p(p, "cmap", default="YlOrRd"))


def _model_from_params(p: Dict[str, Any]) -> str:
    m = str(_p(p, "model", "modelType", "model_type", default="sar")).lower()
    m = {"lag": "sar", "slm": "sar", "spregress_lag": "sar", "error": "sem", "durbin": "sdm", "sarar": "sac",
         "ols_lm": "ols", "lm": "ols", "linear": "ols", "moran_only": "moran"}.get(m, m)
    if m not in sm.MODEL_LABELS and m != "moran":
        raise ValueError(f"알 수 없는 모형: {m} (가능: {', '.join(sm.MODEL_LABELS)})")
    return m


# ---------------------------------------------------------------- 3. cross-section regression
class SpatialAnalysis(BaseAnalysis):
    """Cross-sectional spatial regression (Stata spregress, ml). Backward-compatible keyword arguments."""

    def __init__(self, data: pd.DataFrame, dep: str = "", covs: Optional[List[str]] = None, lat: Optional[str] = None,
                 lon: Optional[str] = None, region_id: Optional[str] = None, weights_method: Optional[str] = None,
                 k: int = 4, model_type: Optional[str] = None, plot_moran: bool = True, plot_map: bool = True, **params):
        p = dict(params)
        p.setdefault("dep", dep)
        p.setdefault("covs", covs or [])
        if lat and "coordY" not in p:
            p["coordY"] = lat
        if lon and "coordX" not in p:
            p["coordX"] = lon
        if region_id and "idVar" not in p and "labelVar" not in p:
            p["labelVar"] = region_id
        if weights_method and "wType" not in p:
            p["wType"] = weights_method
        p.setdefault("k", k)
        if model_type and "model" not in p:
            p["model"] = model_type
        p.setdefault("plotMoran", plot_moran)
        p.setdefault("plotMap", plot_map)
        super().__init__(name="spatialAnalysis", title="공간회귀분석 (횡단면 · Stata spregress)", dataset=data, options=p)
        self.syntax_python = "sess.spatial_regression(" + ", ".join(f"{k_}={v!r}" for k_, v in p.items() if v not in (None, "", [])) + ")"
        self.syntax_r = "spatialreg::lagsarlm / errorsarlm (listw = W)"
        self.run()

    def run(self):
        p = self.options
        _check_map_style(p)
        dep = str(_p(p, "dep", default=""))
        covs = _as_list(_p(p, "covs", default=[]))
        model = _model_from_params(p)
        if not dep or dep not in self.dataset.columns:
            raise ValueError("종속변수(dep)를 지정")
        for c in covs:
            if c not in self.dataset.columns:
                raise ValueError(f"설명변수 '{c}'가 데이터에 없음")
        if model == "moran":
            res = SpatialAutocorrelation(self.dataset, **{**p, "vars": [dep]})
            self.tables, self.plots, self.apa_narrative = res.tables, res.plots, res.apa_narrative
            return
        if not covs:
            raise ValueError("설명변수(covs)를 1개 이상 지정")
        notes: List[str] = []
        ctx = SpatialContext(self.dataset, p)
        cols = [dep] + covs
        base = _filter_period(self.dataset, p, notes)
        frame = _numeric(base, cols).dropna(subset=cols + ctx.required_columns())
        n_missing = len(base) - len(frame)
        frame, keys = _prepare_units(ctx, frame, notes)
        if keys.duplicated().any():
            raise ValueError("지역 식별자 중복: 지역당 1행 자료에 사용 — 연도별 반복 자료는 시점 변수를 지정하거나 「공간패널회귀」 사용")
        units = list(keys)
        geoms = ctx.geometries(units, frame, keys)
        W = _weights_from_params(p, geoms)
        pos = {k_: i for i, k_ in enumerate(units)}
        frame = frame.iloc[[pos[k_] for k_ in W.ids]]
        labels = ctx.unit_labels(W.ids, frame, keys.iloc[[pos[k_] for k_ in W.ids]])
        if n_missing:
            notes.append(f"결측으로 제외한 행 {n_missing}개")
        y = frame[dep].astype(float).values
        X = frame[covs].astype(float).values
        n = len(y)
        if n < len(covs) + 5:
            raise ValueError(f"관측 수 {n}: 설명변수 수에 비해 지역이 너무 적음")
        durbin = _as_list(_p(p, "durbinVars", "durbin", default=[])) or covs
        durbin = [d for d in durbin if d in covs]
        level = float(_p(p, "ciLevel", "level", default=0.95))
        alpha = 1 - level if level > 0.5 else 0.05
        se_type = resolve_choice("표준오차", _p(p, "seType", default="oim"), ["oim", "eim"], {"observed": "oim", "expected": "eim"})
        layout = sm.PanelLayout(n, 1)

        def fitter(m):
            f, _ = sm.fit_spec(m, y, X, covs, W, layout, durbin, True)
            return f

        fit, d = sm.fit_spec(model, y, X, covs, W, layout, durbin, True)
        if se_type == "eim" and model not in ("ols", "slx"):
            V = expected_information_vcov(fit, d, W)
            if V is not None:
                fit.vcov = V

        method = resolve_choice("효과 추론 방법", _p(p, "effectsMethod", default="delta"), ["delta", "sim"],
                                {"simulation": "sim", "mc": "sim", "montecarlo": "sim"})
        nsim = int(_p(p, "nsim", default=1000))
        lag, err, wx = sm.MODEL_PARTS[model]
        rows = _w_rows(W, ctx) + [
            ("모형", sm.MODEL_LABELS[model] + (f", W·X 대상: {', '.join(durbin)}" if wx else "")),
            ("추정 방법", "최우추정(ML)" if model not in ("ols", "slx") else "최소제곱(OLS)"),
            ("오차 가정", "정규분포·등분산(독립) 오차"),
            ("표준오차", {"oim": "관측정보행렬(OIM, Stata 기본)", "eim": "기대정보행렬(EIM, GeoDa·PySAL)"}.get(se_type, se_type) if model not in ("ols", "slx") else "OLS 고전적 표준오차"),
        ]
        self.add_table(_assumption_table(rows))
        self.add_table(_fit_summary_rows(fit))
        ct = _coef_table(fit, f"{sm.MODEL_LABELS[model]} 계수 — 종속변수: {dep}", level) if model not in ("ols", "slx") else _ols_table(fit, f"{sm.MODEL_LABELS[model]} 계수 — 종속변수: {dep}")
        if wx:
            ct.add_note("W·x 계수는 이웃 지역 설명변수의 평균 수준과의 관계. 한계효과 해석은 직접·간접·총효과 표 사용.")
        if lag:
            ct.add_note("ρ가 있는 모형의 계수는 공간 되먹임 때문에 한계효과와 다름 — 직접·간접·총효과 표 참고.")
        if d.dropped:
            ct.add_note(f"공선성으로 제외한 항: {', '.join(d.dropped)}")
        if not fit.hessian_ok:
            ct.add_note("정보행렬이 양정치가 아니어서 일반화 역행렬로 표준오차 계산 — 결과 해석에 주의.")
        for n_ in notes:
            ct.add_note(n_)
        self.add_table(ct)
        eff = []
        if _flag(_p(p, "effects"), True) and (lag or wx):
            eff = sm.effects(fit, W, method, nsim, int(_p(p, "seed", default=42)), level)
            self.add_table(_effects_table(eff, method, nsim, level))

        lm = None
        if _flag(_p(p, "diagnostics"), True):
            ols, d0 = sm.fit_spec("ols", y, X, covs, W, layout, [], True)
            td = ResultsTable(name="olsDiag", title="OLS 기준 진단")
            td.add_column("test", "검정", col_type="text")
            td.add_column("stat", "통계량", col_type="number")
            td.add_column("df", "자유도", col_type="number", fmt="auto")
            td.add_column("p", "p", col_type="number", fmt="pvalue")
            mi = sm.moran_residuals(ols.resid, d0.X, W)
            td.add_row({"test": "잔차 Moran's I", "stat": mi["I"], "df": "", "p": mi["p"]})
            for k_, (v_, df_, pv) in sm.ols_diagnostics(ols, d0).items():
                td.add_row({"test": k_ + ("(설명변수 기준)" if "Koenker" in k_ else ""), "stat": v_, "df": df_ if np.isfinite(df_) else "", "p": pv})
            td.add_note(f"OLS R² = {ols.extra['r2']:.3f}. 잔차 Moran's I의 기댓값 {mi['E']:.4f}, z = {mi['z']:.2f}. 조건수 30 이상은 공선성 의심.")
            self.add_table(td)
            lm = sm.lm_tests(d0, W, 1)
            tl = _lm_table(lm, "공간 의존성 LM 검정 (OLS 잔차 기준)")
            tl.add_note(_lm_guidance(lm, 0.05))
            self.add_table(tl)
        if _flag(_p(p, "compare"), True):
            tc, tr, _ = _compare(["ols", "slx", "sar", "sem", "sdm", "sdem", "sac"], fitter, 0.05)
            self.add_table(tc)
            self.add_table(tr)

        g = geoms.loc[W.ids]
        show_labels = _flag(_p(p, "showLabels"), False)
        scheme = smap.resolve_scheme(_p(p, "scheme", default="quantiles"))
        ncls = int(_p(p, "classes", default=5))
        cmap = smap.resolve_cmap(_p(p, "cmap", default="YlOrRd"))
        if _flag(_p(p, "plotMap"), True):
            self.add_plot(f"주제도 — {dep}", smap.plot_choropleth(g, y, f"{dep} 분포", scheme, ncls, cmap, dep,
                                                               labels=labels if show_labels else None))
        if _flag(_p(p, "plotResid"), True):
            self.add_plot("잔차 지도", smap.plot_choropleth(g, fit.resid, f"{sm.MODEL_LABELS[model]} 잔차", "quantiles", ncls, "RdBu_r", "잔차",
                                                        labels=labels if show_labels else None))
        if _flag(_p(p, "plotMoran"), True):
            z = (fit.resid - fit.resid.mean()) / fit.resid.std()
            mi_r = (z @ (W.W @ z)) / (z @ z) * n / float(W.W.sum())
            self.add_plot("잔차 Moran 산점도", smap.plot_moran_scatter(z, W.W @ z, mi_r, f"잔차 Moran 산점도 (I = {mi_r:.3f})", "표준화 잔차", "공간시차 W·잔차"))

        sp_par = ""
        if fit.rho is not None:
            i = fit.names.index("ρ (W·y)")
            sp_par += f" 공간시차 계수 ρ = {fit.rho:.3f}(SE = {fit.se[i]:.3f})."
        if fit.lam is not None:
            i = fit.names.index("λ (W·u)")
            sp_par += f" 공간오차 계수 λ = {fit.lam:.3f}(SE = {fit.se[i]:.3f})."
        self.apa_narrative = (f"추정 모형: {sm.MODEL_LABELS[model]}, 자료: {n}개 지역, 공간가중행렬: {W.spec.describe()}({NORM_LABELS[W.spec.norm]})."
                              f"{sp_par} 로그우도 {fit.loglik:.2f}, AIC {fit.aic:.2f}. " + (_narrative_effects(eff, dep) if eff else ""))
        self.fit, self.W = fit, W


def expected_information_vcov(fit: sm.FitResult, d: sm.Design, W: SpatialWeights) -> Optional[np.ndarray]:
    """Expected (Fisher) information for the cross-sectional SARAR family (Anselin 1988), as in GeoDa / PySAL."""
    n = W.n
    if len(d.y) != n:
        return None
    Wd = W.dense()
    X = d.X
    k = X.shape[1]
    rho = fit.rho or 0.0
    lam = fit.lam or 0.0
    s2 = fit.sigma2
    beta = fit.coef[:k]
    I_n = np.eye(n)
    A_inv = np.linalg.inv(I_n - rho * Wd)
    B = I_n - lam * Wd
    B_inv = np.linalg.inv(B)
    G = Wd @ A_inv
    Gt = B @ G @ B_inv
    H = Wd @ B_inv
    BX = B @ X
    BGXb = B @ G @ X @ beta
    has_l, has_e = fit.rho is not None, fit.lam is not None
    p = k + int(has_l) + int(has_e) + 1
    I = np.zeros((p, p))
    I[:k, :k] = BX.T @ BX / s2
    j = k
    if has_l:
        I[:k, j] = I[j, :k] = BX.T @ BGXb / s2
        I[j, j] = np.trace(Gt @ Gt) + np.trace(Gt.T @ Gt) + float(BGXb @ BGXb) / s2
        I[j, -1] = I[-1, j] = np.trace(Gt) / s2
        jl = j
        j += 1
    if has_e:
        I[j, j] = np.trace(H @ H) + np.trace(H.T @ H)
        I[j, -1] = I[-1, j] = np.trace(H) / s2
        if has_l:
            I[jl, j] = I[j, jl] = np.trace(H.T @ Gt) + np.trace(H @ Gt)
    I[-1, -1] = n / (2 * s2 ** 2)
    try:
        V = np.linalg.inv(I)
    except np.linalg.LinAlgError:
        return None
    return V[:p - 1, :p - 1]


# ---------------------------------------------------------------- 4. panel regression
class SpatialPanelAnalysis(BaseAnalysis):
    """Spatial panel regression (Stata xsmle / spxtregress)."""

    def __init__(self, data: pd.DataFrame, **params):
        super().__init__(name="spatialPanel", title="공간패널회귀 (Stata xsmle · spxtregress)", dataset=data, options=dict(params))
        self.syntax_python = "sess.spatial_panel(" + ", ".join(f"{k}={v!r}" for k, v in params.items() if v not in (None, "", [])) + ")"
        self.run()

    def run(self):
        p = self.options
        _check_map_style(p)
        dep = str(_p(p, "dep", default=""))
        covs = _as_list(_p(p, "covs", default=[]))
        time_col = str(_p(p, "timeVar", "time", default=""))
        model = _model_from_params({"model": _p(p, "model", default="sdm")})
        if not dep or dep not in self.dataset.columns:
            raise ValueError("종속변수(dep)를 지정")
        if not covs:
            raise ValueError("설명변수(covs)를 1개 이상 지정")
        if not time_col or time_col not in self.dataset.columns:
            raise ValueError("시간 변수(timeVar)를 지정")
        ctx = SpatialContext(self.dataset, p)
        if not ctx.id_col:
            raise ValueError("지역 식별 변수(idVar)를 지정")
        eff_type = resolve_choice("패널 효과", _p(p, "effectsType", "panelEffects", default="fe"), ["fe", "re", "pooled"],
                                  {"fixed": "fe", "random": "re", "none": "pooled", "ols": "pooled"})
        fe = resolve_choice("고정효과 유형", _p(p, "feType", "type", default="ind"), ["ind", "time", "both", "none"],
                            {"individual": "ind", "entity": "ind", "region": "ind", "id": "ind", "period": "time",
                             "twoway": "both", "two_way": "both", "two-way": "both"})
        if eff_type in ("pooled", "none"):
            fe = "none"
        leeyu = _flag(_p(p, "leeyu"), True)
        re_ = eff_type == "re"
        if re_ and model not in ("ols", "slx", "sar", "sdm"):
            raise ValueError("확률효과는 OLS·SLX·SAR·SDM 모형만 지원 — 공간오차 계열(SEM·SDEM·SAC)은 고정효과 선택")
        notes: List[str] = []
        cols = [dep] + covs
        frame = _numeric(self.dataset, cols).dropna(subset=cols + ctx.required_columns() + [time_col])
        n_missing = len(self.dataset) - len(frame)
        frame, keys = _prepare_units(ctx, frame, notes)
        frame = frame.assign(_k=keys.values, _t=frame[time_col].values)
        if frame.duplicated(["_k", "_t"]).any():
            dup = frame[frame.duplicated(["_k", "_t"], keep=False)][["_k", "_t"]].head(5).values.tolist()
            raise ValueError(f"같은 지역·시점의 중복 행 존재(예: {dup})")
        periods = sorted(frame["_t"].unique())
        T = len(periods)
        if T < 2:
            raise ValueError("시점이 2개 이상 필요")
        counts = frame.groupby("_k")["_t"].nunique()
        incomplete = counts[counts < T].index.tolist()
        if incomplete:
            if _flag(_p(p, "dropIncomplete"), True):
                notes.append(f"모든 시점이 관측되지 않은 지역 {len(incomplete)}곳 제외(균형패널 구성): {', '.join(map(str, incomplete[:8]))}{' …' if len(incomplete) > 8 else ''}")
                frame = frame[~frame["_k"].isin(incomplete)]
            else:
                raise ValueError(f"불균형 패널: 일부 시점이 빠진 지역 {len(incomplete)}곳 — 「불완전 지역 제외」를 선택하거나 자료 보완")
        units = _sort_keys(frame["_k"].unique().tolist())
        geoms = ctx.geometries(units, frame, frame["_k"])
        W = _weights_from_params(p, geoms)
        units = W.ids
        frame = frame[frame["_k"].isin(units)]
        N = len(units)
        frame = frame.set_index(["_t", "_k"]).loc[[(t, u) for t in periods for u in units]].reset_index()
        labels = ctx.unit_labels(units, frame, frame["_k"])
        y = frame[dep].astype(float).values
        X = frame[covs].astype(float).values
        durbin = _as_list(_p(p, "durbinVars", "durbin", default=[])) or covs
        durbin = [d for d in durbin if d in covs]
        level = float(_p(p, "ciLevel", "level", default=0.95))
        method = resolve_choice("효과 추론 방법", _p(p, "effectsMethod", default="delta"), ["delta", "sim"],
                                {"simulation": "sim", "mc": "sim", "montecarlo": "sim"})
        nsim = int(_p(p, "nsim", default=1000))
        seed = int(_p(p, "seed", default=42))
        lag, err, wx = sm.MODEL_PARTS[model]
        layout = sm.PanelLayout(N, T, fe=("none" if re_ else fe), leeyu=leeyu)
        row_std = sm._row_std(W)
        if leeyu and fe in ("time", "both") and not row_std and not re_:
            notes.append("Lee–Yu 시점효과 보정은 행 표준화 행렬에서만 적용 — 보정 없이 추정")

        if re_:
            fit = sm.fit_random_effects(y, X, covs, W, N, T, durbin if wx else [], lag)
            fit.model = model
        else:
            fit, d = sm.fit_spec(model, y, X, covs, W, layout, durbin, fe == "none")

        eff_label = "확률효과(개체)" if re_ else sm.FE_LABELS[fe]
        rows = _w_rows(W, ctx) + [
            ("패널 구조", f"지역 {N}곳 × 시점 {T}개 = {N * T:,}관측 (시점: {periods[0]}–{periods[-1]})"),
            ("모형", sm.MODEL_LABELS[model] + (f", W·X 대상: {', '.join(durbin)}" if wx else "")),
            ("패널 효과", eff_label),
            ("편의 보정", ("Lee–Yu(2010) 변환 방식" if (leeyu and fe != "none" and not re_ and (fe == "ind" or row_std)) else "적용하지 않음") if not re_ and fe != "none" else "해당 없음"),
            ("추정 방법", "최우추정(ML)" if model not in ("ols", "slx") or re_ else "최소제곱(OLS, 고정효과 변환)"),
            ("표준오차", "관측정보행렬(OIM)" if (model not in ("ols", "slx") or re_) else "OLS 고전적 표준오차"),
            ("오차 가정", "정규분포·등분산 오차, 시점 간 독립"),
        ]
        self.add_table(_assumption_table(rows))
        extra = [("지역 수 N", N), ("시점 수 T", T)]
        if re_:
            extra += [("개체효과 분산 σ²_μ", fit.extra.get("sigma_mu2", np.nan)), ("φ = σ²/(Tσ²_μ+σ²)", fit.extra.get("phi", np.nan))]
        self.add_table(_fit_summary_rows(fit, extra))
        title = f"{sm.MODEL_LABELS[model]} · {eff_label} 계수 — 종속변수: {dep}"
        ct = _coef_table(fit, title, level) if (model not in ("ols", "slx") or re_) else _ols_table(fit, title)
        if lag:
            ct.add_note("ρ가 있는 모형의 계수는 공간 되먹임 때문에 한계효과와 다름 — 직접·간접·총효과 표 참고.")
        dropped = fit.extra.get("dropped") or []
        if dropped:
            ct.add_note(f"고정효과 변환·공선성으로 제외한 항(시간 불변 변수 등): {', '.join(dropped)}")
        if n_missing:
            notes.append(f"결측으로 제외한 행 {n_missing}개")
        for n_ in notes:
            ct.add_note(n_)
        self.add_table(ct)
        eff = []
        if _flag(_p(p, "effects"), True) and (lag or wx):
            eff = sm.effects(fit, W, method, nsim, seed, level)
            self.add_table(_effects_table(eff, method, nsim, level))

        if _flag(_p(p, "hausman"), False) and model in ("ols", "slx", "sar", "sdm"):
            fe_layout = sm.PanelLayout(N, T, fe="ind", leeyu=leeyu)
            f_fe, _ = sm.fit_spec(model, y, X, covs, W, fe_layout, durbin, False)
            f_re = fit if re_ else sm.fit_random_effects(y, X, covs, W, N, T, durbin if wx else [], lag)
            hz = sm.hausman(f_fe, f_re)
            th = ResultsTable(name="hausman", title="Hausman 검정: 개체 고정효과 대 확률효과")
            th.add_column("stat", "χ²", col_type="number")
            th.add_column("df", "자유도", col_type="integer")
            th.add_column("p", "p", col_type="number", fmt="pvalue")
            th.add_column("decision", "판정", col_type="text")
            th.add_row({"stat": hz["stat"], "df": hz["df"], "p": hz["p"],
                        "decision": "확률효과 기각(고정효과 지지)" if hz["p"] < 0.05 else "확률효과 기각 못함"})
            th.add_note("비교 모형: 지역 고정효과(" + ("Lee–Yu 보정" if leeyu else "보정 없음") + ") 대 확률효과(시점 효과 없음) — 선택한 고정효과 유형과 무관.")
            th.add_note("비교 모수: " + ", ".join(hz["vars"]) + (". 분산 차이 행렬이 양정치가 아니어서 일반화 역행렬 사용." if not hz["pd"] else "."))
            self.add_table(th)

        if _flag(_p(p, "diagnostics"), True) and not re_:
            d0 = sm.build_design(y, X, covs, W, layout, [], fe == "none")
            lm = sm.lm_tests(d0, W, T)
            tl = _lm_table(lm, f"패널 LM 검정 (비공간 {eff_label} 모형 잔차, Elhorst 2014)")
            tl.add_note(_lm_guidance(lm, 0.05))
            self.add_table(tl)
        if _flag(_p(p, "compare"), True) and not re_:
            def fitter(m):
                f, _ = sm.fit_spec(m, y, X, covs, W, layout, durbin, fe == "none")
                return f
            tc, tr, _ = _compare(["ols", "slx", "sar", "sem", "sdm", "sdem", "sac"], fitter, 0.05)
            self.add_table(tc)
            self.add_table(tr)

        g = geoms.loc[units]
        show_labels = _flag(_p(p, "showLabels"), False)
        cmap = smap.resolve_cmap(_p(p, "cmap", default="YlOrRd"))
        ncls = int(_p(p, "classes", default=5))
        if _flag(_p(p, "plotMap"), True):
            mean_y = y.reshape(T, N).mean(axis=0)
            self.add_plot(f"주제도 — {dep} 기간 평균", smap.plot_choropleth(g, mean_y, f"{dep} 평균({periods[0]}–{periods[-1]})",
                                                                     smap.resolve_scheme(_p(p, "scheme", default="quantiles")), ncls, cmap, dep,
                                                                     labels=labels if show_labels else None))
        if _flag(_p(p, "plotResid"), True) and len(fit.resid) == N * T:
            mean_r = fit.resid.reshape(T, N).mean(axis=0)
            self.add_plot("지역별 평균 잔차 지도", smap.plot_choropleth(g, mean_r, "지역별 평균 잔차", "quantiles", ncls, "RdBu_r", "잔차",
                                                                labels=labels if show_labels else None))
        sp_par = ""
        if fit.rho is not None:
            sp_par = f" 공간시차 계수 ρ = {fit.rho:.3f}(SE = {fit.se[fit.names.index('ρ (W·y)')]:.3f})."
        if fit.lam is not None:
            sp_par += f" 공간오차 계수 λ = {fit.lam:.3f}(SE = {fit.se[fit.names.index('λ (W·u)')]:.3f})."
        self.apa_narrative = (f"추정 모형: {sm.MODEL_LABELS[model]}, 패널 효과: {eff_label}, 자료: {N}개 지역 × {T}개 시점 균형패널."
                              f"{sp_par} " + (_narrative_effects(eff, dep) if eff else ""))
        self.fit, self.W = fit, W


# ---------------------------------------------------------------- 5. GWR
class GWRAnalysis(BaseAnalysis):
    def __init__(self, data: pd.DataFrame, **params):
        super().__init__(name="gwr", title="지리가중회귀 (GWR)", dataset=data, options=dict(params))
        self.syntax_python = "sess.gwr(" + ", ".join(f"{k}={v!r}" for k, v in params.items() if v not in (None, "", [])) + ")"
        self.run()

    def run(self):
        try:
            from mgwr.gwr import GWR
            from mgwr.sel_bw import Sel_BW
        except ImportError as e:  # pragma: no cover
            raise ValueError("지리가중회귀에 mgwr 패키지 필요: 'pip install -r requirements.txt'") from e
        from ..spatial_io import to_metric
        import geopandas as gpd
        p = self.options
        dep = str(_p(p, "dep", default=""))
        covs = _as_list(_p(p, "covs", default=[]))
        if not dep or not covs:
            raise ValueError("종속변수와 설명변수를 지정")
        notes: List[str] = []
        ctx = SpatialContext(self.dataset, p)
        cols = [dep] + covs
        frame = _numeric(_filter_period(self.dataset, p, notes), cols).dropna(subset=cols + ctx.required_columns())
        frame, keys = _prepare_units(ctx, frame, notes)
        if keys.duplicated().any():
            raise ValueError("지역 식별자 중복: 지역당 1행 자료에 사용 — 연도별 반복 자료는 시점 변수 지정")
        units = list(keys)
        geoms = ctx.geometries(units, frame, keys)
        labels = ctx.unit_labels(units, frame, keys)
        gdf_m, unit = to_metric(gpd.GeoDataFrame(geometry=list(geoms.values), crs=geoms.crs))
        pts = gdf_m.geometry.representative_point() if not gdf_m.geom_type.isin(["Point"]).all() else gdf_m.geometry
        scale = 1000.0 if unit == "km" else 1.0
        coords = np.column_stack([pts.x.values, pts.y.values]) / scale
        y = frame[dep].astype(float).values.reshape(-1, 1)
        X = frame[covs].astype(float).values
        n = len(y)
        if n < len(covs) + 10:
            raise ValueError(f"관측 수 {n}: 지리가중회귀에는 더 많은 지역 필요")
        standardize = _flag(_p(p, "standardize"), False)
        if standardize:
            X = (X - X.mean(0)) / X.std(0)
            y = (y - y.mean()) / y.std()
        kernel = resolve_choice("커널", _p(p, "kernel", default="bisquare"), ["bisquare", "gaussian", "exponential"],
                                {"bi-square": "bisquare", "gauss": "gaussian", "exp": "exponential"})
        fixed = resolve_choice("대역폭 유형", _p(p, "bwType", default="adaptive"), ["adaptive", "fixed"]) == "fixed"
        criterion = resolve_choice("대역폭 선택 기준", _p(p, "criterion", default="AICc"), ["AICc", "AIC", "BIC", "CV"])
        bw_manual = float(_p(p, "bandwidth", default=0) or 0)
        sel = Sel_BW(coords, y, X, kernel=kernel, fixed=fixed)
        if bw_manual > 0:
            bw = bw_manual if fixed else int(round(bw_manual))
        else:
            kw = {} if fixed else {"bw_min": float(min(n - 1, X.shape[1] + 3)), "bw_max": float(n - 1)}
            bw = sel.search(criterion=criterion, **kw)
        res = GWR(coords, y, X, bw, kernel=kernel, fixed=fixed).fit()
        names = ["(절편)"] + covs

        # global OLS on the same scale
        Xc = np.column_stack([np.ones(n), X])
        b_ols = np.linalg.lstsq(Xc, y.ravel(), rcond=None)[0]
        e = y.ravel() - Xc @ b_ols
        k = Xc.shape[1]
        s2 = float(e @ e) / n
        aicc_ols = n * np.log(s2) + n * np.log(2 * np.pi) + n * (n + k) / (n - 2 - k)
        r2_ols = 1 - float(e @ e) / float(((y - y.mean()) ** 2).sum())

        alpha = float(_p(p, "alpha", default=0.05))
        tv = res.filter_tvals(alpha=alpha)
        sig = tv != 0
        rows = [("공간 정보", ctx.describe()), ("커널", {"bisquare": "이중제곱(bisquare)", "gaussian": "가우시안", "exponential": "지수"}.get(kernel, kernel)),
                ("대역폭 유형", "고정(거리)" if fixed else "적응형(최근접 이웃 수)"),
                ("대역폭 선택", f"직접 지정 {bw_manual:g}" if bw_manual > 0 else f"{criterion} 최소화(황금분할 탐색)"),
                ("거리 단위", unit), ("변수 표준화", "적용" if standardize else "적용하지 않음"),
                ("국지 계수 유의성", f"α = {alpha:g}, 유효 모수 수(ENP)로 보정한 다중검정 기준(da Silva & Fotheringham 2016)")]
        self.add_table(_assumption_table(rows))
        ts = ResultsTable(name="gwrSummary", title="지리가중회귀 적합도")
        ts.add_column("stat", "지표", col_type="text")
        ts.add_column("gwr", "GWR", col_type="number", fmt="auto")
        ts.add_column("ols", "전역 OLS", col_type="number", fmt="auto")
        ts.add_row({"stat": "대역폭" + (f" ({unit})" if fixed else " (이웃 수)"), "gwr": float(bw), "ols": ""})
        ts.add_row({"stat": "유효 모수 수(ENP)", "gwr": float(res.ENP), "ols": float(k)})
        ts.add_row({"stat": "AICc", "gwr": float(res.aicc), "ols": float(aicc_ols)})
        ts.add_row({"stat": "R²", "gwr": float(res.R2), "ols": float(r2_ols)})
        ts.add_row({"stat": "조정 R²", "gwr": float(res.adj_R2), "ols": float(1 - (1 - r2_ols) * (n - 1) / (n - k))})
        ts.add_row({"stat": "잔차 분산 σ²", "gwr": float(res.sigma2), "ols": float(e @ e) / (n - k)})
        ts.add_note(f"관측 수 {n}. AICc 차이가 3 이상이면 GWR 적합 개선으로 해석하는 관행 있음(Fotheringham et al. 2002).")
        for n_ in notes:
            ts.add_note(n_)
        self.add_table(ts)
        tl = ResultsTable(name="gwrLocal", title="국지 계수 요약")
        tl.add_column("term", "변수", col_type="text")
        tl.add_column("ols", "전역 OLS", col_type="number")
        tl.add_column("mean", "평균", col_type="number")
        tl.add_column("sd", "표준편차", col_type="number")
        tl.add_column("min", "최솟값", col_type="number")
        tl.add_column("med", "중앙값", col_type="number")
        tl.add_column("max", "최댓값", col_type="number")
        tl.add_column("pos", "유의(+) 지역", col_type="integer")
        tl.add_column("neg", "유의(−) 지역", col_type="integer")
        for j, nm in enumerate(names):
            col = res.params[:, j]
            tl.add_row({"term": nm, "ols": float(b_ols[j]), "mean": float(col.mean()), "sd": float(col.std(ddof=1)), "min": float(col.min()),
                        "med": float(np.median(col)), "max": float(col.max()), "pos": int(((tv[:, j] > 0)).sum()), "neg": int(((tv[:, j] < 0)).sum())})
        self.add_table(tl)
        if _flag(_p(p, "mcTest"), False):
            iters = int(_p(p, "mcIters", default=50))
            pv = res.spatial_variability(sel, iters)
            tm = ResultsTable(name="gwrMC", title="국지 계수 공간 변이 검정 (몬테카를로)")
            tm.add_column("term", "변수", col_type="text")
            tm.add_column("p", "p", col_type="number", fmt="pvalue")
            for nm, v in zip(names, pv):
                tm.add_row({"term": nm, "p": float(v)})
            tm.add_note(f"좌표 무작위 재배치 {iters}회. p가 작을수록 계수가 지역에 따라 달라진다는 근거.")
            self.add_table(tm)
        g = geoms
        show_labels = _flag(_p(p, "showLabels"), False)
        if _flag(_p(p, "plotCoef"), True):
            mask = _flag(_p(p, "maskInsig"), True)
            for j, nm in enumerate(names):
                if nm == "(절편)" and not _flag(_p(p, "plotIntercept"), False):
                    continue
                self.add_plot(f"국지 계수 지도 — {nm}", smap.plot_coefficient_map(
                    g, res.params[:, j], sig[:, j] if mask else None, f"국지 계수 — {nm}", "계수", labels=labels if show_labels else None))
        if _flag(_p(p, "plotR2"), True):
            self.add_plot("국지 R² 지도", smap.plot_choropleth(g, res.localR2.ravel(), "국지 R²", "quantiles", 5, "Greens", "국지 R²",
                                                          labels=labels if show_labels else None))
        self.apa_narrative = (f"{n}개 지역 자료로 지리가중회귀를 추정하였다({'고정' if fixed else '적응형'} {kernel} 커널, 대역폭 {bw:g}). "
                              f"AICc는 GWR {res.aicc:.2f}, 전역 OLS {aicc_ols:.2f}이고 R²는 {res.R2:.3f}이다.")


# ---------------------------------------------------------------- 6. thematic map
class SpatialMapAnalysis(BaseAnalysis):
    def __init__(self, data: pd.DataFrame, **params):
        super().__init__(name="spatialMap", title="주제도 (단계구분도)", dataset=data, options=dict(params))
        self.syntax_python = "sess.spatial_map(" + ", ".join(f"{k}={v!r}" for k, v in params.items() if v not in (None, "", [])) + ")"
        self.run()

    def run(self):
        p = self.options
        _check_map_style(p)
        vars_ = _as_list(_p(p, "vars", "dep"))
        if not vars_:
            raise ValueError("지도에 표시할 변수(vars)를 지정")
        notes: List[str] = []
        ctx = SpatialContext(self.dataset, p)
        frame = _numeric(_filter_period(self.dataset, p, notes), vars_).dropna(subset=ctx.required_columns())
        frame, keys = _prepare_units(ctx, frame, notes)
        if keys.duplicated().any():
            raise ValueError("지역 식별자 중복: 지역당 1행 자료에서 실행 — 연도별 반복 자료는 시점 변수 지정")
        units = list(keys)
        geoms = ctx.geometries(units, frame, keys)
        labels = ctx.unit_labels(units, frame, keys)
        scheme = smap.resolve_scheme(_p(p, "scheme", default="quantiles"))
        ncls = int(_p(p, "classes", default=5))
        cmap = smap.resolve_cmap(_p(p, "cmap", default="YlOrRd"))
        show_labels = _flag(_p(p, "showLabels"), False)
        t = ResultsTable(name="mapClasses", title="계급 구분")
        t.add_column("var", "변수", col_type="text")
        t.add_column("cls", "계급", col_type="integer")
        t.add_column("lo", "하한", col_type="number")
        t.add_column("hi", "상한", col_type="number")
        t.add_column("n", "지역 수", col_type="integer")
        for v in vars_:
            vals = frame[v].astype(float).values
            fin = vals[np.isfinite(vals)]
            if fin.size == 0:
                continue
            bins = smap.classify(fin, scheme, ncls)
            for i, r in enumerate(smap.class_table(vals, bins)):
                t.add_row({"var": v if i == 0 else "", **r})
            self.add_plot(f"주제도 — {v}", smap.plot_choropleth(geoms, vals, str(_p(p, "title", default="") or v), scheme, ncls, cmap, v,
                                                             labels=labels if show_labels else None))
        t.add_note(f"계급 구분: {smap.SCHEME_LABELS.get(scheme, scheme)}, {ncls}계급. " + ctx.describe() + ".")
        for n_ in notes:
            t.add_note(n_)
        self.add_table(t)
        self.apa_narrative = ""

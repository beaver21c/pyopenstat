"""
Spatial weights matrix W for regional data (GeoDa / PySAL / Stata spmatrix equivalent).

Construction rules (all assumptions are explicit options):
- contiguity: queen (shared edge or vertex) / rook (shared edge), order 1..3, optionally cumulative (lower orders included)
- k-nearest neighbours on centroids, distance band (binary), inverse distance d^-alpha (optional cut-off)
- external matrix file (.gal / .gwt / .csv square matrix)
- units without neighbours (islands): connect to the nearest unit / drop from the analysis / keep as zero rows
- normalisation: row-standardised, spectral (largest |eigenvalue| = 1, Stata default), min-max (Kelejian-Prucha), none
Distances are in km after projecting lon/lat to the local UTM zone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

WEIGHT_KIND_LABELS = {
    "queen": "퀸 인접(경계선·꼭짓점 공유)",
    "rook": "룩 인접(경계선 공유)",
    "knn": "k-최근접 이웃",
    "distance": "거리 임계값(이진)",
    "idw": "역거리 가중",
    "file": "외부 가중행렬 파일",
}
NORM_LABELS = {
    "row": "행 표준화",
    "spectral": "스펙트럼 정규화(최대 고유값 1)",
    "minmax": "최소-최대 정규화",
    "none": "정규화 없음",
}
ISLAND_LABELS = {"nearest": "가장 가까운 지역 1곳과 연결", "drop": "분석에서 제외", "keep": "이웃 없는 상태로 유지"}
_KIND_ALIASES = {
    "inverse_distance": "idw", "inverse": "idw", "idistance": "idw", "k_nearest": "knn", "kneighbors": "knn",
    "contiguity": "queen", "band": "distance", "distance_band": "distance",
}
_NORM_ALIASES = {
    "binary": "none", "raw": "none", "unstandardized": "none", "b": "none",
    "w": "row", "r": "row", "s": "spectral", "m": "minmax",
}


def _f(p: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in p and p[k] not in (None, ""):
            return p[k]
    return default


def _flag(v, default=False) -> bool:
    if v is None or v == "":
        return default
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


@dataclass
class WeightsSpec:
    kind: str = "queen"
    order: int = 1
    lower: bool = True
    k: int = 4
    threshold: float = 0.0  # km; 0 = smallest distance giving every unit at least one neighbour
    power: float = 1.0
    norm: str = "row"
    islands: str = "nearest"
    symmetric: bool = False
    file: str = ""

    @classmethod
    def from_params(cls, p: Dict[str, Any], default_kind: str = "queen") -> "WeightsSpec":
        kind = str(_f(p, "wType", "weightsMethod", "weights_method", "w_type", default=default_kind)).lower()
        kind = _KIND_ALIASES.get(kind, kind)
        if kind not in WEIGHT_KIND_LABELS and kind != "auto":
            raise ValueError(f"알 수 없는 가중행렬 방식: {kind} (가능: {', '.join(WEIGHT_KIND_LABELS)})")
        norm = str(_f(p, "wNorm", "normalize", "w_norm", default="row")).lower()
        norm = _NORM_ALIASES.get(norm, norm)
        if norm not in NORM_LABELS:
            raise ValueError(f"알 수 없는 정규화 방식: {norm}")
        islands = str(_f(p, "wIslands", "islands", default="nearest")).lower()
        if islands not in ISLAND_LABELS:
            raise ValueError(f"알 수 없는 고립 지역 처리 방식: {islands}")
        return cls(
            kind=kind,
            order=int(np.clip(int(float(_f(p, "wOrder", "order", default=1))), 1, 5)),
            lower=_flag(_f(p, "wLower", "lower_order"), True),
            k=int(np.clip(int(float(_f(p, "k", "wK", default=4))), 1, 200)),
            threshold=float(_f(p, "wThreshold", "threshold", default=0.0) or 0.0),
            power=float(_f(p, "wPower", "power", default=1.0) or 1.0),
            norm=norm,
            islands=islands,
            symmetric=_flag(_f(p, "wSymmetric", "symmetric"), False),
            file=str(_f(p, "wFile", "w_file", default="") or ""),
        )

    def describe(self) -> str:
        if self.kind in ("queen", "rook"):
            base = WEIGHT_KIND_LABELS[self.kind]
            if self.order > 1:
                base += f", {self.order}차 인접" + ("(하위 차수 포함)" if self.lower else "(해당 차수만)")
        elif self.kind == "knn":
            base = f"k-최근접 이웃(k={self.k})"
        elif self.kind == "distance":
            base = "거리 임계값(이진)"
        elif self.kind == "idw":
            base = f"역거리 가중(지수 {self.power:g})"
        else:
            base = f"외부 파일({os.path.basename(self.file)})"
        return base


def _centroids(geoms) -> np.ndarray:
    pts = geoms.representative_point() if not all(geoms.geom_type == "Point") else geoms
    return np.column_stack([pts.x.values, pts.y.values])


class SpatialWeights:
    """Spatial weights over an ordered list of unit keys (raw and normalised sparse matrices)."""

    def __init__(self, ids: List[str], raw: sp.csr_matrix, spec: WeightsSpec, unit: str, coords: Optional[np.ndarray],
                 islands: List[str], island_action: str, threshold: Optional[float] = None, dropped: Optional[List[str]] = None):
        self.ids = list(ids)
        self.raw = raw.tocsr()
        self.spec = spec
        self.unit = unit
        self.coords = coords
        self.islands = islands
        self.island_action = island_action
        self.threshold = threshold
        self.dropped = dropped or []
        self.W = self._normalise(self.raw, spec.norm)
        self._dense = None
        self._spectrum = None

    # ---------- construction helpers ----------
    @staticmethod
    def _normalise(raw: sp.csr_matrix, norm: str) -> sp.csr_matrix:
        raw = raw.tocsr().astype(float)
        if norm == "row":
            rs = np.asarray(raw.sum(axis=1)).ravel()
            inv = np.divide(1.0, rs, out=np.zeros_like(rs), where=rs > 0)
            return sp.diags(inv) @ raw
        if norm == "minmax":
            # Kelejian & Prucha (2010); Stata spmatrix normalize(minmax): divide by the smaller of the
            # largest row sum and the largest column sum of absolute values
            a = abs(raw)
            m = min(float(np.asarray(a.sum(axis=1)).max()), float(np.asarray(a.sum(axis=0)).max()))
            return raw / (m if m > 0 else 1.0)
        if norm == "spectral":
            n = raw.shape[0]
            if n <= 2500:
                ev = np.linalg.eigvals(raw.toarray())
                lam = float(np.max(np.abs(ev)))
            else:
                from scipy.sparse.linalg import eigs
                lam = float(np.abs(eigs(raw, k=1, which="LM", return_eigenvectors=False))[0])
            return raw / (lam if lam > 0 else 1.0)
        return raw

    @property
    def n(self) -> int:
        return len(self.ids)

    def dense(self) -> np.ndarray:
        if self._dense is None:
            self._dense = self.W.toarray()
        return self._dense

    def lag(self, v: np.ndarray) -> np.ndarray:
        """W applied period by period to a vector/matrix stacked as T blocks of n rows."""
        v = np.asarray(v, dtype=float)
        n = self.n
        if v.ndim == 1:
            return (self.W @ v.reshape(-1, n).T).T.reshape(-1)
        k = v.shape[1]
        out = np.empty_like(v)
        for j in range(k):
            out[:, j] = (self.W @ v[:, j].reshape(-1, n).T).T.reshape(-1)
        return out

    # ---------- eigen structure for log-determinants and effects ----------
    def spectrum(self) -> Dict[str, Any]:
        """Eigenvalues omega of W and weights a_i with 1'V diag(f(omega)) V^-1 1 = sum a_i f(omega_i)."""
        if self._spectrum is not None:
            return self._spectrum
        n = self.n
        if n > 6000:
            raise ValueError(f"공간 단위 {n:,}개: 최우추정은 6,000개 이하 지원")
        raw = self.raw
        sym_raw = (abs(raw - raw.T) > 1e-12).nnz == 0
        rs = np.asarray(raw.sum(axis=1)).ravel()
        one = np.ones(n)
        if sym_raw and self.spec.norm == "row" and np.all(rs > 0):
            d = 1.0 / np.sqrt(rs)
            S = (sp.diags(d) @ raw @ sp.diags(d)).toarray()
            lam, Q = np.linalg.eigh((S + S.T) / 2.0)
            left = (one * d) @ Q          # 1' D^-1/2 Q
            right = Q.T @ (one / d)       # Q' D^1/2 1
            omega = lam.astype(complex)
            a = (left * right).astype(complex)
            kind = "symmetric-similar"
        elif sym_raw:
            Wd = self.dense()
            lam, Q = np.linalg.eigh((Wd + Wd.T) / 2.0)
            omega = lam.astype(complex)
            a = ((one @ Q) * (Q.T @ one)).astype(complex)
            kind = "symmetric"
        else:
            Wd = self.dense()
            omega, V = np.linalg.eig(Wd)
            Vinv = np.linalg.inv(V)
            a = (one @ V) * (Vinv @ one)
            kind = "general"
        re = omega.real
        lo = 1.0 / re.min() if re.min() < -1e-10 else -1.0
        hi = 1.0 / re.max() if re.max() > 1e-10 else 1.0
        self._spectrum = {"omega": omega, "a": a, "kind": kind, "bounds": (max(lo, -1.0) + 1e-6, min(hi, 1.0) - 1e-6)}
        return self._spectrum

    def logdet(self, rho: float) -> float:
        if rho == 0.0:
            return 0.0
        om = self.spectrum()["omega"]
        val = np.sum(np.log(1.0 - rho * om))
        return float(val.real)

    # ---------- summaries ----------
    def neighbour_counts(self) -> np.ndarray:
        return np.diff(self.raw.indptr)

    def summary(self) -> Dict[str, Any]:
        cnt = self.neighbour_counts()
        n = self.n
        sym = self.raw.maximum(self.raw.T)
        ncomp, labels = connected_components(sym, directed=False)
        return {
            "n": n,
            "links": int(self.raw.nnz),
            "pct": 100.0 * self.raw.nnz / (n * n) if n else 0.0,
            "mean": float(cnt.mean()) if n else 0.0,
            "min": int(cnt.min()) if n else 0,
            "max": int(cnt.max()) if n else 0,
            "median": float(np.median(cnt)) if n else 0.0,
            "islands": list(self.islands),
            "components": int(ncomp),
            "component_sizes": sorted(np.bincount(labels).tolist(), reverse=True),
            "symmetric": bool((abs(self.raw - self.raw.T) > 1e-12).nnz == 0),
        }

    def describe(self) -> str:
        s = self.summary()
        parts = [f"공간가중행렬: {self.spec.describe()}, {NORM_LABELS[self.spec.norm]}", f"지역 {s['n']}곳, 평균 이웃 {s['mean']:.2f}곳(최소 {s['min']}, 최대 {s['max']})"]
        if self.threshold is not None and self.spec.kind in ("distance", "idw"):
            if self.threshold > 0:
                parts.append(f"거리 임계값 {self.threshold:,.2f}{self.unit}")
            else:
                parts.append("거리 임계값 없음(전체 지역 쌍)")
        if self.islands:
            names = ", ".join(self.islands[:6]) + (" …" if len(self.islands) > 6 else "")
            parts.append(f"이웃 없는 지역 {len(self.islands)}곳({names}): {ISLAND_LABELS[self.island_action]}")
        if s["components"] > 1:
            parts.append(f"서로 연결되지 않은 묶음 {s['components']}개")
        return "; ".join(parts) + "."

    def stata_hint(self) -> str:
        norm = {"row": "row", "spectral": "spectral", "minmax": "minmax", "none": "none"}[self.spec.norm]
        if self.spec.kind in ("queen", "rook") and self.spec.order == 1:
            return f"spmatrix create contiguity W{', rook' if self.spec.kind == 'rook' else ','} normalize({norm})"
        if self.spec.kind == "idw" and self.spec.threshold <= 0:
            return f"spmatrix create idistance W, normalize({norm})"
        return "spmatrix fromdata W = w*, normalize(none)  // CSV로 내보낸 행렬 사용"

    # ---------- export ----------
    def to_libpysal(self, normalised: bool = True):
        from libpysal.weights import W as LW
        M = (self.W if normalised else self.raw).tocsr()
        neighbors, weights = {}, {}
        for i, key in enumerate(self.ids):
            cols = M.indices[M.indptr[i]:M.indptr[i + 1]]
            vals = M.data[M.indptr[i]:M.indptr[i + 1]]
            neighbors[key] = [self.ids[j] for j in cols]
            weights[key] = [float(v) for v in vals]
        return LW(neighbors, weights, id_order=self.ids, silence_warnings=True)

    def export(self, path: str) -> str:
        """Write W: .gal (neighbour lists), .gwt (normalised weights) or .csv (Stata 'spmatrix fromdata' layout)."""
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ext = os.path.splitext(path)[1].lower()
        if ext in (".gal", ".gwt"):
            import libpysal
            w = self.to_libpysal(normalised=(ext == ".gwt"))
            f = libpysal.io.open(path, "w")
            try:
                f.write(w)
            finally:
                f.close()
        elif ext == ".csv":
            dense = self.dense()
            numeric = all(str(k).lstrip("-").isdigit() for k in self.ids)
            df = pd.DataFrame(dense, columns=[f"w{i + 1}" for i in range(self.n)])
            df.insert(0, "_ID", [int(k) for k in self.ids] if numeric else np.arange(1, self.n + 1))
            if not numeric:
                df.insert(1, "key", self.ids)
            df.to_csv(path, index=False, encoding="utf-8-sig")
        else:
            raise ValueError("가중행렬 저장 형식: .gal, .gwt, .csv 중 하나")
        return path


# ---------------------------------------------------------------- builders
def _contiguity(gdf_m, kind: str) -> sp.csr_matrix:
    from libpysal.weights import Queen, Rook
    cls = Queen if kind == "queen" else Rook
    w = cls.from_dataframe(gdf_m.reset_index(drop=True), use_index=False, silence_warnings=True)
    order = list(range(len(gdf_m)))
    M = w.sparse.tocsr()
    if list(w.id_order) != order:
        pos = [w.id_order.index(i) for i in order]
        M = M[pos][:, pos]
    M = (M > 0).astype(float)
    M.setdiag(0)
    M.eliminate_zeros()
    return M.tocsr()


def _higher_order(A: sp.csr_matrix, order: int, lower: bool) -> sp.csr_matrix:
    if order <= 1:
        return A
    n = A.shape[0]
    A = (A > 0).astype(np.int8).tocsr()
    reach = A.copy()
    prev = A.copy()
    for _ in range(2, order + 1):
        prev = reach.copy()
        reach = ((reach + reach @ A) > 0).astype(np.int8).tocsr()
    eye = sp.identity(n, format="csr", dtype=np.int8)
    M = reach if lower else ((reach - prev) > 0).astype(np.int8)
    M = (M - M.multiply(eye)).tocsr()
    M.eliminate_zeros()
    return M.astype(float)


def _read_weights_file(path: str, ids: Sequence[str]) -> sp.csr_matrix:
    from ..spatial_io import normalize_keys
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise ValueError(f"가중행렬 파일을 찾을 수 없음: {path}")
    pos = {k: i for i, k in enumerate(ids)}
    n = len(ids)
    ext = os.path.splitext(path)[1].lower()
    rows, cols, vals = [], [], []
    if ext in (".gal", ".gwt"):
        import libpysal
        fh = libpysal.io.open(path)
        try:
            w = fh.read()
        finally:
            fh.close()
        missing = 0
        for i_key, neigh in w.neighbors.items():
            ik = normalize_keys([i_key]).iloc[0]
            if ik not in pos:
                missing += 1
                continue
            for j_key, val in zip(neigh, w.weights[i_key]):
                jk = normalize_keys([j_key]).iloc[0]
                if jk in pos:
                    rows.append(pos[ik]); cols.append(pos[jk]); vals.append(float(val))
        found = {normalize_keys([k]).iloc[0] for k in w.neighbors}
    elif ext == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
        key_col = "key" if "key" in df.columns else df.columns[0]
        wcols = [c for c in df.columns if c not in (key_col, "_ID", "key")]
        if len(wcols) != len(df):
            raise ValueError("CSV 가중행렬은 정방행렬(첫 열 지역키 + 지역 수만큼의 가중치 열) 형식이어야 함")
        keys = normalize_keys(df[key_col]).tolist()
        mat = df[wcols].to_numpy(dtype=float)
        for a, ka in enumerate(keys):
            if ka not in pos:
                continue
            for b, kb in enumerate(keys):
                if kb in pos and mat[a, b] != 0 and a != b:
                    rows.append(pos[ka]); cols.append(pos[kb]); vals.append(mat[a, b])
        found = set(keys)
    else:
        raise ValueError("가중행렬 파일 형식: .gal, .gwt, .csv")
    absent = [k for k in ids if k not in found]
    if absent:
        raise ValueError(f"가중행렬 파일에 없는 지역 {len(absent)}곳: {', '.join(absent[:8])}")
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def build_weights(spec: WeightsSpec, geoms) -> SpatialWeights:
    """Build W over `geoms` (GeoSeries indexed by unit key, any CRS)."""
    import geopandas as gpd
    from ..spatial_io import to_metric

    ids = [str(k) for k in geoms.index]
    n = len(ids)
    if n < 3:
        raise ValueError(f"공간 단위가 {n}곳: 공간가중행렬에 최소 3곳 필요")
    gdf = gpd.GeoDataFrame({"_key": ids}, geometry=list(geoms.values), crs=geoms.crs)
    gdf_m, unit = to_metric(gdf)
    scale = 1000.0 if unit == "km" else 1.0
    xy = _centroids(gdf_m.geometry) / scale
    polygons = bool(gdf_m.geom_type.isin(["Polygon", "MultiPolygon"]).all())
    threshold = None
    if spec.kind == "auto":
        spec.kind = "queen" if polygons else "knn"

    if spec.kind in ("queen", "rook"):
        if not polygons:
            raise ValueError("인접성(퀸·룩) 기준은 면(폴리곤) 경계 파일이 필요함 — 점 좌표 자료는 k-최근접·거리 기준을 선택")
        raw = _higher_order(_contiguity(gdf_m, spec.kind), spec.order, spec.lower)
    elif spec.kind == "knn":
        k = min(spec.k, n - 1)
        _, idx = cKDTree(xy).query(xy, k=k + 1)
        rows, cols = [], []
        for i in range(n):
            nb = [j for j in idx[i] if j != i][:k]
            rows.extend([i] * len(nb)); cols.extend(nb)
        raw = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    elif spec.kind in ("distance", "idw"):
        tree = cKDTree(xy)
        if spec.kind == "distance" or spec.threshold > 0:
            nn_d, _ = tree.query(xy, k=2)
            auto = float(nn_d[:, 1].max()) * (1 + 1e-9)
            threshold = spec.threshold if spec.threshold > 0 else auto
            D = tree.sparse_distance_matrix(tree, threshold, output_type="coo_matrix").tocsr()
            D.setdiag(0)
            D.eliminate_zeros()
            if spec.kind == "distance":
                raw = (D > 0).astype(float)
            else:
                D.data = np.maximum(D.data, 1e-9) ** (-spec.power)
                raw = D
        else:
            if n > 4000:
                raise ValueError("역거리 가중(임계값 없음)은 4,000곳 이하에서 사용 — 거리 임계값을 지정")
            threshold = 0.0
            d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
            with np.errstate(divide="ignore"):
                M = np.where(d > 0, np.maximum(d, 1e-9) ** (-spec.power), 0.0)
            np.fill_diagonal(M, 0.0)
            raw = sp.csr_matrix(M)
    elif spec.kind == "file":
        raw = _read_weights_file(spec.file, ids)
    else:  # pragma: no cover
        raise ValueError(spec.kind)

    raw = raw.tocsr()
    raw.setdiag(0)
    raw.eliminate_zeros()
    if spec.symmetric:
        raw = raw.maximum(raw.T).tocsr()

    rs = np.asarray(abs(raw).sum(axis=1)).ravel()
    island_idx = np.where(rs == 0)[0]
    islands = [ids[i] for i in island_idx]
    dropped: List[str] = []
    if len(island_idx):
        if spec.islands == "nearest":
            _, idx = cKDTree(xy).query(xy[island_idx], k=2)
            raw = raw.tolil()
            for i, nb in zip(island_idx, idx[:, 1]):
                val = 1.0
                if spec.kind == "idw":
                    val = float(np.maximum(np.linalg.norm(xy[i] - xy[nb]), 1e-9) ** (-spec.power))
                raw[i, nb] = val
                if raw[nb, i] == 0:
                    raw[nb, i] = val
            raw = raw.tocsr()
        elif spec.islands == "drop":
            keep = np.setdiff1d(np.arange(n), island_idx)
            raw = raw[keep][:, keep]
            dropped = islands
            ids = [ids[i] for i in keep]
            xy = xy[keep]
    return SpatialWeights(ids, raw, spec, unit, xy, islands, spec.islands, threshold, dropped)

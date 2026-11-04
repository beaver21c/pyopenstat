"""
Boundary files for spatial analysis (polygons or points) and key matching with a data table.

- Reads ESRI Shapefile (.shp with .shx/.dbf, optional .prj/.cpg), zipped shapefiles, GeoJSON and GeoPackage.
- Korean .dbf files without a .cpg are decoded as CP949 when they are not valid UTF-8.
- A row-order key `_GID` (1..n) is always available, so a boundary file can be opened as a dataset directly.
- Keys are compared as normalised strings ("11010", 11010 and 11010.0 match; leading zeros are kept).
- Distances are computed in a metric CRS: geographic coordinates (lon/lat) are projected to the local UTM zone.
"""

from __future__ import annotations

import functools
import os
import re
import zipfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SUPPORTED_BOUNDARY_EXTENSIONS = (".shp", ".zip", ".geojson", ".json", ".gpkg")
SHAPEFILE_SIDECARS = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx", ".qmd")
GID = "_GID"

_CPG_ALIASES = {
    "949": "cp949", "ansi 949": "cp949", "cp949": "cp949", "euc-kr": "cp949", "euckr": "cp949",
    "ks_c_5601-1987": "cp949", "uhc": "cp949", "utf-8": "utf-8", "utf8": "utf-8", "65001": "utf-8",
    "1252": "cp1252", "ansi 1252": "cp1252", "iso-8859-1": "latin-1", "latin1": "latin-1",
}


def require_geopandas():
    try:
        import geopandas as gpd  # noqa: F401
    except ImportError as e:  # pragma: no cover - install profile
        raise ValueError("지도·공간분석에 geopandas 패키지 필요: 'pip install -r requirements.txt' 실행 후 다시 시도") from e
    return gpd


def _zip_member(path: str, ext: str = ".shp") -> Optional[str]:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(ext) and not n.startswith("__MACOSX")]
    return sorted(names, key=len)[0] if names else None


def _cpg_encoding(text: str) -> Optional[str]:
    key = text.strip().lower()
    return _CPG_ALIASES.get(key) or (key if key else None)


def detect_dbf_encoding(path: str) -> str:
    """Encoding of the attribute table: .cpg when present, otherwise UTF-8 if the .dbf decodes, else CP949."""
    base, ext = os.path.splitext(path)
    if ext.lower() == ".zip":
        member = _zip_member(path)
        if not member:
            return "utf-8"
        mbase = os.path.splitext(member)[0]
        with zipfile.ZipFile(path) as z:
            names = {n.lower(): n for n in z.namelist()}
            cpg = names.get((mbase + ".cpg").lower())
            if cpg:
                enc = _cpg_encoding(z.read(cpg).decode("ascii", "ignore"))
                if enc:
                    return enc
            dbf = names.get((mbase + ".dbf").lower())
            raw = z.read(dbf) if dbf else b""
    else:
        for cand in (base + ".cpg", base + ".CPG"):
            if os.path.exists(cand):
                with open(cand, "rb") as f:
                    enc = _cpg_encoding(f.read().decode("ascii", "ignore"))
                if enc:
                    return enc
        dbf = next((c for c in (base + ".dbf", base + ".DBF") if os.path.exists(c)), None)
        raw = open(dbf, "rb").read() if dbf else b""
    if not raw:
        return "utf-8"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp949"


def _source_for(path: str) -> str:
    if path.lower().endswith(".zip"):
        member = _zip_member(path)
        if not member:
            raise ValueError(f"압축 파일 안에 .shp 파일 없음: {os.path.basename(path)}")
        return "/vsizip/" + os.path.abspath(path).replace("\\", "/") + "/" + member
    return path


@functools.lru_cache(maxsize=12)
def _read_cached(path: str, mtime: float):
    gpd = require_geopandas()
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_BOUNDARY_EXTENSIONS:
        raise ValueError(f"지원하지 않는 경계 파일 형식: {ext} (지원: {', '.join(SUPPORTED_BOUNDARY_EXTENSIONS)})")
    if ext == ".shp":
        base = os.path.splitext(path)[0]
        missing = [s for s in (".shx", ".dbf") if not (os.path.exists(base + s) or os.path.exists(base + s.upper()))]
        if missing:
            raise ValueError(f"Shapefile 구성 파일 누락: {', '.join(missing)} (.shp·.shx·.dbf를 같은 폴더에 함께 두어야 함)")
    kwargs = {}
    if ext in (".shp", ".zip"):
        kwargs["encoding"] = detect_dbf_encoding(path)
    gdf = gpd.read_file(_source_for(path), **kwargs)
    gdf = gdf[~gdf.geometry.isna()].reset_index(drop=True)
    if gdf.empty:
        raise ValueError("경계 파일에 도형이 없음")
    if gdf.crs is None:
        minx, miny, maxx, maxy = gdf.total_bounds
        if -180.5 <= minx <= 180.5 and -180.5 <= maxx <= 180.5 and -90.5 <= miny <= 90.5 and -90.5 <= maxy <= 90.5:
            gdf = gdf.set_crs(4326)
    if GID not in gdf.columns:
        gdf.insert(0, GID, np.arange(1, len(gdf) + 1))
    return gdf


def read_boundary(path: str):
    """Read a boundary file as a GeoDataFrame (cached by path and modification time)."""
    if not path:
        raise ValueError("경계 파일 경로가 비어 있음")
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise ValueError(f"경계 파일을 찾을 수 없음: {path}")
    return _read_cached(path, os.path.getmtime(path)).copy()


def normalize_keys(values) -> pd.Series:
    """Compare keys as strings: 11010, 11010.0 and ' 11010 ' are the same key; '01' keeps its leading zero."""
    s = pd.Series(values)
    out = s.astype(object).where(s.notna(), None)

    def _one(v):
        if v is None:
            return None
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return None
            if float(v).is_integer():
                return str(int(v))
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        t = str(v).strip()
        if re.fullmatch(r"-?\d+\.0+", t):
            t = t.split(".")[0]
        return t or None

    return out.map(_one)


def describe_boundary(gdf) -> Dict:
    """Summary used by the upload dialog: geometry type, CRS, attribute columns and key candidates."""
    cols = []
    n = len(gdf)
    for c in gdf.columns:
        if c == "geometry":
            continue
        s = gdf[c]
        keys = normalize_keys(s)
        cols.append({
            "name": c,
            "unique": int(keys.nunique(dropna=True)),
            "isKey": bool(keys.notna().all() and keys.nunique() == n),
            "sample": [str(v) for v in s.head(3).tolist()],
        })
    geom_types = sorted(set(gdf.geom_type.astype(str)))
    crs = gdf.crs
    return {
        "n": n,
        "geomType": ", ".join(geom_types),
        "crs": (crs.to_string() if crs is not None else "정보 없음(.prj 누락)"),
        "crsName": (crs.name if crs is not None else ""),
        "geographic": bool(crs.is_geographic) if crs is not None else None,
        "bounds": [float(v) for v in gdf.total_bounds],
        "columns": cols,
    }


def to_metric(gdf):
    """Project to a metric CRS for distances: lon/lat -> local UTM; projected CRS kept; unknown CRS kept as planar."""
    if gdf.crs is None:
        return gdf, "좌표 단위"
    if gdf.crs.is_geographic:
        try:
            utm = gdf.estimate_utm_crs()
            return gdf.to_crs(utm), "km"
        except Exception:
            return gdf.to_crs(3857), "km"
    unit = ""
    try:
        unit = (gdf.crs.axis_info[0].unit_name or "").lower()
    except Exception:
        pass
    return gdf, ("km" if unit in ("metre", "meter", "m", "") else unit)


def match_keys(data_keys: Sequence, boundary_keys: Sequence) -> Dict:
    dk = normalize_keys(data_keys)
    bk = normalize_keys(boundary_keys)
    d_unique = pd.unique(dk.dropna())
    b_set = set(bk.dropna())
    d_set = set(d_unique)
    matched = [k for k in d_unique if k in b_set]
    dup = bk[bk.duplicated(keep=False) & bk.notna()].unique().tolist()
    return {
        "dataUnits": len(d_unique),
        "boundaryUnits": len(b_set),
        "matched": len(matched),
        "unmatchedData": [k for k in d_unique if k not in b_set][:30],
        "unmatchedDataCount": int(sum(1 for k in d_unique if k not in b_set)),
        "unmatchedBoundary": sorted([k for k in b_set if k not in d_set])[:30],
        "unmatchedBoundaryCount": int(sum(1 for k in b_set if k not in d_set)),
        "duplicateBoundaryKeys": dup[:30],
    }


def unit_geometries(boundary_path: str, boundary_key: str, unit_keys: Sequence[str]):
    """GeoSeries aligned to `unit_keys` (normalised). Multi-part features sharing a key are dissolved."""
    gdf = read_boundary(boundary_path)
    if boundary_key not in gdf.columns:
        raise ValueError(f"경계 파일에 '{boundary_key}' 속성 없음 (사용 가능: {', '.join(c for c in gdf.columns if c != 'geometry')})")
    gdf = gdf[[boundary_key, "geometry"]].copy()
    gdf["_k"] = normalize_keys(gdf[boundary_key]).values
    gdf = gdf[gdf["_k"].notna()]
    if gdf["_k"].duplicated().any():
        gdf = gdf.dissolve(by="_k", as_index=False)
    lookup = gdf.set_index("_k").geometry
    keys = list(unit_keys)
    missing = [k for k in keys if k not in lookup.index]
    if missing:
        raise ValueError(f"경계 파일에서 찾지 못한 지역 {len(missing)}곳: {', '.join(missing[:8])}{' …' if len(missing) > 8 else ''}")
    gpd = require_geopandas()
    return gpd.GeoSeries([lookup[k] for k in keys], index=keys, crs=gdf.crs)


def points_geometries(x: Sequence[float], y: Sequence[float], keys: Sequence[str]):
    """Point geometries from coordinate columns. Values within lon/lat ranges are treated as WGS84."""
    gpd = require_geopandas()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    crs = 4326 if (np.all(np.abs(x) <= 180) and np.all(np.abs(y) <= 90)) else None
    return gpd.GeoSeries(gpd.points_from_xy(x, y), index=list(keys), crs=crs)


def save_uploaded_boundary(files: List[Tuple[str, bytes]], dest_dir: str) -> str:
    """Write uploaded shapefile parts (or one zip/geojson/gpkg) into dest_dir; return the path to read."""
    os.makedirs(dest_dir, exist_ok=True)
    written = []
    for name, data in files:
        base = os.path.basename(name.replace("\\", "/"))
        if not base:
            continue
        with open(os.path.join(dest_dir, base), "wb") as f:
            f.write(data)
        written.append(base)
    if not written:
        raise ValueError("업로드된 파일 없음")
    lower = {w.lower(): w for w in written}
    shp = [w for w in written if w.lower().endswith(".shp")]
    if shp:
        main = shp[0]
        stem = os.path.splitext(main)[0].lower()
        missing = [s for s in (".shx", ".dbf") if stem + s not in lower]
        if missing:
            raise ValueError(f"Shapefile 구성 파일 누락: {', '.join(missing)} — .shp와 같은 이름의 .shx·.dbf(가능하면 .prj·.cpg도)를 함께 선택")
        return os.path.join(dest_dir, main)
    for ext in (".zip", ".gpkg", ".geojson", ".json"):
        cand = [w for w in written if w.lower().endswith(ext)]
        if cand:
            return os.path.join(dest_dir, cand[0])
    raise ValueError("경계 파일(.shp+.shx+.dbf, .zip, .geojson, .gpkg)을 찾지 못함")

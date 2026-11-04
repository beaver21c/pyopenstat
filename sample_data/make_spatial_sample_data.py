"""
Synthetic boundary file and regional datasets for the spatial-analysis modules.
All shapes, names and values are synthetic (Voronoi cells, seed fixed); they do not represent real regions.

- geo/Synthetic_Regions.shp (+ .shx .dbf .prj .cpg): 72 mainland regions + 1 island, EPSG:5179 (Korea 2000 / Unified CS)
  attributes SIG_CD (5-digit code), SIG_KOR_NM (name), AREA_KM2
- Spatial_Panel_Welfare.csv: 73 regions x 8 years (2016-2023), generated from a two-way fixed-effects spatial Durbin model
  y = 0.40 W y + X b + W X t + region effect + year effect + e  (W = queen contiguity, island linked to its nearest region, row-standardised)
- Spatial_CrossSection_Welfare.csv: the same regions in one year, generated from a spatial lag model (rho = 0.45)

Run: python sample_data/make_spatial_sample_data.py
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GEO_DIR = os.path.join(HERE, "geo")
SEED = 2026

PREFIX = ["가람", "누리", "다온", "마루", "바다", "새솔", "아라", "하늬", "보람", "푸른", "솔빛", "한울", "온새", "미리",
          "도담", "라온", "여울", "윤슬", "은빛", "청솔", "해솔", "별빛", "단비", "모래", "구름", "들꽃", "샘골", "갈대",
          "물빛", "숲속", "너울", "꽃뫼", "달빛", "아침", "노을", "새봄", "바람", "이슬", "가온", "나래"]


def make_regions(rng):
    import geopandas as gpd
    from shapely.geometry import MultiPoint, Point, Polygon

    # irregular outline (about 110 km x 95 km)
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    rad = 48000 * (1 + 0.16 * np.sin(3 * ang + 0.7) + 0.08 * np.cos(5 * ang) + rng.normal(0, 0.025, ang.size))
    cx, cy = 1000000.0, 1850000.0
    outline = Polygon(np.column_stack([cx + 1.12 * rad * np.cos(ang), cy + rad * np.sin(ang)])).buffer(0)

    pts = []
    minx, miny, maxx, maxy = outline.bounds
    while len(pts) < 72:
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if outline.contains(p) and all(p.distance(q) > 6500 for q in pts):
            pts.append(p)
    from shapely import voronoi_polygons
    cells = voronoi_polygons(MultiPoint(pts), extend_to=outline.buffer(20000))
    polys = []
    for p in pts:
        cell = next(c for c in cells.geoms if c.contains(p))
        polys.append(cell.intersection(outline))
    island = Point(cx + 1.12 * 48000 * 1.25, cy - 30000).buffer(4200, quad_segs=6)
    polys.append(island)

    order = np.lexsort(([-g.centroid.y for g in polys], [round(g.centroid.x / 15000) for g in polys]))
    polys = [polys[i] for i in order]
    names = []
    for i, _ in enumerate(polys):
        suffix = "군" if i % 4 == 3 else "시"
        names.append(PREFIX[i % len(PREFIX)] + ("" if i < len(PREFIX) else "남") + suffix)
    codes = [f"{90010 + 10 * i}" for i in range(len(polys))]
    gdf = gpd.GeoDataFrame({"SIG_CD": codes, "SIG_KOR_NM": names}, geometry=polys, crs=5179)
    gdf.loc[gdf.geometry.geom_equals(island), "SIG_KOR_NM"] = "섬마을군"
    gdf["AREA_KM2"] = (gdf.area / 1e6).round(1)
    return gdf


def queen_w(gdf):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "reconstructed"))
    from pyjamovi.analyses.spatial_weights import WeightsSpec, build_weights
    geoms = gdf.set_index("SIG_CD").geometry
    return build_weights(WeightsSpec(kind="queen", norm="row", islands="nearest"), geoms)


def make_panel(gdf, W, rng):
    N = len(gdf)
    years = list(range(2016, 2024))
    T = len(years)
    Wd = W.dense()
    A_x = np.linalg.inv(np.eye(N) - 0.55 * Wd)
    base_old = 14 + 6 * (A_x @ rng.normal(0, 1, N)) / 2
    base_single = 28 + 5 * (A_x @ rng.normal(0, 1, N)) / 2
    base_fin = 32 + 12 * (A_x @ rng.normal(0, 1, N)) / 2
    base_fac = 3 + 1.2 * np.abs(A_x @ rng.normal(0, 1, N)) / 2
    mu = 2.0 * (A_x @ rng.normal(0, 1, N)) / 2 + 0.25 * (base_old - base_old.mean())
    tau = np.cumsum(rng.normal(0.35, 0.15, T))
    beta = np.array([0.80, 0.30, -0.15, 0.50])
    theta = np.array([0.30, 0.0, 0.0, -0.20])
    rho = 0.40
    S = np.linalg.inv(np.eye(N) - rho * Wd)
    rows = []
    for t, yr in enumerate(years):
        X = np.column_stack([
            base_old + 0.45 * t + rng.normal(0, 0.6, N),
            base_single + 0.35 * t + rng.normal(0, 0.8, N),
            base_fin - 0.2 * t + rng.normal(0, 1.5, N),
            base_fac + 0.04 * t + rng.normal(0, 0.25, N),
        ])
        eps = rng.normal(0, 1.1, N)
        y = S @ (X @ beta + Wd @ X @ theta + mu + tau[t] + eps)
        for i in range(N):
            rows.append({"지역코드": gdf.SIG_CD.iloc[i], "지역명": gdf.SIG_KOR_NM.iloc[i], "연도": yr,
                         "돌봄이용률": round(y[i], 3), "고령인구비율": round(X[i, 0], 2), "1인가구비율": round(X[i, 1], 2),
                         "재정자립도": round(X[i, 2], 2), "복지시설수": round(X[i, 3], 3)})
    return pd.DataFrame(rows)


def make_cross_section(gdf, W, rng):
    N = len(gdf)
    Wd = W.dense()
    A_x = np.linalg.inv(np.eye(N) - 0.5 * Wd)
    c = gdf.geometry.representative_point()
    lon_like = (c.x - c.x.min()) / (c.x.max() - c.x.min())
    old = 17 + 5 * (A_x @ rng.normal(0, 1, N)) / 2 + 3 * lon_like.values
    poverty = 3.5 + 1.2 * (A_x @ rng.normal(0, 1, N)) / 2
    single = 30 + 6 * rng.normal(0, 1, N)
    rho = 0.45
    y = np.linalg.solve(np.eye(N) - rho * Wd, 20 + 1.6 * old + 2.4 * poverty + 0.25 * single + rng.normal(0, 3.0, N))
    return pd.DataFrame({"지역코드": gdf.SIG_CD.values, "지역명": gdf.SIG_KOR_NM.values,
                         "복지지출": np.round(y, 2), "고령인구비율": np.round(old, 2), "빈곤율": np.round(poverty, 2),
                         "1인가구비율": np.round(single, 2), "면적": gdf.AREA_KM2.values})


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    os.makedirs(GEO_DIR, exist_ok=True)
    gdf = make_regions(rng)
    shp = os.path.join(GEO_DIR, "Synthetic_Regions.shp")
    gdf.to_file(shp, encoding="utf-8")
    with open(os.path.join(GEO_DIR, "Synthetic_Regions.cpg"), "w") as f:
        f.write("UTF-8")
    W = queen_w(gdf)
    panel = make_panel(gdf, W, rng)
    panel.to_csv(os.path.join(HERE, "Spatial_Panel_Welfare.csv"), index=False, encoding="utf-8-sig")
    cs = make_cross_section(gdf, W, rng)
    cs.to_csv(os.path.join(HERE, "Spatial_CrossSection_Welfare.csv"), index=False, encoding="utf-8-sig")
    print(f"{shp}: {len(gdf)} regions | panel {len(panel)} rows | cross-section {len(cs)} rows | islands {W.islands}")

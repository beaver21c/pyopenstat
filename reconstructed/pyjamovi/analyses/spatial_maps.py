"""
Thematic maps for regional analysis: choropleth (class breaks), categorical maps (LISA, Gi*),
neighbour-link maps of W and local-coefficient maps (GWR). Polygons are drawn from the boundary file,
point data as markers. A north arrow and a km scale bar are added when the CRS is known.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

from .plots import fig_to_base64
from .plot_style import get_active_style

SCHEME_LABELS = {
    "quantiles": "분위수(지역 수 균등)",
    "equal_interval": "등간격",
    "natural_breaks": "자연 구분(Fisher–Jenks)",
    "std_mean": "평균·표준편차 구간",
}
CMAP_OPTIONS = ["YlOrRd", "Blues", "Greens", "Purples", "Oranges", "viridis", "RdBu_r", "PuOr_r", "kihasa"]
KIHASA_SEQUENTIAL = ["#EBF3FA", "#93BCE8", "#4A84D1", "#00457C", "#002046"]
_SCHEME_ALIASES = {
    "quantile": "quantiles", "q": "quantiles",
    "equal": "equal_interval", "equalinterval": "equal_interval", "equal_intervals": "equal_interval",
    "fisher_jenks": "natural_breaks", "fisherjenks": "natural_breaks", "jenks": "natural_breaks",
    "natural": "natural_breaks", "naturalbreaks": "natural_breaks",
    "stdmean": "std_mean", "std": "std_mean",
}


def resolve_scheme(name) -> str:
    """Class-break scheme name (aliases accepted); unknown names raise instead of falling back."""
    s = str(name or "quantiles").strip().lower()
    s = _SCHEME_ALIASES.get(s, s)
    if s not in SCHEME_LABELS:
        raise ValueError(f"알 수 없는 계급 구분 방식: {name} (가능: {', '.join(SCHEME_LABELS)})")
    return s


def resolve_cmap(name) -> str:
    """Colormap name: matplotlib names (case-insensitive) or 'kihasa'; unknown names raise."""
    s = str(name or "YlOrRd").strip()
    if s.lower() in ("kihasa", "kihasa_blue"):
        return "kihasa"
    import matplotlib
    if s in matplotlib.colormaps:
        return s
    lower = {k.lower(): k for k in matplotlib.colormaps}
    if s.lower() in lower:
        return lower[s.lower()]
    raise ValueError(f"알 수 없는 색상표: {name} (예: {', '.join(CMAP_OPTIONS)})")


def get_cmap(name, n: int):
    s = resolve_cmap(name)
    if s == "kihasa":
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list("kihasa", KIHASA_SEQUENTIAL, N=n)
    return plt.get_cmap(s, n)

LISA_COLORS = {
    "High-High": "#D7191C", "Low-Low": "#2C7BB6", "Low-High": "#ABD9E9", "High-Low": "#FDAE61",
    "유의하지 않음": "#E5E7EB", "이웃 없음": "#FFFFFF",
}
LISA_LABELS = {"High-High": "고-고(High-High)", "Low-Low": "저-저(Low-Low)", "Low-High": "저-고(Low-High)",
               "High-Low": "고-저(High-Low)", "유의하지 않음": "유의하지 않음", "이웃 없음": "이웃 없음"}
GI_COLORS = {
    "열점 99%": "#B2182B", "열점 95%": "#EF8A62", "열점 90%": "#FDDBC7", "유의하지 않음": "#E5E7EB",
    "냉점 90%": "#D1E5F0", "냉점 95%": "#67A9CF", "냉점 99%": "#2166AC",
}


def _metric(geoms):
    from ..spatial_io import to_metric
    import geopandas as gpd
    gdf = gpd.GeoDataFrame({"_k": list(geoms.index)}, geometry=list(geoms.values), crs=geoms.crs)
    gdf_m, unit = to_metric(gdf)
    return gdf_m, unit


def _is_points(gdf) -> bool:
    return bool(gdf.geom_type.isin(["Point", "MultiPoint"]).all())


def _colors():
    st = get_active_style()
    dark = st.get("theme") == "dark"
    return {
        "fg": "#F1F5F9" if dark else "#1E293B",
        "edge": "#CBD5E1" if dark else "#FFFFFF",
        "outline": "#94A3B8" if dark else "#64748B",
        "halo": "#0F172A" if dark else "#FFFFFF",
        "na": "#F3F4F6" if not dark else "#334155",
    }


def _nice_length(width: float) -> float:
    target = width / 5.0
    if target <= 0:
        return 1.0
    p = 10 ** np.floor(np.log10(target))
    for m in (1, 2, 5, 10):
        if m * p >= target * 0.8:
            return m * p
    return 10 * p


def _decorate(ax, gdf, unit: str, c: Dict[str, str]):
    minx, miny, maxx, maxy = gdf.total_bounds
    w, h = maxx - minx, maxy - miny
    pad = 0.04 * max(w, h, 1e-9)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad * 2.2, maxy + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if gdf.crs is None:
        return
    # north arrow
    ax.annotate("N", xy=(0.965, 0.97), xytext=(0.965, 0.885), xycoords="axes fraction", ha="center", va="center",
                fontsize=10, fontweight="bold", color=c["fg"],
                arrowprops=dict(arrowstyle="-|>", color=c["fg"], lw=1.4, mutation_scale=12))
    # scale bar (units of the metric CRS are metres)
    length_km = _nice_length(w / 1000.0)
    length = length_km * 1000.0
    x0, y0 = minx, miny - pad * 1.6
    ax.plot([x0, x0 + length], [y0, y0], color=c["fg"], lw=2.2, solid_capstyle="butt")
    for xx in (x0, x0 + length / 2, x0 + length):
        ax.plot([xx, xx], [y0, y0 + pad * 0.35], color=c["fg"], lw=1.0)
    lbl = f"{length_km:g} km" if length_km >= 1 else f"{length_km * 1000:g} m"
    ax.text(x0 + length + pad * 0.3, y0, lbl, va="center", ha="left", fontsize=8, color=c["fg"])


def _labels(ax, gdf, labels: Optional[Sequence[str]], label_max: int, c: Dict[str, str]):
    if labels is None or len(gdf) > label_max:
        return
    pts = gdf.geometry.representative_point() if not _is_points(gdf) else gdf.geometry
    fs = 7.5 if len(gdf) > 40 else 8.5
    for p, t in zip(pts, labels):
        if t is None or str(t) == "nan":
            continue
        txt = ax.text(p.x, p.y, str(t), ha="center", va="center", fontsize=fs, color=c["fg"], zorder=6)
        txt.set_path_effects([path_effects.withStroke(linewidth=2.2, foreground=c["halo"])])


def classify(values: np.ndarray, scheme: str = "quantiles", k: int = 5) -> np.ndarray:
    """Upper bounds of the classes (mapclassify)."""
    import mapclassify as mc
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    k = int(max(2, min(k, 9, max(2, len(np.unique(v))))))
    scheme = resolve_scheme(scheme)
    if len(np.unique(v)) <= k:
        return np.unique(v)
    if scheme == "equal_interval":
        cl = mc.EqualInterval(v, k=k)
    elif scheme == "natural_breaks":
        cl = mc.FisherJenks(v, k=k)
    elif scheme == "std_mean":
        cl = mc.StdMean(v)
    else:
        cl = mc.Quantiles(v, k=k)
    return np.unique(np.asarray(cl.bins, dtype=float))


def _fmt(v: float) -> str:
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def class_table(values: np.ndarray, bins: np.ndarray) -> List[Dict]:
    v = np.asarray(values, dtype=float)
    rows = []
    lo = np.nanmin(v)
    for i, b in enumerate(bins):
        mask = (v <= b) & ((v > bins[i - 1]) if i > 0 else np.isfinite(v))
        rows.append({"cls": i + 1, "lo": float(lo if i == 0 else bins[i - 1]), "hi": float(b), "n": int(mask.sum())})
    return rows


def plot_choropleth(geoms, values, title: str, scheme: str = "quantiles", k: int = 5, cmap: str = "YlOrRd",
                    legend_title: str = "", labels: Optional[Sequence[str]] = None, label_max: int = 400,
                    figsize=(7.4, 6.4)) -> str:
    gdf, unit = _metric(geoms)
    c = _colors()
    v = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    finite = np.isfinite(v)
    bins = classify(v[finite], scheme, k) if finite.any() else np.array([0.0])
    nb = len(bins)
    cm = get_cmap(cmap, max(nb, 2))
    colors = [cm(i / max(nb - 1, 1)) for i in range(nb)]
    cls = np.full(len(v), -1)
    for i in range(len(v)):
        if finite[i]:
            cls[i] = int(np.searchsorted(bins, v[i], side="left"))
            cls[i] = min(cls[i], nb - 1)
    fc = [colors[j] if j >= 0 else c["na"] for j in cls]
    if _is_points(gdf):
        gdf.plot(ax=ax, color=fc, markersize=70, edgecolor=c["fg"], linewidth=0.6, zorder=3)
    else:
        gdf.plot(ax=ax, color=fc, edgecolor=c["edge"], linewidth=0.6, zorder=2)
        gdf.boundary.plot(ax=ax, color=c["outline"], linewidth=0.25, zorder=3)
    lo = np.nanmin(v) if finite.any() else 0.0
    handles = []
    for i, b in enumerate(bins):
        a = lo if i == 0 else bins[i - 1]
        n_i = int((cls == i).sum())
        handles.append(Patch(facecolor=colors[i], edgecolor=c["outline"], label=f"{_fmt(a)} – {_fmt(b)}  ({n_i})"))
    if (~finite).any():
        handles.append(Patch(facecolor=c["na"], edgecolor=c["outline"], label=f"자료 없음  ({int((~finite).sum())})"))
    leg = ax.legend(handles=handles, title=legend_title or None, loc="lower right", fontsize=8, title_fontsize=8.5,
                    frameon=True, framealpha=0.9)
    _labels(ax, gdf, labels, label_max, c)
    _decorate(ax, gdf, unit, c)
    ax.set_title(title, fontweight="bold", color=c["fg"])
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_categories(geoms, cats: Sequence[str], palette: Dict[str, str], title: str, legend_title: str = "",
                    label_map: Optional[Dict[str, str]] = None, labels: Optional[Sequence[str]] = None,
                    label_max: int = 400, figsize=(7.4, 6.4)) -> str:
    gdf, unit = _metric(geoms)
    c = _colors()
    fig, ax = plt.subplots(figsize=figsize)
    cats = list(cats)
    fc = [palette.get(x, c["na"]) for x in cats]
    if _is_points(gdf):
        gdf.plot(ax=ax, color=fc, markersize=70, edgecolor=c["fg"], linewidth=0.6, zorder=3)
    else:
        gdf.plot(ax=ax, color=fc, edgecolor=c["edge"], linewidth=0.6, zorder=2)
        gdf.boundary.plot(ax=ax, color=c["outline"], linewidth=0.25, zorder=3)
    handles = []
    for key, col in palette.items():
        n_i = sum(1 for x in cats if x == key)
        if n_i == 0 and key in ("이웃 없음",):
            continue
        name = (label_map or {}).get(key, key)
        handles.append(Patch(facecolor=col, edgecolor=c["outline"], label=f"{name}  ({n_i})"))
    ax.legend(handles=handles, title=legend_title or None, loc="lower right", fontsize=8, title_fontsize=8.5,
              frameon=True, framealpha=0.9)
    _labels(ax, gdf, labels, label_max, c)
    _decorate(ax, gdf, unit, c)
    ax.set_title(title, fontweight="bold", color=c["fg"])
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_links(geoms, W, title: str, labels: Optional[Sequence[str]] = None, label_max: int = 400,
               figsize=(7.4, 6.4)) -> str:
    """Boundaries with a line between every pair of neighbours in W (line width ~ weight)."""
    gdf, unit = _metric(geoms)
    gdf = gdf.set_index("_k").loc[W.ids].reset_index()
    c = _colors()
    st = get_active_style()
    accent = list(st["colors"])[0]
    fig, ax = plt.subplots(figsize=figsize)
    if _is_points(gdf):
        pts = gdf.geometry
    else:
        gdf.plot(ax=ax, color=c["na"], edgecolor=c["outline"], linewidth=0.5, zorder=1)
        pts = gdf.geometry.representative_point()
    xy = np.column_stack([pts.x.values, pts.y.values])
    M = W.raw.tocoo()
    vals = np.abs(M.data)
    vmax = vals.max() if vals.size else 1.0
    from matplotlib.collections import LineCollection
    segs, widths = [], []
    seen = set()
    for i, j, v in zip(M.row, M.col, vals):
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        segs.append([xy[i], xy[j]])
        widths.append(0.5 + 1.6 * (v / vmax if vmax > 0 else 0))
    ax.add_collection(LineCollection(segs, colors=accent, linewidths=widths, alpha=0.75, zorder=3))
    cnt = W.neighbour_counts()
    island = set(W.islands)
    ax.scatter(xy[:, 0], xy[:, 1], s=12 + 3 * cnt, color=accent, edgecolor=c["halo"], linewidth=0.6, zorder=4)
    if island:
        idx = [k for k, key in enumerate(W.ids) if key in island]
        ax.scatter(xy[idx, 0], xy[idx, 1], s=90, facecolor="none", edgecolor="#DC2626", linewidth=1.6, zorder=5,
                   label="원래 이웃이 없던 지역")
        ax.legend(loc="lower right", fontsize=8, frameon=True)
    _labels(ax, gdf, labels, label_max, c)
    _decorate(ax, gdf, unit, c)
    ax.set_title(title, fontweight="bold", color=c["fg"])
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_coefficient_map(geoms, coef, significant: Optional[np.ndarray], title: str, legend_title: str = "",
                         labels: Optional[Sequence[str]] = None, label_max: int = 400, figsize=(7.4, 6.4)) -> str:
    """Local coefficient surface (diverging colours centred at 0); non-significant units hatched."""
    gdf, unit = _metric(geoms)
    c = _colors()
    v = np.asarray(coef, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = "RdBu_r"
    else:
        norm = plt.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-9)
        cmap = "Reds" if vmin >= 0 else "Blues_r"
    gdf = gdf.assign(_v=v)
    if _is_points(gdf):
        gdf.plot(ax=ax, column="_v", cmap=cmap, norm=norm, markersize=70, edgecolor=c["fg"], linewidth=0.5, zorder=3)
    else:
        gdf.plot(ax=ax, column="_v", cmap=cmap, norm=norm, edgecolor=c["edge"], linewidth=0.5, zorder=2)
        if significant is not None and (~significant).any():
            gdf[~significant].plot(ax=ax, facecolor="none", edgecolor=c["outline"], hatch="////", linewidth=0.0, zorder=3)
        gdf.boundary.plot(ax=ax, color=c["outline"], linewidth=0.25, zorder=4)
    sm_ = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm_, ax=ax, shrink=0.62, pad=0.01)
    cb.set_label(legend_title or "국지 계수", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    if significant is not None and (~significant).any() and not _is_points(gdf):
        ax.legend(handles=[Patch(facecolor="none", edgecolor=c["outline"], hatch="////",
                                 label=f"유의하지 않음 ({int((~significant).sum())})")], loc="lower right", fontsize=8)
    _labels(ax, gdf, labels, label_max, c)
    _decorate(ax, gdf, unit, c)
    ax.set_title(title, fontweight="bold", color=c["fg"])
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_moran_scatter(z: np.ndarray, wz: np.ndarray, I: float, title: str, xlabel: str, ylabel: str,
                       cats: Optional[Sequence[str]] = None, figsize=(6.2, 5.4)) -> str:
    c = _colors()
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0, color=c["outline"], lw=0.8, ls="--")
    ax.axhline(0, color=c["outline"], lw=0.8, ls="--")
    if cats is not None:
        col = [LISA_COLORS.get(x, "#9CA3AF") if x != "유의하지 않음" else "#9CA3AF" for x in cats]
    else:
        col = list(get_active_style()["colors"])[0]
    ax.scatter(z, wz, c=col, s=26, alpha=0.85, edgecolor=c["halo"], linewidth=0.5, zorder=3)
    xs = np.linspace(np.nanmin(z), np.nanmax(z), 50)
    b = np.polyfit(z, wz, 1) if len(z) > 2 else (I, 0.0)
    ax.plot(xs, b[0] * xs + b[1], color=c["fg"], lw=1.6, zorder=4, label=f"기울기 = {b[0]:.3f}")
    for (tx, ty, txt, ha, va) in ((0.98, 0.98, "고-고", "right", "top"), (0.02, 0.02, "저-저", "left", "bottom"),
                                  (0.98, 0.02, "고-저", "right", "bottom"), (0.02, 0.98, "저-고", "left", "top")):
        ax.text(tx, ty, txt, transform=ax.transAxes, ha=ha, va=va, fontsize=8.5, color=c["outline"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5)
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_neighbour_hist(counts: np.ndarray, title: str, figsize=(6.0, 3.8)) -> str:
    st = get_active_style()
    fig, ax = plt.subplots(figsize=figsize)
    vals, freq = np.unique(counts, return_counts=True)
    ax.bar(vals, freq, color=list(st["colors"])[0], width=0.7)
    ax.set_xlabel("이웃 수")
    ax.set_ylabel("지역 수")
    ax.set_xticks(vals)
    ax.set_title(title, fontweight="bold")
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64

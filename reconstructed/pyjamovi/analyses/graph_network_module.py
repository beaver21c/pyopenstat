"""
OpenStat Relational Network Module (edge list -> graph analysis)
- Builds a (directed or undirected) weighted graph from an edge list (source, target, weight)
- Network summary: nodes, edges, density, components, transitivity, average path length, modularity
- Node centrality: degree, weighted degree (strength), betweenness, closeness, eigenvector, PageRank
- Louvain community detection (seeded) and a labelled network map with weighted relationship lines

The drawing and centrality helpers are shared with the text-mining module
(keyword co-occurrence networks, cluster relationship maps).
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from .base import BaseAnalysis, ResultsTable
from .plots import fig_to_base64, _pal
from .plot_style import get_active_style


def require_networkx():
    try:
        import networkx as nx
    except ImportError as e:  # pragma: no cover - depends on the install profile
        raise ValueError("네트워크 분석에 networkx 패키지 필요: 'pip install networkx' 실행 후 다시 시도") from e
    return nx


CENTRALITY_LABELS = {
    "degree": "연결 중심성",
    "strength": "가중 연결정도",
    "betweenness": "매개 중심성",
    "closeness": "근접 중심성",
    "eigenvector": "고유벡터 중심성",
    "pagerank": "PageRank",
}


def _set_distance(G, weight: str = "weight"):
    """Path length of an edge = (largest weight) / weight, so the strongest tie has length 1."""
    ws = [float(d.get(weight, 1.0) or 0.0) for _, _, d in G.edges(data=True)]
    wmax = max([w for w in ws if w > 0], default=1.0)
    for _, _, d in G.edges(data=True):
        w = float(d.get(weight, 1.0) or 0.0)
        d["_distance"] = wmax / w if w > 0 else 1e6


def compute_centrality(G, weight: str = "weight") -> pd.DataFrame:
    """Centrality table for every node. Distances for path-based measures use 1 / weight."""
    nx = require_networkx()
    nodes = list(G.nodes())
    if not nodes:
        return pd.DataFrame(columns=["node", *CENTRALITY_LABELS.keys()])
    _set_distance(G, weight)

    degree = nx.degree_centrality(G) if len(nodes) > 1 else {n: 0.0 for n in nodes}
    strength = dict(G.degree(weight=weight))
    betweenness = nx.betweenness_centrality(G, weight="_distance", normalized=True)
    closeness = nx.closeness_centrality(G, distance="_distance")
    try:
        eigen = nx.eigenvector_centrality_numpy(G.to_undirected() if G.is_directed() else G, weight=weight)
    except Exception:
        eigen = {n: float("nan") for n in nodes}
    try:
        pagerank = nx.pagerank(G, weight=weight)
    except Exception:
        pagerank = {n: float("nan") for n in nodes}

    rows = []
    for n in nodes:
        rows.append({
            "node": n,
            "degree": float(degree.get(n, 0.0)),
            "strength": float(strength.get(n, 0.0)),
            "betweenness": float(betweenness.get(n, 0.0)),
            "closeness": float(closeness.get(n, 0.0)),
            "eigenvector": float(eigen.get(n, float("nan"))),
            "pagerank": float(pagerank.get(n, float("nan"))),
        })
    return pd.DataFrame(rows)


def detect_communities(G, weight: str = "weight", resolution: float = 1.0, seed: int = 42) -> Dict[Any, int]:
    """Louvain communities -> {node: community number starting at 1, largest community first}."""
    nx = require_networkx()
    if G.number_of_nodes() == 0:
        return {}
    H = G.to_undirected() if G.is_directed() else G
    if H.number_of_edges() == 0:
        return {n: i + 1 for i, n in enumerate(H.nodes())}
    comms = nx.community.louvain_communities(H, weight=weight, resolution=resolution, seed=seed)
    comms = sorted(comms, key=lambda c: (-len(c), sorted(map(str, c))[0]))
    out = {}
    for i, c in enumerate(comms):
        for n in c:
            out[n] = i + 1
    return out


def community_modularity(G, membership: Dict[Any, int], weight: str = "weight") -> float:
    nx = require_networkx()
    H = G.to_undirected() if G.is_directed() else G
    if H.number_of_edges() == 0 or not membership:
        return float("nan")
    groups: Dict[int, set] = {}
    for n, c in membership.items():
        groups.setdefault(c, set()).add(n)
    try:
        return float(nx.community.modularity(H, list(groups.values()), weight=weight))
    except Exception:
        return float("nan")


def _scale(values: Sequence[float], lo: float, hi: float) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    v = np.where(np.isfinite(v), v, 0.0)
    if v.size == 0:
        return v
    vmin, vmax = float(v.min()), float(v.max())
    if vmax - vmin < 1e-12:
        return np.full_like(v, (lo + hi) / 2.0)
    return lo + (v - vmin) / (vmax - vmin) * (hi - lo)


def _scale_sqrt(values: Sequence[float], lo: float, hi: float) -> np.ndarray:
    """Marker area proportional to the value (sqrt keeps 29 vs 32 documents visually similar)."""
    v = np.asarray(values, dtype=float)
    v = np.clip(np.where(np.isfinite(v), v, 0.0), 0.0, None)
    if v.size == 0:
        return v
    vmax = float(v.max())
    if vmax <= 0:
        return np.full_like(v, lo)
    return lo + (hi - lo) * np.sqrt(v / vmax)


def _spread(pos: Dict[Any, np.ndarray], min_dist: float, iters: int = 60) -> Dict[Any, np.ndarray]:
    """Push apart nodes closer than min_dist so circles and labels do not pile up."""
    keys = list(pos.keys())
    if len(keys) < 2:
        return pos
    P = np.array([pos[k] for k in keys], dtype=float)
    for _ in range(iters):
        D = P[:, None, :] - P[None, :, :]
        dist = np.sqrt((D ** 2).sum(-1)) + np.eye(len(P))
        over = np.clip(min_dist - dist, 0.0, None)
        np.fill_diagonal(over, 0.0)
        if over.max() <= 1e-6:
            break
        push = (D / dist[..., None]) * (over[..., None] * 0.5)
        P += push.sum(axis=1)
    span = np.abs(P).max() or 1.0
    P /= span
    return {k: P[i] for i, k in enumerate(keys)}


def network_layout(G, layout: str = "spring", seed: int = 42, weight: str = "weight") -> Dict[Any, np.ndarray]:
    nx = require_networkx()
    n = max(G.number_of_nodes(), 1)
    H = G.to_undirected() if G.is_directed() else G
    min_dist = 0.9 / np.sqrt(n)
    if layout == "circular":
        return nx.circular_layout(H)
    if layout == "kamada_kawai" and n > 1:
        try:
            _set_distance(H, weight)
            return _spread(nx.kamada_kawai_layout(H, weight="_distance"), min_dist)
        except Exception:
            pass
    k = 2.0 / np.sqrt(n)
    return _spread(nx.spring_layout(H, k=k, weight=weight, seed=seed, iterations=300), min_dist)


def plot_network_map(
    G,
    title: str,
    size_values: Optional[Dict[Any, float]] = None,
    membership: Optional[Dict[Any, int]] = None,
    community_names: Optional[Dict[int, str]] = None,
    labels: Optional[Dict[Any, str]] = None,
    label_top: int = 40,
    layout: str = "spring",
    seed: int = 42,
    weight: str = "weight",
    size_range: Optional[Sequence[float]] = None,
    figsize=(8.6, 7.2),
    legend_title: str = "묶음(커뮤니티)",
) -> str:
    """Labelled network map: node size = importance, colour = community, line width = relationship strength."""
    nx = require_networkx()
    st = get_active_style()
    dark = st.get("theme") == "dark"
    fg = "#F1F5F9" if dark else "#1E293B"
    halo = "#0F172A" if dark else "#FFFFFF"
    edge_base = "#94A3B8" if dark else "#64748B"

    fig, ax = plt.subplots(figsize=figsize)
    nodes = list(G.nodes())
    if not nodes:
        ax.text(0.5, 0.5, "표시할 연결 없음", ha="center", va="center", color=fg)
        ax.axis("off")
        b64 = fig_to_base64(fig)
        plt.close(fig)
        return b64

    pos = network_layout(G, layout=layout, seed=seed, weight=weight)
    size_values = size_values or dict(G.degree(weight=weight))
    if size_range is None:
        hi = float(np.clip(24000.0 / len(nodes), 240.0, 1900.0))
        size_range = (hi * 0.12, hi)
    sizes = _scale_sqrt([size_values.get(n, 0.0) for n in nodes], *size_range)

    membership = membership or {}
    comm_ids = sorted({membership.get(n, 0) for n in nodes})
    pal = _pal(max(len(comm_ids), 1))
    comm_color = {c: pal[i % len(pal)] for i, c in enumerate(comm_ids)}
    node_colors = [comm_color.get(membership.get(n, 0), pal[0]) for n in nodes]

    edges = list(G.edges(data=True))
    if edges:
        ew = [float(d.get(weight, 1.0) or 0.0) for _, _, d in edges]
        widths = _scale(ew, 0.5, 5.2)
        alphas = _scale(ew, 0.22, 0.75)
        edge_colors = []
        for (u, v, _), a in zip(edges, alphas):
            same = membership and membership.get(u) == membership.get(v)
            base = comm_color.get(membership.get(u, 0), edge_base) if same else edge_base
            rgba = matplotlib.colors.to_rgba(base, alpha=float(a))
            edge_colors.append(rgba)
        edge_kw = dict(arrows=True, arrowsize=11, connectionstyle="arc3,rad=0.08", node_size=list(sizes)) if G.is_directed() else {}
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v) for u, v, _ in edges], width=list(widths),
                               edge_color=edge_colors, **edge_kw)

    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes, node_size=list(sizes), node_color=node_colors,
                           edgecolors=halo, linewidths=1.2, alpha=0.93)

    # Labels: most important nodes first; a label that would overlap one already placed is moved
    # above / below its node, and dropped only if every position collides.
    order = sorted(range(len(nodes)), key=lambda i: -sizes[i])[:max(int(label_top), 0)]
    fsz = _scale(sizes, 7.8, 12.5)
    ax.margins(0.08)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    for i in order:
        n = nodes[i]
        x, y = pos[n]
        rad = np.sqrt(sizes[i]) / 2.0 + 5.0
        for dy in (0.0, rad + 4.0, -(rad + 4.0)):
            txt = ax.annotate((labels or {}).get(n, str(n)), xy=(x, y), xytext=(0, dy), textcoords="offset points",
                              ha="center", va="center", fontsize=float(fsz[i]), color=fg, fontweight="bold", zorder=5)
            txt.set_path_effects([path_effects.withStroke(linewidth=3.0, foreground=halo)])
            bb = txt.get_window_extent(renderer).expanded(1.03, 1.08)
            if not any(bb.overlaps(other) for other in placed):
                placed.append(bb)
                break
            txt.remove()

    if membership and len(comm_ids) > 1:
        handles = []
        for c in comm_ids[:10]:
            name = (community_names or {}).get(c, f"묶음 {c}")
            handles.append(plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=comm_color[c],
                                      markeredgecolor=halo, markersize=9, label=name))
        leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8.6, frameon=False,
                        title=legend_title, title_fontsize=9)
        if leg is not None:
            for t in leg.get_texts():
                t.set_color(fg)
            leg.get_title().set_color(fg)

    ax.set_title(title, fontsize=12.5, fontweight="bold", color=fg, pad=10)
    ax.axis("off")
    fig.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def plot_centrality_bars(df_cent: pd.DataFrame, metrics: List[str], top_n: int = 15, title: str = "중심성 상위 노드") -> str:
    st = get_active_style()
    fg = "#F1F5F9" if st.get("theme") == "dark" else "#1E293B"
    metrics = [m for m in metrics if m in df_cent.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics) + 1.2, 0.32 * min(top_n, len(df_cent)) + 1.6))
    axes = np.atleast_1d(axes)
    pal = _pal(len(metrics))
    for ax, m, col in zip(axes, metrics, pal):
        sub = df_cent.sort_values(m, ascending=False).head(top_n).iloc[::-1]
        ax.barh(sub["node"].astype(str), sub[m].fillna(0.0), color=col, alpha=0.9)
        ax.set_title(CENTRALITY_LABELS.get(m, m), fontsize=10.5, fontweight="bold", color=fg)
        ax.tick_params(axis="y", labelsize=8.6)
        ax.tick_params(axis="x", labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.suptitle(title, fontsize=12, fontweight="bold", color=fg)
    fig.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def add_centrality_table(analysis: BaseAnalysis, df_cent: pd.DataFrame, membership: Dict[Any, int], sort_by: str,
                         extra_cols: Optional[Dict[str, Dict[Any, Any]]] = None, title: str = "노드 중심성",
                         node_title: str = "노드", limit: Optional[int] = None) -> ResultsTable:
    tbl = ResultsTable(name="centrality", title=title)
    tbl.add_column("rank", "순위", col_type="integer")
    tbl.add_column("node", node_title, col_type="text")
    for key, spec in (extra_cols or {}).items():
        tbl.add_column(key, spec["title"], col_type=spec.get("type", "number"))
    tbl.add_column("degree", "연결 중심성", col_type="number")
    tbl.add_column("strength", "가중 연결정도", col_type="number", fmt="auto")
    tbl.add_column("betweenness", "매개 중심성", col_type="number")
    tbl.add_column("closeness", "근접 중심성", col_type="number")
    tbl.add_column("eigenvector", "고유벡터 중심성", col_type="number")
    tbl.add_column("pagerank", "PageRank", col_type="number")
    if membership:
        tbl.add_column("community", "묶음", col_type="integer")
    key = sort_by if sort_by in df_cent.columns else "strength"
    ordered = df_cent.sort_values(key, ascending=False).reset_index(drop=True)
    if limit:
        ordered = ordered.head(limit)
    for i, r in ordered.iterrows():
        row = {"rank": i + 1, "node": str(r["node"])}
        for k2, spec in (extra_cols or {}).items():
            row[k2] = spec["values"].get(r["node"])
        for m in CENTRALITY_LABELS:
            row[m] = r[m]
        if membership:
            row["community"] = membership.get(r["node"])
        tbl.add_row(row)
    tbl.add_note(f"정렬 기준: {CENTRALITY_LABELS.get(key, key)}. 매개·근접 중심성의 연결 거리 = 최대 가중치 / 가중치 (가장 강한 연결 = 1).")
    analysis.add_table(tbl)
    return tbl


class GraphNetworkAnalysis(BaseAnalysis):
    """Relational network analysis from an edge list (source, target, optional weight)."""

    def __init__(
        self,
        data: pd.DataFrame,
        source: str,
        target: str,
        weight: Optional[str] = None,
        directed: bool = False,
        min_weight: float = 0.0,
        community: bool = True,
        resolution: float = 1.0,
        size_by: str = "strength",
        layout: str = "spring",
        label_top: int = 30,
        top_n: int = 30,
        plot_network: bool = True,
        plot_centrality: bool = True,
        seed: int = 42,
    ):
        super().__init__(
            name="graphNetwork",
            title="관계망 분석 (Relational Network Analysis)",
            dataset=data,
            options={
                "source": source, "target": target, "weight": weight or "", "directed": directed,
                "minWeight": min_weight, "community": community, "resolution": resolution,
                "sizeBy": size_by, "layout": layout, "labelTop": label_top, "topN": top_n,
                "plotNetwork": plot_network, "plotCentrality": plot_centrality, "seed": seed,
            },
        )
        self.source, self.target, self.weight = source, target, weight
        self.syntax_python = (f"sess.graph_network(source={source!r}, target={target!r}"
                              + (f", weight={weight!r}" if weight else "") + f", directed={directed})")
        self.syntax_r = "igraph::graph_from_data_frame(edges, directed = %s)" % ("TRUE" if directed else "FALSE")
        self.run()

    def run(self):
        nx = require_networkx()
        o = self.options
        df = self.dataset
        cols = [self.source, self.target] + ([self.weight] if self.weight else [])
        sub = df[cols].copy()
        sub = sub.dropna(subset=[self.source, self.target])
        sub[self.source] = sub[self.source].astype(str).str.strip()
        sub[self.target] = sub[self.target].astype(str).str.strip()
        sub = sub[(sub[self.source] != "") & (sub[self.target] != "")]
        if self.weight:
            sub["_w"] = pd.to_numeric(sub[self.weight], errors="coerce")
            sub = sub.dropna(subset=["_w"])
        else:
            sub["_w"] = 1.0
        n_self = int((sub[self.source] == sub[self.target]).sum())
        sub = sub[sub[self.source] != sub[self.target]]
        if not o["directed"]:
            a = sub[[self.source, self.target]].min(axis=1)
            b = sub[[self.source, self.target]].max(axis=1)
            sub[self.source], sub[self.target] = a, b
        agg = sub.groupby([self.source, self.target], as_index=False)["_w"].sum()
        agg = agg[agg["_w"] > float(o["minWeight"] or 0.0)]
        if agg.empty:
            raise ValueError("관계망 분석: 유효한 연결(출발·도착·가중치)이 없음. 변수 지정과 최소 가중치를 확인 필요")

        G = nx.DiGraph() if o["directed"] else nx.Graph()
        for s, t, w in agg.itertuples(index=False):
            G.add_edge(s, t, weight=float(w))

        df_cent = compute_centrality(G)
        membership = detect_communities(G, resolution=float(o["resolution"]), seed=int(o["seed"])) if o["community"] else {}
        modularity = community_modularity(G, membership) if membership else float("nan")

        # 1. Summary
        H = G.to_undirected() if G.is_directed() else G
        comps = list(nx.connected_components(H))
        giant = H.subgraph(max(comps, key=len)).copy() if comps else H
        try:
            apl = nx.average_shortest_path_length(giant) if giant.number_of_nodes() > 1 else float("nan")
        except Exception:
            apl = float("nan")
        summ = ResultsTable(name="summary", title="네트워크 요약")
        summ.add_column("stat", "지표", col_type="text")
        summ.add_column("value", "값", col_type="number", fmt="auto")
        rows = [
            ("노드 수", G.number_of_nodes()), ("연결(링크) 수", G.number_of_edges()),
            ("밀도", nx.density(G)), ("구성요소 수", len(comps)),
            ("최대 구성요소 노드 수", giant.number_of_nodes()), ("평균 경로 길이 (최대 구성요소)", apl),
            ("군집계수 (transitivity)", nx.transitivity(H)), ("평균 연결정도", 2.0 * H.number_of_edges() / max(H.number_of_nodes(), 1)),
        ]
        if membership:
            rows += [("묶음(커뮤니티) 수", len(set(membership.values()))), ("모듈성 Q", modularity)]
        for k, v in rows:
            summ.add_row({"stat": k, "value": float(v) if v is not None else float("nan")})
        summ.add_note(f"{'방향' if o['directed'] else '무방향'} 네트워크. 중복 연결은 가중치 합산"
                      + (f", 자기 연결 {n_self}건 제외" if n_self else "") + ".")
        if membership:
            summ.add_note(f"묶음: Louvain 알고리즘 (해상도 {o['resolution']}, 시드 {o['seed']}).")
        self.add_table(summ)

        # 2. Centrality
        add_centrality_table(self, df_cent, membership, sort_by=o["sizeBy"], limit=int(o["topN"]) or None)

        # 3. Communities
        if membership:
            ctab = ResultsTable(name="communities", title="묶음(커뮤니티) 구성")
            ctab.add_column("community", "묶음", col_type="integer")
            ctab.add_column("n", "노드 수", col_type="integer")
            ctab.add_column("members", "주요 노드 (가중 연결정도 순)", col_type="text")
            strength = dict(zip(df_cent["node"], df_cent["strength"]))
            for c in sorted(set(membership.values())):
                mem = sorted([n for n, cc in membership.items() if cc == c], key=lambda n: -strength.get(n, 0))
                ctab.add_row({"community": c, "n": len(mem), "members": ", ".join(map(str, mem[:12])) + (" …" if len(mem) > 12 else "")})
            self.add_table(ctab)

        if o["plotNetwork"]:
            size_vals = dict(zip(df_cent["node"], df_cent[o["sizeBy"]] if o["sizeBy"] in df_cent.columns else df_cent["strength"]))
            names = {}
            if membership:
                strength = dict(zip(df_cent["node"], df_cent["strength"]))
                for c in sorted(set(membership.values())):
                    top = sorted([n for n, cc in membership.items() if cc == c], key=lambda n: -strength.get(n, 0))[:2]
                    names[c] = f"묶음 {c}: " + "·".join(map(str, top))
            self.add_plot("관계망 그림", plot_network_map(
                G, title="관계망 그림 (노드 크기 = " + CENTRALITY_LABELS.get(o["sizeBy"], "가중 연결정도") + ")",
                size_values=size_vals, membership=membership, community_names=names,
                label_top=int(o["labelTop"]), layout=o["layout"], seed=int(o["seed"])))
        if o["plotCentrality"]:
            self.add_plot("중심성 비교", plot_centrality_bars(df_cent, ["strength", "betweenness", "eigenvector"], top_n=15))

        top = df_cent.sort_values("strength", ascending=False).iloc[0]
        self.apa_narrative = (
            f"관계망은 노드 {G.number_of_nodes()}개, 연결 {G.number_of_edges()}개로 구성되며 밀도는 {nx.density(G):.3f}이다. "
            f"가중 연결정도가 가장 높은 노드는 '{top['node']}'({top['strength']:.2f})이다."
            + (f" Louvain 묶음 {len(set(membership.values()))}개가 도출되었고 모듈성 Q = {modularity:.3f}이다." if membership else "")
        )

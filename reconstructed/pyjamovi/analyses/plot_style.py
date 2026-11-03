"""
OpenStat Studio — Customizable Statistical Graph Design & Styling Engine (PlotStyle)
License: MIT
Allows full programmatic (Python / Claude Code / CLI / MCP) and interactive UI control over:
- Color Palettes (10 academic presets + arbitrary custom hex colors)
- Visual Themes (clean, apa_classic, whitegrid, minimal, dark)
- Typography & Korean CJK Font Auto-Support (KoPubDotum / Malgun / Nanum / DejaVu)
- Custom Titles, Axis Labels, Dimensions (width x height), DPI, Gridlines, Legend placement
- Multi-format Vector & Raster Rendering (.png, .svg, .pdf)
"""

from contextlib import contextmanager
import io
import base64
import logging
import threading
import warnings
from typing import Dict, List, Optional, Any
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


PALETTES: Dict[str, List[str]] = {
    "openstat": ["#0D9488", "#2563EB", "#D97706", "#E11D48", "#7C3AED", "#0284C7"],
    "apa_bw": ["#111827", "#4B5563", "#9CA3AF", "#D1D5DB", "#374151", "#6B7280"],
    "nature": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4"],
    "lancet": ["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", "#FDAF91"],
    "okabe_ito": ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"],
    "kihasa": ["#00457C", "#2A7B9B", "#D97724", "#5C6F84", "#8FA3B8", "#B84A39"],
    "viridis": ["#440154", "#3B528B", "#21918C", "#5EC962", "#FDE725", "#287C8E"],
    "warm": ["#C2410C", "#EA580C", "#F59E0B", "#E11D48", "#9333EA", "#B45309"],
    "pastel": ["#60A5FA", "#34D399", "#FBBF24", "#F87171", "#A78BFA", "#38BDF8"],
    "classic_blue": ["#2B5C8F", "#D97724", "#3B9A6C", "#C94A4A", "#7B61A8", "#2B9EB3"],
}

THEMES = ("clean", "apa_classic", "whitegrid", "minimal", "dark")

_TLS = threading.local()


def _detect_korean_capable_fonts() -> List[str]:
    preferred = [
        "KoPubDotum Bold",
        "KoPubWorldDotum_Pro",
        "NanumGothic",
        "Malgun Gothic",
        "AppleGothic",
        "Noto Sans CJK KR",
        "UnDotum",
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    found = [f for f in preferred if f in installed]
    return found + ["DejaVu Sans", "Segoe UI", "Arial", "sans-serif"]


DEFAULT_FONT_STACK = _detect_korean_capable_fonts()


def normalize_plot_style(style: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    s = dict(style or {})
    pal_name = str(s.get("palette", "openstat")).strip()
    custom_cols = s.get("custom_colors") or s.get("customColors")
    colors: List[str]
    if custom_cols:
        if isinstance(custom_cols, str):
            parsed = [c.strip() for c in custom_cols.split(",") if c.strip()]
        else:
            parsed = [str(c).strip() for c in custom_cols if str(c).strip()]
        colors = parsed if parsed else list(PALETTES["openstat"])
        # Pad to at least 6 colors by cycling
        while len(colors) < 6:
            colors.extend(parsed)
        colors = colors[:8]
    else:
        colors = list(PALETTES.get(pal_name, PALETTES["openstat"]))

    theme = str(s.get("theme", "clean")).strip().lower()
    if theme not in THEMES:
        theme = "clean"

    dpi = int(s.get("dpi", 150) or 150)
    dpi = max(72, min(600, dpi))

    font_scale = float(s.get("font_scale", s.get("fontScale", 1.0)) or 1.0)
    font_scale = max(0.65, min(2.0, font_scale))

    width = s.get("width")
    height = s.get("height")
    w_val = float(width) if width not in (None, "", 0, "0") else None
    h_val = float(height) if height not in (None, "", 0, "0") else None

    show_grid = s.get("show_grid", s.get("showGrid", None))
    if show_grid is not None:
        show_grid = bool(show_grid)

    return {
        "palette": pal_name,
        "colors": colors,
        "primary": colors[0],
        "secondary": colors[1 % len(colors)],
        "accent": colors[2 % len(colors)],
        "danger": colors[3 % len(colors)],
        "heading_color": "#F8FAFC" if theme == "dark" else ("#111111" if theme == "apa_classic" else "#1E3A5F"),
        "theme": theme,
        "title": str(s.get("title", "")).strip() or None,
        "xlabel": str(s.get("xlabel", "")).strip() or None,
        "ylabel": str(s.get("ylabel", "")).strip() or None,
        "width": w_val,
        "height": h_val,
        "dpi": dpi,
        "font_scale": font_scale,
        "show_grid": show_grid,
        "legend_pos": str(s.get("legend_pos", s.get("legendPos", "auto"))).strip().lower(),
    }


def get_active_style() -> Dict[str, Any]:
    st = getattr(_TLS, "style", None)
    if st is None:
        return normalize_plot_style(None)
    return st


@contextmanager
def use_plot_style(style: Optional[Dict[str, Any]] = None):
    prev = getattr(_TLS, "style", None)
    norm = normalize_plot_style(style)
    _TLS.style = norm

    is_dark = (norm["theme"] == "dark")
    bg = "#0F172A" if is_dark else "#FFFFFF"
    fg = "#F1F5F9" if is_dark else "#1E293B"
    edge = "#94A3B8" if is_dark else ("#111111" if norm["theme"] == "apa_classic" else "#334155")
    fs = norm["font_scale"]

    rc_overrides = {
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "savefig.facecolor": bg,
        "text.color": fg,
        "axes.labelcolor": fg,
        "xtick.color": fg,
        "ytick.color": fg,
        "axes.edgecolor": edge,
        "axes.linewidth": 1.1 if norm["theme"] == "apa_classic" else 0.9,
        "font.sans-serif": DEFAULT_FONT_STACK,
        "axes.unicode_minus": False,
        "font.size": 10.0 * fs,
        "axes.titlesize": 11.5 * fs,
        "axes.labelsize": 10.5 * fs,
        "xtick.labelsize": 9.5 * fs,
        "ytick.labelsize": 9.5 * fs,
    }
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Glyph .* missing from font")
            with matplotlib.rc_context(rc_overrides):
                yield norm
    finally:
        _TLS.style = prev


def apply_style_to_axes(ax: plt.Axes):
    st = get_active_style()
    theme = st["theme"]
    show_grid = st["show_grid"]

    if theme == "apa_classic":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#111111")
        ax.spines["bottom"].set_color("#111111")
        ax.grid(False)
    elif theme == "whitegrid":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1")
    elif theme == "minimal":
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", linestyle="-", alpha=0.25, color="#CBD5E1")
    elif theme == "dark":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#0F172A")
        ax.grid(axis="y", linestyle=":", alpha=0.30, color="#475569")
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.35, color="#94A3B8")

    if show_grid is True:
        ax.grid(True, linestyle=":", alpha=0.42)
    elif show_grid is False:
        ax.grid(False)


def finalize_and_serialize_figure(fig: plt.Figure, default_dpi: int = 150) -> str:
    """Applies active PlotStyle overrides (dimensions, custom labels, legend, theme) and captures PNG + Vector SVG."""
    st = get_active_style()
    dpi = st.get("dpi") or default_dpi
    bg = "#0F172A" if st["theme"] == "dark" else "white"

    if st["width"] and st["height"]:
        fig.set_size_inches(st["width"], st["height"])
    elif st["width"]:
        cur_w, cur_h = fig.get_size_inches()
        fig.set_size_inches(st["width"], cur_h * (st["width"] / max(cur_w, 1.0)))
    elif st["height"]:
        cur_w, cur_h = fig.get_size_inches()
        fig.set_size_inches(cur_w * (st["height"] / max(cur_h, 1.0)), st["height"])

    if fig.axes:
        main_ax = fig.axes[0]
        if st["title"]:
            if len(fig.axes) > 1 and fig._suptitle is not None:
                fig.suptitle(st["title"], color=st["heading_color"], fontweight="bold")
            else:
                main_ax.set_title(st["title"], color=st["heading_color"], fontweight="bold")
        elif main_ax.get_title():
            main_ax.title.set_color(st["heading_color"])

        if st["xlabel"]:
            main_ax.set_xlabel(st["xlabel"])
        if st["ylabel"]:
            main_ax.set_ylabel(st["ylabel"])

        leg = main_ax.get_legend()
        lpos = st["legend_pos"]
        if leg is not None and lpos != "auto":
            if lpos == "none":
                leg.remove()
            elif lpos == "top":
                sns_handles, sns_labels = main_ax.get_legend_handles_labels()
                if sns_handles:
                    main_ax.legend(sns_handles, sns_labels, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=min(4, len(sns_labels)), frameon=False)
            elif lpos == "bottom":
                sns_handles, sns_labels = main_ax.get_legend_handles_labels()
                if sns_handles:
                    main_ax.legend(sns_handles, sns_labels, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(4, len(sns_labels)), frameon=False)
            elif lpos == "right":
                sns_handles, sns_labels = main_ax.get_legend_handles_labels()
                if sns_handles:
                    main_ax.legend(sns_handles, sns_labels, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
            elif lpos in ("upper right", "upper left", "lower right", "lower left", "center right"):
                sns_handles, sns_labels = main_ax.get_legend_handles_labels()
                if sns_handles:
                    main_ax.legend(sns_handles, sns_labels, loc=lpos, frameon=True, facecolor=bg)

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Glyph .* missing from font")
        # 1. Render PNG base64
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=bg)
        png_bytes = png_buf.getvalue()
        png_b64 = f"data:image/png;base64,{base64.b64encode(png_bytes).decode('utf-8')}"

        # 2. Render Vector SVG & PDF in memory for lossless design / publication export
        svg_buf = io.BytesIO()
        fig.savefig(svg_buf, format="svg", bbox_inches="tight", facecolor=bg)
        svg_str = svg_buf.getvalue().decode("utf-8")

        pdf_buf = io.BytesIO()
        fig.savefig(pdf_buf, format="pdf", bbox_inches="tight", facecolor=bg)
        pdf_b64 = base64.b64encode(pdf_buf.getvalue()).decode("utf-8")

    if not hasattr(_TLS, "vector_queue"):
        _TLS.vector_queue = []
    _TLS.vector_queue.append({"svg": svg_str, "pdf_b64": pdf_b64, "style": st})

    plt.close(fig)
    return png_b64


def pop_last_vector_bundle() -> Optional[Dict[str, Any]]:
    q = getattr(_TLS, "vector_queue", None)
    if q:
        return q.pop(0)
    return None

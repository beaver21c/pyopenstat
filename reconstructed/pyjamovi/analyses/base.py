"""
pyjamovi Base Analysis & Result Elements Model (v2.0)
Re-creates jmvcore Table, Image/Plot, Syntax, Group, and BaseAnalysis.
"""

from typing import List, Dict, Any, Optional
import json
import math
import numpy as np
import pandas as pd
from tabulate import tabulate


def _is_nan(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, (float, np.floating)):
        return math.isnan(float(val)) or math.isinf(float(val))
    try:
        return bool(pd.isna(val))
    except Exception:
        return False


def resolve_choice(label: str, value: Any, allowed, aliases: Optional[Dict[str, str]] = None,
                   default: Optional[str] = None) -> str:
    """Canonical value of a select-type option.

    Matching ignores case and surrounding blanks, aliases map to canonical keys, and an unknown
    value raises ValueError listing the accepted keys (no silent substitution).
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        if default is None:
            raise ValueError(f"{label}: 값 지정 필요 (가능: {', '.join(allowed)})")
        return default
    raw = str(value).strip()
    keys = list(allowed)
    if raw in keys:
        return raw
    low = raw.lower()
    for k in keys:
        if k.lower() == low:
            return k
    for a, k in (aliases or {}).items():
        if a.lower() == low:
            return k
    raise ValueError(f"{label}: 알 수 없는 값 '{raw}' (가능: {', '.join(keys)})")


class ResultsTable:
    def __init__(self, name: str, title: str, columns: Optional[List[Dict[str, Any]]] = None):
        self.name = name
        self.title = title
        self.columns = columns or []
        self.rows: List[Dict[str, Any]] = []
        self.notes: List[str] = []

    def add_column(self, name: str, title: Optional[str] = None, col_type: str = "number", fmt: str = "", super_title: str = ""):
        self.columns.append({
            "name": name,
            "title": title if title is not None else name,
            "type": col_type,
            "format": fmt,
            "superTitle": super_title
        })

    def add_row(self, row_dict: Dict[str, Any]):
        self.rows.append(row_dict)

    def add_note(self, note: str):
        if note not in self.notes:
            self.notes.append(note)

    @staticmethod
    def display_title(col: Dict[str, Any]) -> str:
        """Column title for flat (single header row) outputs: super title + title when a super title is set."""
        t = col.get("title") or col.get("name") or ""
        st = col.get("superTitle") or ""
        return f"{st} {t}".strip() if st else t

    def to_dataframe(self) -> pd.DataFrame:
        col_names = [c["name"] for c in self.columns]
        data = [[r.get(c) for c in col_names] for r in self.rows]
        titles = [self.display_title(c) for c in self.columns]
        return pd.DataFrame(data, columns=titles)

    def _format_cell(self, val: Any, col_meta: Dict[str, Any]) -> str:
        if val == "":
            return ""
        if _is_nan(val):
            return "—"
        if isinstance(val, (int, np.integer)) and col_meta.get("type") == "integer":
            return str(int(val))
        if isinstance(val, (int, float, np.integer, np.floating)):
            fval = float(val)
            fmt = col_meta.get("format", "")
            if col_meta.get("type") == "integer" and abs(fval - round(fval)) < 1e-9:
                return str(int(round(fval)))
            if fmt == "pvalue":
                if fval < 0.001:
                    return "< .001"
                return f"{fval:.3f}"
            elif fmt == "zto":
                return f"{fval:.3f}"
            elif fmt == "pc":
                return f"{fval:.1f}%"
            elif fmt == "auto" and abs(fval - round(fval)) < 1e-9 and abs(fval) < 1e15:
                return f"{int(round(fval)):,}" if abs(fval) >= 1e4 else str(int(round(fval)))
            else:
                if abs(fval) >= 1e5 or (0 < abs(fval) < 1e-4):
                    return f"{fval:.3e}"
                return f"{fval:.3f}"
        return str(val)

    def to_string(self) -> str:
        headers = [self.display_title(c) for c in self.columns]
        table_data = [[self._format_cell(r.get(c["name"]), c) for c in self.columns] for r in self.rows]
        output = [f"\n{self.title}", tabulate(table_data, headers=headers, tablefmt="simple")]
        for n in self.notes:
            output.append(f"  Note. {n}")
        return "\n".join(output)

    def to_markdown(self) -> str:
        headers = [self.display_title(c) or " " for c in self.columns]
        table_data = [[self._format_cell(r.get(c["name"]), c) for c in self.columns] for r in self.rows]
        output = [f"### {self.title}\n", tabulate(table_data, headers=headers, tablefmt="github")]
        if self.notes:
            output.append("")
            for n in self.notes:
                output.append(f"> *Note.* {n}")
        return "\n".join(output)

    def to_html(self) -> str:
        headers = [self.display_title(c) for c in self.columns]
        html = ["<div class='jamovi-table-container'>", f"<h4>{self.title}</h4>", "<table class='apa-table'>", "<thead><tr>"]
        for h in headers:
            html.append(f"<th>{h}</th>")
        html.append("</tr></thead><tbody>")
        for r in self.rows:
            html.append("<tr>")
            for col in self.columns:
                val = r.get(col["name"])
                cls = " class='num'" if col.get("type") in ("number", "integer") else ""
                html.append(f"<td{cls}>{self._format_cell(val, col)}</td>")
            html.append("</tr>")
        html.append("</tbody></table>")
        for n in self.notes:
            html.append(f"<div class='footnote'><em>Note.</em> {n}</div>")
        html.append("</div>")
        return "".join(html)

    def to_dict(self) -> Dict[str, Any]:
        cleaned_rows = []
        for r in self.rows:
            cleaned_r = {}
            for k, v in r.items():
                if v == "":
                    cleaned_r[k] = ""
                elif _is_nan(v):
                    cleaned_r[k] = None
                elif isinstance(v, (np.integer,)):
                    cleaned_r[k] = int(v)
                elif isinstance(v, (np.floating, float)):
                    cleaned_r[k] = float(v)
                else:
                    cleaned_r[k] = v
            cleaned_rows.append(cleaned_r)
        return {
            "name": self.name,
            "title": self.title,
            "columns": self.columns,
            "rows": cleaned_rows,
            "notes": self.notes
        }


class BaseAnalysis:
    def __init__(self, name: str, title: str, dataset: Optional[Any] = None, options: Optional[Dict[str, Any]] = None):
        self.name = name
        self.title = title
        self.dataset = dataset
        self.options = options or {}
        self.tables: List[ResultsTable] = []
        self.plots: List[Dict[str, Any]] = []  # [{"title": ..., "src": "data:image/png;base64,..."}]
        self.plot_vectors: List[Dict[str, Any]] = []  # [{"svg": "<svg...>", "pdf_b64": "...", "style": {...}}]
        self.apa_narrative: str = ""
        self.syntax_python: str = ""
        self.syntax_r: str = ""
        self.references: List[str] = [
            "OpenStat Studio Contributors (2026). OpenStat Studio v3.0 (MIT License) — Pure-Python Statistical Engine.",
            "Analytical workflow, menu taxonomy, and .omv interoperability designed with reference to The jamovi project (https://www.jamovi.org)."
        ]

    def add_table(self, table: ResultsTable):
        self.tables.append(table)

    def add_plot(self, title: str, src: str):
        if src:
            from .plot_style import pop_last_vector_bundle
            vec = pop_last_vector_bundle() or {}
            self.plots.append({
                "title": title,
                "src": src,
                "style": vec.get("style", {})
            })
            self.plot_vectors.append(vec)

    def attach_figure(self, fig_or_src: Any, title: str = "Attached Visualization", plot_style: Optional[Dict[str, Any]] = None) -> "BaseAnalysis":
        """Attach a custom Matplotlib Figure or base64 PNG data URI directly to this analysis result card."""
        import matplotlib.pyplot as _plt
        if isinstance(fig_or_src, _plt.Figure):
            from .plot_style import use_plot_style, finalize_and_serialize_figure
            with use_plot_style(plot_style):
                png_src = finalize_and_serialize_figure(fig_or_src)
            self.add_plot(title, png_src)
        elif isinstance(fig_or_src, str):
            self.add_plot(title, fig_or_src)
        return self

    def get_table(self, name: str) -> Optional[ResultsTable]:
        for t in self.tables:
            if t.name == name:
                return t
        return None

    def run(self):
        raise NotImplementedError

    def to_string(self) -> str:
        parts = [f"=== {self.title} ==="]
        for t in self.tables:
            parts.append(t.to_string())
        if self.apa_narrative:
            parts.append(f"\n[APA 7th Summary]\n{self.apa_narrative}")
        return "\n".join(parts)

    def to_markdown(self) -> str:
        parts = [f"## {self.title}\n"]
        if self.syntax_python:
            parts.append(f"```python\n{self.syntax_python}\n```\n")
        for t in self.tables:
            parts.append(t.to_markdown())
            parts.append("")
        if self.plots:
            parts.append(f"*Generated {len(self.plots)} plot(s): {', '.join(p['title'] for p in self.plots)}*\n")
        if self.apa_narrative:
            parts.append(f"**APA Style Narrative:**\n> {self.apa_narrative}\n")
        return "\n".join(parts)

    def to_html_snippet(self) -> str:
        parts = ["<div class='analysis-block'>"]
        for t in self.tables:
            parts.append(t.to_html())
        for p in self.plots:
            parts.append(f"<div class='jamovi-plot'><h4>{p['title']}</h4><img src='{p['src']}' style='max-width:100%;height:auto;'/></div>")
        if self.apa_narrative:
            parts.append(f"<p class='apa-summary'><strong>APA Summary:</strong> {self.apa_narrative}</p>")
        parts.append("</div>")
        return "\n".join(parts)

    def to_payload(self, analysis_id: Optional[int] = None) -> Dict[str, Any]:
        return {
            "status": "success",
            "analysisId": analysis_id,
            "name": self.name,
            "title": self.title,
            "options": self.options,
            "tables": [t.to_dict() for t in self.tables],
            "plots": self.plots,
            "apa": self.apa_narrative,
            "markdown": self.to_markdown(),
            "syntax_python": self.syntax_python,
            "syntax_r": self.syntax_r,
            "references": self.references
        }

    def serialize_state(self) -> str:
        state = {
            "name": self.name,
            "title": self.title,
            "options": self.options,
            "tables": [t.to_dict() for t in self.tables],
            "apa_narrative": self.apa_narrative,
            "syntax_python": self.syntax_python,
            "syntax_r": self.syntax_r
        }
        return json.dumps(state, indent=2)

    def save_omv(self, output_path: str):
        from pyjamovi.omv.writer import write_omv
        if self.dataset is None:
            raise ValueError("No dataset attached to analysis to save as .omv")
        return write_omv(self.dataset, output_path, analyses=[self])

    def export_tables(self, output_path: str, fmt: Optional[str] = None, dataset: Optional[Any] = None) -> str:
        """Export analyzed statistical tables & embedded graphs to .xlsx, .docx, .hwpx, .pdf, .tex, .csv, .tsv, .md, .html, .json, or .omv."""
        from pyjamovi.exporters import export_analyses
        ds = dataset if dataset is not None else self.dataset
        return export_analyses([self], output_path=output_path, fmt=fmt, dataset=ds)

    def export_results(self, output_path: str, fmt: Optional[str] = None, dataset: Optional[Any] = None) -> str:
        """Export full analysis results (tables + embedded graphs + APA interpretation) to 11 formats."""
        return self.export_tables(output_path=output_path, fmt=fmt, dataset=dataset)

    def save_plot(self, output_path: str, plot_index: int = 0) -> str:
        """Save a rendered plot to .png, .svg (vector), or .pdf (vector)."""
        import base64 as _b64
        if not self.plots or plot_index >= len(self.plots):
            raise IndexError("Requested plot_index does not exist on this analysis.")
        ext = output_path.rsplit(".", 1)[-1].lower() if "." in output_path else "png"
        vec = self.plot_vectors[plot_index] if plot_index < len(self.plot_vectors) else {}
        if ext == "svg" and vec.get("svg"):
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(vec["svg"])
            return output_path
        if ext == "pdf" and vec.get("pdf_b64"):
            with open(output_path, "wb") as f:
                f.write(_b64.b64decode(vec["pdf_b64"]))
            return output_path
        src = self.plots[plot_index]["src"]
        raw = _b64.b64decode(src.split(",", 1)[1] if "," in src else src)
        with open(output_path, "wb") as f:
            f.write(raw)
        return output_path

    def save_py(self, output_path: str, data_path: str = "dataset.csv") -> str:
        """Save reproducible standalone Python (.py) script for this analysis."""
        from pyjamovi.syntax_engine import generate_python_script
        code = generate_python_script([{"analysis_type": self.name, "title": self.title, "params": self.options}], dataset_path=data_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        return output_path

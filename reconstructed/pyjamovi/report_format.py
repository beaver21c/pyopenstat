"""
OpenStat Studio — social-science report format for exported result tables.
License: MIT

The default export keeps every statistic in its own column (APA three-line tables).
`report_style="social"` applies the conventions of journal and policy-report tables:

- significance stars after the coefficient (or β), with a star legend in the table note
- standard errors in parentheses beside the coefficient, on the row below, or in their own column
- the p column dropped once stars carry the information (optional)
- no leading zero for values bounded by 1 (p, β, R², proportions), fixed decimals
- numbered table captions (<표 n> / Table n.) and merged header cells for grouped columns

Stars, parentheses and the p-column rule apply only to tables marked as coefficient tables
(`ResultsTable.mark_coefficients`). Other tables keep their columns; only the number notation,
captions and header merging change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .analyses.base import ResultsTable, resolve_choice

REPORT_STYLES = ("default", "social")
_STYLE_ALIASES = {"apa": "default", "기본": "default", "none": "default", "standard": "default",
                  "사회과학": "social", "journal": "social", "paper": "social", "학술": "social"}

STAR_PRESETS: Dict[str, List[Tuple[float, str]]] = {
    "standard": [(0.05, "*"), (0.01, "**"), (0.001, "***")],
    "dagger": [(0.10, "†"), (0.05, "*"), (0.01, "**"), (0.001, "***")],
    "econ": [(0.10, "*"), (0.05, "**"), (0.01, "***")],
    "none": [],
}
_STAR_ALIASES = {"apa": "standard", "default": "standard", "기본": "standard", "†": "dagger", "plus": "dagger",
                 "economics": "econ", "경제학": "econ", "off": "none", "없음": "none", "false": "none"}

SE_POSITIONS = ("beside", "below", "column")
_SE_ALIASES = {"inline": "beside", "옆": "beside", "괄호": "beside", "stacked": "below", "under": "below",
               "아래": "below", "separate": "column", "열": "column", "별도": "column"}

STAR_TARGETS = ("estimate", "std_estimate")
_TARGET_ALIASES = {"b": "estimate", "coef": "estimate", "est": "estimate", "계수": "estimate",
                   "beta": "std_estimate", "β": "std_estimate", "std": "std_estimate", "stdest": "std_estimate",
                   "표준화": "std_estimate"}

LANGS = ("ko", "en")


@dataclass
class ReportFormat:
    """Options of the social-science table format (all optional)."""
    stars: Any = "standard"          # preset name or [[threshold, symbol], ...]
    star_target: str = "estimate"    # estimate (B) | std_estimate (β)
    se_position: str = "beside"      # beside: 0.512*** (0.103) | below: SE row under B | column: own column
    keep_p: bool = False             # keep the p column of coefficient tables
    keep_stat: bool = True           # keep t / z columns
    keep_ci: bool = True             # keep confidence-interval columns
    decimals: int = 3                # decimals of estimates and other non-integer statistics
    leading_zero: bool = False       # write 0.042 instead of .042 for p, β, R² and proportions
    star_note: bool = True           # add the star legend (and the SE note) to the table note
    table_numbers: bool = True       # numbered captions <표 n> / Table n.
    lang: str = "ko"                 # language of generated notes and captions
    raw_sheet: bool = True           # xlsx: add a sheet of unformatted numeric values
    star_levels: List[Tuple[float, str]] = field(default_factory=list, repr=False)

    def options(self) -> Dict[str, Any]:
        """Non-default options (for reproducible syntax)."""
        base = ReportFormat()
        out = {}
        for f in fields(self):
            if f.name == "star_levels":
                continue
            v = getattr(self, f.name)
            if v != getattr(base, f.name):
                out[f.name] = v
        return out


OPTION_KEYS = tuple(f.name for f in fields(ReportFormat) if f.name != "star_levels")


def _as_bool(label: str, v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "1", "yes", "y", "on", "예"):
        return True
    if isinstance(v, str) and v.strip().lower() in ("false", "0", "no", "n", "off", "아니오"):
        return False
    raise ValueError(f"{label}: 참/거짓 값 필요 (입력값 {v!r})")


def resolve_stars(value: Any) -> Tuple[Any, List[Tuple[float, str]]]:
    """(canonical option value, [(threshold, symbol)] sorted from the largest threshold)."""
    if value is None or value is False:
        value = "none"
    if isinstance(value, str):
        key = resolve_choice("별표 기준(stars)", value, tuple(STAR_PRESETS), _STAR_ALIASES)
        return key, sorted(STAR_PRESETS[key], key=lambda x: -x[0])
    if isinstance(value, dict):
        value = [[k, v] for k, v in value.items()]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("별표 기준(stars): 'standard'·'dagger'·'econ'·'none' 또는 [[0.05, '*'], ...] 형식 필요")
    levels = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"별표 기준(stars): [기준값, 기호] 쌍 필요 (입력값 {item!r})")
        a, b = item
        if isinstance(a, str) and not isinstance(b, str):
            a, b = b, a
        try:
            thr = float(a)
        except (TypeError, ValueError):
            raise ValueError(f"별표 기준(stars): 기준값은 숫자 (입력값 {a!r})") from None
        if not (0.0 < thr < 1.0):
            raise ValueError(f"별표 기준(stars): 기준값은 0과 1 사이 (입력값 {thr})")
        sym = str(b).strip()
        if not sym:
            raise ValueError("별표 기준(stars): 기호가 빈 문자열")
        levels.append((thr, sym))
    if len({t for t, _ in levels}) != len(levels):
        raise ValueError("별표 기준(stars): 기준값 중복")
    levels.sort(key=lambda x: -x[0])
    return [[t, s] for t, s in levels], levels


def resolve_report_format(report_style: Any = None, report_options: Optional[Dict[str, Any]] = None) -> Optional[ReportFormat]:
    """ReportFormat for `report_style="social"`, None for the default APA export.

    `report_style` may also be a dict of options (implies "social") or a ReportFormat instance.
    Unknown option names and values raise ValueError.
    """
    if isinstance(report_style, ReportFormat):
        if report_options:
            raise ValueError("report_options: ReportFormat 객체와 함께 지정할 수 없음")
        return report_style
    opts = dict(report_options or {})
    if isinstance(report_style, dict):
        opts = {**report_style, **opts}
        report_style = opts.pop("style", "social")
    elif "style" in opts:
        report_style = opts.pop("style") if report_style is None else report_style
        opts.pop("style", None)
    style = resolve_choice("보고 양식(report_style)", report_style, REPORT_STYLES, _STYLE_ALIASES, default="default")
    if style == "default":
        if opts:
            raise ValueError("report_options: report_style='social'일 때만 지정 가능")
        return None
    unknown = [k for k in opts if k not in OPTION_KEYS]
    if unknown:
        raise ValueError(f"report_options: 알 수 없는 항목 {', '.join(unknown)} (가능: {', '.join(OPTION_KEYS)})")
    fmt = ReportFormat()
    fmt.stars, fmt.star_levels = resolve_stars(opts.get("stars", fmt.stars))
    fmt.star_target = resolve_choice("별표 대상(star_target)", opts.get("star_target", fmt.star_target), STAR_TARGETS, _TARGET_ALIASES)
    fmt.se_position = resolve_choice("표준오차 위치(se_position)", opts.get("se_position", fmt.se_position), SE_POSITIONS, _SE_ALIASES)
    fmt.lang = resolve_choice("언어(lang)", opts.get("lang", fmt.lang), LANGS, {"korean": "ko", "한국어": "ko", "english": "en", "영어": "en"})
    for k in ("keep_p", "keep_stat", "keep_ci", "leading_zero", "star_note", "table_numbers", "raw_sheet"):
        if k in opts:
            setattr(fmt, k, _as_bool(k, opts[k]))
    if "decimals" in opts:
        try:
            d = int(opts["decimals"])
        except (TypeError, ValueError):
            raise ValueError(f"decimals: 0~6 정수 필요 (입력값 {opts['decimals']!r})") from None
        if not (0 <= d <= 6) or (isinstance(opts["decimals"], float) and not float(opts["decimals"]).is_integer()):
            raise ValueError(f"decimals: 0~6 정수 필요 (입력값 {opts['decimals']!r})")
        fmt.decimals = d
    return fmt


# ---------------------------------------------------------------------------------------- numbers
def _num(v: Any) -> Optional[float]:
    if v is None or v == "" or isinstance(v, bool):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return None


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (float, np.floating)):
        return math.isnan(float(v)) or math.isinf(float(v))
    return False


def _strip_zero(s: str) -> str:
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def _fixed(f: float, d: int) -> str:
    if f != 0 and abs(f) < 1e-4 and abs(f) < 0.5 * 10 ** (-d):
        return f"{f:.{max(1, d - 1)}e}"
    s = f"{f:,.{d}f}" if abs(f) >= 1e5 else f"{f:.{d}f}"
    if s.lstrip("-").strip("0.,") == "" and s.startswith("-"):
        s = s[1:]
    return s


def format_number(v: Any, meta: Dict[str, Any], fmt: ReportFormat, bounded: bool = False) -> str:
    """Cell text of a numeric value under the report format (strings pass through)."""
    if v == "" or v is None:
        return ""
    if _is_missing(v):
        return "—"
    f = _num(v)
    if f is None:
        return str(v)
    ctype, cfmt = meta.get("type", "number"), meta.get("format", "")
    if ctype == "integer" and abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    if cfmt == "pvalue":
        if f < 0.001:
            return "< 0.001" if fmt.leading_zero else "< .001"
        s = f"{f:.3f}"
        return s if fmt.leading_zero else _strip_zero(s)
    if cfmt == "pc":
        return f"{f:.1f}%"
    if cfmt == "auto" and abs(f - round(f)) < 1e-9 and abs(f) < 1e15:
        return f"{int(round(f)):,}" if abs(f) >= 1e4 else str(int(round(f)))
    s = _fixed(f, fmt.decimals)
    if (bounded or cfmt == "zto") and not fmt.leading_zero:
        s = _strip_zero(s)
    return s


def stars_for(p: Any, fmt: ReportFormat) -> str:
    f = _num(p)
    if f is None:
        return ""
    sym = ""
    for thr, s in fmt.star_levels:          # largest threshold first; keep the most stringent passed
        if f < thr:
            sym = s
    return sym


def star_legend(fmt: ReportFormat) -> str:
    if not fmt.star_levels:
        return ""

    def thr_txt(t: float) -> str:
        s = f"{t:.3f}".rstrip("0")
        if s.endswith("."):
            s += "0"
        if len(s.split(".")[1]) < 2:
            s = f"{t:.2f}"
        return s if fmt.leading_zero else _strip_zero(s)

    if fmt.lang == "en":
        return " ".join(f"{s}p < {thr_txt(t)}." for t, s in fmt.star_levels)
    return ", ".join(f"{s} p<{thr_txt(t)}" for t, s in fmt.star_levels)


def se_note(fmt: ReportFormat) -> str:
    return "Standard errors in parentheses." if fmt.lang == "en" else "괄호 안은 표준오차임."


def caption_text(title: str, number: Optional[int], fmt: Optional[ReportFormat]) -> str:
    if fmt is None or not fmt.table_numbers or number is None:
        return title
    return f"Table {number}. {title}" if fmt.lang == "en" else f"<표 {number}> {title}"


# ---------------------------------------------------------------------------------------- rendering
@dataclass
class HeaderCell:
    text: str
    col: int
    colspan: int = 1
    rowspan: int = 1


@dataclass
class RenderedTable:
    name: str
    title: str
    caption: str
    header_rows: List[List[HeaderCell]]
    flat_headers: List[str]
    body: List[List[str]]
    col_types: List[str]
    aligns: List[str]
    rules_above: List[int]
    notes: List[str]
    is_coefficients: bool = False

    @property
    def n_cols(self) -> int:
        return len(self.flat_headers)


def _header_rows(titles: List[str], supers: List[str]) -> List[List[HeaderCell]]:
    if not any(supers):
        return [[HeaderCell(t, i) for i, t in enumerate(titles)]]
    top, sub = [], []
    i = 0
    while i < len(titles):
        st = supers[i]
        if not st:
            top.append(HeaderCell(titles[i], i, 1, 2))
            i += 1
            continue
        j = i
        while j + 1 < len(titles) and supers[j + 1] == st:
            j += 1
        top.append(HeaderCell(st, i, j - i + 1, 1))
        for k in range(i, j + 1):
            sub.append(HeaderCell(titles[k], k))
        i = j + 1
    return [top, sub]


def _section_rules(rows: Sequence[Dict[str, Any]], row_starts: Sequence[int]) -> List[int]:
    rules, prev = [], None
    for k, r in enumerate(rows):
        sec = r.get("_section", "") if isinstance(r, dict) else ""
        if k > 0 and sec != prev:
            rules.append(row_starts[k])
        prev = sec
    return rules


def _plain_cell(v: Any, meta: Dict[str, Any], ctype: str, fmt: ReportFormat) -> str:
    if ctype == "number":
        return format_number(v, meta, fmt)
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if _is_missing(v):
        return "—"
    if _num(v) is not None:
        return format_number(v, meta, fmt)
    return str(v)


def render_table(tbl: ResultsTable, fmt: Optional[ReportFormat] = None, number: Optional[int] = None) -> RenderedTable:
    """Header rows, formatted body and notes of one result table under the default or social format."""
    if fmt is None:
        titles = [c.get("title") or c.get("name") or "" for c in tbl.columns]
        supers = [c.get("superTitle") or "" for c in tbl.columns]
        body = [[tbl._format_cell(r.get(c["name"]), tbl.cell_meta(r, c)) for c in tbl.columns] for r in tbl.rows]
        types = ["number" if c.get("type") in ("number", "integer") else "text" for c in tbl.columns]
        return RenderedTable(
            name=tbl.name, title=tbl.title, caption=tbl.title,
            header_rows=_header_rows(titles, supers),
            flat_headers=[tbl.display_title(c) for c in tbl.columns],
            body=body, col_types=types,
            aligns=["right" if t == "number" else "left" for t in types],
            rules_above=_section_rules(tbl.rows, list(range(len(tbl.rows)))),
            notes=list(tbl.notes), is_coefficients=tbl.kind == "coefficients",
        )

    groups = tbl.coef_groups if tbl.kind == "coefficients" else []
    role_of: Dict[str, Tuple[int, str]] = {}
    for gi, g in enumerate(groups):
        for role, col in g.items():
            role_of[col] = (gi, role)
    by_name = {c["name"]: c for c in tbl.columns}

    out: List[Dict[str, Any]] = []
    for c in tbl.columns:
        name = c["name"]
        title = c.get("title") or name
        base = {"src": name, "meta": c, "title": title, "super": c.get("superTitle") or "",
                "type": "number" if c.get("type") in ("number", "integer") else "text"}
        if name not in role_of:
            out.append({**base, "kind": "plain"})
            continue
        gi, role = role_of[name]
        g = groups[gi]
        if role == "estimate":
            se_title = (by_name[g["se"]].get("title") or "SE") if g.get("se") else ""
            t = f"{title} ({se_title})" if (fmt.se_position == "beside" and se_title) else title
            out.append({**base, "kind": "est", "group": gi, "title": t})
        elif role == "se":
            if fmt.se_position == "column":
                out.append({**base, "kind": "plain"})
        elif role == "p":
            if fmt.keep_p or not fmt.star_levels:
                out.append({**base, "kind": "plain"})
        elif role == "std_estimate":
            out.append({**base, "kind": "std", "group": gi})
        elif role == "stat":
            if fmt.keep_stat:
                out.append({**base, "kind": "plain"})
        elif role in ("ci_lower", "ci_upper"):
            if fmt.keep_ci:
                out.append({**base, "kind": "plain"})
        else:
            out.append({**base, "kind": "plain"})

    def star_cell(gi: int, r: Dict[str, Any]) -> str:
        """Which cell of group gi carries the stars for row r: 'est', 'std' or ''."""
        g = groups[gi]
        if fmt.star_target == "std_estimate" and g.get("std_estimate"):
            return "std"
        return "est"

    body: List[List[str]] = []
    row_starts: List[int] = []
    se_shown = False
    for r in tbl.rows:
        row_starts.append(len(body))
        cells, below, has_below = [], [], False
        for oc in out:
            meta = tbl.cell_meta(r, oc["meta"])
            v = r.get(oc["src"])
            if oc["kind"] == "est":
                g = groups[oc["group"]]
                txt = format_number(v, meta, fmt)
                if txt and txt != "—" and star_cell(oc["group"], r) == "est":
                    txt += stars_for(r.get(g["p"]), fmt)
                se_v = r.get(g["se"]) if g.get("se") else None
                se_txt = format_number(se_v, by_name[g["se"]], fmt) if _num(se_v) is not None else ""
                if not txt or txt == "—":
                    se_txt = ""
                if se_txt and fmt.se_position == "beside":
                    txt = f"{txt} ({se_txt})"
                    se_shown = True
                below.append(f"({se_txt})" if se_txt and fmt.se_position == "below" else "")
                has_below = has_below or bool(se_txt and fmt.se_position == "below")
                cells.append(txt)
            elif oc["kind"] == "std":
                g = groups[oc["group"]]
                txt = format_number(v, meta, fmt, bounded=True)
                if txt and txt != "—" and star_cell(oc["group"], r) == "std":
                    txt += stars_for(r.get(g["p"]), fmt)
                cells.append(txt)
                below.append("")
            else:
                cells.append(_plain_cell(v, meta, oc["type"], fmt))
                below.append("")
        body.append(cells)
        if fmt.se_position == "below" and has_below:
            body.append(below)
            se_shown = True

    notes = list(tbl.notes)
    if groups and fmt.star_note:
        if se_shown:
            notes.append(se_note(fmt))
        legend = star_legend(fmt)
        if legend:
            notes.append(legend)

    titles = [oc["title"] for oc in out]
    supers = [oc["super"] for oc in out]
    flat = [f"{s} {t}".strip() if s else t for s, t in zip(supers, titles)]
    types = [oc["type"] for oc in out]
    return RenderedTable(
        name=tbl.name, title=tbl.title, caption=caption_text(tbl.title, number, fmt),
        header_rows=_header_rows(titles, supers), flat_headers=flat, body=body, col_types=types,
        aligns=["center" if t == "number" else "left" for t in types],
        rules_above=_section_rules(tbl.rows, row_starts),
        notes=notes, is_coefficients=bool(groups),
    )


def raw_matrix(tbl: ResultsTable) -> Tuple[List[str], List[List[Any]]]:
    """Unformatted values of a table (numbers stay numbers) for spreadsheet raw-data sheets."""
    headers = [tbl.display_title(c) for c in tbl.columns]
    rows = []
    for r in tbl.rows:
        vals = []
        for c in tbl.columns:
            v = r.get(c["name"])
            if v == "" or _is_missing(v):
                vals.append(None)
            elif isinstance(v, (np.integer,)):
                vals.append(int(v))
            elif isinstance(v, (np.floating, float)):
                vals.append(float(v))
            else:
                vals.append(v)
        rows.append(vals)
    return headers, rows


def table_numbering(analyses: Sequence[Any]) -> Dict[int, int]:
    """id(table) -> running table number across all analyses (tables with columns only)."""
    out, n = {}, 0
    for a in analyses:
        for t in getattr(a, "tables", []):
            if t.columns:
                n += 1
                out[id(t)] = n
    return out

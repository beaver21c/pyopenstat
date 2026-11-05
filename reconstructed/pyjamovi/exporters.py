"""
OpenStat Studio — Multi-Format Publication Results, Table & Graph Exporter
License: MIT
Exports full Analysis Results (statistical APA tables, narratives, AND standalone/analysis graphs) into 11 researcher-ready formats:
- .pdf  (Multi-page Publication PDF Report with APA 3-line tables, interpretations & high-res figures)
- .hwpx (Hancom Office Hangul OWPML package with editable native Hangul tables, merged header cells & embedded BinData/*.png `<hp:pic>` graphs)
- .docx (MS Word Document with genuine OOXML APA 3-line tables, headings, notes & embedded charts)
- .xlsx (Excel Workbook with APA 3-line borders, formatted cells, notes & embedded PNG charts)
- .tex  (LaTeX booktabs academic manuscript tables + companion vector .pdf figures & `\\includegraphics`)
- .html (Standalone interactive/printable APA HTML report with inline plots)
- .md   (GitHub / Academic Markdown + companion .png figures)
- .csv  (UTF-8-SIG analyzed tables CSV for Excel / R / Python)
- .tsv  (Tab-separated tables for direct paste)
- .json (Structured machine-readable JSON payload including base64 plots)
- .omv  (jamovi-compatible binary container)
"""

import base64
import csv
import io
import json
import os
import re
import struct
import textwrap
import warnings
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .analyses.base import BaseAnalysis, ResultsTable
from .renderers.html_renderer import render_jamovi_html_report
from .omv.writer import write_omv
from .report_format import (ReportFormat, RenderedTable, render_table, raw_matrix, resolve_report_format,
                            table_numbering)


SUPPORTED_TABLE_FORMATS = ("xlsx", "docx", "hwpx", "pdf", "tex", "csv", "tsv", "md", "html", "json", "omv")


def _rendered_tables(analysis: BaseAnalysis, rf: Optional[ReportFormat], numbers: Optional[Dict[int, int]] = None) -> List[RenderedTable]:
    out = []
    for tbl in analysis.tables:
        if not tbl.columns:
            continue
        out.append(render_table(tbl, rf, (numbers or {}).get(id(tbl))))
    return out


def _formatted_matrix(tbl: ResultsTable, rf: Optional[ReportFormat] = None):
    rt = render_table(tbl, rf)
    col_types = ["number" if t == "number" else "text" for t in rt.col_types]
    return rt.flat_headers, rt.body, col_types


def _note_lines(rt: RenderedTable, rf: Optional[ReportFormat], default_prefix: str) -> List[str]:
    """Note paragraphs: default mode prefixes every note; social mode prefixes the first only."""
    if not rt.notes:
        return []
    if rf is None:
        return [f"{default_prefix}{n}" for n in rt.notes]
    head = "주: " if rf.lang == "ko" else "Note. "
    return [head + rt.notes[0]] + [("    " if rf.lang == "ko" else "") + n for n in rt.notes[1:]]


def _decode_plot_png(plot_dict: dict) -> Optional[bytes]:
    src = (plot_dict or {}).get("src", "")
    if not src:
        return None
    raw_b64 = src.split(",", 1)[1] if "," in src else src
    try:
        return base64.b64decode(raw_b64)
    except Exception:
        return None


def _png_dimensions(png_bytes: bytes) -> Tuple[int, int]:
    if png_bytes and len(png_bytes) >= 24 and png_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        try:
            w, h = struct.unpack(">II", png_bytes[16:24])
            if w > 0 and h > 0:
                return int(w), int(h)
        except Exception:
            pass
    return 930, 600


def export_to_xlsx(analyses: List[BaseAnalysis], output_path: str, report_format: Optional[ReportFormat] = None) -> str:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.utils import get_column_letter

    rf = report_format
    wb = Workbook()
    wb.remove(wb.active)

    thin_dark = Side(border_style="thin", color="334155")
    medium_dark = Side(border_style="medium", color="0F172A")
    hdr_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    numbers = table_numbering(analyses) if rf else {}
    h_align = {"left": "left", "right": "right", "center": "center"}

    used_titles = set()
    for idx, analysis in enumerate(analyses, start=1):
        raw_sheet = re.sub(r"[\[\]\:\*\?\/\\]", "", analysis.title)[:26].strip() or f"Analysis_{idx}"
        sheet_name = raw_sheet
        suffix = 2
        while sheet_name in used_titles:
            sheet_name = f"{raw_sheet[:23]}_{suffix}"
            suffix += 1
        used_titles.add(sheet_name)

        ws = wb.create_sheet(title=sheet_name)
        row_cursor = 1

        ws.cell(row=row_cursor, column=1, value=analysis.title).font = Font(name="Calibri", size=14, bold=True, color="1E3A5F")
        row_cursor += 2

        for rt in _rendered_tables(analysis, rf, numbers):
            ws.cell(row=row_cursor, column=1, value=rt.caption).font = Font(name="Calibri", size=11, bold=True, color="1E293B")
            row_cursor += 1
            if not rt.flat_headers:
                continue
            n_head = len(rt.header_rows)
            for ri, hrow in enumerate(rt.header_rows):
                for hc in hrow:
                    cell = ws.cell(row=row_cursor + ri, column=hc.col + 1, value=hc.text)
                    cell.font = Font(name="Calibri", size=10, bold=True, color="0F172A")
                    cell.fill = hdr_fill
                    al = "center" if (rf or hc.colspan > 1) else h_align[rt.aligns[hc.col]]
                    cell.alignment = Alignment(horizontal=al, vertical="center")
                    if hc.colspan > 1 or hc.rowspan > 1:
                        ws.merge_cells(start_row=row_cursor + ri, start_column=hc.col + 1,
                                       end_row=row_cursor + ri + hc.rowspan - 1, end_column=hc.col + hc.colspan)
            for c_idx in range(1, rt.n_cols + 1):
                ws.cell(row=row_cursor, column=c_idx).border = Border(top=medium_dark, bottom=thin_dark if n_head == 1 else None)
                ws.cell(row=row_cursor + n_head - 1, column=c_idx).border = Border(
                    top=medium_dark if n_head == 1 else None, bottom=thin_dark)
            if n_head == 2:
                for hc in rt.header_rows[0]:
                    if hc.rowspan == 1:
                        for k in range(hc.col, hc.col + hc.colspan):
                            ws.cell(row=row_cursor, column=k + 1).border = Border(top=medium_dark, bottom=thin_dark)
            row_cursor += n_head

            rules = set(rt.rules_above)
            for r_i, r_vals in enumerate(rt.body):
                is_last = (r_i == len(rt.body) - 1)
                for c_idx, val in enumerate(r_vals, start=1):
                    cell = ws.cell(row=row_cursor, column=c_idx, value=val)
                    cell.font = Font(name="Calibri", size=10, color="1E293B")
                    cell.alignment = Alignment(horizontal=h_align[rt.aligns[c_idx - 1]], vertical="center")
                    if is_last or r_i in rules:
                        cell.border = Border(top=thin_dark if r_i in rules else None, bottom=medium_dark if is_last else None)
                row_cursor += 1

            for line in _note_lines(rt, rf, "Note. "):
                ncell = ws.cell(row_cursor, 1, value=line)
                ncell.font = Font(name="Calibri", size=9, italic=rf is None, color="475569")
                row_cursor += 1

            row_cursor += 1

        if analysis.apa_narrative:
            acell = ws.cell(row=row_cursor, column=1, value=f"APA Summary: {analysis.apa_narrative}")
            acell.font = Font(name="Calibri", size=10, italic=True, color="0F172A")
            row_cursor += 2

        # Auto-fit column widths
        for col_idx in range(1, ws.max_column + 1):
            col_letter = get_column_letter(col_idx)
            max_len = 10
            for r_idx in range(2, row_cursor):
                v = ws.cell(row=r_idx, column=col_idx).value
                if v and len(str(v)) < 48:
                    max_len = max(max_len, len(str(v)) + 3)
            ws.column_dimensions[col_letter].width = min(38, max_len)

        # Embed high-res plots below tables
        for p_idx, p in enumerate(analysis.plots, start=1):
            img_bytes = _decode_plot_png(p)
            if img_bytes:
                p_title = p.get("title") or f"Figure {p_idx}"
                pcell = ws.cell(row=row_cursor, column=1, value=f"[Figure {p_idx}] {p_title}")
                pcell.font = Font(name="Calibri", size=10, bold=True, color="0D9488")
                row_cursor += 1
                xl_img = XLImage(io.BytesIO(img_bytes))
                px_w, px_h = _png_dimensions(img_bytes)
                xl_img.width = 480
                xl_img.height = max(220, int(480 * (px_h / max(1, px_w))))
                ws.add_image(xl_img, f"A{row_cursor}")
                row_cursor += max(15, int(xl_img.height / 18) + 2)

    if rf is not None and rf.raw_sheet and analyses:
        # Stars and parentheses turn numbers into text: keep the unformatted values on their own sheet.
        ws = wb.create_sheet(title="원자료(숫자)" if rf.lang == "ko" else "Raw values")
        r = 1
        ws.cell(row=r, column=1, value="보고 양식 적용 전 수치(계산용)" if rf.lang == "ko" else "Unformatted numeric values").font = Font(bold=True, size=12)
        r += 2
        for analysis in analyses:
            for tbl in analysis.tables:
                if not tbl.columns:
                    continue
                cap = f"{analysis.title} — {tbl.title}"
                n = numbers.get(id(tbl))
                ws.cell(row=r, column=1, value=(f"<표 {n}> " if (n and rf.lang == "ko") else (f"Table {n}. " if n else "")) + cap).font = Font(bold=True)
                r += 1
                headers, rows = raw_matrix(tbl)
                for c, h in enumerate(headers, start=1):
                    ws.cell(row=r, column=c, value=h).font = Font(bold=True)
                r += 1
                for vals in rows:
                    for c, v in enumerate(vals, start=1):
                        ws.cell(row=r, column=c, value=v)
                    r += 1
                r += 1

    if not wb.sheetnames:
        ws = wb.create_sheet("Summary")
        ws["A1"] = "No analyses in session."

    wb.save(output_path)
    return output_path


def _set_docx_borders(table, rt: RenderedTable):
    """APA three-line borders: thick top/bottom, thin under the header (and under spanning heads), thin section rules."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge, val, sz in [
        ("top", "single", "12"),
        ("bottom", "single", "12"),
        ("left", "none", "0"),
        ("right", "none", "0"),
        ("insideH", "none", "0"),
        ("insideV", "none", "0"),
    ]:
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), sz)
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "1E293B")
        borders.append(el)
    tblPr.append(borders)

    def _edge(cell, edge, sz="6"):
        tcPr = cell._tc.get_or_add_tcPr()
        tcBorders = tcPr.find(qn("w:tcBorders"))
        if tcBorders is None:
            tcBorders = OxmlElement("w:tcBorders")
            tcPr.append(tcBorders)
        b = OxmlElement(f"w:{edge}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), sz)
        b.set(qn("w:space"), "0")
        b.set(qn("w:color"), "334155")
        tcBorders.append(b)

    n_head = len(rt.header_rows)
    if len(table.rows) >= n_head:
        for cell in table.rows[n_head - 1].cells:
            _edge(cell, "bottom")
        if n_head == 2:
            for hc in rt.header_rows[0]:
                # spanning heads get their own rule; vertically merged heads carry the header rule on the first cell
                _edge(table.rows[0].cells[hc.col], "bottom")
    for bi in rt.rules_above:
        ri = n_head + bi
        if ri < len(table.rows):
            for cell in table.rows[ri].cells:
                _edge(cell, "top")


def export_to_docx(analyses: List[BaseAnalysis], output_path: str, report_format: Optional[ReportFormat] = None) -> str:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_ALIGN_VERTICAL

    rf = report_format

    def _tight(cell, para):
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        pf = para.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.line_spacing = Pt(13)                    # exact: equal row heights for Latin, Hangul and empty cells
    numbers = table_numbering(analyses) if rf else {}
    align_map = {"left": WD_ALIGN_PARAGRAPH.LEFT, "right": WD_ALIGN_PARAGRAPH.RIGHT, "center": WD_ALIGN_PARAGRAPH.CENTER}

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.size = Pt(9.5)                      # also sizes empty table cells (paragraph marks)
    normal.paragraph_format.space_after = Pt(2)
    try:
        from docx.oxml.ns import qn
        normal.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "맑은 고딕")
    except Exception:
        pass
    title_h = doc.add_heading("OpenStat Studio — Statistical Analysis & Graph Report", level=1)
    title_h.runs[0].font.color.rgb = RGBColor(0x1E, 0x3A, 0x5F)

    fig_num = 0
    for analysis in analyses:
        h2 = doc.add_heading(analysis.title, level=2)
        if h2.runs:
            h2.runs[0].font.color.rgb = RGBColor(0x0D, 0x94, 0x88)

        for rt in _rendered_tables(analysis, rf, numbers):
            p_title = doc.add_paragraph()
            run_t = p_title.add_run(rt.caption)
            run_t.bold = True
            run_t.font.size = Pt(10.5)
            if not rt.flat_headers:
                continue

            n_head = len(rt.header_rows)
            w_tbl = doc.add_table(rows=n_head + len(rt.body), cols=rt.n_cols)
            for ri, hrow in enumerate(rt.header_rows):
                for hc in hrow:
                    cell = w_tbl.rows[ri].cells[hc.col]
                    if hc.colspan > 1 or hc.rowspan > 1:
                        cell = cell.merge(w_tbl.rows[ri + hc.rowspan - 1].cells[hc.col + hc.colspan - 1])
                    cell.text = hc.text
                    p = cell.paragraphs[0]
                    _tight(cell, p)
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (rf or hc.colspan > 1) else align_map[rt.aligns[hc.col]]
                    for r in p.runs:
                        r.bold = True
                        r.font.size = Pt(9.5)

            for r_i, r_vals in enumerate(rt.body):
                for c_i, val in enumerate(r_vals):
                    cell = w_tbl.rows[n_head + r_i].cells[c_i]
                    cell.text = str(val)
                    p = cell.paragraphs[0]
                    _tight(cell, p)
                    p.alignment = align_map[rt.aligns[c_i]]
                    for r in p.runs:
                        r.font.size = Pt(9.5)

            _set_docx_borders(w_tbl, rt)

            for i, line in enumerate(_note_lines(rt, rf, "")):
                pn = doc.add_paragraph()
                if rf is None:
                    r1 = pn.add_run("Note. ")
                    r1.italic = True
                    r1.font.size = Pt(8.5)
                r2 = pn.add_run(line)
                r2.font.size = Pt(8.5)

        if analysis.apa_narrative:
            pa = doc.add_paragraph()
            ra_lbl = pa.add_run("APA Interpretation: ")
            ra_lbl.bold = True
            ra_lbl.font.size = Pt(10)
            ra_txt = pa.add_run(analysis.apa_narrative)
            ra_txt.font.size = Pt(10)

        for p in analysis.plots:
            img_bytes = _decode_plot_png(p)
            if img_bytes:
                fig_num += 1
                p_cap = doc.add_paragraph()
                cap = (f"[그림 {fig_num}] " if (rf and rf.lang == "ko") else f"Figure {fig_num}. ") + p.get("title", "Statistical Visualization")
                rc = p_cap.add_run(cap)
                rc.bold = True
                rc.font.size = Pt(9.5)
                doc.add_picture(io.BytesIO(img_bytes), width=Inches(5.5))

    doc.save(output_path)
    return output_path


def export_to_hwpx(analyses: List[BaseAnalysis], output_path: str, report_format: Optional[ReportFormat] = None) -> str:
    """
    Writes a Hancom Office Hangul (.hwpx) report: editable native tables (merged header cells,
    numbered captions under the social format), notes, APA interpretations and embedded figures.
    """
    from .hwpx_writer import HwpxBuilder

    rf = report_format
    numbers = table_numbering(analyses) if rf else {}
    doc = HwpxBuilder(title="OpenStat Studio — 통계 분석 및 시각화 결과 보고서")
    pic_seq = 0
    for analysis in analyses:
        doc.paragraph(f"■ {analysis.title}", kind="heading")
        for rt in _rendered_tables(analysis, rf, numbers):
            doc.paragraph(rt.caption if rf else f"[{rt.caption}]", kind="caption")
            if not rt.flat_headers:
                continue
            doc.table(rt, social=rf is not None)
            for line in _note_lines(rt, rf, "주: "):
                doc.paragraph(line, kind="note")

        if analysis.apa_narrative:
            doc.paragraph(f"해석(APA): {analysis.apa_narrative}")

        for p in analysis.plots:
            png_bytes = _decode_plot_png(p)
            if not png_bytes:
                continue
            pic_seq += 1
            doc.paragraph(f"[그림 {pic_seq}] {p.get('title') or f'시각화 그래프 {pic_seq}'}", kind="caption")
            doc.picture(png_bytes)

    return doc.save(output_path)


def export_to_pdf(analyses: List[BaseAnalysis], output_path: str, report_format: Optional[ReportFormat] = None) -> str:
    """
    Renders a multi-page publication PDF report containing:
    - Report title header
    - APA 3-line statistical tables & footnotes (grouped column heads, numbered captions under the social format)
    - APA interpretations
    - All statistical & standalone visualizations embedded at crisp publication quality
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.backends.backend_pdf import PdfPages
    from .analyses.plot_style import DEFAULT_FONT_STACK

    rf = report_format
    numbers = table_numbering(analyses) if rf else {}

    def _x(c_i, n_cols, x0, col_w, align):
        if align == "right":
            return x0 + (c_i + 0.94) * col_w, "right"
        if align == "center":
            return x0 + (c_i + 0.5) * col_w, "center"
        return x0 + (c_i + 0.04) * col_w, "left"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Glyph .* missing from font")
        with plt.rc_context({"font.family": DEFAULT_FONT_STACK, "font.sans-serif": DEFAULT_FONT_STACK}):
            with PdfPages(output_path) as pdf:
                if not analyses:
                    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
                    fig.text(0.08, 0.92, "OpenStat Studio — Statistical Analysis Report", fontsize=15, fontweight="bold", color="#1E3A5F")
                    fig.text(0.08, 0.87, "No analyses or graphs in current session.", fontsize=10.5, color="#475569")
                    pdf.savefig(fig)
                    plt.close(fig)
                    return output_path

                fig_counter = 0
                for a_idx, analysis in enumerate(analyses, start=1):
                    if analysis.tables or analysis.apa_narrative or not analysis.plots:
                        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
                        ax = fig.add_axes([0, 0, 1, 1])
                        ax.axis("off")
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.autoscale(False)

                        y = 0.94
                        if a_idx == 1:
                            ax.text(0.07, y, "OpenStat Studio — Analysis & Visualization Results", fontsize=14, fontweight="bold", color="#1E3A5F")
                            ax.plot([0.07, 0.93], [y - 0.014, y - 0.014], color="#0D9488", lw=1.6)
                            y -= 0.045

                        ax.text(0.07, y, f"{a_idx}. {analysis.title}", fontsize=12, fontweight="bold", color="#0F172A")
                        y -= 0.032

                        for rt in _rendered_tables(analysis, rf, numbers):
                            if not rt.flat_headers:
                                continue
                            display_rows = rt.body[:28]
                            notes = _note_lines(rt, rf, "Note. ")
                            n_head = len(rt.header_rows)
                            needed_h = 0.035 + 0.022 * n_head + 0.020 * len(display_rows) + 0.018 * len(notes)
                            if y - needed_h < 0.08:
                                pdf.savefig(fig)
                                plt.close(fig)
                                fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
                                ax = fig.add_axes([0, 0, 1, 1])
                                ax.axis("off")
                                ax.set_xlim(0, 1)
                                ax.set_ylim(0, 1)
                                ax.autoscale(False)
                                y = 0.94

                            ax.text(0.07, y, rt.caption, fontsize=10, fontweight="bold", color="#1E293B")
                            y -= 0.016

                            n_cols = rt.n_cols
                            x0, x1 = 0.07, 0.93
                            col_w = (x1 - x0) / n_cols

                            # APA Top Rule
                            ax.plot([x0, x1], [y, y], color="#0F172A", lw=1.2)
                            y -= 0.020
                            if n_head == 2:
                                for hc in rt.header_rows[0]:
                                    if hc.rowspan == 1:
                                        cx = x0 + (hc.col + hc.colspan / 2) * col_w
                                        ax.text(cx, y + 0.004, str(hc.text)[:40], fontsize=8.5, fontweight="bold", color="#0F172A", ha="center", va="bottom")
                                        ax.plot([x0 + (hc.col + 0.04) * col_w, x0 + (hc.col + hc.colspan - 0.04) * col_w],
                                                [y - 0.003, y - 0.003], color="#334155", lw=0.6)
                                y -= 0.020
                            sub = {hc.col: hc.text for hc in rt.header_rows[-1]}
                            for hc in rt.header_rows[0]:
                                if hc.rowspan > 1 or n_head == 1:
                                    sub[hc.col] = hc.text
                            for c_i in range(n_cols):
                                al = "center" if rf else rt.aligns[c_i]
                                tx, ha = _x(c_i, n_cols, x0, col_w, al)
                                ax.text(tx, y + 0.004, str(sub.get(c_i, ""))[:24], fontsize=8.5, fontweight="bold", color="#0F172A", ha=ha, va="bottom")
                            # APA Header Rule
                            ax.plot([x0, x1], [y - 0.003, y - 0.003], color="#334155", lw=0.7)
                            y -= 0.022

                            rules = set(rt.rules_above)
                            for r_i, r_vals in enumerate(display_rows):
                                if r_i in rules:
                                    ax.plot([x0, x1], [y + 0.016, y + 0.016], color="#334155", lw=0.6)
                                for c_i, val in enumerate(r_vals):
                                    tx, ha = _x(c_i, n_cols, x0, col_w, rt.aligns[c_i])
                                    ax.text(tx, y + 0.004, str(val)[:26], fontsize=8.3, color="#1E293B", ha=ha, va="bottom")
                                y -= 0.020

                            # APA Bottom Rule
                            ax.plot([x0, x1], [y + 0.015, y + 0.015], color="#0F172A", lw=1.2)
                            y -= 0.006

                            for note in notes:
                                ax.text(0.07, y, note[:110], fontsize=7.8, style="italic" if rf is None else "normal", color="#475569")
                                y -= 0.018
                            y -= 0.016

                        if analysis.apa_narrative:
                            wrapped = textwrap.wrap(f"APA Interpretation: {analysis.apa_narrative}", width=92)
                            if y - 0.022 * len(wrapped) < 0.08:
                                pdf.savefig(fig)
                                plt.close(fig)
                                fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
                                ax = fig.add_axes([0, 0, 1, 1])
                                ax.axis("off")
                                ax.set_xlim(0, 1)
                                ax.set_ylim(0, 1)
                                ax.autoscale(False)
                                y = 0.94
                            for line in wrapped:
                                ax.text(0.07, y, line, fontsize=9.0, color="#0F172A")
                                y -= 0.020

                        pdf.savefig(fig)
                        plt.close(fig)

                    # Dedicated crisp visualization panel(s) for all plots on this analysis
                    for p in analysis.plots:
                        png_bytes = _decode_plot_png(p)
                        if not png_bytes:
                            continue
                        fig_counter += 1
                        img_arr = mpimg.imread(io.BytesIO(png_bytes), format="png")
                        fig_p = plt.figure(figsize=(8.27, 11.69), facecolor="white")
                        fig_p.text(0.07, 0.93, f"{a_idx}. {analysis.title}", fontsize=11.5, fontweight="bold", color="#1E3A5F")
                        p_title = p.get("title") or f"Figure {fig_counter}"
                        cap = f"[그림 {fig_counter}] {p_title}" if (rf and rf.lang == "ko") else f"Figure {fig_counter}. {p_title}"
                        fig_p.text(0.07, 0.90, cap, fontsize=10.5, fontweight="bold", color="#0D9488")

                        px_w, px_h = _png_dimensions(png_bytes)
                        aspect = px_h / max(1, px_w)
                        box_w = 0.86
                        box_h = min(0.72, box_w * aspect * (8.27 / 11.69))
                        ax_img = fig_p.add_axes([0.07, 0.87 - box_h, box_w, box_h])
                        ax_img.imshow(img_arr)
                        ax_img.axis("off")

                        if analysis.apa_narrative and not analysis.tables:
                            wrapped_cap = textwrap.wrap(analysis.apa_narrative, width=92)
                            cy = 0.84 - box_h
                            for line in wrapped_cap[:5]:
                                fig_p.text(0.07, cy, line, fontsize=9.0, color="#334155")
                                cy -= 0.020

                        pdf.savefig(fig_p)
                        plt.close(fig_p)

    return output_path


def _tex_escape(s: str) -> str:
    rep = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "χ²": r"$\chi^2$",
        "η²p": r"$\eta^2_p$",
        "η²G": r"$\eta^2_G$",
        "η²": r"$\eta^2$",
        "ω²": r"$\omega^2$",
        "ε²": r"$\varepsilon^2$",
        "ε": r"$\varepsilon$",
        "α": r"$\alpha$",
        "ω": r"$\omega$",
        "β": r"$\beta$",
        "†": r"$\dagger$",
        "< .001": r"$< .001$",
    }
    out = str(s)
    for k in ("χ²", "η²p", "η²G", "η²", "ω²", "ε²", "ε", "α", "ω", "β", "†"):
        out = out.replace(k, rep[k])
    out = out.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")
    return out


def export_to_latex(analyses: List[BaseAnalysis], output_path: str, report_format: Optional[ReportFormat] = None) -> str:
    rf = report_format
    lines = [
        r"% OpenStat Studio — Publication LaTeX Tables & Figures (Requires \usepackage{booktabs, graphicx})",
        ""
    ]
    out_dir = os.path.dirname(os.path.abspath(output_path))
    stem = os.path.splitext(os.path.basename(output_path))[0]
    fig_idx = 0

    for analysis in analyses:
        lines.append(f"% === {analysis.title} ===")
        for rt in _rendered_tables(analysis, rf):
            if not rt.flat_headers:
                continue
            align = "".join({"right": "r", "center": "c"}.get(a, "l") for a in rt.aligns)
            lines.append(r"\begin{table}[htbp]")
            lines.append(r"\centering")
            lines.append(f"\\caption{{{_tex_escape(rt.title)}}}")
            lines.append(f"\\begin{{tabular}}{{{align}}}")
            lines.append(r"\toprule")
            if len(rt.header_rows) == 2:
                top, rules = [], []
                for hc in rt.header_rows[0]:
                    if hc.rowspan == 1:
                        top.append(f"\\multicolumn{{{hc.colspan}}}{{c}}{{{_tex_escape(hc.text)}}}")
                        rules.append(f"\\cmidrule(lr){{{hc.col + 1}-{hc.col + hc.colspan}}}")
                    else:
                        top.append("")
                lines.append(" & ".join(top) + r" \\")
                lines.append(" ".join(rules))
                sub = {hc.col: hc.text for hc in rt.header_rows[1]}
                for hc in rt.header_rows[0]:
                    if hc.rowspan > 1:
                        sub[hc.col] = hc.text
                lines.append(" & ".join(_tex_escape(sub.get(i, "")) for i in range(rt.n_cols)) + r" \\")
            else:
                lines.append(" & ".join(_tex_escape(h.text) for h in rt.header_rows[0]) + r" \\")
            lines.append(r"\midrule")
            rules_above = set(rt.rules_above)
            for r_i, r_vals in enumerate(rt.body):
                if r_i in rules_above:
                    lines.append(r"\midrule")
                lines.append(" & ".join(_tex_escape(v) for v in r_vals) + r" \\")
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            if rt.notes:
                note_joined = " ".join(_tex_escape(n) for n in rt.notes)
                lines.append(f"\\\\[4pt]\\footnotesize\\emph{{Note.}} {note_joined}")
            lines.append(r"\end{table}")
            lines.append("")

        for p_i, p in enumerate(analysis.plots):
            fig_idx += 1
            vec = analysis.plot_vectors[p_i] if p_i < len(analysis.plot_vectors) else {}
            if vec.get("pdf_b64"):
                fig_fname = f"{stem}_fig{fig_idx}.pdf"
                with open(os.path.join(out_dir, fig_fname), "wb") as ff:
                    ff.write(base64.b64decode(vec["pdf_b64"]))
            else:
                png_bytes = _decode_plot_png(p)
                if not png_bytes:
                    continue
                fig_fname = f"{stem}_fig{fig_idx}.png"
                with open(os.path.join(out_dir, fig_fname), "wb") as ff:
                    ff.write(png_bytes)
            lines.append(r"\begin{figure}[htbp]")
            lines.append(r"\centering")
            lines.append(f"\\includegraphics[width=0.82\\textwidth]{{{fig_fname}}}")
            lines.append(f"\\caption{{{_tex_escape(p.get('title', f'Figure {fig_idx}'))}}}")
            lines.append(r"\end{figure}")
            lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output_path


def _markdown_tables(analysis: BaseAnalysis, rf: ReportFormat, numbers: Dict[int, int]) -> str:
    from tabulate import tabulate
    parts = [f"## {analysis.title}\n"]
    if analysis.syntax_python:
        parts.append(f"```python\n{analysis.syntax_python}\n```\n")
    for rt in _rendered_tables(analysis, rf, numbers):
        parts.append(f"**{rt.caption}**\n")
        parts.append(tabulate(rt.body, headers=[h or " " for h in rt.flat_headers], tablefmt="github"))
        lines = _note_lines(rt, rf, "")
        if lines:
            parts.append("")
            parts.extend(f"> {ln.strip()}" for ln in lines)
        parts.append("")
    if analysis.apa_narrative:
        parts.append(f"**APA Style Narrative:**\n> {analysis.apa_narrative}\n")
    return "\n".join(parts)


def export_to_markdown(analyses: List[BaseAnalysis], output_path: str, report_format: Optional[ReportFormat] = None) -> str:
    rf = report_format
    numbers = table_numbering(analyses) if rf else {}
    out_dir = os.path.dirname(os.path.abspath(output_path))
    stem = os.path.splitext(os.path.basename(output_path))[0]
    blocks = []
    fig_idx = 0
    for analysis in analyses:
        md_part = analysis.to_markdown() if rf is None else _markdown_tables(analysis, rf, numbers)
        if analysis.plots:
            img_links = []
            for p in analysis.plots:
                png_bytes = _decode_plot_png(p)
                if not png_bytes:
                    continue
                fig_idx += 1
                fig_fname = f"{stem}_fig{fig_idx}.png"
                with open(os.path.join(out_dir, fig_fname), "wb") as ff:
                    ff.write(png_bytes)
                img_links.append(f"![{p.get('title', f'Figure {fig_idx}')}]({fig_fname})")
            if img_links:
                md_part = md_part + "\n" + "\n\n".join(img_links) + "\n"
        blocks.append(md_part)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(blocks))
    return output_path


def export_to_delimited(analyses: List[BaseAnalysis], output_path: str, delimiter: str = ",",
                        report_format: Optional[ReportFormat] = None) -> str:
    rf = report_format
    numbers = table_numbering(analyses) if rf else {}
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=delimiter)
    for analysis in analyses:
        writer.writerow([f"=== {analysis.title} ==="])
        for rt in _rendered_tables(analysis, rf, numbers):
            writer.writerow([f"[{rt.caption}]"])
            writer.writerow(rt.flat_headers)
            for r_vals in rt.body:
                writer.writerow(r_vals)
            for n in _note_lines(rt, rf, "Note. "):
                writer.writerow([n.strip()])
            writer.writerow([])
        if analysis.plots:
            writer.writerow(["Charts Included", "; ".join(p.get("title", "Plot") for p in analysis.plots)])
        if analysis.apa_narrative:
            writer.writerow(["APA Summary", analysis.apa_narrative])
        writer.writerow([])

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(buf.getvalue())
    return output_path


def export_analyses(
    analyses: List[BaseAnalysis],
    output_path: str,
    fmt: Optional[str] = None,
    dataset: Optional[pd.DataFrame] = None,
    report_style: Any = None,
    report_options: Optional[Dict[str, Any]] = None,
) -> str:
    """Export analysis results to one of 11 formats.

    report_style: None/"default" (APA, every statistic in its own column) or "social"
    (stars, SE in parentheses, numbered captions; see `report_format.ReportFormat` for `report_options`).
    The social format applies to .hwpx, .docx, .xlsx, .pdf, .tex, .html, .md, .csv and .tsv;
    .json and .omv always carry the unformatted values.
    """
    ext = (fmt or (output_path.rsplit(".", 1)[-1] if "." in output_path else "md")).lower().lstrip(".")
    rf = resolve_report_format(report_style, report_options)
    if ext not in SUPPORTED_TABLE_FORMATS and ext not in ("latex", "markdown"):
        raise ValueError(f"Unsupported export format '{ext}'. Supported: {', '.join(SUPPORTED_TABLE_FORMATS)}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if ext == "xlsx":
        return export_to_xlsx(analyses, output_path, report_format=rf)
    elif ext == "docx":
        return export_to_docx(analyses, output_path, report_format=rf)
    elif ext == "hwpx":
        return export_to_hwpx(analyses, output_path, report_format=rf)
    elif ext == "pdf":
        return export_to_pdf(analyses, output_path, report_format=rf)
    elif ext in ("tex", "latex"):
        return export_to_latex(analyses, output_path, report_format=rf)
    elif ext == "csv":
        return export_to_delimited(analyses, output_path, delimiter=",", report_format=rf)
    elif ext == "tsv":
        return export_to_delimited(analyses, output_path, delimiter="\t", report_format=rf)
    elif ext in ("md", "markdown"):
        return export_to_markdown(analyses, output_path, report_format=rf)
    elif ext == "html":
        html_doc = render_jamovi_html_report(analyses, title="OpenStat Studio Statistical & Visualization Report", report_format=rf)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        return output_path
    elif ext == "json":
        payload = [a.to_payload(idx + 1) for idx, a in enumerate(analyses)]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return output_path
    elif ext == "omv":
        df = dataset
        if df is None and analyses and analyses[0].dataset is not None:
            df = analyses[0].dataset
        if df is None:
            raise ValueError("Dataset is required to export .omv archive.")
        write_omv(df, output_path, analyses=analyses)
        return output_path

    raise ValueError(f"Unsupported export format '{ext}'. Supported: {', '.join(SUPPORTED_TABLE_FORMATS)}")

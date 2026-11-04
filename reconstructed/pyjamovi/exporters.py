"""
OpenStat Studio — Multi-Format Publication Results, Table & Graph Exporter
License: MIT
Exports full Analysis Results (statistical APA tables, narratives, AND standalone/analysis graphs) into 11 researcher-ready formats:
- .pdf  (Multi-page Publication PDF Report with APA 3-line tables, interpretations & high-res figures)
- .hwpx (Hancom Office Hangul OWPML package with editable native Hangul tables & embedded BinData/*.png `<hp:pic>` graphs)
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
import zipfile
from typing import List, Optional, Any, Tuple
from xml.sax.saxutils import escape as xml_escape

import pandas as pd

from .analyses.base import BaseAnalysis, ResultsTable
from .renderers.html_renderer import render_jamovi_html_report
from .omv.writer import write_omv


SUPPORTED_TABLE_FORMATS = ("xlsx", "docx", "hwpx", "pdf", "tex", "csv", "tsv", "md", "html", "json", "omv")


def _formatted_matrix(tbl: ResultsTable):
    headers = [tbl.display_title(c) for c in tbl.columns]
    rows = [[tbl._format_cell(r.get(c["name"]), c) for c in tbl.columns] for r in tbl.rows]
    col_types = [c.get("type", "text") for c in tbl.columns]
    return headers, rows, col_types


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


def export_to_xlsx(analyses: List[BaseAnalysis], output_path: str) -> str:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)

    thin_dark = Side(border_style="thin", color="334155")
    medium_dark = Side(border_style="medium", color="0F172A")
    hdr_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")

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

        for tbl in analysis.tables:
            ws.cell(row=row_cursor, column=1, value=tbl.title).font = Font(name="Calibri", size=11, bold=True, color="1E293B")
            row_cursor += 1

            headers, rows, col_types = _formatted_matrix(tbl)
            if not headers:
                continue

            # Header Row (APA top medium border + bottom thin border)
            for c_idx, h_text in enumerate(headers, start=1):
                cell = ws.cell(row=row_cursor, column=c_idx, value=h_text)
                cell.font = Font(name="Calibri", size=10, bold=True, color="0F172A")
                cell.fill = hdr_fill
                cell.alignment = Alignment(horizontal="right" if col_types[c_idx - 1] in ("number", "integer") else "left", vertical="center")
                cell.border = Border(top=medium_dark, bottom=thin_dark)
            row_cursor += 1

            # Data Rows
            for r_i, r_vals in enumerate(rows):
                is_last = (r_i == len(rows) - 1)
                for c_idx, val in enumerate(r_vals, start=1):
                    cell = ws.cell(row=row_cursor, column=c_idx, value=val)
                    cell.font = Font(name="Calibri", size=10, color="1E293B")
                    cell.alignment = Alignment(horizontal="right" if col_types[c_idx - 1] in ("number", "integer") else "left", vertical="center")
                    if is_last:
                        cell.border = Border(bottom=medium_dark)
                row_cursor += 1

            for note in tbl.notes:
                ncell = ws.cell(row_cursor, 1, value=f"Note. {note}")
                ncell.font = Font(name="Calibri", size=9, italic=True, color="475569")
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

    if not wb.sheetnames:
        ws = wb.create_sheet("Summary")
        ws["A1"] = "No analyses in session."

    wb.save(output_path)
    return output_path


def _set_docx_apa_borders(table):
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

    # Bottom border on header row
    if len(table.rows) > 0:
        for cell in table.rows[0].cells:
            tcPr = cell._tc.get_or_add_tcPr()
            tcBorders = OxmlElement("w:tcBorders")
            b_bot = OxmlElement("w:bottom")
            b_bot.set(qn("w:val"), "single")
            b_bot.set(qn("w:sz"), "6")
            b_bot.set(qn("w:space"), "0")
            b_bot.set(qn("w:color"), "334155")
            tcBorders.append(b_bot)
            tcPr.append(tcBorders)


def export_to_docx(analyses: List[BaseAnalysis], output_path: str) -> str:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()
    title_h = doc.add_heading("OpenStat Studio — Statistical Analysis & Graph Report", level=1)
    title_h.runs[0].font.color.rgb = RGBColor(0x1E, 0x3A, 0x5F)

    fig_num = 0
    for analysis in analyses:
        h2 = doc.add_heading(analysis.title, level=2)
        if h2.runs:
            h2.runs[0].font.color.rgb = RGBColor(0x0D, 0x94, 0x88)

        for tbl in analysis.tables:
            p_title = doc.add_paragraph()
            run_t = p_title.add_run(tbl.title)
            run_t.bold = True
            run_t.font.size = Pt(10.5)

            headers, rows, col_types = _formatted_matrix(tbl)
            if not headers:
                continue

            w_tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
            for c_i, h_txt in enumerate(headers):
                cell = w_tbl.rows[0].cells[c_i]
                cell.text = h_txt
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if col_types[c_i] in ("number", "integer") else WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    r.bold = True
                    r.font.size = Pt(9.5)

            for r_i, r_vals in enumerate(rows):
                for c_i, val in enumerate(r_vals):
                    cell = w_tbl.rows[r_i + 1].cells[c_i]
                    cell.text = str(val)
                    p = cell.paragraphs[0]
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if col_types[c_i] in ("number", "integer") else WD_ALIGN_PARAGRAPH.LEFT
                    for r in p.runs:
                        r.font.size = Pt(9.5)

            _set_docx_apa_borders(w_tbl)

            for note in tbl.notes:
                pn = doc.add_paragraph()
                r1 = pn.add_run("Note. ")
                r1.italic = True
                r1.font.size = Pt(8.5)
                r2 = pn.add_run(note)
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
                rc = p_cap.add_run(f"Figure {fig_num}. {p.get('title', 'Statistical Visualization')}")
                rc.bold = True
                rc.font.size = Pt(9.5)
                doc.add_picture(io.BytesIO(img_bytes), width=Inches(5.5))

    doc.save(output_path)
    return output_path


def export_to_hwpx(analyses: List[BaseAnalysis], output_path: str) -> str:
    """
    Generates a valid Hancom Office OWPML (.hwpx) document containing:
    - Editable native Hangul APA 3-line tables (`<hp:tbl>`)
    - Embedded high-resolution statistical & standalone graphs (`BinData/BIN*.png` + `<hp:pic>`)
    - APA narrative interpretations
    """
    version_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n'
        '<hv:HCFVersion xmlns:hv="http://www.hancom.co.kr/hwpml/2011/version" '
        'tagetApplication="WORDPROCESSOR" major="5" minor="1" micro="0" buildNumber="1" '
        'os="1" xmlVersion="1.4" application="OpenStat Studio" appVersion="3.0"/>'
    )

    container_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n'
        '<ocf:container xmlns:ocf="urn:oasis:names:tc:opendocument:xmlns:container" xmlns:hpf="http://www.hancom.co.kr/schema/2011/hpf">'
        '<ocf:rootfiles><ocf:rootfile full-path="Contents/content.hpf" media-type="application/hwpml-package+xml"/></ocf:rootfiles>'
        '</ocf:container>'
    )

    header_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n'
        '<hh:head xmlns:hh="http://www.hancom.co.kr/hwpml/2011/head" version="1.4" secCnt="1">'
        '<hh:beginNum page="1" footnote="1" endnote="1" pic="1" tbl="1" equation="1"/>'
        '<hh:refList>'
        '<hh:fontfaces itemCnt="1"><hh:fontface lang="HANGUL" fontCnt="1"><hh:font id="0" face="함초롬돋움" type="TTF" isEmbedded="0"/></hh:fontface></hh:fontfaces>'
        '<hh:borderFills itemCnt="3">'
        '<hh:borderFill id="1" threeD="0" shadow="0" centerLine="NONE"><hh:leftBorder type="NONE" width="0.1 mm" color="#000000"/><hh:rightBorder type="NONE" width="0.1 mm" color="#000000"/><hh:topBorder type="NONE" width="0.1 mm" color="#000000"/><hh:bottomBorder type="NONE" width="0.1 mm" color="#000000"/></hh:borderFill>'
        '<hh:borderFill id="2" threeD="0" shadow="0" centerLine="NONE"><hh:leftBorder type="NONE" width="0.1 mm" color="#000000"/><hh:rightBorder type="NONE" width="0.1 mm" color="#000000"/><hh:topBorder type="SOLID" width="0.4 mm" color="#1E293B"/><hh:bottomBorder type="SOLID" width="0.25 mm" color="#334155"/></hh:borderFill>'
        '<hh:borderFill id="3" threeD="0" shadow="0" centerLine="NONE"><hh:leftBorder type="NONE" width="0.1 mm" color="#000000"/><hh:rightBorder type="NONE" width="0.1 mm" color="#000000"/><hh:topBorder type="NONE" width="0.1 mm" color="#000000"/><hh:bottomBorder type="SOLID" width="0.4 mm" color="#1E293B"/></hh:borderFill>'
        '</hh:borderFills>'
        '<hh:charProperties itemCnt="2">'
        '<hh:charPr id="0" height="1000" textColor="#1E293B" shadeColor="none" useFontSpace="0" useKerning="0" symMark="NONE" borderFillIDRef="1"/>'
        '<hh:charPr id="1" height="1100" textColor="#0F172A" shadeColor="none" useFontSpace="0" useKerning="0" symMark="NONE" borderFillIDRef="1"><hh:bold/></hh:charPr>'
        '</hh:charProperties>'
        '<hh:paraProperties itemCnt="2">'
        '<hh:paraPr id="0" tabPrIDRef="0" condense="0" fontLineHeight="0" snapToGrid="1"><hh:align horizontal="LEFT" vertical="BASELINE"/></hh:paraPr>'
        '<hh:paraPr id="1" tabPrIDRef="0" condense="0" fontLineHeight="0" snapToGrid="1"><hh:align horizontal="RIGHT" vertical="BASELINE"/></hh:paraPr>'
        '</hh:paraProperties>'
        '</hh:refList>'
        '</hh:head>'
    )

    paras: List[str] = []
    embedded_images: List[Dict[str, Any]] = []

    def _p(text: str, bold: bool = False, right: bool = False) -> str:
        c_id = "1" if bold else "0"
        p_id = "1" if right else "0"
        return f'<hp:p paraPrIDRef="{p_id}" styleIDRef="0"><hp:run charPrIDRef="{c_id}"><hp:t>{xml_escape(str(text))}</hp:t></hp:run></hp:p>'

    paras.append(_p("OpenStat Studio — 통계 분석 및 시각화 결과 보고서", bold=True))

    tbl_id = 100
    pic_seq = 0
    for analysis in analyses:
        paras.append(_p(f"■ {analysis.title}", bold=True))
        for tbl in analysis.tables:
            paras.append(_p(f"[{tbl.title}]", bold=True))
            headers, rows, col_types = _formatted_matrix(tbl)
            if not headers:
                continue
            r_cnt = 1 + len(rows)
            c_cnt = len(headers)
            col_w = max(3500, 42520 // max(1, c_cnt))
            tbl_id += 1
            tr_xml = []

            # Header row
            th_cells = []
            for c_i, h_txt in enumerate(headers):
                is_num = col_types[c_i] in ("number", "integer")
                th_cells.append(
                    f'<hp:tc borderFillIDRef="2"><hp:subList id="" textDirection="HORIZONTAL" lineWrap="BREAK" vertAlign="CENTER">'
                    f'{_p(h_txt, bold=True, right=is_num)}</hp:subList>'
                    f'<hp:cellAddr colAddr="{c_i}" rowAddr="0"/><hp:cellSpan colSpan="1" rowSpan="1"/>'
                    f'<hp:cellSz width="{col_w}" height="1800"/><hp:cellMargin left="280" right="280" top="140" bottom="140"/></hp:tc>'
                )
            tr_xml.append(f"<hp:tr>{''.join(th_cells)}</hp:tr>")

            # Body rows
            for r_i, r_vals in enumerate(rows):
                bf_id = "3" if r_i == len(rows) - 1 else "1"
                td_cells = []
                for c_i, val in enumerate(r_vals):
                    is_num = col_types[c_i] in ("number", "integer")
                    td_cells.append(
                        f'<hp:tc borderFillIDRef="{bf_id}"><hp:subList id="" textDirection="HORIZONTAL" lineWrap="BREAK" vertAlign="CENTER">'
                        f'{_p(val, bold=False, right=is_num)}</hp:subList>'
                        f'<hp:cellAddr colAddr="{c_i}" rowAddr="{r_i + 1}"/><hp:cellSpan colSpan="1" rowSpan="1"/>'
                        f'<hp:cellSz width="{col_w}" height="1650"/><hp:cellMargin left="280" right="280" top="120" bottom="120"/></hp:tc>'
                    )
                tr_xml.append(f"<hp:tr>{''.join(td_cells)}</hp:tr>")

            tbl_xml = (
                f'<hp:p paraPrIDRef="0" styleIDRef="0"><hp:run charPrIDRef="0">'
                f'<hp:tbl id="{tbl_id}" zOrder="0" numberingType="TABLE" textWrap="TOP_AND_BOTTOM" rowCnt="{r_cnt}" colCnt="{c_cnt}" cellSpacing="0" borderFillIDRef="1">'
                f'{"".join(tr_xml)}</hp:tbl></hp:run></hp:p>'
            )
            paras.append(tbl_xml)

            for note in tbl.notes:
                paras.append(_p(f"주: {note}"))

        if analysis.apa_narrative:
            paras.append(_p(f"해석(APA): {analysis.apa_narrative}"))

        # Embed all figures as native OWPML <hp:pic> + BinData/*.png
        for p in analysis.plots:
            png_bytes = _decode_plot_png(p)
            if not png_bytes:
                continue
            pic_seq += 1
            img_id = f"BIN{pic_seq:04d}"
            img_fname = f"{img_id}.png"
            px_w, px_h = _png_dimensions(png_bytes)
            w_hwp = 40000
            h_hwp = max(12000, min(48000, int(w_hwp * (px_h / max(1, px_w)))))
            embedded_images.append({
                "id": img_id,
                "filename": img_fname,
                "data": png_bytes,
            })
            p_title = p.get("title") or f"시각화 그래프 {pic_seq}"
            paras.append(_p(f"[그림 {pic_seq}] {p_title}", bold=True))
            pic_xml = (
                f'<hp:p paraPrIDRef="0" styleIDRef="0"><hp:run charPrIDRef="0">'
                f'<hp:pic id="{200000000 + pic_seq}" zOrder="{pic_seq}" '
                f'numberingType="PICTURE" textWrap="TOP_AND_BOTTOM" textFlow="BOTH_SIDES" '
                f'lock="0" dropcapstyle="None" href="" groupLevel="0" instid="{30000000 + pic_seq}" reverse="0">'
                f'<hp:offset x="0" y="0"/>'
                f'<hp:orgSz width="{w_hwp}" height="{h_hwp}"/>'
                f'<hp:curSz width="{w_hwp}" height="{h_hwp}"/>'
                f'<hp:flip horizontal="0" vertical="0"/>'
                f'<hp:rotationInfo angle="0" centerX="0" centerY="0" rotateimage="1"/>'
                f'<hp:renderingInfo>'
                f'<hc:transMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/>'
                f'<hc:scaMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/>'
                f'<hc:rotMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/>'
                f'</hp:renderingInfo>'
                f'<hp:imgRect><hc:pt0 x="0" y="0"/><hc:pt1 x="{w_hwp}" y="0"/>'
                f'<hc:pt2 x="{w_hwp}" y="{h_hwp}"/><hc:pt3 x="0" y="{h_hwp}"/></hp:imgRect>'
                f'<hp:imgClip left="0" right="0" top="0" bottom="0"/>'
                f'<hp:inMargin left="0" right="0" top="0" bottom="0"/>'
                f'<hc:img binaryItemIDRef="{img_id}" bright="0" contrast="0" effect="REAL_PIC" alpha="0"/>'
                f'<hp:effects/>'
                f'<hp:sz width="{w_hwp}" widthRelTo="ABSOLUTE" height="{h_hwp}" heightRelTo="ABSOLUTE" protect="0"/>'
                f'<hp:pos treatAsChar="1" affectLSpacing="0" flowWithText="1" allowOverlap="1" '
                f'holdAnchorAndSO="0" vertRelTo="PARA" horzRelTo="COLUMN" vertAlign="TOP" horzAlign="LEFT" '
                f'vertOffset="0" horzOffset="0"/>'
                f'<hp:outMargin left="0" right="0" top="140" bottom="280"/>'
                f'</hp:pic></hp:run></hp:p>'
            )
            paras.append(pic_xml)

    manifest_img_entries = "".join(
        f'<odf:file-entry full-path="BinData/{img["filename"]}" media-type="image/png"/>'
        for img in embedded_images
    )
    manifest_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n'
        '<odf:manifest xmlns:odf="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0">'
        '<odf:file-entry full-path="/" media-type="application/hwp+zip"/>'
        '<odf:file-entry full-path="Contents/header.xml" media-type="application/xml"/>'
        '<odf:file-entry full-path="Contents/section0.xml" media-type="application/xml"/>'
        f'{manifest_img_entries}'
        '</odf:manifest>'
    )

    hpf_img_items = "".join(
        f'<opf:item id="{img["id"]}" href="BinData/{img["filename"]}" media-type="image/png" isEmbeded="1"/>'
        for img in embedded_images
    )
    content_hpf = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n'
        '<opf:package xmlns:opf="http://www.idpf.org/2007/opf/" version="2.0" unique-identifier="openstat-hwpx">'
        '<opf:metadata><opf:title>OpenStat Studio Statistical Report</opf:title><opf:language>ko</opf:language></opf:metadata>'
        '<opf:manifest>'
        '<opf:item id="header" href="header.xml" media-type="application/xml"/>'
        '<opf:item id="section0" href="section0.xml" media-type="application/xml"/>'
        f'{hpf_img_items}'
        '</opf:manifest>'
        '<opf:spine><opf:itemref idref="header" linear="no"/><opf:itemref idref="section0" linear="yes"/></opf:spine>'
        '</opf:package>'
    )

    section_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n'
        '<hs:sec xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section" '
        'xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph" '
        'xmlns:hc="http://www.hancom.co.kr/hwpml/2011/core">'
        + "".join(paras)
        + "</hs:sec>"
    )

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/hwp+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("version.xml", version_xml)
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("META-INF/manifest.xml", manifest_xml)
        zf.writestr("Contents/content.hpf", content_hpf)
        zf.writestr("Contents/header.xml", header_xml)
        zf.writestr("Contents/section0.xml", section_xml)
        for img in embedded_images:
            zf.writestr(f'BinData/{img["filename"]}', img["data"])

    return output_path


def export_to_pdf(analyses: List[BaseAnalysis], output_path: str) -> str:
    """
    Renders a multi-page publication PDF report containing:
    - Report title header
    - APA 3-line statistical tables & footnotes
    - APA interpretations
    - All statistical & standalone visualizations embedded at crisp publication quality
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.backends.backend_pdf import PdfPages
    from .analyses.plot_style import DEFAULT_FONT_STACK

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
                    # Page(s) for tables + APA summary (when present, or title card when no plots either)
                    if analysis.tables or analysis.apa_narrative or not analysis.plots:
                        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
                        ax = fig.add_axes([0, 0, 1, 1])
                        ax.axis("off")

                        y = 0.94
                        if a_idx == 1:
                            ax.text(0.07, y, "OpenStat Studio — Analysis & Visualization Results", fontsize=14, fontweight="bold", color="#1E3A5F")
                            ax.plot([0.07, 0.93], [y - 0.014, y - 0.014], color="#0D9488", lw=1.6)
                            y -= 0.045

                        ax.text(0.07, y, f"{a_idx}. {analysis.title}", fontsize=12, fontweight="bold", color="#0F172A")
                        y -= 0.032

                        for tbl in analysis.tables:
                            headers, rows, col_types = _formatted_matrix(tbl)
                            if not headers:
                                continue
                            display_rows = rows[:28]
                            needed_h = 0.035 + 0.024 * (1 + len(display_rows)) + 0.02 * len(tbl.notes)
                            if y - needed_h < 0.08:
                                pdf.savefig(fig)
                                plt.close(fig)
                                fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
                                ax = fig.add_axes([0, 0, 1, 1])
                                ax.axis("off")
                                y = 0.94

                            ax.text(0.07, y, tbl.title, fontsize=10, fontweight="bold", color="#1E293B")
                            y -= 0.016

                            n_cols = len(headers)
                            x0, x1 = 0.07, 0.93
                            col_w = (x1 - x0) / n_cols

                            # APA Top Rule
                            ax.plot([x0, x1], [y, y], color="#0F172A", lw=1.2)
                            y -= 0.020
                            for c_i, h_txt in enumerate(headers):
                                is_num = col_types[c_i] in ("number", "integer")
                                tx = (x0 + (c_i + 0.94) * col_w) if is_num else (x0 + (c_i + 0.04) * col_w)
                                ha = "right" if is_num else "left"
                                ax.text(tx, y + 0.004, str(h_txt)[:24], fontsize=8.5, fontweight="bold", color="#0F172A", ha=ha, va="bottom")
                            # APA Header Rule
                            ax.plot([x0, x1], [y - 0.003, y - 0.003], color="#334155", lw=0.7)
                            y -= 0.022

                            for r_vals in display_rows:
                                for c_i, val in enumerate(r_vals):
                                    is_num = col_types[c_i] in ("number", "integer")
                                    tx = (x0 + (c_i + 0.94) * col_w) if is_num else (x0 + (c_i + 0.04) * col_w)
                                    ha = "right" if is_num else "left"
                                    ax.text(tx, y + 0.004, str(val)[:26], fontsize=8.3, color="#1E293B", ha=ha, va="bottom")
                                y -= 0.020

                            # APA Bottom Rule
                            ax.plot([x0, x1], [y + 0.015, y + 0.015], color="#0F172A", lw=1.2)
                            y -= 0.006

                            for note in tbl.notes:
                                ax.text(0.07, y, f"Note. {note}"[:110], fontsize=7.8, style="italic", color="#475569")
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
                        fig_p.text(0.07, 0.90, f"Figure {fig_counter}. {p_title}", fontsize=10.5, fontweight="bold", color="#0D9488")

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
        "< .001": r"$< .001$",
    }
    out = str(s)
    for k in ("χ²", "η²p", "η²G", "η²", "ω²", "ε²", "ε", "α", "ω", "β"):
        out = out.replace(k, rep[k])
    out = out.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")
    return out


def export_to_latex(analyses: List[BaseAnalysis], output_path: str) -> str:
    lines = [
        r"% OpenStat Studio — Publication LaTeX Tables & Figures (Requires \usepackage{booktabs, graphicx})",
        ""
    ]
    out_dir = os.path.dirname(os.path.abspath(output_path))
    stem = os.path.splitext(os.path.basename(output_path))[0]
    fig_idx = 0

    for analysis in analyses:
        lines.append(f"% === {analysis.title} ===")
        for tbl in analysis.tables:
            headers, rows, col_types = _formatted_matrix(tbl)
            if not headers:
                continue
            align = "".join("r" if t in ("number", "integer") else "l" for t in col_types)
            lines.append(r"\begin{table}[htbp]")
            lines.append(r"\centering")
            lines.append(f"\\caption{{{_tex_escape(tbl.title)}}}")
            lines.append(f"\\begin{{tabular}}{{{align}}}")
            lines.append(r"\toprule")
            lines.append(" & ".join(_tex_escape(h) for h in headers) + r" \\")
            lines.append(r"\midrule")
            for r_vals in rows:
                lines.append(" & ".join(_tex_escape(v) for v in r_vals) + r" \\")
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            if tbl.notes:
                note_joined = " ".join(_tex_escape(n) for n in tbl.notes)
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


def export_to_markdown(analyses: List[BaseAnalysis], output_path: str) -> str:
    out_dir = os.path.dirname(os.path.abspath(output_path))
    stem = os.path.splitext(os.path.basename(output_path))[0]
    blocks = []
    fig_idx = 0
    for analysis in analyses:
        md_part = analysis.to_markdown()
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


def export_to_delimited(analyses: List[BaseAnalysis], output_path: str, delimiter: str = ",") -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=delimiter)
    for analysis in analyses:
        writer.writerow([f"=== {analysis.title} ==="])
        for tbl in analysis.tables:
            writer.writerow([f"[{tbl.title}]"])
            headers, rows, _ = _formatted_matrix(tbl)
            writer.writerow(headers)
            for r_vals in rows:
                writer.writerow(r_vals)
            for n in tbl.notes:
                writer.writerow([f"Note. {n}"])
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
    dataset: Optional[pd.DataFrame] = None
) -> str:
    ext = (fmt or (output_path.rsplit(".", 1)[-1] if "." in output_path else "md")).lower().lstrip(".")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if ext == "xlsx":
        return export_to_xlsx(analyses, output_path)
    elif ext == "docx":
        return export_to_docx(analyses, output_path)
    elif ext == "hwpx":
        return export_to_hwpx(analyses, output_path)
    elif ext == "pdf":
        return export_to_pdf(analyses, output_path)
    elif ext in ("tex", "latex"):
        return export_to_latex(analyses, output_path)
    elif ext == "csv":
        return export_to_delimited(analyses, output_path, delimiter=",")
    elif ext == "tsv":
        return export_to_delimited(analyses, output_path, delimiter="\t")
    elif ext in ("md", "markdown"):
        return export_to_markdown(analyses, output_path)
    elif ext == "html":
        html_doc = render_jamovi_html_report(analyses, title="OpenStat Studio Statistical & Visualization Report")
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

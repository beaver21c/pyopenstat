"""
OpenStat Studio — OWPML (.hwpx) package writer for analysis reports.
License: MIT

Writes a complete Hancom Office Hangul package: namespace-complete header (fonts, border fills,
character/paragraph properties, styles), section properties (A4 page), native editable tables with
merged header cells (`hp:cellSpan`, covered cells omitted as Hangul does), numbered captions and
embedded PNG figures (`BinData/` + `hp:pic`). Element order and required children follow files
saved by Hangul 2020+ (table `hp:sz`/`hp:pos`/`hp:outMargin`/`hp:inMargin`, first-paragraph `hp:secPr`).
"""

from __future__ import annotations

import re
import struct
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape as _esc

from .report_format import RenderedTable

NS = (
    'xmlns:ha="http://www.hancom.co.kr/hwpml/2011/app" '
    'xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph" '
    'xmlns:hp10="http://www.hancom.co.kr/hwpml/2016/paragraph" '
    'xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section" '
    'xmlns:hc="http://www.hancom.co.kr/hwpml/2011/core" '
    'xmlns:hh="http://www.hancom.co.kr/hwpml/2011/head" '
    'xmlns:hhs="http://www.hancom.co.kr/hwpml/2011/history" '
    'xmlns:hm="http://www.hancom.co.kr/hwpml/2011/master-page" '
    'xmlns:hpf="http://www.hancom.co.kr/schema/2011/hpf" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'xmlns:opf="http://www.idpf.org/2007/opf/" '
    'xmlns:ooxmlchart="http://www.hancom.co.kr/hwpml/2016/ooxmlchart" '
    'xmlns:hwpunitchar="http://www.hancom.co.kr/hwpml/2016/HwpUnitChar" '
    'xmlns:epub="http://www.idpf.org/2007/ops" '
    'xmlns:config="urn:oasis:names:tc:opendocument:xmlns:config:1.0"'
)
XML_DECL = '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>'

PAGE_W, PAGE_H = 59528, 84188          # A4 in HWPUNIT (1/7200 inch)
MARGIN_LR, MARGIN_T, MARGIN_B = 8504, 5668, 4252
TEXT_W = PAGE_W - 2 * MARGIN_LR         # 42520

THICK, THIN = "0.4 mm", "0.12 mm"
TEXT_COLOR, NOTE_COLOR, HEAD_COLOR = "#1E293B", "#475569", "#0F172A"

# character properties: (id, height(1/100 pt), color, bold)
CHAR = {"body": 0, "heading": 1, "cell": 2, "cell_bold": 3, "note": 4, "title": 5, "caption": 6}
_CHAR_DEFS = [
    (0, 1000, TEXT_COLOR, False),
    (1, 1150, HEAD_COLOR, True),
    (2, 950, TEXT_COLOR, False),
    (3, 950, HEAD_COLOR, True),
    (4, 850, NOTE_COLOR, False),
    (5, 1400, "#1E3A5F", True),
    (6, 1000, HEAD_COLOR, True),
]
# paragraph properties: (id, align, line spacing %, space before, space after, keepWithNext)
PARA = {"body": 0, "cell_left": 1, "cell_right": 2, "cell_center": 3, "caption": 4, "note": 5, "figure": 6, "heading": 7}
_PARA_DEFS = [
    (0, "JUSTIFY", 160, 0, 400, 0),
    (1, "LEFT", 130, 0, 0, 0),
    (2, "RIGHT", 130, 0, 0, 0),
    (3, "CENTER", 130, 0, 0, 0),
    (4, "LEFT", 140, 600, 200, 1),
    (5, "LEFT", 130, 0, 0, 0),
    (6, "CENTER", 100, 0, 400, 0),
    (7, "LEFT", 160, 800, 300, 1),
]
# border fills: id -> (top, bottom); 1 = page, 2 = character
BF_NONE, BF_HEAD, BF_SUBHEAD, BF_LAST, BF_RULE, BF_RULE_LAST, BF_PLAIN = 3, 4, 5, 6, 7, 8, 9
_BF_DEFS = {
    1: (None, None), 2: (None, None), 3: (None, None),
    4: (THICK, THIN), 5: (None, THIN), 6: (None, THICK), 7: (THIN, None), 8: (THIN, THICK), 9: (None, None),
}


def _t(text: Any) -> str:
    s = "" if text is None else str(text)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s).replace("\r", " ").replace("\n", " ")
    return _esc(s)


def _border(tag: str, width: Optional[str]) -> str:
    if width:
        return f'<hh:{tag} type="SOLID" width="{width}" color="#1E293B"/>'
    return f'<hh:{tag} type="NONE" width="0.1 mm" color="#000000"/>'


def _border_fill(bf_id: int, top: Optional[str], bottom: Optional[str]) -> str:
    return (
        f'<hh:borderFill id="{bf_id}" threeD="0" shadow="0" centerLine="NONE" breakCellSeparateLine="0">'
        '<hh:slash type="NONE" Crooked="0" isCounter="0"/><hh:backSlash type="NONE" Crooked="0" isCounter="0"/>'
        f'{_border("leftBorder", None)}{_border("rightBorder", None)}{_border("topBorder", top)}{_border("bottomBorder", bottom)}'
        '<hh:diagonal type="SOLID" width="0.1 mm" color="#000000"/>'
        '<hc:fillBrush><hc:winBrush faceColor="none" hatchColor="#999999" alpha="0"/></hc:fillBrush>'
        '</hh:borderFill>'
    )


_LANGS = ("hangul", "latin", "hanja", "japanese", "other", "symbol", "user")


def _char_pr(cid: int, height: int, color: str, bold: bool) -> str:
    def attrs(v):
        return " ".join(f'{k}="{v}"' for k in _LANGS)
    return (
        f'<hh:charPr id="{cid}" height="{height}" textColor="{color}" shadeColor="none" useFontSpace="0" '
        f'useKerning="0" symMark="NONE" borderFillIDRef="2">'
        f'<hh:fontRef {attrs(0)}/><hh:ratio {attrs(100)}/><hh:spacing {attrs(0)}/>'
        f'<hh:relSz {attrs(100)}/><hh:offset {attrs(0)}/>'
        + ('<hh:bold/>' if bold else '')
        + '<hh:underline type="NONE" shape="SOLID" color="#000000"/><hh:strikeout shape="NONE" color="#000000"/>'
        '<hh:outline type="NONE"/><hh:shadow type="NONE" color="#C0C0C0" offsetX="10" offsetY="10"/></hh:charPr>'
    )


def _para_pr(pid: int, align: str, spacing: int, prev: int, nxt: int, keep_next: int) -> str:
    margin = (
        '<hh:margin><hc:intent value="0" unit="HWPUNIT"/><hc:left value="0" unit="HWPUNIT"/>'
        '<hc:right value="0" unit="HWPUNIT"/>'
        f'<hc:prev value="{prev}" unit="HWPUNIT"/><hc:next value="{nxt}" unit="HWPUNIT"/></hh:margin>'
        f'<hh:lineSpacing type="PERCENT" value="{spacing}" unit="HWPUNIT"/>'
    )
    return (
        f'<hh:paraPr id="{pid}" tabPrIDRef="0" condense="0" fontLineHeight="0" snapToGrid="1" '
        f'suppressLineNumbers="0" checked="0" textDir="LTR">'
        f'<hh:align horizontal="{align}" vertical="BASELINE"/><hh:heading type="NONE" idRef="0" level="0"/>'
        f'<hh:breakSetting breakLatinWord="KEEP_WORD" breakNonLatinWord="BREAK_WORD" widowOrphan="0" '
        f'keepWithNext="{keep_next}" keepLines="0" pageBreakBefore="0" lineWrap="BREAK"/>'
        '<hh:autoSpacing eAsianEng="0" eAsianNum="0"/>'
        f'<hp:switch><hp:case hp:required-namespace="http://www.hancom.co.kr/hwpml/2016/HwpUnitChar">{margin}</hp:case>'
        f'<hp:default>{margin}</hp:default></hp:switch>'
        '<hh:border borderFillIDRef="2" offsetLeft="0" offsetRight="0" offsetTop="0" offsetBottom="0" connect="0" ignoreMargin="0"/>'
        '</hh:paraPr>'
    )


def _numbering() -> str:
    fmts = [("DIGIT", "^1."), ("HANGUL_SYLLABLE", "^2."), ("DIGIT", "^3)"), ("HANGUL_SYLLABLE", "^4)"),
            ("DIGIT", "(^5)"), ("HANGUL_SYLLABLE", "(^6)"), ("CIRCLED_DIGIT", "^7")]
    heads = "".join(
        f'<hh:paraHead start="1" level="{i + 1}" align="LEFT" useInstWidth="1" autoIndent="1" widthAdjust="0" '
        f'textOffsetType="PERCENT" textOffset="50" numFormat="{nf}" charPrIDRef="4294967295" checkable="0">{txt}</hh:paraHead>'
        for i, (nf, txt) in enumerate(fmts)
    )
    return f'<hh:numberings itemCnt="1"><hh:numbering id="1" start="0">{heads}</hh:numbering></hh:numberings>'


def header_xml(font: str = "함초롬돋움") -> str:
    font_xml = (
        f'<hh:font id="0" face="{_t(font)}" type="TTF" isEmbedded="0"><hh:typeInfo familyType="FCAT_GOTHIC" weight="6" '
        'proportion="4" contrast="0" strokeVariation="1" armStyle="1" letterform="1" midline="1" xHeight="1"/></hh:font>'
    )
    faces = "".join(f'<hh:fontface lang="{lang}" fontCnt="1">{font_xml}</hh:fontface>'
                    for lang in ("HANGUL", "LATIN", "HANJA", "JAPANESE", "OTHER", "SYMBOL", "USER"))
    bfs = "".join(_border_fill(i, *_BF_DEFS[i]) for i in sorted(_BF_DEFS))
    chars = "".join(_char_pr(*d) for d in _CHAR_DEFS)
    paras = "".join(_para_pr(*d) for d in _PARA_DEFS)
    return (
        f'{XML_DECL}<hh:head {NS} version="1.5" secCnt="1">'
        '<hh:beginNum page="1" footnote="1" endnote="1" pic="1" tbl="1" equation="1"/>'
        '<hh:refList>'
        f'<hh:fontfaces itemCnt="7">{faces}</hh:fontfaces>'
        f'<hh:borderFills itemCnt="{len(_BF_DEFS)}">{bfs}</hh:borderFills>'
        f'<hh:charProperties itemCnt="{len(_CHAR_DEFS)}">{chars}</hh:charProperties>'
        '<hh:tabProperties itemCnt="1"><hh:tabPr id="0" autoTabLeft="0" autoTabRight="0"/></hh:tabProperties>'
        f'{_numbering()}'
        f'<hh:paraProperties itemCnt="{len(_PARA_DEFS)}">{paras}</hh:paraProperties>'
        '<hh:styles itemCnt="1"><hh:style id="0" type="PARA" name="바탕글" engName="Normal" paraPrIDRef="0" '
        'charPrIDRef="0" nextStyleIDRef="0" langID="1042" lockForm="0"/></hh:styles>'
        '</hh:refList>'
        '<hh:compatibleDocument targetProgram="HWP201X"><hh:layoutCompatibility/></hh:compatibleDocument>'
        '<hh:docOption><hh:linkinfo path="" pageInherit="0" footnoteInherit="0"/></hh:docOption>'
        '<hh:trackchageConfig flags="56"/>'
        '</hh:head>'
    )


def _sec_pr() -> str:
    note_line = '<hp:noteLine length="-1" type="SOLID" width="0.12 mm" color="#000000"/>'
    page_bf = "".join(
        f'<hp:pageBorderFill type="{t}" borderFillIDRef="1" textBorder="PAPER" headerInside="0" footerInside="0" '
        'fillArea="PAPER"><hp:offset left="1417" right="1417" top="1417" bottom="1417"/></hp:pageBorderFill>'
        for t in ("BOTH", "EVEN", "ODD")
    )
    return (
        '<hp:secPr id="" textDirection="HORIZONTAL" spaceColumns="1134" tabStop="8000" tabStopVal="4000" '
        'tabStopUnit="HWPUNIT" outlineShapeIDRef="1" memoShapeIDRef="0" textVerticalWidthHead="0" masterPageCnt="0">'
        '<hp:grid lineGrid="0" charGrid="0" wonggojiFormat="0"/>'
        '<hp:startNum pageStartsOn="BOTH" page="0" pic="0" tbl="0" equation="0"/>'
        '<hp:visibility hideFirstHeader="0" hideFirstFooter="0" hideFirstMasterPage="0" border="SHOW_ALL" '
        'fill="SHOW_ALL" hideFirstPageNum="0" hideFirstEmptyLine="0" showLineNumber="0"/>'
        '<hp:lineNumberShape restartType="0" countBy="0" distance="0" startNumber="0"/>'
        f'<hp:pagePr landscape="WIDELY" width="{PAGE_W}" height="{PAGE_H}" gutterType="LEFT_ONLY">'
        f'<hp:margin header="4252" footer="4252" gutter="0" left="{MARGIN_LR}" right="{MARGIN_LR}" top="{MARGIN_T}" bottom="{MARGIN_B}"/></hp:pagePr>'
        '<hp:footNotePr><hp:autoNumFormat type="DIGIT" userChar="" prefixChar="" suffixChar=")" supscript="0"/>'
        f'{note_line}<hp:noteSpacing betweenNotes="283" belowLine="567" aboveLine="850"/>'
        '<hp:numbering type="CONTINUOUS" newNum="1"/><hp:placement place="EACH_COLUMN" beneathText="0"/></hp:footNotePr>'
        '<hp:endNotePr><hp:autoNumFormat type="DIGIT" userChar="" prefixChar="" suffixChar=")" supscript="0"/>'
        f'{note_line}<hp:noteSpacing betweenNotes="0" belowLine="567" aboveLine="850"/>'
        '<hp:numbering type="CONTINUOUS" newNum="1"/><hp:placement place="END_OF_DOCUMENT" beneathText="0"/></hp:endNotePr>'
        f'{page_bf}</hp:secPr>'
        '<hp:ctrl><hp:colPr id="" type="NEWSPAPER" layout="LEFT" colCount="1" sameSz="1" sameGap="0"/></hp:ctrl>'
    )


CELL_PAD_LR = 283      # left/right cell margin (1 mm)


def _text_width(s: str) -> int:
    """Approximate rendered width (HWPUNIT) of a 9.5 pt line: full-width (Hangul/CJK) ≈ 1 em, digits ≈ 0.55 em."""
    w = 0
    for ch in s:
        o = ord(ch)
        if o >= 0x1100:
            w += 950
        elif ch in " .,:;'|()[]!":
            w += 290
        elif ch.isupper() or ch in "%@#&≤≥±×÷→←":
            w += 660
        else:
            w += 530
    return w


def _need(s: str) -> int:
    return _text_width(s) + 2 * CELL_PAD_LR + 260


def png_size(data: bytes) -> Tuple[int, int]:
    if data and len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        try:
            w, h = struct.unpack(">II", data[16:24])
            if w > 0 and h > 0:
                return int(w), int(h)
        except struct.error:
            pass
    return 930, 600


class HwpxBuilder:
    """Accumulates paragraphs, tables and pictures, then writes the .hwpx package."""

    def __init__(self, title: str = "OpenStat Studio Report", font: str = "함초롬돋움"):
        self.title = title
        self.font = font
        self.paras: List[str] = []
        self.images: List[Dict[str, Any]] = []
        self.preview: List[str] = []
        self._obj_id = 1000000
        self._z = 0

    # ------------------------------------------------------------------ primitives
    def _next_id(self) -> int:
        self._obj_id += 1
        return self._obj_id

    @staticmethod
    def _run(text: str, char: int) -> str:
        if text == "":
            return f'<hp:run charPrIDRef="{char}"/>'
        return f'<hp:run charPrIDRef="{char}"><hp:t>{_t(text)}</hp:t></hp:run>'

    def _p(self, inner: str, para: int) -> str:
        return f'<hp:p id="0" paraPrIDRef="{para}" styleIDRef="0" pageBreak="0" columnBreak="0" merged="0">{inner}</hp:p>'

    def paragraph(self, text: str, kind: str = "body") -> None:
        char = {"body": CHAR["body"], "heading": CHAR["heading"], "title": CHAR["title"],
                "caption": CHAR["caption"], "note": CHAR["note"]}[kind]
        para = {"body": PARA["body"], "heading": PARA["heading"], "title": PARA["heading"],
                "caption": PARA["caption"], "note": PARA["note"]}[kind]
        self.paras.append(self._p(self._run(text, char), para))
        self.preview.append(str(text))

    # ------------------------------------------------------------------ tables
    def _col_widths(self, rt: RenderedTable) -> List[int]:
        """Column widths filling the text width: numeric columns keep one-line width, text columns absorb the rest."""
        n = rt.n_cols
        body = [0] * n
        word = [0] * n
        head = [0] * n
        for row in rt.body:
            for i, v in enumerate(row):
                body[i] = max(body[i], _need(v))
        for row in rt.header_rows:
            for hc in row:
                if hc.colspan == 1:
                    longest = max((_need(t) for t in hc.text.split()), default=0)
                    word[hc.col] = max(word[hc.col], longest)
                    head[hc.col] = max(head[hc.col], min(_need(hc.text), 2 * max(body[hc.col], longest)))
        pref = [max(body[i], word[i], head[i], 2200) for i in range(n)]
        floor = [max(word[i], 2200, min(pref[i], 4 * 950 + 2 * CELL_PAD_LR)) for i in range(n)]
        total = sum(pref)
        if total <= TEXT_W:
            extra = TEXT_W - total
            widths = [p + int(extra * p / total) for p in pref]
        else:
            num = [i for i in range(n) if rt.col_types[i] == "number"]
            txt = [i for i in range(n) if rt.col_types[i] != "number"]
            fixed = sum(pref[i] for i in num)
            remain = TEXT_W - fixed
            t_min = sum(floor[i] for i in txt)
            t_pref = sum(pref[i] for i in txt)
            if txt and remain >= t_min and t_pref > t_min:
                k = (remain - t_min) / (t_pref - t_min)
                widths = list(pref)
                for i in txt:
                    widths[i] = int(floor[i] + (pref[i] - floor[i]) * min(1.0, k))
            else:
                widths = [int(p * TEXT_W / total) for p in pref]
        widths[-1] += TEXT_W - sum(widths)
        return widths

    def _cell(self, text: str, col: int, row: int, width: int, height: int, bf: int, char: int, para: int,
              colspan: int = 1, rowspan: int = 1, header: bool = False) -> str:
        return (
            f'<hp:tc name="" header="{1 if header else 0}" hasMargin="0" protect="0" editable="0" dirty="0" borderFillIDRef="{bf}">'
            '<hp:subList id="" textDirection="HORIZONTAL" lineWrap="BREAK" vertAlign="CENTER" linkListIDRef="0" '
            'linkListNextIDRef="0" textWidth="0" textHeight="0" hasTextRef="0" hasNumRef="0">'
            f'{self._p(self._run(text, char), para)}</hp:subList>'
            f'<hp:cellAddr colAddr="{col}" rowAddr="{row}"/><hp:cellSpan colSpan="{colspan}" rowSpan="{rowspan}"/>'
            f'<hp:cellSz width="{width}" height="{height}"/><hp:cellMargin left="283" right="283" top="141" bottom="141"/></hp:tc>'
        )

    def table(self, rt: RenderedTable, social: bool = False) -> None:
        n = rt.n_cols
        if n == 0:
            return
        widths = self._col_widths(rt)
        h_head, h_body = 1300, 1150
        align_para = {"left": PARA["cell_left"], "right": PARA["cell_right"], "center": PARA["cell_center"]}
        rows_xml: List[str] = []
        n_head = len(rt.header_rows)
        for ri, hrow in enumerate(rt.header_rows):
            cells = []
            for hc in hrow:
                w = sum(widths[hc.col: hc.col + hc.colspan])
                bf = BF_HEAD if ri == 0 else BF_SUBHEAD
                if social or hc.colspan > 1:
                    para = PARA["cell_center"]
                else:
                    para = align_para.get(rt.aligns[hc.col], PARA["cell_left"])
                cells.append(self._cell(hc.text, hc.col, ri, w, h_head * hc.rowspan, bf, CHAR["cell_bold"], para,
                                        hc.colspan, hc.rowspan, header=True))
            rows_xml.append(f"<hp:tr>{''.join(cells)}</hp:tr>")
        rules = set(rt.rules_above)
        last = len(rt.body) - 1
        for bi, brow in enumerate(rt.body):
            r = n_head + bi
            if bi == last:
                bf = BF_RULE_LAST if bi in rules else BF_LAST
            else:
                bf = BF_RULE if bi in rules else BF_NONE
            cells = [self._cell(v, ci, r, widths[ci], h_body, bf, CHAR["cell"],
                                align_para.get(rt.aligns[ci], PARA["cell_left"]))
                     for ci, v in enumerate(brow)]
            rows_xml.append(f"<hp:tr>{''.join(cells)}</hp:tr>")
        row_cnt = n_head + len(rt.body)
        height = h_head * n_head + h_body * len(rt.body)
        self._z += 1
        tbl = (
            f'<hp:tbl id="{self._next_id()}" zOrder="{self._z}" numberingType="TABLE" textWrap="TOP_AND_BOTTOM" '
            'textFlow="BOTH_SIDES" lock="0" dropcapstyle="None" pageBreak="CELL" repeatHeader="1" '
            f'rowCnt="{row_cnt}" colCnt="{n}" cellSpacing="0" borderFillIDRef="{BF_PLAIN}" noAdjust="0">'
            f'<hp:sz width="{TEXT_W}" widthRelTo="ABSOLUTE" height="{height}" heightRelTo="ABSOLUTE" protect="0"/>'
            '<hp:pos treatAsChar="1" affectLSpacing="0" flowWithText="1" allowOverlap="0" holdAnchorAndSO="0" '
            'vertRelTo="PARA" horzRelTo="PARA" vertAlign="TOP" horzAlign="LEFT" vertOffset="0" horzOffset="0"/>'
            '<hp:outMargin left="0" right="0" top="0" bottom="0"/>'
            '<hp:inMargin left="283" right="283" top="141" bottom="141"/>'
            f'{"".join(rows_xml)}</hp:tbl>'
        )
        self.paras.append(self._p(f'<hp:run charPrIDRef="{CHAR["body"]}">{tbl}</hp:run>', PARA["cell_left"]))
        self.preview.append(" | ".join(rt.flat_headers))

    # ------------------------------------------------------------------ pictures
    def picture(self, png: bytes, max_width: int = 40000, max_height: int = 48000) -> None:
        seq = len(self.images) + 1
        item_id = f"image{seq}"
        self.images.append({"id": item_id, "href": f"BinData/{item_id}.png", "data": png})
        px_w, px_h = png_size(png)
        w = max_width
        h = int(w * px_h / max(1, px_w))
        if h > max_height:
            h = max_height
            w = int(h * px_w / max(1, px_h))
        dim_w, dim_h = px_w * 75, px_h * 75
        self._z += 1
        pic = (
            f'<hp:pic id="{self._next_id()}" zOrder="{self._z}" numberingType="PICTURE" textWrap="TOP_AND_BOTTOM" '
            f'textFlow="BOTH_SIDES" lock="0" dropcapstyle="None" href="" groupLevel="0" instid="{self._next_id()}" reverse="0">'
            f'<hp:offset x="0" y="0"/><hp:orgSz width="{w}" height="{h}"/><hp:curSz width="{w}" height="{h}"/>'
            '<hp:flip horizontal="0" vertical="0"/>'
            f'<hp:rotationInfo angle="0" centerX="{w // 2}" centerY="{h // 2}" rotateimage="1"/>'
            '<hp:renderingInfo><hc:transMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/>'
            '<hc:scaMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/>'
            '<hc:rotMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/></hp:renderingInfo>'
            f'<hc:img binaryItemIDRef="{item_id}" bright="0" contrast="0" effect="REAL_PIC" alpha="0"/>'
            f'<hp:imgRect><hc:pt0 x="0" y="0"/><hc:pt1 x="{w}" y="0"/><hc:pt2 x="{w}" y="{h}"/><hc:pt3 x="0" y="{h}"/></hp:imgRect>'
            f'<hp:imgClip left="0" right="{dim_w}" top="0" bottom="{dim_h}"/>'
            '<hp:inMargin left="0" right="0" top="0" bottom="0"/>'
            f'<hp:imgDim dimwidth="{dim_w}" dimheight="{dim_h}"/><hp:effects/>'
            f'<hp:sz width="{w}" widthRelTo="ABSOLUTE" height="{h}" heightRelTo="ABSOLUTE" protect="0"/>'
            '<hp:pos treatAsChar="1" affectLSpacing="0" flowWithText="1" allowOverlap="0" holdAnchorAndSO="0" '
            'vertRelTo="PARA" horzRelTo="COLUMN" vertAlign="TOP" horzAlign="CENTER" vertOffset="0" horzOffset="0"/>'
            '<hp:outMargin left="0" right="0" top="0" bottom="0"/></hp:pic>'
        )
        self.paras.append(self._p(f'<hp:run charPrIDRef="{CHAR["body"]}">{pic}</hp:run>', PARA["figure"]))

    # ------------------------------------------------------------------ package
    def section_xml(self) -> str:
        first = self._p(f'<hp:run charPrIDRef="{CHAR["body"]}">{_sec_pr()}</hp:run>' + self._run(self.title, CHAR["title"]),
                        PARA["heading"])
        return f'{XML_DECL}<hs:sec {NS}>{first}{"".join(self.paras)}</hs:sec>'

    def save(self, path: str) -> str:
        img_items = "".join(
            f'<opf:item id="{im["id"]}" href="{im["href"]}" media-type="image/png" isEmbeded="1"/>' for im in self.images
        )
        content_hpf = (
            f'{XML_DECL}<opf:package {NS} version="" unique-identifier="" id="">'
            f'<opf:metadata><opf:title>{_t(self.title)}</opf:title><opf:language>ko</opf:language>'
            '<opf:meta name="creator" content="text">OpenStat Studio</opf:meta></opf:metadata>'
            '<opf:manifest><opf:item id="header" href="Contents/header.xml" media-type="application/xml"/>'
            f'{img_items}<opf:item id="section0" href="Contents/section0.xml" media-type="application/xml"/>'
            '<opf:item id="settings" href="settings.xml" media-type="application/xml"/></opf:manifest>'
            '<opf:spine><opf:itemref idref="header" linear="yes"/><opf:itemref idref="section0" linear="yes"/></opf:spine>'
            '</opf:package>'
        )
        version_xml = (
            f'{XML_DECL}<hv:HCFVersion xmlns:hv="http://www.hancom.co.kr/hwpml/2011/version" '
            'tagetApplication="WORDPROCESSOR" major="5" minor="1" micro="1" buildNumber="0" os="1" xmlVersion="1.5" '
            'application="OpenStat Studio" appVersion="3.0"/>'
        )
        container_xml = (
            f'{XML_DECL}<ocf:container xmlns:ocf="urn:oasis:names:tc:opendocument:xmlns:container" '
            'xmlns:hpf="http://www.hancom.co.kr/schema/2011/hpf"><ocf:rootfiles>'
            '<ocf:rootfile full-path="Contents/content.hpf" media-type="application/hwpml-package+xml"/>'
            '<ocf:rootfile full-path="Preview/PrvText.txt" media-type="text/plain"/>'
            '</ocf:rootfiles></ocf:container>'
        )
        manifest_xml = f'{XML_DECL}<odf:manifest xmlns:odf="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"/>'
        settings_xml = (
            f'{XML_DECL}<ha:HWPApplicationSetting xmlns:ha="http://www.hancom.co.kr/hwpml/2011/app" '
            'xmlns:config="urn:oasis:names:tc:opendocument:xmlns:config:1.0">'
            '<ha:CaretPosition listIDRef="0" paraIDRef="0" pos="0"/></ha:HWPApplicationSetting>'
        )
        preview = "\r\n".join([self.title] + self.preview)[:1024]
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("mimetype", "application/hwp+zip", compress_type=zipfile.ZIP_STORED)
            zf.writestr("version.xml", version_xml)
            zf.writestr("Contents/header.xml", header_xml(self.font))
            zf.writestr("Contents/section0.xml", self.section_xml())
            zf.writestr("Preview/PrvText.txt", preview)
            zf.writestr("settings.xml", settings_xml)
            zf.writestr("META-INF/container.xml", container_xml)
            zf.writestr("META-INF/manifest.xml", manifest_xml)
            zf.writestr("Contents/content.hpf", content_hpf)
            for im in self.images:
                zf.writestr(im["href"], im["data"])
        return path

"""
OpenStat Text Import: turn text files into a dataset (one row per document).
Supported: .txt .md .csv-free plain text, .docx (python-docx), .pdf (pypdf), .hwpx (Hancom OWPML zip).
Legacy binary .hwp is not supported (save as .hwpx in Hangul first).

Columns of the resulting DataFrame: 문서ID, 파일명, 폴더, 형식, 글자수, 본문
"""

import io
import os
import re
import zipfile
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd

SUPPORTED_TEXT_EXTENSIONS = (".txt", ".md", ".docx", ".pdf", ".hwpx")
TEXT_COLUMNS = ["문서ID", "파일명", "폴더", "형식", "글자수", "본문"]


def _decode_plain(raw: bytes) -> str:
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _docx_text(raw: bytes) -> str:
    import docx
    d = docx.Document(io.BytesIO(raw))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for tbl in d.tables:
        for row in tbl.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _pdf_text(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise ValueError("PDF 읽기에 pypdf 패키지 필요: 'pip install pypdf'") from e
    reader = PdfReader(io.BytesIO(raw))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _hwpx_text(raw: bytes) -> str:
    import xml.etree.ElementTree as ET
    out: List[str] = []

    def walk(el, para, order, texts):
        # Each <hp:t> belongs to its nearest enclosing <hp:p>, so table-cell paragraphs are not duplicated.
        if el.tag.endswith("}p"):
            para = el
            order.append(el)
            texts[el] = []
        elif el.tag.endswith("}t") and para is not None:
            texts[para].append("".join(el.itertext()))
            return
        for ch in el:
            walk(ch, para, order, texts)

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = [n for n in zf.namelist() if re.match(r"Contents/section\d+\.xml$", n)]
        names.sort(key=lambda n: int(re.findall(r"\d+", n)[-1]))
        for name in names:
            order: List = []
            texts: dict = {}
            walk(ET.fromstring(zf.read(name)), None, order, texts)
            for p in order:
                line = "".join(texts[p]).strip()
                if line:
                    out.append(line)
    return "\n".join(out)


def extract_text(raw: bytes, filename: str) -> str:
    """Extract plain text from file bytes, dispatching on the file extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".txt", ".md"):
        return _decode_plain(raw)
    if ext == ".docx":
        return _docx_text(raw)
    if ext == ".pdf":
        return _pdf_text(raw)
    if ext == ".hwpx":
        return _hwpx_text(raw)
    if ext == ".hwp":
        raise ValueError("HWP(구형 형식) 미지원: 한글에서 HWPX로 저장 후 사용")
    raise ValueError(f"지원하지 않는 형식: {ext or '(확장자 없음)'}")


def texts_to_dataframe(items: Iterable[Tuple[str, bytes]]) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    items: (relative path, file bytes). Returns (dataset, skipped [(path, reason)]).
    The folder column holds the immediate parent folder, useful as a grouping variable.
    """
    rows = []
    skipped: List[Tuple[str, str]] = []
    for rel, raw in items:
        rel = (rel or "").replace("\\", "/").lstrip("/")
        base = os.path.basename(rel)
        if not base or base.startswith(("~$", ".")):
            continue
        ext = os.path.splitext(base)[1].lower()
        if ext not in SUPPORTED_TEXT_EXTENSIONS and ext != ".hwp":
            skipped.append((rel, f"지원하지 않는 형식 {ext or '(없음)'}"))
            continue
        try:
            text = extract_text(raw, base)
        except Exception as e:
            skipped.append((rel, str(e)[:120]))
            continue
        text = text.replace("\x00", "").strip()
        if not text:
            skipped.append((rel, "추출된 글자 없음 (스캔 PDF 등)"))
            continue
        parent = os.path.basename(os.path.dirname(rel)) or "(최상위)"
        rows.append({"파일명": base, "폴더": parent, "형식": ext.lstrip("."), "글자수": len(text), "본문": text, "_path": rel})
    df = pd.DataFrame(rows, columns=["파일명", "폴더", "형식", "글자수", "본문", "_path"])
    if not df.empty:
        df = df.sort_values("_path", kind="stable").reset_index(drop=True)
    df = df.drop(columns=["_path"])
    df.insert(0, "문서ID", range(1, len(df) + 1))
    return df, skipped


def load_texts(source: Union[str, List[str]], recursive: bool = True) -> pd.DataFrame:
    """
    Build a text dataset from a folder (recursively) or a list of file paths.

    >>> df = load_texts("reports/interviews")      # every .txt/.md/.docx/.pdf/.hwpx inside
    >>> sess = ost.Session(df)
    """
    paths: List[Tuple[str, str]] = []
    if isinstance(source, str) and os.path.isdir(source):
        root = os.path.abspath(source)
        walker = os.walk(root) if recursive else [(root, [], os.listdir(root))]
        for d, _, files in walker:
            for f in files:
                full = os.path.join(d, f)
                if os.path.isfile(full):
                    paths.append((os.path.relpath(full, os.path.dirname(root)), full))
    else:
        for p in ([source] if isinstance(source, str) else list(source)):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"파일 없음: {p}")
            paths.append((os.path.basename(p), p))

    def _items():
        for rel, full in paths:
            with open(full, "rb") as fh:
                yield rel, fh.read()

    df, skipped = texts_to_dataframe(_items())
    if df.empty:
        raise ValueError("불러온 문서 없음. 지원 형식: " + ", ".join(SUPPORTED_TEXT_EXTENSIONS)
                         + (f" (제외 {len(skipped)}건: {skipped[0][1]})" if skipped else ""))
    df.attrs["skipped"] = skipped
    return df

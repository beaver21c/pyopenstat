"""Cumulative command record of an OpenStat Studio session.

Every command from loading a data file onward (data steps, analyses, code runs, exports) is kept in one
runnable Python file, ``workspace/command_log.py``, together with a structured copy
(``workspace/command_log.json``) that :mod:`code_export` converts to Stata (.do) or R (.R) on request.

Re-running an analysis card while its options are edited replaces the card's last entry instead of adding
one line per click, so the record lists the final settings of each analysis in the order they were run.
"""
from __future__ import annotations

import json
import os
import pprint
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

from .syntax_engine import _clean_params_for_code, _format_op_call, _portable_path

LOG_MARKER = "# OpenStat Studio 명령 기록"

OP_TITLES = {
    "compute": "변수 계산", "recode": "재코딩", "filter_rows": "행 필터", "merge": "자료 결합", "joinby": "자료 결합(joinby)",
    "append": "자료 추가(append)", "reshape": "구조 변환", "collapse": "집계(collapse)", "egen": "집단 통계 변수(egen)",
    "convert_type": "변수 형식 변환", "encode": "문자→숫자 부호화(encode)", "decode": "숫자→문자(decode)",
    "set_value_labels": "값 라벨", "set_measure": "측정 수준", "standardize": "표준화", "rename_var": "변수 이름 변경",
    "drop_vars": "변수 삭제", "set_value": "셀 값 수정", "add_rows": "행 추가", "drop_rows": "행 삭제",
    "add_var": "변수 추가", "set_label": "변수 라벨", "panel_predict": "패널 예측값(predict)",
}

HEADER = '''{marker} — 파일 불러오기부터 실행한 모든 명령을 순서대로 기록한 파일
# 처음 기록: {started} · 마지막 기록: {updated} · 명령 {n}건
# 다시 실행: 저장소 폴더에서  python workspace/command_log.py   (또는 프로그램의 명령어 편집기 [▶ 전체 실행])
# Stata(.do)·R(.R) 코드는 [코드 내보내기]에서 변환하여 받음
import os
import sys

import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) if "__file__" in globals() else os.getcwd()
if os.path.join(_ROOT, "reconstructed") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "reconstructed"))
import openstat as ost


def _data(path):
    """Project-relative data path -> absolute path."""
    return path if os.path.isabs(path) else os.path.join(_ROOT, path)
'''


def _now() -> datetime:
    return datetime.now()


class CommandLog:
    def __init__(self, workspace_dir: str):
        self.dir = workspace_dir
        self.py_path = os.path.join(workspace_dir, "command_log.py")
        self.json_path = os.path.join(workspace_dir, "command_log.json")
        self.entries: List[Dict[str, Any]] = []
        self.started = _now().strftime("%Y-%m-%d %H:%M:%S")
        self.paused_code_logging = False
        self._load()

    # ------------------------------------------------------------------ storage
    def _load(self) -> None:
        try:
            with open(self.json_path, encoding="utf-8") as f:
                doc = json.load(f)
            self.entries = list(doc.get("entries") or [])
            self.started = doc.get("started") or self.started
        except (OSError, ValueError):
            self.entries = []

    def save(self) -> None:
        try:
            os.makedirs(self.dir, exist_ok=True)
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump({"started": self.started, "entries": self.entries}, f, ensure_ascii=False, indent=1, default=str)
            with open(self.py_path, "w", encoding="utf-8") as f:
                f.write(self.render_python())
        except OSError:
            pass

    def reset(self, archive: bool = True) -> Optional[str]:
        """Start a new record; the previous files are kept as command_log_YYYYmmdd_HHMMSS.py/.json."""
        kept = None
        if archive and self.entries:
            stamp = _now().strftime("%Y%m%d_%H%M%S")
            for src, ext in ((self.py_path, "py"), (self.json_path, "json")):
                if os.path.isfile(src):
                    dst = os.path.join(self.dir, f"command_log_{stamp}.{ext}")
                    shutil.copyfile(src, dst)
                    if ext == "py":
                        kept = dst
        self.entries = []
        self.started = _now().strftime("%Y-%m-%d %H:%M:%S")
        self.save()
        return kept

    # ------------------------------------------------------------------ recording
    def _add(self, kind: str, title: str, code: str, **meta: Any) -> Dict[str, Any]:
        e = {"kind": kind, "title": title, "code": code.rstrip(), "time": _now().strftime("%H:%M:%S"), **meta}
        last = self.entries[-1] if self.entries else None
        if (kind == "analysis" and last and last.get("kind") == "analysis"
                and last.get("aid") == meta.get("aid") and meta.get("aid") is not None):
            self.entries[-1] = e          # option edits of the same card: keep its final settings only
        else:
            self.entries.append(e)
        self.save()
        return e

    def load_dataset(self, path: str, name: Optional[str] = None) -> None:
        p = _portable_path(path)
        self._add("load", f"데이터 불러오기: {name or os.path.basename(str(p))}", f"sess = ost.Session(_data({p!r}))", path=p)

    def new_dataset(self, rows: int, cols: int) -> None:
        self._add("new", "빈 데이터셋", f"sess = ost.Session(ost.blank_dataset(rows={rows}, cols={cols})) if hasattr(ost, 'blank_dataset') else ost.Session()", rows=rows, cols=cols)

    def data_op(self, op: Dict[str, Any]) -> None:
        name = op.get("op", "")
        self._add("data", OP_TITLES.get(name, name), _format_op_call(op), op=json.loads(json.dumps(op, default=str)))

    def special(self, title: str, code: str, **meta: Any) -> None:
        """Data commands outside the op list (undo, filter removal, boundary link, ...)."""
        self._add("data_special", title, code, **meta)

    def analysis(self, aid: Optional[int], a_type: str, title: str, params: Dict[str, Any], syntax_r: Optional[str] = None) -> None:
        p = _clean_params_for_code(params or {})
        p.pop("plotStyle", None)
        lines = [f"res = sess.run({a_type!r},"]
        for k, v in p.items():
            lines.append(f"    {k}={pprint.pformat(v, width=96, sort_dicts=False)},")
        code = "\n".join(lines) + "\n)" if p else f"res = sess.run({a_type!r})"
        self._add("analysis", f"분석: {title}", code, aid=aid, analysis_type=a_type, params=json.loads(json.dumps(p, default=str)),
                  syntax_r=syntax_r)

    def code_run(self, code: str, whole: bool, source: Optional[str] = None) -> None:
        if LOG_MARKER in code or (source and LOG_MARKER in source):
            self._add("note", "명령 기록 전체 다시 실행", "# (위 명령 기록을 처음부터 다시 실행함)")
            return
        src = (source or "").strip()
        body = code.strip()
        if src and src != body:
            # typed as Stata commands: keep what was entered (comment) above the Python that ran
            body = "\n".join(["# 입력한 명령:"] + [f"#   {ln}" for ln in src.splitlines() if ln.strip()] + [body])
        self._add("code", "명령·코드 직접 실행" + (" (전체, 기존 결과 대체)" if whole else ""), body,
                  source=src, whole=bool(whole))

    def export(self, fmt: str, filename: str, report_style: Optional[str] = None) -> None:
        extra = f", report_style={report_style!r}" if report_style else ""
        self._add("export", f"결과 내보내기 (.{fmt})", f"sess.export_results(_data({('reports/' + filename)!r}){extra})", fmt=fmt, filename=filename)

    def note(self, title: str, text: str = "") -> None:
        self._add("note", title, "# " + (text or title))

    # ------------------------------------------------------------------ output
    def render_python(self) -> str:
        updated = self.entries[-1]["time"] if self.entries else "—"
        out = [HEADER.format(marker=LOG_MARKER, started=self.started, updated=updated, n=len(self.entries))]
        if not self.entries:
            out.append("\n# (아직 기록된 명령 없음)\n")
        for i, e in enumerate(self.entries, 1):
            out.append(f"\n# [{i}] {e.get('time', '')} {e.get('title', '')}\n{e.get('code', '')}\n")
        return "".join(out)

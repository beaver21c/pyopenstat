"""
OpenStat Studio — panel-data description (Stata xtdescribe, xtsum, xttab, xttrans).
License: MIT

Analysis id ``panelDescribe`` (aliases xtdescribe, xtsum, xttab, xttrans). Descriptive tables only; the
computations are pandas/numpy group operations defined as in the Stata [XT] manual entries:

    table                         definition (source)                                              Stata
    panel structure, T_i, patterns  n, distinct times, span, T_i percentiles (summarize, detail),   xtdescribe, patterns(#)
                                  participation patterns sorted by frequency then pattern
    overall / between / within    between = panel means xbar_i; within = x_it - xbar_i + xbar        xtsum
                                  ([XT] xtsum Methods and formulas)
    overall / between / within    between = panels ever observed with the value; within = mean of    xttab
    frequencies                   the share of T_i with the value over those panels ([XT] xttab)
    transition matrix             consecutive observations of a panel, both nonmissing, time gaps    xttrans [, freq]
                                  not filled ([XT] xttrans)

Verified against Stata 19.5 (xtdescribe, xtsum, xttab, xttrans on sample_data/Grunfeld_Panel.csv and its unbalanced
variant); reference values are in tests/test_panel_postestimation.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseAnalysis, ResultsTable
from .panel_module import _as_list, _as_single, _flag, _opt, _split_opts, _err, _xt_index_args, stata_percentile, \
    time_positions, xtset_lines


def _num(v) -> str:
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


class PanelDescribe(BaseAnalysis):
    """Panel structure and overall/between/within descriptive statistics."""

    def __init__(self, data: pd.DataFrame, entity: str, time: Optional[str] = None, sum_vars=None, tab_vars=None,
                 trans_vars=None, describe: bool = True, patterns: int = 9, trans_freq: bool = False, delta: float = 1.0):
        sum_vars, tab_vars, trans_vars = _as_list(sum_vars or []), _as_list(tab_vars or []), _as_list(trans_vars or [])
        if not entity:
            raise ValueError("패널 자료 요약: 개체 변수(entity) 지정 필요 (Stata 명령 입력 시 xtset 개체변수 시점변수 먼저 실행)")
        need_time = describe or bool(trans_vars)
        if need_time and not time:
            raise ValueError("패널 자료 요약: 패널 구조(xtdescribe)·전이표(xttrans)에는 시점 변수(time) 필요")
        cols = [entity] + ([time] if time else []) + sum_vars + tab_vars + trans_vars
        absent = [c for c in dict.fromkeys(cols) if c and c not in data.columns]
        if absent:
            raise ValueError(f"패널 자료 요약: 데이터에 없는 변수 — {', '.join(absent)}. 사용 가능한 변수: {list(data.columns)}")
        if not (describe or sum_vars or tab_vars or trans_vars):
            raise ValueError("패널 자료 요약: 패널 구조 표시 또는 요약할 변수(sumVars·tabVars·transVars) 1개 이상 지정 필요")
        try:
            patterns = int(patterns)
        except (TypeError, ValueError):
            raise ValueError("패턴 수(patterns)는 0 이상 정수")
        if patterns < 0:
            raise ValueError("패턴 수(patterns)는 0 이상 정수")
        super().__init__(
            name="panelDescribe",
            title="패널 자료 요약 (xtdescribe · xtsum · xttab · xttrans)",
            dataset=data,
            options={"entity": entity, "time": time or "", "sumVars": sum_vars, "tabVars": tab_vars,
                     "transVars": trans_vars, "describe": bool(describe), "patterns": patterns,
                     "transFreq": bool(trans_freq), "delta": float(delta or 1.0)},
        )
        self.syntax_python = self._py_syntax()
        self.syntax_r = "\n".join(r_code("panelDescribe", self.options) or [])
        self.run()

    def _py_syntax(self) -> str:
        o = self.options
        parts = [f"entity={o['entity']!r}"]
        if o["time"]:
            parts.append(f"time={o['time']!r}")
        for k in ("sumVars", "tabVars", "transVars"):
            if o[k]:
                parts.append(f"{k}={o[k]!r}")
        for k, dv in (("describe", True), ("patterns", 9), ("transFreq", False), ("delta", 1.0)):
            if o[k] != dv:
                parts.append(f"{k}={o[k]!r}")
        return f"res = sess.run('panelDescribe', {', '.join(parts)})"

    # ------------------------------------------------------------------ run
    def run(self):
        o = self.options
        df = self.dataset
        e = o["entity"]
        self._done = {}
        if o["describe"]:
            self._describe(df, e, o["time"], o["patterns"], o["delta"])
        if o["sumVars"]:
            self._xtsum(df, e, o["sumVars"])
        for v in o["tabVars"]:
            self._xttab(df, e, v)
        for v in o["transVars"]:
            self._xttrans(df, e, o["time"], v, o["transFreq"], o["delta"])
        bits = []
        if "describe" in self._done:
            bits.append(f"개체 {self._done['describe'][0]}개, 시점 {self._done['describe'][1]}개 패널")
        self.apa_narrative = (", ".join(bits) + " — 패널 구조와 집단 간·집단 내 변동 요약.") if bits else \
            "패널 자료의 전체·집단 간·집단 내 변동 요약."

    # ------------------------------------------------------------------ xtdescribe
    def _describe(self, df: pd.DataFrame, e: str, tm: str, n_pat: int, delta: float):
        sub = df[[e, tm]].dropna()
        if sub.empty:
            raise ValueError("패널 구조: 개체·시점 변수가 모두 있는 관측치 없음")
        tpos, tnote = time_positions(sub[tm], delta)
        sub = sub.assign(_tp=tpos)
        g, labels = pd.factorize(sub[e], sort=True)
        sub = sub.assign(_g=g)
        n = len(labels)
        times = np.unique(tpos)
        span = int(times.max() - times.min()) + 1
        dup = bool(sub.duplicated(["_g", "_tp"]).any())
        Ti = sub.drop_duplicates(["_g", "_tp"]).groupby("_g").size().to_numpy()
        tvals = sub.drop_duplicates("_tp").sort_values("_tp")[tm].tolist()
        t = ResultsTable(name="xtdescribe", title="패널 구조 (xtdescribe)")
        t.add_column("item", "항목", col_type="text")
        t.add_column("val", "값", col_type="text")
        first = [_num(v) for v in tvals[:2]]
        t.add_row({"item": f"개체 변수 {e}", "val": f"{n}개 (n)"})
        t.add_row({"item": f"시점 변수 {tm}", "val": f"{_num(tvals[0])}, {', '.join(first[1:])}, ..., {_num(tvals[-1])} — "
                                                  f"서로 다른 시점 {len(times)}개 (T)"})
        t.add_row({"item": f"Delta({tm})", "val": f"{delta:g} 단위"})
        t.add_row({"item": f"Span({tm})", "val": f"{span}개 기간"})
        balanced = (not dup) and np.all(Ti == len(times))
        t.add_row({"item": "균형 여부", "val": "강균형 (strongly balanced)" if balanced else
                   ("불균형 (unbalanced)" + (", 시점 공백 있음" if span > len(times) or not balanced else ""))})
        t.add_row({"item": "고유 식별", "val": f"{e}×{tm} 조합이 관측치를 고유하게 식별" if not dup else
                   f"{e}×{tm} 조합 중복 있음 (Stata: repeated time values within panel)"})
        if tnote:
            t.add_note(tnote)
        self.add_table(t)
        d = ResultsTable(name="xtdescribe_Ti", title="개체별 관측 시점 수 T_i 분포")
        for key, lab in (("min", "최소"), ("p5", "5%"), ("p25", "25%"), ("p50", "50%"), ("p75", "75%"), ("p95", "95%"),
                         ("max", "최대")):
            d.add_column(key, lab, col_type="number")
        d.add_row({"min": int(Ti.min()), "p5": stata_percentile(Ti, 5), "p25": stata_percentile(Ti, 25),
                   "p50": stata_percentile(Ti, 50), "p75": stata_percentile(Ti, 75), "p95": stata_percentile(Ti, 95),
                   "max": int(Ti.max())})
        d.add_note("백분위수: Stata summarize, detail 정의 (개체 단위)")
        self.add_table(d)
        if n_pat > 0:
            base = int(times.min())
            pats = []
            for gi in range(n):
                obs = set((sub.loc[sub["_g"] == gi, "_tp"].to_numpy() - base).astype(int).tolist())
                pats.append("".join("1" if k in obs else "." for k in range(span)))
            cnt = pd.Series(pats).value_counts()
            order = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
            p = ResultsTable(name="xtdescribe_patterns", title="참여 형태 (Participation patterns)")
            p.add_column("freq", "빈도", col_type="integer")
            p.add_column("pct", "백분율", col_type="number")
            p.add_column("cum", "누적", col_type="number")
            p.add_column("pattern", "형태", col_type="text")
            cum = 0
            for pat, c in order[:n_pat]:
                cum += c
                p.add_row({"freq": int(c), "pct": 100 * c / n, "cum": 100 * cum / n, "pattern": pat})
            rest = sum(c for _, c in order[n_pat:])
            if rest:
                p.add_row({"freq": int(rest), "pct": 100 * rest / n, "cum": 100.0, "pattern": "(그 밖의 형태)"})
            p.add_row({"freq": n, "pct": 100.0, "cum": None, "pattern": "X" * span})
            p.add_note("1 = 관측, . = 미관측; 빈도 순·같은 빈도는 형태 문자열 순 (Stata xtdescribe)")
            self.add_table(p)
        self._done["describe"] = (n, len(times))

    # ------------------------------------------------------------------ xtsum
    def _xtsum(self, df: pd.DataFrame, e: str, vars_: List[str]):
        t = ResultsTable(name="xtsum", title="전체·집단 간·집단 내 기술통계 (xtsum)")
        t.add_column("var", "변수", col_type="text")
        t.add_column("part", "구분", col_type="text")
        t.add_column("mean", "평균", col_type="number")
        t.add_column("sd", "표준편차", col_type="number")
        t.add_column("min", "최소", col_type="number")
        t.add_column("max", "최대", col_type="number")
        t.add_column("obs", "관측", col_type="text")
        for v in vars_:
            x = pd.to_numeric(df[v], errors="coerce")
            if x.notna().sum() < df[v].notna().sum():
                raise ValueError(f"xtsum: '{v}'는 숫자형 변수 필요 (문자 값 포함)")
            sub = pd.DataFrame({"g": df[e], "x": x}).dropna()
            if sub.empty:
                raise ValueError(f"xtsum: '{v}' 관측치 없음")
            xm = sub.groupby("g", sort=True)["x"].transform("mean")
            pm = sub.groupby("g", sort=True)["x"].mean()
            grand = float(sub["x"].mean())
            w = sub["x"] - xm + grand
            N, n = len(sub), len(pm)
            Tbar = N / n
            balanced = sub.groupby("g").size().nunique() == 1
            t.add_row({"var": v, "part": "전체 (overall)", "mean": grand, "sd": float(sub["x"].std(ddof=1)),
                       "min": float(sub["x"].min()), "max": float(sub["x"].max()), "obs": f"N = {N}"})
            t.add_row({"var": "", "part": "집단 간 (between)", "mean": None, "sd": float(pm.std(ddof=1)),
                       "min": float(pm.min()), "max": float(pm.max()), "obs": f"n = {n}"})
            t.add_row({"var": "", "part": "집단 내 (within)", "mean": None, "sd": float(w.std(ddof=1)),
                       "min": float(w.min()), "max": float(w.max()),
                       "obs": f"T = {int(Tbar)}" if balanced else f"T-bar = {Tbar:.6g}"})
        t.add_note("집단 간: 개체 평균 x̄_i의 분포; 집단 내: x_it − x̄_i + x̄ (전체 평균을 더해 원척도 유지) — Stata [XT] xtsum")
        t.add_note("변수별로 결측이 아닌 관측치 사용")
        self.add_table(t)

    # ------------------------------------------------------------------ xttab
    def _xttab(self, df: pd.DataFrame, e: str, v: str):
        sub = df[[e, v]].dropna()
        if sub.empty:
            raise ValueError(f"xttab: '{v}' 관측치 없음")
        Ti = sub.groupby(e).size()
        n = len(Ti)
        cnt = sub.groupby([v, e]).size().rename("c").reset_index()
        cnt["share"] = cnt["c"] / cnt[e].map(Ti)
        t = ResultsTable(name=f"xttab_{v}", title=f"전체·집단 간·집단 내 빈도 (xttab {v})")
        t.add_column("val", v, col_type="text")
        t.add_column("of", "전체 빈도", col_type="integer", super_title="전체 (Overall)")
        t.add_column("op", "전체 %", col_type="number", super_title="전체 (Overall)")
        t.add_column("bf", "집단 간 빈도", col_type="integer", super_title="집단 간 (Between)")
        t.add_column("bp", "집단 간 %", col_type="number", super_title="집단 간 (Between)")
        t.add_column("wp", "집단 내 %", col_type="number", super_title="집단 내 (Within)")
        tot_b = 0
        for val, grp in cnt.groupby(v, sort=True):
            of = int(grp["c"].sum())
            bf = int(len(grp))
            tot_b += bf
            t.add_row({"val": _num(val), "of": of, "op": 100 * of / len(sub), "bf": bf, "bp": 100 * bf / n,
                       "wp": 100 * float(grp["share"].mean())})
        t.add_row({"val": "합계", "of": len(sub), "op": 100.0, "bf": tot_b, "bp": 100 * tot_b / n, "wp": 100 * n / tot_b})
        t.add_note(f"n = {n}; 집단 간 빈도: 해당 값을 한 번이라도 가진 개체 수; 집단 내 %: 그 개체들에서 해당 값을 가진 시점 비율의 평균 "
                   "(Stata [XT] xttab); 합계의 집단 내 % = n / 집단 간 빈도 합")
        self.add_table(t)

    # ------------------------------------------------------------------ xttrans
    def _xttrans(self, df: pd.DataFrame, e: str, tm: str, v: str, freq: bool, delta: float):
        sub = df[[e, tm, v]].copy()
        sub = sub[sub[e].notna() & sub[tm].notna()]
        tpos, _ = time_positions(sub[tm], delta)
        sub = sub.assign(_tp=tpos).sort_values([e, "_tp"], kind="mergesort")
        same = sub[e].to_numpy()[1:] == sub[e].to_numpy()[:-1]
        a, b = sub[v].to_numpy()[:-1], sub[v].to_numpy()[1:]
        ok = same & pd.notna(a) & pd.notna(b)
        pairs = pd.DataFrame({"from": a[ok], "to": b[ok]})
        if pairs.empty:
            raise ValueError(f"xttrans: '{v}'의 연속 관측 쌍 없음")
        tab = pd.crosstab(pairs["from"], pairs["to"])
        levels = sorted(set(tab.index) | set(tab.columns))
        tab = tab.reindex(index=levels, columns=levels, fill_value=0)
        t = ResultsTable(name=f"xttrans_{v}", title=f"전이 확률 (xttrans {v}) — 행: t 시점, 열: 다음 관측 시점")
        t.add_column("from", v, col_type="text")
        keys = [f"c{i}" for i in range(len(levels))]
        for k, lv in zip(keys, levels):
            t.add_column(k, _num(lv), col_type="number")
        t.add_column("tot", "합계", col_type="number")
        for lv in levels:
            row = tab.loc[lv]
            s = float(row.sum())
            r = {"from": _num(lv), "tot": 100.0 if s else None}
            for k, lv2 in zip(keys, levels):
                r[k] = 100 * float(row[lv2]) / s if s else None
            t.add_row(r)
            if freq:
                rf = {"from": "  빈도", "tot": s}
                for k, lv2 in zip(keys, levels):
                    rf[k] = float(row[lv2])
                rf["_rowfmt"] = "integer"
                t.add_row(rf)
        colsum = tab.sum(axis=0)
        tot = float(colsum.sum())
        r = {"from": "합계", "tot": 100.0}
        for k, lv2 in zip(keys, levels):
            r[k] = 100 * float(colsum[lv2]) / tot
        t.add_row(r)
        if freq:
            rf = {"from": "  빈도", "tot": tot, "_rowfmt": "integer"}
            for k, lv2 in zip(keys, levels):
                rf[k] = float(colsum[lv2])
            t.add_row(rf)
        t.add_note(f"전이 {int(tot)}건: 같은 개체의 연속 관측치 쌍(시점 공백은 채우지 않음, 두 값 모두 결측 아님) — Stata [XT] xttrans")
        t.add_note("행 백분율; 합계 행은 도착 상태의 분포")
        self.add_table(t)


# ----------------------------------------------------------------------------- glue
def _kwargs(p: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        entity=_as_single(_opt(p, "entity", "idVar", "panelVar")),
        time=_as_single(_opt(p, "time", "timeVar")),
        sum_vars=_as_list(_opt(p, "sumVars", "sum_vars", "vars", default=[])),
        tab_vars=_as_list(_opt(p, "tabVars", "tab_vars", default=[])),
        trans_vars=_as_list(_opt(p, "transVars", "trans_vars", default=[])),
        describe=_flag(p, "describe", default=True),
        patterns=_opt(p, "patterns", default=9),
        trans_freq=_flag(p, "transFreq", "trans_freq", "freq", default=False),
        delta=float(_opt(p, "delta", default=1.0)),
    )


def dispatch(t: str, df: pd.DataFrame, params: Dict[str, Any]):
    if t != "panelDescribe":
        return None
    return PanelDescribe(df, **_kwargs(params or {}))


def stata_code(t: str, p: Dict[str, Any]) -> Optional[List[str]]:
    if t != "panelDescribe":
        return None
    kw = _kwargs(p)
    e, tm = kw["entity"], kw["time"]
    out = xtset_lines(e, tm, kw["delta"]) if tm else [f"xtset {e}"]
    if kw["describe"]:
        out.append("xtdescribe" + (f", patterns({int(kw['patterns'])})" if int(kw["patterns"]) != 9 else ""))
    if kw["sum_vars"]:
        out.append("xtsum " + " ".join(kw["sum_vars"]))
    out += [f"xttab {v}" for v in kw["tab_vars"]]
    out += [f"xttrans {v}" + (", freq" if kw["trans_freq"] else "") for v in kw["trans_vars"]]
    return out


def r_code(t: str, p: Dict[str, Any]) -> Optional[List[str]]:
    if t != "panelDescribe":
        return None
    kw = _kwargs(p)
    e, tm = kw["entity"], kw["time"]
    out = []
    if kw["describe"] and tm:
        out += [f"pdata <- plm::pdata.frame(data, index = c(\"{e}\", \"{tm}\"))", "plm::pdim(pdata)  # 균형 여부·n·T",
                f"table(table(data${e}))  # T_i 분포"]
    for v in kw["sum_vars"]:
        out += [f"# xtsum {v}: 전체·집단 간·집단 내",
                f"c(overall = sd(data${v}, na.rm = TRUE), between = sd(tapply(data${v}, data${e}, mean, na.rm = TRUE)), "
                f"within = sd(data${v} - ave(data${v}, data${e}, FUN = function(z) mean(z, na.rm = TRUE)), na.rm = TRUE))"]
    for v in kw["tab_vars"]:
        out += [f"table(data${v})  # xttab 전체 빈도", f"colSums(table(data${e}, data${v}) > 0)  # 집단 간 빈도"]
    for v in kw["trans_vars"]:
        out += [f"d <- data[order(data${e}, data${tm}), ]",
                f"nx <- ave(d${v}, d${e}, FUN = function(z) c(z[-1], NA))",
                f"prop.table(table(d${v}, nx), 1) * 100  # xttrans {v}"]
    return out


def transpile(line: str) -> Optional[str]:
    import re
    s = line.strip()
    m = re.match(r"^(xtdescribe|xtdes|xtsum|xttab|xttrans)\b\s*([^,]*?)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if not m:
        return None
    cmd = m.group(1).lower()
    vars_ = m.group(2).split()
    opts = _split_opts(m.group(3) or "")
    idx = _xt_index_args()
    if cmd in ("xtdescribe", "xtdes"):
        pat = opts.pop("patterns", None)
        if opts:
            return _err(f"xtdescribe 옵션 미지원: {', '.join(opts)} — " + s)
        extra = f", patterns={int(pat)}" if pat not in (None, "") else ""
        return f"res = sess.run('panelDescribe', {idx}{extra})  # [Stata] {s}"
    if not vars_:
        return _err(f"{cmd}: 변수 지정 필요 — " + s)
    if cmd == "xtsum":
        if opts:
            return _err(f"xtsum 옵션 미지원: {', '.join(opts)} — " + s)
        return f"res = sess.run('panelDescribe', {idx}, sumVars={vars_!r}, describe=False)  # [Stata] {s}"
    if cmd == "xttab":
        if opts or len(vars_) != 1:
            return _err("xttab: 변수 1개, 옵션 없음 — " + s)
        return f"res = sess.run('panelDescribe', {idx}, tabVars={vars_!r}, describe=False)  # [Stata] {s}"
    freq = "freq" in opts
    opts.pop("freq", None)
    if opts or len(vars_) != 1:
        return _err("xttrans: 변수 1개, freq 옵션만 지원 — " + s)
    return (f"res = sess.run('panelDescribe', {idx}, transVars={vars_!r}, describe=False"
            + (", transFreq=True" if freq else "") + f")  # [Stata] {s}")


EXTENSION = {
    "catalog": [{"id": "panelDescribe", "group": "Linear Models",
                 "title": "Panel Data Description (xtdescribe, xtsum, xttab, xttrans)",
                 "module": "openstat", "icon": "analysis-linearmodels.svg"}],
    "aliases": {"xtdescribe": "panelDescribe", "xtsum": "panelDescribe", "xttab": "panelDescribe",
                "xttrans": "panelDescribe", "panel_describe": "panelDescribe"},
    "dispatch": dispatch,
    "stata": stata_code,
    "r": r_code,
    "transpile": transpile,
}

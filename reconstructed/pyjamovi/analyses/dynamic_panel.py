"""
OpenStat Studio — linear dynamic panel GMM (Stata xtabond / xtdpdsys conventions).
License: MIT

Model:  y_it = sum_j a_j y_i,t-j + x_it b1 + w_it b2 + v_i + e_it

method="difference"  Arellano–Bond (1991) difference GMM                        [xtabond]
method="system"      Arellano–Bover (1995) / Blundell–Bond (1998) system GMM     [xtdpdsys]

Implemented from the published estimators (Arellano & Bond 1991; Blundell & Bond 1998; Windmeijer 2005; Stata [XT]
xtdpd Methods and formulas) with numpy / scipy.sparse — library evaluated and not used as the core: pydynpd 0.2.2 (MIT) follows
xtabond2 conventions and was compared with Stata 19.5 on abdata: it reproduces two-step WC-robust difference GMM, but
its one-step AR test differs (AR(1) -3.4929 vs Stata -3.5996), its one-step "Hansen" is the two-step statistic, it has
no conventional VCE / Sargan, its system GMM uses xtabond2 instrument sets (L1.n 0.9028 vs xtdpdsys 0.8235), and it
calls sys.exit() on input errors. linearmodels' IVGMM has no dynamic-panel instrument construction. pydynpd and
R plm::pgmm serve as cross-checks (tests/test_panel_extended.py).

Instrument sets follow Stata ([XT] xtabond, xtdpdsys, Methods and formulas in [XT] xtdpd):
  differenced equation   GMM-type  L(2/.).y, predetermined L(1/.).L^q.w, endogenous L(2/.).L^q.w
                         standard  D. of every strictly exogenous term (and time dummies)
  level equation         GMM-type  LD.y, predetermined L^q D.w, endogenous L^(q+1) D.w        (system only)
                         standard  _cons (the constant is estimated from the level errors; also in xtabond)
  One column per period and lag ("GMM-style"); ``collapse`` = xtabond2's collapsed form (checked with plm::pgmm).
  lagstruct(q, m): m further lags (pre L(1/m), endogenous L(2/m+1)); global maxlags(M): pre L(1/M), endogenous L(2/M).

Weighting: one-step H = [H_d 0; 0 0.5 I] with H_d = 1 on the diagonal and -0.5 next to it (consecutive periods);
two-step from one-step residuals. VCE: GMM (sigma^2 W1^-1 with sigma^2 = e*'e*/(N_diff - K); W2^-1), robust one-step
(Arellano–Bond), two-step WC-robust (Windmeijer 2005). AR(m) tests and Sargan statistics as in [XT] xtdpd
postestimation (robust and two-step forms with the residuals of the reported step).

Lags and differences use the time variable (gaps give missing values), not row order. Redundant instruments
(for example differenced time dummies) are dropped; Stata keeps them, which changes its two-step WC-robust standard
errors (checked: with the redundant dummy left out Stata reproduces these values).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats

from .base import BaseAnalysis, ResultsTable, resolve_choice
from .panel_module import (_as_list, _as_single, _flag, _opt, _split_opts, independent_columns, lag_index,
                           time_positions, _fmt_val, xtset_lines)

METHODS = ("difference", "system")
METHOD_ALIASES = {"diff": "difference", "ab": "difference", "arellano-bond": "difference", "xtabond": "difference",
                  "d": "difference", "sys": "system", "bb": "system", "blundell-bond": "system", "xtdpdsys": "system",
                  "ld": "system"}
VCES = ("gmm", "robust")
VCE_ALIASES = {"conventional": "gmm", "unadjusted": "gmm", "ols": "gmm", "wc": "robust", "windmeijer": "robust",
               "cluster": "robust", "clustered": "robust"}

_LAGSPEC = re.compile(r"^L(?:(\d+)|\((\d+)\s*/\s*(\d+)\))?\.(.+)$", re.IGNORECASE)


def parse_terms(spec: Any, default_lags: int = 0) -> List[Tuple[str, int]]:
    """'w', 'L.w', 'L2.w', 'L(0/2).w', 'L(0/2).(k ys)' -> [(var, lag), ...]; plain names get lags 0..default_lags."""
    out: List[Tuple[str, int]] = []
    items = spec if isinstance(spec, (list, tuple)) else [spec]
    toks: List[str] = []
    for it in items:
        if it is None or it == "":
            continue
        s = str(it).strip()
        # keep "L(0/2).(k ys)" together
        toks += re.findall(r"L\(\d+\s*/\s*\d+\)\.\([^)]*\)|L\d*\.\([^)]*\)|[^\s,]+", s)
    for tk in toks:
        m = _LAGSPEC.match(tk)
        if not m:
            if re.search(r"[.#()]", tk):
                raise ValueError(f"변수 표기 해석 불가: '{tk}' (지원: x, L.x, L2.x, L(0/2).x, L(0/2).(x z))")
            out += [(tk, l) for l in range(0, int(default_lags) + 1)]
            continue
        if m.group(2) is not None:
            lo, hi = int(m.group(2)), int(m.group(3))
            if hi < lo:
                raise ValueError(f"시차 범위 오류: '{tk}'")
            lags = list(range(lo, hi + 1))
        else:
            lags = [int(m.group(1)) if m.group(1) else 1]
        body = m.group(4).strip()
        vars_ = body[1:-1].split() if body.startswith("(") and body.endswith(")") else [body]
        for v in vars_:
            if re.search(r"[.#()]", v):
                raise ValueError(f"변수 표기 해석 불가: '{tk}'")
            out += [(v, l) for l in lags]
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def term_name(v: str, l: int) -> str:
    return v if l == 0 else f"L{l}.{v}"


def _parse_sets(spec: Any, lags: int, maxlags: Optional[int], label: str) -> List[Dict[str, Any]]:
    """pre/endog -> [{'vars': [...], 'lags': q, 'maxlags': m or None}] (Stata pre(varlist, lagstruct(q, m)))."""
    if spec is None or spec == "" or spec == []:
        return []
    if isinstance(spec, dict):
        spec = [spec]
    if isinstance(spec, (list, tuple)) and spec and all(isinstance(s, dict) for s in spec):
        out = []
        for s in spec:
            vs = _as_list(s.get("vars") or s.get("varlist"))
            if not vs:
                raise ValueError(f"{label}: vars 지정 필요")
            ml = s.get("maxlags", s.get("maxLags"))
            out.append({"vars": vs, "lags": int(s.get("lags", 0) or 0), "maxlags": None if ml in (None, "", ".") else int(ml)})
        return out
    vs = _as_list(spec)
    return [{"vars": vs, "lags": int(lags or 0), "maxlags": maxlags}] if vs else []


class DynData:
    """All rows (no listwise deletion: lags may use rows that are not themselves in the estimation sample)."""

    def __init__(self, df: pd.DataFrame, entity: str, time: str, needed: Sequence[str], delta: float = 1.0):
        absent = [c for c in dict.fromkeys([entity, time] + list(needed)) if c and c not in df.columns]
        if absent:
            raise ValueError(f"동적 패널 GMM: 데이터에 없는 변수 — {', '.join(absent)}. 사용 가능한 변수: {list(df.columns)}")
        sub = df[list(dict.fromkeys([entity, time] + list(needed)))].copy()
        sub = sub[sub[entity].notna() & sub[time].notna()]
        for c in needed:
            if not pd.api.types.is_numeric_dtype(sub[c]):
                conv = pd.to_numeric(sub[c], errors="coerce")
                if conv.notna().sum() < sub[c].notna().sum():
                    raise ValueError(f"동적 패널 GMM: '{c}'는 숫자형 변수 필요")
                sub[c] = conv
        tpos, self.time_note = time_positions(sub[time], delta)
        sub["_tpos"] = tpos
        if sub.duplicated([entity, "_tpos"]).any():
            raise ValueError("동적 패널 GMM: 같은 개체·시점 관측치 중복 (Stata: repeated time values within panel)")
        sub = sub.sort_values([entity, "_tpos"], kind="mergesort").reset_index(drop=True)
        self.df = sub
        self.g, self.labels = pd.factorize(sub[entity], sort=True)
        self.tpos = sub["_tpos"].to_numpy(np.int64)
        self.R = len(sub)
        self.time_values = sub[time].to_numpy()
        self._lag_cache: Dict[int, np.ndarray] = {}
        self.entity, self.time = entity, time

    def lag_rows(self, l: int) -> np.ndarray:
        if l not in self._lag_cache:
            self._lag_cache[l] = np.arange(self.R) if l == 0 else lag_index(self.g, self.tpos, l)
        return self._lag_cache[l]

    def lagged(self, v: str, l: int) -> np.ndarray:
        base = self.df[v].to_numpy(float)
        if l == 0:
            return base
        idx = self.lag_rows(l)
        out = np.full(self.R, np.nan)
        ok = idx >= 0
        out[ok] = base[idx[ok]]
        return out

    def time_label(self, tp: int) -> str:
        i = np.where(self.tpos == tp)[0]
        return _fmt_val(self.time_values[i[0]]) if len(i) else str(tp)


class DynamicPanelGMM(BaseAnalysis):
    """Arellano–Bond difference / Blundell–Bond system GMM."""

    def __init__(self, data: pd.DataFrame, entity: str, time: str, dep: str, exog: Any = None,
                 method: str = "difference", lags: int = 1, maxldep: Optional[int] = None,
                 pre: Any = None, pre_lags: int = 0, pre_maxlags: Optional[int] = None,
                 endog: Any = None, endog_lags: int = 0, endog_maxlags: Optional[int] = None,
                 maxlags: Optional[int] = None, exog_lags: int = 0, time_dummies: bool = False,
                 twostep: bool = False, vce: str = "gmm", collapse: bool = False, constant: bool = True,
                 artests: int = 2, delta: float = 1.0):
        meth = resolve_choice("추정 방법(method)", method, METHODS, METHOD_ALIASES, default="difference")
        vc = resolve_choice("표준오차(vce)", vce, VCES, VCE_ALIASES, default="gmm")
        if not entity or not time or not dep:
            miss = [n for n, v in (("개체 변수(entity)", entity), ("시점 변수(time)", time), ("종속변수(dep)", dep)) if not v]
            raise ValueError(f"동적 패널 GMM: 지정 필요 — {', '.join(miss)} (Stata 명령 입력 시 xtset 먼저 실행). "
                             f"사용 가능한 변수: {list(data.columns)}")
        lags = int(lags)
        if lags < 1:
            raise ValueError("종속변수 시차 수(lags)는 1 이상")
        maxldep = None if maxldep in (None, "", ".") else int(maxldep)
        if maxldep is not None and maxldep < 1:
            raise ValueError("maxldep은 1 이상")
        maxlags = None if maxlags in (None, "", ".") else int(maxlags)
        pre_maxlags = None if pre_maxlags in (None, "", ".") else int(pre_maxlags)
        endog_maxlags = None if endog_maxlags in (None, "", ".") else int(endog_maxlags)
        artests = int(artests)
        if artests < 1:
            raise ValueError("artests는 1 이상")
        exog_terms = parse_terms(exog, exog_lags)
        pre_sets = _parse_sets(pre, pre_lags, pre_maxlags, "선결변수(pre)")
        end_sets = _parse_sets(endog, endog_lags, endog_maxlags, "내생변수(endog)")
        # Stata: lagstruct(q, m) gives m further lags (pre L(1/m).L^q.w, endogenous L(2/m+1).L^q.w);
        # the global maxlags(M) gives pre L(1/M) and endogenous L(2/M) (M >= 2) for sets without their own m
        if maxlags is not None:
            if maxlags < 1:
                raise ValueError("maxlags는 1 이상")
            for st in pre_sets:
                if st["maxlags"] is None:
                    st["maxlags"] = maxlags
            for st in end_sets:
                if st["maxlags"] is None:
                    if maxlags < 2:
                        raise ValueError("maxlags: 내생변수 도구변수는 2기 시차부터이므로 2 이상 필요 (Stata: lag(2 #) invalid)")
                    st["maxlags"] = maxlags - 1
        for st in pre_sets + end_sets:
            if st["maxlags"] is not None and st["maxlags"] < 1:
                raise ValueError("선결·내생변수 최대 시차 수는 1 이상")
        allv = [v for v, _ in exog_terms] + [v for s in pre_sets + end_sets for v in s["vars"]]
        if dep in allv:
            raise ValueError(f"동적 패널 GMM: 종속변수 '{dep}'를 설명변수로 지정할 수 없음 (시차는 lags 옵션)")
        dup = {v for v in allv if [x for s in pre_sets + end_sets for x in s["vars"]].count(v) > 1
               or (v in [x for s in pre_sets + end_sets for x in s["vars"]] and v in [w for w, _ in exog_terms])}
        if dup:
            raise ValueError(f"동적 패널 GMM: 같은 변수가 외생·선결·내생에 중복 지정 — {', '.join(sorted(dup))}")
        title = ("동적 패널 GMM — 차분 GMM (xtabond)" if meth == "difference" else "동적 패널 GMM — 시스템 GMM (xtdpdsys)")
        super().__init__(name="dynPanel", title=title, dataset=data, options={
            "entity": entity, "time": time, "dep": dep, "exog": [term_name(v, l) for v, l in exog_terms],
            "method": meth, "lags": lags, "maxldep": maxldep, "pre": pre_sets, "endog": end_sets,
            "timeDummies": bool(time_dummies), "twostep": bool(twostep), "vce": vc, "collapse": bool(collapse),
            "constant": bool(constant), "artests": artests, "delta": float(delta or 1.0),
        })
        self.exog_terms, self.pre_sets, self.end_sets = exog_terms, pre_sets, end_sets
        self.syntax_python = self._py_syntax()
        self.syntax_r = "\n".join(r_code("dynPanel", self.options) or [])
        self.run()

    # ------------------------------------------------------------------ syntax
    def _py_syntax(self) -> str:
        o = self.options
        parts = [f"entity={o['entity']!r}", f"time={o['time']!r}", f"dep={o['dep']!r}", f"exog={o['exog']!r}",
                 f"method={o['method']!r}", f"lags={o['lags']}"]
        if o["pre"]:
            parts.append(f"pre={o['pre']!r}")
        if o["endog"]:
            parts.append(f"endog={o['endog']!r}")
        for k, dv in (("maxldep", None), ("timeDummies", False), ("twostep", False), ("vce", "gmm"), ("collapse", False),
                      ("constant", True), ("artests", 2), ("delta", 1.0)):
            if o[k] != dv:
                parts.append(f"{k}={o[k]!r}")
        return f"res = sess.run('dynPanel', {', '.join(parts)})"

    # ------------------------------------------------------------------ build design
    def run(self):
        o = self.options
        dep, p = o["dep"], o["lags"]
        needed = [dep] + [v for v, _ in self.exog_terms] + [v for s in self.pre_sets + self.end_sets for v in s["vars"]]
        D = DynData(self.dataset, o["entity"], o["time"], list(dict.fromkeys(needed)), o["delta"])
        self.data_ = D
        system = o["method"] == "system"
        has_cons = o["constant"]
        # regressor terms (var, lag, kind)
        # Stata's coefficient order: lagged y, predetermined, endogenous, strictly exogenous (then time dummies)
        terms: List[Tuple[str, int, str]] = [(dep, j, "y") for j in range(1, p + 1)]
        for s in self.pre_sets:
            terms += [(v, l, "pre") for v in s["vars"] for l in range(0, s["lags"] + 1)]
        for s in self.end_sets:
            terms += [(v, l, "endog") for v in s["vars"] for l in range(0, s["lags"] + 1)]
        terms += [(v, l, "exog") for v, l in self.exog_terms]
        names = [term_name(v, l) for v, l, _ in terms]
        XL = np.column_stack([D.lagged(v, l) for v, l, _ in terms])
        XP = np.column_stack([D.lagged(v, l + 1) for v, l, _ in terms])
        yL = D.lagged(dep, 0)
        yP = D.lagged(dep, 1)
        dummy_names: List[str] = []
        if o["timeDummies"]:
            uniq = np.unique(D.tpos)
            prev = D.lag_rows(1)
            for u in uniq[1:]:
                cur = (D.tpos == u).astype(float)
                XL = np.column_stack([XL, cur])          # dummy at t-1 (XP) is 1 when the previous period equals u
                XP = np.column_stack([XP, np.where(prev >= 0, (D.tpos[np.maximum(prev, 0)] == u).astype(float), np.nan)])
                dummy_names.append(f"{D.time_label(int(u))}.{D.time}")
        all_names = names + dummy_names
        n_terms = len(all_names)
        dX = XL - XP
        dy = yL - yP
        diff_rows = np.where(np.isfinite(dy) & np.all(np.isfinite(dX), axis=1))[0]
        lev_ok = np.isfinite(yL) & np.all(np.isfinite(XL), axis=1)
        if system:
            lev_rows = np.where(lev_ok)[0]
        elif has_cons:
            lev_rows = np.where(lev_ok)[0]
        else:
            lev_rows = np.zeros(0, np.int64)
        if len(diff_rows) == 0:
            raise ValueError("동적 패널 GMM: 차분식에 쓸 관측치가 없음 (개체별 연속 시점 수가 시차 수 + 2 이상 필요)")
        self._build_and_estimate(D, terms, all_names, n_terms, dX, dy, XL, yL, diff_rows, lev_rows, system, has_cons)

    def _gmm_blocks(self, D: DynData, terms):
        """List of GMM-type instrument specs: (eq, var, lag_lo, lag_hi, label)."""
        o = self.options
        dep, p = o["dep"], o["lags"]
        specs = []
        lo_y = int(getattr(self, "_y_lag_lo", 2))        # internal: xtdpd dgmmiv(y, lag(# .)) for verification
        hi = None if o["maxldep"] is None else o["maxldep"] + lo_y - 1
        specs.append(("d", dep, lo_y, hi, f"L({lo_y}/{'.' if hi is None else hi}).{dep}"))
        for s in self.pre_sets:
            q = s["lags"]
            for v in s["vars"]:
                hi_ = None if s["maxlags"] is None else q + s["maxlags"]
                inner = v if q == 0 else f"L{q}.{v}"
                specs.append(("d", v, q + 1, hi_, f"L(1/{'.' if s['maxlags'] is None else s['maxlags']}).{inner}"))
        for s in self.end_sets:
            q = s["lags"]
            for v in s["vars"]:
                hi_ = None if s["maxlags"] is None else q + 1 + s["maxlags"]
                inner = v if q == 0 else f"L{q}.{v}"
                specs.append(("d", v, q + 2, hi_, f"L(2/{'.' if s['maxlags'] is None else s['maxlags'] + 1}).{inner}"))
        if o["method"] == "system":
            ql = lo_y - 1
            specs.append(("l", dep, ql, ql, ("LD." if ql == 1 else f"L{ql}D.") + dep))
            for s in self.pre_sets:
                for v in s["vars"]:
                    q = s["lags"]
                    specs.append(("l", v, q, q, ("D." if q == 0 else ("LD." if q == 1 else f"L{q}D.")) + v))
            for s in self.end_sets:
                for v in s["vars"]:
                    q = s["lags"] + 1
                    specs.append(("l", v, q, q, ("LD." if q == 1 else f"L{q}D.") + v))
        return specs

    def _build_and_estimate(self, D, terms, names, K0, dX, dy, XL, yL, diff_rows, lev_rows, system, has_cons):
        o = self.options
        collapse = o["collapse"]
        Rd, Rl = len(diff_rows), len(lev_rows)
        # ---- stacked regressors
        Xd = dX[diff_rows]
        Xl = XL[lev_rows] if Rl else np.zeros((0, K0))
        X = np.vstack([Xd, Xl])
        yv = np.concatenate([dy[diff_rows], yL[lev_rows] if Rl else np.zeros(0)])
        if has_cons:
            X = np.column_stack([X, np.concatenate([np.zeros(Rd), np.ones(Rl)])])
            names = names + ["Intercept"]
        n_dum = K0 - len(terms)
        order = ([X.shape[1] - 1] if has_cons else []) + list(range(len(terms))) + list(range(K0 - 1, len(terms) - 1, -1))
        keep = independent_columns(X, protect=order)
        omitted = [names[j] for j in range(len(names)) if j not in keep]
        X = X[:, keep]
        names = [names[j] for j in keep]
        K = X.shape[1]
        g_st = np.concatenate([D.g[diff_rows], D.g[lev_rows] if Rl else np.zeros(0, np.int64)]).astype(np.int64)
        t_st = np.concatenate([D.tpos[diff_rows], D.tpos[lev_rows] if Rl else np.zeros(0, np.int64)])
        is_d = np.concatenate([np.ones(Rd, bool), np.zeros(Rl, bool)])
        # ---- instruments
        cols, rows_, vals = [], [], []
        col_info: List[Dict[str, Any]] = []
        ncol = 0
        tmin = int(D.tpos.min())
        summary_rows: List[Dict[str, Any]] = []
        for eq, v, lo, hi, label in self._gmm_blocks(D, terms):
            r_idx = diff_rows if eq == "d" else lev_rows
            if len(r_idx) == 0:
                continue
            off = 0 if eq == "d" else Rd
            tp = D.tpos[r_idx]
            start = ncol
            if eq == "d":
                max_l = int(tp.max() - tmin)
                hi_ = max_l if hi is None else min(hi, max_l)
                lag_list = list(range(lo, hi_ + 1))
                key_cols: Dict[Any, int] = {}
                for a in lag_list:
                    val = D.lagged(v, a)[r_idx]
                    ok = np.isfinite(val) & (val != 0)
                    if not ok.any():
                        continue
                    if collapse:
                        keys = np.full(len(r_idx), a)
                    else:
                        keys = tp * 1000 + a
                    for kk in np.unique(keys[ok]):
                        if kk not in key_cols:
                            key_cols[kk] = ncol
                            ncol += 1
                    cc = np.array([key_cols[kk] for kk in keys[ok]])
                    rows_.append(off + np.where(ok)[0])
                    cols.append(cc)
                    vals.append(val[ok])
            else:
                val = D.lagged(v, lo) - D.lagged(v, lo + 1)
                val = val[r_idx]
                ok = np.isfinite(val) & (val != 0)
                if ok.any():
                    keys = np.zeros(len(r_idx), np.int64) if collapse else tp
                    key_cols = {}
                    for kk in np.unique(keys[ok]):
                        key_cols[kk] = ncol
                        ncol += 1
                    rows_.append(off + np.where(ok)[0])
                    cols.append(np.array([key_cols[kk] for kk in keys[ok]]))
                    vals.append(val[ok])
            summary_rows.append({"eq": eq, "type": "GMM", "label": label, "start": start, "end": ncol})
        # standard instruments for the differenced equation: D. of strictly exogenous terms and time dummies
        excl = set(getattr(self, "_div_exclude", ()))   # internal: xtdpd div() subsets for verification
        exog_idx = [j for j, (v, l, kind) in enumerate(terms) if kind == "exog" and term_name(v, l) not in excl] \
            + list(range(len(terms), K0))
        std_cols = []
        if exog_idx:
            Zs = dX[diff_rows][:, exog_idx]
            n_ex = len(exog_idx) - (K0 - len(terms))
            ks = independent_columns(Zs, protect=list(range(n_ex)) + list(range(len(exog_idx) - 1, n_ex - 1, -1)))
            dropped_iv = [j for j in range(len(exog_idx)) if j not in ks]
            iv_names = [("D." + (term_name(terms[j][0], terms[j][1]) if j < len(terms) else self._dummy_name(j - len(terms))))
                        for j in exog_idx]
            self._iv_omitted = [iv_names[j] for j in dropped_iv]
            start = ncol
            for jj in ks:
                col = Zs[:, jj]
                nz = np.where(col != 0)[0]
                rows_.append(nz)
                cols.append(np.full(len(nz), ncol))
                vals.append(col[nz])
                std_cols.append(iv_names[jj])
                ncol += 1
            if ks:
                summary_rows.append({"eq": "d", "type": "표준", "label": " ".join(std_cols), "start": start, "end": ncol})
        else:
            self._iv_omitted = []
        if has_cons and Rl:
            rows_.append(Rd + np.arange(Rl))
            cols.append(np.full(Rl, ncol))
            vals.append(np.ones(Rl))
            summary_rows.append({"eq": "l", "type": "표준", "label": "_cons", "start": ncol, "end": ncol + 1})
            ncol += 1
        Rtot = Rd + Rl
        Z = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows_), np.concatenate(cols))), shape=(Rtot, ncol))
        self._estimate(D, X, yv, Z, names, omitted, g_st, t_st, is_d, diff_rows, lev_rows, summary_rows, system, has_cons)

    def _dummy_name(self, j: int) -> str:
        D = self.data_
        uniq = np.unique(D.tpos)
        return f"{D.time_label(int(uniq[1:][j]))}.{D.time}"

    # ------------------------------------------------------------------ GMM
    @staticmethod
    def _sym_inv(M: np.ndarray) -> np.ndarray:
        """Inverse of a symmetric p.s.d. matrix: Cholesky-based solve when nonsingular, else Moore–Penrose."""
        try:
            c = np.linalg.cholesky(M)
            ci = np.linalg.solve(c, np.eye(M.shape[0]))
            out = ci.T @ ci
            if np.all(np.isfinite(out)):
                return out
        except np.linalg.LinAlgError:
            pass
        return np.linalg.pinv(M, hermitian=True)

    def _estimate(self, D, X, y, Z, names, omitted, g, t, is_d, diff_rows, lev_rows, summary_rows, system, has_cons):
        o = self.options
        R, K = X.shape
        L = Z.shape[1]
        n_ind = int(D.g.max()) + 1
        # column equilibration (estimates, tests and VCE are invariant; avoids loss of precision when a regressor
        # such as a calendar-year trend sits next to the level-equation constant)
        sx = np.sqrt(np.sum(X ** 2, axis=0))
        sx[sx == 0] = 1.0
        X = X / sx
        sz = np.sqrt(np.asarray(Z.multiply(Z).sum(axis=0)).ravel())
        sz[sz == 0] = 1.0
        Z = (Z @ sparse.diags(1.0 / sz)).tocsr()
        # H1 = [H_d 0; 0 0.5 I]; H_d: 1 on diagonal, -0.5 for consecutive periods of the same panel
        Zd = Z[np.where(is_d)[0]]
        Zl = Z[np.where(~is_d)[0]]
        gd, td = g[is_d], t[is_d]
        nxt = np.where((gd[1:] == gd[:-1]) & (td[1:] - td[:-1] == 1))[0]
        ZHZ = (Zd.T @ Zd).toarray()
        if len(nxt):
            C = (Zd[nxt].T @ Zd[nxt + 1]).toarray()
            ZHZ -= 0.5 * (C + C.T)
        if Zl.shape[0]:
            ZHZ += 0.5 * (Zl.T @ Zl).toarray()
        # drop all-zero instrument columns and collinear ones (generalized inverse as Stata)
        zrank = int(np.linalg.matrix_rank(ZHZ, tol=None))
        A1 = np.linalg.pinv(ZHZ, hermitian=True)
        ZX = (Z.T @ X)
        ZX = np.asarray(ZX)
        Zy = np.asarray(Z.T @ y).ravel()
        W1 = ZX.T @ A1 @ ZX
        W1i = self._sym_inv(W1)
        b1 = W1i @ (ZX.T @ A1 @ Zy)
        e1 = y - X @ b1
        ed1 = e1[is_d]
        Nd = int(is_d.sum())
        sig2_1 = float(ed1 @ ed1) / (Nd - K)      # Stata: differenced residuals, N = differenced observations
        # per-panel moment contributions
        Ze1 = self._panel_sum(Z, e1, g, n_ind)                     # n x L
        Omega1 = Ze1.T @ Ze1                                        # sum_i Z_i' e_i e_i' Z_i
        A2 = np.linalg.pinv(Omega1, hermitian=True)
        V1_gmm = sig2_1 * W1i
        V1_rob = W1i @ ZX.T @ A1 @ Omega1 @ A1 @ ZX @ W1i
        res = {"b1": b1, "e1": e1, "sig2_1": sig2_1, "A1": A1, "A2": A2, "W1i": W1i}
        if o["twostep"]:
            W2 = ZX.T @ A2 @ ZX
            W2i = self._sym_inv(W2)
            b2 = W2i @ (ZX.T @ A2 @ Zy)
            e2 = y - X @ b2
            V2 = W2i
            if o["vce"] == "robust":
                Ze2 = Z.T @ e2
                Dm = np.zeros((K, K))
                for k in range(K):
                    Gk = self._panel_sum(Z, X[:, k], g, n_ind)      # sum_i Z_i' x_ik
                    dOm = Gk.T @ Ze1 + Ze1.T @ Gk                   # sum_i Z_i'(x_ik e_i' + e_i x_ik')Z_i
                    Dm[:, k] = W2i @ ZX.T @ A2 @ dOm @ A2 @ np.asarray(Ze2).ravel()
                V = V2 + Dm @ V2 + V2 @ Dm.T + Dm @ V1_rob @ Dm.T
                vlab = "WC-강건 (Windmeijer 2005)"
            else:
                V = V2
                vlab = "GMM (2단계, 하향 편의 가능)"
            b, e, A, Wi = b2, e2, A2, W2i
        else:
            b, e, A, Wi = b1, e1, A1, W1i
            V = V1_rob if o["vce"] == "robust" else V1_gmm
            vlab = "강건 (Arellano–Bond 1단계)" if o["vce"] == "robust" else "GMM (1단계)"
        V = (V + V.T) / 2
        V_scaled = V
        b = b / sx                          # back to the original units
        V = V / np.outer(sx, sx)
        res["b1"] = res["b1"] / sx
        self.b, self.V, self.names, self.omitted = b, V, names, omitted
        self.res_ = res
        se = np.sqrt(np.clip(np.diag(V), 0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0, b / se, np.nan)
        pz = 2 * stats.norm.sf(np.abs(z))
        crit = stats.norm.ppf(0.975)
        # ---- tables
        n_lev = int((~is_d).sum())
        rep_rows = lev_rows if system else diff_rows
        Ti = np.bincount(D.g[rep_rows], minlength=n_ind)
        Ti = Ti[Ti > 0]
        slope_idx = [i for i, nm in enumerate(names) if nm != "Intercept"]
        wald = None
        if slope_idx:
            bs, Vs = b[slope_idx], V[np.ix_(slope_idx, slope_idx)]
            W = float(bs @ np.linalg.pinv(Vs) @ bs)
            wald = (W, len(slope_idx), float(stats.chi2.sf(W, len(slope_idx))))
        self._summary(D, len(rep_rows), Nd, n_lev, Ti, L, zrank, wald, vlab, system)
        self._coef_table(names, b, se, z, pz, crit, omitted, vlab)
        self._instrument_table(summary_rows, Z, system)
        # AR tests
        self._ar_tests(D, X, Z, e, res, A, Wi, V_scaled, g, t, is_d, n_ind, K)   # scaled X, Z, V (invariant)
        self._overid(Z, e, res, A1, A2, zrank, K, g, n_ind)
        stats_ = {"N": int(len(rep_rows)), "Groups": int(len(Ti))}
        if wald:
            stats_["Chi2"] = (wald[0], wald[2])
        self.model_stats = stats_
        meth = "차분 GMM (Arellano–Bond)" if not system else "시스템 GMM (Blundell–Bond)"
        step = "2단계" if o["twostep"] else "1단계"
        txt = (f"{meth} {step} 추정 (개체 {len(Ti)}개, 관측치 {len(rep_rows)}개, 도구변수 {zrank}개)")
        if wald:
            pv = "< .001" if wald[2] < 0.001 else f"= {wald[2]:.3f}"
            txt += f", Wald χ²({wald[1]}) = {wald[0]:.2f}, p {pv}"
        ar = self.get_table("ar")
        if ar is not None:
            for r in ar.rows:
                if r.get("order") == 2 and r.get("z") is not None:
                    txt += f"; AR(2) z = {r['z']:.2f}, p = {r['p']:.3f}"
        self.apa_narrative = txt + "."

    @staticmethod
    def _panel_sum(Z, v: np.ndarray, g: np.ndarray, n: int) -> np.ndarray:
        """Rows: sum over rows of panel i of Z_r' v_r  (n x L)."""
        S = sparse.csr_matrix((v, (g, np.arange(len(v)))), shape=(n, len(v)))
        return np.asarray((S @ Z).todense())

    def _summary(self, D, N, Nd, Nl, Ti, L, zrank, wald, vlab, system):
        o = self.options
        t = ResultsTable(name="summary", title=f"모형 요약 ({'xtdpdsys' if system else 'xtabond'})")
        t.add_column("metric", "항목", col_type="text")
        t.add_column("val", "값", col_type="number")
        rows = [("관측치 수 (N)", N, "integer")]
        if system:
            rows += [("  차분식 관측치", Nd, "integer"), ("  수준식 관측치", Nl, "integer")]
        rows += [("집단 수", len(Ti), "integer"), ("집단당 관측치 최소", int(Ti.min()), "integer"),
                 ("집단당 관측치 평균", float(Ti.mean()), ""), ("집단당 관측치 최대", int(Ti.max()), "integer"),
                 ("도구변수 수", zrank, "integer")]
        if wald:
            rows += [(f"Wald χ²({wald[1]})", wald[0], ""), ("Prob > χ²", wald[2], "pvalue")]
        for lab, v, fmt in rows:
            r = {"metric": lab, "val": v}
            if fmt:
                r["_rowfmt"] = fmt
            t.add_row(r)
        t.add_note(("2단계" if o["twostep"] else "1단계") + f" 추정, 표준오차: {vlab}")
        if system:
            t.add_note("관측치 수는 수준식 기준 (Stata xtdpdsys 표기)")
        if o["collapse"]:
            t.add_note("GMM형 도구변수 축약 (xtabond2 collapse 방식, Stata xtabond·xtdpdsys에는 없는 옵션)")
        if D.time_note:
            t.add_note(D.time_note)
        if L != zrank:
            t.add_note(f"도구변수 행렬 열 {L}개 중 선형독립 {zrank}개 (일반화 역행렬 사용)")
        self.add_table(t)

    def _coef_table(self, names, b, se, z, p, crit, omitted, vlab):
        t = ResultsTable(name="coef", title="계수 추정치 (Coefficients)")
        t.add_column("var", "변수", col_type="text")
        t.add_column("b", "추정치", col_type="number")
        t.add_column("se", "표준오차", col_type="number")
        t.add_column("z", "z", col_type="number")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        t.add_column("cil", "95% CI 하한", col_type="number")
        t.add_column("ciu", "95% CI 상한", col_type="number")
        for i, nm in enumerate(names):
            t.add_row({"var": nm, "b": float(b[i]), "se": float(se[i]), "z": float(z[i]), "p": float(p[i]),
                       "cil": float(b[i] - crit * se[i]), "ciu": float(b[i] + crit * se[i])})
        t.add_note(f"표준오차: {vlab}; 계수는 수준식 모형 기준 (Stata xtabond 표기)")
        if self.options["vce"] == "robust":
            t.add_note("강건 표준오차는 개체(패널) 군집 기준")
        if omitted:
            t.add_note("공선성으로 제외 (Stata omitted): " + ", ".join(omitted))
        if getattr(self, "_iv_omitted", None):
            t.add_note("공선성으로 표준 도구변수에서 제외 (Stata omitted from div()): " + ", ".join(self._iv_omitted))
        if "Intercept" in names:
            t.add_note("상수항(Intercept)은 수준식 오차로 식별 (도구변수 _cons)")
        t.mark_coefficients(estimate="b", se="se", p="p", stat="z", ci=("cil", "ciu"), label="var")
        self.add_table(t)

    def _instrument_table(self, summary_rows, Z, system):
        t = ResultsTable(name="instruments", title="도구변수 구성 (Instruments)")
        t.add_column("eq", "식", col_type="text")
        t.add_column("type", "유형", col_type="text")
        t.add_column("spec", "도구변수", col_type="text")
        t.add_column("ncol", "열 수", col_type="integer")
        for r in summary_rows:
            t.add_row({"eq": "차분식" if r["eq"] == "d" else "수준식", "type": "GMM형" if r["type"] == "GMM" else "표준(IV형)",
                       "spec": r["label"], "ncol": int(r["end"] - r["start"])})
        t.add_note("GMM형: 시점별(축약 시 시차별) 열, 없는 값은 0; 표준: 변수당 1열")
        t.add_note("L(a/b).x = x의 a~b기 시차 수준값, '.' = 가용한 모든 시차; LD.x = 1기 시차 차분")
        self.add_table(t)

    # ------------------------------------------------------------------ specification tests
    def _ar_tests(self, D, X, Z, e, res, A, Wi, V, g, t, is_d, n_ind, K):
        o = self.options
        system = o["method"] == "system"
        has_cons = o["constant"]
        tab = ResultsTable(name="ar", title="Arellano–Bond 자기상관 검정 (estat abond)")
        tab.add_column("order", "차수", col_type="integer")
        tab.add_column("z", "z", col_type="number")
        tab.add_column("p", "Prob > z", col_type="number", fmt="pvalue")
        tab.add_note("H0: 차분 오차에 해당 차수 자기상관 없음 (1차는 구조적으로 존재, 2차 기각 시 모형 설정 문제)")
        sys_like = system or (has_cons and (~is_d).any())
        if sys_like and not o["twostep"] and o["vce"] == "gmm":
            tab.add_note("Stata: 1단계 시스템 추정(상수항 포함 xtabond 포함) 후에는 vce(robust)일 때만 계산 — 미산출")
            for m in range(1, o["artests"] + 1):
                tab.add_row({"order": m, "z": None, "p": None})
            self.add_table(tab)
            return
        di = np.where(is_d)[0]
        ed = e[di]
        gd, td = g[di], t[di]
        Xd = X[di]
        key = pd.Series(np.arange(len(di)), index=pd.MultiIndex.from_arrays([gd, td]))
        robust = o["vce"] == "robust" or o["twostep"]
        sig2 = res["sig2_1"]
        nxt = np.where((gd[1:] == gd[:-1]) & (td[1:] - td[:-1] == 1))[0]
        for m in range(1, o["artests"] + 1):
            li = key.reindex(pd.MultiIndex.from_arrays([gd, td - m])).to_numpy()
            ok = ~np.isnan(li)
            u = np.zeros(len(di))
            u[ok] = ed[li[ok].astype(int)]
            s0 = float(u @ ed)
            qx = u @ Xd                                            # 1 x K
            if not robust:
                # homoskedastic one-step: H = sigma^2 [1 on diag, -0.5 next]
                Hu = u.copy()
                if len(nxt):
                    Hu[nxt] -= 0.5 * u[nxt + 1]
                    Hu[nxt + 1] -= 0.5 * u[nxt]
                s1 = sig2 * float(u @ Hu)
                Hu_full = np.zeros(len(e))
                Hu_full[di] = sig2 * Hu
                Qzu = np.asarray(Z.T @ Hu_full).ravel()
            else:
                # robust / two-step form: H_i = e_i e_i' from the residuals of the reported step
                # (matches Stata's published abdata results for one-step robust and two-step WC-robust)
                c = np.bincount(gd, weights=u * ed, minlength=n_ind)       # e_i' u_i per panel
                s1 = float(np.sum(c ** 2))
                Qzu = np.asarray(Z.T @ (e * c[g])).ravel()
            ZX = np.asarray(Z.T @ X)
            s2 = -2.0 * float(qx @ Wi @ ZX.T @ A @ Qzu)
            s3 = float(qx @ V @ qx)
            den = s1 + s2 + s3
            if den <= 0 or not np.isfinite(den):
                tab.add_row({"order": m, "z": None, "p": None})
                continue
            zv = s0 / np.sqrt(den)
            tab.add_row({"order": m, "z": zv, "p": float(2 * stats.norm.sf(abs(zv)))})
        if robust:
            tab.add_note("강건 형태 (잔차 외적 사용)")
        self.add_table(tab)

    def _overid(self, Z, e, res, A1, A2, zrank, K, g, n_ind):
        o = self.options
        df = zrank - K
        t = ResultsTable(name="overid", title="과대식별 검정 (Sargan / Hansen)")
        t.add_column("test", "검정", col_type="text")
        t.add_column("chi2", "χ²", col_type="number")
        t.add_column("df", "df", col_type="integer")
        t.add_column("p", "p", col_type="number", fmt="pvalue")
        if df <= 0:
            t.add_row({"test": "과대식별 제약 없음 (도구변수 수 ≤ 계수 수)", "chi2": None, "df": df, "p": None})
            self.add_table(t)
            return
        Ze1 = np.asarray(Z.T @ res["e1"]).ravel()
        S1 = float(Ze1 @ A1 @ Ze1) / res["sig2_1"]
        t.add_row({"test": "Sargan (1단계 잔차, 동분산 가정)", "chi2": S1, "df": df, "p": float(stats.chi2.sf(S1, df))})
        if o["twostep"]:
            Ze2 = np.asarray(Z.T @ e).ravel()
            J = float(Ze2 @ A2 @ Ze2)
            t.add_row({"test": "Hansen J (2단계 잔차, 2단계 가중행렬)", "chi2": J, "df": df, "p": float(stats.chi2.sf(J, df))})
        elif o["vce"] == "robust":
            J = float(Ze1 @ A2 @ Ze1)
            t.add_row({"test": "Hansen J (1단계 잔차, 강건 가중행렬)", "chi2": J, "df": df, "p": float(stats.chi2.sf(J, df))})
        t.add_note("Sargan: 이분산 하에서 χ² 분포가 아님 — Stata estat sargan은 1단계 비강건 추정 후 이 값을 보고, "
                   "vce(robust) 지정 시 미보고")
        if o["twostep"]:
            t.add_note("2단계 비강건 추정 후 Stata estat sargan 값 = Hansen J 행 (이분산에 강건, 도구변수가 많으면 검정력 약화)")
        elif o["vce"] == "robust":
            t.add_note("Hansen J는 xtabond2의 1단계 강건 추정 보고 방식 (Stata xtabond 미보고)")
        self.add_table(t)


# ----------------------------------------------------------------------------- registry glue
def _sets_from_params(p: Dict[str, Any], key_vars: Tuple[str, ...], key_lags: Tuple[str, ...], key_max: Tuple[str, ...]):
    v = _opt(p, *key_vars)
    return v, int(_opt(p, *key_lags, default=0) or 0), _opt(p, *key_max)


def _limit(v: Any) -> Optional[int]:
    """Lag limits: None, '', '.' or 0 (UI number field) mean 'all available lags'."""
    if v in (None, "", ".", 0, "0"):
        return None
    return int(v)


def _kwargs(p: Dict[str, Any]) -> Dict[str, Any]:
    pre, pl, pm_ = _sets_from_params(p, ("pre", "predetermined"), ("preLags", "pre_lags"), ("preMaxLags", "pre_maxlags"))
    en, el, em = _sets_from_params(p, ("endog", "endogenous"), ("endogLags", "endog_lags"), ("endogMaxLags", "endog_maxlags"))
    pm_, em = _limit(pm_), _limit(em)
    return dict(
        entity=_as_single(_opt(p, "entity", "entity_col", "idVar", "panelVar")),
        time=_as_single(_opt(p, "time", "time_col", "timeVar")),
        dep=_as_single(_opt(p, "dep", "depvar")),
        exog=_opt(p, "exog", "covs", "indeps", default=[]),
        method=str(_opt(p, "method", "estimator", "transformation", default="difference")),
        lags=int(_opt(p, "lags", default=1)),
        maxldep=_limit(_opt(p, "maxldep", "maxLdep", "max_ldep")),
        pre=pre, pre_lags=pl, pre_maxlags=pm_,
        endog=en, endog_lags=el, endog_maxlags=em,
        maxlags=_limit(_opt(p, "maxlags", "maxLags")),
        exog_lags=int(_opt(p, "exogLags", "exog_lags", default=0) or 0),
        time_dummies=_flag(p, "timeDummies", "time_dummies", default=False),
        twostep=_flag(p, "twostep", "twoStep", "two_step", default=False),
        vce=str(_opt(p, "vce", "covType", "cov_type", default="gmm")),
        collapse=_flag(p, "collapse", default=False),
        constant=not _flag(p, "noconstant", "noConstant", default=False) and _flag(p, "constant", default=True),
        artests=int(_opt(p, "artests", "arTests", default=2)),
        delta=float(_opt(p, "delta", default=1.0)),
    )


def dispatch(t: str, df: pd.DataFrame, params: Dict[str, Any]):
    if t != "dynPanel":
        return None
    return DynamicPanelGMM(df, **_kwargs(params or {}))


def _set_spec(s: Dict[str, Any]) -> str:
    vs = " ".join(s["vars"])
    if s["lags"] or s["maxlags"] is not None:
        return f"{vs}, lagstruct({s['lags']}, {'.' if s['maxlags'] is None else s['maxlags']})"
    return vs


def stata_code(t: str, p: Dict[str, Any]) -> Optional[List[str]]:
    if t != "dynPanel":
        return None
    kw = _kwargs(p)
    meth = resolve_choice("method", kw["method"], METHODS, METHOD_ALIASES, default="difference")
    vce = resolve_choice("vce", kw["vce"], VCES, VCE_ALIASES, default="gmm")
    terms = parse_terms(kw["exog"], kw["exog_lags"])
    by_var: Dict[str, List[int]] = {}
    for v, l in terms:
        by_var.setdefault(v, []).append(l)
    rhs = []
    for v, ls in by_var.items():
        ls = sorted(ls)
        if ls == [0]:
            rhs.append(v)
        elif ls == list(range(ls[0], ls[-1] + 1)):
            rhs.append(f"L({ls[0]}/{ls[-1]}).{v}")
        else:
            rhs += [term_name(v, l) for l in ls]
    out = xtset_lines(kw["entity"], kw["time"], kw["delta"])
    if kw["time_dummies"]:
        # xtabond / xtdpdsys do not accept factor variables: generate the period dummies (first period = base)
        out += [f"quietly tabulate {kw['time']}, generate(_yr)", "local _nyr = r(r)"]
        rhs.append("`_td'")
    opts = [f"lags({kw['lags']})"]
    if kw["maxldep"] not in (None, ""):
        opts.append(f"maxldep({kw['maxldep']})")
    if kw["maxlags"] not in (None, ""):
        opts.append(f"maxlags({kw['maxlags']})")
    for s in _parse_sets(kw["pre"], kw["pre_lags"], None if kw["pre_maxlags"] in (None, "") else int(kw["pre_maxlags"]), "pre"):
        opts.append(f"pre({_set_spec(s)})")
    for s in _parse_sets(kw["endog"], kw["endog_lags"], None if kw["endog_maxlags"] in (None, "") else int(kw["endog_maxlags"]), "endog"):
        opts.append(f"endogenous({_set_spec(s)})")
    if kw["twostep"]:
        opts.append("twostep")
    if vce == "robust":
        opts.append("vce(robust)")
    if not kw["constant"]:
        opts.append("noconstant")
    if kw["artests"] != 2:
        opts.append(f"artests({kw['artests']})")
    cmd = "xtabond" if meth == "difference" else "xtdpdsys"
    if kw["collapse"]:
        out.append("* collapse 옵션은 xtabond2(ssc install xtabond2)에서만 지원 — 아래 명령은 축약 없이 추정")
    line = f"{cmd} {kw['dep']} {' '.join(rhs)}, {' '.join(opts)}".replace("  ", " ")
    if kw["time_dummies"]:
        # first pass: Stata omits the dummies that are collinear after differencing but keeps their D. instruments,
        # which changes the WC-robust VCE; keep only the dummies it estimates and fit again
        out += ["* 1차 추정으로 공선성 제외(omitted) 시점 더미를 찾아 빼고 다시 추정 (중복 도구변수 방지)",
                "local _td \"_yr2-_yr`_nyr'\"", "quietly " + line, "local _td \"\"",
                "foreach v of varlist _yr2-_yr`_nyr' {",
                "    local _chk = 0",
                "    capture local _chk = _se[`v']",
                "    if `_chk' > 0 & `_chk' < . local _td \"`_td' `v'\"",
                "}"]
    out.append(line)
    out.append("estat abond" + (f", artests({kw['artests']})" if kw["artests"] != 2 else ""))
    if vce == "gmm":
        out.append("estat sargan")
    return out


def r_code(t: str, p: Dict[str, Any]) -> Optional[List[str]]:
    if t != "dynPanel":
        return None
    kw = _kwargs(p)
    meth = resolve_choice("method", kw["method"], METHODS, METHOD_ALIASES, default="difference")
    vce = resolve_choice("vce", kw["vce"], VCES, VCE_ALIASES, default="gmm")
    y = kw["dep"]
    terms = parse_terms(kw["exog"], kw["exog_lags"])
    rhs = [f"lag({y}, 1:{kw['lags']})" if kw["lags"] > 1 else f"lag({y}, 1)"]
    by_var: Dict[str, List[int]] = {}
    for v, l in terms:
        by_var.setdefault(v, []).append(l)
    for v, ls in by_var.items():
        rhs.append(v if ls == [0] else f"lag({v}, c({', '.join(map(str, sorted(ls)))}))")
    gmm = [f"lag({y}, 2:{99 if kw['maxldep'] in (None, '') else int(kw['maxldep']) + 1})"]
    for s in _parse_sets(kw["pre"], kw["pre_lags"], None if kw["pre_maxlags"] in (None, "") else int(kw["pre_maxlags"]), "pre"):
        for v in s["vars"]:
            rhs.append(v if s["lags"] == 0 else f"lag({v}, 0:{s['lags']})")
            hi = 99 if s["maxlags"] is None else s["lags"] + s["maxlags"]
            gmm.append(f"lag({v}, {s['lags'] + 1}:{hi})")
    for s in _parse_sets(kw["endog"], kw["endog_lags"], None if kw["endog_maxlags"] in (None, "") else int(kw["endog_maxlags"]), "endog"):
        for v in s["vars"]:
            rhs.append(v if s["lags"] == 0 else f"lag({v}, 0:{s['lags']})")
            hi = 99 if s["maxlags"] is None else s["lags"] + 1 + s["maxlags"]
            gmm.append(f"lag({v}, {s['lags'] + 2}:{hi})")
    exo = " + ".join(by_var.keys()) if by_var else ""
    form = f"{y} ~ {' + '.join(rhs)} | {' + '.join(gmm)}" + (f" | {' + '.join(rhs[1:len(by_var) + 1])}" if by_var else "")
    q = lambda s: '"' + s + '"'
    out = [f"pdata <- plm::pdata.frame(data, index = c({q(kw['entity'])}, {q(kw['time'])}))",
           "# 주의: plm::pgmm은 Stata와 기본값이 다름 (수준식 상수항·시점더미 처리, 1단계 가중행렬 fsm, 불균형 패널 표본 절단)",
           f"fit <- plm::pgmm({form}, data = pdata, effect = {q('twoways' if kw['time_dummies'] else 'individual')}, "
           f"model = {q('twosteps' if kw['twostep'] else 'onestep')}, transformation = {q('d' if meth == 'difference' else 'ld')}"
           + (", fsm = \"GI\"" if meth == "system" else "") + (", collapse = TRUE" if kw["collapse"] else "") + ")",
           f"summary(fit, robust = {'TRUE' if vce == 'robust' else 'FALSE'})"]
    return out


# ----------------------------------------------------------------------------- Stata command input
def _lagstruct(val: str) -> Tuple[List[str], int, Optional[int]]:
    """'w k, lagstruct(1, .)' / 'w, lag(1,3)' -> (vars, lags, maxlags)."""
    if "," in val:
        vs, rest = val.split(",", 1)
        m = re.search(r"lag(?:struct|s)?\s*\(\s*(\d+)\s*,\s*(\.|\d+)\s*\)", rest, re.IGNORECASE)
        if not m:
            raise ValueError(f"lagstruct 해석 불가: {val}")
        return vs.split(), int(m.group(1)), None if m.group(2) == "." else int(m.group(2))
    return val.split(), 0, None


def transpile(line: str) -> Optional[str]:
    s = line.strip()
    m = re.match(r"^(xtabond|xtdpdsys)\s+([^,]+?)\s*(?:,\s*(.*))?$", s, re.IGNORECASE)
    if m:
        cmd = m.group(1).lower()
        body = m.group(2).strip()
        toks = re.findall(r"L\(\d+\s*/\s*\d+\)\.\([^)]*\)|L\d*\.\([^)]*\)|[^\s]+", body)
        if not toks:
            return f"raise NotImplementedError({('Stata ' + cmd + ': 종속변수 필요 — ' + s)!r})"
        dep, rhs = toks[0], toks[1:]
        if re.search(r"[.#(]", dep):
            return f"raise NotImplementedError({('종속변수에 연산자 사용 불가 — ' + s)!r})"
        time_d = False
        exog = []
        for r in rhs:
            mm = re.fullmatch(r"i\.(\w+)", r, re.IGNORECASE)
            if mm:
                time_d = True
                continue
            if re.fullmatch(r"\w+", r) or _LAGSPEC.match(r.replace("l(", "L(").replace("l.", "L.")):
                exog.append(re.sub(r"^l", "L", r) if r[:1] == "l" and re.match(r"^l[\d(.]", r) else r)
            else:
                return f"raise NotImplementedError({('지원하지 않는 변수 표기: ' + r + ' — ' + s)!r})"
        opts = _split_opts(m.group(3) or "")
        kw = [f"dep={dep!r}", f"exog={exog!r}", f"method={'difference' if cmd == 'xtabond' else 'system'!r}"]
        pre_sets, end_sets = [], []
        # _split_opts keeps only the last of repeated pre()/endogenous(); parse them directly
        for mm in re.finditer(r"\b(pre|endogenous|endog)\s*\(((?:[^()]|\([^()]*\))*)\)", m.group(3) or "", re.IGNORECASE):
            try:
                vs, lq, mx = _lagstruct(mm.group(2))
            except ValueError as e:
                return f"raise NotImplementedError({(str(e) + ' — ' + s)!r})"
            (pre_sets if mm.group(1).lower() == "pre" else end_sets).append({"vars": vs, "lags": lq, "maxlags": mx})
        for k_ in ("pre", "endogenous", "endog"):
            opts.pop(k_, None)
        if pre_sets:
            kw.append(f"pre={pre_sets!r}")
        if end_sets:
            kw.append(f"endog={end_sets!r}")
        for name, key in (("lags", "lags"), ("maxldep", "maxldep"), ("maxlags", "maxlags"), ("artests", "artests")):
            if name in opts:
                try:
                    kw.append(f"{key}={int(opts.pop(name))}")
                except ValueError:
                    return f"raise NotImplementedError({(name + '() 값은 정수 — ' + s)!r})"
        if "twostep" in opts:
            opts.pop("twostep")
            kw.append("twostep=True")
        if "vce" in opts:
            v = opts.pop("vce").strip().lower()
            if "robust".startswith(v) and v:
                kw.append("vce='robust'")
            elif v in ("gmm", "conventional"):
                pass
            else:
                return f"raise NotImplementedError({('vce(' + v + ') 미지원 (gmm·robust) — ' + s)!r})"
        if "noconstant" in opts or "nocons" in opts or "noc" in opts:
            for k_ in ("noconstant", "nocons", "noc"):
                opts.pop(k_, None)
            kw.append("constant=False")
        if time_d:
            kw.append("timeDummies=True")
        if opts:
            return f"raise NotImplementedError({(cmd + ' 옵션 미지원: ' + ', '.join(opts) + ' — ' + s)!r})"
        idx = ("entity=getattr(sess, 'panel_spec', {}).get('idVar'), time=getattr(sess, 'panel_spec', {}).get('timeVar')")
        return f"res = sess.run('dynPanel', {idx}, {', '.join(kw)})"
    m = re.match(r"^estat\s+(abond|sargan)\b(.*)$", s, re.IGNORECASE)
    if m:
        what = m.group(1).lower()
        rest = m.group(2)
        mm = re.search(r"artests\s*\(\s*(\d+)\s*\)", rest, re.IGNORECASE)
        if what == "abond" and mm:
            return ("res = sess.run('dynPanel', **dict(next(a for a in reversed(sess.history) if getattr(a, 'name', '') == 'dynPanel').options, "
                    f"artests={int(mm.group(1))}))  # [Stata] {s}")
        return f"# [Stata] {s}  → 동적 패널 GMM 결과의 '{'Arellano–Bond 자기상관 검정' if what == 'abond' else '과대식별 검정'}' 표에 포함"
    if re.match(r"^(xtdpd|xtabond2)\b", s, re.IGNORECASE):
        return (f"raise NotImplementedError({('xtdpd·xtabond2 일반 구문 미지원 — xtabond·xtdpdsys 형식 또는 '
                                               'sess.run(\"dynPanel\", ...)으로 지정: ' + s)!r})")
    return None


EXTENSION = {
    "catalog": [{"id": "dynPanel", "group": "Linear Models",
                 "title": "Dynamic Panel GMM (Arellano–Bond xtabond / Blundell–Bond xtdpdsys)",
                 "module": "openstat", "icon": "analysis-linearmodels.svg"}],
    "aliases": {"xtabond": "dynPanel", "xtdpdsys": "dynPanel", "dynamic_panel": "dynPanel", "dynamicPanel": "dynPanel",
                "pgmm": "dynPanel", "gmm_panel": "dynPanel", "arellano_bond": "dynPanel", "system_gmm": "dynPanel"},
    "dispatch": dispatch,
    "stata": stata_code,
    "r": r_code,
    "transpile": transpile,
}
